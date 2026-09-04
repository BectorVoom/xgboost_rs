//! The CPU `hist` tree updater — `grow_quantile_histmaker`.
//!
//! One boosting round grows one tree:
//!
//! ```text
//! root histogram ─▶ split evaluation ─▶ apply split ─▶ partition rows
//!        ▲                                                   │
//!        └──── child histograms (one built, one subtracted) ◀─┘
//! ```
//!
//! Three details are reproduced exactly from upstream because they change
//! results, not just speed:
//!
//! * **Which child is built.** The child with the smaller Hessian sum is built
//!   from the data and its sibling is derived as `parent - built`
//!   (`AssignNodes`). Swapping them changes the floating point rounding.
//! * **Row order within a node.** Partitioning is stable, so a node's rows stay
//!   in their original relative order.
//! * **Tie-breaking.** The best split is the maximum of `(loss change, then
//!   smallest feature index)`, a total order, so merging candidates in any
//!   order gives the same answer.
//!
//! # Determinism under parallelism
//!
//! Rows are cut into fixed-size blocks and blocks are dealt to a fixed number
//! of *lanes*, each with its own partial histogram; lanes are then summed in
//! lane order. Both the block size and the lane count depend only on the data,
//! never on the thread count, so a model trained on 1 core and on 32 cores is
//! bit-identical.
//!
//! # Parallelism across growth policies
//!
//! Queue selection itself is policy-specific and cheap: monotonically allocated
//! node ids make `depthwise` a FIFO, while `lossguide` uses a gain-ordered heap.
//! This keeps queue work linear or `O(n log n)` as `max_leaves` grows instead of
//! repeatedly scanning and shifting a flat pending-node vector.
//!
//! `depthwise` expands a whole level, so small nodes naturally run concurrently.
//! `lossguide` expands one node at a time; when that node has fewer row lanes
//! than workers, its dense histogram is additionally divided into contiguous,
//! disjoint feature groups. Every feature stays in one task and visits rows in
//! the same order as the ordinary row-major kernel, so this uses otherwise idle
//! workers without changing a bin's floating-point addition sequence.
//!
//! Lowering `BLOCK_ROWS` would also create more work, but it changes the order
//! the `f64` partial histograms are reduced in and therefore changes the model.
//! The feature-group fallback is deliberately limited to dense lossguide nodes
//! with enough bin updates to repay nested task setup; sparse data and
//! depth-wise batches keep the cache-friendly row-lane kernel.

use super::cat;
use super::column_sampler::{ColumnSampler, FeatureSet};
use super::evaluator::{InteractionConstraints, SplitEvaluator};
use super::model::RegTree;
use super::param::{GradStats, GrowPolicy, RT_EPS, SplitEntry, TrainParam, calc_weight};
use crate::context::Context;
use crate::data::gradient_index::{DenseColumn, GHistIndex, dispatch_bins};
use crate::data::DMatrix;
use crate::objective::GradientPair;
use crate::rng::Mt19937;
use crate::threading;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};

/// Rows per histogram block. Large enough to amortise per-block overhead,
/// small enough that lanes stay balanced.
const BLOCK_ROWS: usize = 4096;
/// Upper bound on partial histograms per node.
const MAX_LANES: usize = 16;
/// Memory ceiling for a node's partial histograms, in bytes.
const LANE_BUDGET: usize = 16 << 20;
/// Nodes below this size are partitioned serially; the parallel path costs more
/// than it saves.
const PARALLEL_PARTITION_MIN: usize = 1 << 15;
/// Minimum bin updates before nested feature-group parallelism pays for its
/// task setup on a lossguide node with too few row lanes.
const FEATURE_GROUP_MIN_UPDATES: usize = 1 << 15;
/// Bins per task when reducing or subtracting histograms.
const REDUCE_CHUNK_BINS: usize = 2048;
/// Maximum number of nodes expanded together by the depth-wise policy.
const MAX_NODE_BATCH_SIZE: usize = 256;

/// A bin index stored in the compressed feature matrix.
pub(crate) trait BinIdx: Copy + Send + Sync {
    fn idx(self) -> usize;
}

impl BinIdx for u8 {
    #[inline(always)]
    fn idx(self) -> usize {
        self as usize
    }
}
impl BinIdx for u16 {
    #[inline(always)]
    fn idx(self) -> usize {
        self as usize
    }
}
impl BinIdx for u32 {
    #[inline(always)]
    fn idx(self) -> usize {
        self as usize
    }
}

/// A node waiting to be split, upstream's `CPUExpandEntry`.
#[derive(Clone, Debug, Default)]
struct ExpandEntry {
    nid: usize,
    depth: i32,
    split: SplitEntry,
    /// The categories this split sends right, when `split.is_cat`. Kept beside
    /// the split rather than inside it because it is variable-length and
    /// [`SplitEntry`] is copied by the thousand during the search.
    cat_bits: Vec<u32>,
}

impl ExpandEntry {
    /// Whether this candidate may still be applied to the tree.
    fn is_valid(&self, p: &TrainParam, num_leaves: i32) -> bool {
        if self.split.loss_chg <= RT_EPS {
            return false;
        }
        if self.split.left_sum.sum_hess == 0.0 || self.split.right_sum.sum_hess == 0.0 {
            return false;
        }
        if self.split.loss_chg < p.min_split_loss {
            return false;
        }
        if p.max_depth > 0 && self.depth == p.max_depth {
            return false;
        }
        if p.max_leaves > 0 && num_leaves == p.max_leaves {
            return false;
        }
        true
    }
}

/// Heap wrapper whose maximum is the loss-guide candidate expanded next.
///
/// Split gains entering the queue are finite and positive. `total_cmp` still
/// gives the heap a complete ordering, while reversing the node-id comparison
/// preserves upstream's smaller-id tie break.
#[derive(Debug)]
struct LossGuideEntry(ExpandEntry);

impl PartialEq for LossGuideEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for LossGuideEntry {}

impl PartialOrd for LossGuideEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LossGuideEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .split
            .loss_chg
            .total_cmp(&other.0.split.loss_chg)
            .then_with(|| other.0.nid.cmp(&self.0.nid))
    }
}

/// Policy-specific expansion queue.
///
/// Tree node ids are allocated monotonically, so depth-wise growth is a FIFO:
/// new children always follow every node already waiting. Loss-guide needs a
/// real priority queue because every child can have a different gain.
enum ExpandQueue {
    DepthWise(VecDeque<ExpandEntry>),
    LossGuide(BinaryHeap<LossGuideEntry>),
}

impl ExpandQueue {
    fn new(policy: GrowPolicy) -> Self {
        match policy {
            GrowPolicy::DepthWise => Self::DepthWise(VecDeque::new()),
            GrowPolicy::LossGuide => Self::LossGuide(BinaryHeap::new()),
        }
    }

    fn push(&mut self, entry: ExpandEntry) {
        match self {
            Self::DepthWise(queue) => {
                debug_assert!(queue.back().is_none_or(|last| last.nid < entry.nid));
                queue.push_back(entry);
            }
            Self::LossGuide(queue) => queue.push(LossGuideEntry(entry)),
        }
    }

    /// Fill `out` with the next expansion batch, retaining its allocation for
    /// the next call. This is one entry for loss-guide and up to 256 entries
    /// from one level for depth-wise growth.
    fn pop_batch(
        &mut self,
        param: &TrainParam,
        num_leaves: &mut i32,
        out: &mut Vec<ExpandEntry>,
    ) {
        out.clear();
        match self {
            Self::LossGuide(queue) => {
                let Some(LossGuideEntry(entry)) = queue.pop() else {
                    return;
                };
                if entry.is_valid(param, *num_leaves) {
                    *num_leaves += 1;
                    out.push(entry);
                }
            }
            Self::DepthWise(queue) => {
                let Some(level) = queue.front().map(|entry| entry.depth) else {
                    return;
                };
                while out.len() < MAX_NODE_BATCH_SIZE
                    && queue.front().is_some_and(|entry| entry.depth == level)
                {
                    let entry = queue.pop_front().expect("front entry was present");
                    if entry.is_valid(param, *num_leaves) {
                        *num_leaves += 1;
                        out.push(entry);
                    }
                }
            }
        }
    }
}

/// Node statistics carried alongside the tree during growth.
#[derive(Clone, Copy, Debug, Default)]
struct NodeEntry {
    stats: GradStats,
    root_gain: f32,
}

/// Row sets per node: a permutation of row ids plus one slice per live node.
struct Partitioner {
    row_indices: Vec<u32>,
    /// `(begin, end)` into `row_indices` for each node id, if live.
    segments: Vec<Option<(usize, usize)>>,
    scratch: Vec<u32>,
    /// One cached left/right decision per row while a large node is partitioned.
    decisions: Vec<u8>,
}

/// One node's partitioning job.
struct SplitTask {
    nid: usize,
    left: usize,
    right: usize,
    fidx: u32,
    /// Feature-local bin threshold; rows at or below it go left. Unused by a
    /// categorical split, which tests [`Self::cat_bits`] instead.
    local_cond: i64,
    default_left: bool,
    /// The categories that go right, when this is a categorical split.
    cat_bits: Vec<u32>,
}

type PartitionJob<'a> = (usize, &'a mut [u32], &'a mut [u32], &'a mut [u8]);
type PartitionWork<'task, 'data, 'out> = (
    &'task SplitTask,
    &'data mut [u32],
    &'data mut [u32],
    &'data mut [u8],
    &'out mut usize,
);

impl Partitioner {
    fn new(num_row: usize) -> Self {
        Self {
            row_indices: (0..num_row as u32).collect(),
            segments: vec![Some((0, num_row))],
            scratch: vec![0; num_row],
            decisions: vec![0; num_row],
        }
    }

    fn reset(&mut self, num_row: usize) {
        self.row_indices.clear();
        self.row_indices.extend(0..num_row as u32);
        self.segments.clear();
        self.segments.push(Some((0, num_row)));
        self.scratch.resize(num_row, 0);
        self.decisions.resize(num_row, 0);
    }

    #[inline]
    fn rows(&self, nid: usize) -> &[u32] {
        match self.segments[nid] {
            Some((b, e)) => &self.row_indices[b..e],
            None => &[],
        }
    }

    fn ensure(&mut self, n: usize) {
        if self.segments.len() < n {
            self.segments.resize(n, None);
        }
    }

    /// Stable-partition every task's rows into its two children.
    ///
    /// Nodes own disjoint slices of `row_indices`, so a whole level is
    /// partitioned in one parallel pass; the root, which is one huge node, adds
    /// a second level of block parallelism inside its own slice.
    fn split_all(&mut self, tasks: &[SplitTask], gi: &GHistIndex) {
        if tasks.is_empty() {
            return;
        }
        let mut ranges: Vec<(usize, usize, usize)> = tasks
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let (b, e) = self.segments[t.nid].expect("splitting a node with no rows");
                (b, e, i)
            })
            .collect();
        // Carving disjoint borrows requires the ranges in address order.
        ranges.sort_unstable_by_key(|r| r.0);

        let mut src_rest: &mut [u32] = &mut self.row_indices;
        let mut dst_rest: &mut [u32] = &mut self.scratch;
        let mut decision_rest: &mut [u8] = &mut self.decisions;
        let mut consumed = 0usize;
        let mut jobs: Vec<PartitionJob<'_>> = Vec::with_capacity(ranges.len());
        for &(begin, end, task_idx) in &ranges {
            let (_, s_tail) = src_rest.split_at_mut(begin - consumed);
            let (_, d_tail) = dst_rest.split_at_mut(begin - consumed);
            let (_, decision_tail) = decision_rest.split_at_mut(begin - consumed);
            let (s_here, s_tail) = s_tail.split_at_mut(end - begin);
            let (d_here, d_tail) = d_tail.split_at_mut(end - begin);
            let (decision_here, decision_tail) = decision_tail.split_at_mut(end - begin);
            src_rest = s_tail;
            dst_rest = d_tail;
            decision_rest = decision_tail;
            consumed = end;
            jobs.push((task_idx, s_here, d_here, decision_here));
        }

        let mut n_left = vec![0usize; tasks.len()];
        {
            let slots: Vec<(usize, &mut usize)> =
                n_left.iter_mut().enumerate().collect();
            let mut by_task: Vec<Option<&mut usize>> = (0..tasks.len()).map(|_| None).collect();
            for (i, v) in slots {
                by_task[i] = Some(v);
            }
            let mut work: Vec<PartitionWork<'_, '_, '_>> = Vec::with_capacity(jobs.len());
            for (task_idx, src, dst, decisions) in jobs {
                let out = by_task[task_idx].take().expect("each task appears once");
                work.push((&tasks[task_idx], src, dst, decisions, out));
            }
            work.into_par_iter().for_each(|(task, src, dst, decisions, out)| {
                *out = if src.len() >= PARALLEL_PARTITION_MIN {
                    partition_blocked(src, dst, decisions, task, gi)
                } else {
                    let n = partition_block(src, dst, task, gi);
                    src.copy_from_slice(dst);
                    n
                };
            });
        }

        for (i, task) in tasks.iter().enumerate() {
            let (begin, end) = self.segments[task.nid].expect("checked above");
            self.ensure(task.left.max(task.right) + 1);
            self.segments[task.nid] = None;
            self.segments[task.left] = Some((begin, begin + n_left[i]));
            self.segments[task.right] = Some((begin + n_left[i], end));
        }
    }

}

/// Blocked partition of one large node: count per block, prefix-sum, scatter.
///
/// Each block keeps its rows in order on both sides, so the result is identical
/// to the serial partition. `src` is overwritten with the result.
fn partition_blocked(
    src: &mut [u32],
    dst: &mut [u32],
    decisions: &mut [u8],
    task: &SplitTask,
    gi: &GHistIndex,
) -> usize {
    debug_assert_eq!(src.len(), decisions.len());
    let n_blocks = src.len().div_ceil(BLOCK_ROWS);

    // Cache routing decisions during the counting pass. The scatter pass used
    // to call `goes_left` again, repeating the feature-column lookup for every
    // row in every large node.
    let left_counts = match gi.columns.dense_column(task.fidx) {
        Some((DenseColumn::U8(bins), missing)) if task.cat_bits.is_empty() => {
            dense_numeric_decisions(decisions, src, bins, missing, task)
        }
        Some((DenseColumn::U16(bins), missing)) if task.cat_bits.is_empty() => {
            dense_numeric_decisions(decisions, src, bins, missing, task)
        }
        Some((DenseColumn::U32(bins), missing)) if task.cat_bits.is_empty() => {
            dense_numeric_decisions(decisions, src, bins, missing, task)
        }
        _ => decisions
            .par_chunks_mut(BLOCK_ROWS)
            .zip(src.par_chunks(BLOCK_ROWS))
            .map(|(flags, rows)| {
                flags.iter_mut().zip(rows).fold(0usize, |count, (flag, &rid)| {
                    let left = goes_left(rid, task, gi);
                    *flag = u8::from(left);
                    count + usize::from(left)
                })
            })
            .collect(),
    };
    let n_left: usize = left_counts.iter().sum();

    // Each block owns one contiguous slice on each side, handed out as disjoint
    // borrows rather than shared pointers.
    let (left_part, right_part) = dst.split_at_mut(n_left);
    let mut left_rest = left_part;
    let mut right_rest = right_part;
    let mut jobs: Vec<(usize, &mut [u32], &mut [u32])> = Vec::with_capacity(n_blocks);
    for b in 0..n_blocks {
        let block_len = (((b + 1) * BLOCK_ROWS).min(src.len())) - b * BLOCK_ROWS;
        let (l, l_rest) = left_rest.split_at_mut(left_counts[b]);
        let (r, r_rest) = right_rest.split_at_mut(block_len - left_counts[b]);
        left_rest = l_rest;
        right_rest = r_rest;
        jobs.push((b, l, r));
    }

    let src_ro: &[u32] = src;
    jobs.into_par_iter().for_each(|(b, l_out, r_out)| {
        let lo = b * BLOCK_ROWS;
        let hi = ((b + 1) * BLOCK_ROWS).min(src_ro.len());
        let (mut li, mut ri) = (0usize, 0usize);
        for (&rid, &left) in src_ro[lo..hi].iter().zip(&decisions[lo..hi]) {
            if left != 0 {
                debug_assert!(li < l_out.len());
                // SAFETY: `l_out` was sized from this block's cached count.
                unsafe { *l_out.get_unchecked_mut(li) = rid };
                li += 1;
            } else {
                debug_assert!(ri < r_out.len());
                // SAFETY: the complementary count sized `r_out` exactly.
                unsafe { *r_out.get_unchecked_mut(ri) = rid };
                ri += 1;
            }
        }
        debug_assert_eq!(li, l_out.len());
        debug_assert_eq!(ri, r_out.len());
    });

    src.copy_from_slice(dst);
    n_left
}

/// Route a dense numerical column after dispatching its bin width once.
fn dense_numeric_decisions<T>(
    decisions: &mut [u8],
    rows: &[u32],
    bins: &[T],
    missing: u32,
    task: &SplitTask,
) -> Vec<usize>
where
    T: Copy + Into<u32> + Sync,
{
    if missing == u32::MAX {
        decisions
            .par_chunks_mut(BLOCK_ROWS)
            .zip(rows.par_chunks(BLOCK_ROWS))
            .map(|(flags, rows)| {
                flags.iter_mut().zip(rows).fold(0usize, |count, (flag, &rid)| {
                    debug_assert!((rid as usize) < bins.len());
                    // SAFETY: the column has one bin per matrix row, and row
                    // ids originate from `0..num_row`.
                    let bin: u32 = unsafe { (*bins.get_unchecked(rid as usize)).into() };
                    let left = (bin as i64) <= task.local_cond;
                    *flag = u8::from(left);
                    count + usize::from(left)
                })
            })
            .collect()
    } else {
        decisions
            .par_chunks_mut(BLOCK_ROWS)
            .zip(rows.par_chunks(BLOCK_ROWS))
            .map(|(flags, rows)| {
                flags.iter_mut().zip(rows).fold(0usize, |count, (flag, &rid)| {
                    debug_assert!((rid as usize) < bins.len());
                    // SAFETY: as above; this branch additionally handles the
                    // dense missing-value sentinel.
                    let bin: u32 = unsafe { (*bins.get_unchecked(rid as usize)).into() };
                    let left = if bin == missing {
                        task.default_left
                    } else {
                        (bin as i64) <= task.local_cond
                    };
                    *flag = u8::from(left);
                    count + usize::from(left)
                })
            })
            .collect()
    }
}

/// Serial partition of one node, used by the small-node path.
fn partition_block(
    src: &[u32],
    dst: &mut [u32],
    task: &SplitTask,
    gi: &GHistIndex,
) -> usize {
    let mut l = 0usize;
    let mut r = src.len();
    // Right-hand rows are written from the back and reversed afterwards, which
    // keeps a single pass while preserving order on both sides.
    for &rid in src {
        if goes_left(rid, task, gi) {
            dst[l] = rid;
            l += 1;
        } else {
            r -= 1;
            dst[r] = rid;
        }
    }
    dst[l..].reverse();
    l
}

/// Which side of the split row `rid` belongs to.
///
/// Reads the transposed index: consecutive rows of one feature are adjacent, so
/// a whole node is scanned sequentially. A categorical feature's local bin *is*
/// its category code, so membership is tested against the bin directly and no
/// value has to be recovered.
#[inline]
fn goes_left(rid: u32, task: &SplitTask, gi: &GHistIndex) -> bool {
    match gi.columns.get(task.fidx, rid as usize) {
        Some(bin) if task.cat_bits.is_empty() => (bin as i64) <= task.local_cond,
        Some(bin) => !cat::check_bit(&task.cat_bits, bin),
        None => task.default_left,
    }
}

/// Per-node gradient histograms, indexed by node id.
///
/// A vector-leaf tree keeps one histogram *per target* per node, which is how
/// upstream stores them too: the accumulation kernel stays scalar and is simply
/// run once per target, so nothing on the hot path has to know about targets.
struct HistCollection {
    /// Indexed `nid * n_targets + target`.
    data: Vec<Vec<GradStats>>,
    /// Buffers of released nodes, kept for reuse: a histogram is hundreds of
    /// kilobytes and every level would otherwise allocate a fresh set.
    free: Vec<Vec<GradStats>>,
    total_bins: usize,
    n_targets: usize,
    /// `max_cached_hist_node`: how many buffers may be held for reuse. Past
    /// this the released ones are dropped instead, which trades allocator
    /// traffic for a smaller resident set on a wide tree.
    max_cached: usize,
}

impl HistCollection {
    fn new(total_bins: usize, n_targets: usize, max_cached: usize) -> Self {
        Self { data: Vec::new(), free: Vec::new(), total_bins, n_targets, max_cached }
    }

    /// Slot of one `(node, target)` histogram.
    #[inline]
    fn slot(&self, nid: usize, target: usize) -> usize {
        nid * self.n_targets + target
    }

    /// Give node `nid` a buffer per target. Their contents are undefined;
    /// callers either fill them completely or zero them first.
    fn allocate(&mut self, nid: usize) {
        let last = self.slot(nid, self.n_targets - 1);
        if self.data.len() <= last {
            self.data.resize_with(last + 1, Vec::new);
        }
        for t in 0..self.n_targets {
            let slot = self.slot(nid, t);
            if self.data[slot].len() == self.total_bins {
                continue;
            }
            self.data[slot] = match self.free.pop() {
                Some(buf) => buf,
                None => vec![GradStats::default(); self.total_bins],
            };
        }
    }

    /// Release a node's histograms, which will not be read again.
    fn release(&mut self, nid: usize) {
        for t in 0..self.n_targets {
            let slot = self.slot(nid, t);
            if slot < self.data.len() && !self.data[slot].is_empty() {
                let buf = std::mem::take(&mut self.data[slot]);
                if self.free.len() < self.max_cached {
                    self.free.push(buf);
                }
            }
        }
    }

    fn clear(&mut self) {
        for i in 0..self.data.len() {
            let buf = std::mem::take(&mut self.data[i]);
            if !buf.is_empty() && self.free.len() < self.max_cached {
                self.free.push(buf);
            }
        }
        self.data.clear();
    }

    #[inline]
    fn get(&self, nid: usize) -> &[GradStats] {
        &self.data[nid * self.n_targets]
    }

    /// One target's histogram of node `nid`.
    #[inline]
    fn get_target(&self, nid: usize, target: usize) -> &[GradStats] {
        &self.data[self.slot(nid, target)]
    }
}

/// Accumulate one block of dense rows into `hist`.
#[inline]
fn build_dense<T: BinIdx>(
    hist: &mut [GradStats],
    rows: &[u32],
    index: &[T],
    stride: usize,
    offsets: &[u32],
    gpair: &[GradientPair],
) {
    debug_assert_eq!(offsets.len(), stride);
    for &rid in rows {
        let rid = rid as usize;
        debug_assert!(rid < gpair.len());
        debug_assert!(rid.checked_mul(stride).is_some_and(|base| base + stride <= index.len()));

        // SAFETY: row ids originate as `0..num_row` in `Partitioner`; a dense
        // gradient index stores exactly `stride` bins for every row. Cut
        // offsets turn each feature-local bin into a global bin below
        // `hist.len()` (covered by `histogram_kernels_stay_in_bounds`). Avoiding
        // these repeated checks matters because this is the training hot loop.
        let g = unsafe { *gpair.get_unchecked(rid) };
        let (gd, hd) = (g.grad as f64, g.hess as f64);
        let base = rid * stride;
        for feature in 0..stride {
            // SAFETY: justified by the invariants above.
            let bin = unsafe { (*index.get_unchecked(base + feature)).idx() };
            let off = unsafe { *offsets.get_unchecked(feature) as usize };
            let cell = unsafe { hist.get_unchecked_mut(off + bin) };
            cell.sum_grad += gd;
            cell.sum_hess += hd;
        }
    }
}

/// Accumulate one row lane using parallel, disjoint groups of features.
///
/// Each feature remains in exactly one task, and that task visits the lane's
/// rows in the same order as [`build_dense`]. The grouping therefore exposes
/// spare parallelism without changing any bin's floating-point sum order.
fn build_dense_feature_groups(
    hist: &mut [GradStats],
    rows: &[u32],
    gi: &GHistIndex,
    gpair: &[GradientPair],
    first_block: usize,
    block_step: usize,
    groups: usize,
) {
    dispatch_bins!(&gi.index, |index| build_dense_feature_groups_typed(
        hist,
        rows,
        index,
        gi.row_stride,
        &gi.offsets,
        gpair,
        first_block,
        block_step,
        groups,
    ))
}

#[allow(clippy::too_many_arguments)]
fn build_dense_feature_groups_typed<T: BinIdx>(
    hist: &mut [GradStats],
    rows: &[u32],
    index: &[T],
    stride: usize,
    offsets: &[u32],
    gpair: &[GradientPair],
    first_block: usize,
    block_step: usize,
    groups: usize,
) {
    let groups = groups.min(stride).max(1);
    let hist_len = hist.len();
    let mut rest = hist;
    let mut consumed = 0usize;
    let mut tasks = Vec::with_capacity(groups);
    for group in 0..groups {
        let first_feature = group * stride / groups;
        let end_feature = (group + 1) * stride / groups;
        let begin = offsets[first_feature] as usize;
        let end = if end_feature < stride {
            offsets[end_feature] as usize
        } else {
            hist_len
        };
        debug_assert_eq!(begin, consumed);
        let (group_hist, tail) = rest.split_at_mut(end - begin);
        rest = tail;
        consumed = end;
        tasks.push((first_feature, end_feature, begin, group_hist));
    }

    tasks.into_par_iter().for_each(
        |(first_feature, end_feature, hist_offset, group_hist)| {
            let n_blocks = rows.len().div_ceil(BLOCK_ROWS);
            let mut block = first_block;
            while block < n_blocks {
                let lo = block * BLOCK_ROWS;
                let hi = ((block + 1) * BLOCK_ROWS).min(rows.len());
                for &rid in &rows[lo..hi] {
                    let rid = rid as usize;
                    debug_assert!(rid < gpair.len());
                    debug_assert!(rid * stride + end_feature <= index.len());
                    let g = unsafe { *gpair.get_unchecked(rid) };
                    let (gd, hd) = (g.grad as f64, g.hess as f64);
                    let row_base = rid * stride;
                    for feature in first_feature..end_feature {
                        // SAFETY: the same dense-index and global-bin
                        // invariants as `build_dense` apply. `group_hist`
                        // covers precisely this task's feature range.
                        let bin = unsafe { (*index.get_unchecked(row_base + feature)).idx() };
                        let off = unsafe { *offsets.get_unchecked(feature) as usize };
                        let cell = unsafe { group_hist.get_unchecked_mut(off - hist_offset + bin) };
                        cell.sum_grad += gd;
                        cell.sum_hess += hd;
                    }
                }
                block += block_step;
            }
        },
    );
}

/// Accumulate one block of sparse rows into `hist`.
///
/// Sparse entries hold global bins, which are `< total_bins` by construction.
#[inline]
fn build_sparse<T: BinIdx>(
    hist: &mut [GradStats],
    rows: &[u32],
    index: &[T],
    row_ptr: &[usize],
    gpair: &[GradientPair],
) {
    for &rid in rows {
        let rid = rid as usize;
        debug_assert!(rid < gpair.len());
        debug_assert!(rid + 1 < row_ptr.len());

        // SAFETY: `Partitioner` contains only valid row ids. `row_ptr` was
        // built with one terminal entry, and every stored sparse bin is a
        // global bin below `hist.len()`; the latter invariant is exercised by
        // `histogram_kernels_stay_in_bounds`.
        let g = unsafe { *gpair.get_unchecked(rid) };
        let (gd, hd) = (g.grad as f64, g.hess as f64);
        let begin = unsafe { *row_ptr.get_unchecked(rid) };
        let end = unsafe { *row_ptr.get_unchecked(rid + 1) };
        debug_assert!(begin <= end && end <= index.len());
        for position in begin..end {
            let bin = unsafe { (*index.get_unchecked(position)).idx() };
            let cell = unsafe { hist.get_unchecked_mut(bin) };
            cell.sum_grad += gd;
            cell.sum_hess += hd;
        }
    }
}

/// Accumulate `rows` into `hist`, dispatching on the index width.
fn build_hist(hist: &mut [GradStats], rows: &[u32], gi: &GHistIndex, gpair: &[GradientPair]) {
    if gi.is_dense {
        dispatch_bins!(&gi.index, |v| build_dense(
            hist,
            rows,
            v,
            gi.row_stride,
            &gi.offsets,
            gpair
        ))
    } else {
        dispatch_bins!(&gi.index, |v| build_sparse(hist, rows, v, &gi.row_ptr, gpair))
    }
}

/// The per-target sums added together, which is what the scalar bookkeeping —
/// node cover, split validity, the tree's own statistics — is stated in.
fn sum_targets(stats: &[GradStats]) -> GradStats {
    let mut out = GradStats::default();
    for s in stats {
        out.add_stats(s);
    }
    out
}

/// Lanes to use for a node: bounded by its block count, the lane cap, and the
/// memory budget. Depends only on the data, never on the thread count.
fn lanes_for(n_rows: usize, total_bins: usize) -> usize {
    let by_blocks = n_rows.div_ceil(BLOCK_ROWS).max(1);
    let by_memory = (LANE_BUDGET / (total_bins * size_of::<GradStats>()).max(1)).max(1);
    by_blocks.min(MAX_LANES).min(by_memory)
}

/// Grows one tree in place. This is the entry point of a boosting round.
pub struct HistGrower<'a> {
    param: &'a TrainParam,
    gi: &'a GHistIndex,
    dmat: &'a DMatrix,
    hist: HistCollection,
    partitioner: Partitioner,
    snode: Vec<NodeEntry>,
    /// Partial histograms, reused across nodes and rounds.
    lane_buf: Vec<GradStats>,
    /// Monotonicity: the per-node weight boxes and the split gain rule.
    evaluator: SplitEvaluator,
    /// Which features each node may split on.
    constraints: InteractionConstraints,
    /// Which features each node *has* to choose from, after column sampling.
    column_sampler: ColumnSampler,
    /// Outputs a leaf carries. `1` is the ordinary one-tree-per-target fit;
    /// more makes this a vector-leaf grower.
    n_targets: usize,
    /// Per-`(node, target)` gradient sums, indexed `nid * n_targets + t`. Only
    /// used by the vector-leaf path; the scalar path keeps them in [`Self::snode`].
    snode_targets: Vec<GradStats>,
    /// Per-target gradient columns, extracted once per round so the scalar
    /// accumulation kernel can be reused unchanged.
    target_gpair: Vec<Vec<GradientPair>>,
}

impl<'a> HistGrower<'a> {
    pub fn new(param: &'a TrainParam, gi: &'a GHistIndex, dmat: &'a DMatrix) -> Self {
        Self::new_multi(param, gi, dmat, 1)
    }

    /// A grower for vector-leaf trees: one tree covering `n_targets` outputs.
    pub fn new_multi(
        param: &'a TrainParam,
        gi: &'a GHistIndex,
        dmat: &'a DMatrix,
        n_targets: usize,
    ) -> Self {
        let n_features = dmat.num_col();
        let n_targets = n_targets.max(1);
        Self {
            param,
            gi,
            dmat,
            hist: HistCollection::new(
                gi.total_bins(),
                n_targets,
                param.max_cached_hist_node.min(usize::MAX as u64) as usize,
            ),
            partitioner: Partitioner::new(dmat.num_row()),
            snode: Vec::new(),
            lane_buf: Vec::new(),
            evaluator: SplitEvaluator::new(&param.monotone_constraints, n_features),
            constraints: InteractionConstraints::new(
                param.interaction_constraints.as_ref(),
                n_features,
            ),
            column_sampler: ColumnSampler::weighted(
                param.colsample_bynode,
                param.colsample_bylevel,
                param.colsample_bytree,
                &dmat.info().feature_weights,
            ),
            n_targets,
            snode_targets: Vec::new(),
            target_gpair: Vec::new(),
        }
    }

    /// Whether this grower produces vector leaves.
    #[inline]
    fn is_multi(&self) -> bool {
        self.n_targets > 1
    }

    /// Split the round's `(row, target)` gradients into one column per target.
    ///
    /// The histogram kernels take a gradient per row, so a vector-leaf round
    /// hands them one column at a time rather than teaching them a stride.
    fn split_gpair(&mut self, gpair: &[GradientPair]) {
        if !self.is_multi() {
            return;
        }
        let n_rows = self.dmat.num_row();
        self.target_gpair.resize_with(self.n_targets, Vec::new);
        for (t, column) in self.target_gpair.iter_mut().enumerate() {
            column.clear();
            column.extend((0..n_rows).map(|r| gpair[r * self.n_targets + t]));
        }
    }

    /// Per-`(node, target)` sums, as a slice over the node's targets.
    #[inline]
    fn node_targets(&self, nid: usize) -> &[GradStats] {
        &self.snode_targets[nid * self.n_targets..(nid + 1) * self.n_targets]
    }

    /// Reset for another tree over the same data, keeping allocations.
    pub fn reset(&mut self) {
        self.partitioner.reset(self.dmat.num_row());
        self.hist.clear();
        self.snode.clear();
        self.evaluator.reset();
        self.constraints.reset();
    }

    /// Grow `tree` from `gpair`, mirroring upstream's `UpdateTree`.
    ///
    /// `ctx` supplies the thread count and the session random engine; the
    /// engine is advanced here by the per-tree, per-level and per-node column
    /// samples, in that order.
    pub fn grow(&mut self, ctx: &mut Context, gpair: &[GradientPair], tree: &mut RegTree) {
        let threads = ctx.threads();
        self.split_gpair(gpair);
        // The column sample for a whole tree is drawn before any split is
        // considered, as `ColumnSampler::Init` is called from the evaluator's
        // constructor once per tree.
        self.column_sampler.reset(self.dmat.num_col(), ctx.rng());
        let rng = ctx.rng();
        threading::install_with(threads, || self.grow_inner(rng, gpair, tree));
    }

    fn grow_inner(&mut self, rng: &mut Mt19937, gpair: &[GradientPair], tree: &mut RegTree) {
        let mut num_leaves: i32 = 1;
        let root = self.init_root(rng, gpair, tree);

        let mut queue = ExpandQueue::new(self.param.grow_policy);
        if root.split.loss_chg > RT_EPS {
            queue.push(root);
        }
        let batch_capacity = match self.param.grow_policy {
            GrowPolicy::DepthWise => MAX_NODE_BATCH_SIZE,
            GrowPolicy::LossGuide => 1,
        };
        let mut expand_set = Vec::with_capacity(batch_capacity);
        queue.pop_batch(self.param, &mut num_leaves, &mut expand_set);

        while !expand_set.is_empty() {
            let mut valid_candidates = Vec::new();
            let mut tasks = Vec::with_capacity(expand_set.len());
            for candidate in &expand_set {
                self.apply_split(candidate, tree);
                tasks.push(self.split_task(candidate, tree));
                if self.is_child_valid(candidate, num_leaves) {
                    valid_candidates.push(candidate.clone());
                }
            }

            self.partitioner.split_all(&tasks, self.gi);

            // Histograms for every child of every expandable candidate are
            // built in one parallel pass, then their siblings are subtracted.
            let built = self.build_children_hists(&valid_candidates, tree, gpair);

            let depths: Vec<i32> = built.iter().map(|&nid| tree.depth(nid)).collect();
            let features = self.feature_sets(&depths, rng);
            let mut best_splits = Vec::with_capacity(built.len());
            let splits = self.evaluate_splits(&built, &features);
            for ((nid, (split, cat_bits)), depth) in built.iter().zip(splits).zip(depths) {
                best_splits.push(ExpandEntry { nid: *nid, depth, split, cat_bits });
            }

            // Parent histograms are dead once both children exist.
            for candidate in &expand_set {
                self.hist.release(candidate.nid);
            }

            for e in best_splits {
                if e.split.loss_chg > RT_EPS {
                    queue.push(e);
                }
            }
            queue.pop_batch(self.param, &mut num_leaves, &mut expand_set);
        }
    }

    /// Add each row's leaf value to `preds`, using the row sets rather than
    /// re-traversing the tree.
    ///
    /// `preds` is row-major `(row, group)`, so a multi-output fit writes into
    /// the column of the group this tree belongs to. `weight` is the tree's
    /// own weight, which is `1` outside DART.
    pub fn update_predictions(
        &self,
        tree: &RegTree,
        preds: &mut [f32],
        n_groups: usize,
        group: usize,
        weight: f32,
    ) {
        for nid in 0..tree.num_nodes() {
            if !tree.nodes[nid].is_leaf() {
                continue;
            }
            // A vector leaf writes every output column of the row; a scalar
            // one writes the single column its tree belongs to.
            let values = tree.leaf_value(nid);
            for &rid in self.partitioner.rows(nid) {
                let base = rid as usize * n_groups;
                for (t, v) in values.iter().enumerate() {
                    preds[base + group + t] += v * weight;
                }
            }
        }
    }

    /// Build the root histogram, set the root leaf, and evaluate its split.
    fn init_root(
        &mut self,
        rng: &mut Mt19937,
        gpair: &[GradientPair],
        tree: &mut RegTree,
    ) -> ExpandEntry {
        self.hist.allocate(0);
        self.build_hists(&[0], gpair);

        // Upstream sums the first feature's bins for dense data instead of the
        // gradients themselves; the two differ in rounding, so match the choice.
        let root_target_sum = |t: usize, this: &Self| -> GradStats {
            let mut sum = GradStats::default();
            if this.dmat.is_dense() {
                let cuts = &this.gi.cuts;
                let (b, e) = (cuts.cut_ptrs[0] as usize, cuts.cut_ptrs[1] as usize);
                for bin in b..e {
                    let cell = this.hist.get_target(0, t)[bin];
                    sum.add(cell.sum_grad, cell.sum_hess);
                }
            } else if this.is_multi() {
                for g in &this.target_gpair[t] {
                    sum.add(g.grad as f64, g.hess as f64);
                }
            }
            sum
        };

        let mut root_targets: Vec<GradStats> = Vec::with_capacity(self.n_targets);
        for t in 0..self.n_targets {
            let mut sum = root_target_sum(t, self);
            if !self.dmat.is_dense() && !self.is_multi() {
                sum = GradStats::default();
                for g in gpair {
                    sum.add(g.grad as f64, g.hess as f64);
                }
            }
            root_targets.push(sum);
        }
        let root_sum = sum_targets(&root_targets);

        self.snode.clear();
        self.snode.push(NodeEntry {
            stats: root_sum,
            root_gain: self.evaluator.calc_gain(0, self.param, &root_sum),
        });
        self.snode_targets.clear();
        self.snode_targets.extend_from_slice(&root_targets);

        if self.is_multi() {
            let weights: Vec<f32> = root_targets
                .iter()
                .map(|s| self.param.learning_rate * self.evaluator.calc_weight(0, self.param, s))
                .collect();
            tree.stats[0].sum_hess = root_sum.sum_hess as f32;
            tree.stats[0].base_weight = self.evaluator.calc_weight(0, self.param, &root_sum);
            tree.set_leaf_vector(0, &weights);
        } else {
            let weight = self.evaluator.calc_weight(0, self.param, &root_sum);
            tree.stats[0].sum_hess = root_sum.sum_hess as f32;
            tree.stats[0].base_weight = weight;
            tree.set_leaf(0, self.param.learning_rate * weight);
        }

        let features = self.feature_sets(&[0], rng);
        let (split, cat_bits) = self.evaluate_splits(&[0], &features).pop().unwrap_or_default();
        ExpandEntry { nid: 0, depth: 0, split, cat_bits }
    }

    /// The candidate features for each node in an expansion batch, in the same
    /// order as `depths`.
    ///
    /// Drawn here rather than inside the parallel evaluation because with
    /// `colsample_bynode < 1` each call advances the random engine, and the
    /// order of those draws is part of what the model is.
    fn feature_sets(&mut self, depths: &[i32], rng: &mut Mt19937) -> Vec<FeatureSet> {
        depths.iter().map(|&depth| self.column_sampler.feature_set(depth, rng)).collect()
    }

    fn is_child_valid(&self, parent: &ExpandEntry, num_leaves: i32) -> bool {
        if self.param.max_depth > 0 && parent.depth + 1 >= self.param.max_depth {
            return false;
        }
        if self.param.max_leaves > 0 && num_leaves >= self.param.max_leaves {
            return false;
        }
        true
    }

    fn apply_split(&mut self, candidate: &ExpandEntry, tree: &mut RegTree) {
        if self.is_multi() {
            self.apply_split_multi(candidate, tree);
            return;
        }
        let split = &candidate.split;
        let mut parent_sum = split.left_sum;
        parent_sum.add_stats(&split.right_sum);

        // All three weights are bounded by the *parent's* box: the children do
        // not have one until `add_split` below derives it from these values.
        let nid = candidate.nid;
        let base_weight = self.evaluator.calc_weight(nid, self.param, &parent_sum);
        let left_weight = self.evaluator.calc_weight(nid, self.param, &split.left_sum);
        let right_weight = self.evaluator.calc_weight(nid, self.param, &split.right_sum);
        let lr = self.param.learning_rate;

        if split.is_cat {
            tree.expand_categorical(
                candidate.nid,
                split.split_index(),
                &candidate.cat_bits,
                split.default_left(),
                base_weight,
                left_weight * lr,
                right_weight * lr,
                split.loss_chg,
                parent_sum.sum_hess as f32,
                split.left_sum.sum_hess as f32,
                split.right_sum.sum_hess as f32,
            );
        } else {
            tree.expand_node(
                candidate.nid,
                split.split_index(),
                split.split_value,
                split.default_left(),
                base_weight,
                left_weight * lr,
                right_weight * lr,
                split.loss_chg,
                parent_sum.sum_hess as f32,
                split.left_sum.sum_hess as f32,
                split.right_sum.sum_hess as f32,
            );
        }

        let node = tree.nodes[candidate.nid];
        let (left, right) = (node.left as usize, node.right as usize);
        self.evaluator.add_split(nid, left, right, node.split_index, left_weight, right_weight);

        if self.snode.len() < tree.num_nodes() {
            self.snode.resize(tree.num_nodes(), NodeEntry::default());
        }
        // Upstream evaluates both children's gains against the parent's node
        // id, so the parent's bounds — not the ones just derived — apply here.
        self.snode[left] = NodeEntry {
            stats: split.left_sum,
            root_gain: self.evaluator.calc_gain(nid, self.param, &split.left_sum),
        };
        self.snode[right] = NodeEntry {
            stats: split.right_sum,
            root_gain: self.evaluator.calc_gain(nid, self.param, &split.right_sum),
        };

        self.constraints.split(nid, node.split_index, left, right);
    }

    /// [`apply_split`](Self::apply_split) for a vector-leaf tree: one shared
    /// split decision, one leaf value per target.
    fn apply_split_multi(&mut self, candidate: &ExpandEntry, tree: &mut RegTree) {
        let split = &candidate.split;
        let nid = candidate.nid;
        let (left_sums, right_sums) = self.multi_child_sums(nid, split);

        let mut parent_sum = split.left_sum;
        parent_sum.add_stats(&split.right_sum);
        let base_weight = self.evaluator.calc_weight(nid, self.param, &parent_sum);
        let lr = self.param.learning_rate;
        let left_weights: Vec<f32> = left_sums
            .iter()
            .map(|s| lr * self.evaluator.calc_weight(nid, self.param, s))
            .collect();
        let right_weights: Vec<f32> = right_sums
            .iter()
            .map(|s| lr * self.evaluator.calc_weight(nid, self.param, s))
            .collect();

        tree.expand_node_multi(
            nid,
            split.split_index(),
            split.split_value,
            split.default_left(),
            base_weight,
            &left_weights,
            &right_weights,
            split.loss_chg,
            parent_sum.sum_hess as f32,
            split.left_sum.sum_hess as f32,
            split.right_sum.sum_hess as f32,
        );

        let node = tree.nodes[nid];
        let (left, right) = (node.left as usize, node.right as usize);
        // Monotonicity is stated per feature, not per target, so the bounds are
        // handed on from the summed weights just as the scalar path does.
        self.evaluator.add_split(
            nid,
            left,
            right,
            node.split_index,
            self.evaluator.calc_weight(nid, self.param, &split.left_sum),
            self.evaluator.calc_weight(nid, self.param, &split.right_sum),
        );

        if self.snode.len() < tree.num_nodes() {
            self.snode.resize(tree.num_nodes(), NodeEntry::default());
        }
        if self.snode_targets.len() < tree.num_nodes() * self.n_targets {
            self.snode_targets.resize(tree.num_nodes() * self.n_targets, GradStats::default());
        }
        for (child, sums) in [(left, &left_sums), (right, &right_sums)] {
            let total = sum_targets(sums);
            self.snode[child] = NodeEntry {
                stats: total,
                root_gain: self.evaluator.calc_gain(nid, self.param, &total),
            };
            self.snode_targets[child * self.n_targets..(child + 1) * self.n_targets]
                .copy_from_slice(sums);
        }

        self.constraints.split(nid, node.split_index, left, right);
    }

    fn split_task(&self, candidate: &ExpandEntry, tree: &RegTree) -> SplitTask {
        let node = tree.nodes[candidate.nid];
        let fidx = node.split_index;
        if candidate.split.is_cat {
            return SplitTask {
                nid: candidate.nid,
                left: node.left as usize,
                right: node.right as usize,
                fidx,
                local_cond: -1,
                default_left: node.default_left,
                cat_bits: candidate.cat_bits.clone(),
            };
        }
        let global_cond = self.find_split_condition(fidx, node.value);
        // Feature-local threshold for the dense path. `-1` (no matching cut)
        // stays negative so every present row goes right.
        let local_cond = if global_cond < 0 {
            -1
        } else {
            global_cond - self.gi.cuts.cut_ptrs[fidx as usize] as i64
        };
        SplitTask {
            nid: candidate.nid,
            left: node.left as usize,
            right: node.right as usize,
            fidx,
            local_cond,
            default_left: node.default_left,
            cat_bits: Vec::new(),
        }
    }

    /// Bin index whose cut value equals the split threshold, or `-1`.
    fn find_split_condition(&self, fidx: u32, split_pt: f32) -> i64 {
        let cuts = &self.gi.cuts;
        let (lo, hi) = (cuts.cut_ptrs[fidx as usize], cuts.cut_ptrs[fidx as usize + 1]);
        let mut split_cond: i64 = -1;
        for bound in lo..hi {
            if split_pt == cuts.cut_values[bound as usize] {
                split_cond = bound as i64;
            }
        }
        split_cond
    }

    /// Build the cheaper child of every candidate, then subtract for the other.
    ///
    /// Returns every child node id, built and subtracted alike.
    fn build_children_hists(
        &mut self,
        candidates: &[ExpandEntry],
        tree: &RegTree,
        gpair: &[GradientPair],
    ) -> Vec<usize> {
        if candidates.is_empty() {
            return Vec::new();
        }
        let mut to_build = Vec::with_capacity(candidates.len());
        let mut to_subtract = Vec::with_capacity(candidates.len());
        for c in candidates {
            let node = tree.nodes[c.nid];
            let (left, right) = (node.left as usize, node.right as usize);
            // Build the child with the smaller Hessian sum.
            let fewer_right = c.split.right_sum.sum_hess < c.split.left_sum.sum_hess;
            let (build, subtract) = if fewer_right { (right, left) } else { (left, right) };
            to_build.push(build);
            to_subtract.push((subtract, c.nid, build));
        }

        for nid in &to_build {
            self.hist.allocate(*nid);
        }
        self.build_hists(&to_build, gpair);

        for (subtract, _, _) in &to_subtract {
            self.hist.allocate(*subtract);
        }
        // The subtraction is per `(node, target)`: a vector-leaf node holds one
        // histogram per target, and each is `parent - built` in its own right.
        let n_targets = self.n_targets;
        let slots: Vec<(usize, usize, usize)> = to_subtract
            .iter()
            .flat_map(|&(subtract, parent, built)| {
                (0..n_targets).map(move |t| {
                    (subtract * n_targets + t, parent * n_targets + t, built * n_targets + t)
                })
            })
            .collect();
        let mut dsts: Vec<Vec<GradStats>> =
            slots.iter().map(|&(dst, _, _)| std::mem::take(&mut self.hist.data[dst])).collect();
        {
            let hist = &self.hist;
            dsts.par_iter_mut().enumerate().for_each(|(i, dst)| {
                let (_, parent, built) = slots[i];
                let p = &hist.data[parent];
                let b = &hist.data[built];
                dst.par_chunks_mut(REDUCE_CHUNK_BINS).enumerate().for_each(|(c, out)| {
                    let lo = c * REDUCE_CHUNK_BINS;
                    for (k, o) in out.iter_mut().enumerate() {
                        o.sum_grad = p[lo + k].sum_grad - b[lo + k].sum_grad;
                        o.sum_hess = p[lo + k].sum_hess - b[lo + k].sum_hess;
                    }
                });
            });
        }
        for (&(dst_slot, _, _), dst) in slots.iter().zip(dsts) {
            self.hist.data[dst_slot] = dst;
        }

        let mut out = Vec::with_capacity(candidates.len() * 2);
        for c in candidates {
            let node = tree.nodes[c.nid];
            out.push(node.left as usize);
            out.push(node.right as usize);
        }
        out
    }

    /// Build histograms for `nodes` in one parallel pass.
    ///
    /// Every per-bin step — zeroing, accumulation, reduction — runs in
    /// parallel; leaving any of them serial caps the whole level's speedup.
    fn build_hists(&mut self, nodes: &[usize], gpair: &[GradientPair]) {
        if self.is_multi() {
            // One scalar pass per target, over the same row sets.
            for t in 0..self.n_targets {
                let column = std::mem::take(&mut self.target_gpair[t]);
                self.build_hists_target(nodes, &column, t);
                self.target_gpair[t] = column;
            }
            return;
        }
        self.build_hists_target(nodes, gpair, 0);
    }

    /// Build one target's histogram for every node in `nodes`.
    fn build_hists_target(
        &mut self,
        nodes: &[usize],
        gpair: &[GradientPair],
        target: usize,
    ) {
        let total_bins = self.hist.total_bins;
        let lanes: Vec<usize> = nodes
            .iter()
            .map(|&nid| lanes_for(self.partitioner.rows(nid).len(), total_bins))
            .collect();

        // Multi-lane nodes get a slice of the shared partial-histogram buffer;
        // single-lane nodes accumulate straight into their own histogram.
        let mut lane_owner: Vec<(usize, usize)> = Vec::new();
        let mut lane_start = vec![usize::MAX; nodes.len()];
        for (i, &l) in lanes.iter().enumerate() {
            if l > 1 {
                lane_start[i] = lane_owner.len();
                for local in 0..l {
                    lane_owner.push((i, local));
                }
            }
        }

        self.lane_buf.resize(lane_owner.len() * total_bins, GradStats::default());
        let gi = self.gi;
        let partitioner = &self.partitioner;
        let feature_groups = self.param.grow_policy == GrowPolicy::LossGuide && gi.is_dense;
        let mut targets: Vec<Vec<GradStats>> =
            nodes.iter().map(|&nid| std::mem::take(&mut self.hist.data[nid * self.n_targets + target])).collect();

        // Single-lane nodes: zero and accumulate in one pass over the node set.
        //
        // Zeroing here is a bulk write with no ordering to preserve, so it
        // could in principle be spread further when the batch is one node —
        // that was measured and made `lossguide` slower, because the nested
        // dispatch costs more than the memset it splits. Accumulation could not
        // be spread further in any case: which rows land in which lane fixes
        // the summation order, and that is part of the model.
        targets
            .par_iter_mut()
            .enumerate()
            .filter(|(i, _)| lanes[*i] == 1)
            .for_each(|(i, dst)| {
                dst.fill(GradStats::default());
                let rows = partitioner.rows(nodes[i]);
                let threads = rayon::current_num_threads();
                if feature_groups
                    && threads > 1
                    && rows.len() * gi.row_stride >= FEATURE_GROUP_MIN_UPDATES
                {
                    build_dense_feature_groups(dst, rows, gi, gpair, 0, 1, threads);
                } else {
                    build_hist(dst, rows, gi, gpair);
                }
            });

        if !lane_owner.is_empty() {
            // Lane `l` of a node takes blocks `l, l + lanes, ...` in increasing
            // order, so its partial sum is fixed by the data alone.
            self.lane_buf
                .par_chunks_mut(total_bins)
                .zip(lane_owner.par_iter())
                .for_each(|(hist, &(node_idx, local_lane))| {
                    hist.fill(GradStats::default());
                    let rows = partitioner.rows(nodes[node_idx]);
                    let n_blocks = rows.len().div_ceil(BLOCK_ROWS);
                    let threads = rayon::current_num_threads();
                    if feature_groups
                        && lanes[node_idx] * 2 <= threads
                        && rows.len() * gi.row_stride >= FEATURE_GROUP_MIN_UPDATES
                    {
                        build_dense_feature_groups(
                            hist,
                            rows,
                            gi,
                            gpair,
                            local_lane,
                            lanes[node_idx],
                            threads.div_ceil(lanes[node_idx]),
                        );
                        return;
                    }
                    let mut b = local_lane;
                    while b < n_blocks {
                        let lo = b * BLOCK_ROWS;
                        let hi = ((b + 1) * BLOCK_ROWS).min(rows.len());
                        build_hist(hist, &rows[lo..hi], gi, gpair);
                        b += lanes[node_idx];
                    }
                });

            // Reduce lanes in lane order. Nodes are independent, and within a
            // node the bin range is split so a single big node still scales.
            let lane_buf = &self.lane_buf;
            targets
                .par_iter_mut()
                .enumerate()
                .filter(|(i, _)| lanes[*i] > 1)
                .for_each(|(i, dst)| {
                    let start = lane_start[i];
                    let n = lanes[i];
                    dst.par_chunks_mut(REDUCE_CHUNK_BINS).enumerate().for_each(|(c, out)| {
                        let lo = c * REDUCE_CHUNK_BINS;
                        let hi = lo + out.len();
                        out.copy_from_slice(&lane_buf[start * total_bins + lo..start * total_bins + hi]);
                        for lane in 1..n {
                            let base = (start + lane) * total_bins;
                            for (d, s) in out.iter_mut().zip(&lane_buf[base + lo..base + hi]) {
                                d.sum_grad += s.sum_grad;
                                d.sum_hess += s.sum_hess;
                            }
                        }
                    });
                });
        }

        for (i, &nid) in nodes.iter().enumerate() {
            self.hist.data[nid * self.n_targets + target] = std::mem::take(&mut targets[i]);
        }
    }

    /// Best split for each node, considering only that node's candidate
    /// features.
    ///
    /// Candidates are evaluated over a flat `(node, feature)` task list so a
    /// level with many small nodes parallelises as well as one big node, then
    /// merged per node. The merge rule is a total order on `(loss change,
    /// smallest feature index)`, so the result does not depend on how the work
    /// was scheduled.
    ///
    /// Chunking the task list into fatter jobs was measured and made no
    /// difference: rayon's adaptive splitting already stops well short of
    /// costing more than it saves, on both growth policies.
    fn evaluate_splits(
        &self,
        nodes: &[usize],
        features: &[FeatureSet],
    ) -> Vec<(SplitEntry, Vec<u32>)> {
        debug_assert_eq!(nodes.len(), features.len());
        // Flat task list: one entry per (node, candidate feature) pair.
        let tasks: Vec<(usize, u32)> = features
            .iter()
            .enumerate()
            .flat_map(|(node_idx, set)| set.iter().map(move |&fidx| (node_idx, fidx)))
            .collect();

        let per_task: Vec<(SplitEntry, Vec<u32>)> = tasks
            .par_iter()
            .map(|&(node_idx, fidx)| {
                if self.is_multi() {
                    (self.evaluate_one_multi(nodes[node_idx], fidx), Vec::new())
                } else {
                    self.evaluate_one(nodes[node_idx], fidx)
                }
            })
            .collect();

        let mut out = vec![(SplitEntry::default(), Vec::new()); nodes.len()];
        for ((node_idx, _), (candidate, bits)) in tasks.iter().zip(&per_task) {
            let slot = &mut out[*node_idx];
            if slot.0.update_entry(candidate) {
                // The categories belong to the split that just won; a numeric
                // winner clears whatever a categorical feature had proposed.
                slot.1.clear();
                slot.1.extend_from_slice(bits);
            }
        }
        out
    }

    /// The best split of one node on one feature, with the categories it sends
    /// right when that split is categorical.
    #[inline]
    fn evaluate_one(&self, nid: usize, fidx: u32) -> (SplitEntry, Vec<u32>) {
        let mut best = SplitEntry::default();
        if !self.constraints.query(nid, fidx) {
            return (best, Vec::new());
        }
        let hist = self.hist.get(nid);
        let parent = self.snode[nid];

        if self.gi.cuts.is_cat(fidx as usize) {
            let n_bins = self.gi.cuts.feature_bins(fidx as usize);
            // `common::UseOneHot`: few enough categories that testing each one
            // on its own beats partitioning them.
            let bits = if (n_bins as u32) < self.param.max_cat_to_onehot {
                self.enumerate_one_hot(nid, fidx, hist, &parent, &mut best)
            } else {
                self.enumerate_partition(nid, fidx, hist, &parent, &mut best)
            };
            return (best, bits);
        }

        let non_missing = self.enumerate_forward(nid, fidx, hist, &parent, &mut best);
        // A feature has missing values in this node exactly when its bins do
        // not account for the node's whole gradient sum.
        if non_missing.sum_grad != parent.stats.sum_grad
            || non_missing.sum_hess != parent.stats.sum_hess
        {
            self.enumerate_backward(nid, fidx, hist, &parent, &mut best);
        }
        (best, Vec::new())
    }

    /// `EnumerateOneHot` — try each category on its own against all the others.
    ///
    /// Each category is scanned twice, once with the feature's missing rows
    /// grouped with the other categories and once with the chosen one, which is
    /// how the default direction is learned without a dedicated missing bin.
    fn enumerate_one_hot(
        &self,
        nid: usize,
        fidx: u32,
        hist: &[GradStats],
        parent: &NodeEntry,
        p_best: &mut SplitEntry,
    ) -> Vec<u32> {
        let cuts = &self.gi.cuts;
        let f = fidx as usize;
        let (ibegin, iend) = (cuts.cut_ptrs[f] as usize, cuts.cut_ptrs[f + 1] as usize);

        // Rows whose value is missing: whatever the feature's bins do not hold.
        let mut feature_sum = GradStats::default();
        for cell in &hist[ibegin..iend] {
            feature_sum.add(cell.sum_grad, cell.sum_hess);
        }
        let mut missing = GradStats::default();
        missing.set_subtract(&parent.stats, &feature_sum);

        let mut best = SplitEntry::default();
        for i in ibegin..iend {
            let split_pt = cuts.cut_values[i];

            // Missing on the left: the chosen category alone goes right.
            let mut right_sum = GradStats::new(hist[i].sum_grad, hist[i].sum_hess);
            let mut left_sum = GradStats::default();
            left_sum.set_subtract(&parent.stats, &right_sum);
            let gain = self.evaluator.calc_split_gain(nid, fidx, self.param, &left_sum, &right_sum);
            if gain.is_finite() {
                best.update_cat(gain - parent.root_gain, fidx, split_pt, true, left_sum, right_sum);
            }

            // Missing on the right: grouped with the chosen category.
            right_sum.add_stats(&missing);
            left_sum.set_subtract(&parent.stats, &right_sum);
            let gain = self.evaluator.calc_split_gain(nid, fidx, self.param, &left_sum, &right_sum);
            if gain.is_finite() {
                best.update_cat(gain - parent.root_gain, fidx, split_pt, false, left_sum, right_sum);
            }
        }

        let mut bits = Vec::new();
        if best.is_cat {
            bits = vec![0u32; cat::storage_size(iend - ibegin + 1)];
            cat::set_bit(&mut bits, cat::as_cat(best.split_value));
        }
        p_best.update_entry(&best);
        bits
    }

    /// `EnumeratePart` — partition the categories by the weight their gradients
    /// imply, then split that order like an ordinary numeric feature.
    ///
    /// Sorting by weight is what makes a contiguous run of the sorted order an
    /// optimal category subset. Both scan directions are tried, because they
    /// differ in which side the feature's missing rows land on.
    fn enumerate_partition(
        &self,
        nid: usize,
        fidx: u32,
        hist: &[GradStats],
        parent: &NodeEntry,
        p_best: &mut SplitEntry,
    ) -> Vec<u32> {
        let cuts = &self.gi.cuts;
        let f = fidx as usize;
        let (f_begin, f_end) = (cuts.cut_ptrs[f] as usize, cuts.cut_ptrs[f + 1] as usize);
        let n_bins_feature = f_end - f_begin;
        // `max_cat_threshold` caps how many categories one side may name.
        let n_bins = (self.param.max_cat_threshold as usize).min(n_bins_feature);
        if n_bins < 2 {
            return Vec::new();
        }

        // `CalcWeightCat`: the unconstrained weight. Categories carry no
        // monotonicity, so the node's weight box is deliberately not applied.
        let mut sorted_idx: Vec<usize> = (0..n_bins_feature).collect();
        sorted_idx.sort_by(|&l, &r| {
            let wl = calc_weight(self.param, &hist[f_begin + l]);
            let wr = calc_weight(self.param, &hist[f_begin + r]);
            wl.total_cmp(&wr).then_with(|| l.cmp(&r))
        });

        let mut best = SplitEntry::default();
        // How many of the sorted categories the winning split sends right.
        let mut best_partition: Option<usize> = None;

        for forward in [true, false] {
            let mut left_sum = GradStats::default();
            let mut right_sum = GradStats::default();
            for step in 0..n_bins - 1 {
                let j = if forward { step } else { n_bins_feature - 1 - step };
                let cell = hist[f_begin + sorted_idx[j]];
                if forward {
                    // Scanning up the order, the accumulated head goes right
                    // and the feature's missing rows stay left.
                    right_sum.add(cell.sum_grad, cell.sum_hess);
                    left_sum.set_subtract(&parent.stats, &right_sum);
                } else {
                    left_sum.add(cell.sum_grad, cell.sum_hess);
                    right_sum.set_subtract(&parent.stats, &left_sum);
                }
                let gain =
                    self.evaluator.calc_split_gain(nid, fidx, self.param, &left_sum, &right_sum);
                if gain.is_finite()
                    && best.update_cat(
                        gain - parent.root_gain,
                        fidx,
                        f32::NAN,
                        forward,
                        left_sum,
                        right_sum,
                    )
                {
                    // Forward: the first `step + 1` of the order go right.
                    // Backward: everything from `n_bins_feature - 1 - step`
                    // upwards is the left side, so the right side is the head.
                    best_partition =
                        Some(if forward { step + 1 } else { n_bins_feature - 1 - step });
                }
            }
        }

        let mut bits = Vec::new();
        if let Some(partition) = best_partition {
            debug_assert!(partition > 0 && partition <= n_bins_feature);
            bits = vec![0u32; cat::storage_size(n_bins_feature)];
            // The head of the order is the right-hand side either way: a
            // forward scan accumulates it into `right_sum` directly, and a
            // backward scan leaves it as whatever the left side did not absorb.
            for &c in &sorted_idx[..partition] {
                cat::set_bit(&mut bits, cat::as_cat(cuts.cut_values[f_begin + c]));
            }
        }
        p_best.update_entry(&best);
        bits
    }

    /// The gain of a vector-leaf node, `TreeEvaluator::CalcGain` over targets.
    ///
    /// Every target contributes its own regularised gain at its own weight;
    /// the split itself is shared, which is the whole point of a vector leaf.
    fn multi_gain(&self, nid: usize, stats: &[GradStats]) -> f32 {
        stats.iter().map(|s| self.evaluator.calc_gain(nid, self.param, s)).sum()
    }

    /// `CalcSplitGain` for vector leaves: the summed per-target gain, rejected
    /// as a whole when the children's *mean* hessian is too small.
    ///
    /// The validity test uses the mean rather than each target's own hessian
    /// because the split is one decision for all of them, so `min_child_weight`
    /// is a statement about the node, not about any single output.
    fn multi_split_gain(
        &self,
        nid: usize,
        fidx: u32,
        left: &[GradStats],
        right: &[GradStats],
    ) -> f32 {
        let k = self.n_targets as f64;
        let left_hess: f64 = left.iter().map(|s| s.sum_hess).sum::<f64>() / k;
        let right_hess: f64 = right.iter().map(|s| s.sum_hess).sum::<f64>() / k;
        let mean_left = GradStats::new(0.0, left_hess);
        let mean_right = GradStats::new(0.0, right_hess);
        // Reuse the scalar validity rule, which is what upstream does with the
        // averaged hessians.
        if !self.evaluator.calc_split_gain(nid, fidx, self.param, &mean_left, &mean_right).is_finite()
        {
            return f32::NEG_INFINITY;
        }
        let mut gain = 0.0f32;
        for (l, r) in left.iter().zip(right) {
            gain += self.evaluator.calc_gain(nid, self.param, l);
            gain += self.evaluator.calc_gain(nid, self.param, r);
        }
        gain
    }

    /// The best split of one node on one feature, for a vector-leaf tree.
    ///
    /// Structurally the scalar scan, run over `n_targets` histograms at once:
    /// one accumulator per target, one shared gain.
    fn evaluate_one_multi(&self, nid: usize, fidx: u32) -> SplitEntry {
        let mut best = SplitEntry::default();
        if !self.constraints.query(nid, fidx) {
            return best;
        }
        let cuts = &self.gi.cuts;
        let f = fidx as usize;
        let (ibegin, iend) = (cuts.cut_ptrs[f] as usize, cuts.cut_ptrs[f + 1] as usize);
        let parent = self.node_targets(nid);
        let parent_gain = self.multi_gain(nid, parent);

        let mut left = vec![GradStats::default(); self.n_targets];
        let mut right = vec![GradStats::default(); self.n_targets];

        // Forward: the head of the feature's bins goes left, and the rows with
        // no value for it go right.
        let mut non_missing = GradStats::default();
        for i in ibegin..iend {
            for t in 0..self.n_targets {
                let cell = self.hist.get_target(nid, t)[i];
                left[t].add(cell.sum_grad, cell.sum_hess);
                right[t].set_subtract(&parent[t], &left[t]);
            }
            let gain = self.multi_split_gain(nid, fidx, &left, &right);
            if gain.is_finite() {
                best.update(
                    gain - parent_gain,
                    fidx,
                    cuts.cut_values[i],
                    false,
                    sum_targets(&left),
                    sum_targets(&right),
                );
            }
        }
        for t in 0..self.n_targets {
            non_missing.add_stats(&left[t]);
        }

        // Backward, only when the feature has missing rows in this node.
        let parent_total = sum_targets(parent);
        if non_missing.sum_grad != parent_total.sum_grad
            || non_missing.sum_hess != parent_total.sum_hess
        {
            for s in right.iter_mut() {
                *s = GradStats::default();
            }
            for i in (ibegin..iend).rev() {
                for t in 0..self.n_targets {
                    let cell = self.hist.get_target(nid, t)[i];
                    // Scanning backwards, the accumulator is the right side.
                    right[t].add(cell.sum_grad, cell.sum_hess);
                    left[t].set_subtract(&parent[t], &right[t]);
                }
                let gain = self.multi_split_gain(nid, fidx, &left, &right);
                if gain.is_finite() {
                    best.update(
                        gain - parent_gain,
                        fidx,
                        cuts.backward_split_point(f, i),
                        true,
                        sum_targets(&left),
                        sum_targets(&right),
                    );
                }
            }
        }
        best
    }

    /// Recover a chosen split's per-target child sums by re-scanning the
    /// node's histograms.
    ///
    /// The split search keeps only the summed-over-targets totals, because
    /// [`SplitEntry`] is copied by the thousand and a per-target vector cannot
    /// ride along. Rebuilding them here costs one pass over one feature's bins
    /// per applied split, against a full scan per candidate during the search.
    fn multi_child_sums(
        &self,
        nid: usize,
        split: &SplitEntry,
    ) -> (Vec<GradStats>, Vec<GradStats>) {
        let cuts = &self.gi.cuts;
        let f = split.split_index() as usize;
        let (ibegin, iend) = (cuts.cut_ptrs[f] as usize, cuts.cut_ptrs[f + 1] as usize);
        let parent = self.node_targets(nid);

        // Bins at or below `cond` take the left branch; the rows with no value
        // are on whichever side `default_left` names, and are recovered as the
        // difference from the parent.
        let cond = self.find_split_condition(split.split_index(), split.split_value);
        let mut accumulated = vec![GradStats::default(); self.n_targets];
        let range: Box<dyn Iterator<Item = usize>> = if split.default_left() {
            // The right side is the bins strictly above `cond`.
            Box::new((cond.max(-1) as usize + 1).max(ibegin)..iend)
        } else {
            Box::new(ibegin..=(cond as usize).min(iend - 1))
        };
        for i in range {
            for t in 0..self.n_targets {
                let cell = self.hist.get_target(nid, t)[i];
                accumulated[t].add(cell.sum_grad, cell.sum_hess);
            }
        }

        let mut left = vec![GradStats::default(); self.n_targets];
        let mut right = vec![GradStats::default(); self.n_targets];
        for t in 0..self.n_targets {
            if split.default_left() {
                right[t] = accumulated[t];
                left[t].set_subtract(&parent[t], &right[t]);
            } else {
                left[t] = accumulated[t];
                right[t].set_subtract(&parent[t], &left[t]);
            }
        }
        (left, right)
    }

    fn enumerate_forward(
        &self,
        nid: usize,
        fidx: u32,
        hist: &[GradStats],
        parent: &NodeEntry,
        p_best: &mut SplitEntry,
    ) -> GradStats {
        let cuts = &self.gi.cuts;
        let p = self.param;
        let f = fidx as usize;
        let (ibegin, iend) = (cuts.cut_ptrs[f] as usize, cuts.cut_ptrs[f + 1] as usize);

        let mut left_sum = GradStats::default();
        let mut best = SplitEntry::default();
        for i in ibegin..iend {
            left_sum.add(hist[i].sum_grad, hist[i].sum_hess);
            let mut right_sum = GradStats::default();
            right_sum.set_subtract(&parent.stats, &left_sum);
            let gain = self.evaluator.calc_split_gain(nid, fidx, p, &left_sum, &right_sum);
            if gain.is_finite() {
                best.update(
                    gain - parent.root_gain,
                    fidx,
                    cuts.cut_values[i],
                    false,
                    left_sum,
                    right_sum,
                );
            }
        }
        p_best.update_entry(&best);
        left_sum
    }

    fn enumerate_backward(
        &self,
        nid: usize,
        fidx: u32,
        hist: &[GradStats],
        parent: &NodeEntry,
        p_best: &mut SplitEntry,
    ) {
        let cuts = &self.gi.cuts;
        let p = self.param;
        let f = fidx as usize;
        let (ibegin, iend) = (cuts.cut_ptrs[f] as usize, cuts.cut_ptrs[f + 1] as usize);

        let mut left_sum = GradStats::default();
        let mut best = SplitEntry::default();
        for i in (ibegin..iend).rev() {
            // Scanning backwards, `left_sum` accumulates the right-hand side.
            left_sum.add(hist[i].sum_grad, hist[i].sum_hess);
            let mut right_sum = GradStats::default();
            right_sum.set_subtract(&parent.stats, &left_sum);
            let gain = self.evaluator.calc_split_gain(nid, fidx, p, &right_sum, &left_sum);
            if gain.is_finite() {
                let split_pt = cuts.backward_split_point(f, i);
                best.update(
                    gain - parent.root_gain,
                    fidx,
                    split_pt,
                    true,
                    right_sum,
                    left_sum,
                );
            }
        }
        p_best.update_entry(&best);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::cuts::build_cuts;
    use crate::data::gradient_index::build_gradient_index;

    /// A step function: rows with x < 0.5 have gradient -1, the rest +1.
    fn step_data(n: usize) -> (DMatrix, Vec<GradientPair>) {
        let x: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
        let d = DMatrix::from_dense(&x, n, 1, f32::NAN).unwrap();
        let gpair = (0..n)
            .map(|i| {
                let g = if (i as f32 / n as f32) < 0.5 { -1.0 } else { 1.0 };
                GradientPair { grad: g, hess: 1.0 }
            })
            .collect();
        (d, gpair)
    }

    fn grow(d: &DMatrix, gpair: &[GradientPair], param: &TrainParam) -> RegTree {
        let cuts = build_cuts(d, param.max_bin).unwrap();
        let gi = build_gradient_index(d, &cuts).unwrap();
        let mut tree = RegTree::new(d.num_col());
        HistGrower::new(param, &gi, d).grow(&mut Context::default(), gpair, &mut tree);
        tree
    }

    #[test]
    fn blocked_partition_matches_serial_with_dense_missing_values() {
        let n = PARALLEL_PARTITION_MIN + 17;
        let values: Vec<f32> = (0..n)
            .map(|i| {
                if i % 7 == 0 { f32::NAN } else { ((i * 37) % 101) as f32 / 101.0 }
            })
            .collect();
        let d = DMatrix::from_dense(&values, n, 1, f32::NAN).unwrap();
        let cuts = build_cuts(&d, 32).unwrap();
        let gi = build_gradient_index(&d, &cuts).unwrap();
        let task = SplitTask {
            nid: 0,
            left: 1,
            right: 2,
            fidx: 0,
            local_cond: 15,
            default_left: true,
            cat_bits: Vec::new(),
        };

        let mut serial_src: Vec<u32> = (0..n as u32).rev().collect();
        let mut serial_dst = vec![0; n];
        let serial_left = partition_block(&serial_src, &mut serial_dst, &task, &gi);
        serial_src.copy_from_slice(&serial_dst);

        let mut blocked_src: Vec<u32> = (0..n as u32).rev().collect();
        let mut blocked_dst = vec![0; n];
        let mut decisions = vec![0; n];
        let blocked_left =
            partition_blocked(&mut blocked_src, &mut blocked_dst, &mut decisions, &task, &gi);

        assert_eq!(blocked_left, serial_left);
        assert_eq!(blocked_src, serial_src);
    }

    #[test]
    fn finds_the_obvious_split_and_signs_the_leaves() {
        let (d, gpair) = step_data(64);
        let param =
            TrainParam { max_depth: 1, learning_rate: 1.0, max_bin: 32, ..Default::default() };
        let tree = grow(&d, &gpair, &param);

        assert_eq!(tree.num_nodes(), 3, "one split expected");
        assert_eq!(tree.nodes[0].split_index, 0);
        assert!(tree.nodes[0].value > 0.45 && tree.nodes[0].value <= 0.55, "split near 0.5");
        // Leaf weight is -G/(H+lambda): negative gradients give a positive leaf.
        assert!(tree.nodes[1].value > 0.0, "left leaf takes the negative gradients");
        assert!(tree.nodes[2].value < 0.0);
    }

    #[test]
    fn max_leaves_one_leaves_a_stump() {
        let (d, gpair) = step_data(64);
        let param =
            TrainParam { max_depth: 1, max_leaves: 1, max_bin: 32, ..Default::default() };
        assert_eq!(grow(&d, &gpair, &param).num_nodes(), 1);
    }

    #[test]
    fn min_child_weight_blocks_tiny_children() {
        let (d, gpair) = step_data(64);
        // Every child would need 1000 hessian; only 64 rows exist.
        let param = TrainParam { min_child_weight: 1000.0, max_bin: 32, ..Default::default() };
        assert_eq!(grow(&d, &gpair, &param).num_nodes(), 1);
    }

    fn expand_entry(nid: usize, depth: i32, loss_chg: f32) -> ExpandEntry {
        let child = GradStats { sum_hess: 1.0, ..Default::default() };
        ExpandEntry {
            nid,
            depth,
            split: SplitEntry { loss_chg, left_sum: child, right_sum: child, ..Default::default() },
            ..Default::default()
        }
    }

    #[test]
    fn depthwise_queue_preserves_node_order_and_level_batches() {
        let param = TrainParam { grow_policy: GrowPolicy::DepthWise, ..Default::default() };
        let mut queue = ExpandQueue::new(param.grow_policy);
        for entry in [
            expand_entry(1, 1, 1.0),
            expand_entry(2, 1, 3.0),
            expand_entry(3, 2, 2.0),
        ] {
            queue.push(entry);
        }

        let mut num_leaves = 1;
        let mut batch = Vec::new();
        queue.pop_batch(&param, &mut num_leaves, &mut batch);
        assert_eq!(batch.iter().map(|entry| entry.nid).collect::<Vec<_>>(), [1, 2]);
        queue.pop_batch(&param, &mut num_leaves, &mut batch);
        assert_eq!(batch.iter().map(|entry| entry.nid).collect::<Vec<_>>(), [3]);
    }

    #[test]
    fn lossguide_queue_prioritises_gain_then_smaller_node_id() {
        let param = TrainParam { grow_policy: GrowPolicy::LossGuide, ..Default::default() };
        let mut queue = ExpandQueue::new(param.grow_policy);
        for entry in [
            expand_entry(1, 1, 1.0),
            expand_entry(3, 1, 3.0),
            expand_entry(2, 1, 3.0),
        ] {
            queue.push(entry);
        }

        let mut num_leaves = 1;
        let mut batch = Vec::new();
        let mut order = Vec::new();
        for _ in 0..3 {
            queue.pop_batch(&param, &mut num_leaves, &mut batch);
            order.push(batch[0].nid);
        }
        assert_eq!(order, [2, 3, 1]);
    }

    #[test]
    fn partitioning_is_stable_and_covers_every_row() {
        let (d, gpair) = step_data(64);
        let param = TrainParam { max_depth: 3, max_bin: 32, ..Default::default() };
        let cuts = build_cuts(&d, param.max_bin).unwrap();
        let gi = build_gradient_index(&d, &cuts).unwrap();
        let mut tree = RegTree::new(1);
        let mut grower = HistGrower::new(&param, &gi, &d);
        grower.grow(&mut Context::default(), &gpair, &mut tree);

        let mut seen: Vec<u32> = grower.partitioner.row_indices.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..d.num_row() as u32).collect::<Vec<_>>());

        // Every leaf's rows are ascending: partitioning preserved input order.
        for nid in 0..tree.num_nodes() {
            if tree.nodes[nid].is_leaf() {
                let rows = grower.partitioner.rows(nid);
                assert!(rows.windows(2).all(|w| w[0] < w[1]), "node {nid} lost row order");
            }
        }
    }

    /// The parallel partition path (large nodes) must agree with the serial one.
    #[test]
    fn parallel_and_serial_partitioning_agree() {
        let n = PARALLEL_PARTITION_MIN * 2;
        let (d, gpair) = step_data(n);
        let param = TrainParam { max_depth: 4, max_bin: 64, ..Default::default() };
        let cuts = build_cuts(&d, param.max_bin).unwrap();
        let gi = build_gradient_index(&d, &cuts).unwrap();

        let mut tree = RegTree::new(1);
        let mut grower = HistGrower::new(&param, &gi, &d);
        grower.grow(&mut Context::default(), &gpair, &mut tree);
        for nid in 0..tree.num_nodes() {
            if tree.nodes[nid].is_leaf() {
                let rows = grower.partitioner.rows(nid);
                assert!(rows.windows(2).all(|w| w[0] < w[1]), "node {nid} lost row order");
            }
        }
    }

    /// Bin indices must always address a real histogram slot, for dense and
    /// sparse indices alike.
    #[test]
    fn histogram_kernels_stay_in_bounds() {
        for sparsity in [0.0f32, 0.4] {
            let (rows, cols) = (500usize, 7usize);
            let x: Vec<f32> = (0..rows * cols)
                .map(|i| {
                    let v = ((i * 37) % 101) as f32 / 101.0;
                    if sparsity > 0.0 && (i * 7919) % 5 == 0 { f32::NAN } else { v }
                })
                .collect();
            let d = DMatrix::from_dense(&x, rows, cols, f32::NAN).unwrap();
            let cuts = build_cuts(&d, 16).unwrap();
            let gi = build_gradient_index(&d, &cuts).unwrap();
            let total = gi.total_bins();
            for r in 0..rows {
                for bin in gi.row_global_bins(r) {
                    assert!((bin as usize) < total, "row {r} produced bin {bin} of {total}");
                }
            }
        }
    }

    /// Lane counts must not depend on how many threads happen to be available.
    #[test]
    fn lane_count_depends_only_on_data() {
        assert_eq!(lanes_for(1, 1000), 1);
        assert_eq!(lanes_for(BLOCK_ROWS, 1000), 1);
        assert_eq!(lanes_for(BLOCK_ROWS * 4, 1000), 4);
        assert_eq!(lanes_for(BLOCK_ROWS * 1000, 1000), MAX_LANES);
        // A huge bin count falls back to fewer lanes to respect the budget.
        assert!(lanes_for(BLOCK_ROWS * 1000, 10_000_000) < MAX_LANES);
    }
}
