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

use super::model::RegTree;
use super::param::{
    GradStats, GrowPolicy, RT_EPS, SplitEntry, TrainParam, calc_gain, calc_split_gain, calc_weight,
};
use crate::data::gradient_index::{GHistIndex, dispatch_bins};
use crate::data::DMatrix;
use crate::objective::GradientPair;
use crate::threading;
use rayon::prelude::*;

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
/// Bins per task when reducing or subtracting histograms.
const REDUCE_CHUNK_BINS: usize = 2048;

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
}

/// One node's partitioning job.
struct SplitTask {
    nid: usize,
    left: usize,
    right: usize,
    fidx: u32,
    /// Feature-local bin threshold; rows at or below it go left.
    local_cond: i64,
    default_left: bool,
}

impl Partitioner {
    fn new(num_row: usize) -> Self {
        Self {
            row_indices: (0..num_row as u32).collect(),
            segments: vec![Some((0, num_row))],
            scratch: vec![0; num_row],
        }
    }

    fn reset(&mut self, num_row: usize) {
        self.row_indices.clear();
        self.row_indices.extend(0..num_row as u32);
        self.segments.clear();
        self.segments.push(Some((0, num_row)));
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
        let mut consumed = 0usize;
        let mut jobs: Vec<(usize, &mut [u32], &mut [u32])> = Vec::with_capacity(ranges.len());
        for &(begin, end, task_idx) in &ranges {
            let (_, s_tail) = src_rest.split_at_mut(begin - consumed);
            let (_, d_tail) = dst_rest.split_at_mut(begin - consumed);
            let (s_here, s_tail) = s_tail.split_at_mut(end - begin);
            let (d_here, d_tail) = d_tail.split_at_mut(end - begin);
            src_rest = s_tail;
            dst_rest = d_tail;
            consumed = end;
            jobs.push((task_idx, s_here, d_here));
        }

        let mut n_left = vec![0usize; tasks.len()];
        {
            let slots: Vec<(usize, &mut usize)> =
                n_left.iter_mut().enumerate().map(|(i, v)| (i, v)).collect();
            let mut by_task: Vec<Option<&mut usize>> = (0..tasks.len()).map(|_| None).collect();
            for (i, v) in slots {
                by_task[i] = Some(v);
            }
            let mut work: Vec<(&SplitTask, &mut [u32], &mut [u32], &mut usize)> =
                Vec::with_capacity(jobs.len());
            for (task_idx, src, dst) in jobs {
                let out = by_task[task_idx].take().expect("each task appears once");
                work.push((&tasks[task_idx], src, dst, out));
            }
            work.into_par_iter().for_each(|(task, src, dst, out)| {
                *out = if src.len() >= PARALLEL_PARTITION_MIN {
                    partition_blocked(src, dst, task, gi)
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
    task: &SplitTask,
    gi: &GHistIndex,
) -> usize {
    let n_blocks = src.len().div_ceil(BLOCK_ROWS);

    let mut left_counts: Vec<usize> = vec![0; n_blocks];
    left_counts.par_iter_mut().enumerate().for_each(|(b, count)| {
        let lo = b * BLOCK_ROWS;
        let hi = ((b + 1) * BLOCK_ROWS).min(src.len());
        *count = src[lo..hi].iter().filter(|&&rid| goes_left(rid, task, gi)).count();
    });
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
        for &rid in &src_ro[lo..hi] {
            if goes_left(rid, task, gi) {
                l_out[li] = rid;
                li += 1;
            } else {
                r_out[ri] = rid;
                ri += 1;
            }
        }
    });

    src.copy_from_slice(dst);
    n_left
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
/// a whole node is scanned sequentially.
#[inline]
fn goes_left(rid: u32, task: &SplitTask, gi: &GHistIndex) -> bool {
    match gi.columns.get(task.fidx, rid as usize) {
        Some(bin) => (bin as i64) <= task.local_cond,
        None => task.default_left,
    }
}

/// Per-node gradient histograms, indexed by node id.
struct HistCollection {
    data: Vec<Vec<GradStats>>,
    /// Buffers of released nodes, kept for reuse: a histogram is hundreds of
    /// kilobytes and every level would otherwise allocate a fresh set.
    free: Vec<Vec<GradStats>>,
    total_bins: usize,
}

impl HistCollection {
    fn new(total_bins: usize) -> Self {
        Self { data: Vec::new(), free: Vec::new(), total_bins }
    }

    /// Give node `nid` a buffer of the right size. Its contents are undefined;
    /// callers either fill it completely or zero it first.
    fn allocate(&mut self, nid: usize) {
        if self.data.len() <= nid {
            self.data.resize_with(nid + 1, Vec::new);
        }
        if self.data[nid].len() == self.total_bins {
            return;
        }
        self.data[nid] = match self.free.pop() {
            Some(buf) => buf,
            None => vec![GradStats::default(); self.total_bins],
        };
    }

    /// Release a histogram that will not be read again.
    fn release(&mut self, nid: usize) {
        if nid < self.data.len() && !self.data[nid].is_empty() {
            let buf = std::mem::take(&mut self.data[nid]);
            self.free.push(buf);
        }
    }

    fn clear(&mut self) {
        for i in 0..self.data.len() {
            self.release(i);
        }
        self.data.clear();
    }

    #[inline]
    fn get(&self, nid: usize) -> &[GradStats] {
        &self.data[nid]
    }
}

/// Accumulate one block of dense rows into `hist`.
///
/// Every bin index is in range by construction — a dense entry holds a
/// feature-local bin `< feature_bins(f)`, and `offsets[f] + feature_bins(f) ==
/// cut_ptrs[f + 1] <= total_bins == hist.len()`, which
/// `histogram_kernels_stay_in_bounds` pins down. Skipping the bounds check
/// anyway was measured and made no difference: the loop is bound by memory
/// latency on the scattered `hist` updates, not by the compare.
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
        let g = gpair[rid as usize];
        let (gd, hd) = (g.grad as f64, g.hess as f64);
        let base = rid as usize * stride;
        let row = &index[base..base + stride];
        for (off, bin) in offsets.iter().zip(row) {
            let cell = &mut hist[*off as usize + bin.idx()];
            cell.sum_grad += gd;
            cell.sum_hess += hd;
        }
    }
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
        let g = gpair[rid as usize];
        let (gd, hd) = (g.grad as f64, g.hess as f64);
        for bin in &index[row_ptr[rid as usize]..row_ptr[rid as usize + 1]] {
            let cell = &mut hist[bin.idx()];
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
}

impl<'a> HistGrower<'a> {
    pub fn new(param: &'a TrainParam, gi: &'a GHistIndex, dmat: &'a DMatrix) -> Self {
        Self {
            param,
            gi,
            dmat,
            hist: HistCollection::new(gi.total_bins()),
            partitioner: Partitioner::new(dmat.num_row()),
            snode: Vec::new(),
            lane_buf: Vec::new(),
        }
    }

    /// Reset for another tree over the same data, keeping allocations.
    pub fn reset(&mut self) {
        self.partitioner.reset(self.dmat.num_row());
        self.hist.clear();
        self.snode.clear();
    }

    /// Grow `tree` from `gpair`, mirroring upstream's `UpdateTree`.
    pub fn grow(&mut self, gpair: &[GradientPair], tree: &mut RegTree) {
        threading::install(|| self.grow_inner(gpair, tree));
    }

    fn grow_inner(&mut self, gpair: &[GradientPair], tree: &mut RegTree) {
        let mut num_leaves: i32 = 1;
        let root = self.init_root(gpair, tree);

        let mut queue: Vec<ExpandEntry> = Vec::new();
        if root.split.loss_chg > RT_EPS {
            queue.push(root);
        }
        let mut expand_set = self.pop(&mut queue, &mut num_leaves);

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

            let mut best_splits = Vec::with_capacity(built.len());
            let splits = self.evaluate_splits(&built);
            for (nid, split) in built.iter().zip(splits) {
                best_splits.push(ExpandEntry { nid: *nid, depth: tree.depth(*nid), split });
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
            expand_set = self.pop(&mut queue, &mut num_leaves);
        }
    }

    /// Add each row's leaf value to `preds`, using the row sets rather than
    /// re-traversing the tree.
    pub fn update_predictions(&self, tree: &RegTree, preds: &mut [f32]) {
        for nid in 0..tree.num_nodes() {
            if !tree.nodes[nid].is_leaf() {
                continue;
            }
            let value = tree.nodes[nid].value;
            for &rid in self.partitioner.rows(nid) {
                preds[rid as usize] += value;
            }
        }
    }

    /// Build the root histogram, set the root leaf, and evaluate its split.
    fn init_root(&mut self, gpair: &[GradientPair], tree: &mut RegTree) -> ExpandEntry {
        self.hist.allocate(0);
        self.build_hists(&[0], gpair);

        // Upstream sums the first feature's bins for dense data instead of the
        // gradients themselves; the two differ in rounding, so match the choice.
        let mut root_sum = GradStats::default();
        if self.dmat.is_dense() {
            let cuts = &self.gi.cuts;
            let (b, e) = (cuts.cut_ptrs[0] as usize, cuts.cut_ptrs[1] as usize);
            for bin in b..e {
                let cell = self.hist.get(0)[bin];
                root_sum.add(cell.sum_grad, cell.sum_hess);
            }
        } else {
            for g in gpair {
                root_sum.add(g.grad as f64, g.hess as f64);
            }
        }

        self.snode.clear();
        self.snode.push(NodeEntry {
            stats: root_sum,
            root_gain: calc_gain(self.param, &root_sum),
        });
        let weight = calc_weight(self.param, &root_sum);

        tree.stats[0].sum_hess = root_sum.sum_hess as f32;
        tree.stats[0].base_weight = weight;
        tree.set_leaf(0, self.param.learning_rate * weight);

        let split = self.evaluate_splits(&[0]).pop().unwrap_or_default();
        ExpandEntry { nid: 0, depth: 0, split }
    }

    /// The node batch to expand next, following upstream's `Driver::Pop`.
    fn pop(&self, queue: &mut Vec<ExpandEntry>, num_leaves: &mut i32) -> Vec<ExpandEntry> {
        if queue.is_empty() {
            return Vec::new();
        }
        let pick = |q: &Vec<ExpandEntry>| -> usize {
            match self.param.grow_policy {
                // Depth-wise: smallest node id first.
                GrowPolicy::DepthWise => {
                    let mut best = 0;
                    for i in 1..q.len() {
                        if q[i].nid < q[best].nid {
                            best = i;
                        }
                    }
                    best
                }
                // Loss-guide: largest loss change, ties to the smaller node id.
                GrowPolicy::LossGuide => {
                    let mut best = 0;
                    for i in 1..q.len() {
                        let (a, b) = (&q[i], &q[best]);
                        if a.split.loss_chg > b.split.loss_chg
                            || (a.split.loss_chg == b.split.loss_chg && a.nid < b.nid)
                        {
                            best = i;
                        }
                    }
                    best
                }
            }
        };

        if self.param.grow_policy == GrowPolicy::LossGuide {
            let e = queue.remove(pick(queue));
            return if e.is_valid(self.param, *num_leaves) {
                *num_leaves += 1;
                vec![e]
            } else {
                Vec::new()
            };
        }

        let mut result = Vec::new();
        let level = queue[pick(queue)].depth;
        // Cap the batch the way upstream's `max_node_batch_size` does.
        const MAX_BATCH: usize = 256;
        while !queue.is_empty() && result.len() < MAX_BATCH {
            let i = pick(queue);
            if queue[i].depth != level {
                break;
            }
            let e = queue.remove(i);
            if e.is_valid(self.param, *num_leaves) {
                *num_leaves += 1;
                result.push(e);
            }
        }
        result
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
        let split = &candidate.split;
        let mut parent_sum = split.left_sum;
        parent_sum.add_stats(&split.right_sum);

        let base_weight = calc_weight(self.param, &parent_sum);
        let left_weight = calc_weight(self.param, &split.left_sum);
        let right_weight = calc_weight(self.param, &split.right_sum);
        let lr = self.param.learning_rate;

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

        let node = tree.nodes[candidate.nid];
        let (left, right) = (node.left as usize, node.right as usize);
        if self.snode.len() < tree.num_nodes() {
            self.snode.resize(tree.num_nodes(), NodeEntry::default());
        }
        self.snode[left] =
            NodeEntry { stats: split.left_sum, root_gain: calc_gain(self.param, &split.left_sum) };
        self.snode[right] = NodeEntry {
            stats: split.right_sum,
            root_gain: calc_gain(self.param, &split.right_sum),
        };
    }

    fn split_task(&self, candidate: &ExpandEntry, tree: &RegTree) -> SplitTask {
        let node = tree.nodes[candidate.nid];
        let fidx = node.split_index;
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
        let mut dsts: Vec<Vec<GradStats>> = to_subtract
            .iter()
            .map(|(subtract, _, _)| std::mem::take(&mut self.hist.data[*subtract]))
            .collect();
        {
            let hist = &self.hist;
            dsts.par_iter_mut().enumerate().for_each(|(i, dst)| {
                let (_, parent, built) = to_subtract[i];
                let p = hist.get(parent);
                let b = hist.get(built);
                dst.par_chunks_mut(REDUCE_CHUNK_BINS).enumerate().for_each(|(c, out)| {
                    let lo = c * REDUCE_CHUNK_BINS;
                    for (k, o) in out.iter_mut().enumerate() {
                        o.sum_grad = p[lo + k].sum_grad - b[lo + k].sum_grad;
                        o.sum_hess = p[lo + k].sum_hess - b[lo + k].sum_hess;
                    }
                });
            });
        }
        for ((subtract, _, _), dst) in to_subtract.iter().zip(dsts) {
            self.hist.data[*subtract] = dst;
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
        let mut targets: Vec<Vec<GradStats>> =
            nodes.iter().map(|&nid| std::mem::take(&mut self.hist.data[nid])).collect();

        // Single-lane nodes: zero and accumulate in one pass over the node set.
        targets
            .par_iter_mut()
            .enumerate()
            .filter(|(i, _)| lanes[*i] == 1)
            .for_each(|(i, dst)| {
                dst.fill(GradStats::default());
                build_hist(dst, partitioner.rows(nodes[i]), gi, gpair);
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
            self.hist.data[nid] = std::mem::take(&mut targets[i]);
        }
    }

    /// Best split for each node.
    ///
    /// Candidates are evaluated over a flat `(node, feature)` task list so a
    /// level with many small nodes parallelises as well as one big node, then
    /// merged per node in feature order. The merge rule is a total order on
    /// `(loss change, smallest feature index)`, so the result does not depend
    /// on how the work was scheduled.
    fn evaluate_splits(&self, nodes: &[usize]) -> Vec<SplitEntry> {
        let n_features = self.gi.cuts.num_features();
        let n_tasks = nodes.len() * n_features;
        let per_task: Vec<SplitEntry> = (0..n_tasks)
            .into_par_iter()
            .map(|task| {
                let (node_idx, fidx) = (task / n_features, task % n_features);
                let nid = nodes[node_idx];
                let hist = self.hist.get(nid);
                let parent = self.snode[nid];
                let mut best = SplitEntry::default();
                let non_missing = self.enumerate_forward(fidx, hist, &parent, &mut best);
                // A feature has missing values in this node exactly when its
                // bins do not account for the node's whole gradient sum.
                if non_missing.sum_grad != parent.stats.sum_grad
                    || non_missing.sum_hess != parent.stats.sum_hess
                {
                    self.enumerate_backward(fidx, hist, &parent, &mut best);
                }
                best
            })
            .collect();

        per_task
            .chunks(n_features)
            .map(|candidates| {
                let mut best = SplitEntry::default();
                for candidate in candidates {
                    best.update_entry(candidate);
                }
                best
            })
            .collect()
    }

    fn enumerate_forward(
        &self,
        fidx: usize,
        hist: &[GradStats],
        parent: &NodeEntry,
        p_best: &mut SplitEntry,
    ) -> GradStats {
        let cuts = &self.gi.cuts;
        let p = self.param;
        let (ibegin, iend) = (cuts.cut_ptrs[fidx] as usize, cuts.cut_ptrs[fidx + 1] as usize);

        let mut left_sum = GradStats::default();
        let mut best = SplitEntry::default();
        for i in ibegin..iend {
            left_sum.add(hist[i].sum_grad, hist[i].sum_hess);
            let mut right_sum = GradStats::default();
            right_sum.set_subtract(&parent.stats, &left_sum);
            if is_valid_split(p, &left_sum, &right_sum) {
                let loss_chg = calc_split_gain(p, &left_sum, &right_sum) - parent.root_gain;
                best.update(loss_chg, fidx as u32, cuts.cut_values[i], false, left_sum, right_sum);
            }
        }
        p_best.update_entry(&best);
        left_sum
    }

    fn enumerate_backward(
        &self,
        fidx: usize,
        hist: &[GradStats],
        parent: &NodeEntry,
        p_best: &mut SplitEntry,
    ) {
        let cuts = &self.gi.cuts;
        let p = self.param;
        let (ibegin, iend) = (cuts.cut_ptrs[fidx] as usize, cuts.cut_ptrs[fidx + 1] as usize);

        let mut left_sum = GradStats::default();
        let mut best = SplitEntry::default();
        for i in (ibegin..iend).rev() {
            // Scanning backwards, `left_sum` accumulates the right-hand side.
            left_sum.add(hist[i].sum_grad, hist[i].sum_hess);
            let mut right_sum = GradStats::default();
            right_sum.set_subtract(&parent.stats, &left_sum);
            if is_valid_split(p, &left_sum, &right_sum) {
                let loss_chg = calc_split_gain(p, &right_sum, &left_sum) - parent.root_gain;
                let split_pt = cuts.backward_split_point(fidx, i);
                best.update(loss_chg, fidx as u32, split_pt, true, right_sum, left_sum);
            }
        }
        p_best.update_entry(&best);
    }
}

/// Both children must carry at least `min_child_weight` Hessian.
#[inline]
fn is_valid_split(p: &TrainParam, left: &GradStats, right: &GradStats) -> bool {
    left.sum_hess >= p.min_child_weight as f64 && right.sum_hess >= p.min_child_weight as f64
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
        HistGrower::new(param, &gi, d).grow(gpair, &mut tree);
        tree
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

    #[test]
    fn partitioning_is_stable_and_covers_every_row() {
        let (d, gpair) = step_data(64);
        let param = TrainParam { max_depth: 3, max_bin: 32, ..Default::default() };
        let cuts = build_cuts(&d, param.max_bin).unwrap();
        let gi = build_gradient_index(&d, &cuts).unwrap();
        let mut tree = RegTree::new(1);
        let mut grower = HistGrower::new(&param, &gi, &d);
        grower.grow(&gpair, &mut tree);

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
        grower.grow(&gpair, &mut tree);
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
