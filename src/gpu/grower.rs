//! The GPU tree driver — `grow_gpu_hist`.
//!
//! Ported from `xgboost/src/tree/updater_gpu_hist.cu` (`GPUHistMaker` /
//! `GPUHistMakerDevice::UpdateTree`), and deliberately kept step for step
//! alongside the CPU grower in [`crate::tree::hist`]: same expansion queue,
//! same validity rules, same subtraction-trick choice, same order of random
//! draws, same tree write-back. What differs is only *where* the work runs.
//!
//! # Histogram lifetime
//!
//! A node's histogram must survive from the moment it is built until its own
//! children are built, because the subtraction trick reads it. Rather than a
//! pool with eviction, each expansion batch allocates one buffer holding its
//! children's histograms side by side. That has two payoffs: a whole batch is
//! evaluated in a single launch — the split evaluator reads one buffer and
//! indexes nodes by bin offset — and a buffer frees itself once no queued node
//! still references it, because [`Handle`] is reference-counted.
//!
//! That is also why a depth-wise batch is a whole level rather than the CPU
//! grower's capped 256: every node of a batch has to share one allocation.

use std::collections::{BinaryHeap, VecDeque};

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::ellpack::{DeviceEllpack, build_ellpack};
use super::evaluate_splits::{DeviceSplitCandidate, NodeInput, SplitConfig, SplitEvaluatorGpu};
use super::histogram::{HistogramBuilder, HistogramEngine};
use super::quantiser::GradientQuantiser;
use super::row_partitioner::{RowPartitioner, SegmentSplit};
use super::{DeviceGpairs, DeviceRows, GradientPairInt64};
use crate::data::DMatrix;
use crate::objective::GradientPair;
use crate::data::cuts::HistogramCuts;
use crate::error::Result;
use crate::rng::Mt19937;
use crate::tree::column_sampler::ColumnSampler;
use crate::tree::evaluator::{InteractionConstraints, SplitEvaluator, direction};
use crate::tree::model::RegTree;
use crate::tree::param::{GradStats, GrowPolicy, RT_EPS, TrainParam};

/// A node waiting to be split; the GPU counterpart of `CPUExpandEntry`.
#[derive(Clone, Debug)]
struct ExpandEntry {
    nid: usize,
    depth: i32,
    /// This node's rows, as a range of the partitioner's `ridx`.
    seg_begin: u32,
    seg_len: u32,
    /// Buffer holding this node's histogram, its total bin count, and the bin
    /// offset of this node within it.
    hist: Handle,
    hist_bins: usize,
    slot: u32,
    /// Quantised node sum, and the gain of leaving this node a leaf.
    sum: GradientPairInt64,
    root_gain: f32,
    /// Best split found for this node, and its children's decoded sums.
    split: DeviceSplitCandidate,
    left_stats: GradStats,
    right_stats: GradStats,
}

impl ExpandEntry {
    /// `ExpandEntry::is_valid`, unchanged from the CPU grower.
    fn is_valid(&self, p: &TrainParam, num_leaves: i32) -> bool {
        if self.split.loss_chg <= RT_EPS {
            return false;
        }
        if self.left_stats.sum_hess == 0.0 || self.right_stats.sum_hess == 0.0 {
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

/// Heap wrapper whose maximum is the loss-guide candidate expanded next; the
/// smaller node id wins a tie, as upstream.
struct LossGuideEntry(ExpandEntry);

impl PartialEq for LossGuideEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}
impl Eq for LossGuideEntry {}
impl PartialOrd for LossGuideEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for LossGuideEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .split
            .loss_chg
            .total_cmp(&other.0.split.loss_chg)
            .then_with(|| other.0.nid.cmp(&self.0.nid))
    }
}

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
            Self::DepthWise(q) => q.push_back(entry),
            Self::LossGuide(q) => q.push(LossGuideEntry(entry)),
        }
    }

    /// The next batch: one entry for loss-guide, a whole level for depth-wise.
    fn pop_batch(&mut self, param: &TrainParam, num_leaves: &mut i32) -> Vec<ExpandEntry> {
        let mut out = Vec::new();
        match self {
            Self::LossGuide(q) => {
                if let Some(LossGuideEntry(entry)) = q.pop()
                    && entry.is_valid(param, *num_leaves)
                {
                    *num_leaves += 1;
                    out.push(entry);
                }
            }
            Self::DepthWise(q) => {
                let Some(level) = q.front().map(|e| e.depth) else {
                    return out;
                };
                while q.front().is_some_and(|e| e.depth == level) {
                    let entry = q.pop_front().expect("front was present");
                    if entry.is_valid(param, *num_leaves) {
                        *num_leaves += 1;
                        out.push(entry);
                    }
                }
            }
        }
        out
    }
}

/// One expandable candidate: its position in the batch, and its two children.
struct Expandable {
    batch_pos: usize,
    left: usize,
    right: usize,
}

/// Grows trees on the device.
///
/// Built once per fit — the ELLPACK and the cut layout are uploaded here, not
/// per tree — then [`grow`](Self::grow)n once per boosting round.
pub struct GpuHistGrower<R: Runtime> {
    client: ComputeClient<R>,
    ell: DeviceEllpack,
    engine: HistogramEngine<R>,
    cuts: HistogramCuts,
    param: TrainParam,
    n_rows: usize,
    n_features: usize,
    n_bins: usize,
    column_sampler: ColumnSampler,
    constraints: InteractionConstraints,
    /// Quantiser factors of the tree currently being grown.
    to_float_grad: f64,
    to_float_hess: f64,
}

impl<R: Runtime> GpuHistGrower<R> {
    /// Bin `dmat` with `cuts` and upload everything that outlives a tree.
    pub fn new(
        client: ComputeClient<R>,
        dmat: &DMatrix,
        cuts: HistogramCuts,
        param: TrainParam,
    ) -> Result<Self> {
        let matrix = build_ellpack(dmat, &cuts);
        let ell = DeviceEllpack::upload(&client, &matrix);
        let engine = HistogramBuilder::new(&client).build_shared(&matrix, &ell)?;
        let n_features = dmat.num_col();

        Ok(Self {
            client,
            ell,
            engine,
            n_rows: dmat.num_row(),
            n_features,
            n_bins: cuts.total_bins(),
            column_sampler: ColumnSampler::new(
                param.colsample_bynode,
                param.colsample_bylevel,
                param.colsample_bytree,
            ),
            constraints: InteractionConstraints::new(
                param.interaction_constraints.as_ref(),
                n_features,
            ),
            to_float_grad: 1.0,
            to_float_hess: 1.0,
            cuts,
            param,
        })
    }

    /// Grow one tree from `gpair`.
    ///
    /// Returns the final row index: rows grouped by leaf, in the partitioner's
    /// segment order. The caller pairs it with the leaf segments to update a
    /// prediction cache without re-traversing the tree.
    pub fn grow(
        &mut self,
        gpair: &[GradientPair],
        tree: &mut RegTree,
        rng: &mut Mt19937,
    ) -> Result<GrownTree> {
        // Drawn once per tree, before any split is considered, as
        // `ColumnSampler::Init` is.
        self.column_sampler.reset(self.n_features, rng);
        self.constraints.reset();
        let mut evaluator = SplitEvaluator::new(&self.param.monotone_constraints, self.n_features);

        // Fixed point, so histogram sums are exact whatever order they commit.
        // The quantiser works in the GPU module's own pair type; it is the
        // same two floats, so this is a view, not a conversion of meaning.
        let device_pairs: Vec<super::GradientPair> = gpair
            .iter()
            .map(|g| super::GradientPair { grad: g.grad, hess: g.hess })
            .collect();
        let quantiser = GradientQuantiser::new(&device_pairs, self.n_rows as u64);
        self.to_float_grad = quantiser.to_floating_point.grad;
        self.to_float_hess = quantiser.to_floating_point.hess;
        let qpairs: Vec<GradientPairInt64> =
            device_pairs.iter().map(|g| quantiser.to_fixed_point(*g)).collect();
        let dev_gpairs = self.engine.upload_gpairs(&qpairs)?;

        let split_eval = SplitEvaluatorGpu::<R>::new(
            self.client.clone(),
            &self.cuts.cut_ptrs,
            &self.cuts.cut_values,
            &self.cuts.min_values,
            SplitConfig {
                lambda: self.param.reg_lambda,
                alpha: self.param.reg_alpha,
                max_delta_step: self.param.max_delta_step,
                min_child_weight: self.param.min_child_weight,
                to_float_grad: self.to_float_grad,
                to_float_hess: self.to_float_hess,
                monotone: self.param.monotone_constraints.iter().map(direction).collect(),
            },
        )?;

        let mut partitioner = RowPartitioner::<R>::all_rows(self.client.clone(), self.n_rows);
        let mut num_leaves: i32 = 1;
        let mut leaf_segments: Vec<(usize, u32, u32)> = Vec::new();

        let root =
            self.init_root(&dev_gpairs, &qpairs, &split_eval, &evaluator, tree, rng, &partitioner)?;

        let mut queue = ExpandQueue::new(self.param.grow_policy);
        if root.split.loss_chg > RT_EPS {
            queue.push(root.clone());
        } else {
            leaf_segments.push((0, root.seg_begin, root.seg_len));
        }

        let mut batch = queue.pop_batch(&self.param, &mut num_leaves);
        while !batch.is_empty() {
            // Apply every split first, so node ids exist before anything reads
            // them — the CPU grower's order.
            let mut splits = Vec::with_capacity(batch.len());
            let mut expandable = Vec::new();
            for (pos, entry) in batch.iter().enumerate() {
                let (left, right) = self.apply_split(entry, tree, &mut evaluator);
                splits.push(self.segment_split(entry, tree));
                if self.child_can_expand(entry, num_leaves) {
                    expandable.push(Expandable { batch_pos: pos, left, right });
                }
            }

            let left_counts = partitioner.partition(&self.ell, &splits)?;

            // Children that cannot expand are leaves already; record their
            // segments now, because nothing will look at them again.
            for (pos, entry) in batch.iter().enumerate() {
                if expandable.iter().any(|e| e.batch_pos == pos) {
                    continue;
                }
                let node = tree.nodes[entry.nid];
                let n_left = left_counts[pos];
                leaf_segments.push((node.left as usize, entry.seg_begin, n_left));
                leaf_segments
                    .push((node.right as usize, entry.seg_begin + n_left, entry.seg_len - n_left));
            }

            let children = self.build_children(
                &expandable,
                &batch,
                &left_counts,
                &dev_gpairs,
                &partitioner,
                &evaluator,
            );
            let evaluated = self.evaluate(&children, &split_eval, &evaluator, rng)?;

            for child in evaluated {
                if child.split.loss_chg > RT_EPS {
                    queue.push(child);
                } else {
                    leaf_segments.push((child.nid, child.seg_begin, child.seg_len));
                }
            }

            batch = queue.pop_batch(&self.param, &mut num_leaves);
        }

        // Anything still queued when the loop ended failed `is_valid` and stays
        // a leaf.
        if let ExpandQueue::LossGuide(q) = &queue {
            for LossGuideEntry(e) in q.iter() {
                leaf_segments.push((e.nid, e.seg_begin, e.seg_len));
            }
        }
        if let ExpandQueue::DepthWise(q) = &queue {
            for e in q.iter() {
                leaf_segments.push((e.nid, e.seg_begin, e.seg_len));
            }
        }

        Ok(GrownTree { ridx: partitioner.read(), leaf_segments })
    }

    /// Build the root histogram, set the root leaf, and evaluate its split.
    #[allow(clippy::too_many_arguments)]
    fn init_root(
        &mut self,
        dev_gpairs: &DeviceGpairs,
        qpairs: &[GradientPairInt64],
        split_eval: &SplitEvaluatorGpu<R>,
        evaluator: &SplitEvaluator,
        tree: &mut RegTree,
        rng: &mut Mt19937,
        partitioner: &RowPartitioner<R>,
    ) -> Result<ExpandEntry> {
        let hist = self.zeroed_frontier(1);
        let rows = DeviceRows::slice(partitioner.ridx().clone(), 0, self.n_rows);
        self.engine.build_into(dev_gpairs, &rows, &hist, self.n_bins, 0);

        // Exact in fixed point, so — unlike the CPU grower — there is no need
        // to choose between summing the first feature's bins and summing the
        // gradients. Both give this.
        let sum = qpairs.iter().fold(GradientPairInt64::default(), |a, b| a + *b);
        let stats = self.decode(sum);
        let root_gain = evaluator.calc_gain(0, &self.param, &stats);
        let weight = evaluator.calc_weight(0, &self.param, &stats);

        tree.stats[0].sum_hess = stats.sum_hess as f32;
        tree.stats[0].base_weight = weight;
        tree.set_leaf(0, self.param.learning_rate * weight);

        let entry = ExpandEntry {
            nid: 0,
            depth: 0,
            seg_begin: 0,
            seg_len: self.n_rows as u32,
            hist,
            hist_bins: self.n_bins,
            slot: 0,
            sum,
            root_gain,
            split: DeviceSplitCandidate::default(),
            left_stats: GradStats::default(),
            right_stats: GradStats::default(),
        };
        let evaluated = self.evaluate(&[entry], split_eval, evaluator, rng)?;
        Ok(evaluated.into_iter().next().expect("one node in, one node out"))
    }

    /// Evaluate a batch of nodes that share one histogram buffer.
    fn evaluate(
        &mut self,
        nodes: &[ExpandEntry],
        split_eval: &SplitEvaluatorGpu<R>,
        evaluator: &SplitEvaluator,
        rng: &mut Mt19937,
    ) -> Result<Vec<ExpandEntry>> {
        if nodes.is_empty() {
            return Ok(Vec::new());
        }
        let hist = nodes[0].hist.clone();
        let hist_bins = nodes[0].hist_bins;
        debug_assert!(
            nodes.iter().all(|n| n.hist_bins == hist_bins),
            "a batch must share one histogram allocation"
        );

        // Column samples are drawn here, one per node, because with
        // `colsample_bynode < 1` each draw advances the engine and the order of
        // those draws is part of what the model is.
        let mut mask = vec![0u32; nodes.len() * self.n_features];
        for (i, node) in nodes.iter().enumerate() {
            let features = self.column_sampler.feature_set(node.depth, rng);
            for &f in features.iter() {
                if self.constraints.query(node.nid, f) {
                    mask[i * self.n_features + f as usize] = 1;
                }
            }
        }

        let inputs: Vec<NodeInput> = nodes
            .iter()
            .map(|n| {
                let (lower, upper) = evaluator.bounds(n.nid);
                NodeInput {
                    hist_base: n.slot,
                    parent_grad: n.sum.grad,
                    parent_hess: n.sum.hess,
                    root_gain: n.root_gain,
                    lower,
                    upper,
                }
            })
            .collect();

        let candidates = split_eval.evaluate(&hist, hist_bins, &inputs, &mask)?;

        Ok(nodes
            .iter()
            .zip(candidates)
            .map(|(node, split)| {
                let left = GradientPairInt64 { grad: split.left_grad, hess: split.left_hess };
                let right = node.sum - left;
                ExpandEntry {
                    split,
                    left_stats: self.decode(left),
                    right_stats: self.decode(right),
                    ..node.clone()
                }
            })
            .collect())
    }

    /// Build one histogram per expandable candidate's smaller child, subtract
    /// for the sibling, into a single allocation for the batch.
    fn build_children(
        &mut self,
        expandable: &[Expandable],
        batch: &[ExpandEntry],
        left_counts: &[u32],
        dev_gpairs: &DeviceGpairs,
        partitioner: &RowPartitioner<R>,
        evaluator: &SplitEvaluator,
    ) -> Vec<ExpandEntry> {
        if expandable.is_empty() {
            return Vec::new();
        }
        let frontier = self.zeroed_frontier(expandable.len() * 2);
        let frontier_bins = self.n_bins * expandable.len() * 2;
        let n_bins = self.n_bins as u32;
        let mut out = Vec::with_capacity(expandable.len() * 2);

        for (i, e) in expandable.iter().enumerate() {
            let parent = &batch[e.batch_pos];
            let n_left = left_counts[e.batch_pos];
            let left_seg = (parent.seg_begin, n_left);
            let right_seg = (parent.seg_begin + n_left, parent.seg_len - n_left);
            let left_sum =
                GradientPairInt64 { grad: parent.split.left_grad, hess: parent.split.left_hess };
            let right_sum = parent.sum - left_sum;

            // Build the child with the smaller hessian sum; subtract the other.
            let fewer_right = parent.right_stats.sum_hess < parent.left_stats.sum_hess;
            let (build_slot, sub_slot) = ((i * 2) as u32, (i * 2 + 1) as u32);
            let (build, subtract) = if fewer_right {
                (
                    (e.right, right_seg, right_sum, parent.right_stats, build_slot),
                    (e.left, left_seg, left_sum, parent.left_stats, sub_slot),
                )
            } else {
                (
                    (e.left, left_seg, left_sum, parent.left_stats, build_slot),
                    (e.right, right_seg, right_sum, parent.right_stats, sub_slot),
                )
            };

            let rows = DeviceRows::slice(
                partitioner.ridx().clone(),
                build.1.0 as usize,
                build.1.1 as usize,
            );
            self.engine.build_into(
                dev_gpairs,
                &rows,
                &frontier,
                frontier_bins,
                build_slot * n_bins,
            );
            self.engine.subtract_into(
                &parent.hist,
                parent.hist_bins,
                parent.slot,
                &frontier,
                frontier_bins,
                build_slot * n_bins,
                sub_slot * n_bins,
            );

            for (nid, seg, sum, stats, slot) in [build, subtract] {
                out.push(ExpandEntry {
                    nid,
                    depth: parent.depth + 1,
                    seg_begin: seg.0,
                    seg_len: seg.1,
                    hist: frontier.clone(),
                    hist_bins: frontier_bins,
                    slot: slot * n_bins,
                    sum,
                    // Upstream evaluates a child's gain against the *parent's*
                    // node id, so the parent's weight box applies here.
                    root_gain: evaluator.calc_gain(parent.nid, &self.param, &stats),
                    split: DeviceSplitCandidate::default(),
                    left_stats: GradStats::default(),
                    right_stats: GradStats::default(),
                });
            }
        }

        // The depth-wise queue expects ascending node ids, and the smaller
        // child is not always the left one.
        out.sort_by_key(|e| e.nid);
        out
    }

    /// Write one split into the tree and hand both children their weight boxes.
    fn apply_split(
        &mut self,
        entry: &ExpandEntry,
        tree: &mut RegTree,
        evaluator: &mut SplitEvaluator,
    ) -> (usize, usize) {
        let mut parent_sum = entry.left_stats;
        parent_sum.add_stats(&entry.right_stats);
        let nid = entry.nid;
        let p = &self.param;

        // All three weights are bounded by the *parent's* box; the children do
        // not have one until `add_split` derives it below.
        let base_weight = evaluator.calc_weight(nid, p, &parent_sum);
        let left_weight = evaluator.calc_weight(nid, p, &entry.left_stats);
        let right_weight = evaluator.calc_weight(nid, p, &entry.right_stats);
        let lr = p.learning_rate;

        tree.expand_node(
            nid,
            entry.split.split_index(),
            entry.split.split_value,
            entry.split.default_left(),
            base_weight,
            left_weight * lr,
            right_weight * lr,
            entry.split.loss_chg,
            parent_sum.sum_hess as f32,
            entry.left_stats.sum_hess as f32,
            entry.right_stats.sum_hess as f32,
        );

        let node = tree.nodes[nid];
        let (left, right) = (node.left as usize, node.right as usize);
        evaluator.add_split(nid, left, right, node.split_index, left_weight, right_weight);
        self.constraints.split(nid, node.split_index, left, right);
        (left, right)
    }

    /// `HistGrower::is_child_valid`.
    fn child_can_expand(&self, parent: &ExpandEntry, num_leaves: i32) -> bool {
        if self.param.max_depth > 0 && parent.depth + 1 >= self.param.max_depth {
            return false;
        }
        if self.param.max_leaves > 0 && num_leaves >= self.param.max_leaves {
            return false;
        }
        true
    }

    /// The partitioner's view of one split.
    ///
    /// `cond` is derived from the chosen threshold exactly as the CPU grower's
    /// `find_split_condition` does, so the two partition identically.
    fn segment_split(&self, entry: &ExpandEntry, tree: &RegTree) -> SegmentSplit {
        let node = tree.nodes[entry.nid];
        let fidx = node.split_index;
        let global_cond = self.find_split_condition(fidx, node.value);
        // `-1` (no matching cut) stays negative, so every present row goes
        // right and only the missing ones follow `default_left`.
        let cond = if global_cond < 0 {
            -1
        } else {
            global_cond - self.cuts.cut_ptrs[fidx as usize] as i64
        };
        SegmentSplit {
            begin: entry.seg_begin,
            len: entry.seg_len,
            fidx,
            cond,
            default_left: node.default_left,
            cat_bits: Vec::new(),
        }
    }

    /// Bin whose cut value equals the split threshold, or `-1`.
    fn find_split_condition(&self, fidx: u32, split_pt: f32) -> i64 {
        let (lo, hi) = (self.cuts.cut_ptrs[fidx as usize], self.cuts.cut_ptrs[fidx as usize + 1]);
        for bound in lo..hi {
            if split_pt == self.cuts.cut_values[bound as usize] {
                return bound as i64;
            }
        }
        -1
    }

    /// Fixed point back to the `f64` sums the split arithmetic runs on.
    fn decode(&self, g: GradientPairInt64) -> GradStats {
        GradStats::new(g.grad as f64 * self.to_float_grad, g.hess as f64 * self.to_float_hess)
    }

    /// A zeroed buffer holding `slots` node histograms side by side.
    fn zeroed_frontier(&self, slots: usize) -> Handle {
        // Four u32 accumulator words per bin, which is also two i64 words.
        self.client.create_from_slice(bytemuck::cast_slice(&vec![0u32; self.n_bins * 4 * slots]))
    }
}

/// What a grown tree leaves behind: which rows ended in which leaf.
///
/// The GPU analogue of the CPU grower's row sets, and what
/// `update_predictions` needs to add leaf values without re-traversing.
#[derive(Clone, Debug)]
pub struct GrownTree {
    /// Rows grouped by leaf, in segment order.
    pub ridx: Vec<u32>,
    /// `(leaf node id, segment begin, segment length)`.
    pub leaf_segments: Vec<(usize, u32, u32)>,
}

impl GrownTree {
    /// Add each row's leaf value to `preds`, mirroring
    /// `HistGrower::update_predictions`.
    pub fn update_predictions(
        &self,
        tree: &RegTree,
        preds: &mut [f32],
        n_groups: usize,
        group: usize,
        weight: f32,
    ) {
        for &(nid, begin, len) in &self.leaf_segments {
            let values = tree.leaf_value(nid);
            for &rid in &self.ridx[begin as usize..(begin + len) as usize] {
                let base = rid as usize * n_groups;
                for (t, v) in values.iter().enumerate() {
                    preds[base + group + t] += v * weight;
                }
            }
        }
    }
}
