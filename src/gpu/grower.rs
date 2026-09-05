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
use rayon::prelude::*;

use super::categorical;
use super::ellpack::{DeviceEllpack, build_ellpack};
use super::evaluate_splits::{
    DeviceSplitCandidate, MultiNodeInput, NodeInput, SplitConfig, SplitEvaluatorGpu,
};
use super::histogram::{HistogramBuilder, HistogramEngine, NodeHistJob};
use super::quantiser::{
    DeviceGpairsF32, GradientQuantiser, quantise_device_column, sum_device_column,
};
use super::row_partitioner::{RowPartitioner, SegmentSplit};
use super::{DeviceGpairs, DeviceRows, GradientPairInt64, PhaseLog};
use crate::data::{DMatrix, MetaInfo as DMatrixInfo};
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
    /// Categories the split sends right, as a bit set over feature-local
    /// bins. Empty until `split.is_cat` — set only for a categorical winner.
    cat_bits: Vec<u32>,
    /// This node's sum and its children's sums, one entry per target. Empty
    /// outside a vector-leaf tree, where [`Self::sum`] and the two `_stats`
    /// already say everything: they are the same numbers added over targets.
    target_sums: Vec<GradientPairInt64>,
    left_target_sums: Vec<GradientPairInt64>,
    right_target_sums: Vec<GradientPairInt64>,
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
    ///
    /// An entry that fails `is_valid` here — its gain under `min_split_loss`,
    /// the leaf budget spent — stays the leaf it already is, and `leaves`
    /// records its rows: the prediction update adds every leaf's value by
    /// these segments, where the CPU grower walks the tree's leaves, so a
    /// leaf left off the list is a leaf whose value never reaches the
    /// predictions.
    fn pop_batch(
        &mut self,
        param: &TrainParam,
        num_leaves: &mut i32,
        leaves: &mut Vec<(usize, u32, u32)>,
    ) -> Vec<ExpandEntry> {
        let mut out = Vec::new();
        let mut take = |entry: ExpandEntry, out: &mut Vec<ExpandEntry>| {
            if entry.is_valid(param, *num_leaves) {
                *num_leaves += 1;
                out.push(entry);
            } else {
                leaves.push((entry.nid, entry.seg_begin, entry.seg_len));
            }
        };
        match self {
            Self::LossGuide(q) => {
                if let Some(LossGuideEntry(entry)) = q.pop() {
                    take(entry, &mut out);
                }
            }
            Self::DepthWise(q) => {
                let Some(level) = q.front().map(|e| e.depth) else {
                    return out;
                };
                while q.front().is_some_and(|e| e.depth == level) {
                    let entry = q.pop_front().expect("front was present");
                    take(entry, &mut out);
                }
            }
        }
        out
    }
}

/// One expandable candidate: its position in the batch, and its two children.
/// A zeroed frontier buffer that batches take their slots from in turn.
///
/// A loss-guided tree builds two histograms a batch, sixty-three batches a
/// tree; zeroing a two-slot buffer for each was a launch per batch. Slots
/// are handed out from a buffer of [`ARENA_SLOTS`] zeroed once, and the
/// parent and children of a batch may share it — `subtract_batch` reads the
/// parent's slot and writes the sibling's, which never overlap.
struct Arena {
    handle: Handle,
    slots: usize,
    next: usize,
}

/// Slots zeroed at a time. Every level of a depth-6 tree fits in one.
const ARENA_SLOTS: usize = 64;

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
    /// Built once: the cut layout it uploads is fixed for the fit.
    split_eval: SplitEvaluatorGpu<R>,
    /// Outputs a leaf carries. `1` is the ordinary one-tree-per-target fit;
    /// more makes this a vector-leaf grower, the device counterpart of
    /// `HistGrower::new_multi`.
    n_targets: usize,
    /// Quantiser factors of the tree currently being grown.
    to_float_grad: f64,
    to_float_hess: f64,
    /// Frontier buffers, with their capacity in `u32` words, kept for reuse
    /// across levels and rounds. A buffer is free when this is its only
    /// holder (`Handle::can_mut`): the expand entries that reference a level's
    /// histograms drop theirs once the level's children are built. See
    /// `HistogramEngine::prepare_frontier` for why reuse matters.
    frontiers: Vec<(usize, Handle)>,
    /// The frontier slots are handed out from; see [`Self::take_slots`].
    arena: Option<Arena>,
    /// The device-resident round — predictions, labels, weights — once a
    /// round has been done that way; see `gpu::objective`.
    round: Option<super::objective::DeviceRound>,
}

impl<R: Runtime> GpuHistGrower<R> {
    /// Bin `dmat` with `cuts` and upload everything that outlives a tree.
    pub fn new(
        client: ComputeClient<R>,
        dmat: &DMatrix,
        cuts: HistogramCuts,
        param: TrainParam,
    ) -> Result<Self> {
        Self::new_multi(client, dmat, None, cuts, param, 1)
    }

    /// A grower for vector-leaf trees: one tree covering `n_targets` outputs.
    ///
    /// Categorical features are refused rather than mis-fitted. The vector-leaf
    /// evaluator scans a feature's bins in ascending order — meaningless for
    /// category codes — and the host-side route [`super::categorical`] provides
    /// for the scalar path scores one target at a time, which is not the
    /// decision a vector leaf makes. `multi_strategy=multi_output_tree` has no
    /// categorical split on the CPU either, and [`crate::api::train`] rejects
    /// the combination before it ever reaches here; this is the backstop.
    ///
    /// `values` is the matrix already on the device when the sketch put it
    /// there (`gbm` hands it over): a second upload of the same 100 MB was
    /// 25 ms of a 260 ms fit on a T4, 53 ms with a CSR matrix.
    pub fn new_multi(
        client: ComputeClient<R>,
        dmat: &DMatrix,
        values: Option<super::sketch::DeviceValues>,
        cuts: HistogramCuts,
        param: TrainParam,
        n_targets: usize,
    ) -> Result<Self> {
        let n_targets = n_targets.max(1);
        if n_targets > 1 && cuts.has_categorical() {
            return Err(crate::error::Error::invalid(
                "multi_strategy",
                "`multi_output_tree` has no categorical split; one-hot encode the \
                 categories, or use `one_output_per_tree`",
            ));
        }
        let mut phases = PhaseLog::new(client.clone());
        // Binned on the device where there is one: the host route is three
        // passes over every entry and two uploads, and it cost more than a
        // whole upstream fit at the sizes `docs/gpu-benchmarks.md` measures.
        // A categorical feature bins by category code rather than by cut
        // search, which only the host route does; the CPU runtime's cores
        // are the host's, so it keeps the host route too.
        let on_device = super::launch::has_planes(&client)
            && !cuts.has_categorical()
            && super::sketch::DeviceValues::readable(dmat);
        let (ell, engine) = if on_device {
            let values = match values {
                Some(v) if v.n_rows == dmat.num_row() && v.n_cols == dmat.num_col() => v,
                _ => super::sketch::DeviceValues::upload(&client, dmat),
            };
            phases.mark("upload values");
            let (ell, shape) = DeviceEllpack::bin_on_device(&client, &values, &cuts);
            phases.mark("bin+pack on device");
            let engine = HistogramBuilder::new(&client).build_device(&ell, &shape)?;
            (ell, engine)
        } else {
            let matrix = build_ellpack(dmat, &cuts);
            phases.mark("bin");
            let ell = DeviceEllpack::upload(&client, &matrix);
            phases.mark("pack+upload");
            let engine = HistogramBuilder::new(&client).build_shared(&matrix, &ell)?;
            (ell, engine)
        };
        phases.mark("engine");
        phases.report_as("setup");
        let n_features = dmat.num_col();

        let split_eval = SplitEvaluatorGpu::<R>::new(
            client.clone(),
            &cuts.cut_ptrs,
            &cuts.cut_values,
            &cuts.min_values,
            SplitConfig {
                lambda: param.reg_lambda,
                alpha: param.reg_alpha,
                max_delta_step: param.max_delta_step,
                min_child_weight: param.min_child_weight,
                monotone: param.monotone_constraints.iter().map(direction).collect(),
            },
        )?;

        Ok(Self {
            split_eval,
            client,
            ell,
            engine,
            n_rows: dmat.num_row(),
            n_features,
            n_bins: cuts.total_bins(),
            column_sampler: ColumnSampler::weighted(
                param.colsample_bynode,
                param.colsample_bylevel,
                param.colsample_bytree,
                &dmat.info().feature_weights,
            ),
            constraints: InteractionConstraints::new(
                param.interaction_constraints.as_ref(),
                n_features,
            ),
            n_targets,
            to_float_grad: 1.0,
            to_float_hess: 1.0,
            frontiers: Vec::new(),
            arena: None,
            round: None,
            cuts,
            param,
        })
    }

    /// Whether this grower produces vector leaves.
    #[inline]
    fn is_multi(&self) -> bool {
        self.n_targets > 1
    }

    /// Bins in a frontier of `slots` nodes: a node holds one histogram per
    /// target, laid out side by side from its own slot.
    #[inline]
    fn frontier_bins(&self, slots: usize) -> usize {
        self.n_bins * self.n_targets * slots
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
        let k = self.n_targets;
        let columns: Vec<DeviceGpairsF32> = if k == 1 {
            vec![DeviceGpairsF32::upload_objective(&self.client, gpair)]
        } else {
            (0..k)
                .map(|t| {
                    let column: Vec<GradientPair> =
                        (0..self.n_rows).into_par_iter().map(|r| gpair[r * k + t]).collect();
                    DeviceGpairsF32::upload_objective(&self.client, &column)
                })
                .collect()
        };
        let (partitioner, leaf_segments) = self.grow_columns(columns, tree, rng)?;
        let ridx = partitioner.read();
        Ok(GrownTree { ridx, leaf_segments })
    }

    /// A whole round on the device: the gradients of `kind` at the
    /// device-resident predictions, one tree, and the tree's leaf values
    /// added back into those predictions — nothing crosses to the host but
    /// the split decisions.
    ///
    /// The predictions must be on the device already ([`Self::device_round`]
    /// makes them so from the host's copy); afterwards the host's copy is
    /// stale until [`Self::sync_preds`].
    pub fn grow_device(
        &mut self,
        kind: crate::objective::DeviceObjective,
        tree: &mut RegTree,
        rng: &mut Mt19937,
        weight: f32,
    ) -> Result<()> {
        let round = self.round.as_ref().expect("device predictions are set up first");
        let gpairs = round.gradients(&self.client, kind);
        let (partitioner, leaf_segments) = self.grow_columns(vec![gpairs], tree, rng)?;
        let segments: Vec<(u32, u32, u32, f32)> = leaf_segments
            .iter()
            .map(|&(nid, begin, len)| {
                (partitioner.side_of(begin), begin, len, tree.leaf_value(nid)[0])
            })
            .collect();
        let round = self.round.as_mut().expect("set above");
        round.update_predictions(&self.client, partitioner.buffers(), &segments, weight);
        Ok(())
    }

    /// Put the host's predictions on the device, for [`Self::grow_device`].
    /// Uploads only when the device does not already hold the current ones.
    pub fn device_round(&mut self, info: &DMatrixInfo, preds: &[f32]) {
        match &mut self.round {
            Some(round) if !round.host_stale => round.upload_preds(&self.client, preds),
            Some(_) => {}
            None => self.round = Some(super::objective::DeviceRound::new(&self.client, info, preds)),
        }
    }

    /// Whether the device holds predictions the host copy is behind.
    pub fn preds_on_device(&self) -> bool {
        self.round.as_ref().is_some_and(|r| r.host_stale)
    }

    /// Bring the host's predictions up to date with the device's.
    pub fn sync_preds(&mut self, preds: &mut [f32]) {
        if let Some(round) = &mut self.round
            && round.host_stale
        {
            round.download(&self.client, preds);
        }
    }

    /// The training `rmse` over the device's predictions, when they are the
    /// current ones.
    pub fn device_rmse(&self) -> Option<f64> {
        let round = self.round.as_ref()?;
        round.host_stale.then(|| round.rmse(&self.client))
    }

    /// Grow one tree from device-resident gradient columns, returning the
    /// partitioner (its row index says which leaf every row reached) and the
    /// leaf segments.
    fn grow_columns(
        &mut self,
        columns: Vec<DeviceGpairsF32>,
        tree: &mut RegTree,
        rng: &mut Mt19937,
    ) -> Result<(RowPartitioner<R>, Vec<(usize, u32, u32)>)> {
        // Drawn once per tree, before any split is considered, as
        // `ColumnSampler::Init` is.
        let mut phases = PhaseLog::new(self.client.clone());
        // The last tree's arena goes back to the pool: nothing holds its
        // slots once the tree's entries are gone.
        self.arena = None;
        self.column_sampler.reset(self.n_features, rng);
        self.constraints.reset();
        let mut evaluator = SplitEvaluator::new(&self.param.monotone_constraints, self.n_features);

        // One gradient column per target, which is what makes the vector-leaf
        // path reuse the scalar histogram kernel unchanged: it takes a gradient
        // per row, so a round hands it one column at a time rather than
        // teaching it a stride. `HistGrower::split_gpair` does the same.
        //
        // Fixed point, so histogram sums are exact whatever order they commit.
        // The quantiser works in the GPU module's own pair type; it is the
        // same two floats, so this is a view, not a conversion of meaning.
        //
        // All three passes — the bound, the conversion, the sum — run on the
        // device, and the upload is the `f32` pairs: the host quantiser's
        // passes plus its `i64` upload were 9 ms of a 30 ms round on a cloud
        // VM at 500 000 rows, 44 ms at a million.
        let quantiser = GradientQuantiser::new_on_device(&self.client, &columns, self.n_rows as u64);
        self.to_float_grad = quantiser.to_floating_point.grad;
        self.to_float_hess = quantiser.to_floating_point.hess;
        let dev_gpairs: Vec<DeviceGpairs> = columns
            .iter()
            .map(|c| quantise_device_column(&self.client, c, &quantiser))
            .collect();
        // Exact in fixed point, so — unlike the CPU grower — there is no need
        // to choose between summing the first feature's bins and summing the
        // gradients. Both give this.
        let target_sums: Vec<GradientPairInt64> =
            dev_gpairs.iter().map(|d| sum_device_column(&self.client, d)).collect();
        phases.mark("quantise+upload");

        let mut partitioner = RowPartitioner::<R>::all_rows(self.client.clone(), self.n_rows);
        phases.mark("ridx");
        let mut num_leaves: i32 = 1;
        let mut leaf_segments: Vec<(usize, u32, u32)> = Vec::new();

        let root =
            self.init_root(&dev_gpairs, &target_sums, &evaluator, tree, rng, &partitioner)?;
        phases.mark("root");

        let mut queue = ExpandQueue::new(self.param.grow_policy);
        if root.split.loss_chg > RT_EPS {
            queue.push(root.clone());
        } else {
            leaf_segments.push((0, root.seg_begin, root.seg_len));
        }

        let mut batch = queue.pop_batch(&self.param, &mut num_leaves, &mut leaf_segments);
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

            phases.mark("host");
            let left_counts = partitioner.partition(&self.ell, &splits)?;
            phases.mark("partition");

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
            phases.mark("histogram");
            let evaluated = self.evaluate(&children, &evaluator, rng)?;
            phases.mark("evaluate");

            for child in evaluated {
                if child.split.loss_chg > RT_EPS {
                    queue.push(child);
                } else {
                    leaf_segments.push((child.nid, child.seg_begin, child.seg_len));
                }
            }

            batch = queue.pop_batch(&self.param, &mut num_leaves, &mut leaf_segments);
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

        phases.report();
        Ok((partitioner, leaf_segments))
    }

    /// Build the root histogram, set the root leaf, and evaluate its split.
    #[allow(clippy::too_many_arguments)]
    fn init_root(
        &mut self,
        dev_gpairs: &[DeviceGpairs],
        target_sums: &[GradientPairInt64],
        evaluator: &SplitEvaluator,
        tree: &mut RegTree,
        rng: &mut Mt19937,
        partitioner: &RowPartitioner<R>,
    ) -> Result<ExpandEntry> {
        let (hist, base, slots) = self.take_slots(1);
        let hist_bins = self.frontier_bins(slots);
        let slot = (base * self.n_bins * self.n_targets) as u32;
        let rows = DeviceRows::slice(partitioner.ridx().clone(), 0, self.n_rows);
        for (t, gpairs) in dev_gpairs.iter().enumerate() {
            self.engine.build_into(gpairs, &rows, &hist, hist_bins, slot + (t * self.n_bins) as u32);
        }

        let sum = target_sums.iter().fold(GradientPairInt64::default(), |a, b| a + *b);
        let stats = self.decode(sum);
        let weight = evaluator.calc_weight(0, &self.param, &stats);

        // The scalar bookkeeping — node cover and the tree's own statistics —
        // is stated over the targets together, as `sum_targets` states it.
        tree.stats[0].sum_hess = stats.sum_hess as f32;
        tree.stats[0].base_weight = weight;
        let root_gain = if self.is_multi() {
            let weights: Vec<f32> = target_sums
                .iter()
                .map(|s| {
                    self.param.learning_rate * evaluator.calc_weight(0, &self.param, &self.decode(*s))
                })
                .collect();
            tree.set_leaf_vector(0, &weights);
            // `HistGrower::multi_gain`: every target's own regularised gain.
            target_sums.iter().map(|s| evaluator.calc_gain(0, &self.param, &self.decode(*s))).sum()
        } else {
            tree.set_leaf(0, self.param.learning_rate * weight);
            evaluator.calc_gain(0, &self.param, &stats)
        };

        let entry = ExpandEntry {
            nid: 0,
            depth: 0,
            seg_begin: 0,
            seg_len: self.n_rows as u32,
            hist,
            hist_bins,
            slot,
            sum,
            root_gain,
            split: DeviceSplitCandidate::default(),
            left_stats: GradStats::default(),
            right_stats: GradStats::default(),
            cat_bits: Vec::new(),
            target_sums: if self.is_multi() { target_sums.to_vec() } else { Vec::new() },
            left_target_sums: Vec::new(),
            right_target_sums: Vec::new(),
        };
        let evaluated = self.evaluate(&[entry], evaluator, rng)?;
        Ok(evaluated.into_iter().next().expect("one node in, one node out"))
    }

    /// Evaluate a batch of nodes that share one histogram buffer.
    fn evaluate(
        &mut self,
        nodes: &[ExpandEntry],
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
        //
        // A categorical feature is left out of `mask` — the device kernel
        // scans a feature's bins in ascending order, which is meaningless for
        // a category code — and collected in `cat_features` instead, so it can
        // be scored host-side below.
        let mut mask = vec![0u32; nodes.len() * self.n_features];
        let mut cat_features: Vec<Vec<u32>> = vec![Vec::new(); nodes.len()];
        for (i, node) in nodes.iter().enumerate() {
            let features = self.column_sampler.feature_set(node.depth, rng);
            for &f in features.iter() {
                if self.constraints.query(node.nid, f) {
                    if self.cuts.is_cat(f as usize) {
                        cat_features[i].push(f);
                    } else {
                        mask[i * self.n_features + f as usize] = 1;
                    }
                }
            }
        }

        if self.is_multi() {
            // A vector-leaf fit has no categorical feature — `new_multi`
            // refuses one — so `mask` already names every candidate.
            debug_assert!(cat_features.iter().all(Vec::is_empty));
            return self.evaluate_multi(nodes, &hist, hist_bins, &mask, evaluator);
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

        let mut candidates = self.split_eval.evaluate(
            &hist,
            hist_bins,
            &inputs,
            &mask,
            self.to_float_grad,
            self.to_float_hess,
        )?;

        let mut cat_bits = vec![Vec::new(); nodes.len()];
        if cat_features.iter().any(|f| !f.is_empty()) {
            // The batch's slots only: the buffer is an arena the batch
            // shares with others (`take_slots`).
            let lo = nodes.iter().map(|n| n.slot as usize).min().unwrap_or(0);
            let hi = nodes.iter().map(|n| n.slot as usize + self.n_bins).max().unwrap_or(0);
            let all_bins = self.read_histogram(&hist, hist_bins, lo, hi - lo)?;
            for (i, node) in nodes.iter().enumerate() {
                if cat_features[i].is_empty() {
                    continue;
                }
                let at = node.slot as usize - lo;
                let node_bins = &all_bins[at..at + self.n_bins];
                let mut best = candidates[i];
                for &f in &cat_features[i] {
                    let (ib, ie) = (
                        self.cuts.cut_ptrs[f as usize] as usize,
                        self.cuts.cut_ptrs[f as usize + 1] as usize,
                    );
                    let (cand, bits) = categorical::enumerate(
                        evaluator,
                        &self.param,
                        node.nid,
                        f,
                        &self.cuts.cut_values[ib..ie],
                        &node_bins[ib..ie],
                        node.sum,
                        node.root_gain,
                        self.to_float_grad,
                        self.to_float_hess,
                    );
                    if categorical::replaces(cand.loss_chg, f, best.loss_chg, best.split_index()) {
                        best = DeviceSplitCandidate {
                            loss_chg: cand.loss_chg,
                            sindex: if cand.default_left { f | (1 << 31) } else { f },
                            split_value: cand.split_value,
                            left_grad: cand.left.grad,
                            left_hess: cand.left.hess,
                            is_cat: true,
                        };
                        cat_bits[i] = bits;
                    }
                }
                candidates[i] = best;
            }
        }

        Ok(nodes
            .iter()
            .zip(candidates)
            .zip(cat_bits)
            .map(|((node, split), bits)| {
                let left = GradientPairInt64 { grad: split.left_grad, hess: split.left_hess };
                let right = node.sum - left;
                ExpandEntry {
                    split,
                    left_stats: self.decode(left),
                    right_stats: self.decode(right),
                    cat_bits: bits,
                    ..node.clone()
                }
            })
            .collect())
    }

    /// [`evaluate`](Self::evaluate) for a vector-leaf tree.
    ///
    /// The split search itself is one device call, as it is for a scalar tree.
    /// What it cannot return is the winner's per-target child sums — a
    /// candidate carries one `(grad, hess)` pair, summed over targets, and is
    /// copied by the thousand — so they are read out of the same prefix sums
    /// afterwards, which is exactly why `HistGrower::multi_child_sums` exists.
    fn evaluate_multi(
        &mut self,
        nodes: &[ExpandEntry],
        hist: &Handle,
        hist_bins: usize,
        mask: &[u32],
        evaluator: &SplitEvaluator,
    ) -> Result<Vec<ExpandEntry>> {
        let inputs: Vec<MultiNodeInput> = nodes
            .iter()
            .map(|n| {
                let (lower, upper) = evaluator.bounds(n.nid);
                MultiNodeInput {
                    hist_base: n.slot,
                    root_gain: n.root_gain,
                    lower,
                    upper,
                    parent: n.target_sums.clone(),
                }
            })
            .collect();

        let (candidates, scan) = self.split_eval.evaluate_multi(
            hist,
            hist_bins,
            self.n_bins,
            self.n_targets,
            &inputs,
            mask,
            self.to_float_grad,
            self.to_float_hess,
        )?;

        // The chosen threshold back to a bin, by the same rule the row
        // partitioner uses, so the sums and the partition name the same split.
        let splits: Vec<(u32, i64, bool)> = candidates
            .iter()
            .map(|c| {
                let fidx = c.split_index();
                (fidx, self.find_split_condition(fidx, c.split_value), c.default_left())
            })
            .collect();
        let sums = self.split_eval.multi_child_sums(
            &scan,
            self.n_bins,
            self.n_targets,
            &inputs,
            &splits,
        );

        Ok(nodes
            .iter()
            .zip(candidates)
            .enumerate()
            .map(|(i, (node, split))| {
                let pairs = &sums[i * self.n_targets..(i + 1) * self.n_targets];
                let left_target_sums: Vec<GradientPairInt64> = pairs.iter().map(|p| p.0).collect();
                let right_target_sums: Vec<GradientPairInt64> = pairs.iter().map(|p| p.1).collect();
                let left = GradientPairInt64 { grad: split.left_grad, hess: split.left_hess };
                debug_assert!(
                    split.loss_chg <= RT_EPS
                        || left
                            == left_target_sums
                                .iter()
                                .fold(GradientPairInt64::default(), |a, b| a + *b),
                    "the per-target child sums must add up to the candidate's own"
                );
                ExpandEntry {
                    split,
                    left_stats: self.decode(left),
                    right_stats: self.decode(node.sum - left),
                    left_target_sums,
                    right_target_sums,
                    ..node.clone()
                }
            })
            .collect())
    }

    /// Read a shared frontier buffer back to host, decoding the accumulator's
    /// 4 `u32` words per bin into quantised `[grad, hess]` pairs.
    ///
    /// Only called when a batch has at least one categorical feature to
    /// evaluate — the common all-numeric case never pays for this readback.
    /// Mirrors [`HistogramEngine::read`](super::histogram::HistogramEngine::read),
    /// checked rather than debug-asserted for the same reason it is: a wrong
    /// bin count here would otherwise index the per-node slice out of bounds.
    fn read_histogram(
        &self,
        hist: &Handle,
        hist_bins: usize,
        first_bin: usize,
        n_bins: usize,
    ) -> Result<Vec<GradientPairInt64>> {
        // Four accumulator words a bin.
        let word = size_of::<u32>() as u64;
        if first_bin + n_bins > hist_bins {
            return Err(crate::error::Error::HistogramBins {
                expected: first_bin + n_bins,
                got: hist_bins,
            });
        }
        let range = hist
            .clone()
            .offset_start(first_bin as u64 * 4 * word)
            .offset_end((hist_bins - first_bin - n_bins) as u64 * 4 * word);
        let bytes = self.client.read_one_unchecked(range);
        let words: &[u32] = bytemuck::cast_slice(&bytes);
        // A pooled buffer can be wider than the bins it was asked for.
        if words.len() < n_bins * 4 {
            return Err(crate::error::Error::HistogramBins {
                expected: n_bins,
                got: words.len() / 4,
            });
        }
        Ok(words[..n_bins * 4]
            .chunks_exact(4)
            .map(|w| GradientPairInt64 {
                grad: (w[0] as i64) | ((w[1] as i64) << 32),
                hess: (w[2] as i64) | ((w[3] as i64) << 32),
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
        dev_gpairs: &[DeviceGpairs],
        partitioner: &RowPartitioner<R>,
        evaluator: &SplitEvaluator,
    ) -> Vec<ExpandEntry> {
        if expandable.is_empty() {
            return Vec::new();
        }
        let (frontier, base, slots) = self.take_slots(expandable.len() * 2);
        let frontier_bins = self.frontier_bins(slots);
        let n_bins = self.n_bins as u32;
        // Bins between one node's slot and the next: a node holds one histogram
        // per target, side by side.
        let stride = n_bins * self.n_targets as u32;
        let mut out = Vec::with_capacity(expandable.len() * 2);
        // Collected across the batch so the level costs one histogram launch
        // and one subtraction launch, not two per node.
        let mut hist_jobs = Vec::with_capacity(expandable.len());
        let mut sub_slots = Vec::with_capacity(expandable.len());

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
            let (build_slot, sub_slot) =
                ((base + i * 2) as u32 * stride, (base + i * 2 + 1) as u32 * stride);
            let left_side =
                (e.left, left_seg, left_sum, parent.left_stats, &parent.left_target_sums);
            let right_side =
                (e.right, right_seg, right_sum, parent.right_stats, &parent.right_target_sums);
            let (build, subtract) = if fewer_right {
                (right_side, left_side)
            } else {
                (left_side, right_side)
            };

            hist_jobs.push(NodeHistJob {
                ridx_base: build.1.0,
                n_ridx: build.1.1,
                slot: build_slot,
            });
            sub_slots.push((parent.slot, build_slot, sub_slot));

            for ((nid, seg, sum, stats, targets), slot) in
                [(build, build_slot), (subtract, sub_slot)]
            {
                // Upstream evaluates a child's gain against the *parent's* node
                // id, so the parent's weight box applies here.
                let root_gain = if self.is_multi() {
                    targets
                        .iter()
                        .map(|s| evaluator.calc_gain(parent.nid, &self.param, &self.decode(*s)))
                        .sum()
                } else {
                    evaluator.calc_gain(parent.nid, &self.param, &stats)
                };
                out.push(ExpandEntry {
                    nid,
                    depth: parent.depth + 1,
                    seg_begin: seg.0,
                    seg_len: seg.1,
                    hist: frontier.clone(),
                    hist_bins: frontier_bins,
                    slot,
                    sum,
                    root_gain,
                    split: DeviceSplitCandidate::default(),
                    left_stats: GradStats::default(),
                    right_stats: GradStats::default(),
                    cat_bits: Vec::new(),
                    target_sums: targets.clone(),
                    left_target_sums: Vec::new(),
                    right_target_sums: Vec::new(),
                });
            }
        }

        // One launch for every node's histogram, then one for every sibling.
        // Every parent of a batch shares a buffer, which is what lets the
        // subtraction be batched too.
        let parent_hist = batch[expandable[0].batch_pos].hist.clone();
        let parent_bins = batch[expandable[0].batch_pos].hist_bins;
        debug_assert!(
            expandable.iter().all(|e| batch[e.batch_pos].hist_bins == parent_bins),
            "a batch's parents must share one histogram allocation"
        );
        //
        // A vector-leaf level runs the same two launches once per target, over
        // the same row sets and into the target's own slice of each node's
        // slot — the device counterpart of `HistGrower::build_hists` running
        // its scalar kernel once per column.
        //
        // Single target: the multi-chunk nodes' partials are left for the
        // subtraction below to fold in, which saves the level a launch on a
        // runtime where a launch is a thread hand-off. The vector-leaf path
        // keeps the merge launch: its subtraction slots are per (node, target)
        // and the pending tables are per node.
        let mut pending = None;
        for (t, gpairs) in dev_gpairs.iter().enumerate() {
            let offset = t as u32 * n_bins;
            let jobs: Vec<NodeHistJob> = if offset == 0 {
                hist_jobs.clone()
            } else {
                hist_jobs.iter().map(|j| NodeHistJob { slot: j.slot + offset, ..*j }).collect()
            };
            if self.is_multi() {
                self.engine.build_into_batch(
                    gpairs,
                    partitioner.ridx(),
                    partitioner.n_rows(),
                    &jobs,
                    &frontier,
                    frontier_bins,
                );
            } else {
                pending = self.engine.build_into_batch_deferred(
                    gpairs,
                    partitioner.ridx(),
                    partitioner.n_rows(),
                    &jobs,
                    &frontier,
                    frontier_bins,
                );
            }
        }
        // The subtraction is per `(node, target)`: each target's histogram is
        // `parent - built` in its own right.
        let slots: Vec<(u32, u32, u32)> = if self.is_multi() {
            sub_slots
                .iter()
                .flat_map(|&(p, b, o)| {
                    (0..self.n_targets as u32)
                        .map(move |t| (p + t * n_bins, b + t * n_bins, o + t * n_bins))
                })
                .collect()
        } else {
            sub_slots
        };
        self.engine.subtract_batch(
            &parent_hist,
            parent_bins,
            &frontier,
            frontier_bins,
            &slots,
            pending.as_ref(),
        );

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
        if self.is_multi() {
            return self.apply_split_multi(entry, tree, evaluator);
        }
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

        if entry.split.is_cat {
            tree.expand_categorical(
                nid,
                entry.split.split_index(),
                &entry.cat_bits,
                entry.split.default_left(),
                base_weight,
                left_weight * lr,
                right_weight * lr,
                entry.split.loss_chg,
                parent_sum.sum_hess as f32,
                entry.left_stats.sum_hess as f32,
                entry.right_stats.sum_hess as f32,
            );
        } else {
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
        }

        let node = tree.nodes[nid];
        let (left, right) = (node.left as usize, node.right as usize);
        evaluator.add_split(nid, left, right, node.split_index, left_weight, right_weight);
        self.constraints.split(nid, node.split_index, left, right);
        (left, right)
    }

    /// [`apply_split`](Self::apply_split) for a vector-leaf tree: one shared
    /// split decision, one leaf value per target.
    ///
    /// Port of `HistGrower::apply_split_multi`, including where the summed
    /// statistics are used instead of the per-target ones — the node's cover,
    /// the tree's recorded gain, and the monotone box handed to the children,
    /// which is stated per feature rather than per target.
    fn apply_split_multi(
        &mut self,
        entry: &ExpandEntry,
        tree: &mut RegTree,
        evaluator: &mut SplitEvaluator,
    ) -> (usize, usize) {
        let mut parent_sum = entry.left_stats;
        parent_sum.add_stats(&entry.right_stats);
        let nid = entry.nid;
        let p = &self.param;
        let lr = p.learning_rate;

        let base_weight = evaluator.calc_weight(nid, p, &parent_sum);
        let left_weights: Vec<f32> = entry
            .left_target_sums
            .iter()
            .map(|s| lr * evaluator.calc_weight(nid, p, &self.decode(*s)))
            .collect();
        let right_weights: Vec<f32> = entry
            .right_target_sums
            .iter()
            .map(|s| lr * evaluator.calc_weight(nid, p, &self.decode(*s)))
            .collect();

        tree.expand_node_multi(
            nid,
            entry.split.split_index(),
            entry.split.split_value,
            entry.split.default_left(),
            base_weight,
            &left_weights,
            &right_weights,
            entry.split.loss_chg,
            parent_sum.sum_hess as f32,
            entry.left_stats.sum_hess as f32,
            entry.right_stats.sum_hess as f32,
        );

        let node = tree.nodes[nid];
        let (left, right) = (node.left as usize, node.right as usize);
        evaluator.add_split(
            nid,
            left,
            right,
            node.split_index,
            evaluator.calc_weight(nid, p, &entry.left_stats),
            evaluator.calc_weight(nid, p, &entry.right_stats),
        );
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
        // A categorical split ignores `cond` entirely — the partitioner tests
        // `cat_bits` instead — so there is no threshold to derive.
        let cond = if entry.split.is_cat {
            -1
        } else {
            let global_cond = self.find_split_condition(fidx, node.value);
            // `-1` (no matching cut) stays negative, so every present row goes
            // right and only the missing ones follow `default_left`.
            if global_cond < 0 { -1 } else { global_cond - self.cuts.cut_ptrs[fidx as usize] as i64 }
        };
        SegmentSplit {
            begin: entry.seg_begin,
            len: entry.seg_len,
            fidx,
            cond,
            default_left: node.default_left,
            cat_bits: if entry.split.is_cat { entry.cat_bits.clone() } else { Vec::new() },
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

    /// A buffer holding `slots` node histograms side by side, ready for the
    /// engine to build into — zeroed where its kernel only adds, and left
    /// untouched where its kernel writes every bin (see
    /// `HistogramEngine::frontier`).
    /// `n` zeroed histogram slots: the buffer, the first slot's index, and
    /// the buffer's slot count. From the current arena while it has room,
    /// else from a fresh one.
    fn take_slots(&mut self, n: usize) -> (Handle, usize, usize) {
        if let Some(a) = &mut self.arena
            && a.next + n <= a.slots
        {
            let base = a.next;
            a.next += n;
            return (a.handle.clone(), base, a.slots);
        }
        let slots = n.max(ARENA_SLOTS);
        let handle = self.zeroed_frontier(slots);
        self.arena = Some(Arena { handle: handle.clone(), slots, next: n });
        (handle, 0, slots)
    }

    fn zeroed_frontier(&mut self, slots: usize) -> Handle {
        // Four u32 accumulator words per bin, which is also two i64 words.
        let words = self.frontier_bins(slots) * 4;
        // A buffer nobody else holds, wide enough; a level's histograms are
        // read only through the handles its entries carry, so the pool's
        // being the sole holder is exactly "no longer in use".
        let reused = self.frontiers.iter().find(|(cap, h)| *cap >= words && h.can_mut());
        match reused {
            Some((_, h)) => {
                let h = h.clone();
                self.engine.prepare_frontier(&h, words);
                h
            }
            None => {
                let h = self.engine.frontier(words);
                self.frontiers.push((words, h.clone()));
                h
            }
        }
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
