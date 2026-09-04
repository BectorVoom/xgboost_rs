//! The `exact` tree updater — `grow_colmaker`.
//!
//! A port of `xgboost::tree::ColMaker` (`src/tree/updater_colmaker.cc`).
//!
//! Where `hist` bins every value up front and evaluates splits over a
//! histogram, `exact` keeps every distinct value as a split candidate. It scans
//! each feature's column *in value order*, accumulating a running gradient sum,
//! and proposes a split at every boundary between two different values:
//!
//! ```text
//! column, sorted by value:  v0 <= v1 <= v2 <= ... <= vn
//! candidate split points:      ^     ^        ^
//!                              (v0+v1)/2, (v1+v2)/2, ...
//! ```
//!
//! That makes it exact and makes it slow: the cost is the whole matrix per
//! level, with no binning to compress it.
//!
//! # Row position, not row partition
//!
//! `hist` physically partitions a row-index array so each node owns a
//! contiguous slice. `exact` cannot: it walks columns, and a column visits rows
//! of every node interleaved. Instead each row carries the node it currently
//! sits in (`position`), and one pass over a column touches every live node at
//! once — which is why the split scan keeps a separate running sum per node.
//!
//! # Missing values and `default_direction`
//!
//! A forward (ascending) scan puts the values it has already seen on the
//! *left*, so everything unseen — including the missing rows, which are simply
//! absent from the column — goes right. A backward scan is the mirror image.
//! Running both is how `learn` finds the better direction; `left` and `right`
//! pin it by running only one:
//!
//! | `default_direction` | forward scan | backward scan |
//! |---|---|---|
//! | `learn` | only when the column is sparser than `opt_dense_col` | yes |
//! | `left` | no | yes |
//! | `right` | yes | no |
//!
//! A fully dense column has no missing rows for the direction to matter to, so
//! `learn` skips its forward scan — that is exactly what `opt_dense_col` buys,
//! and why lowering it is a speed knob that can also move the `default_left`
//! flag recorded on dense splits.
//!
//! # Gamma is not applied here
//!
//! `ColMaker` splits on any positive loss change and leaves `gamma`
//! (`min_split_loss`) to the `prune` updater that follows it in the pipeline —
//! which is why `tree_method=exact` resolves to `grow_colmaker,prune` rather
//! than to the grower alone. Pruning after the fact can remove a split whose
//! children turned out to be worth keeping only together, which a greedy
//! pre-test cannot.

use rayon::prelude::*;

use super::column_sampler::ColumnSampler;
use super::evaluator::{InteractionConstraints, SplitEvaluator};
use super::model::RegTree;
use super::param::{GradStats, RT_EPS, SplitEntry, TrainParam};
use crate::context::Context;
use crate::data::DMatrix;
use crate::data::csc::CscPage;
use crate::gbm::feature_value;
use crate::objective::GradientPair;
use crate::parameters::DefaultDirection;
use crate::rng::{Mt19937, canonical_f64_from_mt};

/// Per-node state for the level being expanded, upstream's `NodeEntry`.
#[derive(Clone, Copy, Debug, Default)]
struct NodeEntry {
    stats: GradStats,
    root_gain: f32,
    weight: f32,
    best: SplitEntry,
}

/// The running state of one node inside one column scan, upstream's
/// `ThreadEntry`.
#[derive(Clone, Copy, Debug, Default)]
struct ScanEntry {
    stats: GradStats,
    last_fvalue: f32,
    best: SplitEntry,
}

/// How one node's split routes a row, gathered once per level so the row pass
/// needs no tree lookups.
#[derive(Clone, Copy, Debug)]
struct RouteRule {
    fidx: u32,
    cond: f32,
    default_left: bool,
    left: i32,
    right: i32,
}

/// Grows one tree with the exact greedy algorithm.
pub struct ColMaker<'a> {
    param: &'a TrainParam,
    dmat: &'a DMatrix,
    /// Every column in ascending value order, built once per booster.
    page: &'a CscPage,
    /// Stored fraction of each column, for the `opt_dense_col` rule.
    density: Vec<f32>,
    /// Node each row currently sits in.
    position: Vec<i32>,
    /// Whether a row still contributes; row sampling and negative hessians
    /// clear it. An invalid row keeps travelling down the tree so it still
    /// lands in a leaf and still gets a prediction.
    valid: Vec<bool>,
    snode: Vec<NodeEntry>,
    evaluator: SplitEvaluator,
    constraints: InteractionConstraints,
    column_sampler: ColumnSampler,
}

impl<'a> ColMaker<'a> {
    pub fn new(param: &'a TrainParam, page: &'a CscPage, dmat: &'a DMatrix) -> Self {
        let n_features = dmat.num_col();
        let n_rows = dmat.num_row();
        let density = (0..n_features)
            .map(|f| {
                if n_rows == 0 {
                    1.0
                } else {
                    page.column_len(f) as f32 / n_rows as f32
                }
            })
            .collect();
        Self {
            param,
            dmat,
            page,
            density,
            position: vec![0; n_rows],
            valid: vec![true; n_rows],
            snode: Vec::new(),
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
        }
    }

    /// Reset for another tree over the same data, keeping allocations.
    pub fn reset(&mut self) {
        self.position.fill(0);
        self.valid.fill(true);
        self.snode.clear();
        self.evaluator.reset();
        self.constraints.reset();
    }

    /// Grow `tree` from `gpair`, mirroring `ColMaker::Builder::Update`.
    ///
    /// `gpair` is the objective's gradients *unsampled*: unlike the `hist`
    /// path, `exact` draws its own row sample here, because it excludes rows by
    /// marking their position rather than by zeroing their gradients.
    pub fn grow(&mut self, ctx: &mut Context, gpair: &[GradientPair], tree: &mut RegTree) {
        let threads = ctx.threads();
        self.init_data(ctx.rng(), gpair);
        self.column_sampler.reset(self.dmat.num_col(), ctx.rng());
        let rng = ctx.rng();
        crate::threading::install_with(threads, || self.grow_inner(rng, gpair, tree));
    }

    fn grow_inner(&mut self, rng: &mut Mt19937, gpair: &[GradientPair], tree: &mut RegTree) {
        let mut expand: Vec<usize> = vec![0];
        self.init_node_stats(&expand, gpair, tree);
        self.snode[0].weight = self.evaluator.calc_weight(0, self.param, &self.snode[0].stats);
        self.snode[0].root_gain = self.evaluator.calc_gain(0, self.param, &self.snode[0].stats);

        for depth in 0..self.param.max_depth {
            self.find_split(depth, &expand, gpair, tree, rng);
            self.reset_position(&expand, tree);

            let mut next = Vec::with_capacity(expand.len() * 2);
            for &nid in &expand {
                if !tree.nodes[nid].is_leaf() {
                    next.push(tree.nodes[nid].left as usize);
                    next.push(tree.nodes[nid].right as usize);
                }
            }
            if next.is_empty() {
                expand.clear();
                break;
            }
            self.init_node_stats(&next, gpair, tree);
            for &nid in &next {
                let parent = tree.nodes[nid].parent as usize;
                let stats = self.snode[nid].stats;
                // Both children are weighed against the *parent's* monotone
                // box; their own is only derived by `add_split` below.
                self.snode[nid].weight = self.evaluator.calc_weight(parent, self.param, &stats);
                self.snode[nid].root_gain = self.evaluator.calc_gain(parent, self.param, &stats);
            }
            for &nid in &expand {
                if tree.nodes[nid].is_leaf() {
                    continue;
                }
                let node = tree.nodes[nid];
                let (left, right) = (node.left as usize, node.right as usize);
                self.evaluator.add_split(
                    nid,
                    left,
                    right,
                    node.split_index,
                    self.snode[left].weight,
                    self.snode[right].weight,
                );
                self.constraints.split(nid, node.split_index, left, right);
            }
            expand = next;
        }

        // Whatever is still queued when the depth limit is reached becomes a
        // leaf at its own optimal weight.
        for &nid in &expand {
            tree.set_leaf(nid, self.snode[nid].weight * self.param.learning_rate);
        }
        for nid in 0..tree.num_nodes() {
            tree.stats[nid].loss_chg = self.snode[nid].best.loss_chg;
            tree.stats[nid].base_weight = self.snode[nid].weight;
            tree.stats[nid].sum_hess = self.snode[nid].stats.sum_hess as f32;
        }
    }

    /// Add each row's leaf value to `preds`, using the row positions the grow
    /// left behind.
    ///
    /// Positions survive pruning only until the tree is renumbered, so this
    /// must be called before any `prune` pass; the caller does exactly that.
    pub fn update_predictions(
        &self,
        tree: &RegTree,
        preds: &mut [f32],
        n_groups: usize,
        group: usize,
        weight: f32,
    ) {
        for (r, &nid) in self.position.iter().enumerate() {
            let value = tree.nodes[nid as usize].value * weight;
            preds[r * n_groups + group] += value;
        }
    }

    /// `ColMaker::Builder::InitData`: exclude the rows this tree does not see.
    ///
    /// A negative hessian marks a row the objective has dropped;
    /// `subsample < 1` drops more. Upstream draws the coin flips sequentially
    /// from the session engine, so the draw order is part of the model and is
    /// reproduced here rather than parallelised.
    fn init_data(&mut self, rng: &mut Mt19937, gpair: &[GradientPair]) {
        self.position.fill(0);
        self.valid.fill(true);
        for (r, g) in gpair.iter().enumerate() {
            if g.hess < 0.0 {
                self.valid[r] = false;
            }
        }
        if self.param.subsample < 1.0 {
            let p = self.param.subsample as f64;
            for r in 0..self.valid.len() {
                if !self.valid[r] {
                    continue;
                }
                if canonical_f64_from_mt(rng) >= p {
                    self.valid[r] = false;
                }
            }
        }
    }

    /// Total each listed node's gradients from the rows currently sitting in
    /// it.
    fn init_node_stats(&mut self, nodes: &[usize], gpair: &[GradientPair], tree: &RegTree) {
        if self.snode.len() < tree.num_nodes() {
            self.snode.resize(tree.num_nodes(), NodeEntry::default());
        }
        for &nid in nodes {
            self.snode[nid] = NodeEntry::default();
        }
        let mut wanted = vec![false; self.snode.len()];
        for &nid in nodes {
            wanted[nid] = true;
        }
        for (r, &nid) in self.position.iter().enumerate() {
            if !self.valid[r] {
                continue;
            }
            let nid = nid as usize;
            if wanted[nid] {
                let g = gpair[r];
                self.snode[nid].stats.add(g.grad as f64, g.hess as f64);
            }
        }
    }

    /// `ColMaker::Builder::FindSplit`: score every candidate feature for every
    /// node of the level, then apply the winners.
    fn find_split(
        &mut self,
        depth: i32,
        expand: &[usize],
        gpair: &[GradientPair],
        tree: &mut RegTree,
        rng: &mut Mt19937,
    ) {
        // Slot of each expanding node, so a column scan can keep one running
        // sum per node in a dense little array rather than one per tree node.
        let mut slot = vec![usize::MAX; self.snode.len()];
        for (i, &nid) in expand.iter().enumerate() {
            slot[nid] = i;
        }

        // The column sample is drawn once per level, in level order, because
        // the draw advances the session engine and its position in that
        // sequence is part of the model.
        let features = self.column_sampler.feature_set(depth, rng);

        let per_feature: Vec<Vec<SplitEntry>> = features
            .par_iter()
            .map(|&fid| self.scan_feature(fid, expand, &slot, gpair))
            .collect();

        for candidates in &per_feature {
            for (i, &nid) in expand.iter().enumerate() {
                let best = candidates[i];
                self.snode[nid].best.update_entry(&best);
            }
        }

        for &nid in expand {
            let e = self.snode[nid];
            if e.best.loss_chg > RT_EPS {
                let lr = self.param.learning_rate;
                let left_weight = self.evaluator.calc_weight(nid, self.param, &e.best.left_sum);
                let right_weight = self.evaluator.calc_weight(nid, self.param, &e.best.right_sum);
                tree.expand_node(
                    nid,
                    e.best.split_index(),
                    e.best.split_value,
                    e.best.default_left(),
                    e.weight,
                    left_weight * lr,
                    right_weight * lr,
                    e.best.loss_chg,
                    e.stats.sum_hess as f32,
                    e.best.left_sum.sum_hess as f32,
                    e.best.right_sum.sum_hess as f32,
                );
            } else {
                tree.set_leaf(nid, e.weight * self.param.learning_rate);
            }
        }
    }

    /// Both scan directions over one column, as `UpdateSolution` chooses them.
    fn scan_feature(
        &self,
        fid: u32,
        expand: &[usize],
        slot: &[usize],
        gpair: &[GradientPair],
    ) -> Vec<SplitEntry> {
        let mut temp = vec![ScanEntry::default(); expand.len()];
        let (rows, values) = self.page.column(fid as usize);
        // An "indicator" column holds a single value, so a forward scan can
        // only ever propose the degenerate all-on-one-side split.
        let indicator = !values.is_empty() && values[0] == values[values.len() - 1];

        let direction = self.param.default_direction;
        let forward = direction == DefaultDirection::Right
            || (direction == DefaultDirection::Learn
                && self.density[fid as usize] < self.param.opt_dense_col
                && !indicator);
        let backward = direction != DefaultDirection::Right;

        if forward {
            self.enumerate(fid, rows.iter().zip(values), false, expand, slot, gpair, &mut temp);
        }
        if backward {
            // Only the running sums restart; the best candidate found so far
            // carries over, exactly as upstream's per-thread scratch does.
            for e in temp.iter_mut() {
                e.stats = GradStats::default();
            }
            self.enumerate(
                fid,
                rows.iter().rev().zip(values.iter().rev()),
                true,
                expand,
                slot,
                gpair,
                &mut temp,
            );
        }
        temp.into_iter().map(|e| e.best).collect()
    }

    /// `ColMaker::Builder::EnumerateSplit` for one direction.
    ///
    /// `backward` is upstream's `d_step == -1`: the running sum is then the
    /// *right* child and the missing rows join the left, which is what sets
    /// `default_left` on the candidates this direction produces.
    #[allow(clippy::too_many_arguments)]
    fn enumerate<'e, I>(
        &self,
        fid: u32,
        entries: I,
        backward: bool,
        expand: &[usize],
        slot: &[usize],
        gpair: &[GradientPair],
        temp: &mut [ScanEntry],
    ) where
        I: Iterator<Item = (&'e u32, &'e f32)>,
    {
        let p = self.param;
        for (&rid, &fvalue) in entries {
            let rid = rid as usize;
            if !self.valid[rid] {
                continue;
            }
            let nid = self.position[rid] as usize;
            let s = slot[nid];
            if s == usize::MAX || !self.constraints.query(nid, fid) {
                continue;
            }
            let g = gpair[rid];
            let e = &mut temp[s];
            // `GradStats::Empty()`: the first row of this node in this column.
            if e.stats.sum_hess == 0.0 {
                e.stats.add(g.grad as f64, g.hess as f64);
                e.last_fvalue = fvalue;
                continue;
            }
            if fvalue != e.last_fvalue && e.stats.sum_hess >= p.min_child_weight as f64 {
                let mut other = GradStats::default();
                other.set_subtract(&self.snode[nid].stats, &e.stats);
                if other.sum_hess >= p.min_child_weight as f64 {
                    // The midpoint, unless rounding collapses it onto the new
                    // value — then the previous value is the only threshold
                    // that still separates the two sides.
                    let proposed = (fvalue + e.last_fvalue) * 0.5;
                    let split_pt = if proposed == fvalue { e.last_fvalue } else { proposed };
                    let (left, right) =
                        if backward { (other, e.stats) } else { (e.stats, other) };
                    let gain = self.evaluator.calc_split_gain(nid, fid, p, &left, &right)
                        - self.snode[nid].root_gain;
                    e.best.update(gain, fid, split_pt, backward, left, right);
                }
            }
            e.stats.add(g.grad as f64, g.hess as f64);
            e.last_fvalue = fvalue;
        }

        // The last candidate: everything scanned on one side, everything
        // missing (and nothing else) on the other. Its threshold sits just past
        // the final value seen.
        for (i, &nid) in expand.iter().enumerate() {
            let e = &mut temp[i];
            let mut other = GradStats::default();
            other.set_subtract(&self.snode[nid].stats, &e.stats);
            if e.stats.sum_hess >= p.min_child_weight as f64
                && other.sum_hess >= p.min_child_weight as f64
            {
                let gap = e.last_fvalue.abs() + RT_EPS;
                let delta = if backward { -gap } else { gap };
                let (left, right) = if backward { (other, e.stats) } else { (e.stats, other) };
                let gain = self.evaluator.calc_split_gain(nid, fid, p, &left, &right)
                    - self.snode[nid].root_gain;
                e.best.update(gain, fid, e.last_fvalue + delta, backward, left, right);
            }
        }
    }

    /// `ColMaker::Builder::ResetPosition`: move every row into the child its
    /// split sends it to, and retire the rows of nodes that stopped growing.
    fn reset_position(&mut self, expand: &[usize], tree: &RegTree) {
        let mut split_of: Vec<Option<RouteRule>> = vec![None; self.snode.len()];
        let mut terminal = vec![false; self.snode.len()];
        for &nid in expand {
            let node = tree.nodes[nid];
            if node.is_leaf() {
                terminal[nid] = true;
            } else {
                split_of[nid] = Some(RouteRule {
                    fidx: node.split_index,
                    cond: node.value,
                    default_left: node.default_left,
                    left: node.left,
                    right: node.right,
                });
            }
        }

        for r in 0..self.position.len() {
            let nid = self.position[r] as usize;
            if terminal[nid] {
                self.valid[r] = false;
                continue;
            }
            let Some(rule) = split_of[nid] else {
                continue;
            };
            let goes_left = match feature_value(self.dmat, r, rule.fidx) {
                Some(v) => v < rule.cond,
                None => rule.default_left,
            };
            self.position[r] = if goes_left { rule.left } else { rule.right };
        }
    }
}
