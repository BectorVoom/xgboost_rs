//! Gradient boosted model: the tree ensemble and one boosting round.
//!
//! A round grows `num_output_group * num_parallel_tree` trees. The output
//! groups are what `num_class` and `num_target` become inside the booster:
//! each group has its own column of gradients and its own column of the
//! prediction buffer, and `tree_info` records which group a tree belongs to so
//! prediction can put the leaf values back in the right column.

use crate::context::Context;
use crate::data::{DMatrix, cuts::HistogramCuts, gradient_index::GHistIndex};
use crate::linear::GBLinear;
use crate::objective::GradientPair;
use crate::parameters::{DartNormalizeType, DartParameters, DartSampleType};
use crate::predictor::TreeRange;
use crate::data::csc::CscPage;
use crate::parameters::{ProcessType, TreeUpdaterName};
use crate::tree::colmaker::ColMaker;
use crate::tree::hist::HistGrower;
use crate::tree::model::RegTree;
use crate::tree::param::TrainParam;
use crate::tree::refresh::refresh;
use crate::tree::sampler::RowSampler;

/// `kRtEps`, the floor the weighted dropout sampler compares against.
const RT_EPS: f32 = 1e-6;

/// The DART dropout settings a booster runs with.
#[derive(Clone, Copy, Debug)]
pub struct DartConfig {
    pub sample_type: DartSampleType,
    pub normalize_type: DartNormalizeType,
    pub rate_drop: f32,
    pub one_drop: bool,
    pub skip_drop: f32,
    /// The learning rate, which DART's normalisation reads.
    pub learning_rate: f32,
}

impl DartConfig {
    pub fn from_parameters(dart: &DartParameters) -> Self {
        Self {
            sample_type: dart.sample_type,
            normalize_type: dart.normalize_type,
            rate_drop: dart.rate_drop,
            one_drop: dart.one_drop,
            skip_drop: dart.skip_drop,
            learning_rate: dart.tree.eta,
        }
    }
}

/// The gradient booster a fit runs: an ensemble of trees, or a linear model.
///
/// This is XGBoost's `GradientBooster` interface reduced to what a fit and a
/// prediction actually ask of it. It is an enum rather than a trait object
/// because the two implementations are the whole set — upstream registers no
/// others — and because prediction needs to ask which one it has (a linear
/// model has no leaves and no tree range).
pub enum Booster {
    /// `gbtree` or `dart`.
    Tree(GBTree),
    /// `gblinear`. Boxed because a tree booster is much the larger of the two
    /// and every `Booster` would otherwise be sized for the bigger one.
    Linear(Box<GBLinear>),
}

impl Booster {
    /// The trees, for the callers that only make sense for a tree model.
    pub fn tree(&self) -> Option<&GBTree> {
        match self {
            Self::Tree(t) => Some(t),
            Self::Linear(_) => None,
        }
    }

    /// The linear model, if this is a `gblinear` booster.
    pub fn linear(&self) -> Option<&GBLinear> {
        match self {
            Self::Linear(l) => Some(l),
            Self::Tree(_) => None,
        }
    }

    /// The XGBoost `booster` name this model records itself under.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Tree(t) => {
                if t.is_dart() {
                    "dart"
                } else {
                    "gbtree"
                }
            }
            Self::Linear(_) => "gblinear",
        }
    }

    pub fn num_feature(&self) -> usize {
        match self {
            Self::Tree(t) => t.model.num_feature,
            Self::Linear(l) => l.model.num_feature,
        }
    }

    pub fn num_output_group(&self) -> usize {
        match self {
            Self::Tree(t) => t.model.num_output_group.max(1),
            Self::Linear(l) => l.model.num_output_group.max(1),
        }
    }

    pub fn set_num_output_group(&mut self, n: usize) {
        match self {
            Self::Tree(t) => t.set_num_output_group(n),
            Self::Linear(l) => l.set_num_output_group(n),
        }
    }

    pub fn boosted_rounds(&self) -> usize {
        match self {
            Self::Tree(t) => t.model.num_rounds(),
            Self::Linear(l) => l.model.num_boosted_rounds,
        }
    }

    /// Tell a tree booster whether the objective's hessian is constant, which
    /// only `approx` reads. A no-op for `gblinear`.
    pub fn set_constant_hessian(&mut self, constant: bool) {
        if let Self::Tree(t) = self {
            t.set_constant_hessian(constant);
        }
    }

    /// Prepare for training on `dtrain`. Idempotent.
    pub fn configure(&mut self, dtrain: &DMatrix) -> crate::Result<()> {
        match self {
            Self::Tree(t) => t.configure(dtrain),
            Self::Linear(l) => l.configure(dtrain),
        }
    }

    /// Work done before the round's gradients are taken. Only DART has any.
    pub fn pre_boost(&mut self, ctx: &mut Context, dtrain: &DMatrix, preds: &mut [f32]) {
        if let Self::Tree(t) = self {
            t.pre_boost(ctx, dtrain, preds);
        }
    }

    /// Run one boosting round, folding its output into `preds`.
    pub fn do_boost(
        &mut self,
        ctx: &mut Context,
        dtrain: &DMatrix,
        gpair: &[GradientPair],
        preds: &mut [f32],
    ) -> crate::Result<()> {
        match self {
            Self::Tree(t) => t.do_boost(ctx, dtrain, gpair, preds),
            Self::Linear(l) => l.do_boost(ctx, dtrain, gpair, preds),
        }
    }

    /// Raw margins for `dmat`. `trees` is ignored by the linear booster, which
    /// has no rounds to slice — the API layer rejects a range there rather than
    /// silently ignoring one.
    pub fn predict_margin(
        &self,
        dmat: &DMatrix,
        base_margin: &[f32],
        trees: TreeRange,
    ) -> Vec<f32> {
        match self {
            Self::Tree(t) => crate::predictor::predict_margin(&t.model, dmat, base_margin, trees),
            Self::Linear(l) => l.predict_margin(dmat, base_margin),
        }
    }
}

/// The ensemble produced by boosting, upstream's `GBTreeModel`.
#[derive(Clone, Debug, Default)]
pub struct GBTreeModel {
    pub trees: Vec<RegTree>,
    /// Output group of each tree, upstream's `tree_info`.
    pub tree_info: Vec<u32>,
    /// Per-tree weight, upstream's `weight_drop`. Every entry is `1` for
    /// `gbtree`; DART rescales them as it drops and grows trees.
    pub tree_weight: Vec<f32>,
    pub num_feature: usize,
    /// Trees grown per boosting round *per output group*; `> 1` makes each
    /// round a small forest.
    pub num_parallel_tree: u32,
    /// Outputs per row. One column of predictions per group.
    pub num_output_group: usize,
}

impl GBTreeModel {
    pub fn new(num_feature: usize) -> Self {
        Self {
            trees: Vec::new(),
            tree_info: Vec::new(),
            tree_weight: Vec::new(),
            num_feature,
            num_parallel_tree: 1,
            num_output_group: 1,
        }
    }

    /// Weight of tree `i`; `1` unless DART has rescaled it.
    #[inline]
    pub fn weight_of(&self, i: usize) -> f32 {
        self.tree_weight.get(i).copied().unwrap_or(1.0)
    }

    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

    /// Trees a single boosting round contributes.
    pub fn trees_per_round(&self) -> usize {
        self.num_parallel_tree.max(1) as usize * self.num_output_group.max(1)
    }

    /// Boosting rounds represented by the ensemble.
    pub fn num_rounds(&self) -> usize {
        self.trees.len() / self.trees_per_round()
    }

    /// The output group tree `i` belongs to.
    #[inline]
    pub fn group_of(&self, i: usize) -> usize {
        self.tree_info.get(i).copied().unwrap_or(0) as usize
    }
}

/// `gbtree` booster over the CPU `hist` tree method.
pub struct GBTree {
    pub model: GBTreeModel,
    param: TrainParam,
    /// Binned training matrix, built once and reused every round. Only the
    /// histogram growers need it.
    gindex: Option<GHistIndex>,
    /// Built once per fit: it owns the device-resident ELLPACK, which is by
    /// far the most expensive thing a GPU fit uploads.
    #[cfg(feature = "gpu")]
    gpu_grower: Option<crate::gpu::grower::GpuHistGrower<crate::gpu::DefaultRuntime>>,
    /// Value-sorted column view, which only the `exact` grower needs.
    sorted: Option<CscPage>,
    /// Scratch for one group's gradients, and for the sampled copy of them.
    group_gpair: Vec<GradientPair>,
    sampled: Vec<GradientPair>,
    /// Dropout settings; `None` makes this an ordinary `gbtree`.
    dart: Option<DartConfig>,
    /// Trees dropped for the round being grown.
    dropped: Vec<usize>,
    /// The ensemble `process_type=update` is revisiting. Each round moves one
    /// round's worth of trees out of here and into the model, rewritten.
    trees_to_update: Vec<RegTree>,
    /// Whether the objective's hessian never changes, which lets `approx`
    /// sketch its quantiles once instead of once per round.
    constant_hessian: bool,
}

impl GBTree {
    pub fn new(num_feature: usize, param: TrainParam) -> Self {
        let mut model = GBTreeModel::new(num_feature);
        model.num_parallel_tree = param.num_parallel_tree;
        Self {
            model,
            param,
            gindex: None,
            #[cfg(feature = "gpu")]
            gpu_grower: None,
            sorted: None,
            group_gpair: Vec::new(),
            sampled: Vec::new(),
            dart: None,
            dropped: Vec::new(),
            trees_to_update: Vec::new(),
            constant_hessian: false,
        }
    }

    /// Tell the booster whether the objective's hessian is constant. Only
    /// `approx` reads it.
    pub fn set_constant_hessian(&mut self, constant: bool) {
        self.constant_hessian = constant;
    }

    /// Turn this booster into `dart`: every round drops a random subset of the
    /// existing trees before computing gradients, and rescales what survives.
    pub fn set_dart(&mut self, dart: DartConfig) {
        self.dart = Some(dart);
    }

    /// Whether dropout is in effect.
    pub fn is_dart(&self) -> bool {
        self.dart.is_some()
    }

    pub fn param(&self) -> &TrainParam {
        &self.param
    }

    /// Set how many output groups a round grows trees for.
    pub fn set_num_output_group(&mut self, n: usize) {
        self.model.num_output_group = n.max(1);
    }

    /// Quantile cuts of the cached binned matrix, if it has been built.
    pub fn cuts(&self) -> Option<&HistogramCuts> {
        self.gindex.as_ref().map(|g| &g.cuts)
    }

    /// Bin `dtrain` if it has not been binned yet.
    ///
    /// Cuts depend only on the data and `max_bin`, so this is done once per
    /// booster rather than once per round.
    pub fn configure(&mut self, dtrain: &DMatrix) -> crate::Result<()> {
        // `GBTreeModel::InitTreesToUpdate`: under `update` the model starts
        // empty and the existing ensemble is set aside. Each round then moves
        // one round of it back, rewritten — so a fit that stops early leaves a
        // *shorter* model rather than a half-rewritten one, and the prediction
        // cache grows exactly as it does for an ordinary fit.
        if self.param.process_type == ProcessType::Update && self.trees_to_update.is_empty() {
            self.trees_to_update = std::mem::take(&mut self.model.trees);
            self.model.tree_info.clear();
            self.model.tree_weight.clear();
        }
        match self.grower_kind() {
            Some(TreeUpdaterName::GrowColMaker) => {
                if self.sorted.is_none() {
                    self.sorted = Some(crate::threading::install(|| {
                        CscPage::build_sorted(dtrain, 0..dtrain.num_row())
                    }));
                }
            }
            // `approx` re-sketches per round from the current hessians, so
            // there is nothing to build ahead of time.
            Some(TreeUpdaterName::GrowHistMaker) => {}
            // The GPU grower bins into its own ELLPACK and never reads the
            // CPU binned matrix, so only the cuts are built here.
            #[cfg(feature = "gpu")]
            Some(TreeUpdaterName::GrowGpuHist) => {
                if self.gpu_grower.is_none() {
                    let cuts = crate::data::cuts::build_cuts(dtrain, self.param.max_bin)?;
                    let client = crate::gpu::default_client(self.param.device.ordinal().unwrap_or(0).max(0) as usize);
                    // A vector-leaf fit grows one tree covering every output,
                    // so the device grower needs the target count up front:
                    // it sizes the per-`(node, target)` histograms.
                    let n_targets = if self.param.multi_output_tree {
                        self.model.num_output_group.max(1)
                    } else {
                        1
                    };
                    self.gpu_grower = Some(crate::gpu::grower::GpuHistGrower::new_multi(
                        client,
                        dtrain,
                        cuts,
                        self.param.clone(),
                        n_targets,
                    )?);
                }
            }
            Some(_) if self.gindex.is_none() => {
                let cuts = crate::data::cuts::build_cuts(dtrain, self.param.max_bin)?;
                self.gindex = Some(crate::data::gradient_index::build_gradient_index_with(
                    dtrain,
                    &cuts,
                    self.param.sparse_threshold,
                )?);
            }
            Some(_) => {}
            // `process_type=update` has no grower: it rewrites existing trees.
            None => {}
        }
        Ok(())
    }

    /// The updater that grows trees this round, or `None` under
    /// `process_type=update` where every stage only modifies existing ones.
    fn grower_kind(&self) -> Option<TreeUpdaterName> {
        self.param.updaters.iter().copied().find(|u| !u.can_modify_tree())
    }

    /// Choose this round's dropped trees and remove them from `preds`.
    ///
    /// Called before the objective computes gradients, because the whole point
    /// of DART is that a round's gradients are the residuals of a *thinned*
    /// ensemble. A no-op for `gbtree`.
    pub fn pre_boost(&mut self, ctx: &mut Context, dtrain: &DMatrix, preds: &mut [f32]) {
        self.dropped = self.choose_dropped(ctx.rng());
        self.remove_trees(&self.dropped.clone(), dtrain, preds);
    }

    /// `Dart::DropTrees` — pick the trees this round leaves out.
    ///
    /// Empty for `gbtree`, for an empty ensemble, and whenever `skip_drop`
    /// fires.
    fn choose_dropped(&self, rng: &mut crate::rng::Mt19937) -> Vec<usize> {
        let mut dropped = Vec::new();
        let Some(dart) = self.dart else { return dropped };
        if self.model.trees.is_empty() {
            return dropped;
        }

        // `skip_drop` skips the whole dropout for this round.
        if dart.skip_drop > 0.0 && (rng.next_f64() as f32) < dart.skip_drop {
            return dropped;
        }

        let n = self.model.trees.len();
        match dart.sample_type {
            DartSampleType::Uniform => {
                for i in 0..n {
                    if (rng.next_f64() as f32) < dart.rate_drop {
                        dropped.push(i);
                    }
                }
                if dart.one_drop && dropped.is_empty() {
                    let pick = (rng.next_f64() * n as f64) as usize;
                    dropped.push(pick.min(n - 1));
                }
            }
            DartSampleType::Weighted => {
                let sum_weight: f32 = self.model.tree_weight.iter().sum();
                if sum_weight > RT_EPS {
                    for i in 0..n {
                        let p = dart.rate_drop * n as f32 * self.model.weight_of(i) / sum_weight;
                        if (rng.next_f64() as f32) < p {
                            dropped.push(i);
                        }
                    }
                    if dart.one_drop && dropped.is_empty() {
                        // A weight-proportional draw, the discrete
                        // distribution upstream falls back to.
                        let target = rng.next_f64() as f32 * sum_weight;
                        let mut acc = 0.0f32;
                        let mut pick = n - 1;
                        for i in 0..n {
                            acc += self.model.weight_of(i);
                            if acc >= target {
                                pick = i;
                                break;
                            }
                        }
                        dropped.push(pick);
                    }
                } else {
                    // Every weight has decayed to nothing: fall back to uniform.
                    for i in 0..n {
                        if (rng.next_f64() as f32) < dart.rate_drop {
                            dropped.push(i);
                        }
                    }
                }
            }
        }
        dropped
    }

    /// Subtract the named trees' contributions from `preds`.
    fn remove_trees(&self, trees: &[usize], dmat: &DMatrix, preds: &mut [f32]) {
        let n_groups = self.model.num_output_group.max(1);
        for &t in trees {
            let tree = &self.model.trees[t];
            let group = self.model.group_of(t);
            let weight = self.model.weight_of(t);
            for r in 0..dmat.num_row() {
                let leaf = tree.leaf_index(|f| feature_value(dmat, r, f));
                preds[r * n_groups + group] -= weight * tree.nodes[leaf].value;
            }
        }
    }

    /// Apply a DART dropout to an already-computed margin, which is what a
    /// prediction asked for with `training = true` wants: the thinned ensemble
    /// a training round would see, not the full one.
    ///
    /// A no-op for `gbtree`, which drops nothing. The draw is seeded from
    /// `seed` rather than the session engine, so repeating the call repeats the
    /// answer — a prediction is not a round, and must not advance the state a
    /// later round depends on.
    pub fn apply_training_dropout(&self, seed: u32, dmat: &DMatrix, preds: &mut [f32]) -> usize {
        if self.dart.is_none() {
            return 0;
        }
        let mut rng = crate::rng::Mt19937::new(seed);
        let dropped = self.choose_dropped(&mut rng);
        self.remove_trees(&dropped, dmat, preds);
        dropped.len()
    }

    /// Run one round of the updater pipeline and fold its output into
    /// `preds`.
    ///
    /// `gpair` and `preds` are both row-major `(row, group)`. With
    /// `num_parallel_tree > 1` every tree of a group is grown from the same
    /// gradients but its own row and column samples, which is what turns the
    /// fit into a boosted forest.
    ///
    /// Under `process_type=update` nothing is grown: the round takes the trees
    /// the model already holds at this round's slots and hands them to the
    /// tree-modifying stages instead, which is why the prediction update
    /// subtracts the old contribution before adding the new one.
    pub fn do_boost(
        &mut self,
        ctx: &mut Context,
        dtrain: &DMatrix,
        gpair: &[GradientPair],
        preds: &mut [f32],
    ) -> crate::Result<()> {
        self.configure(dtrain)?;
        let n_groups = self.model.num_output_group.max(1);
        let n_rows = dtrain.num_row();

        // DART's rescaling depends only on how many trees were dropped and how
        // many are about to be added, both of which are known now.
        let size_new = n_groups * self.param.num_parallel_tree.max(1) as usize;
        let (new_weight, drop_factor) = self.dart_weights(size_new);

        match self.grower_kind() {
            Some(kind) => self.grow_round(
                ctx, dtrain, gpair, preds, kind, n_groups, n_rows, new_weight,
            )?,
            None => self.update_round(dtrain, gpair, preds, n_groups, n_rows),
        }

        self.rescale_dropped(dtrain, preds, drop_factor);
        Ok(())
    }

    /// A `process_type=default` round: grow this round's trees.
    #[allow(clippy::too_many_arguments)]
    fn grow_round(
        &mut self,
        ctx: &mut Context,
        dtrain: &DMatrix,
        gpair: &[GradientPair],
        preds: &mut [f32],
        kind: TreeUpdaterName,
        n_groups: usize,
        n_rows: usize,
        new_weight: f32,
    ) -> crate::Result<()> {
        // One tree covering every output, rather than one tree per output.
        if self.param.multi_output_tree && n_groups > 1 {
            return self.grow_round_multi(ctx, dtrain, gpair, preds, n_groups, n_rows, new_weight);
        }

        // `approx` rebuilds its binned matrix from each group's hessians, so
        // it cannot share one grower across the round the way the other two do.
        if kind == TreeUpdaterName::GrowHistMaker {
            return self.grow_round_approx(ctx, dtrain, gpair, preds, n_groups, n_rows, new_weight);
        }

        let sampler = RowSampler::new(self.param.sampling_method, self.param.subsample);
        // The `exact` updater draws its own row sample, because it excludes a
        // row by marking its position rather than by zeroing its gradient.
        let exact = kind == TreeUpdaterName::GrowColMaker;
        let is_sampling = !exact && sampler.is_sampling(n_rows);

        #[cfg(feature = "gpu")]
        let gpu = self.gpu_grower.take();
        let param = &self.param;
        let mut grower = if exact {
            Grower::Exact(ColMaker::new(param, self.sorted.as_ref().expect("configured"), dtrain))
        } else {
            #[cfg(feature = "gpu")]
            match gpu {
                Some(g) => Grower::Gpu(Box::new(g), None),
                None => Grower::Hist(HistGrower::new(
                    param,
                    self.gindex.as_ref().expect("configured"),
                    dtrain,
                )),
            }
            #[cfg(not(feature = "gpu"))]
            Grower::Hist(HistGrower::new(param, self.gindex.as_ref().expect("configured"), dtrain))
        };
        let mut first_tree = true;

        for gid in 0..n_groups {
            // One group's gradients, gathered out of the interleaved buffer.
            let group_gpair: &[GradientPair] = if n_groups == 1 {
                gpair
            } else {
                self.group_gpair.clear();
                self.group_gpair.extend((0..n_rows).map(|r| gpair[r * n_groups + gid]));
                &self.group_gpair
            };

            for _ in 0..self.param.num_parallel_tree {
                if !first_tree {
                    grower.reset();
                }
                first_tree = false;

                // Sampling zeroes gradients, so it needs a copy the objective's
                // buffer can survive; without it the original is used untouched.
                let tree_gpair = if is_sampling {
                    self.sampled.clear();
                    self.sampled.extend_from_slice(group_gpair);
                    let seed = ctx.rng().next_u32() as u64;
                    sampler.sample(&mut self.sampled, seed, ctx.threads());
                    &self.sampled[..]
                } else {
                    group_gpair
                };

                let mut tree = RegTree::new(self.model.num_feature);
                grower.grow(ctx, tree_gpair, &mut tree);

                // The tree-modifying stages run after the grower, in the order
                // the pipeline names them. They can move leaf values and
                // renumber nodes, so the prediction cache is only safe to
                // update from the grower's row sets when nothing changed.
                let modified =
                    run_modifiers(&mut tree, &self.param.updaters, &self.param, dtrain, tree_gpair);
                if modified {
                    add_tree_predictions(&tree, dtrain, preds, n_groups, gid, new_weight);
                } else {
                    // The row sets already say which leaf each row reached, so
                    // no tree traversal is needed.
                    grower.update_predictions(&tree, preds, n_groups, gid, new_weight);
                }
                self.model.trees.push(tree);
                self.model.tree_info.push(gid as u32);
                self.model.tree_weight.push(new_weight);
            }
        }
        // Put the device grower back, so the next round reuses the uploaded
        // ELLPACK rather than binning and uploading the matrix again.
        #[cfg(feature = "gpu")]
        if let Grower::Gpu(g, _) = grower {
            self.gpu_grower = Some(*g);
        }
        Ok(())
    }

    /// A `process_type=default` round under `multi_strategy=multi_output_tree`.
    ///
    /// The difference from [`grow_round`](Self::grow_round) is what a tree is:
    /// one tree carries a vector leaf covering every output, so the round grows
    /// `num_parallel_tree` trees instead of `num_parallel_tree * n_groups`, and
    /// each split is a single decision that all the outputs share. That is the
    /// point of the strategy — the outputs are modelled as related rather than
    /// as independent problems — and it is also why the ensemble is smaller.
    #[allow(clippy::too_many_arguments)]
    fn grow_round_multi(
        &mut self,
        ctx: &mut Context,
        dtrain: &DMatrix,
        gpair: &[GradientPair],
        preds: &mut [f32],
        n_groups: usize,
        n_rows: usize,
        new_weight: f32,
    ) -> crate::Result<()> {
        let sampler = RowSampler::new(self.param.sampling_method, self.param.subsample);
        let is_sampling = sampler.is_sampling(n_rows);
        #[cfg(feature = "gpu")]
        let gpu = self.gpu_grower.take();
        let param = &self.param;
        let mut grower = {
            #[cfg(feature = "gpu")]
            match gpu {
                Some(g) => Grower::Gpu(Box::new(g), None),
                None => Grower::Hist(HistGrower::new_multi(
                    param,
                    self.gindex.as_ref().expect("configured"),
                    dtrain,
                    n_groups,
                )),
            }
            #[cfg(not(feature = "gpu"))]
            Grower::Hist(HistGrower::new_multi(
                param,
                self.gindex.as_ref().expect("configured"),
                dtrain,
                n_groups,
            ))
        };

        for i in 0..self.param.num_parallel_tree {
            if i > 0 {
                grower.reset();
            }

            // Row sampling drops whole rows, not single outputs: a row the
            // tree does not see contributes to none of them. The draw is made
            // against each row's summed gradient, so gradient-based sampling
            // still weighs a row by how much work it represents.
            let tree_gpair: &[GradientPair] = if is_sampling {
                let mut row_totals: Vec<GradientPair> = (0..n_rows)
                    .map(|r| {
                        let mut total = GradientPair::default();
                        for t in 0..n_groups {
                            let g = gpair[r * n_groups + t];
                            total.grad += g.grad;
                            total.hess += g.hess;
                        }
                        total
                    })
                    .collect();
                let seed = ctx.rng().next_u32() as u64;
                sampler.sample(&mut row_totals, seed, ctx.threads());

                self.sampled.clear();
                self.sampled.extend_from_slice(gpair);
                for (r, total) in row_totals.iter().enumerate() {
                    if total.hess == 0.0 && total.grad == 0.0 {
                        for t in 0..n_groups {
                            self.sampled[r * n_groups + t] = GradientPair::default();
                        }
                    }
                }
                &self.sampled
            } else {
                gpair
            };

            let mut tree = RegTree::new_multi(self.model.num_feature, n_groups);
            grower.grow(ctx, tree_gpair, &mut tree);
            grower.update_predictions(&tree, preds, n_groups, 0, new_weight);

            self.model.trees.push(tree);
            // A vector-leaf tree belongs to no single output, so it is recorded
            // under group 0 and prediction reads its leaf vector instead.
            self.model.tree_info.push(0);
            self.model.tree_weight.push(new_weight);
        }
        // Put the device grower back, so the next round reuses the uploaded
        // ELLPACK rather than binning and uploading the matrix again.
        #[cfg(feature = "gpu")]
        if let Grower::Gpu(g, _) = grower {
            self.gpu_grower = Some(*g);
        }
        Ok(())
    }

    /// A `process_type=default` round under the `approx` tree method.
    ///
    /// The difference from [`grow_round`](Self::grow_round) is where the
    /// quantile cuts come from: `approx` re-sketches the matrix each round,
    /// weighting every row by its hessian, so the bins track wherever the
    /// current model is least certain. That also fixes two smaller
    /// differences, both of which upstream has too — the row sample is drawn
    /// once per group rather than once per tree (the sketch has to be built
    /// from *some* sample), and the binned matrix is rebuilt per group.
    #[allow(clippy::too_many_arguments)]
    fn grow_round_approx(
        &mut self,
        ctx: &mut Context,
        dtrain: &DMatrix,
        gpair: &[GradientPair],
        preds: &mut [f32],
        n_groups: usize,
        n_rows: usize,
        new_weight: f32,
    ) -> crate::Result<()> {
        let sampler = RowSampler::new(self.param.sampling_method, self.param.subsample);
        let is_sampling = sampler.is_sampling(n_rows);

        for gid in 0..n_groups {
            let mut group_gpair: Vec<GradientPair> = if n_groups == 1 {
                gpair.to_vec()
            } else {
                (0..n_rows).map(|r| gpair[r * n_groups + gid]).collect()
            };
            if is_sampling {
                let seed = ctx.rng().next_u32() as u64;
                sampler.sample(&mut group_gpair, seed, ctx.threads());
            }

            // Skipped once built when the objective's hessian is constant:
            // re-sketching would give the same cuts, and upstream skips it too.
            if !(self.constant_hessian && self.gindex.is_some()) {
                self.gindex =
                    Some(build_approx_index(dtrain, &self.param, &group_gpair)?);
            }

            let param = &self.param;
            let mut grower =
                HistGrower::new(param, self.gindex.as_ref().expect("rebuilt above"), dtrain);
            for i in 0..self.param.num_parallel_tree {
                if i > 0 {
                    grower.reset();
                }
                let mut tree = RegTree::new(self.model.num_feature);
                grower.grow(ctx, &group_gpair, &mut tree);

                let modified = run_modifiers(
                    &mut tree,
                    &self.param.updaters,
                    &self.param,
                    dtrain,
                    &group_gpair,
                );
                if modified {
                    add_tree_predictions(&tree, dtrain, preds, n_groups, gid, new_weight);
                } else {
                    grower.update_predictions(&tree, preds, n_groups, gid, new_weight);
                }
                self.model.trees.push(tree);
                self.model.tree_info.push(gid as u32);
                self.model.tree_weight.push(new_weight);
            }
        }
        Ok(())
    }

    /// A `process_type=update` round: move one round of the set-aside
    /// ensemble back into the model, rewritten by the modifying stages.
    ///
    /// The gradients each round sees are the ones the *partially rebuilt*
    /// model produces, not the full original ensemble's — which is what makes
    /// `refresh` idempotent on an unchanged dataset: round `i` re-fits tree `i`
    /// against exactly the residuals it was originally grown from.
    fn update_round(
        &mut self,
        dtrain: &DMatrix,
        gpair: &[GradientPair],
        preds: &mut [f32],
        n_groups: usize,
        n_rows: usize,
    ) {
        let per_tree = self.param.num_parallel_tree.max(1) as usize;
        let round = self.model.trees.len() / (per_tree * n_groups);

        for gid in 0..n_groups {
            let group_gpair: &[GradientPair] = if n_groups == 1 {
                gpair
            } else {
                self.group_gpair.clear();
                self.group_gpair.extend((0..n_rows).map(|r| gpair[r * n_groups + gid]));
                &self.group_gpair
            };

            for i in 0..per_tree {
                let slot = round * per_tree * n_groups + gid * per_tree + i;
                let Some(tree) = self.trees_to_update.get_mut(slot) else {
                    // Upstream refuses to run more update rounds than the base
                    // model has; stopping leaves a valid, shorter ensemble.
                    return;
                };
                let mut tree = std::mem::replace(tree, RegTree::new(0));
                run_modifiers(&mut tree, &self.param.updaters, &self.param, dtrain, group_gpair);
                add_tree_predictions(&tree, dtrain, preds, n_groups, gid, 1.0);
                self.model.trees.push(tree);
                self.model.tree_info.push(gid as u32);
                self.model.tree_weight.push(1.0);
            }
        }
    }

    /// `Dart::NormalizeTrees`: the weight new trees take, and the factor the
    /// dropped ones are scaled by.
    fn dart_weights(&self, size_new: usize) -> (f32, f32) {
        let Some(dart) = self.dart else { return (1.0, 1.0) };
        let num_drop = self.dropped.len();
        if num_drop == 0 {
            return (1.0, 1.0);
        }
        let lr = dart.learning_rate / size_new.max(1) as f32;
        match dart.normalize_type {
            // `forest`: the whole dropped set counts as one tree.
            DartNormalizeType::Forest => {
                let factor = 1.0 / (1.0 + lr);
                (factor, factor)
            }
            // `tree`: the new tree replaces `num_drop` of them.
            DartNormalizeType::Tree => {
                let k = num_drop as f32;
                (1.0 / (k + lr), k / (k + lr))
            }
        }
    }

    /// Put the dropped trees back into the prediction at their new weight.
    fn rescale_dropped(&mut self, dtrain: &DMatrix, preds: &mut [f32], drop_factor: f32) {
        if self.dropped.is_empty() {
            return;
        }
        let n_groups = self.model.num_output_group.max(1);
        for &t in &self.dropped {
            self.model.tree_weight[t] *= drop_factor;
            let tree = &self.model.trees[t];
            let group = self.model.group_of(t);
            let weight = self.model.tree_weight[t];
            for r in 0..dtrain.num_row() {
                let leaf = tree.leaf_index(|f| feature_value(dtrain, r, f));
                preds[r * n_groups + group] += weight * tree.nodes[leaf].value;
            }
        }
        self.dropped.clear();
    }
}

/// Sketch and bin the matrix from a round's hessians, which is what makes
/// `approx` approximate: the bin boundaries follow the current model's
/// uncertainty rather than the raw data distribution.
///
/// A row the sampler excluded has a zero hessian and so a zero sketch weight,
/// which drops it from the quantiles as well as from the histograms.
fn build_approx_index(
    dtrain: &DMatrix,
    param: &TrainParam,
    gpair: &[GradientPair],
) -> crate::Result<GHistIndex> {
    let info = dtrain.info();
    let weights: Vec<f32> =
        gpair.iter().enumerate().map(|(r, g)| g.hess * info.weight(r)).collect();
    let cuts = crate::data::cuts::build_cuts_weighted(dtrain, param.max_bin, Some(&weights))?;
    crate::data::gradient_index::build_gradient_index_with(dtrain, &cuts, param.sparse_threshold)
}

/// The grower a round runs, chosen by the pipeline's first non-modifying
/// updater.
enum Grower<'a> {
    Hist(HistGrower<'a>),
    Exact(ColMaker<'a>),
    /// Owned rather than borrowed, because it holds device buffers the
    /// caller's `&mut self` cannot lend out for the length of a round; the
    /// caller takes it out of the model and puts it back.
    #[cfg(feature = "gpu")]
    Gpu(
        Box<crate::gpu::grower::GpuHistGrower<crate::gpu::DefaultRuntime>>,
        Option<crate::gpu::grower::GrownTree>,
    ),
}

impl Grower<'_> {
    fn reset(&mut self) {
        match self {
            Self::Hist(g) => g.reset(),
            Self::Exact(g) => g.reset(),
            // The device grower keeps nothing between trees but the uploaded
            // matrix, which is exactly what must survive.
            #[cfg(feature = "gpu")]
            Self::Gpu(_, last) => *last = None,
        }
    }

    fn grow(&mut self, ctx: &mut Context, gpair: &[GradientPair], tree: &mut RegTree) {
        match self {
            Self::Hist(g) => g.grow(ctx, gpair, tree),
            Self::Exact(g) => g.grow(ctx, gpair, tree),
            #[cfg(feature = "gpu")]
            Self::Gpu(g, last) => {
                *last = Some(
                    g.grow(gpair, tree, ctx.rng()).expect("device grow"),
                );
            }
        }
    }

    fn update_predictions(
        &self,
        tree: &RegTree,
        preds: &mut [f32],
        n_groups: usize,
        group: usize,
        weight: f32,
    ) {
        match self {
            Self::Hist(g) => g.update_predictions(tree, preds, n_groups, group, weight),
            Self::Exact(g) => g.update_predictions(tree, preds, n_groups, group, weight),
            #[cfg(feature = "gpu")]
            Self::Gpu(_, last) => last
                .as_ref()
                .expect("grow ran before update_predictions")
                .update_predictions(tree, preds, n_groups, group, weight),
        }
    }
}

/// Run every tree-modifying stage of `updaters`, in order.
///
/// Returns whether any of them actually changed the tree, which is what tells
/// the caller the grower's row sets are no longer a valid shortcut to the
/// prediction update.
fn run_modifiers(
    tree: &mut RegTree,
    updaters: &[TreeUpdaterName],
    param: &TrainParam,
    dtrain: &DMatrix,
    gpair: &[GradientPair],
) -> bool {
    let mut modified = false;
    for updater in updaters {
        match updater {
            TreeUpdaterName::Refresh => {
                refresh(tree, dtrain, gpair, param, param.refresh_leaf);
                modified = true;
            }
            TreeUpdaterName::Prune => {
                let pruned =
                    tree.prune(param.min_split_loss, param.max_depth, param.learning_rate);
                modified |= pruned > 0;
            }
            _ => {}
        }
    }
    modified
}

/// Add one tree's leaf values to `preds`, by traversal.
///
/// The slow path, used when the grower's row sets no longer describe the tree —
/// after a modifying stage — and when a round has to take an existing tree's
/// contribution back out.
fn add_tree_predictions(
    tree: &RegTree,
    dmat: &DMatrix,
    preds: &mut [f32],
    n_groups: usize,
    group: usize,
    weight: f32,
) {
    for r in 0..dmat.num_row() {
        let leaf = tree.leaf_index(|f| feature_value(dmat, r, f));
        preds[r * n_groups + group] += weight * tree.nodes[leaf].value;
    }
}

/// Value of feature `fidx` in row `r`, or `None` when missing.
#[inline]
pub fn feature_value(dmat: &DMatrix, r: usize, fidx: u32) -> Option<f32> {
    let (idx, val) = dmat.row(r);
    match idx.binary_search(&fidx) {
        Ok(k) => Some(val[k]),
        Err(_) => None,
    }
}
