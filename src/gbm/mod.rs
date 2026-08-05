//! Gradient boosted model: the tree ensemble and one boosting round.
//!
//! A round grows `num_output_group * num_parallel_tree` trees. The output
//! groups are what `num_class` and `num_target` become inside the booster:
//! each group has its own column of gradients and its own column of the
//! prediction buffer, and `tree_info` records which group a tree belongs to so
//! prediction can put the leaf values back in the right column.

use crate::context::Context;
use crate::data::{DMatrix, cuts::HistogramCuts, gradient_index::GHistIndex};
use crate::objective::GradientPair;
use crate::parameters::{DartNormalizeType, DartParameters, DartSampleType};
use crate::tree::hist::HistGrower;
use crate::tree::model::RegTree;
use crate::tree::param::TrainParam;
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
    /// Binned training matrix, built once and reused every round.
    gindex: Option<GHistIndex>,
    /// Scratch for one group's gradients, and for the sampled copy of them.
    group_gpair: Vec<GradientPair>,
    sampled: Vec<GradientPair>,
    /// Dropout settings; `None` makes this an ordinary `gbtree`.
    dart: Option<DartConfig>,
    /// Trees dropped for the round being grown.
    dropped: Vec<usize>,
}

impl GBTree {
    pub fn new(num_feature: usize, param: TrainParam) -> Self {
        let mut model = GBTreeModel::new(num_feature);
        model.num_parallel_tree = param.num_parallel_tree;
        Self {
            model,
            param,
            gindex: None,
            group_gpair: Vec::new(),
            sampled: Vec::new(),
            dart: None,
            dropped: Vec::new(),
        }
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
        if self.gindex.is_none() {
            let cuts = crate::data::cuts::build_cuts(dtrain, self.param.max_bin)?;
            self.gindex = Some(crate::data::gradient_index::build_gradient_index(dtrain, &cuts)?);
        }
        Ok(())
    }

    /// Choose this round's dropped trees and remove them from `preds`.
    ///
    /// Called before the objective computes gradients, because the whole point
    /// of DART is that a round's gradients are the residuals of a *thinned*
    /// ensemble. A no-op for `gbtree`.
    pub fn pre_boost(&mut self, ctx: &mut Context, dtrain: &DMatrix, preds: &mut [f32]) {
        self.dropped.clear();
        let Some(dart) = self.dart else { return };
        if self.model.trees.is_empty() {
            return;
        }

        // `skip_drop` skips the whole dropout for this round.
        if dart.skip_drop > 0.0 && (ctx.rng().next_f64() as f32) < dart.skip_drop {
            return;
        }

        let n = self.model.trees.len();
        match dart.sample_type {
            DartSampleType::Uniform => {
                for i in 0..n {
                    if (ctx.rng().next_f64() as f32) < dart.rate_drop {
                        self.dropped.push(i);
                    }
                }
                if dart.one_drop && self.dropped.is_empty() {
                    let pick = (ctx.rng().next_f64() * n as f64) as usize;
                    self.dropped.push(pick.min(n - 1));
                }
            }
            DartSampleType::Weighted => {
                let sum_weight: f32 = self.model.tree_weight.iter().sum();
                if sum_weight > RT_EPS {
                    for i in 0..n {
                        let p = dart.rate_drop * n as f32 * self.model.weight_of(i) / sum_weight;
                        if (ctx.rng().next_f64() as f32) < p {
                            self.dropped.push(i);
                        }
                    }
                    if dart.one_drop && self.dropped.is_empty() {
                        // A weight-proportional draw, the discrete
                        // distribution upstream falls back to.
                        let target = ctx.rng().next_f64() as f32 * sum_weight;
                        let mut acc = 0.0f32;
                        let mut pick = n - 1;
                        for i in 0..n {
                            acc += self.model.weight_of(i);
                            if acc >= target {
                                pick = i;
                                break;
                            }
                        }
                        self.dropped.push(pick);
                    }
                } else {
                    // Every weight has decayed to nothing: fall back to uniform.
                    for i in 0..n {
                        if (ctx.rng().next_f64() as f32) < dart.rate_drop {
                            self.dropped.push(i);
                        }
                    }
                }
            }
        }

        // Take the dropped trees back out of the running prediction.
        let n_groups = self.model.num_output_group.max(1);
        for &t in &self.dropped {
            let tree = &self.model.trees[t];
            let group = self.model.group_of(t);
            let weight = self.model.weight_of(t);
            for r in 0..dtrain.num_row() {
                let leaf = tree.leaf_index(|f| feature_value(dtrain, r, f));
                preds[r * n_groups + group] -= weight * tree.nodes[leaf].value;
            }
        }
    }

    /// Grow this round's trees from `gpair` and fold their output into `preds`.
    ///
    /// `gpair` and `preds` are both row-major `(row, group)`. With
    /// `num_parallel_tree > 1` every tree of a group is grown from the same
    /// gradients but its own row and column samples, which is what turns the
    /// fit into a boosted forest.
    pub fn do_boost(
        &mut self,
        ctx: &mut Context,
        dtrain: &DMatrix,
        gpair: &[GradientPair],
        preds: &mut [f32],
    ) -> crate::Result<()> {
        self.configure(dtrain)?;
        let gindex = self.gindex.as_ref().expect("configured above");
        let n_groups = self.model.num_output_group.max(1);
        let n_rows = dtrain.num_row();

        let sampler = RowSampler::new(self.param.sampling_method, self.param.subsample);
        let is_sampling = sampler.is_sampling(n_rows);
        let mut grower = HistGrower::new(&self.param, gindex, dtrain);
        let mut first_tree = true;

        // DART's rescaling depends only on how many trees were dropped and how
        // many are about to be added, both of which are known now.
        let size_new = n_groups * self.param.num_parallel_tree.max(1) as usize;
        let (new_weight, drop_factor) = self.dart_weights(size_new);

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
                // Prediction cache update: the row sets already say which leaf
                // each row reached, so no tree traversal is needed.
                grower.update_predictions(&tree, preds, n_groups, gid, new_weight);
                self.model.trees.push(tree);
                self.model.tree_info.push(gid as u32);
                self.model.tree_weight.push(new_weight);
            }
        }

        self.rescale_dropped(dtrain, preds, drop_factor);
        Ok(())
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

/// Value of feature `fidx` in row `r`, or `None` when missing.
#[inline]
pub fn feature_value(dmat: &DMatrix, r: usize, fidx: u32) -> Option<f32> {
    let (idx, val) = dmat.row(r);
    match idx.binary_search(&fidx) {
        Ok(k) => Some(val[k]),
        Err(_) => None,
    }
}
