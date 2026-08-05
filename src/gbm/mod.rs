//! Gradient boosted model: the tree ensemble and one boosting round.

use crate::context::Context;
use crate::data::{DMatrix, cuts::HistogramCuts, gradient_index::GHistIndex};
use crate::objective::GradientPair;
use crate::tree::hist::HistGrower;
use crate::tree::model::RegTree;
use crate::tree::param::TrainParam;
use crate::tree::sampler::RowSampler;

/// The ensemble produced by boosting, upstream's `GBTreeModel`.
#[derive(Clone, Debug, Default)]
pub struct GBTreeModel {
    pub trees: Vec<RegTree>,
    pub num_feature: usize,
    /// Trees grown per boosting round; `> 1` makes each round a small forest.
    pub num_parallel_tree: u32,
}

impl GBTreeModel {
    pub fn new(num_feature: usize) -> Self {
        Self { trees: Vec::new(), num_feature, num_parallel_tree: 1 }
    }

    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

    /// Boosting rounds represented by the ensemble, which is the tree count
    /// only while `num_parallel_tree` is 1.
    pub fn num_rounds(&self) -> usize {
        self.trees.len() / self.num_parallel_tree.max(1) as usize
    }
}

/// `gbtree` booster over the CPU `hist` tree method.
pub struct GBTree {
    pub model: GBTreeModel,
    param: TrainParam,
    /// Binned training matrix, built once and reused every round.
    gindex: Option<GHistIndex>,
    /// Scratch for the sampled gradients of one tree.
    sampled: Vec<GradientPair>,
}

impl GBTree {
    pub fn new(num_feature: usize, param: TrainParam) -> Self {
        let mut model = GBTreeModel::new(num_feature);
        model.num_parallel_tree = param.num_parallel_tree;
        Self { model, param, gindex: None, sampled: Vec::new() }
    }

    pub fn param(&self) -> &TrainParam {
        &self.param
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

    /// Grow this round's trees from `gpair` and fold their output into `preds`.
    ///
    /// With `num_parallel_tree > 1` every tree in the round is grown from the
    /// same gradients but its own row and column samples, which is what turns
    /// the fit into a boosted forest.
    pub fn do_boost(
        &mut self,
        ctx: &mut Context,
        dtrain: &DMatrix,
        gpair: &[GradientPair],
        preds: &mut [f32],
    ) -> crate::Result<()> {
        self.configure(dtrain)?;
        let gindex = self.gindex.as_ref().expect("configured above");

        let sampler = RowSampler::new(self.param.sampling_method, self.param.subsample);
        let is_sampling = sampler.is_sampling(gpair.len());
        let mut grower = HistGrower::new(&self.param, gindex, dtrain);

        for tree_idx in 0..self.param.num_parallel_tree {
            if tree_idx > 0 {
                grower.reset();
            }
            // Sampling zeroes gradients, so it needs a copy the objective's
            // buffer can survive; without it the original is used untouched.
            let gpair = if is_sampling {
                self.sampled.clear();
                self.sampled.extend_from_slice(gpair);
                let seed = ctx.rng().next_u32() as u64;
                sampler.sample(&mut self.sampled, seed, ctx.threads());
                &self.sampled[..]
            } else {
                gpair
            };

            let mut tree = RegTree::new(self.model.num_feature);
            grower.grow(ctx, gpair, &mut tree);
            // Prediction cache update: the row sets already say which leaf each
            // row reached, so no tree traversal is needed.
            grower.update_predictions(&tree, preds);
            self.model.trees.push(tree);
        }
        Ok(())
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
