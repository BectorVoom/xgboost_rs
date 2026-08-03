//! Gradient boosted model: the tree ensemble and one boosting round.

use crate::data::{DMatrix, cuts::HistogramCuts, gradient_index::GHistIndex};
use crate::objective::GradientPair;
use crate::tree::hist::HistGrower;
use crate::tree::model::RegTree;
use crate::tree::param::TrainParam;

/// The ensemble produced by boosting, upstream's `GBTreeModel`.
#[derive(Clone, Debug, Default)]
pub struct GBTreeModel {
    pub trees: Vec<RegTree>,
    pub num_feature: usize,
}

impl GBTreeModel {
    pub fn new(num_feature: usize) -> Self {
        Self { trees: Vec::new(), num_feature }
    }

    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }
}

/// `gbtree` booster over the CPU `hist` tree method.
pub struct GBTree {
    pub model: GBTreeModel,
    param: TrainParam,
    /// Binned training matrix, built once and reused every round.
    gindex: Option<GHistIndex>,
}

impl GBTree {
    pub fn new(num_feature: usize, param: TrainParam) -> Self {
        Self { model: GBTreeModel::new(num_feature), param, gindex: None }
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

    /// Grow one tree from `gpair` and fold its output into `preds`.
    pub fn do_boost(
        &mut self,
        dtrain: &DMatrix,
        gpair: &[GradientPair],
        preds: &mut [f32],
    ) -> crate::Result<()> {
        self.configure(dtrain)?;
        let gindex = self.gindex.as_ref().expect("configured above");

        let mut tree = RegTree::new(self.model.num_feature);
        let mut grower = HistGrower::new(&self.param, gindex, dtrain);
        grower.grow(gpair, &mut tree);
        // Prediction cache update: the row sets already say which leaf each row
        // reached, so no tree traversal is needed.
        grower.update_predictions(&tree, preds);

        self.model.trees.push(tree);
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
