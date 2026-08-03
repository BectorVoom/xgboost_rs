//! CPU prediction: tree traversal over a [`DMatrix`].

use crate::data::DMatrix;
use crate::gbm::{GBTreeModel, feature_value};

/// Raw margins: `base_score` (or `base_margin`) plus every tree's leaf value.
pub fn predict_margin(model: &GBTreeModel, dmat: &DMatrix, base_score: f32) -> Vec<f32> {
    let n = dmat.num_row();
    let mut preds = match &dmat.info().base_margin {
        Some(m) => m.clone(),
        None => vec![base_score; n],
    };
    for tree in &model.trees {
        for (r, p) in preds.iter_mut().enumerate() {
            let leaf = tree.leaf_index(|f| feature_value(dmat, r, f));
            *p += tree.nodes[leaf].value;
        }
    }
    preds
}

/// Leaf index reached in each tree, one row per input row.
pub fn predict_leaf(model: &GBTreeModel, dmat: &DMatrix) -> Vec<Vec<u32>> {
    (0..dmat.num_row())
        .map(|r| {
            model
                .trees
                .iter()
                .map(|t| t.leaf_index(|f| feature_value(dmat, r, f)) as u32)
                .collect()
        })
        .collect()
}
