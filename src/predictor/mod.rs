//! CPU prediction: tree traversal over a [`DMatrix`].
//!
//! Every entry point is group-aware: a model with `num_output_group > 1`
//! predicts `num_row * num_output_group` values laid out row-major, and each
//! tree adds its leaf value to the column its `tree_info` names.

use crate::data::DMatrix;
use crate::gbm::{GBTreeModel, feature_value};

/// Which trees a prediction uses, as a half-open range into the tree list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TreeRange {
    pub begin: usize,
    pub end: usize,
}

impl TreeRange {
    /// Every tree in the model.
    pub fn all(model: &GBTreeModel) -> Self {
        Self { begin: 0, end: model.num_trees() }
    }

    /// The trees of rounds `[begin, end)`, upstream's `detail::LayerToTree`.
    pub fn from_rounds(model: &GBTreeModel, begin: u32, end: u32) -> Self {
        let per_round = model.trees_per_round();
        Self {
            begin: (begin as usize * per_round).min(model.num_trees()),
            end: (end as usize * per_round).min(model.num_trees()),
        }
    }
}

/// Raw margins: `base_score` (or `base_margin`) plus every tree's leaf value.
///
/// `base_score` holds one margin per output group; a single value is
/// broadcast, which is what a scalar `base_score` parameter means.
pub fn predict_margin(
    model: &GBTreeModel,
    dmat: &DMatrix,
    base_score: &[f32],
    trees: TreeRange,
) -> Vec<f32> {
    let n = dmat.num_row();
    let n_groups = model.num_output_group.max(1);
    let mut preds = init_margin(dmat, base_score, n, n_groups);

    for t in trees.begin..trees.end {
        let tree = &model.trees[t];
        let group = model.group_of(t);
        // DART gives each tree a weight; `gbtree` leaves them all at 1.
        let weight = model.weight_of(t);
        for r in 0..n {
            let leaf = tree.leaf_index(|f| feature_value(dmat, r, f));
            // A vector-leaf tree contributes to every output at once; an
            // ordinary one has a single value and a single group.
            let base = r * n_groups + group;
            for (t_idx, v) in tree.leaf_value(leaf).iter().enumerate() {
                preds[base + t_idx] += weight * v;
            }
        }
    }
    preds
}

/// The margin a fit starts from, before any tree.
pub fn init_margin(dmat: &DMatrix, base_score: &[f32], n_rows: usize, n_groups: usize) -> Vec<f32> {
    if let Some(m) = &dmat.info().base_margin {
        // A supplied margin may be one value per row or one per (row, group).
        if m.len() == n_rows * n_groups {
            return m.clone();
        }
        if m.len() == n_rows {
            let mut out = Vec::with_capacity(n_rows * n_groups);
            for v in m {
                out.extend(std::iter::repeat_n(*v, n_groups));
            }
            return out;
        }
    }
    let fallback = base_score.first().copied().unwrap_or(0.5);
    let mut out = Vec::with_capacity(n_rows * n_groups);
    for _ in 0..n_rows {
        for g in 0..n_groups {
            out.push(base_score.get(g).copied().unwrap_or(fallback));
        }
    }
    out
}

/// Leaf index reached in each tree, one row per input row.
pub fn predict_leaf(model: &GBTreeModel, dmat: &DMatrix, trees: TreeRange) -> Vec<Vec<u32>> {
    (0..dmat.num_row())
        .map(|r| {
            model.trees[trees.begin..trees.end]
                .iter()
                .map(|t| t.leaf_index(|f| feature_value(dmat, r, f)) as u32)
                .collect()
        })
        .collect()
}

/// SHAP feature contributions, `num_feature + 1` values per `(row, group)`;
/// the last column is the bias.
///
/// `approximate` selects the attribution `pred_contribs` computes with
/// `approx_contribs=true`: each split on the path a row takes assigns its whole
/// change in node value to the feature it split on. The exact, path-dependent
/// TreeSHAP is used otherwise.
pub fn predict_contribution(
    model: &GBTreeModel,
    dmat: &DMatrix,
    base_score: &[f32],
    trees: TreeRange,
    approximate: bool,
) -> Vec<f32> {
    let n = dmat.num_row();
    let n_groups = model.num_output_group.max(1);
    let ncol = model.num_feature;
    let width = ncol + 1;
    let mut out = vec![0.0f32; n * n_groups * width];

    // The bias column starts at the intercept, as upstream's does.
    let margins = init_margin(dmat, base_score, n, n_groups);
    for r in 0..n {
        for g in 0..n_groups {
            out[(r * n_groups + g) * width + ncol] = margins[r * n_groups + g];
        }
    }

    for t in trees.begin..trees.end {
        let tree = &model.trees[t];
        let group = model.group_of(t);
        let weight = model.weight_of(t);
        for r in 0..n {
            let row = |f: u32| feature_value(dmat, r, f);
            let base = (r * n_groups + group) * width;
            let slot = &mut out[base..base + width];
            if approximate {
                tree.add_saabas_contributions(&row, weight, slot);
            } else {
                tree.add_shap_contributions(&row, weight, slot);
            }
        }
    }
    out
}

/// SHAP interaction values, `(num_feature + 1)^2` per `(row, group)`.
///
/// The diagonal holds each feature's main effect and the off-diagonal the
/// pairwise interaction, split evenly between the two features so the matrix is
/// symmetric and every row sums back to that feature's total contribution —
/// the invariants `PredictInteractionContributions` guarantees.
pub fn predict_interaction(
    model: &GBTreeModel,
    dmat: &DMatrix,
    base_score: &[f32],
    trees: TreeRange,
    approximate: bool,
) -> Vec<f32> {
    let n = dmat.num_row();
    let n_groups = model.num_output_group.max(1);
    let ncol = model.num_feature;
    let width = ncol + 1;
    let mut out = vec![0.0f32; n * n_groups * width * width];

    // Total contributions, which every row of the matrix must sum back to.
    let total = predict_contribution(model, dmat, base_score, trees, approximate);

    let contributions_without = |r: usize, g: usize, held_out: u32| -> Vec<f32> {
        let mut acc = vec![0.0f32; width];
        for t in trees.begin..trees.end {
            if model.group_of(t) != g {
                continue;
            }
            let tree = &model.trees[t];
            let weight = model.weight_of(t);
            let row = |f: u32| if f == held_out { None } else { feature_value(dmat, r, f) };
            if approximate {
                tree.add_saabas_contributions(&row, weight, &mut acc);
            } else {
                tree.add_shap_contributions(&row, weight, &mut acc);
            }
        }
        acc
    };

    for r in 0..n {
        for g in 0..n_groups {
            let slot = r * n_groups + g;
            let contrib = &total[slot * width..(slot + 1) * width];
            let block = &mut out[slot * width * width..(slot + 1) * width * width];

            for i in 0..ncol {
                let without = contributions_without(r, g, i as u32);
                for j in 0..ncol {
                    if i == j {
                        continue;
                    }
                    // Half the paired difference each way keeps the matrix
                    // symmetric.
                    let interaction = (contrib[j] - without[j]) * 0.5;
                    block[i * width + j] += interaction;
                    block[j * width + i] += interaction;
                }
            }
            // The diagonal absorbs whatever the off-diagonal did not, so each
            // row still sums to the feature's total contribution.
            for i in 0..width {
                let off: f32 = (0..width).filter(|j| *j != i).map(|j| block[i * width + j]).sum();
                block[i * width + i] = contrib[i] - off;
            }
        }
    }
    out
}
