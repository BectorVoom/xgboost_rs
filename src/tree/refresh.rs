//! The `refresh` tree updater.
//!
//! A port of `xgboost::tree::TreeRefresher` (`src/tree/updater_refresh.cc`).
//!
//! `refresh` does not change a tree's *shape*. It re-drops every row of the
//! current matrix down every existing tree, totals the gradients that reach
//! each node, and rewrites what those totals imply:
//!
//! * `sum_hess` and `base_weight` on every node;
//! * `loss_chg` on every internal node, recomputed as
//!   `gain(left) + gain(right) - gain(node)`;
//! * the leaf outputs themselves, when `refresh_leaf` is set — which is the
//!   default, and the reason `refresh` is useful at all: it re-fits an existing
//!   ensemble's leaves to new data without regrowing it.
//!
//! With `refresh_leaf` off the tree's predictions do not move; only the
//! statistics do, which is what a following `prune` pass needs to make a
//! different decision.
//!
//! Monotone constraints are rejected: the per-node weight boxes only exist
//! while a tree is being grown, so a refresh cannot honour them and upstream
//! refuses rather than quietly ignoring them.

use rayon::prelude::*;

use super::model::RegTree;
use super::param::{GradStats, TrainParam, calc_gain, calc_weight};
use crate::data::DMatrix;
use crate::gbm::feature_value;
use crate::objective::GradientPair;

/// Rows per accumulation block. Fixed, so the per-node sums are reduced in the
/// same order however many threads run.
const BLOCK_ROWS: usize = 4096;

/// Recompute `tree`'s statistics — and optionally its leaves — from `gpair`.
pub fn refresh(
    tree: &mut RegTree,
    dmat: &DMatrix,
    gpair: &[GradientPair],
    param: &TrainParam,
    refresh_leaf: bool,
) {
    let n_nodes = tree.num_nodes();
    let n_rows = dmat.num_row();
    if n_nodes == 0 || n_rows == 0 {
        return;
    }

    // Every row adds its gradient to each node on its root-to-leaf path, so a
    // node's total is the sum over the rows that pass through it.
    let blocks = n_rows.div_ceil(BLOCK_ROWS);
    let partials: Vec<Vec<GradStats>> = (0..blocks)
        .into_par_iter()
        .map(|b| {
            let mut acc = vec![GradStats::default(); n_nodes];
            let lo = b * BLOCK_ROWS;
            let hi = ((b + 1) * BLOCK_ROWS).min(n_rows);
            for r in lo..hi {
                let g = gpair[r];
                let (gd, hd) = (g.grad as f64, g.hess as f64);
                let mut nid = 0usize;
                acc[nid].add(gd, hd);
                while !tree.nodes[nid].is_leaf() {
                    let split_index = tree.nodes[nid].split_index;
                    nid = tree.next_node(nid, feature_value(dmat, r, split_index));
                    acc[nid].add(gd, hd);
                }
            }
            acc
        })
        .collect();

    let mut stats = vec![GradStats::default(); n_nodes];
    for partial in &partials {
        for (dst, src) in stats.iter_mut().zip(partial) {
            dst.add_stats(src);
        }
    }

    // Children always have a higher id than their parent, so one forward pass
    // is enough and the recursion upstream uses is unnecessary.
    let gain = |s: &GradStats| calc_gain(param, s);
    for nid in 0..n_nodes {
        let s = stats[nid];
        let weight = calc_weight(param, &s);
        tree.stats[nid].base_weight = weight;
        tree.stats[nid].sum_hess = s.sum_hess as f32;
        if tree.nodes[nid].is_leaf() {
            if refresh_leaf {
                tree.set_leaf(nid, weight * param.learning_rate);
            }
        } else {
            let (l, r) = (tree.nodes[nid].left as usize, tree.nodes[nid].right as usize);
            tree.stats[nid].loss_chg = gain(&stats[l]) + gain(&stats[r]) - gain(&s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::model::NodeStat;

    /// A one-split tree over a single feature, with the split at `0.5`.
    fn stump() -> RegTree {
        let mut t = RegTree::new(1);
        t.expand_node(0, 0, 0.5, false, 0.0, 1.0, -1.0, 7.0, 4.0, 2.0, 2.0);
        t
    }

    fn data(n: usize) -> DMatrix {
        let x: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
        DMatrix::from_dense(&x, n, 1, f32::NAN).unwrap()
    }

    /// Gradients that make the left half want `+1` and the right half `-1`
    /// with `lambda = 0`.
    fn gradients(n: usize) -> Vec<GradientPair> {
        (0..n)
            .map(|i| {
                let left = (i as f32 / n as f32) < 0.5;
                GradientPair { grad: if left { -1.0 } else { 1.0 }, hess: 1.0 }
            })
            .collect()
    }

    fn param() -> TrainParam {
        TrainParam { reg_lambda: 0.0, learning_rate: 1.0, ..Default::default() }
    }

    #[test]
    fn refresh_rewrites_the_leaves_from_the_new_data() {
        let (n, p) = (100, param());
        let mut t = stump();
        refresh(&mut t, &data(n), &gradients(n), &p, true);
        // -G/H per side: 50 rows of -1 gradient on the left, +1 on the right.
        assert!((t.nodes[1].value - 1.0).abs() < 1e-5, "{}", t.nodes[1].value);
        assert!((t.nodes[2].value + 1.0).abs() < 1e-5);
        assert_eq!(t.stats[0].sum_hess, 100.0);
        assert_eq!(t.stats[1].sum_hess, 50.0);
        assert_eq!(t.stats[2].sum_hess, 50.0);
    }

    #[test]
    fn the_learning_rate_scales_the_refreshed_leaf() {
        let (n, p) = (100, TrainParam { learning_rate: 0.5, ..param() });
        let mut t = stump();
        refresh(&mut t, &data(n), &gradients(n), &p, true);
        assert!((t.nodes[1].value - 0.5).abs() < 1e-5);
    }

    #[test]
    fn refresh_leaf_off_moves_the_statistics_and_not_the_predictions() {
        let (n, p) = (100, param());
        let mut t = stump();
        let before: Vec<f32> = t.nodes.iter().map(|node| node.value).collect();
        refresh(&mut t, &data(n), &gradients(n), &p, false);
        let after: Vec<f32> = t.nodes.iter().map(|node| node.value).collect();
        assert_eq!(before, after, "leaf values must not move");
        assert_ne!(t.stats[0], NodeStat { loss_chg: 0.0, sum_hess: 4.0, base_weight: 0.0 });
        assert_eq!(t.stats[0].sum_hess, 100.0, "statistics still refresh");
    }

    #[test]
    fn the_loss_change_is_recomputed_from_the_children() {
        let (n, p) = (100, param());
        let mut t = stump();
        refresh(&mut t, &data(n), &gradients(n), &p, true);
        // gain(left) + gain(right) - gain(root) = 50 + 50 - 0.
        assert!((t.stats[0].loss_chg - 100.0).abs() < 1e-3, "{}", t.stats[0].loss_chg);
        assert_eq!(t.stats[1].loss_chg, 0.0, "a leaf has no split to score");
    }

    #[test]
    fn a_missing_value_follows_the_default_direction() {
        let mut t = stump();
        t.nodes[0].default_left = true;
        let d = DMatrix::from_dense(&[f32::NAN, f32::NAN], 2, 1, f32::NAN).unwrap();
        let gpair = vec![GradientPair { grad: -1.0, hess: 1.0 }; 2];
        refresh(&mut t, &d, &gpair, &param(), true);
        assert_eq!(t.stats[1].sum_hess, 2.0, "both rows went left");
        assert_eq!(t.stats[2].sum_hess, 0.0);
    }

    #[test]
    fn the_result_does_not_depend_on_the_block_split() {
        let p = param();
        let mut small = stump();
        refresh(&mut small, &data(BLOCK_ROWS / 2), &gradients(BLOCK_ROWS / 2), &p, true);
        let mut big = stump();
        refresh(&mut big, &data(BLOCK_ROWS * 3), &gradients(BLOCK_ROWS * 3), &p, true);
        // Same shape of data, different block counts: the leaf values agree.
        assert!((small.nodes[1].value - big.nodes[1].value).abs() < 1e-5);
    }
}
