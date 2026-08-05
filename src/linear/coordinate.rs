//! The coordinate-descent arithmetic shared by both linear updaters.
//!
//! A port of `xgboost::linear::coordinate_common.h`. Every sum is accumulated
//! in `f64` and the resulting step narrowed to `f32`, as upstream does.

use crate::data::csc::CscPage;
use crate::objective::GradientPair;

/// `CoordinateDelta`: the elastic-net step for one weight.
///
/// The `-w` clamps are what makes this a *proximal* step: a weight may be
/// driven exactly to zero by the L1 term but never past it, which is what gives
/// `alpha` its sparsifying behaviour.
pub fn coordinate_delta(
    sum_grad: f64,
    sum_hess: f64,
    w: f64,
    reg_alpha: f64,
    reg_lambda: f64,
) -> f64 {
    // Upstream compares against the `float` literal `1e-5f`.
    if sum_hess < 1e-5f32 as f64 {
        return 0.0;
    }
    let sum_grad_l2 = sum_grad + reg_lambda * w;
    let sum_hess_l2 = sum_hess + reg_lambda;
    let tmp = w - sum_grad_l2 / sum_hess_l2;
    if tmp >= 0.0 {
        (-(sum_grad_l2 + reg_alpha) / sum_hess_l2).max(-w)
    } else {
        (-(sum_grad_l2 - reg_alpha) / sum_hess_l2).min(-w)
    }
}

/// `CoordinateDeltaBias`: the unregularised Newton step for the intercept.
///
/// A round with no hessian at all gives `0/0`; upstream maps that (and any
/// infinity) to no movement rather than letting a NaN into the model.
pub fn coordinate_delta_bias(sum_grad: f64, sum_hess: f64) -> f64 {
    let b = -sum_grad / sum_hess;
    if b.is_nan() || b.is_infinite() { 0.0 } else { b }
}

/// Entries reduced per block. Fixed, so a column's sum does not depend on how
/// many threads happened to be free.
const BLOCK: usize = 4096;

/// Sum `(g * x, h * x * x)` over one column, skipping rows whose hessian is
/// negative — upstream's marker for a row the objective has excluded.
pub fn column_gradient(
    page: &CscPage,
    fidx: usize,
    group: usize,
    n_groups: usize,
    gpair: &[GradientPair],
) -> (f64, f64) {
    let (rows, values) = page.column(fidx);
    reduce_blocks(rows.len(), |lo, hi| {
        let mut sum_grad = 0.0f64;
        let mut sum_hess = 0.0f64;
        for k in lo..hi {
            let p = gpair[rows[k] as usize * n_groups + group];
            if p.hess < 0.0 {
                continue;
            }
            let v = values[k] as f64;
            sum_grad += p.grad as f64 * v;
            sum_hess += p.hess as f64 * v * v;
        }
        (sum_grad, sum_hess)
    })
}

/// Sum `(g, h)` over every row of `rows`, which is the intercept's "column".
pub fn bias_gradient(
    rows: std::ops::Range<usize>,
    group: usize,
    n_groups: usize,
    gpair: &[GradientPair],
) -> (f64, f64) {
    let base = rows.start;
    reduce_blocks(rows.len(), |lo, hi| {
        let mut sum_grad = 0.0f64;
        let mut sum_hess = 0.0f64;
        for r in (base + lo)..(base + hi) {
            let p = gpair[r * n_groups + group];
            if p.hess >= 0.0 {
                sum_grad += p.grad as f64;
                sum_hess += p.hess as f64;
            }
        }
        (sum_grad, sum_hess)
    })
}

/// Fold `[0, n)` in fixed blocks, summing partial results in block order.
///
/// Reducing in block order rather than completion order is what keeps the
/// result independent of the thread count, the same rule the histogram lanes
/// follow.
fn reduce_blocks(n: usize, f: impl Fn(usize, usize) -> (f64, f64) + Sync) -> (f64, f64) {
    use rayon::prelude::*;
    if n <= BLOCK {
        return f(0, n);
    }
    let partials: Vec<(f64, f64)> = (0..n.div_ceil(BLOCK))
        .into_par_iter()
        .map(|b| f(b * BLOCK, ((b + 1) * BLOCK).min(n)))
        .collect();
    partials.iter().fold((0.0, 0.0), |acc, p| (acc.0 + p.0, acc.1 + p.1))
}

/// Apply a weight change to the residual gradients of one column.
///
/// `g += h * x * dw` is the first-order correction that lets the next
/// coordinate be chosen against an already-updated model without recomputing
/// every prediction.
pub fn update_residual(
    page: &CscPage,
    fidx: usize,
    group: usize,
    n_groups: usize,
    dw: f32,
    gpair: &mut [GradientPair],
) {
    if dw == 0.0 {
        return;
    }
    let (rows, values) = page.column(fidx);
    for (&r, &v) in rows.iter().zip(values) {
        let p = &mut gpair[r as usize * n_groups + group];
        if p.hess < 0.0 {
            continue;
        }
        p.grad += p.hess * v * dw;
    }
}

/// The same correction for the intercept, which every row sees.
pub fn update_bias_residual(
    rows: std::ops::Range<usize>,
    group: usize,
    n_groups: usize,
    dbias: f32,
    gpair: &mut [GradientPair],
) {
    if dbias == 0.0 {
        return;
    }
    for r in rows {
        let p = &mut gpair[r * n_groups + group];
        if p.hess < 0.0 {
            continue;
        }
        p.grad += p.hess * dbias;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_unregularised_step_is_the_newton_step() {
        // -g/h with no penalties.
        assert_eq!(coordinate_delta(4.0, 2.0, 0.0, 0.0, 0.0), -2.0);
    }

    #[test]
    fn a_column_without_curvature_does_not_move() {
        assert_eq!(coordinate_delta(4.0, 0.0, 1.0, 0.0, 0.0), 0.0);
        assert_eq!(coordinate_delta(4.0, 1e-6, 1.0, 0.0, 0.0), 0.0);
    }

    #[test]
    fn l1_can_zero_a_weight_but_never_overshoot_it() {
        // A weight of 1 with a large alpha lands exactly on zero.
        let dw = coordinate_delta(0.0, 1.0, 1.0, 100.0, 0.0);
        assert_eq!(dw, -1.0, "the step is clamped at -w");

        let dw = coordinate_delta(0.0, 1.0, -1.0, 100.0, 0.0);
        assert_eq!(dw, 1.0);
    }

    #[test]
    fn l2_shrinks_the_step() {
        let plain = coordinate_delta(4.0, 2.0, 1.0, 0.0, 0.0);
        let ridged = coordinate_delta(4.0, 2.0, 1.0, 0.0, 10.0);
        assert!(ridged.abs() < plain.abs() || ridged > plain);
    }

    #[test]
    fn a_bias_without_hessian_stays_put() {
        assert_eq!(coordinate_delta_bias(1.0, 0.0), 0.0);
        assert_eq!(coordinate_delta_bias(0.0, 0.0), 0.0);
        assert_eq!(coordinate_delta_bias(4.0, 2.0), -2.0);
    }

    #[test]
    fn reducing_in_blocks_matches_the_serial_sum() {
        let n = BLOCK * 3 + 7;
        let serial: f64 = (0..n).map(|i| i as f64).sum();
        let (blocked, _) = reduce_blocks(n, |lo, hi| ((lo..hi).map(|i| i as f64).sum(), 0.0));
        assert_eq!(blocked, serial);
    }
}
