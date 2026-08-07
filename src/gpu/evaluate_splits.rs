//! On-device split evaluation, ported from
//! `xgboost/src/tree/gpu_hist/evaluate_splits.{cu,cuh}`
//! (`EvaluateSplitsKernel` / `EvaluateFeature` / `DeviceSplitCandidate`).
//!
//! # Why the scan is exact
//!
//! The CPU grower accumulates `f64` gradient sums in bin order, so its prefix
//! sums depend on that order. The device histogram holds *quantised* `i64`
//! bins ([`super::quantiser`]), and integer prefix sums are exact regardless of
//! how the work is split across threads. That is what lets this kernel scan a
//! feature in parallel and still produce, bit for bit, the split the sequential
//! scan would have produced — the same property XGBoost's own gpu_hist relies
//! on. Gains are then computed from the decoded `f64` sums with the `f32`/`f64`
//! mix of `TreeEvaluator::SplitEvaluator`, and — as `EvaluateFeature` does — a
//! sibling is subtracted in fixed point and decoded once, never decoded twice
//! and subtracted in `f64`.
//!
//! One caveat, measured rather than assumed (`tests/gpu_split_eval.rs`): the
//! reference narrows each gain term to `f32` *before* dividing, while the
//! device backend evaluates the expression at `f64` and narrows once. Forcing
//! the narrowing — through a typed function boundary, or a `RuntimeCell` — does
//! not change what the backend emits. The reported `loss_chg` therefore differs
//! from the CPU grower's in the last ulp or two. It is the more accurate of the
//! two, it is far inside the 1e-5 parity target, and it has not moved a split
//! decision: the chosen feature, threshold, default direction and child sums
//! are bit-identical in every case tested.
//!
//! # Candidate ordering
//!
//! `SplitEntry::need_replace` is a total order: prefer the larger `loss_chg`,
//! and on a tie the candidate the sequential scan would have seen first. Within
//! a feature that is "lower rank", where rank counts the forward scan's bins
//! first and the backward scan's after them; across features it is "lower
//! feature index". Both reductions here use that comparator, so the result does
//! not depend on block size or launch geometry.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::error::Result;

/// Threads per split-evaluation workgroup.
pub const EVAL_BLOCK: u32 = 256;

// --------------------------------------------------------- split maths ----
//
// CubeCL will not let a local seeded from a constant be mutated, so every
// accumulator here is written as an `if`/`else` *expression* and every loop
// counter lives in a `RuntimeCell`.

/// `L1(g, a) = sign(g) * max(|g| - a, 0)`; port of `tree::ThresholdL1`.
#[cube]
fn threshold_l1(sum_grad: f64, alpha: f32) -> f64 {
    let a = f64::cast_from(alpha);
    if sum_grad > a {
        sum_grad - a
    } else {
        if sum_grad < -a { sum_grad + a } else { 0.0f64.into() }
    }
}

/// The `max_delta_step` clip of `SplitEvaluator::CalcWeight`.
#[cube]
fn clamp_delta(dw: f64, max_delta_step: f32, #[comptime] has_mds: bool) -> f64 {
    if has_mds {
        let mds = f64::cast_from(max_delta_step);
        let mag = if dw < 0.0 { -dw } else { dw };
        if mag > mds {
            if dw < 0.0 { -mds } else { mds }
        } else {
            dw
        }
    } else {
        dw
    }
}

/// Port of `SplitEvaluator::CalcWeight`: the regularised Newton step, computed
/// in `f64` and narrowed to `f32`.
#[cube]
fn calc_weight(
    g: f64,
    h: f64,
    lambda: f32,
    alpha: f32,
    max_delta_step: f32,
    #[comptime] has_mds: bool,
) -> f32 {
    if h > 0.0 {
        let dw = -threshold_l1(g, alpha) / (h + f64::cast_from(lambda));
        f32::cast_from(clamp_delta(dw, max_delta_step, has_mds))
    } else {
        0.0f32.into()
    }
}

/// `a / b` on operands that are already `f32`.
///
/// `CalcGainGivenWeight` narrows both operands to `f32` *before* dividing, and
/// near-ties depend on it. Written as its own `f32`-typed function so the
/// operands cross a typed boundary: inline, the backend is free to keep them
/// in `f64` and divide at the wider precision, which drifts by an ulp.
#[cube]
fn narrowed_div(a: f32, b: f32) -> f32 {
    a / b
}

/// Port of `tree::CalcGainGivenWeight`. The `max_delta_step == 0` branch
/// narrows to `f32` *before* dividing, which is what keeps the reference's
/// rounding; the other branch is all-`f32`.
#[cube]
fn calc_gain_given_weight(
    g: f64,
    h: f64,
    w: f32,
    lambda: f32,
    alpha: f32,
    #[comptime] has_mds: bool,
) -> f32 {
    if h > 0.0 {
        if has_mds {
            let gf = f32::cast_from(g);
            let hf = f32::cast_from(h);
            let aw = if w < 0.0 { -w } else { w };
            -(2.0f32 * gf * w + (hf + lambda) * w * w + 2.0f32 * alpha * aw)
        } else {
            let num = threshold_l1(g, alpha);
            narrowed_div(f32::cast_from(num * num), f32::cast_from(h + f64::cast_from(lambda)))
        }
    } else {
        0.0f32.into()
    }
}

/// A child's weight, clipped into the node's monotone box
/// (`SplitEvaluator::CalcWeight`'s bounded overload). With no constrained
/// feature the box is the whole line.
#[cube]
#[allow(clippy::too_many_arguments)]
fn child_weight(
    g: f64,
    h: f64,
    lower: f32,
    upper: f32,
    lambda: f32,
    alpha: f32,
    max_delta_step: f32,
    #[comptime] has_constraint: bool,
    #[comptime] has_mds: bool,
) -> f32 {
    let w = calc_weight(g, h, lambda, alpha, max_delta_step, has_mds);
    if has_constraint {
        if w < lower {
            lower
        } else {
            if w > upper { upper } else { w }
        }
    } else {
        w
    }
}

// ------------------------------------------------------- the reduction ----

/// `SplitEntry::need_replace` restricted to one comparison: is `(gain, rank)`
/// a better candidate than `(best_gain, best_rank)`? Larger gain wins; a tie
/// goes to the lower rank, which is the one the sequential scan met first.
#[cube]
fn better(gain: f32, rank: u32, best_gain: f32, best_rank: u32) -> bool {
    gain > best_gain || (gain == best_gain && rank < best_rank)
}

/// Score one candidate split and keep it if it beats this thread's best.
///
/// Folds in `tree::IsValidSplit` and the monotone direction check the way
/// `CalcSplitGain` does — a rejected split simply never reaches [`better`],
/// which is equivalent to upstream returning `-inf` and failing `is_finite`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn consider_split(
    s_gain: &mut SharedMemory<f32>,
    s_rank: &mut SharedMemory<u32>,
    s_bin: &mut SharedMemory<u32>,
    s_dir: &mut SharedMemory<u32>,
    s_left: &mut SharedMemory<i64>,
    lg: f64,
    lh: f64,
    rg: f64,
    rh: f64,
    lg_i: i64,
    lh_i: i64,
    rank: u32,
    bin: u32,
    dir: u32,
    parent_gain: f32,
    lower: f32,
    upper: f32,
    monotone: i32,
    lambda: f32,
    alpha: f32,
    max_delta_step: f32,
    min_child_weight: f32,
    #[comptime] has_constraint: bool,
    #[comptime] has_mds: bool,
) {
    let t = UNIT_POS_X as usize;
    let mcw = f64::cast_from(min_child_weight);

    let wl = child_weight(
        lg, lh, lower, upper, lambda, alpha, max_delta_step, has_constraint, has_mds,
    );
    let wr = child_weight(
        rg, rh, lower, upper, lambda, alpha, max_delta_step, has_constraint, has_mds,
    );

    // `tree::IsValidSplit`: both children must carry hessian, and at least
    // `min_child_weight` of it. Then the monotone direction check.
    let valid = lh > 0.0 && rh > 0.0 && lh >= mcw && rh >= mcw;
    let monotone_ok = if has_constraint {
        (monotone == 0i32) || (monotone > 0i32 && wl <= wr) || (monotone < 0i32 && wl >= wr)
    } else {
        true.into()
    };

    if valid && monotone_ok {
        let gain = calc_gain_given_weight(lg, lh, wl, lambda, alpha, has_mds)
            + calc_gain_given_weight(rg, rh, wr, lambda, alpha, has_mds);
        let chg = gain - parent_gain;
        if better(chg, rank, s_gain[t], s_rank[t]) {
            s_gain[t] = chg;
            s_rank[t] = rank;
            s_bin[t] = bin;
            s_dir[t] = dir;
            s_left[t * 2usize] = lg_i;
            s_left[t * 2usize + 1] = lh_i;
        }
    }
}

/// Inclusive prefix scan of one tile of bins, in shared memory.
///
/// Hillis-Steele over `i64`, so the result is exact and independent of the
/// block size — the property the module docs rest on.
#[cube]
fn scan_tile(s_scan: &mut SharedMemory<i64>, #[comptime] block: usize) {
    let t = UNIT_POS_X as usize;
    let off = RuntimeCell::<u32>::new(1u32);
    while off.read() < block as u32 {
        let d = off.read();
        let ag = if UNIT_POS_X >= d { s_scan[(t - d as usize) * 2usize] } else { 0i64.into() };
        let ah = if UNIT_POS_X >= d { s_scan[(t - d as usize) * 2usize + 1] } else { 0i64.into() };
        sync_cube();
        if UNIT_POS_X >= d {
            s_scan[t * 2usize] += ag;
            s_scan[t * 2usize + 1] += ah;
        }
        sync_cube();
        off.store(d * 2u32);
    }
}

/// Reduce the per-thread bests held in `s_gain` / `s_rank` (and the payload
/// arrays) down to slot 0, using [`better`].
#[cube]
fn reduce_best(
    s_gain: &mut SharedMemory<f32>,
    s_rank: &mut SharedMemory<u32>,
    s_bin: &mut SharedMemory<u32>,
    s_dir: &mut SharedMemory<u32>,
    s_left: &mut SharedMemory<i64>,
    #[comptime] block: usize,
) {
    let stride = RuntimeCell::<u32>::new((block / 2usize) as u32);
    while stride.read() > 0u32 {
        let d = stride.read();
        sync_cube();
        if UNIT_POS_X < d {
            let a = UNIT_POS_X as usize;
            let b = (UNIT_POS_X + d) as usize;
            if better(s_gain[b], s_rank[b], s_gain[a], s_rank[a]) {
                s_gain[a] = s_gain[b];
                s_rank[a] = s_rank[b];
                s_bin[a] = s_bin[b];
                s_dir[a] = s_dir[b];
                s_left[a * 2usize] = s_left[b * 2usize];
                s_left[a * 2usize + 1] = s_left[b * 2usize + 1];
            }
        }
        stride.store(d / 2u32);
    }
    sync_cube();
}

// ----------------------------------------------------------- the kernel ----

/// Evaluate every split of one `(node, feature)` pair.
///
/// Grid is `(n_nodes, n_features)`: `CUBE_POS_X` selects the node, `CUBE_POS_Y`
/// the feature, mirroring the `dim3` launch of `EvaluateSplitsKernel`.
///
/// Both scan directions run, exactly as `EnumerateForward` / `EnumerateBackward`
/// do, and the backward pass is skipped when the feature's bins already account
/// for the node's whole gradient sum — i.e. when the feature has no missing
/// value in this node.
///
/// * `hist` — interleaved `[grad, hess]` `i64` per bin, one block of bins per
///   node; `node_hist_base` gives each node's first bin.
/// * `feature_mask` — per `(node, feature)`, zero for a feature this node may
///   not split on (column sampling and interaction constraints, applied host
///   side exactly as `HistGrower::constraints.query` does).
/// * outputs — one candidate per `(node, feature)`, later reduced by
///   [`reduce_candidates_kernel`]. `out_left` holds the *quantised* left child
///   sum; the right child is the parent minus it.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn evaluate_feature_kernel(
    hist: &Array<i64>,
    cut_ptrs: &Array<u32>,
    cut_values: &Array<f32>,
    min_values: &Array<f32>,
    node_hist_base: &Array<u32>,
    parent_sum: &Array<i64>,
    root_gain: &Array<f32>,
    node_lower: &Array<f32>,
    node_upper: &Array<f32>,
    feature_mask: &Array<u32>,
    monotone: &Array<i32>,
    out_loss_chg: &mut Array<f32>,
    out_sindex: &mut Array<u32>,
    out_split_value: &mut Array<f32>,
    out_left: &mut Array<i64>,
    to_float_grad: f64,
    to_float_hess: f64,
    lambda: f32,
    alpha: f32,
    max_delta_step: f32,
    min_child_weight: f32,
    n_features: u32,
    #[comptime] has_constraint: bool,
    #[comptime] has_mds: bool,
    #[comptime] block: usize,
) {
    let node = CUBE_POS_X;
    let fidx = CUBE_POS_Y;
    let cand = (node * n_features + fidx) as usize;

    let mut s_gain = SharedMemory::<f32>::new(block);
    let mut s_rank = SharedMemory::<u32>::new(block);
    let mut s_bin = SharedMemory::<u32>::new(block);
    let mut s_dir = SharedMemory::<u32>::new(block);
    let mut s_left = SharedMemory::<i64>::new(block * 2usize);
    // Tile scan buffer, plus two slots for the running carry across tiles.
    let mut s_scan = SharedMemory::<i64>::new(block * 2usize);
    let mut s_carry = SharedMemory::<i64>::new(2usize);

    let t = UNIT_POS_X as usize;
    // Seed of `SplitEntry::default()`: gain 0 at rank 0. Rank 0 is what makes
    // ties never replace the seed, so — as upstream — a split has to be a
    // *strict* improvement to be recorded at all.
    s_gain[t] = 0.0f32;
    s_rank[t] = 0u32;
    s_bin[t] = 0u32;
    s_dir[t] = 0u32;
    s_left[t * 2usize] = 0i64;
    s_left[t * 2usize + 1] = 0i64;

    let masked = feature_mask[cand] == 0u32;

    let ibegin = cut_ptrs[fidx as usize];
    let iend = cut_ptrs[(fidx + 1u32) as usize];
    // A masked-out feature scans nothing and so yields the empty candidate.
    // Zeroing the width rather than returning early keeps every thread of the
    // workgroup on the same `sync_cube` sequence.
    let n_bins_feature = if masked { 0u32.into() } else { iend - ibegin };
    let base = node_hist_base[node as usize];

    let pg = parent_sum[(node * 2u32) as usize];
    let ph = parent_sum[(node * 2u32 + 1u32) as usize];
    let parent_gain = root_gain[node as usize];
    let lower = node_lower[node as usize];
    let upper = node_upper[node as usize];
    let mono = monotone[fidx as usize];

    // ---- forward scan: `left_sum` grows over ascending bins ----
    if UNIT_POS_X == 0u32 {
        s_carry[0usize] = 0i64;
        s_carry[1usize] = 0i64;
    }
    sync_cube();

    let tile = RuntimeCell::<u32>::new(0u32);
    while tile.read() < n_bins_feature {
        let i = tile.read() + UNIT_POS_X;
        let live = i < n_bins_feature;
        let cell = (base + ibegin + i) as usize;
        s_scan[t * 2usize] = if live { hist[cell * 2usize] } else { 0i64.into() };
        s_scan[t * 2usize + 1] = if live { hist[cell * 2usize + 1] } else { 0i64.into() };
        sync_cube();

        scan_tile(&mut s_scan, block);

        if live {
            let lg_i = s_carry[0usize] + s_scan[t * 2usize];
            let lh_i = s_carry[1usize] + s_scan[t * 2usize + 1];
            // The sibling is subtracted in *fixed point* and decoded once, as
            // `EvaluateFeature` does — decoding both sides and subtracting in
            // `f64` would round differently.
            let rg_i = pg - lg_i;
            let rh_i = ph - lh_i;
            consider_split(
                &mut s_gain,
                &mut s_rank,
                &mut s_bin,
                &mut s_dir,
                &mut s_left,
                f64::cast_from(lg_i) * to_float_grad,
                f64::cast_from(lh_i) * to_float_hess,
                f64::cast_from(rg_i) * to_float_grad,
                f64::cast_from(rh_i) * to_float_hess,
                lg_i,
                lh_i,
                i,
                ibegin + i,
                0u32,
                parent_gain,
                lower,
                upper,
                mono,
                lambda,
                alpha,
                max_delta_step,
                min_child_weight,
                has_constraint,
                has_mds,
            );
        }

        sync_cube();
        if UNIT_POS_X == block as u32 - 1u32 {
            s_carry[0usize] += s_scan[t * 2usize];
            s_carry[1usize] += s_scan[t * 2usize + 1];
        }
        sync_cube();
        tile.store(tile.read() + block as u32);
    }

    // The feature has missing rows in this node exactly when its bins do not
    // account for the node's whole sum.
    let has_missing = (s_carry[0usize] != pg || s_carry[1usize] != ph) && !masked;

    // ---- backward scan: only needed when there are missing rows ----
    if has_missing {
        if UNIT_POS_X == 0u32 {
            s_carry[0usize] = 0i64;
            s_carry[1usize] = 0i64;
        }
        sync_cube();

        let btile = RuntimeCell::<u32>::new(0u32);
        while btile.read() < n_bins_feature {
            let step = btile.read() + UNIT_POS_X;
            let live = step < n_bins_feature;
            // Descending bin order.
            let i = if live { n_bins_feature - 1u32 - step } else { 0u32.into() };
            let cell = (base + ibegin + i) as usize;
            s_scan[t * 2usize] = if live { hist[cell * 2usize] } else { 0i64.into() };
            s_scan[t * 2usize + 1] = if live { hist[cell * 2usize + 1] } else { 0i64.into() };
            sync_cube();

            scan_tile(&mut s_scan, block);

            if live {
                // Scanning backwards the accumulator is the *right* side.
                let rg_i = s_carry[0usize] + s_scan[t * 2usize];
                let rh_i = s_carry[1usize] + s_scan[t * 2usize + 1];
                let lg_i = pg - rg_i;
                let lh_i = ph - rh_i;
                consider_split(
                    &mut s_gain,
                    &mut s_rank,
                    &mut s_bin,
                    &mut s_dir,
                    &mut s_left,
                    f64::cast_from(lg_i) * to_float_grad,
                    f64::cast_from(lh_i) * to_float_hess,
                    f64::cast_from(rg_i) * to_float_grad,
                    f64::cast_from(rh_i) * to_float_hess,
                    lg_i,
                    lh_i,
                    // Backward candidates rank after every forward one, so a
                    // tie keeps the forward split — `update_entry`'s rule.
                    n_bins_feature + step,
                    ibegin + i,
                    1u32,
                    parent_gain,
                    lower,
                    upper,
                    mono,
                    lambda,
                    alpha,
                    max_delta_step,
                    min_child_weight,
                    has_constraint,
                    has_mds,
                );
            }

            sync_cube();
            if UNIT_POS_X == block as u32 - 1u32 {
                s_carry[0usize] += s_scan[t * 2usize];
                s_carry[1usize] += s_scan[t * 2usize + 1];
            }
            sync_cube();
            btile.store(btile.read() + block as u32);
        }
    }

    reduce_best(&mut s_gain, &mut s_rank, &mut s_bin, &mut s_dir, &mut s_left, block);

    if UNIT_POS_X == 0u32 {
        let bin = s_bin[0usize];
        let default_left = s_dir[0usize] == 1u32;
        // A backward split's threshold is the cut *before* the bin, which is
        // what `Cuts::backward_split_point` returns — the feature's minimum
        // when the bin is the first one.
        let value = if default_left {
            if bin == ibegin {
                min_values[fidx as usize]
            } else {
                cut_values[(bin - 1u32) as usize]
            }
        } else {
            cut_values[bin as usize]
        };

        out_loss_chg[cand] = s_gain[0usize];
        out_sindex[cand] = if default_left { fidx | (1u32 << 31u32) } else { fidx };
        out_split_value[cand] = value;
        out_left[cand * 2usize] = s_left[0usize];
        out_left[cand * 2usize + 1] = s_left[1usize];
    }
}

/// Reduce the per-feature candidates of each node to a single best split.
///
/// One workgroup per node; ties resolve to the lower feature index, matching
/// `SplitEntry::need_replace`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn reduce_candidates_kernel(
    in_loss_chg: &Array<f32>,
    in_sindex: &Array<u32>,
    in_split_value: &Array<f32>,
    in_left: &Array<i64>,
    // Packed so a batch costs two host round trips rather than four: floats
    // in one buffer, integers in the other. Each read back is a full pipeline
    // drain, and a level pays them per launch.
    out_f32: &mut Array<f32>,
    out_i64: &mut Array<i64>,
    n_features: u32,
    #[comptime] block: usize,
) {
    let node = CUBE_POS_X;

    let mut s_gain = SharedMemory::<f32>::new(block);
    let mut s_rank = SharedMemory::<u32>::new(block);
    let mut s_sindex = SharedMemory::<u32>::new(block);
    let mut s_value = SharedMemory::<f32>::new(block);
    let mut s_left = SharedMemory::<i64>::new(block * 2usize);

    let t = UNIT_POS_X as usize;
    // Same seed as the per-feature pass: `SplitEntry::default()` at rank 0, so
    // a zero-gain candidate never displaces "no split".
    s_gain[t] = 0.0f32;
    s_rank[t] = 0u32;
    s_sindex[t] = 0u32;
    s_value[t] = 0.0f32;
    s_left[t * 2usize] = 0i64;
    s_left[t * 2usize + 1] = 0i64;

    let f = RuntimeCell::<u32>::new(UNIT_POS_X);
    while f.read() < n_features {
        let fi = f.read();
        let idx = (node * n_features + fi) as usize;
        let gain = in_loss_chg[idx];
        // Rank by feature index, so a tie keeps the lower one.
        if better(gain, fi, s_gain[t], s_rank[t]) {
            s_gain[t] = gain;
            s_rank[t] = fi;
            s_sindex[t] = in_sindex[idx];
            s_value[t] = in_split_value[idx];
            s_left[t * 2usize] = in_left[idx * 2usize];
            s_left[t * 2usize + 1] = in_left[idx * 2usize + 1];
        }
        f.store(fi + block as u32);
    }

    let stride = RuntimeCell::<u32>::new((block / 2usize) as u32);
    while stride.read() > 0u32 {
        let d = stride.read();
        sync_cube();
        if UNIT_POS_X < d {
            let a = UNIT_POS_X as usize;
            let b = (UNIT_POS_X + d) as usize;
            if better(s_gain[b], s_rank[b], s_gain[a], s_rank[a]) {
                s_gain[a] = s_gain[b];
                s_rank[a] = s_rank[b];
                s_sindex[a] = s_sindex[b];
                s_value[a] = s_value[b];
                s_left[a * 2usize] = s_left[b * 2usize];
                s_left[a * 2usize + 1] = s_left[b * 2usize + 1];
            }
        }
        stride.store(d / 2u32);
    }
    sync_cube();

    if UNIT_POS_X == 0u32 {
        let n = node as usize;
        out_f32[n * 2usize] = s_gain[0usize];
        out_f32[n * 2usize + 1] = s_value[0usize];
        out_i64[n * 3usize] = i64::cast_from(s_sindex[0usize]);
        out_i64[n * 3usize + 1] = s_left[0usize];
        out_i64[n * 3usize + 2] = s_left[1usize];
    }
}

// ------------------------------------------------------------- host API ----

/// One node's chosen split, read back from the device.
///
/// The layout mirrors [`crate::tree::param::SplitEntry`]; child sums stay
/// quantised so the caller can decode them with the same quantiser the
/// histogram used.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DeviceSplitCandidate {
    pub loss_chg: f32,
    /// Feature index with the default-left flag in bit 31.
    pub sindex: u32,
    pub split_value: f32,
    /// Quantised left-child sum; the right child is the parent minus this.
    pub left_grad: i64,
    pub left_hess: i64,
    /// Whether this split names a set of categories rather than a threshold.
    /// Always `false` out of the device kernel itself — it has no categorical
    /// path — and set only by [`super::grower`]'s host-side merge.
    pub is_cat: bool,
}

impl DeviceSplitCandidate {
    pub fn split_index(&self) -> u32 {
        self.sindex & ((1 << 31) - 1)
    }

    pub fn default_left(&self) -> bool {
        (self.sindex >> 31) != 0
    }
}

/// Everything the split kernels need that does not change between nodes.
#[derive(Clone, Debug)]
pub struct SplitConfig {
    pub lambda: f32,
    pub alpha: f32,
    pub max_delta_step: f32,
    pub min_child_weight: f32,
    /// `-1`, `0` or `+1` per feature; all-zero means unconstrained.
    pub monotone: Vec<i32>,
}

impl SplitConfig {
    pub fn has_constraint(&self) -> bool {
        self.monotone.iter().any(|&c| c != 0)
    }
}

/// Per-node inputs: the parent sums, its gain, and its monotone weight box.
#[derive(Clone, Copy, Debug)]
pub struct NodeInput {
    /// Offset of this node's histogram within the `hist` buffer, in bins.
    pub hist_base: u32,
    pub parent_grad: i64,
    pub parent_hess: i64,
    pub root_gain: f32,
    pub lower: f32,
    pub upper: f32,
}

/// Device-resident split evaluator: uploads the cut layout once, then answers
/// any number of node batches.
pub struct SplitEvaluatorGpu<R: Runtime> {
    client: ComputeClient<R>,
    cut_ptrs: Handle,
    cut_values: Handle,
    min_values: Handle,
    monotone: Handle,
    n_features: usize,
    n_bins: usize,
    has_constraint: bool,
    cfg: SplitConfig,
}

impl<R: Runtime> SplitEvaluatorGpu<R> {
    /// `cut_ptrs` has `n_features + 1` entries; `cut_values` one per bin.
    pub fn new(
        client: ComputeClient<R>,
        cut_ptrs: &[u32],
        cut_values: &[f32],
        min_values: &[f32],
        cfg: SplitConfig,
    ) -> Result<Self> {
        let n_features = cut_ptrs.len() - 1;
        let mut monotone = cfg.monotone.clone();
        monotone.resize(n_features, 0);
        let has_constraint = monotone.iter().any(|&c| c != 0);

        let cut_ptrs_dev = client.create_from_slice(bytemuck::cast_slice(cut_ptrs));
        let cut_values_dev = client.create_from_slice(bytemuck::cast_slice(cut_values));
        let min_values_dev = client.create_from_slice(bytemuck::cast_slice(min_values));
        let monotone_dev = client.create_from_slice(bytemuck::cast_slice(&monotone));

        Ok(Self {
            client,
            cut_ptrs: cut_ptrs_dev,
            cut_values: cut_values_dev,
            min_values: min_values_dev,
            monotone: monotone_dev,
            n_features,
            n_bins: cut_values.len(),
            has_constraint,
            cfg,
        })
    }

    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Evaluate one batch of nodes against `hist`, returning the best split per
    /// node in the order the nodes were given.
    ///
    /// `hist` holds interleaved `[grad, hess]` `i64` bins; each node's slice
    /// starts at its own `hist_base`. `feature_mask` is `nodes.len() *
    /// n_features` flags — zero for a feature the node may not split on.
    /// `to_float_*` are the quantiser's factors for the tree being grown; they
    /// change per tree, while everything else here is fixed for the fit, which
    /// is why the evaluator is built once and they are passed per call.
    pub fn evaluate(
        &self,
        hist: &Handle,
        hist_bins: usize,
        nodes: &[NodeInput],
        feature_mask: &[u32],
        to_float_grad: f64,
        to_float_hess: f64,
    ) -> Result<Vec<DeviceSplitCandidate>> {
        let n_nodes = nodes.len();
        if n_nodes == 0 {
            return Ok(Vec::new());
        }
        debug_assert_eq!(feature_mask.len(), n_nodes * self.n_features);

        let c = &self.client;
        let base: Vec<u32> = nodes.iter().map(|n| n.hist_base).collect();
        let parent: Vec<i64> =
            nodes.iter().flat_map(|n| [n.parent_grad, n.parent_hess]).collect();
        let gain: Vec<f32> = nodes.iter().map(|n| n.root_gain).collect();
        let lower: Vec<f32> = nodes.iter().map(|n| n.lower).collect();
        let upper: Vec<f32> = nodes.iter().map(|n| n.upper).collect();

        let base_d = c.create_from_slice(bytemuck::cast_slice(&base));
        let parent_d = c.create_from_slice(bytemuck::cast_slice(&parent));
        let gain_d = c.create_from_slice(bytemuck::cast_slice(&gain));
        let lower_d = c.create_from_slice(bytemuck::cast_slice(&lower));
        let upper_d = c.create_from_slice(bytemuck::cast_slice(&upper));
        let mask_d = c.create_from_slice(bytemuck::cast_slice(feature_mask));

        let n_cand = n_nodes * self.n_features;
        let cand_chg = c.empty(n_cand * size_of::<f32>());
        let cand_sindex = c.empty(n_cand * size_of::<u32>());
        let cand_value = c.empty(n_cand * size_of::<f32>());
        let cand_left = c.empty(n_cand * 2 * size_of::<i64>());

        let has_mds = self.cfg.max_delta_step != 0.0;

        evaluate_feature_kernel::launch::<R>(
            c,
            CubeCount::Static(n_nodes as u32, self.n_features as u32, 1),
            CubeDim::new_1d(EVAL_BLOCK),
            unsafe { ArrayArg::from_raw_parts(hist.clone(), hist_bins * 2) },
            unsafe { ArrayArg::from_raw_parts(self.cut_ptrs.clone(), self.n_features + 1) },
            unsafe { ArrayArg::from_raw_parts(self.cut_values.clone(), self.n_bins) },
            unsafe { ArrayArg::from_raw_parts(self.min_values.clone(), self.n_features) },
            unsafe { ArrayArg::from_raw_parts(base_d, n_nodes) },
            unsafe { ArrayArg::from_raw_parts(parent_d, n_nodes * 2) },
            unsafe { ArrayArg::from_raw_parts(gain_d, n_nodes) },
            unsafe { ArrayArg::from_raw_parts(lower_d, n_nodes) },
            unsafe { ArrayArg::from_raw_parts(upper_d, n_nodes) },
            unsafe { ArrayArg::from_raw_parts(mask_d, n_cand) },
            unsafe { ArrayArg::from_raw_parts(self.monotone.clone(), self.n_features) },
            unsafe { ArrayArg::from_raw_parts(cand_chg.clone(), n_cand) },
            unsafe { ArrayArg::from_raw_parts(cand_sindex.clone(), n_cand) },
            unsafe { ArrayArg::from_raw_parts(cand_value.clone(), n_cand) },
            unsafe { ArrayArg::from_raw_parts(cand_left.clone(), n_cand * 2) },
            to_float_grad,
            to_float_hess,
            self.cfg.lambda,
            self.cfg.alpha,
            self.cfg.max_delta_step,
            self.cfg.min_child_weight,
            self.n_features as u32,
            self.has_constraint,
            has_mds,
            EVAL_BLOCK as usize,
        );

        let best_f32 = c.empty(n_nodes * 2 * size_of::<f32>());
        let best_i64 = c.empty(n_nodes * 3 * size_of::<i64>());

        reduce_candidates_kernel::launch::<R>(
            c,
            CubeCount::Static(n_nodes as u32, 1, 1),
            CubeDim::new_1d(EVAL_BLOCK),
            unsafe { ArrayArg::from_raw_parts(cand_chg, n_cand) },
            unsafe { ArrayArg::from_raw_parts(cand_sindex, n_cand) },
            unsafe { ArrayArg::from_raw_parts(cand_value, n_cand) },
            unsafe { ArrayArg::from_raw_parts(cand_left, n_cand * 2) },
            unsafe { ArrayArg::from_raw_parts(best_f32.clone(), n_nodes * 2) },
            unsafe { ArrayArg::from_raw_parts(best_i64.clone(), n_nodes * 3) },
            self.n_features as u32,
            EVAL_BLOCK as usize,
        );

        let floats: Vec<f32> = read_vec(c, best_f32);
        let ints: Vec<i64> = read_vec(c, best_i64);

        Ok((0..n_nodes)
            .map(|i| DeviceSplitCandidate {
                loss_chg: floats[2 * i],
                split_value: floats[2 * i + 1],
                sindex: ints[3 * i] as u32,
                left_grad: ints[3 * i + 1],
                left_hess: ints[3 * i + 2],
                is_cat: false,
            })
            .collect())
    }
}

fn read_vec<R: Runtime, T: bytemuck::Pod>(client: &ComputeClient<R>, handle: Handle) -> Vec<T> {
    let bytes = client.read_one_unchecked(handle);
    bytemuck::cast_slice(&bytes).to_vec()
}
