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

use super::GradientPairInt64;
use super::launch;
use super::tables::TableBuilder;
use crate::error::Result;

/// Threads per split-evaluation workgroup a device with planes is *asked* for.
///
/// Not the number a launch uses: every launch site runs this through
/// [`launch::scan_block_1d`], which adapts it to the runtime and guarantees the
/// power of two the scan and reduction loops need. A plane-less runtime gets a
/// far smaller block, and gets the same answer — the scans are exact `i64`, so
/// nothing here depends on how the bins were split across units.
pub const EVAL_BLOCK: u32 = 256;

/// `i64` words per node in [`reduce_candidates_kernel`]'s packed output.
const SPLIT_WORDS: usize = 5;

/// Per-unit scratch slots the serial (non-cooperative) shape declares.
///
/// An upper bound on the elementwise cube width, not the width itself: it is a
/// comptime argument, so pinning it keeps the kernel to one compilation even as
/// the frontier changes the launch geometry from level to level. Matches
/// `launch`'s ceiling on a plane-less workgroup.
const SERIAL_SLOTS: usize = 64;

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

/// Port of `SplitEvaluator::CalcGainGivenWeight`.
///
/// The closed form `G²/(H+λ)` is the gain only when `w` really is the
/// unconstrained optimum, so it is used only when nothing can have moved the
/// weight away from it — neither `max_delta_step` nor a monotone box. It is
/// preferred where it applies because it carries less floating point error,
/// and it narrows to `f32` *before* dividing, which is what keeps the
/// reference's rounding. The general branch is all-`f32`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn calc_gain_given_weight(
    g: f64,
    h: f64,
    w: f32,
    lambda: f32,
    alpha: f32,
    #[comptime] has_mds: bool,
    #[comptime] has_constraint: bool,
) -> f32 {
    if h > 0.0 {
        if has_mds || has_constraint {
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

/// Whether a bin holds any row of the node.
///
/// A bin with no rows adds nothing to the running sums, so the split *after*
/// it is the split after the previous bin — same child sums, same gain — at
/// a later rank, and [`better`] keeps the earlier of two equal candidates. So
/// an empty bin's candidate can never win, in either scan direction, and is
/// not scored. The reference scan scores it and rejects it by rank; the
/// result is identical, and at depth, where a node of a few hundred rows
/// spreads over thousands of bins, most of the search is skipped. Both words
/// are tested: a row with zero hessian still carries a gradient.
#[cube]
fn bin_is_live(g: i64, h: i64) -> bool {
    g != 0i64 || h != 0i64
}

/// Score one candidate split: its gain over the parent, or zero when it is
/// not a valid split.
///
/// Folds in `tree::IsValidSplit` and the monotone direction check the way
/// `CalcSplitGain` does, and tests validity *first*: a rejected candidate then
/// costs two comparisons rather than two Newton steps and two gains, and on a
/// small node most of a feature's bins are rejected at one end or the other.
/// Zero is the right answer for "rejected" — the search is seeded at
/// `SplitEntry::default()`, gain 0 at rank 0, and [`better`] accepts only a
/// strict improvement, so a zero never displaces anything. That is what
/// upstream's `-inf` failing `is_finite` amounts to.
#[cube]
#[allow(clippy::too_many_arguments)]
fn split_gain(
    lg: f64,
    lh: f64,
    rg: f64,
    rh: f64,
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
) -> f32 {
    // `tree::IsValidSplit`: both children must carry hessian, and at least
    // `min_child_weight` of it.
    let mcw = f64::cast_from(min_child_weight);
    let valid = lh > 0.0 && rh > 0.0 && lh >= mcw && rh >= mcw;
    if valid {
        if has_constraint || has_mds {
            let wl = child_weight(
                lg, lh, lower, upper, lambda, alpha, max_delta_step, has_constraint, has_mds,
            );
            let wr = child_weight(
                rg, rh, lower, upper, lambda, alpha, max_delta_step, has_constraint, has_mds,
            );
            let monotone_ok = if has_constraint {
                (monotone == 0i32)
                    || (monotone > 0i32 && wl <= wr)
                    || (monotone < 0i32 && wl >= wr)
            } else {
                true.into()
            };
            if monotone_ok {
                let gain = calc_gain_given_weight(
                    lg, lh, wl, lambda, alpha, has_mds, has_constraint,
                ) + calc_gain_given_weight(
                    rg, rh, wr, lambda, alpha, has_mds, has_constraint,
                );
                gain - parent_gain
            } else {
                0.0f32.into()
            }
        } else {
            // Nothing can move a child's weight off the unconstrained optimum,
            // so the gain is the closed form and the weights are never needed.
            // `calc_gain_given_weight` would compute them anyway — two `f64`
            // divisions per candidate that the JIT does not remove as dead,
            // because they sit behind a branch on `h > 0`. Both children are
            // known positive here (`valid`), so this is the same closed form.
            closed_form_gain(lg, lh, lambda, alpha) + closed_form_gain(rg, rh, lambda, alpha)
                - parent_gain
        }
    } else {
        0.0f32.into()
    }
}

/// `CalcGainGivenWeight`'s closed form `L1(G)² / (H + λ)` for a child whose
/// hessian is already known positive: the `h > 0` branch of
/// [`calc_gain_given_weight`] with neither `max_delta_step` nor a monotone
/// box, kept as one expression so both call sites round identically.
#[cube]
fn closed_form_gain(g: f64, h: f64, lambda: f32, alpha: f32) -> f32 {
    let num = threshold_l1(g, alpha);
    narrowed_div(f32::cast_from(num * num), f32::cast_from(h + f64::cast_from(lambda)))
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
/// # Two shapes, one body
///
/// `coop` is the *cooperation width* — how many units share one
/// `(node, feature)` — and it is comptime, so a build only ever emits one of
/// the two:
///
/// * **`coop`** — grid `(n_nodes, n_features)`, `CUBE_POS_X` the node and
///   `CUBE_POS_Y` the feature, mirroring the `dim3` launch of
///   `EvaluateSplitsKernel`. The cube's units split the feature's bins between
///   them, exchange the running sum through `s_scan`/`s_carry`, and finish with
///   a `reduce_best` over `s_gain`. This is what a GPU wants: the work of one
///   pair is only a few hundred bins, so spreading it over a workgroup is the
///   only way to fill the machine.
/// * **not `coop`** — one *unit* owns a whole pair, indexed by `ABSOLUTE_POS`,
///   and walks its bins with an ordinary running accumulator. No scan, no
///   reduction, no `sync_cube`, and the shared arrays degrade to per-unit
///   scratch that no other unit addresses. This is what a runtime whose units
///   are OS threads wants, because there a barrier costs a scheduler quantum
///   (see [`launch::cooperative`]) and a kernel that synchronises cannot be
///   given more than one unit — which leaves it no parallelism at all.
///
/// The two agree bit for bit, and not by luck: every candidate is scored by the
/// same [`consider_split`], off exact `i64` prefix sums, and `better` is a total
/// order on `(gain, rank)`, so the winner does not depend on the order the
/// candidates were visited or on how they were split across units. That is the
/// same property the module docs already rest on for the block size.
///
/// Measured on the CPU runtime, 8 cores, 256 nodes x 64 features x 256 bins:
/// 81.8 ms cooperative, 18.3 ms serial, identical results. In a depth-10 fit
/// the whole `evaluate` call goes 2.63 s -> 1.67 s (`train_bench`, 15 rounds),
/// the rest being the per-call uploads and readbacks the shape does not touch.
///
/// Both scan directions run, exactly as `EnumerateForward` / `EnumerateBackward`
/// do, and the backward pass is skipped when the feature's bins already account
/// for the node's whole gradient sum — i.e. when the feature has no missing
/// value in this node.
///
/// * `hist` — interleaved `[grad, hess]` `i64` per bin, one block of bins per
///   node; `node_hist_base` gives each node's first bin. Two scalar loads,
///   not one `Vector<i64, 2>`: on the CPU runtime the vector view was
///   measured 38% *slower* here (the lane extracts cost more than the load
///   they save at `-O0`), the opposite of the histogram kernel's result.
/// * `feature_mask` — per `(node, feature)`, zero for a feature this node may
///   not split on (column sampling and interaction constraints, applied host
///   side exactly as `HistGrower::constraints.query` does).
/// * outputs — one candidate per `(node, feature)`, later reduced by
///   [`reduce_candidates_kernel`]. `out_left` holds the *quantised* left child
///   sum; the right child is the parent minus it.
#[cube(launch_unchecked)]
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
    n_cands: u32,
    #[comptime] has_constraint: bool,
    #[comptime] has_mds: bool,
    #[comptime] coop: bool,
    #[comptime] block: usize,
) {
    // Cooperation width, comptime. `coop` — a whole cube walks one
    // `(node, feature)`, splitting its bins across the units and exchanging the
    // running sum through shared memory. Otherwise one *unit* owns the whole
    // pair and walks its bins serially, and the launch is elementwise. See the
    // module docs for why the second shape exists.
    let raw_cand = if coop { CUBE_POS_X * n_features + CUBE_POS_Y } else { ABSOLUTE_POS as u32 };
    // An elementwise grid overprovisions and a cooperative one is exact, so
    // only the second shape needs a bound. A dead unit is folded onto candidate
    // 0 rather than branched away, so that every index below stays in range;
    // its result is simply not written out.
    let live_cand = coop || raw_cand < n_cands;
    let cand_ix = if live_cand { raw_cand } else { 0u32.into() };
    let cand = cand_ix as usize;
    // The cooperative grid already carries the two coordinates on its axes;
    // only the flattened one has to divide them back out.
    let node = if coop { CUBE_POS_X } else { cand_ix / n_features };
    let fidx = if coop { CUBE_POS_Y } else { cand_ix - node * n_features };

    let mut s_gain = SharedMemory::<f32>::new(block);
    let mut s_rank = SharedMemory::<u32>::new(block);
    let mut s_bin = SharedMemory::<u32>::new(block);
    let mut s_dir = SharedMemory::<u32>::new(block);
    let mut s_left = SharedMemory::<i64>::new(block * 2usize);
    // Tile scan buffer, plus two slots for the running carry across tiles.
    let mut s_scan = SharedMemory::<i64>::new(block * 2usize);
    let mut s_carry = SharedMemory::<i64>::new(2usize);

    let t = UNIT_POS_X as usize;
    // This unit's running best, in registers; it reaches the shared slot only
    // once the scans are done. Seeded at `SplitEntry::default()`: gain 0 at
    // rank 0, which is what makes ties never replace the seed, so — as
    // upstream — a split has to be a *strict* improvement to be recorded.
    let best_gain = RuntimeCell::<f32>::new(0.0f32);
    let best_rank = RuntimeCell::<u32>::new(0u32);
    let best_bin = RuntimeCell::<u32>::new(0u32);
    let best_dir = RuntimeCell::<u32>::new(0u32);
    let best_lg = RuntimeCell::<i64>::new(0i64);
    let best_lh = RuntimeCell::<i64>::new(0i64);

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
    //
    // Serial when one unit owns the pair: the running sum is an ordinary
    // accumulator, so there is no tile, no shared scan buffer and no barrier.
    // The candidate arithmetic is `consider_split`, identical to the
    // cooperative path below — only the walk differs.
    let fwd_g = RuntimeCell::<i64>::new(0i64);
    let fwd_h = RuntimeCell::<i64>::new(0i64);
    if !coop {
        let i = RuntimeCell::<u32>::new(0u32);
        while i.read() < n_bins_feature {
            let b = i.read();
            let cell = (base + ibegin + b) as usize;
            let bg = hist[cell * 2usize];
            let bh = hist[cell * 2usize + 1];
            let lg_i = fwd_g.read() + bg;
            let lh_i = fwd_h.read() + bh;
            fwd_g.store(lg_i);
            fwd_h.store(lh_i);
            // An empty bin leaves the sums where they were, so its candidate
            // is the previous bin's split at a later rank, and `better` would
            // never take it: skipping it is exact, and at depth most of a
            // node's bins are empty. See `bin_is_live`.
            if bin_is_live(bg, bh) {
                let rg_i = pg - lg_i;
                let rh_i = ph - lh_i;
                let chg = split_gain(
                    f64::cast_from(lg_i) * to_float_grad,
                    f64::cast_from(lh_i) * to_float_hess,
                    f64::cast_from(rg_i) * to_float_grad,
                    f64::cast_from(rh_i) * to_float_hess,
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
                if better(chg, b, best_gain.read(), best_rank.read()) {
                    best_gain.store(chg);
                    best_rank.store(b);
                    best_bin.store(ibegin + b);
                    best_dir.store(0u32);
                    best_lg.store(lg_i);
                    best_lh.store(lh_i);
                }
            }
            i.store(b + 1u32);
        }
    }

    if coop {
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
        let own_g = if live { hist[cell * 2usize] } else { 0i64.into() };
        let own_h = if live { hist[cell * 2usize + 1] } else { 0i64.into() };
        s_scan[t * 2usize] = own_g;
        s_scan[t * 2usize + 1] = own_h;
        sync_cube();

        scan_tile(&mut s_scan, block);

        // An empty bin's prefix is its left neighbour's, whose unit scores
        // the same split at a lower rank; see `bin_is_live`.
        if live && bin_is_live(own_g, own_h) {
            let lg_i = s_carry[0usize] + s_scan[t * 2usize];
            let lh_i = s_carry[1usize] + s_scan[t * 2usize + 1];
            // The sibling is subtracted in *fixed point* and decoded once, as
            // `EvaluateFeature` does — decoding both sides and subtracting in
            // `f64` would round differently.
            let rg_i = pg - lg_i;
            let rh_i = ph - lh_i;
            let chg = split_gain(
                f64::cast_from(lg_i) * to_float_grad,
                f64::cast_from(lh_i) * to_float_hess,
                f64::cast_from(rg_i) * to_float_grad,
                f64::cast_from(rh_i) * to_float_hess,
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
            if better(chg, i, best_gain.read(), best_rank.read()) {
                best_gain.store(chg);
                best_rank.store(i);
                best_bin.store(ibegin + i);
                best_dir.store(0u32);
                best_lg.store(lg_i);
                best_lh.store(lh_i);
            }
        }

        sync_cube();
        if UNIT_POS_X == block as u32 - 1u32 {
            s_carry[0usize] += s_scan[t * 2usize];
            s_carry[1usize] += s_scan[t * 2usize + 1];
        }
        sync_cube();
        tile.store(tile.read() + block as u32);
    }
    }

    // The feature has missing rows in this node exactly when its bins do not
    // account for the node's whole sum. Each walk left that total where it
    // accumulated it — the shared carry, or the serial accumulator.
    let tot_g = if coop { s_carry[0usize] } else { fwd_g.read() };
    let tot_h = if coop { s_carry[1usize] } else { fwd_h.read() };
    let has_missing = (tot_g != pg || tot_h != ph) && !masked;

    // ---- backward scan: only needed when there are missing rows ----
    if has_missing && !coop {
        let bwd_g = RuntimeCell::<i64>::new(0i64);
        let bwd_h = RuntimeCell::<i64>::new(0i64);
        let s = RuntimeCell::<u32>::new(0u32);
        while s.read() < n_bins_feature {
            let step = s.read();
            // Descending bin order; the accumulator is the *right* side.
            let b = n_bins_feature - 1u32 - step;
            let cell = (base + ibegin + b) as usize;
            let bg = hist[cell * 2usize];
            let bh = hist[cell * 2usize + 1];
            let rg_i = bwd_g.read() + bg;
            let rh_i = bwd_h.read() + bh;
            bwd_g.store(rg_i);
            bwd_h.store(rh_i);
            if bin_is_live(bg, bh) {
                let lg_i = pg - rg_i;
                let lh_i = ph - rh_i;
                let chg = split_gain(
                    f64::cast_from(lg_i) * to_float_grad,
                    f64::cast_from(lh_i) * to_float_hess,
                    f64::cast_from(rg_i) * to_float_grad,
                    f64::cast_from(rh_i) * to_float_hess,
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
                // Backward candidates rank after every forward one, so a tie
                // keeps the forward split — `update_entry`'s rule.
                let rank = n_bins_feature + step;
                if better(chg, rank, best_gain.read(), best_rank.read()) {
                    best_gain.store(chg);
                    best_rank.store(rank);
                    best_bin.store(ibegin + b);
                    best_dir.store(1u32);
                    best_lg.store(lg_i);
                    best_lh.store(lh_i);
                }
            }
            s.store(step + 1u32);
        }
    }

    if has_missing && coop {
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
            let own_g = if live { hist[cell * 2usize] } else { 0i64.into() };
            let own_h = if live { hist[cell * 2usize + 1] } else { 0i64.into() };
            s_scan[t * 2usize] = own_g;
            s_scan[t * 2usize + 1] = own_h;
            sync_cube();

            scan_tile(&mut s_scan, block);

            if live && bin_is_live(own_g, own_h) {
                // Scanning backwards the accumulator is the *right* side.
                let rg_i = s_carry[0usize] + s_scan[t * 2usize];
                let rh_i = s_carry[1usize] + s_scan[t * 2usize + 1];
                let lg_i = pg - rg_i;
                let lh_i = ph - rh_i;
                let chg = split_gain(
                    f64::cast_from(lg_i) * to_float_grad,
                    f64::cast_from(lh_i) * to_float_hess,
                    f64::cast_from(rg_i) * to_float_grad,
                    f64::cast_from(rh_i) * to_float_hess,
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
                // Backward candidates rank after every forward one, so a
                // tie keeps the forward split — `update_entry`'s rule.
                let rank = n_bins_feature + step;
                if better(chg, rank, best_gain.read(), best_rank.read()) {
                    best_gain.store(chg);
                    best_rank.store(rank);
                    best_bin.store(ibegin + i);
                    best_dir.store(1u32);
                    best_lg.store(lg_i);
                    best_lh.store(lh_i);
                }
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

    // The unit's best reaches its shared slot once, here.
    s_gain[t] = best_gain.read();
    s_rank[t] = best_rank.read();
    s_bin[t] = best_bin.read();
    s_dir[t] = best_dir.read();
    s_left[t * 2usize] = best_lg.read();
    s_left[t * 2usize + 1] = best_lh.read();

    // Only the cooperative walk spread the candidates across units, so only it
    // has anything to reduce.
    if coop {
        reduce_best(&mut s_gain, &mut s_rank, &mut s_bin, &mut s_dir, &mut s_left, block);
    }

    // The winner is in slot 0 after that reduction; without it, each unit's own
    // slot already holds its own candidate's winner.
    let slot = (if coop { 0u32.into() } else { UNIT_POS_X }) as usize;
    let writes = if coop { UNIT_POS_X == 0u32 } else { live_cand };
    if writes {
        let bin = s_bin[slot];
        let default_left = s_dir[slot] == 1u32;
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

        out_loss_chg[cand] = s_gain[slot];
        out_sindex[cand] = if default_left { fidx | (1u32 << 31u32) } else { fidx };
        out_split_value[cand] = value;
        out_left[cand * 2usize] = s_left[slot * 2usize];
        out_left[cand * 2usize + 1] = s_left[slot * 2usize + 1];
    }

    // Hold the whole cube here while unit 0 reads slot 0. On a runtime that
    // runs cubes sequentially out of one shared buffer (see the module docs),
    // the other units would otherwise loop straight on to the next
    // `(node, feature)` and re-seed `s_gain`/`s_carry` underneath this read.
    //
    // The serial walk needs none of it: every unit only ever touches its own
    // slot, so nothing it writes is another unit's to read, in this cube or the
    // next. That is the whole point of the shape — no barrier, so the launch
    // can use every core.
    if coop {
        sync_cube();
    }
}

/// Reduce the per-feature candidates of each node to a single best split.
///
/// Ties resolve to the lower feature index, matching `SplitEntry::need_replace`.
/// Two shapes, as [`evaluate_feature_kernel`]: cooperative, a cube per node
/// whose units stride the features and tree-reduce their bests; serial, a
/// unit per node walking every feature, with no barrier.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn reduce_candidates_kernel(
    in_loss_chg: &Array<f32>,
    in_sindex: &Array<u32>,
    in_split_value: &Array<f32>,
    in_left: &Array<i64>,
    // Packed so a batch costs one host round trip rather than four: five
    // `i64` per node, `[sindex, left_grad, left_hess, gain_bits, value_bits]`,
    // the two floats carried as their `u32` bit patterns. Each read back is a
    // full pipeline drain — on the CPU runtime, a thread hand-off — and a
    // level pays it once.
    out: &mut Array<i64>,
    n_features: u32,
    n_nodes: u32,
    #[comptime] coop: bool,
    #[comptime] block: usize,
) {
    // The cooperative grid is exact; the elementwise one overprovisions, and a
    // dead unit is folded onto node 0 so every index stays in range.
    let raw_node = if coop { CUBE_POS_X } else { ABSOLUTE_POS as u32 };
    let live = coop || raw_node < n_nodes;
    let node = if live { raw_node } else { 0u32.into() };

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

    // The cube's units share a node's features in the cooperative shape; a
    // unit takes them all in the serial one.
    let first = if coop { UNIT_POS_X } else { 0u32.into() };
    let stride = if coop { CUBE_DIM_X } else { 1u32.into() };
    let f = RuntimeCell::<u32>::new(first);
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
        f.store(fi + stride);
    }

    if coop {
        let half = RuntimeCell::<u32>::new((block / 2usize) as u32);
        while half.read() > 0u32 {
            let d = half.read();
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
            half.store(d / 2u32);
        }
        sync_cube();
    }

    // The winner is in slot 0 after the reduction; without one, each unit's
    // own slot holds its node's winner.
    let slot = (if coop { 0u32.into() } else { UNIT_POS_X }) as usize;
    let writes = if coop { UNIT_POS_X == 0u32 } else { live };
    if writes {
        let n = node as usize;
        out[n * SPLIT_WORDS] = i64::cast_from(s_sindex[slot]);
        out[n * SPLIT_WORDS + 1usize] = s_left[slot * 2usize];
        out[n * SPLIT_WORDS + 2usize] = s_left[slot * 2usize + 1];
        out[n * SPLIT_WORDS + 3usize] = i64::cast_from(u32::reinterpret(s_gain[slot]));
        out[n * SPLIT_WORDS + 4usize] = i64::cast_from(u32::reinterpret(s_value[slot]));
    }

    // As in `evaluate_feature_kernel`: the next cube must not re-seed the
    // shared slots while unit 0 is still reading this cube's winner. The
    // serial shape shares nothing between units.
    if coop {
        sync_cube();
    }
}

// ------------------------------------------------- vector-leaf kernels ----
//
// `multi_strategy=multi_output_tree` grows one tree whose leaves carry a value
// per target. The split is a single decision shared by every target, scored
// from one histogram per `(node, target)` — the shape `HistGrower` uses on the
// CPU, where the scalar accumulation kernel is simply run once per target.
//
// The scalar evaluator scans a feature and accumulates as it goes, so its
// running sum is the candidate. That does not carry over: a vector-leaf
// candidate needs `n_targets` running sums at once, and holding them in shared
// memory would blow the 16 KiB workgroup budget the portable backends allow.
// So the scan is split out into [`prefix_scan_kernel`], which materialises the
// inclusive prefix sums once per `(node, feature, target)`, and
// [`evaluate_feature_multi_kernel`] then reads whichever sums a candidate bin
// needs. The prefix sums are exact `i64`, so — exactly as for the scalar path —
// the result does not depend on how the work was split across threads.
//
// The backward scan needs no second pass: the suffix sum at bin `i` is
// `total - prefix[i - 1]`, in fixed point, which is bit-identical to
// accumulating it from the top.

/// Inclusive prefix sums over each `(node, feature, target)` run of bins.
///
/// `out` has the same layout as `hist`: bin `b` of target `t` of the node at
/// `node_hist_base[node]` lives at `node_hist_base[node] + t * node_bins + b`.
///
/// Two shapes: cooperative, grid `(n_nodes, n_features, n_targets)` with a
/// cube scanning one run tile by tile through shared memory; serial, a unit
/// per run indexed by `ABSOLUTE_POS` over `n_lanes = n_nodes * n_features *
/// n_targets`, walking the bins with a running sum. Exact `i64` either way.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
// The outer `if !coop` is a comptime branch and the inner one a runtime one;
// they cannot be joined with `&&` across that boundary.
#[allow(clippy::collapsible_if)]
pub fn prefix_scan_kernel(
    hist: &Array<i64>,
    cut_ptrs: &Array<u32>,
    node_hist_base: &Array<u32>,
    out: &mut Array<i64>,
    node_bins: u32,
    n_features: u32,
    n_targets: u32,
    n_lanes: u32,
    #[comptime] coop: bool,
    #[comptime] block: usize,
) {
    let lane = ABSOLUTE_POS as u32;
    let live_lane = coop || lane < n_lanes;
    let lane_ix = if live_lane { lane } else { 0u32.into() };
    let lane_node = lane_ix / (n_features * n_targets);
    let lane_rest = lane_ix - lane_node * (n_features * n_targets);
    let node = if coop { CUBE_POS_X } else { lane_node };
    let fidx = if coop { CUBE_POS_Y } else { lane_rest / n_targets };
    let target = if coop { CUBE_POS_Z } else { lane_rest - (lane_rest / n_targets) * n_targets };

    let mut s_scan = SharedMemory::<i64>::new(block * 2usize);
    let mut s_carry = SharedMemory::<i64>::new(2usize);

    let t = UNIT_POS_X as usize;
    let ibegin = cut_ptrs[fidx as usize];
    let n_bins_feature = cut_ptrs[(fidx + 1u32) as usize] - ibegin;
    let base = node_hist_base[node as usize] + target * node_bins;

    if !coop {
        if live_lane {
            let g = RuntimeCell::<i64>::new(0i64);
            let h = RuntimeCell::<i64>::new(0i64);
            let i = RuntimeCell::<u32>::new(0u32);
            while i.read() < n_bins_feature {
                let cell = (base + ibegin + i.read()) as usize;
                g.store(g.read() + hist[cell * 2usize]);
                h.store(h.read() + hist[cell * 2usize + 1]);
                out[cell * 2usize] = g.read();
                out[cell * 2usize + 1] = h.read();
                i.store(i.read() + 1u32);
            }
        }
    }

    if coop {
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
                out[cell * 2usize] = s_carry[0usize] + s_scan[t * 2usize];
                out[cell * 2usize + 1] = s_carry[1usize] + s_scan[t * 2usize + 1];
            }

            sync_cube();
            if UNIT_POS_X == block as u32 - 1u32 {
                s_carry[0usize] += s_scan[t * 2usize];
                s_carry[1usize] += s_scan[t * 2usize + 1];
            }
            sync_cube();
            tile.store(tile.read() + block as u32);
        }

        // The tile loop runs a different number of times per `(node, feature,
        // target)`, so units leave it at different moments. Without this, a
        // unit whose feature had fewer bins would reach the next cube's
        // `s_carry` reset while a slower one was still scanning out of the
        // same buffer.
        sync_cube();
    }
}

/// Score one vector-leaf candidate split and keep it if it beats this thread's
/// best; the multi-target [`consider_split`].
///
/// Port of `HistGrower::multi_split_gain`: every target contributes its own
/// regularised gain at its own bounded weight, and the whole candidate is
/// rejected when the children's *mean* hessian fails `tree::IsValidSplit` —
/// the split is one decision for all the outputs, so `min_child_weight` is a
/// statement about the node rather than about any single target.
///
/// The monotone *direction* check that `CalcSplitGain` folds in is absent for
/// the same reason it is a no-op on the CPU: it is applied to the mean-hessian
/// children, whose gradient sum is zero, so both weights are `CalcWeight(0, h)`
/// clipped into the node's box — always equal, and so always passing whichever
/// direction is asked for. The bounds still shape the *gain*, through
/// [`child_weight`] on each target's real sums.
#[cube]
#[allow(clippy::too_many_arguments)]
fn consider_split_multi(
    s_gain: &mut SharedMemory<f32>,
    s_rank: &mut SharedMemory<u32>,
    s_bin: &mut SharedMemory<u32>,
    s_dir: &mut SharedMemory<u32>,
    s_left: &mut SharedMemory<i64>,
    prefix: &Array<i64>,
    parent_sum: &Array<i64>,
    node: u32,
    node_base: u32,
    node_bins: u32,
    n_targets: u32,
    ibegin: u32,
    n_bins_feature: u32,
    bin_local: u32,
    dir: u32,
    rank: u32,
    parent_gain: f32,
    lower: f32,
    upper: f32,
    to_float_grad: f64,
    to_float_hess: f64,
    lambda: f32,
    alpha: f32,
    max_delta_step: f32,
    min_child_weight: f32,
    #[comptime] has_constraint: bool,
    #[comptime] has_mds: bool,
) {
    let t = UNIT_POS_X as usize;
    let backward = dir == 1u32;

    let sum_lg = RuntimeCell::<i64>::new(0i64);
    let sum_lh = RuntimeCell::<i64>::new(0i64);
    let hess_l = RuntimeCell::<f64>::new(0.0f64);
    let hess_r = RuntimeCell::<f64>::new(0.0f64);
    let gain = RuntimeCell::<f32>::new(0.0f32);

    let tt = RuntimeCell::<u32>::new(0u32);
    while tt.read() < n_targets {
        let target = tt.read();
        let tbase = node_base + target * node_bins;
        let psum = ((node * n_targets + target) * 2u32) as usize;
        let pg = parent_sum[psum];
        let ph = parent_sum[psum + 1];

        let cell = (tbase + ibegin + bin_local) as usize;
        let last = (tbase + ibegin + n_bins_feature - 1u32) as usize;
        // Bin `bin_local - 1`, or the bin itself when there is no predecessor;
        // the value is then discarded, so it only has to stay in bounds.
        let head = if bin_local == 0u32 { cell } else { cell - 1usize };
        let head_g = if bin_local == 0u32 { 0i64.into() } else { prefix[head * 2usize] };
        let head_h = if bin_local == 0u32 { 0i64.into() } else { prefix[head * 2usize + 1] };

        // Forward, the accumulator is the left child. Backward, it is the right
        // one: the suffix sum `total - prefix[bin - 1]`, subtracted from the
        // parent in fixed point exactly as `EvaluateFeature` does.
        let lg_i = if backward {
            pg - (prefix[last * 2usize] - head_g)
        } else {
            prefix[cell * 2usize]
        };
        let lh_i = if backward {
            ph - (prefix[last * 2usize + 1] - head_h)
        } else {
            prefix[cell * 2usize + 1]
        };
        let rg_i = pg - lg_i;
        let rh_i = ph - lh_i;

        sum_lg.store(sum_lg.read() + lg_i);
        sum_lh.store(sum_lh.read() + lh_i);

        let lgf = f64::cast_from(lg_i) * to_float_grad;
        let lhf = f64::cast_from(lh_i) * to_float_hess;
        let rgf = f64::cast_from(rg_i) * to_float_grad;
        let rhf = f64::cast_from(rh_i) * to_float_hess;
        hess_l.store(hess_l.read() + lhf);
        hess_r.store(hess_r.read() + rhf);

        let wl = child_weight(
            lgf, lhf, lower, upper, lambda, alpha, max_delta_step, has_constraint, has_mds,
        );
        let wr = child_weight(
            rgf, rhf, lower, upper, lambda, alpha, max_delta_step, has_constraint, has_mds,
        );
        // One `f32` accumulator, left term then right term per target, which is
        // the order `multi_split_gain` sums them in.
        gain.store(
            gain.read()
                + calc_gain_given_weight(lgf, lhf, wl, lambda, alpha, has_mds, has_constraint)
                + calc_gain_given_weight(rgf, rhf, wr, lambda, alpha, has_mds, has_constraint),
        );

        tt.store(target + 1u32);
    }

    let k = f64::cast_from(n_targets);
    let mean_l = hess_l.read() / k;
    let mean_r = hess_r.read() / k;
    let mcw = f64::cast_from(min_child_weight);
    if mean_l > 0.0 && mean_r > 0.0 && mean_l >= mcw && mean_r >= mcw {
        let chg = gain.read() - parent_gain;
        if better(chg, rank, s_gain[t], s_rank[t]) {
            s_gain[t] = chg;
            s_rank[t] = rank;
            s_bin[t] = ibegin + bin_local;
            s_dir[t] = dir;
            s_left[t * 2usize] = sum_lg.read();
            s_left[t * 2usize + 1] = sum_lh.read();
        }
    }
}

/// [`evaluate_feature_kernel`] for a vector-leaf tree.
///
/// Same grid, same candidate ordering, same outputs — `out_left` holds the
/// left child's sum *added over the targets*, which is what
/// [`crate::tree::param::SplitEntry`] carries on the CPU too. The per-target
/// child sums the applied split needs are recovered afterwards by
/// [`multi_child_sums_kernel`], for the same reason `multi_child_sums` exists:
/// a per-target vector cannot ride along on a candidate that is copied by the
/// thousand.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn evaluate_feature_multi_kernel(
    prefix: &Array<i64>,
    cut_ptrs: &Array<u32>,
    cut_values: &Array<f32>,
    min_values: &Array<f32>,
    node_hist_base: &Array<u32>,
    parent_sum: &Array<i64>,
    root_gain: &Array<f32>,
    node_lower: &Array<f32>,
    node_upper: &Array<f32>,
    feature_mask: &Array<u32>,
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
    n_targets: u32,
    node_bins: u32,
    n_cands: u32,
    #[comptime] has_constraint: bool,
    #[comptime] has_mds: bool,
    #[comptime] coop: bool,
    #[comptime] block: usize,
) {
    // Cooperation width, exactly as `evaluate_feature_kernel` states it: a
    // cube per `(node, feature)`, or a unit per pair on an elementwise grid
    // with a dead unit folded onto candidate 0.
    let raw_cand = if coop { CUBE_POS_X * n_features + CUBE_POS_Y } else { ABSOLUTE_POS as u32 };
    let live_cand = coop || raw_cand < n_cands;
    let cand_ix = if live_cand { raw_cand } else { 0u32.into() };
    let cand = cand_ix as usize;
    let node = if coop { CUBE_POS_X } else { cand_ix / n_features };
    let fidx = if coop { CUBE_POS_Y } else { cand_ix - node * n_features };

    let mut s_gain = SharedMemory::<f32>::new(block);
    let mut s_rank = SharedMemory::<u32>::new(block);
    let mut s_bin = SharedMemory::<u32>::new(block);
    let mut s_dir = SharedMemory::<u32>::new(block);
    let mut s_left = SharedMemory::<i64>::new(block * 2usize);

    let t = UNIT_POS_X as usize;
    s_gain[t] = 0.0f32;
    s_rank[t] = 0u32;
    s_bin[t] = 0u32;
    s_dir[t] = 0u32;
    s_left[t * 2usize] = 0i64;
    s_left[t * 2usize + 1] = 0i64;

    let masked = feature_mask[cand] == 0u32;
    let ibegin = cut_ptrs[fidx as usize];
    let iend = cut_ptrs[(fidx + 1u32) as usize];
    let n_bins_feature = if masked { 0u32.into() } else { iend - ibegin };
    let node_base = node_hist_base[node as usize];
    let parent_gain = root_gain[node as usize];
    let lower = node_lower[node as usize];
    let upper = node_upper[node as usize];

    // ---- forward: the head of the feature's bins goes left ----
    //
    // No `sync_cube` inside either scan — the prefix sums are already on
    // device — so the threads may stride through the bins independently. In
    // the serial shape the one unit takes every bin.
    let first = if coop { UNIT_POS_X } else { 0u32.into() };
    let stride = if coop { CUBE_DIM_X } else { 1u32.into() };
    let step = RuntimeCell::<u32>::new(first);
    while step.read() < n_bins_feature {
        let i = step.read();
        consider_split_multi(
            &mut s_gain,
            &mut s_rank,
            &mut s_bin,
            &mut s_dir,
            &mut s_left,
            prefix,
            parent_sum,
            node,
            node_base,
            node_bins,
            n_targets,
            ibegin,
            n_bins_feature,
            i,
            0u32,
            i,
            parent_gain,
            lower,
            upper,
            to_float_grad,
            to_float_hess,
            lambda,
            alpha,
            max_delta_step,
            min_child_weight,
            has_constraint,
            has_mds,
        );
        step.store(i + stride);
    }

    // The feature has missing rows in this node exactly when its bins do not
    // account for the node's whole sum — stated over the targets together, as
    // `evaluate_one_multi` states it.
    let tot_g = RuntimeCell::<i64>::new(0i64);
    let tot_h = RuntimeCell::<i64>::new(0i64);
    let par_g = RuntimeCell::<i64>::new(0i64);
    let par_h = RuntimeCell::<i64>::new(0i64);
    let tt = RuntimeCell::<u32>::new(0u32);
    while tt.read() < n_targets {
        let target = tt.read();
        if n_bins_feature > 0u32 {
            let last = (node_base + target * node_bins + ibegin + n_bins_feature - 1u32) as usize;
            tot_g.store(tot_g.read() + prefix[last * 2usize]);
            tot_h.store(tot_h.read() + prefix[last * 2usize + 1]);
        }
        let psum = ((node * n_targets + target) * 2u32) as usize;
        par_g.store(par_g.read() + parent_sum[psum]);
        par_h.store(par_h.read() + parent_sum[psum + 1]);
        tt.store(target + 1u32);
    }
    let has_missing = (tot_g.read() != par_g.read() || tot_h.read() != par_h.read()) && !masked;

    // ---- backward: only needed when there are missing rows ----
    if has_missing {
        let bstep = RuntimeCell::<u32>::new(first);
        while bstep.read() < n_bins_feature {
            let s = bstep.read();
            consider_split_multi(
                &mut s_gain,
                &mut s_rank,
                &mut s_bin,
                &mut s_dir,
                &mut s_left,
                prefix,
                parent_sum,
                node,
                node_base,
                node_bins,
                n_targets,
                ibegin,
                n_bins_feature,
                // Descending bin order; backward candidates rank after every
                // forward one, so a tie keeps the forward split.
                n_bins_feature - 1u32 - s,
                1u32,
                n_bins_feature + s,
                parent_gain,
                lower,
                upper,
                to_float_grad,
                to_float_hess,
                lambda,
                alpha,
                max_delta_step,
                min_child_weight,
                has_constraint,
                has_mds,
            );
            bstep.store(s + stride);
        }
    }

    // Only the cooperative walk spread the candidates across units, so only it
    // has anything to reduce; see `evaluate_feature_kernel`.
    if coop {
        reduce_best(&mut s_gain, &mut s_rank, &mut s_bin, &mut s_dir, &mut s_left, block);
    }

    let slot = (if coop { 0u32.into() } else { UNIT_POS_X }) as usize;
    let writes = if coop { UNIT_POS_X == 0u32 } else { live_cand };
    if writes {
        let bin = s_bin[slot];
        let default_left = s_dir[slot] == 1u32;
        let value = if default_left {
            if bin == ibegin {
                min_values[fidx as usize]
            } else {
                cut_values[(bin - 1u32) as usize]
            }
        } else {
            cut_values[bin as usize]
        };

        out_loss_chg[cand] = s_gain[slot];
        out_sindex[cand] = if default_left { fidx | (1u32 << 31u32) } else { fidx };
        out_split_value[cand] = value;
        out_left[cand * 2usize] = s_left[slot * 2usize];
        out_left[cand * 2usize + 1] = s_left[slot * 2usize + 1];
    }

    // As in `evaluate_feature_kernel`: the next cube must not re-seed the
    // shared slots while unit 0 is still reading this cube's winner.
    if coop {
        sync_cube();
    }
}

/// Recover a chosen split's per-target child sums from the prefix sums.
///
/// Port of `HistGrower::multi_child_sums`, down to how `node_cond` is read:
/// `-1` (the threshold matched no cut) means the whole feature run is the
/// accumulated side, which leaves the other child holding only the rows with
/// no value for the feature.
///
/// One thread per `(node, target)`; `out` holds `[left, right]` interleaved
/// `[grad, hess]` per pair, so four `i64` each.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn multi_child_sums_kernel(
    prefix: &Array<i64>,
    cut_ptrs: &Array<u32>,
    node_hist_base: &Array<u32>,
    parent_sum: &Array<i64>,
    node_fidx: &Array<u32>,
    node_cond: &Array<i32>,
    node_default_left: &Array<u32>,
    out: &mut Array<i64>,
    n_targets: u32,
    node_bins: u32,
    n: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < n {
        let node = idx / n_targets;
        let target = idx % n_targets;
        let fidx = node_fidx[node as usize];
        let ibegin = cut_ptrs[fidx as usize];
        let iend = cut_ptrs[(fidx + 1u32) as usize];
        let tbase = node_hist_base[node as usize] + target * node_bins;

        let last = (tbase + iend - 1u32) as usize;
        let total_g = prefix[last * 2usize];
        let total_h = prefix[last * 2usize + 1];

        let cond = node_cond[node as usize];
        let below = cond < i32::cast_from(ibegin);
        // Clamped so both arms of the reads below stay in bounds; the value is
        // discarded when `below`.
        let at = (tbase + if below { ibegin } else { u32::cast_from(cond) }) as usize;
        let pfx_g = if below { 0i64.into() } else { prefix[at * 2usize] };
        let pfx_h = if below { 0i64.into() } else { prefix[at * 2usize + 1] };

        let dl = node_default_left[node as usize] == 1u32;
        // `default_left` accumulates the bins *above* the threshold — the right
        // child; otherwise the run at or below it, the left child.
        let acc_g = if dl {
            total_g - pfx_g
        } else {
            if below { total_g } else { pfx_g }
        };
        let acc_h = if dl {
            total_h - pfx_h
        } else {
            if below { total_h } else { pfx_h }
        };

        let pg = parent_sum[(idx * 2u32) as usize];
        let ph = parent_sum[(idx * 2u32 + 1u32) as usize];
        let lg = if dl { pg - acc_g } else { acc_g };
        let lh = if dl { ph - acc_h } else { acc_h };

        out[(idx * 4u32) as usize] = lg;
        out[(idx * 4u32 + 1u32) as usize] = lh;
        out[(idx * 4u32 + 2u32) as usize] = pg - lg;
        out[(idx * 4u32 + 3u32) as usize] = ph - lh;
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

/// Per-node inputs for the vector-leaf evaluator.
///
/// The difference from [`NodeInput`] is that a node holds one histogram and one
/// parent sum *per target*, and that `root_gain` is their summed gain rather
/// than one node's.
#[derive(Clone, Debug)]
pub struct MultiNodeInput {
    /// Bin offset of this node's first target histogram within `hist`; target
    /// `t` follows at `hist_base + t * node_bins`.
    pub hist_base: u32,
    /// `Σ_t CalcGain(parent_t)` — `HistGrower::multi_gain` of the parent.
    pub root_gain: f32,
    pub lower: f32,
    pub upper: f32,
    /// Quantised parent sum, one per target.
    pub parent: Vec<GradientPairInt64>,
}

/// A batch's prefix sums, held on device between the split search that produced
/// them and the per-target child sums read out of them afterwards.
pub struct MultiScan {
    prefix: Handle,
    /// Length in bins, matching the histogram the scan was taken over.
    bins: usize,
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
    /// Candidate and winner buffers, kept across batches and grown on
    /// demand: five allocations a level through the runtime's pool were
    /// measurable host time for buffers whose size only ever reaches the
    /// widest level's.
    scratch: std::cell::RefCell<Scratch>,
}

/// The evaluator's per-batch buffers; see `SplitEvaluatorGpu::scratch`.
#[derive(Default)]
struct Scratch {
    /// Candidates the buffers hold; zero before the first batch.
    n_cand: usize,
    chg: Option<Handle>,
    sindex: Option<Handle>,
    value: Option<Handle>,
    left: Option<Handle>,
    /// Nodes the winner buffer holds.
    n_nodes: usize,
    best: Option<Handle>,
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
        if !super::supports_f64(&client) {
            return Err(crate::error::Error::NoF64Support);
        }
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
            scratch: Default::default(),
        })
    }

    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// The candidate buffers for `n_cand` `(node, feature)` pairs.
    fn candidate_buffers(&self, n_cand: usize) -> (Handle, Handle, Handle, Handle) {
        let mut sc = self.scratch.borrow_mut();
        if sc.n_cand < n_cand || sc.chg.is_none() {
            let c = &self.client;
            sc.chg = Some(c.empty(n_cand * size_of::<f32>()));
            sc.sindex = Some(c.empty(n_cand * size_of::<u32>()));
            sc.value = Some(c.empty(n_cand * size_of::<f32>()));
            sc.left = Some(c.empty(n_cand * 2 * size_of::<i64>()));
            sc.n_cand = n_cand;
        }
        (
            sc.chg.clone().expect("set above"),
            sc.sindex.clone().expect("set above"),
            sc.value.clone().expect("set above"),
            sc.left.clone().expect("set above"),
        )
    }

    /// The packed-winner buffer for `n_nodes` nodes.
    fn best_buffer(&self, n_nodes: usize) -> Handle {
        let mut sc = self.scratch.borrow_mut();
        if sc.n_nodes < n_nodes || sc.best.is_none() {
            sc.best = Some(self.client.empty(n_nodes * SPLIT_WORDS * size_of::<i64>()));
            sc.n_nodes = n_nodes;
        }
        sc.best.clone().expect("set above")
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
        let n_cand_u32 = (n_nodes * self.n_features) as u32;
        // Cooperation width: a cube per `(node, feature)` where a barrier is
        // cheap, a unit per pair where it is not. See `launch::cooperative`.
        let coop = launch::cooperative(c);
        let (eval_count, eval_dim) = if coop {
            (
                CubeCount::Static(n_nodes as u32, self.n_features as u32, 1),
                CubeDim::new_1d(launch::scan_block_1d(c, EVAL_BLOCK)),
            )
        } else {
            // A unit walks one feature's bins twice over (forward, then
            // backward when the feature has missing rows), so it is worth far
            // more than the one scalar operation `elementwise` assumes.
            launch::elementwise_with_work(c, n_cand_u32 as usize, 2 * self.n_bins / self.n_features)
        };
        // `block` is a *comptime* argument, so every distinct value it takes is
        // a separate kernel compilation — and on a JIT backend that is paid at
        // run time, per level, as the frontier changes the elementwise cube
        // width. In the serial shape `block` only has to be an upper bound on
        // the number of per-unit scratch slots, so it is pinned to a constant
        // and the kernel compiles once for the whole fit.
        let eval_block = if coop { eval_dim.x as usize } else { SERIAL_SLOTS };
        debug_assert!(eval_dim.x as usize <= eval_block);
        let base: Vec<u32> = nodes.iter().map(|n| n.hist_base).collect();
        let parent: Vec<i64> =
            nodes.iter().flat_map(|n| [n.parent_grad, n.parent_hess]).collect();
        let gain: Vec<f32> = nodes.iter().map(|n| n.root_gain).collect();
        let lower: Vec<f32> = nodes.iter().map(|n| n.lower).collect();
        let upper: Vec<f32> = nodes.iter().map(|n| n.upper).collect();

        // One upload for the batch's six tables (see `gpu::tables`).
        let mut tb = TableBuilder::new();
        let t_base = tb.push(&base);
        let t_parent = tb.push(&parent);
        let t_gain = tb.push(&gain);
        let t_lower = tb.push(&lower);
        let t_upper = tb.push(&upper);
        let t_mask = tb.push(feature_mask);
        let tables = tb.upload(c);

        let n_cand = n_nodes * self.n_features;
        let (cand_chg, cand_sindex, cand_value, cand_left) = self.candidate_buffers(n_cand);

        let has_mds = self.cfg.max_delta_step != 0.0;

        // SAFETY: the kernel guards every index against the lengths it is
        // given; see the `gpu` module docs on unchecked launches.
        unsafe {
            evaluate_feature_kernel::launch_unchecked::<R>(
                c,
                eval_count,
                eval_dim,
                ArrayArg::from_raw_parts(hist.clone(), hist_bins * 2),
                ArrayArg::from_raw_parts(self.cut_ptrs.clone(), self.n_features + 1),
                ArrayArg::from_raw_parts(self.cut_values.clone(), self.n_bins),
                ArrayArg::from_raw_parts(self.min_values.clone(), self.n_features),
                tables.arg(t_base, n_nodes),
                tables.arg(t_parent, n_nodes * 2),
                tables.arg(t_gain, n_nodes),
                tables.arg(t_lower, n_nodes),
                tables.arg(t_upper, n_nodes),
                tables.arg(t_mask, n_cand),
                ArrayArg::from_raw_parts(self.monotone.clone(), self.n_features),
                ArrayArg::from_raw_parts(cand_chg.clone(), n_cand),
                ArrayArg::from_raw_parts(cand_sindex.clone(), n_cand),
                ArrayArg::from_raw_parts(cand_value.clone(), n_cand),
                ArrayArg::from_raw_parts(cand_left.clone(), n_cand * 2),
                to_float_grad,
                to_float_hess,
                self.cfg.lambda,
                self.cfg.alpha,
                self.cfg.max_delta_step,
                self.cfg.min_child_weight,
                self.n_features as u32,
                n_cand_u32,
                self.has_constraint,
                has_mds,
                coop,
                eval_block,
            );
        }

        Ok(self.reduce_candidates(n_nodes, cand_chg, cand_sindex, cand_value, cand_left))
    }

    /// Reduce a batch's per-`(node, feature)` candidates to one per node and
    /// read them back: the tail both evaluators share.
    fn reduce_candidates(
        &self,
        n_nodes: usize,
        cand_chg: Handle,
        cand_sindex: Handle,
        cand_value: Handle,
        cand_left: Handle,
    ) -> Vec<DeviceSplitCandidate> {
        let c = &self.client;
        let n_cand = n_nodes * self.n_features;
        let best = self.best_buffer(n_nodes);

        // A cube per node where a barrier is cheap, a unit per node where it
        // is not; `block` is only the cooperative cube width, and pinned to a
        // constant bound on the serial scratch slots otherwise.
        let coop = launch::cooperative(c);
        let (count, dim, block) = if coop {
            let block = launch::scan_block_1d(c, EVAL_BLOCK);
            (CubeCount::Static(n_nodes as u32, 1, 1), CubeDim::new_1d(block), block as usize)
        } else {
            let (count, dim) = launch::elementwise_with_work(c, n_nodes, self.n_features);
            (count, dim, SERIAL_SLOTS)
        };
        debug_assert!(dim.x as usize <= block);

        // SAFETY: the kernel guards every index against the lengths it is
        // given; see the `gpu` module docs on unchecked launches.
        unsafe {
            reduce_candidates_kernel::launch_unchecked::<R>(
                c,
                count,
                dim,
                ArrayArg::from_raw_parts(cand_chg, n_cand),
                ArrayArg::from_raw_parts(cand_sindex, n_cand),
                ArrayArg::from_raw_parts(cand_value, n_cand),
                ArrayArg::from_raw_parts(cand_left, n_cand * 2),
                ArrayArg::from_raw_parts(best.clone(), n_nodes * SPLIT_WORDS),
                self.n_features as u32,
                n_nodes as u32,
                coop,
                block,
            );
        }

        // The pooled buffer can be wider than this batch: only its first
        // `n_nodes` winners are this batch's.
        let words: Vec<i64> = read_vec(c, best);
        words
            .chunks_exact(SPLIT_WORDS)
            .take(n_nodes)
            .map(|w| DeviceSplitCandidate {
                sindex: w[0] as u32,
                left_grad: w[1],
                left_hess: w[2],
                loss_chg: f32::from_bits(w[3] as u32),
                split_value: f32::from_bits(w[4] as u32),
                is_cat: false,
            })
            .collect()
    }

    /// [`evaluate`](Self::evaluate) for a vector-leaf tree.
    ///
    /// `node_bins` is the bin count of one target's histogram — the fit's total
    /// bin count — and a node's `n_targets` histograms sit side by side from
    /// its `hist_base`. The returned [`MultiScan`] holds the prefix sums the
    /// search ran over; feed it to [`multi_child_sums`](Self::multi_child_sums)
    /// to recover the chosen split's per-target child sums.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_multi(
        &self,
        hist: &Handle,
        hist_bins: usize,
        node_bins: usize,
        n_targets: usize,
        nodes: &[MultiNodeInput],
        feature_mask: &[u32],
        to_float_grad: f64,
        to_float_hess: f64,
    ) -> Result<(Vec<DeviceSplitCandidate>, MultiScan)> {
        let n_nodes = nodes.len();
        debug_assert!(n_nodes > 0, "an empty batch has nothing to scan");
        debug_assert_eq!(feature_mask.len(), n_nodes * self.n_features);
        debug_assert!(nodes.iter().all(|n| n.parent.len() == n_targets));

        let c = &self.client;
        // Cooperation width, as `evaluate` chooses it; both kernels below take
        // the same two shapes.
        let coop = launch::cooperative(c);
        let block = launch::scan_block_1d(c, EVAL_BLOCK);
        let n_cand = n_nodes * self.n_features;
        let base: Vec<u32> = nodes.iter().map(|n| n.hist_base).collect();
        let parent: Vec<i64> =
            nodes.iter().flat_map(|n| n.parent.iter().flat_map(|p| [p.grad, p.hess])).collect();
        let gain: Vec<f32> = nodes.iter().map(|n| n.root_gain).collect();
        let lower: Vec<f32> = nodes.iter().map(|n| n.lower).collect();
        let upper: Vec<f32> = nodes.iter().map(|n| n.upper).collect();

        // One upload for the batch's six tables (see `gpu::tables`).
        let mut tb = TableBuilder::new();
        let t_base = tb.push(&base);
        let t_parent = tb.push(&parent);
        let t_gain = tb.push(&gain);
        let t_lower = tb.push(&lower);
        let t_upper = tb.push(&upper);
        let t_mask = tb.push(feature_mask);
        let tables = tb.upload(c);

        // Same shape as the histogram it scans: one inclusive prefix per bin.
        let prefix = c.empty(hist_bins * 2 * size_of::<i64>());
        let n_runs = n_cand * n_targets;
        let (scan_count, scan_dim, scan_block) = if coop {
            (
                CubeCount::Static(n_nodes as u32, self.n_features as u32, n_targets as u32),
                CubeDim::new_1d(block),
                block as usize,
            )
        } else {
            // A unit walks one feature's bins once.
            let (count, dim) = launch::elementwise_with_work(c, n_runs, self.n_bins / self.n_features);
            (count, dim, 1)
        };
        // SAFETY: the kernel guards every index against the lengths it is
        // given; see the `gpu` module docs on unchecked launches.
        unsafe {
            prefix_scan_kernel::launch_unchecked::<R>(
                c,
                scan_count,
                scan_dim,
                ArrayArg::from_raw_parts(hist.clone(), hist_bins * 2),
                ArrayArg::from_raw_parts(self.cut_ptrs.clone(), self.n_features + 1),
                tables.arg(t_base, n_nodes),
                ArrayArg::from_raw_parts(prefix.clone(), hist_bins * 2),
                node_bins as u32,
                self.n_features as u32,
                n_targets as u32,
                n_runs as u32,
                coop,
                scan_block,
            );
        }

        let (cand_chg, cand_sindex, cand_value, cand_left) = self.candidate_buffers(n_cand);

        let has_mds = self.cfg.max_delta_step != 0.0;

        let (eval_count, eval_dim, eval_block) = if coop {
            (
                CubeCount::Static(n_nodes as u32, self.n_features as u32, 1),
                CubeDim::new_1d(block),
                block as usize,
            )
        } else {
            // A unit walks one feature's bins up to twice, reading every
            // target's prefix at each.
            let (count, dim) = launch::elementwise_with_work(
                c,
                n_cand,
                2 * n_targets * self.n_bins / self.n_features,
            );
            (count, dim, SERIAL_SLOTS)
        };
        debug_assert!(eval_dim.x as usize <= eval_block);

        // SAFETY: the kernel guards every index against the lengths it is
        // given; see the `gpu` module docs on unchecked launches.
        unsafe {
            evaluate_feature_multi_kernel::launch_unchecked::<R>(
                c,
                eval_count,
                eval_dim,
                ArrayArg::from_raw_parts(prefix.clone(), hist_bins * 2),
                ArrayArg::from_raw_parts(self.cut_ptrs.clone(), self.n_features + 1),
                ArrayArg::from_raw_parts(self.cut_values.clone(), self.n_bins),
                ArrayArg::from_raw_parts(self.min_values.clone(), self.n_features),
                tables.arg(t_base, n_nodes),
                tables.arg(t_parent, n_nodes * n_targets * 2),
                tables.arg(t_gain, n_nodes),
                tables.arg(t_lower, n_nodes),
                tables.arg(t_upper, n_nodes),
                tables.arg(t_mask, n_cand),
                ArrayArg::from_raw_parts(cand_chg.clone(), n_cand),
                ArrayArg::from_raw_parts(cand_sindex.clone(), n_cand),
                ArrayArg::from_raw_parts(cand_value.clone(), n_cand),
                ArrayArg::from_raw_parts(cand_left.clone(), n_cand * 2),
                to_float_grad,
                to_float_hess,
                self.cfg.lambda,
                self.cfg.alpha,
                self.cfg.max_delta_step,
                self.cfg.min_child_weight,
                self.n_features as u32,
                n_targets as u32,
                node_bins as u32,
                n_cand as u32,
                self.has_constraint,
                has_mds,
                coop,
                eval_block,
            );
        }

        // The per-feature reduction is target-blind: a candidate is already one
        // number by the time it gets here.
        let candidates = self.reduce_candidates(n_nodes, cand_chg, cand_sindex, cand_value, cand_left);
        Ok((candidates, MultiScan { prefix, bins: hist_bins }))
    }

    /// Per-target child sums of the splits `evaluate_multi` chose.
    ///
    /// `splits` gives `(feature, condition bin, default_left)` per node, with
    /// the condition derived from the chosen threshold exactly as the CPU
    /// grower's `find_split_condition` derives it. The result is `n_targets`
    /// `(left, right)` pairs per node, in node order.
    pub fn multi_child_sums(
        &self,
        scan: &MultiScan,
        node_bins: usize,
        n_targets: usize,
        nodes: &[MultiNodeInput],
        splits: &[(u32, i64, bool)],
    ) -> Vec<(GradientPairInt64, GradientPairInt64)> {
        debug_assert_eq!(nodes.len(), splits.len());
        let n_nodes = nodes.len();
        let n = n_nodes * n_targets;
        if n == 0 {
            return Vec::new();
        }

        let c = &self.client;
        let base: Vec<u32> = nodes.iter().map(|n| n.hist_base).collect();
        let parent: Vec<i64> =
            nodes.iter().flat_map(|n| n.parent.iter().flat_map(|p| [p.grad, p.hess])).collect();
        let fidx: Vec<u32> = splits.iter().map(|s| s.0).collect();
        let cond: Vec<i32> = splits.iter().map(|s| s.1 as i32).collect();
        let default_left: Vec<u32> = splits.iter().map(|s| u32::from(s.2)).collect();

        let base_d = c.create_from_slice(bytemuck::cast_slice(&base));
        let parent_d = c.create_from_slice(bytemuck::cast_slice(&parent));
        let fidx_d = c.create_from_slice(bytemuck::cast_slice(&fidx));
        let cond_d = c.create_from_slice(bytemuck::cast_slice(&cond));
        let dl_d = c.create_from_slice(bytemuck::cast_slice(&default_left));
        let out = c.empty(n * 4 * size_of::<i64>());

        let (cube_count, cube_dim) = super::launch::elementwise(c, n);
        // SAFETY: the kernel guards every index against the lengths it is
        // given; see the `gpu` module docs on unchecked launches.
        unsafe {
            multi_child_sums_kernel::launch_unchecked::<R>(
                c,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(scan.prefix.clone(), scan.bins * 2),
                ArrayArg::from_raw_parts(self.cut_ptrs.clone(), self.n_features + 1),
                ArrayArg::from_raw_parts(base_d, n_nodes),
                ArrayArg::from_raw_parts(parent_d, n * 2),
                ArrayArg::from_raw_parts(fidx_d, n_nodes),
                ArrayArg::from_raw_parts(cond_d, n_nodes),
                ArrayArg::from_raw_parts(dl_d, n_nodes),
                ArrayArg::from_raw_parts(out.clone(), n * 4),
                n_targets as u32,
                node_bins as u32,
                n as u32,
            );
        }

        let words: Vec<i64> = read_vec(c, out);
        words
            .chunks_exact(4)
            .map(|w| {
                (
                    GradientPairInt64 { grad: w[0], hess: w[1] },
                    GradientPairInt64 { grad: w[2], hess: w[3] },
                )
            })
            .collect()
    }
}

fn read_vec<R: Runtime, T: bytemuck::Pod>(client: &ComputeClient<R>, handle: Handle) -> Vec<T> {
    let bytes = client.read_one_unchecked(handle);
    bytemuck::cast_slice(&bytes).to_vec()
}

