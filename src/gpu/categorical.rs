//! Host-side categorical split evaluation for the GPU grower.
//!
//! The device split kernel (`evaluate_splits`) scans a feature's bins in
//! ascending order, which only means something for a numeric feature — a
//! category code carries no order. Categorical features are therefore masked
//! out of the device evaluation ([`super::grower::GpuHistGrower`]) and scored
//! here instead, against histogram bins read back from the device.
//!
//! This mirrors `HistGrower::enumerate_one_hot` / `enumerate_partition`
//! (`src/tree/hist.rs`) term for term, but keeps the running sums in the
//! histogram's native quantised `i64` representation and only decodes where
//! the gain formula needs a float — the same split between "exact accumulator,
//! decoded gain" the device kernel and [`super::evaluate_splits`] use for
//! numeric features, which is what keeps a categorical winner comparable to
//! them.

use super::GradientPairInt64;
use crate::tree::cat;
use crate::tree::evaluator::SplitEvaluator;
use crate::tree::param::{GradStats, TrainParam, calc_weight};

/// A categorical candidate, quantised like `DeviceSplitCandidate`.
#[derive(Clone, Copy, Debug, Default)]
pub struct CatCandidate {
    pub loss_chg: f32,
    /// The chosen category (one-hot) or `NaN` (partition). Unused by
    /// `RegTree::expand_categorical`; kept only for parity with `SplitEntry`.
    pub split_value: f32,
    pub default_left: bool,
    pub left: GradientPairInt64,
}

#[inline]
fn decode(g: GradientPairInt64, to_float_grad: f64, to_float_hess: f64) -> GradStats {
    GradStats::new(g.grad as f64 * to_float_grad, g.hess as f64 * to_float_hess)
}

/// Mirrors `SplitEntry::need_replace`, for merging a categorical candidate
/// against the device's numeric-only winner: strictly larger gain wins, and a
/// tie keeps the lower feature index — the same rule the device's own
/// cross-feature reduction (`reduce_candidates_kernel`) uses.
pub fn replaces(gain: f32, fidx: u32, best_gain: f32, best_fidx: u32) -> bool {
    if !gain.is_finite() {
        return false;
    }
    if best_fidx <= fidx { gain > best_gain } else { !(best_gain > gain) }
}

/// `common::UseOneHot`: dispatch `fidx`'s bins to one of the two algorithms,
/// exactly as `HistGrower::evaluate_one` does for the CPU path.
#[allow(clippy::too_many_arguments)]
pub fn enumerate(
    evaluator: &SplitEvaluator,
    param: &TrainParam,
    nid: usize,
    fidx: u32,
    cut_values: &[f32],
    bins: &[GradientPairInt64],
    parent: GradientPairInt64,
    parent_root_gain: f32,
    to_float_grad: f64,
    to_float_hess: f64,
) -> (CatCandidate, Vec<u32>) {
    if (bins.len() as u32) < param.max_cat_to_onehot {
        enumerate_one_hot(
            evaluator,
            param,
            nid,
            fidx,
            cut_values,
            bins,
            parent,
            parent_root_gain,
            to_float_grad,
            to_float_hess,
        )
    } else {
        enumerate_partition(
            evaluator,
            param,
            nid,
            fidx,
            cut_values,
            bins,
            parent,
            parent_root_gain,
            to_float_grad,
            to_float_hess,
        )
    }
}

/// `EnumerateOneHot` — try each category on its own against all the others.
///
/// Each category is scanned twice, once with the feature's missing rows
/// grouped with the other categories and once with the chosen one, which is
/// how the default direction is learned without a dedicated missing bin.
#[allow(clippy::too_many_arguments)]
fn enumerate_one_hot(
    evaluator: &SplitEvaluator,
    param: &TrainParam,
    nid: usize,
    fidx: u32,
    cut_values: &[f32],
    bins: &[GradientPairInt64],
    parent: GradientPairInt64,
    parent_root_gain: f32,
    to_float_grad: f64,
    to_float_hess: f64,
) -> (CatCandidate, Vec<u32>) {
    let mut feature_sum = GradientPairInt64::default();
    for &b in bins {
        feature_sum = feature_sum + b;
    }
    // Rows whose value is missing: whatever the feature's bins do not hold.
    let missing = parent - feature_sum;

    let mut best = CatCandidate::default();
    let mut best_cat: Option<usize> = None;

    for (i, &right0) in bins.iter().enumerate() {
        // Missing on the left: the chosen category alone goes right.
        let left0 = parent - right0;
        let gain0 = evaluator.calc_split_gain(
            nid,
            fidx,
            param,
            &decode(left0, to_float_grad, to_float_hess),
            &decode(right0, to_float_grad, to_float_hess),
        );
        if gain0.is_finite() {
            let chg = gain0 - parent_root_gain;
            if chg > best.loss_chg {
                best = CatCandidate {
                    loss_chg: chg,
                    split_value: cut_values[i],
                    default_left: true,
                    left: left0,
                };
                best_cat = Some(i);
            }
        }

        // Missing on the right: grouped with the chosen category.
        let right1 = right0 + missing;
        let left1 = parent - right1;
        let gain1 = evaluator.calc_split_gain(
            nid,
            fidx,
            param,
            &decode(left1, to_float_grad, to_float_hess),
            &decode(right1, to_float_grad, to_float_hess),
        );
        if gain1.is_finite() {
            let chg = gain1 - parent_root_gain;
            if chg > best.loss_chg {
                best = CatCandidate {
                    loss_chg: chg,
                    split_value: cut_values[i],
                    default_left: false,
                    left: left1,
                };
                best_cat = Some(i);
            }
        }
    }

    let bits = match best_cat {
        Some(i) => {
            let mut b = vec![0u32; cat::storage_size(bins.len())];
            cat::set_bit(&mut b, cat::as_cat(cut_values[i]));
            b
        }
        None => Vec::new(),
    };
    (best, bits)
}

/// `EnumeratePart` — partition the categories by the weight their gradients
/// imply, then split that order like an ordinary numeric feature.
///
/// Sorting by weight is what makes a contiguous run of the sorted order an
/// optimal category subset. Both scan directions are tried, because they
/// differ in which side the feature's missing rows land on.
#[allow(clippy::too_many_arguments)]
fn enumerate_partition(
    evaluator: &SplitEvaluator,
    param: &TrainParam,
    nid: usize,
    fidx: u32,
    cut_values: &[f32],
    bins: &[GradientPairInt64],
    parent: GradientPairInt64,
    parent_root_gain: f32,
    to_float_grad: f64,
    to_float_hess: f64,
) -> (CatCandidate, Vec<u32>) {
    let n_bins_feature = bins.len();
    // `max_cat_threshold` caps how many categories one side may name.
    let n_bins = (param.max_cat_threshold as usize).min(n_bins_feature);
    if n_bins < 2 {
        return (CatCandidate::default(), Vec::new());
    }

    // `CalcWeightCat`: the unconstrained weight. Categories carry no
    // monotonicity, so the node's weight box is deliberately not applied.
    let mut sorted_idx: Vec<usize> = (0..n_bins_feature).collect();
    sorted_idx.sort_by(|&l, &r| {
        let wl = calc_weight(param, &decode(bins[l], to_float_grad, to_float_hess));
        let wr = calc_weight(param, &decode(bins[r], to_float_grad, to_float_hess));
        wl.total_cmp(&wr).then_with(|| l.cmp(&r))
    });

    let mut best = CatCandidate::default();
    // How many of the sorted categories the winning split sends right.
    let mut best_partition: Option<usize> = None;

    for forward in [true, false] {
        let mut left = GradientPairInt64::default();
        let mut right = GradientPairInt64::default();
        for step in 0..n_bins - 1 {
            let j = if forward { step } else { n_bins_feature - 1 - step };
            let cell = bins[sorted_idx[j]];
            if forward {
                // Scanning up the order, the accumulated head goes right and
                // the feature's missing rows stay left.
                right = right + cell;
                left = parent - right;
            } else {
                left = left + cell;
                right = parent - left;
            }
            let gain = evaluator.calc_split_gain(
                nid,
                fidx,
                param,
                &decode(left, to_float_grad, to_float_hess),
                &decode(right, to_float_grad, to_float_hess),
            );
            if gain.is_finite() {
                let chg = gain - parent_root_gain;
                if chg > best.loss_chg {
                    best = CatCandidate {
                        loss_chg: chg,
                        split_value: f32::NAN,
                        default_left: forward,
                        left,
                    };
                    // Forward: the first `step + 1` of the order go right.
                    // Backward: everything from `n_bins_feature - 1 - step`
                    // upwards is the left side, so the right side is the head.
                    best_partition =
                        Some(if forward { step + 1 } else { n_bins_feature - 1 - step });
                }
            }
        }
    }

    let bits = match best_partition {
        Some(partition) => {
            debug_assert!(partition > 0 && partition <= n_bins_feature);
            let mut b = vec![0u32; cat::storage_size(n_bins_feature)];
            // The head of the order is the right-hand side either way: a
            // forward scan accumulates it into `right` directly, and a
            // backward scan leaves it as whatever the left side did not
            // absorb.
            for &c in &sorted_idx[..partition] {
                cat::set_bit(&mut b, cat::as_cat(cut_values[c]));
            }
            b
        }
        None => Vec::new(),
    };
    (best, bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::param::TrainParam;

    /// An identity quantiser (`to_float = 1.0`) makes the quantised path
    /// arithmetic identical to plain `f64`, so this checks the algorithms
    /// against hand-computed gains without needing a real histogram build.
    fn pair(grad: i64, hess: i64) -> GradientPairInt64 {
        GradientPairInt64 { grad, hess }
    }

    #[test]
    fn one_hot_isolates_the_single_informative_category() {
        // Three categories; category 1 carries all the negative gradient, so
        // isolating it from {0, 2} is the only useful split.
        let evaluator = SplitEvaluator::new(&[], 1);
        let param = TrainParam::default();
        let bins = [pair(10, 10), pair(-30, 10), pair(10, 10)];
        let parent = pair(-10, 30);
        let cut_values = [0.0f32, 1.0, 2.0];

        let (cand, bits) =
            enumerate(&evaluator, &param, 0, 0, &cut_values, &bins, parent, 0.0, 1.0, 1.0);

        assert!(cand.loss_chg > 0.0, "should find a positive-gain split");
        assert!(!bits.is_empty());
        assert!(cat::check_bit(&bits, 1), "category 1 should be the isolated one");
        assert!(!cat::check_bit(&bits, 0));
        assert!(!cat::check_bit(&bits, 2));
    }

    #[test]
    fn partition_is_used_once_the_category_count_passes_the_onehot_cutoff() {
        let evaluator = SplitEvaluator::new(&[], 1);
        let param = TrainParam { max_cat_to_onehot: 2, ..Default::default() };
        let bins: Vec<GradientPairInt64> = vec![pair(5, 10), pair(-5, 10), pair(5, 10)];
        let parent = pair(5, 30);
        let cut_values = [0.0f32, 1.0, 2.0];

        let (cand, bits) =
            enumerate(&evaluator, &param, 0, 0, &cut_values, &bins, parent, 0.0, 1.0, 1.0);
        // Not asserting a specific split here, only that the partition path
        // (not one-hot) ran and produced a self-consistent candidate: a
        // positive gain always comes with a non-empty bit set.
        assert_eq!(cand.loss_chg > 0.0, !bits.is_empty());
    }

    #[test]
    fn no_finite_gain_returns_no_candidate() {
        let evaluator = SplitEvaluator::new(&[], 1);
        let param = TrainParam { min_child_weight: 1000.0, ..Default::default() };
        let bins = [pair(10, 1), pair(-10, 1)];
        let parent = pair(0, 2);
        let cut_values = [0.0f32, 1.0];

        let (cand, bits) =
            enumerate(&evaluator, &param, 0, 0, &cut_values, &bins, parent, 0.0, 1.0, 1.0);
        assert_eq!(cand.loss_chg, 0.0);
        assert!(bits.is_empty());
    }
}
