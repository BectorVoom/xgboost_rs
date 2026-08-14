//! Tree training parameters and the regularised split arithmetic.
//!
//! The float/double mix here is deliberate and load-bearing: XGBoost
//! accumulates gradient sums in `double` but evaluates weights and gains in
//! `float` (`src/tree/split_evaluator.h`, `TreeEvaluator::SplitEvaluator`).
//! Doing it all in `f64` would produce split decisions that drift from the
//! reference on near-ties, so the widths are matched exactly.

use crate::parameters::{
    DefaultDirection, Device, MonotoneConstraint, ProcessType, SamplingMethod, TreeUpdaterName,
};

/// Growth order for the node expansion queue.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GrowPolicy {
    /// Expand all nodes of a level before descending.
    #[default]
    DepthWise,
    /// Expand the node with the largest loss change first.
    LossGuide,
}

/// The subset of XGBoost's `tree::TrainParam` that the CPU `hist` path reads.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainParam {
    pub learning_rate: f32,
    /// `gamma`: minimum loss reduction required to split.
    pub min_split_loss: f32,
    /// `0` means unlimited.
    pub max_depth: i32,
    /// `0` means unlimited.
    pub max_leaves: i32,
    pub max_bin: u32,
    pub grow_policy: GrowPolicy,
    pub min_child_weight: f32,
    pub reg_lambda: f32,
    pub reg_alpha: f32,
    pub max_delta_step: f32,

    /// Row subsample ratio, drawn once per tree.
    pub subsample: f32,
    /// How rows are drawn when `subsample < 1`.
    pub sampling_method: SamplingMethod,
    /// Column subsample ratio, drawn once per tree.
    pub colsample_bytree: f32,
    /// Column subsample ratio, drawn once per depth level.
    pub colsample_bylevel: f32,
    /// Column subsample ratio, drawn once per node.
    pub colsample_bynode: f32,
    /// Trees grown per boosting round.
    pub num_parallel_tree: u32,
    /// One monotonicity direction per feature; empty means unconstrained.
    pub monotone_constraints: Vec<MonotoneConstraint>,
    /// Feature groups allowed to interact; `None` means unconstrained.
    pub interaction_constraints: Option<Vec<Vec<u32>>>,
    /// Histogram buffers the grower may keep for reuse. Bounds the memory a
    /// deep or wide tree holds; it changes speed, never the model.
    pub max_cached_hist_node: u64,

    /// Whether one tree covers every output with a vector leaf, rather than
    /// one tree being grown per output. `multi_strategy=multi_output_tree`.
    pub multi_output_tree: bool,

    /// Density below which a column is stored sparsely in the transposed copy
    /// the row partitioner reads. Trades lookup speed for memory; it changes
    /// neither the split search nor the model.
    pub sparse_threshold: f64,

    /// Category count below which a categorical feature is split one-hot —
    /// "equals this category" — instead of by partitioning the categories.
    pub max_cat_to_onehot: u32,
    /// Largest number of categories a partition-based split may send right.
    pub max_cat_threshold: u32,

    /// Where the `exact` updater sends rows whose split feature is missing.
    pub default_direction: DefaultDirection,
    /// Density above which the `exact` updater skips a column's forward scan.
    pub opt_dense_col: f32,

    /// The resolved updater pipeline: one grower, then any tree-modifying
    /// stages. Under [`ProcessType::Update`] there is no grower and every
    /// stage rewrites a tree the model already holds.
    pub updaters: Vec<TreeUpdaterName>,
    /// Whether a round grows new trees or revisits the existing ones.
    pub process_type: ProcessType,
    /// Whether the `refresh` updater also rewrites leaf values.
    pub refresh_leaf: bool,
    /// Which processor runs the fit. The grower reads it to pick a device
    /// ordinal; every other stage is device-agnostic.
    pub device: Device,
}

impl Default for TrainParam {
    fn default() -> Self {
        Self {
            learning_rate: 0.3,
            min_split_loss: 0.0,
            max_depth: 6,
            max_leaves: 0,
            max_bin: 256,
            grow_policy: GrowPolicy::DepthWise,
            min_child_weight: 1.0,
            reg_lambda: 1.0,
            reg_alpha: 0.0,
            max_delta_step: 0.0,
            subsample: 1.0,
            sampling_method: SamplingMethod::Uniform,
            colsample_bytree: 1.0,
            colsample_bylevel: 1.0,
            colsample_bynode: 1.0,
            num_parallel_tree: 1,
            monotone_constraints: Vec::new(),
            interaction_constraints: None,
            // `HistMakerTrainParam::CpuDefaultNodes`.
            max_cached_hist_node: 1 << 16,
            multi_output_tree: false,
            sparse_threshold: 0.2,
            max_cat_to_onehot: 4,
            max_cat_threshold: 64,
            default_direction: DefaultDirection::Learn,
            opt_dense_col: 1.0,
            updaters: vec![TreeUpdaterName::GrowQuantileHistMaker],
            process_type: ProcessType::Default,
            refresh_leaf: true,
            device: Device::Cpu,
        }
    }
}

/// Gradient/Hessian sums for a node or a candidate child.
///
/// Accumulated in `f64`, matching upstream's `GradStats`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GradStats {
    pub sum_grad: f64,
    pub sum_hess: f64,
}

impl GradStats {
    #[inline]
    pub fn new(sum_grad: f64, sum_hess: f64) -> Self {
        Self { sum_grad, sum_hess }
    }

    #[inline]
    pub fn add(&mut self, grad: f64, hess: f64) {
        self.sum_grad += grad;
        self.sum_hess += hess;
    }

    #[inline]
    pub fn add_stats(&mut self, other: &Self) {
        self.sum_grad += other.sum_grad;
        self.sum_hess += other.sum_hess;
    }

    /// `self = a - b`, the subtraction trick.
    #[inline]
    pub fn set_subtract(&mut self, a: &Self, b: &Self) {
        self.sum_grad = a.sum_grad - b.sum_grad;
        self.sum_hess = a.sum_hess - b.sum_hess;
    }
}

/// `L1(g, a) = sign(g) * max(|g| - a, 0)`.
#[inline]
pub fn threshold_l1(sum_grad: f64, alpha: f32) -> f64 {
    let alpha = alpha as f64;
    if sum_grad > alpha {
        sum_grad - alpha
    } else if sum_grad < -alpha {
        sum_grad + alpha
    } else {
        0.0
    }
}

/// Optimal leaf weight for the given sums, clipped by `max_delta_step`.
///
/// Computed in `f64` and returned as `f32`, as upstream's
/// `SplitEvaluator::CalcWeight` does.
#[inline]
pub fn calc_weight(p: &TrainParam, stats: &GradStats) -> f32 {
    if stats.sum_hess <= 0.0 {
        return 0.0;
    }
    let mut dw = -threshold_l1(stats.sum_grad, p.reg_alpha) / (stats.sum_hess + p.reg_lambda as f64);
    if p.max_delta_step != 0.0 && dw.abs() > p.max_delta_step as f64 {
        dw = (p.max_delta_step as f64).copysign(dw);
    }
    dw as f32
}

/// Loss reduction of a node whose weight is the *unconstrained* optimum.
///
/// `tree::CalcGain`. This is the closed form `G²/(H+λ)`, which is only the
/// gain when the weight really is `-G/(H+λ)` — so it is not usable where a
/// monotone box may have clipped the weight, and
/// [`calc_gain_given_weight`] is used there instead.
///
/// The narrowing to `f32` *before* the division is upstream's and is kept:
/// it is what keeps average floating point error low, and reproducing it is
/// required for tie-for-tie agreement on split choices.
#[inline]
pub fn calc_gain(p: &TrainParam, stats: &GradStats) -> f32 {
    if stats.sum_hess <= 0.0 {
        return 0.0;
    }
    if p.max_delta_step == 0.0 {
        let num = threshold_l1(stats.sum_grad, p.reg_alpha);
        let num = (num * num) as f32;
        let den = (stats.sum_hess + p.reg_lambda as f64) as f32;
        return num / den;
    }
    calc_gain_given_weight(p, stats, calc_weight(p, stats))
}

/// Loss reduction contributed by a node **at the weight it will actually
/// take**.
///
/// `tree::CalcGainGivenWeight<ParamT, float>`, all-`f32` arithmetic. Unlike
/// [`calc_gain`] this reads `w`, which is the whole point: under a monotone
/// constraint the weight is clipped into the node's box and the closed form
/// — which assumes the unclipped optimum — would report a gain the split
/// cannot deliver. Getting that wrong lets a constrained node split where
/// upstream makes it a leaf.
#[inline]
pub fn calc_gain_given_weight(p: &TrainParam, stats: &GradStats, w: f32) -> f32 {
    if stats.sum_hess <= 0.0 {
        return 0.0;
    }
    let (g, h) = (stats.sum_grad as f32, stats.sum_hess as f32);
    -(2.0 * g * w + (h + p.reg_lambda) * w * w + 2.0 * p.reg_alpha * w.abs())
}

/// A candidate split, mirroring `SplitEntry`.
///
/// `Copy` because it is pure data that the split search moves around by the
/// thousand: the per-task results of a wide level are merged one by one, and a
/// memcpy beats a `clone` call there.
#[derive(Clone, Copy, Debug, Default)]
pub struct SplitEntry {
    pub loss_chg: f32,
    /// Feature index with the default-left flag in bit 31.
    pub sindex: u32,
    pub split_value: f32,
    pub left_sum: GradStats,
    pub right_sum: GradStats,
    /// Whether this split tests category membership rather than
    /// `value < split_value`. The chosen categories are not stored here: they
    /// are a variable-length bit set, and this type is deliberately `Copy`.
    pub is_cat: bool,
}

impl SplitEntry {
    #[inline]
    pub fn split_index(&self) -> u32 {
        self.sindex & ((1 << 31) - 1)
    }

    #[inline]
    pub fn default_left(&self) -> bool {
        (self.sindex >> 31) != 0
    }

    /// Whether `new_loss_chg` on `split_index` should replace this entry.
    ///
    /// Ties resolve towards the lower feature index, which is what makes
    /// multi-threaded evaluation reproducible upstream.
    #[inline]
    fn need_replace(&self, new_loss_chg: f32, split_index: u32) -> bool {
        if new_loss_chg.is_infinite() {
            // Can be inf/NaN when lambda and min_child_weight are both zero.
            return false;
        }
        if self.split_index() <= split_index {
            new_loss_chg > self.loss_chg
        } else {
            !(self.loss_chg > new_loss_chg)
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        new_loss_chg: f32,
        split_index: u32,
        new_split_value: f32,
        default_left: bool,
        left_sum: GradStats,
        right_sum: GradStats,
    ) -> bool {
        self.update_impl(new_loss_chg, split_index, new_split_value, default_left, left_sum, right_sum, false)
    }

    /// [`update`](Self::update) for a split on category membership.
    ///
    /// `new_split_value` carries no threshold — a categorical split has none —
    /// but a one-hot split still passes its category through it, which is how
    /// upstream's `EnumerateOneHot` recovers the single bit to set.
    #[allow(clippy::too_many_arguments)]
    pub fn update_cat(
        &mut self,
        new_loss_chg: f32,
        split_index: u32,
        new_split_value: f32,
        default_left: bool,
        left_sum: GradStats,
        right_sum: GradStats,
    ) -> bool {
        self.update_impl(new_loss_chg, split_index, new_split_value, default_left, left_sum, right_sum, true)
    }

    #[allow(clippy::too_many_arguments)]
    fn update_impl(
        &mut self,
        new_loss_chg: f32,
        split_index: u32,
        new_split_value: f32,
        default_left: bool,
        left_sum: GradStats,
        right_sum: GradStats,
        is_cat: bool,
    ) -> bool {
        if !self.need_replace(new_loss_chg, split_index) {
            return false;
        }
        self.loss_chg = new_loss_chg;
        self.sindex = if default_left { split_index | (1 << 31) } else { split_index };
        self.split_value = new_split_value;
        self.left_sum = left_sum;
        self.right_sum = right_sum;
        self.is_cat = is_cat;
        true
    }

    /// Merge another candidate into this one, keeping the better split.
    pub fn update_entry(&mut self, other: &SplitEntry) -> bool {
        if !self.need_replace(other.loss_chg, other.split_index()) {
            return false;
        }
        *self = *other;
        true
    }
}

/// `kRtEps` upstream: the smallest loss change worth splitting on.
pub const RT_EPS: f32 = 1e-6;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_is_the_regularised_newton_step() {
        let p = TrainParam { reg_lambda: 1.0, ..Default::default() };
        let s = GradStats::new(4.0, 3.0);
        assert_eq!(calc_weight(&p, &s), -1.0); // -4 / (3 + 1)
    }

    #[test]
    fn weight_is_clipped_by_max_delta_step() {
        let p = TrainParam { reg_lambda: 0.0, max_delta_step: 0.25, ..Default::default() };
        assert_eq!(calc_weight(&p, &GradStats::new(4.0, 1.0)), -0.25);
        assert_eq!(calc_weight(&p, &GradStats::new(-4.0, 1.0)), 0.25);
    }

    #[test]
    fn zero_hessian_has_no_weight_and_no_gain() {
        let p = TrainParam::default();
        let s = GradStats::new(1.0, 0.0);
        assert_eq!(calc_weight(&p, &s), 0.0);
        assert_eq!(calc_gain_given_weight(&p, &s, calc_weight(&p, &s)), 0.0);
    }

    #[test]
    fn l1_shrinks_the_gradient_towards_zero() {
        assert_eq!(threshold_l1(2.0, 0.5), 1.5);
        assert_eq!(threshold_l1(-2.0, 0.5), -1.5);
        assert_eq!(threshold_l1(0.25, 0.5), 0.0);
    }

    #[test]
    fn ties_prefer_the_lower_feature_index() {
        let mut best = SplitEntry::default();
        best.update(1.0, 3, 0.5, false, GradStats::default(), GradStats::default());
        // Same gain on a lower index replaces; on a higher index it does not.
        assert!(best.update(1.0, 1, 0.5, false, GradStats::default(), GradStats::default()));
        assert_eq!(best.split_index(), 1);
        assert!(!best.update(1.0, 2, 0.5, false, GradStats::default(), GradStats::default()));
    }

    #[test]
    fn default_left_rides_in_the_index_high_bit() {
        let mut e = SplitEntry::default();
        e.update(1.0, 7, 0.5, true, GradStats::default(), GradStats::default());
        assert_eq!(e.split_index(), 7);
        assert!(e.default_left());
    }
}
