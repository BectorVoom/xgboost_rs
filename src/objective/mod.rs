//! Objectives: the loss being boosted.
//!
//! Every `objective` XGBoost accepts is implemented here. An objective supplies
//! first and second derivatives per row, the transform applied to raw margins
//! at prediction time, the intercept (`base_score`) boosting starts from, and
//! how many outputs it produces per row.
//!
//! # Outputs per row
//!
//! Most objectives produce one output per row, but `multi:softmax` produces
//! `num_class`, `reg:quantileerror` one per `quantile_alpha`, and the
//! elementwise regression losses one per `num_target`. That count —
//! [`Objective::num_output_group`] — is what makes a boosting round grow a
//! *group* of trees rather than a single one; predictions are laid out
//! row-major `(row, output)` throughout.
//!
//! # `base_score` lives in prediction space
//!
//! Upstream stores `base_score` on the *prediction* scale (a probability for
//! the logistic losses, a rate for the log-link ones) and converts it to a
//! margin with `ProbToMargin` when a fit starts. The same split is kept here:
//! [`Objective::init_estimation`] returns prediction-space values and
//! [`Objective::prob_to_margin`] converts them, so a `base_score` a caller sets
//! means what the XGBoost documentation says it means.

pub mod classification;
pub mod ranking;
pub mod regression;
pub mod survival;

use crate::data::MetaInfo;
use crate::parameters::Objective as ObjectiveSpec;
use crate::{Error, Result};

pub use classification::{BinaryHinge, LogitRaw, SoftmaxMultiClass};
pub use ranking::{LambdaRank, RankLoss};
pub use regression::{
    ExpectileRegression, GammaRegression, MeanAbsoluteError, PoissonRegression, PseudoHuber,
    QuantileRegression, RegLossObj, SquaredError, TweedieRegression,
};
pub use survival::{AftSurvival, CoxRegression};

/// `LearnerModelParam::kDefaultBaseScore` — the intercept an objective that
/// does not estimate one is left with.
pub const DEFAULT_BASE_SCORE: f32 = 0.5;

/// First and second derivative of the loss at one `(row, output)`.
///
/// An objective the device computes gradients for; see `gpu::objective`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DeviceObjective {
    /// `reg:squarederror`, with `scale_pos_weight` applied to rows labelled
    /// exactly one as `RegLossObj::weight` applies it.
    SquaredError { scale_pos_weight: f32 },
}

/// Mirrors `xgboost::GradientPair`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GradientPair {
    pub grad: f32,
    pub hess: f32,
}

/// A boosting objective.
pub trait Objective {
    /// The XGBoost parameter value that selects this objective.
    fn name(&self) -> &'static str;

    /// Outputs per row. One tree per output is grown each round.
    ///
    /// The default reads the label matrix, which is `ObjFunction::Targets`.
    fn num_output_group(&self, info: &MetaInfo) -> usize {
        info.n_targets()
    }

    /// Fill `out` with per-`(row, output)` gradients for the current margins.
    ///
    /// Takes `&mut self` because an objective may carry state that a round
    /// updates: unbiased LambdaMART re-estimates its position propensities from
    /// every round's pairs, and upstream keeps that state on the objective too,
    /// down to saving it in the model.
    fn get_gradient(
        &mut self,
        preds: &[f32],
        info: &MetaInfo,
        iter: i32,
        out: &mut Vec<GradientPair>,
    );

    /// The objective's device form, for a fit whose trees grow on the device
    /// to compute its gradients there too; `None` keeps the host path.
    fn device_kind(&self) -> Option<DeviceObjective> {
        None
    }

    /// Whether this objective draws from the *session* random engine before
    /// each gradient computation.
    ///
    /// Only `rank:*` under `lambdarank_pair_method=mean` does, and upstream
    /// draws exactly then (`LambdaRankObj::GetGradient`). The draw has to stay
    /// conditional: the engine is shared with column and row sampling, so an
    /// unconditional one would shift every other objective's samples.
    fn wants_pair_seed(&self) -> bool {
        false
    }

    /// Hand over the draw [`wants_pair_seed`](Self::wants_pair_seed) asked for.
    fn set_pair_seed(&mut self, seed: u32) {
        let _ = seed;
    }

    /// Map raw margins to the reported prediction scale.
    ///
    /// Takes a `Vec` because `multi:softmax` shortens it: one class index
    /// replaces the `num_class` margins of a row.
    fn pred_transform(&self, preds: &mut Vec<f32>) {
        let _ = preds;
    }

    /// The transform metrics see, upstream's `EvalTransform`.
    ///
    /// It differs from [`pred_transform`](Self::pred_transform) exactly where a
    /// metric needs more than the user-facing prediction: `multi:softmax`
    /// reports a class index but `mlogloss` needs the probabilities, and
    /// `survival:aft` reports a time but `aft-nloglik` needs the margin.
    fn eval_transform(&self, preds: &mut Vec<f32>) {
        self.pred_transform(preds);
    }

    /// Convert an intercept from prediction space to margin space, in place.
    fn prob_to_margin(&self, base_score: &mut [f32]) -> Result<()> {
        let _ = base_score;
        Ok(())
    }

    /// Estimate the intercept boosting starts from, in *prediction* space, one
    /// value per output.
    fn init_estimation(&mut self, info: &MetaInfo) -> Vec<f32>;

    /// Whether every row's hessian is the same in every round, which is
    /// `ObjInfo::const_hess` upstream.
    ///
    /// Only `reg:squarederror` (and its `reg:linear` alias) qualifies, because
    /// its second derivative is `1` whatever the prediction. It is what lets
    /// `approx` sketch its quantiles once instead of once per round: the
    /// sketch is weighted by the hessian, and a hessian that never moves gives
    /// a sketch that never moves.
    fn has_constant_hessian(&self) -> bool {
        false
    }

    /// Metric reported when the caller did not choose one.
    fn default_metric(&self) -> String;

    /// Build a metric, giving it the objective's own parameters where it needs
    /// them.
    ///
    /// This is upstream's `DefaultMetricConfig`, generalised: `aft-nloglik`
    /// needs the AFT distribution and scale, and `quantile`/`expectile` need
    /// the alphas, none of which the metric's *name* can carry. Objectives that
    /// own such a metric override this; everything else builds from the name.
    fn make_metric(&self, spec: &crate::parameters::EvalMetric) -> Result<Box<dyn crate::metric::Metric>> {
        crate::metric::create_from(spec)
    }

    /// Reject data this objective cannot consume — a label out of range, a
    /// missing censoring bound — before the first round rather than at the
    /// first row.
    fn validate_data(&self, info: &MetaInfo) -> Result<()> {
        let _ = info;
        Ok(())
    }
}

/// `FitIntercept::InitEstimation`: one Newton step from a zero margin.
///
/// The step is `-sum(grad) / sum(hess)` per output — the weight a single-leaf
/// tree would take — mapped back to prediction space so the value round-trips
/// through [`Objective::prob_to_margin`].
pub fn fit_intercept<O: Objective + ?Sized>(obj: &mut O, info: &MetaInfo) -> Vec<f32> {
    let n_groups = obj.num_output_group(info);
    if info.num_row == 0 || n_groups == 0 {
        return vec![0.0; n_groups.max(1)];
    }
    let preds = vec![0.0f32; info.num_row * n_groups];
    let mut gpair = Vec::new();
    obj.get_gradient(&preds, info, 0, &mut gpair);

    let mut out = vec![0.0f32; n_groups];
    for (t, slot) in out.iter_mut().enumerate() {
        let (mut g, mut h) = (0.0f64, 0.0f64);
        for i in 0..info.num_row {
            let p = gpair[i * n_groups + t];
            g += p.grad as f64;
            h += p.hess as f64;
        }
        // `FitStump` leaves the weight at zero when there is no curvature.
        *slot = if h < 1e-6 { 0.0 } else { (-g / h) as f32 };
    }
    // `FitIntercept` applies the prediction transform so the stored intercept
    // is on the scale a user-supplied `base_score` would be. A transform that
    // changes the length (`multi:softmax`) is not one an intercept can take.
    let mut transformed = out.clone();
    obj.pred_transform(&mut transformed);
    if transformed.len() == out.len() { transformed } else { out }
}

/// `FitInterceptGlmLike::InitEstimation`: the weighted label mean.
///
/// Used by the objectives whose link makes the mean the natural starting
/// point; the result is already in prediction space.
pub fn fit_intercept_glm_like(info: &MetaInfo, n_groups: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_groups];
    if info.num_row == 0 {
        return out;
    }
    let targets = info.n_targets();
    for (t, slot) in out.iter_mut().enumerate() {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for i in 0..info.num_row {
            let w = info.weight(i) as f64;
            num += info.label(i, t.min(targets - 1)) as f64 * w;
            den += w;
        }
        *slot = if den == 0.0 { 0.0 } else { (num / den) as f32 };
    }
    out
}

/// Quantile of `values` at `alpha`, weighted when weights are given.
///
/// Mirrors `common::Quantile` / `common::WeightedQuantile`.
pub(crate) fn weighted_quantile(alpha: f32, values: &[f32], weights: Option<&[f32]>) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));

    let Some(weights) = weights else {
        // `common::Quantile`: linear interpolation at `alpha * (n - 1)`.
        let pos = alpha as f64 * (values.len() - 1) as f64;
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        let frac = pos - lo as f64;
        let (a, b) = (values[order[lo]] as f64, values[order[hi]] as f64);
        return (a + (b - a) * frac) as f32;
    };

    let total: f64 = order.iter().map(|&i| weights[i] as f64).sum();
    if total <= 0.0 {
        return values[order[0]];
    }
    // The value whose cumulative weight first reaches `alpha`, averaging the
    // two straddling values when the boundary falls exactly between them.
    let target = alpha as f64 * total;
    let mut acc = 0.0f64;
    for (k, &i) in order.iter().enumerate() {
        let next = acc + weights[i] as f64;
        if next >= target {
            if k + 1 == order.len() || next > target {
                return values[i];
            }
            return ((values[i] as f64 + values[order[k + 1]] as f64) / 2.0) as f32;
        }
        acc = next;
    }
    values[*order.last().expect("non-empty")]
}

/// `common::Sigmoid`.
#[inline]
pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Construct the objective a [`ObjectiveSpec`] selects, with the learning-task
/// parameters objectives read.
pub fn create(spec: &ObjectiveSpec, scale_pos_weight: f32) -> Result<Box<dyn Objective>> {
    use ObjectiveSpec::*;
    Ok(match spec {
        RegSquaredError | RegLinear => {
            Box::new(RegLossObj::new(regression::Loss::SquaredError, scale_pos_weight))
        }
        RegSquaredLogError => {
            Box::new(RegLossObj::new(regression::Loss::SquaredLogError, scale_pos_weight))
        }
        RegLogistic => Box::new(RegLossObj::new(regression::Loss::Logistic, scale_pos_weight)),
        BinaryLogistic => {
            Box::new(RegLossObj::new(regression::Loss::BinaryLogistic, scale_pos_weight))
        }
        BinaryLogitRaw => Box::new(LogitRaw::new(scale_pos_weight)),
        BinaryHinge => Box::new(classification::BinaryHinge),
        RegPseudoHuberError { huber_slope } => Box::new(PseudoHuber::new(*huber_slope)),
        RegAbsoluteError => Box::new(MeanAbsoluteError),
        RegQuantileError { quantile_alpha } => {
            Box::new(QuantileRegression::new(quantile_alpha.clone())?)
        }
        RegExpectileError { expectile_alpha } => {
            Box::new(ExpectileRegression::new(expectile_alpha.clone())?)
        }
        RegGamma => Box::new(GammaRegression),
        RegTweedie { tweedie_variance_power } => {
            Box::new(TweedieRegression::new(*tweedie_variance_power))
        }
        CountPoisson { max_delta_step } => Box::new(PoissonRegression::new(*max_delta_step)),
        SurvivalCox => Box::new(CoxRegression),
        SurvivalAft { aft_loss_distribution, aft_loss_distribution_scale } => {
            Box::new(AftSurvival::new(*aft_loss_distribution, *aft_loss_distribution_scale))
        }
        MultiSoftmax { num_class } => Box::new(SoftmaxMultiClass::new(*num_class as usize, false)),
        MultiSoftprob { num_class } => Box::new(SoftmaxMultiClass::new(*num_class as usize, true)),
        RankPairwise(p) => Box::new(LambdaRank::new(RankLoss::Pairwise, *p)),
        RankNdcg(p) => Box::new(LambdaRank::new(RankLoss::Ndcg, *p)),
        RankMap(p) => Box::new(LambdaRank::new(RankLoss::Map, *p)),
    })
}

/// Construct an objective from its XGBoost name at that objective's defaults.
///
/// Used when loading a saved model, where the tuning parameters that do not
/// change prediction are not all recorded.
pub fn create_by_name(name: &str) -> Result<Box<dyn Objective>> {
    let spec: ObjectiveSpec = name.parse()?;
    create(&spec, 1.0)
}

/// Reject a label the objective's link cannot represent.
pub(crate) fn check_labels(
    name: &'static str,
    info: &MetaInfo,
    ok: impl Fn(f32) -> bool,
    reason: &str,
) -> Result<()> {
    if let Some(bad) = info.labels.iter().copied().find(|y| !ok(*y)) {
        return Err(Error::invalid("objective", format!("`{name}`: {reason}, got label {bad}")));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameters::Objective as Spec;

    fn info(labels: &[f32]) -> MetaInfo {
        MetaInfo {
            num_row: labels.len(),
            num_col: 1,
            labels: labels.to_vec(),
            num_target: 1,
            ..Default::default()
        }
    }

    #[test]
    fn every_objective_spelling_constructs() {
        for name in [
            "reg:squarederror",
            "reg:squaredlogerror",
            "reg:logistic",
            "reg:pseudohubererror",
            "reg:absoluteerror",
            "reg:gamma",
            "reg:tweedie",
            "reg:linear",
            "count:poisson",
            "survival:cox",
            "survival:aft",
            "binary:logistic",
            "binary:logitraw",
            "binary:hinge",
            "rank:pairwise",
            "rank:ndcg",
            "rank:map",
        ] {
            let obj = create_by_name(name).unwrap_or_else(|e| panic!("{name}: {e}"));
            // `reg:linear` is the deprecated spelling of squared error and
            // reports the objective it actually is, as upstream does.
            let expected = if name == "reg:linear" { "reg:squarederror" } else { name };
            assert_eq!(obj.name(), expected);
            assert!(!obj.default_metric().is_empty(), "{name} has no default metric");
        }
        for spec in [
            Spec::MultiSoftmax { num_class: 3 },
            Spec::MultiSoftprob { num_class: 3 },
            Spec::RegQuantileError { quantile_alpha: vec![0.5] },
            Spec::RegExpectileError { expectile_alpha: vec![0.5] },
        ] {
            let obj = create(&spec, 1.0).unwrap();
            assert_eq!(obj.name(), spec.name());
        }
    }

    #[test]
    fn the_generic_intercept_is_the_newton_step() {
        // Squared error at a zero margin: -sum(grad)/sum(hess) is the mean.
        let mut obj = create(&Spec::RegSquaredError, 1.0).unwrap();
        assert_eq!(obj.init_estimation(&info(&[1.0, 2.0, 3.0])), vec![2.0]);
    }

    #[test]
    fn quantiles_bracket_the_data() {
        let values = [1.0f32, 2.0, 3.0, 4.0];
        assert_eq!(weighted_quantile(0.0, &values, None), 1.0);
        assert_eq!(weighted_quantile(1.0, &values, None), 4.0);
        assert_eq!(weighted_quantile(0.5, &values, None), 2.5);
        // A weight concentrated on one value pulls every quantile onto it.
        assert_eq!(weighted_quantile(0.5, &values, Some(&[0.0, 0.0, 1.0, 0.0])), 3.0);
    }
}
