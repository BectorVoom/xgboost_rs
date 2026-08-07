//! The regression objectives.
//!
//! Ports of `src/objective/regression_obj.cu`, `regression_loss.h` and
//! `quantile_obj.cu`. The float widths are upstream's: gradients are `f32`
//! because that is what the histogram accumulates, while the scale statistics
//! the smooth losses derive are `f64`.

use super::{
    GradientPair, Objective, check_labels, fit_intercept, fit_intercept_glm_like, sigmoid,
    weighted_quantile,
};
use crate::data::MetaInfo;
use crate::{Error, Result};

/// `kRtEps`, the epsilon upstream bounds intercepts and denominators with.
const RT_EPS: f32 = 1e-6;

// ------------------------------------------------------------ RegLossObj ----

/// The pointwise losses `RegLossObj` is instantiated with.
///
/// One enum rather than a generic parameter: the objective is chosen by a
/// runtime string, so the dispatch has to be runtime anyway, and keeping the
/// five losses side by side makes them easy to compare with `regression_loss.h`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Loss {
    /// `reg:squarederror` — `LinearSquareLoss`.
    SquaredError,
    /// `reg:squaredlogerror` — `SquaredLogError`.
    SquaredLogError,
    /// `reg:logistic` — `LogisticRegression`.
    Logistic,
    /// `binary:logistic` — `LogisticClassification`, logistic with a
    /// classification default metric.
    BinaryLogistic,
}

impl Loss {
    const fn name(self) -> &'static str {
        match self {
            Self::SquaredError => "reg:squarederror",
            Self::SquaredLogError => "reg:squaredlogerror",
            Self::Logistic => "reg:logistic",
            Self::BinaryLogistic => "binary:logistic",
        }
    }

    const fn default_metric(self) -> &'static str {
        match self {
            Self::SquaredError | Self::Logistic => "rmse",
            Self::SquaredLogError => "rmsle",
            Self::BinaryLogistic => "logloss",
        }
    }

    #[inline]
    fn pred_transform(self, x: f32) -> f32 {
        match self {
            Self::SquaredError | Self::SquaredLogError => x,
            Self::Logistic | Self::BinaryLogistic => sigmoid(x),
        }
    }

    #[inline]
    fn first_order(self, predt: f32, label: f32) -> f32 {
        match self {
            Self::SquaredError | Self::Logistic | Self::BinaryLogistic => predt - label,
            Self::SquaredLogError => {
                let p = predt.max(-1.0 + 1e-6);
                (p.ln_1p() - label.ln_1p()) / (p + 1.0)
            }
        }
    }

    #[inline]
    fn second_order(self, predt: f32, label: f32) -> f32 {
        match self {
            Self::SquaredError => 1.0,
            Self::Logistic | Self::BinaryLogistic => (predt * (1.0 - predt)).max(1e-16),
            Self::SquaredLogError => {
                let p = predt.max(-1.0 + 1e-6);
                ((-p.ln_1p() + label.ln_1p() + 1.0) / (p + 1.0).powi(2)).max(1e-6)
            }
        }
    }

    /// Whether this loss's link needs the intercept converted, and how.
    fn prob_to_margin(self, base_score: f32) -> Result<f32> {
        match self {
            Self::SquaredError | Self::SquaredLogError => Ok(base_score),
            Self::Logistic | Self::BinaryLogistic => {
                if !(0.0..=1.0).contains(&base_score) {
                    return Err(Error::invalid(
                        "base_score",
                        "base_score must be in (0,1) for the logistic loss",
                    ));
                }
                let bounded = base_score.clamp(RT_EPS, 1.0 - RT_EPS);
                Ok((bounded / (1.0 - bounded)).ln())
            }
        }
    }

    fn check_labels(self, info: &MetaInfo) -> Result<()> {
        match self {
            Self::SquaredError => Ok(()),
            Self::SquaredLogError => check_labels(
                self.name(),
                info,
                |y| y > -1.0,
                "label must be greater than -1 so that log(label + 1) is valid",
            ),
            Self::Logistic | Self::BinaryLogistic => check_labels(
                self.name(),
                info,
                |y| (0.0..=1.0).contains(&y),
                "label must be in [0, 1] for logistic regression",
            ),
        }
    }
}

/// The elementwise regression objectives, upstream's `RegLossObj<Loss>`.
///
/// `scale_pos_weight` multiplies the weight of rows labelled exactly `1`, for
/// every loss in the family rather than only the classification ones — that is
/// where upstream applies it too.
#[derive(Clone, Copy, Debug)]
pub struct RegLossObj {
    loss: Loss,
    scale_pos_weight: f32,
}

/// The squared-error objective at its defaults, kept as a name because it is
/// the crate's default and appears in a lot of call sites.
pub type SquaredError = RegLossObj;

impl Default for RegLossObj {
    fn default() -> Self {
        Self::new(Loss::SquaredError, 1.0)
    }
}

impl RegLossObj {
    pub fn new(loss: Loss, scale_pos_weight: f32) -> Self {
        Self { loss, scale_pos_weight }
    }

    /// Effective weight of row `i`.
    #[inline]
    fn weight(&self, info: &MetaInfo, i: usize, label: f32) -> f32 {
        let w = info.weight(i);
        if label == 1.0 { w * self.scale_pos_weight } else { w }
    }
}

impl Objective for RegLossObj {
    fn name(&self) -> &'static str {
        self.loss.name()
    }

    /// `LinearSquareLoss::Info` is the only one upstream marks constant: its
    /// second derivative is the row weight and nothing else.
    fn has_constant_hessian(&self) -> bool {
        self.loss == Loss::SquaredError
    }

    fn get_gradient(&mut self, preds: &[f32], info: &MetaInfo, _iter: i32, out: &mut Vec<GradientPair>) {
        let n_targets = info.n_targets();
        out.clear();
        out.reserve(preds.len());
        for idx in 0..preds.len() {
            let p = self.loss.pred_transform(preds[idx]);
            let label = info.labels[idx];
            let w = self.weight(info, idx / n_targets, label);
            out.push(GradientPair {
                grad: self.loss.first_order(p, label) * w,
                hess: self.loss.second_order(p, label) * w,
            });
        }
    }

    fn pred_transform(&self, preds: &mut Vec<f32>) {
        for p in preds.iter_mut() {
            *p = self.loss.pred_transform(*p);
        }
    }

    fn prob_to_margin(&self, base_score: &mut [f32]) -> Result<()> {
        for b in base_score.iter_mut() {
            *b = self.loss.prob_to_margin(*b)?;
        }
        Ok(())
    }

    fn init_estimation(&mut self, info: &MetaInfo) -> Vec<f32> {
        // `RegLossObj::InitEstimation`: the weighted label mean is only the
        // right answer while `scale_pos_weight` is 1, because the mean cannot
        // see the extra weight on the positives.
        //
        // `reg:squaredlogerror` is the exception in the family: its link makes
        // the label mean a poor starting point, and upstream fits the intercept
        // from the gradient instead. On the pinned 3.4.0 oracle the label mean
        // gives 1.1359 where XGBoost reports 0.4178, which is exactly the
        // Newton step — see tests/oracle_string_parameters.rs.
        if self.loss == Loss::SquaredLogError || (self.scale_pos_weight - 1.0).abs() > RT_EPS {
            fit_intercept(self, info)
        } else {
            fit_intercept_glm_like(info, self.num_output_group(info))
        }
    }

    fn default_metric(&self) -> String {
        self.loss.default_metric().to_owned()
    }

    fn validate_data(&self, info: &MetaInfo) -> Result<()> {
        self.loss.check_labels(info)
    }
}

// ----------------------------------------------------------- PseudoHuber ----

/// `reg:pseudohubererror`.
#[derive(Clone, Copy, Debug)]
pub struct PseudoHuber {
    slope: f32,
}

impl PseudoHuber {
    pub fn new(slope: f32) -> Self {
        Self { slope }
    }
}

impl Objective for PseudoHuber {
    fn name(&self) -> &'static str {
        "reg:pseudohubererror"
    }

    fn get_gradient(&mut self, preds: &[f32], info: &MetaInfo, _iter: i32, out: &mut Vec<GradientPair>) {
        let n_targets = info.n_targets();
        let slope_sq = self.slope * self.slope;
        out.clear();
        out.reserve(preds.len());
        for idx in 0..preds.len() {
            let z = preds[idx] - info.labels[idx];
            let scale_sqrt = (1.0 + z * z / slope_sq).sqrt();
            let scale = slope_sq + z * z;
            let w = info.weight(idx / n_targets);
            out.push(GradientPair {
                grad: z / scale_sqrt * w,
                hess: slope_sq / (scale * scale_sqrt) * w,
            });
        }
    }

    fn init_estimation(&mut self, info: &MetaInfo) -> Vec<f32> {
        fit_intercept(self, info)
    }

    fn default_metric(&self) -> String {
        "mphe".to_owned()
    }
}

// --------------------------------------------------- MeanAbsoluteError ----

/// Weighted mean of `sqrt(|residual|)`, squared: the automatic smoothing scale
/// `reg:absoluteerror` and `reg:quantileerror` share.
///
/// Returns one scale per output, and `0` when the total weight is zero.
fn smoothing_scale(preds: &[f32], info: &MetaInfo, n_groups: usize, label_col: usize) -> Vec<f32> {
    let mut scale = vec![0.0f32; n_groups];
    let n_targets = info.n_targets();
    let total_weight: f64 = (0..info.num_row).map(|i| info.weight(i) as f64).sum();
    if total_weight <= 0.0 {
        return scale;
    }
    for (t, slot) in scale.iter_mut().enumerate() {
        // `label_col` is the target the residual is measured against: the
        // matching target for MAE, always column 0 for the quantile loss,
        // whose outputs are quantiles of a single target.
        let col = if label_col == usize::MAX { t.min(n_targets - 1) } else { label_col };
        let mut acc = 0.0f64;
        for i in 0..info.num_row {
            let r = preds[i * n_groups + t] - info.label(i, col);
            acc += info.weight(i) as f64 * (r.abs() as f64).sqrt();
        }
        let root_mean = acc / total_weight;
        *slot = (root_mean * root_mean) as f32;
    }
    scale
}

/// `reg:absoluteerror` — the smooth majorisation of the mean absolute error.
///
/// Upstream replaced the adaptive-leaf L1 fit with a per-iteration
/// pseudo-Huber surrogate whose delta is the automatic scale above: the
/// gradient approaches `sign(residual)` as the residuals contract, and the
/// hessian `delta / hypot(delta, residual)` is the majorisation curvature that
/// keeps the IRLS step stable.
#[derive(Clone, Copy, Debug, Default)]
pub struct MeanAbsoluteError;

impl Objective for MeanAbsoluteError {
    fn name(&self) -> &'static str {
        "reg:absoluteerror"
    }

    fn get_gradient(&mut self, preds: &[f32], info: &MetaInfo, _iter: i32, out: &mut Vec<GradientPair>) {
        let n_groups = self.num_output_group(info);
        let scale = smoothing_scale(preds, info, n_groups, usize::MAX);
        out.clear();
        out.resize(preds.len(), GradientPair::default());
        for i in 0..info.num_row {
            let w = info.weight(i);
            for t in 0..n_groups {
                let idx = i * n_groups + t;
                let residual = preds[idx] - info.label(i, t);
                let delta = scale[t];
                let norm = delta.hypot(residual);
                let curvature = if norm > 0.0 { delta / norm } else { 1.0 };
                out[idx] = GradientPair { grad: w * residual * curvature, hess: w * curvature };
            }
        }
    }

    fn init_estimation(&mut self, info: &MetaInfo) -> Vec<f32> {
        // `MeanAbsoluteError::InitEstimation`: a Newton step taken *from the
        // label mean* rather than from zero, then added back to the mean.
        let n_groups = self.num_output_group(info);
        let mean = fit_intercept_glm_like(info, n_groups);
        if info.num_row == 0 {
            return mean;
        }
        let mut preds = Vec::with_capacity(info.num_row * n_groups);
        for _ in 0..info.num_row {
            preds.extend_from_slice(&mean);
        }
        let mut gpair = Vec::new();
        self.get_gradient(&preds, info, 0, &mut gpair);

        let mut out = mean;
        for (t, slot) in out.iter_mut().enumerate() {
            let (mut g, mut h) = (0.0f64, 0.0f64);
            for i in 0..info.num_row {
                let p = gpair[i * n_groups + t];
                g += p.grad as f64;
                h += p.hess as f64;
            }
            if h >= 1e-6 {
                *slot += (-g / h) as f32;
            }
        }
        out
    }

    fn default_metric(&self) -> String {
        "mae".to_owned()
    }
}

// ------------------------------------------------------ QuantileRegression --

/// Bandwidth factor `c` of the logistic smoothing, upstream's
/// `kSmoothingScale`.
const QUANTILE_SMOOTHING_SCALE: f32 = 0.04;
/// Relative curvature floor, upstream's `kMinSurrogateRatio`.
const QUANTILE_MIN_RATIO: f32 = 3.0e-4;

/// `reg:quantileerror` — the logistic-smoothed pinball loss.
///
/// One output per `quantile_alpha`, so a three-quantile fit grows three trees
/// a round and predicts three values a row.
#[derive(Clone, Debug)]
pub struct QuantileRegression {
    alpha: Vec<f32>,
}

impl QuantileRegression {
    pub fn new(alpha: Vec<f32>) -> Result<Self> {
        if alpha.is_empty() {
            return Err(Error::invalid("quantile_alpha", "at least one value is required"));
        }
        Ok(Self { alpha })
    }
}

impl Objective for QuantileRegression {
    fn name(&self) -> &'static str {
        "reg:quantileerror"
    }

    fn num_output_group(&self, _info: &MetaInfo) -> usize {
        self.alpha.len()
    }

    fn get_gradient(&mut self, preds: &[f32], info: &MetaInfo, _iter: i32, out: &mut Vec<GradientPair>) {
        let n_groups = self.alpha.len();
        let scale = smoothing_scale(preds, info, n_groups, 0);
        out.clear();
        out.resize(preds.len(), GradientPair::default());
        for i in 0..info.num_row {
            let w = info.weight(i);
            let y = info.label(i, 0);
            for (t, &alpha) in self.alpha.iter().enumerate() {
                let idx = i * n_groups + t;
                let residual = preds[idx] - y;
                let residual_scale = scale[t];
                if !(residual_scale > 0.0) || w == 0.0 {
                    continue;
                }
                let x = residual / (QUANTILE_SMOOTHING_SCALE * residual_scale);
                let tanh_x = x.tanh();
                let ratio = if x == 0.0 { 1.0 } else { tanh_x / x }.max(QUANTILE_MIN_RATIO);
                out[idx] = GradientPair {
                    grad: w * 0.5 * residual_scale * (tanh_x + 1.0 - 2.0 * alpha),
                    hess: w * 0.5 / QUANTILE_SMOOTHING_SCALE * ratio,
                };
            }
        }
    }

    /// Sort each row's quantiles so they cannot cross, as upstream does.
    fn pred_transform(&self, preds: &mut Vec<f32>) {
        let n = self.alpha.len();
        if n < 2 {
            return;
        }
        for row in preds.chunks_mut(n) {
            row.sort_by(f32::total_cmp);
        }
    }

    fn init_estimation(&mut self, info: &MetaInfo) -> Vec<f32> {
        self.alpha
            .iter()
            .map(|&a| weighted_quantile(a, &info.labels, info.weights.as_deref()))
            .collect()
    }

    fn default_metric(&self) -> String {
        "quantile".to_owned()
    }

    /// The pinball loss weights each output by its own alpha, which the
    /// metric's name does not carry.
    fn make_metric(
        &self,
        spec: &crate::parameters::EvalMetric,
    ) -> Result<Box<dyn crate::metric::Metric>> {
        match spec {
            crate::parameters::EvalMetric::Quantile => Ok(Box::new(
                crate::metric::elementwise::PinballLoss::quantile()
                    .with_alpha(self.alpha.clone()),
            )),
            other => crate::metric::create_from(other),
        }
    }
}

// ----------------------------------------------------- ExpectileRegression --

/// `common::SoftPlus`.
#[inline]
fn soft_plus(x: f32) -> f32 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// `common::SoftPlusInv`.
#[inline]
fn soft_plus_inv(x: f32) -> f32 {
    // log(exp(x) - 1), computed so a small `x` does not lose every digit.
    if x > 20.0 { x } else { (x.exp() - 1.0).max(f32::MIN_POSITIVE).ln() }
}

/// `reg:expectileerror` — asymmetric least squares at several expectiles.
///
/// The outputs after the first are *increments* passed through a softplus, so
/// the reported expectiles cannot cross whatever the trees learn. That
/// reparameterisation is why the gradient of output `j` accumulates every
/// expectile from `j` upwards.
#[derive(Clone, Debug)]
pub struct ExpectileRegression {
    alpha: Vec<f32>,
}

impl ExpectileRegression {
    pub fn new(alpha: Vec<f32>) -> Result<Self> {
        if alpha.is_empty() {
            return Err(Error::invalid("expectile_alpha", "at least one value is required"));
        }
        Ok(Self { alpha })
    }
}

impl Objective for ExpectileRegression {
    fn name(&self) -> &'static str {
        "reg:expectileerror"
    }

    fn num_output_group(&self, _info: &MetaInfo) -> usize {
        self.alpha.len()
    }

    fn get_gradient(&mut self, preds: &[f32], info: &MetaInfo, _iter: i32, out: &mut Vec<GradientPair>) {
        let n = self.alpha.len();
        out.clear();
        out.resize(preds.len(), GradientPair::default());
        for i in 0..info.num_row {
            let label = info.label(i, 0);
            let w = info.weight(i);
            for j in 0..n {
                let mut pred = preds[i * n];
                let (mut grad_sum, mut hess_sum) = (0.0f32, 0.0f32);
                for k in 0..n {
                    if k > 0 {
                        pred += RT_EPS + soft_plus(preds[i * n + k]);
                    }
                    if k >= j {
                        let diff = pred - label;
                        let weight_scale =
                            if diff >= 0.0 { 1.0 - self.alpha[k] } else { self.alpha[k] };
                        grad_sum += weight_scale * diff * w;
                        hess_sum += weight_scale * w;
                    }
                }
                let scale = if j == 0 { 1.0 } else { sigmoid(preds[i * n + j]) };
                out[i * n + j] =
                    GradientPair { grad: scale * grad_sum, hess: scale * scale * hess_sum };
            }
        }
    }

    fn pred_transform(&self, preds: &mut Vec<f32>) {
        let n = self.alpha.len();
        if n < 2 {
            return;
        }
        for row in preds.chunks_mut(n) {
            let mut pred = row[0];
            for j in 1..n {
                pred += RT_EPS + soft_plus(row[j]);
                row[j] = pred;
            }
        }
    }

    fn prob_to_margin(&self, base_score: &mut [f32]) -> Result<()> {
        for j in (1..base_score.len()).rev() {
            let gap = base_score[j] - base_score[j - 1];
            base_score[j] = soft_plus_inv(gap - RT_EPS);
        }
        Ok(())
    }

    fn init_estimation(&mut self, info: &MetaInfo) -> Vec<f32> {
        let n = self.alpha.len();
        let mean = fit_intercept_glm_like(info, 1)[0];
        let mut out = vec![mean; n];
        if info.num_row == 0 {
            return out;
        }
        // One Newton step per expectile, taken at the label mean.
        for (j, slot) in out.iter_mut().enumerate() {
            let (mut g, mut h) = (0.0f64, 0.0f64);
            for i in 0..info.num_row {
                let diff = mean - info.label(i, 0);
                let weight_scale = if diff >= 0.0 { 1.0 - self.alpha[j] } else { self.alpha[j] };
                let w = info.weight(i);
                g += (weight_scale * diff * w) as f64;
                h += (weight_scale * w) as f64;
            }
            if h >= 1e-6 {
                *slot += (-g / h) as f32;
            }
        }
        // The expectiles are reported in ascending order.
        for j in 1..n {
            out[j] = out[j].max(out[j - 1]);
        }
        out
    }

    fn default_metric(&self) -> String {
        "expectile".to_owned()
    }

    /// As for the quantile loss, the metric needs the objective's alphas.
    fn make_metric(
        &self,
        spec: &crate::parameters::EvalMetric,
    ) -> Result<Box<dyn crate::metric::Metric>> {
        match spec {
            crate::parameters::EvalMetric::Expectile => Ok(Box::new(
                crate::metric::elementwise::PinballLoss::expectile()
                    .with_alpha(self.alpha.clone()),
            )),
            other => crate::metric::create_from(other),
        }
    }
}

// --------------------------------------------------------- the log links ----

/// `reg:gamma` — gamma deviance with a log link.
#[derive(Clone, Copy, Debug, Default)]
pub struct GammaRegression;

impl Objective for GammaRegression {
    fn name(&self) -> &'static str {
        "reg:gamma"
    }

    fn get_gradient(&mut self, preds: &[f32], info: &MetaInfo, _iter: i32, out: &mut Vec<GradientPair>) {
        let n_targets = info.n_targets();
        out.clear();
        out.reserve(preds.len());
        for idx in 0..preds.len() {
            let p = preds[idx].exp();
            let y = info.labels[idx];
            let w = info.weight(idx / n_targets);
            out.push(GradientPair { grad: (1.0 - y / p) * w, hess: (y / p) * w });
        }
    }

    fn pred_transform(&self, preds: &mut Vec<f32>) {
        for p in preds.iter_mut() {
            *p = p.exp();
        }
    }

    fn prob_to_margin(&self, base_score: &mut [f32]) -> Result<()> {
        log_link_margin("reg:gamma", base_score)
    }

    fn init_estimation(&mut self, info: &MetaInfo) -> Vec<f32> {
        fit_intercept_glm_like(info, self.num_output_group(info))
    }

    fn default_metric(&self) -> String {
        "gamma-deviance".to_owned()
    }

    fn validate_data(&self, info: &MetaInfo) -> Result<()> {
        check_labels(self.name(), info, |y| y > 0.0, "label must be positive for gamma regression")
    }
}

/// `count:poisson` — Poisson regression with a log link.
///
/// `max_delta_step` is the objective's own, defaulting to `0.7` rather than the
/// tree booster's `0`: it inflates the hessian, which caps the leaf weights and
/// keeps `exp` from overflowing.
#[derive(Clone, Copy, Debug)]
pub struct PoissonRegression {
    max_delta_step: f32,
}

impl PoissonRegression {
    pub fn new(max_delta_step: f32) -> Self {
        Self { max_delta_step }
    }
}

impl Objective for PoissonRegression {
    fn name(&self) -> &'static str {
        "count:poisson"
    }

    fn get_gradient(&mut self, preds: &[f32], info: &MetaInfo, _iter: i32, out: &mut Vec<GradientPair>) {
        let n_targets = info.n_targets();
        out.clear();
        out.reserve(preds.len());
        for idx in 0..preds.len() {
            let p = preds[idx];
            let y = info.labels[idx];
            let w = info.weight(idx / n_targets);
            out.push(GradientPair {
                grad: (p.exp() - y) * w,
                hess: (p + self.max_delta_step).exp() * w,
            });
        }
    }

    fn pred_transform(&self, preds: &mut Vec<f32>) {
        for p in preds.iter_mut() {
            *p = p.exp();
        }
    }

    fn prob_to_margin(&self, base_score: &mut [f32]) -> Result<()> {
        log_link_margin("count:poisson", base_score)
    }

    fn init_estimation(&mut self, info: &MetaInfo) -> Vec<f32> {
        fit_intercept_glm_like(info, self.num_output_group(info))
    }

    fn default_metric(&self) -> String {
        "poisson-nloglik".to_owned()
    }

    fn validate_data(&self, info: &MetaInfo) -> Result<()> {
        check_labels(
            self.name(),
            info,
            |y| y >= 0.0,
            "label must be non-negative for Poisson regression",
        )
    }
}

/// `reg:tweedie` — Tweedie regression with a log link.
#[derive(Clone, Copy, Debug)]
pub struct TweedieRegression {
    rho: f32,
}

impl TweedieRegression {
    pub fn new(rho: f32) -> Self {
        Self { rho }
    }
}

impl Objective for TweedieRegression {
    fn name(&self) -> &'static str {
        "reg:tweedie"
    }

    fn get_gradient(&mut self, preds: &[f32], info: &MetaInfo, _iter: i32, out: &mut Vec<GradientPair>) {
        let n_targets = info.n_targets();
        let rho = self.rho;
        out.clear();
        out.reserve(preds.len());
        for idx in 0..preds.len() {
            let p = preds[idx];
            let y = info.labels[idx];
            let w = info.weight(idx / n_targets);
            let grad = -y * ((1.0 - rho) * p).exp() + ((2.0 - rho) * p).exp();
            let hess =
                -y * (1.0 - rho) * ((1.0 - rho) * p).exp() + (2.0 - rho) * ((2.0 - rho) * p).exp();
            out.push(GradientPair { grad: grad * w, hess: hess * w });
        }
    }

    fn pred_transform(&self, preds: &mut Vec<f32>) {
        for p in preds.iter_mut() {
            *p = p.exp();
        }
    }

    fn prob_to_margin(&self, base_score: &mut [f32]) -> Result<()> {
        log_link_margin("reg:tweedie", base_score)
    }

    fn init_estimation(&mut self, info: &MetaInfo) -> Vec<f32> {
        fit_intercept_glm_like(info, self.num_output_group(info))
    }

    /// The variance power is part of the metric's name, so a Tweedie fit
    /// reports `tweedie-nloglik@1.5` rather than an unparameterised name.
    fn default_metric(&self) -> String {
        format!("tweedie-nloglik@{}", self.rho)
    }

    fn validate_data(&self, info: &MetaInfo) -> Result<()> {
        check_labels(
            self.name(),
            info,
            |y| y >= 0.0,
            "label must be non-negative for Tweedie regression",
        )
    }
}

/// `ProbToMargin` for the log-link objectives, with upstream's positivity check.
fn log_link_margin(name: &'static str, base_score: &mut [f32]) -> Result<()> {
    for b in base_score.iter_mut() {
        if *b <= 0.0 {
            return Err(Error::invalid(
                "base_score",
                format!("`base_score` must be greater than 0 for `{name}`, got {b}"),
            ));
        }
        *b = b.ln();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(labels: &[f32], weights: Option<&[f32]>) -> MetaInfo {
        MetaInfo {
            num_row: labels.len(),
            num_col: 1,
            labels: labels.to_vec(),
            num_target: 1,
            weights: weights.map(|w| w.to_vec()),
            ..Default::default()
        }
    }

    fn gradient(obj: &mut dyn Objective, preds: &[f32], info: &MetaInfo) -> Vec<GradientPair> {
        let mut out = Vec::new();
        obj.get_gradient(preds, info, 0, &mut out);
        out
    }

    #[test]
    fn squared_error_gradient_is_the_residual() {
        let mut obj = RegLossObj::default();
        let g = gradient(&mut obj, &[1.0, 2.0], &info(&[0.5, 3.0], None));
        assert_eq!(g, vec![
            GradientPair { grad: 0.5, hess: 1.0 },
            GradientPair { grad: -1.0, hess: 1.0 },
        ]);
    }

    #[test]
    fn scale_pos_weight_only_touches_rows_labelled_one() {
        let mut obj = RegLossObj::new(Loss::SquaredError, 3.0);
        let g = gradient(&mut obj, &[0.0, 0.0], &info(&[1.0, 2.0], None));
        assert_eq!(g, vec![
            GradientPair { grad: -3.0, hess: 3.0 },
            GradientPair { grad: -2.0, hess: 1.0 },
        ]);
    }

    #[test]
    fn logistic_transforms_the_margin_and_the_intercept() {
        let mut obj = RegLossObj::new(Loss::BinaryLogistic, 1.0);
        let mut preds = vec![0.0f32];
        obj.pred_transform(&mut preds);
        assert_eq!(preds, vec![0.5]);

        // Half the labels positive: the intercept is 0.5 in probability space
        // and 0 in margin space.
        let intercept = obj.init_estimation(&info(&[0.0, 1.0], None));
        assert_eq!(intercept, vec![0.5]);
        let mut margin = intercept;
        obj.prob_to_margin(&mut margin).unwrap();
        assert!(margin[0].abs() < 1e-6, "logit(0.5) is 0, got {}", margin[0]);

        assert!(obj.prob_to_margin(&mut [1.5]).is_err(), "a probability above 1 is rejected");
    }

    #[test]
    fn the_log_link_objectives_reject_a_non_positive_intercept() {
        for obj in [
            Box::new(GammaRegression) as Box<dyn Objective>,
            Box::new(PoissonRegression::new(0.7)),
            Box::new(TweedieRegression::new(1.5)),
        ] {
            assert!(obj.prob_to_margin(&mut [0.0]).is_err(), "{}", obj.name());
            let mut b = [std::f32::consts::E];
            obj.prob_to_margin(&mut b).unwrap();
            assert!((b[0] - 1.0).abs() < 1e-6, "{} gave {}", obj.name(), b[0]);
        }
    }

    #[test]
    fn label_ranges_are_checked_per_objective() {
        assert!(GammaRegression.validate_data(&info(&[0.0], None)).is_err());
        assert!(GammaRegression.validate_data(&info(&[1.0], None)).is_ok());
        assert!(PoissonRegression::new(0.7).validate_data(&info(&[-1.0], None)).is_err());
        assert!(
            RegLossObj::new(Loss::SquaredLogError, 1.0).validate_data(&info(&[-2.0], None)).is_err()
        );
        assert!(RegLossObj::new(Loss::Logistic, 1.0).validate_data(&info(&[1.5], None)).is_err());
    }

    #[test]
    fn tweedie_names_its_metric_after_the_variance_power() {
        assert_eq!(TweedieRegression::new(1.2).default_metric(), "tweedie-nloglik@1.2");
    }

    #[test]
    fn pseudo_huber_gradient_saturates_at_the_slope() {
        let mut obj = PseudoHuber::new(1.0);
        // A huge residual: the gradient tends to +-1 rather than growing.
        let g = gradient(&mut obj, &[1000.0], &info(&[0.0], None));
        assert!((g[0].grad - 1.0).abs() < 1e-3, "{:?}", g[0]);
        assert!(g[0].hess > 0.0 && g[0].hess < 1e-5);
    }

    #[test]
    fn absolute_error_gradient_tracks_the_sign_of_the_residual() {
        let mut obj = MeanAbsoluteError;
        let d = info(&[0.0, 0.0, 0.0, 0.0], None);
        let g = gradient(&mut obj, &[-2.0, -1.0, 1.0, 2.0], &d);
        assert!(g[0].grad < 0.0 && g[1].grad < 0.0);
        assert!(g[2].grad > 0.0 && g[3].grad > 0.0);
        assert!(g.iter().all(|p| p.hess > 0.0), "the majorisation curvature is positive");
    }

    #[test]
    fn quantile_regression_has_one_output_per_alpha() {
        let mut obj = QuantileRegression::new(vec![0.1, 0.5, 0.9]).unwrap();
        let d = info(&[1.0, 2.0, 3.0, 4.0], None);
        assert_eq!(obj.num_output_group(&d), 3);
        // The intercept is the label quantile at each alpha, ascending.
        let intercept = obj.init_estimation(&d);
        assert_eq!(intercept.len(), 3);
        assert!(intercept[0] <= intercept[1] && intercept[1] <= intercept[2], "{intercept:?}");

        // The transform sorts each row so quantiles cannot cross.
        let mut preds = vec![3.0f32, 1.0, 2.0];
        obj.pred_transform(&mut preds);
        assert_eq!(preds, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn quantile_gradient_leans_the_way_alpha_asks() {
        // At a zero residual scale everything is zero, so use real residuals.
        let d = info(&[0.0; 8], None);
        let preds: Vec<f32> = (0..8).map(|i| i as f32 - 4.0).collect();

        let mut low = QuantileRegression::new(vec![0.1]).unwrap();
        let mut high = QuantileRegression::new(vec![0.9]).unwrap();
        let gl: f32 = gradient(&mut low, &preds, &d).iter().map(|p| p.grad).sum();
        let gh: f32 = gradient(&mut high, &preds, &d).iter().map(|p| p.grad).sum();
        assert!(gl > gh, "a low quantile pushes predictions down harder: {gl} vs {gh}");
    }

    #[test]
    fn expectiles_are_reported_in_ascending_order() {
        let obj = ExpectileRegression::new(vec![0.2, 0.5, 0.8]).unwrap();
        let mut preds = vec![1.0f32, 0.0, 0.0];
        obj.pred_transform(&mut preds);
        assert!(preds[0] < preds[1] && preds[1] < preds[2], "{preds:?}");
    }

    #[test]
    fn the_expectile_margin_round_trips_through_the_reparameterisation() {
        let obj = ExpectileRegression::new(vec![0.2, 0.5, 0.8]).unwrap();
        let mut margin = vec![1.0f32, 2.0, 4.0];
        obj.prob_to_margin(&mut margin).unwrap();
        let mut back = margin;
        obj.pred_transform(&mut back);
        for (got, want) in back.iter().zip([1.0f32, 2.0, 4.0]) {
            assert!((got - want).abs() < 1e-4, "{back:?}");
        }
    }
}
