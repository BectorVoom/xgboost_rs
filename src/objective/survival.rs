//! The survival objectives.
//!
//! Ports of `src/objective/regression_obj.cu` (`CoxRegression`),
//! `src/objective/aft_obj.cu` and `src/common/survival_util.h`. Everything here
//! accumulates in `f64`: the AFT gradients divide small densities by small
//! probabilities, and Cox's partial likelihood sums `exp(margin)` over the
//! whole risk set.

use super::{GradientPair, Objective, fit_intercept_glm_like};
use crate::data::MetaInfo;
use crate::parameters::AftDistribution;
use crate::{Error, Result};

// --------------------------------------------------------------- Cox ------

/// `survival:cox` — Cox proportional hazards on the partial likelihood.
///
/// Labels carry the censoring in their sign: a negative label is a
/// right-censored observation at time `|label|`. Rows are processed in
/// ascending `|label|` order, and ties share a risk set (Breslow's method).
#[derive(Clone, Copy, Debug, Default)]
pub struct CoxRegression;

impl Objective for CoxRegression {
    fn name(&self) -> &'static str {
        "survival:cox"
    }

    fn num_output_group(&self, _info: &MetaInfo) -> usize {
        1
    }

    fn get_gradient(&self, preds: &[f32], info: &MetaInfo, _iter: i32, out: &mut Vec<GradientPair>) {
        out.clear();
        out.resize(preds.len(), GradientPair::default());
        if preds.is_empty() {
            return;
        }
        // `MetaInfo::LabelAbsSort`.
        let mut order: Vec<usize> = (0..preds.len()).collect();
        order.sort_by(|&a, &b| info.labels[a].abs().total_cmp(&info.labels[b].abs()));

        let mut exp_p_sum: f64 = order.iter().map(|&i| (preds[i] as f64).exp()).sum();

        let (mut r_k, mut s_k) = (0.0f64, 0.0f64);
        let (mut last_exp_p, mut last_abs_y, mut accumulated_sum) = (0.0f64, 0.0f64, 0.0f64);
        for &ind in &order {
            let exp_p = (preds[ind] as f64).exp();
            let w = info.weight(ind) as f64;
            let y = info.labels[ind] as f64;
            let abs_y = y.abs();

            // The denominator only shrinks once time moves forward, so ties
            // share one risk set.
            accumulated_sum += last_exp_p;
            if last_abs_y < abs_y {
                exp_p_sum -= accumulated_sum;
                accumulated_sum = 0.0;
            }

            if y > 0.0 {
                r_k += 1.0 / exp_p_sum;
                s_k += 1.0 / (exp_p_sum * exp_p_sum);
            }

            let grad = exp_p * r_k - f64::from(y > 0.0);
            let hess = exp_p * r_k - exp_p * exp_p * s_k;
            out[ind] = GradientPair { grad: (grad * w) as f32, hess: (hess * w) as f32 };

            last_abs_y = abs_y;
            last_exp_p = exp_p;
        }
    }

    fn pred_transform(&self, preds: &mut Vec<f32>) {
        for p in preds.iter_mut() {
            *p = p.exp();
        }
    }

    fn prob_to_margin(&self, base_score: &mut [f32]) -> Result<()> {
        for b in base_score.iter_mut() {
            if *b <= 0.0 {
                return Err(Error::invalid(
                    "base_score",
                    format!("`base_score` must be greater than 0 for `survival:cox`, got {b}"),
                ));
            }
            *b = b.ln();
        }
        Ok(())
    }

    fn init_estimation(&self, info: &MetaInfo) -> Vec<f32> {
        // Cox has no closed-form intercept upstream: the hazard ratio is
        // relative, so the fit starts from the neutral ratio of 1.
        let _ = info;
        vec![1.0]
    }

    fn default_metric(&self) -> String {
        "cox-nloglik".to_owned()
    }
}

// --------------------------------------------------------------- AFT ------

/// Allowable gradient/hessian range, `aft::kMinGradient` and friends.
const MIN_GRADIENT: f64 = -15.0;
const MAX_GRADIENT: f64 = 15.0;
const MIN_HESSIAN: f64 = 1e-16;
const MAX_HESSIAN: f64 = 15.0;
/// `aft::kEps` — the floor a denominator is allowed to reach.
const AFT_EPS: f64 = 1e-12;

/// How a row is censored, `xgboost::common::CensoringType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Censoring {
    Uncensored,
    RightCensored,
    LeftCensored,
    IntervalCensored,
}

/// The noise distributions of the AFT model.
impl AftDistribution {
    fn pdf(self, z: f64) -> f64 {
        match self {
            Self::Normal => (-z * z / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt(),
            Self::Logistic => {
                let w = z.exp();
                if w.is_infinite() || (w * w).is_infinite() { 0.0 } else { w / ((1.0 + w) * (1.0 + w)) }
            }
            Self::Extreme => {
                let w = z.exp();
                if w.is_infinite() { 0.0 } else { w * (-w).exp() }
            }
        }
    }

    fn cdf(self, z: f64) -> f64 {
        match self {
            Self::Normal => 0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2)),
            Self::Logistic => {
                let w = z.exp();
                if w.is_infinite() { 1.0 } else { w / (1.0 + w) }
            }
            Self::Extreme => 1.0 - (-z.exp()).exp(),
        }
    }

    fn grad_pdf(self, z: f64) -> f64 {
        match self {
            Self::Normal => -z * self.pdf(z),
            Self::Logistic => {
                let w = z.exp();
                if w.is_infinite() { 0.0 } else { self.pdf(z) * (1.0 - w) / (1.0 + w) }
            }
            Self::Extreme => {
                let w = z.exp();
                if w.is_infinite() { 0.0 } else { (1.0 - w) * self.pdf(z) }
            }
        }
    }

    fn hess_pdf(self, z: f64) -> f64 {
        match self {
            Self::Normal => (z * z - 1.0) * self.pdf(z),
            Self::Logistic => {
                let w = z.exp();
                if w.is_infinite() || (w * w).is_infinite() {
                    0.0
                } else {
                    self.pdf(z) * (w * w - 4.0 * w + 1.0) / ((1.0 + w) * (1.0 + w))
                }
            }
            Self::Extreme => {
                let w = z.exp();
                if w.is_infinite() || (w * w).is_infinite() {
                    0.0
                } else {
                    (w * w - 3.0 * w + 1.0) * self.pdf(z)
                }
            }
        }
    }

    /// `aft::GetLimitGradAtInfPred` — the limit the gradient takes when the
    /// denominator underflows.
    fn limit_grad(self, censor: Censoring, sign: bool, sigma: f64) -> f64 {
        use Censoring::*;
        match (self, censor) {
            (Self::Normal, Uncensored) | (Self::Normal, IntervalCensored) => {
                if sign { MIN_GRADIENT } else { MAX_GRADIENT }
            }
            (Self::Normal, RightCensored) => {
                if sign { MIN_GRADIENT } else { 0.0 }
            }
            (Self::Normal, LeftCensored) => {
                if sign { 0.0 } else { MAX_GRADIENT }
            }
            (Self::Logistic, Uncensored) | (Self::Logistic, IntervalCensored) => {
                if sign { -1.0 / sigma } else { 1.0 / sigma }
            }
            (Self::Logistic, RightCensored) => {
                if sign { -1.0 / sigma } else { 0.0 }
            }
            (Self::Logistic, LeftCensored) => {
                if sign { 0.0 } else { 1.0 / sigma }
            }
            (Self::Extreme, Uncensored) | (Self::Extreme, IntervalCensored) => {
                if sign { MIN_GRADIENT } else { 1.0 / sigma }
            }
            (Self::Extreme, RightCensored) => {
                if sign { MIN_GRADIENT } else { 0.0 }
            }
            (Self::Extreme, LeftCensored) => {
                if sign { 0.0 } else { 1.0 / sigma }
            }
        }
    }

    /// `aft::GetLimitHessAtInfPred`.
    fn limit_hess(self, censor: Censoring, sign: bool, sigma: f64) -> f64 {
        use Censoring::*;
        match (self, censor) {
            (Self::Normal, Uncensored) | (Self::Normal, IntervalCensored) => 1.0 / (sigma * sigma),
            (Self::Normal, RightCensored) => {
                if sign { 1.0 / (sigma * sigma) } else { MIN_HESSIAN }
            }
            (Self::Normal, LeftCensored) => {
                if sign { MIN_HESSIAN } else { 1.0 / (sigma * sigma) }
            }
            (Self::Logistic, _) => MIN_HESSIAN,
            (Self::Extreme, Uncensored) | (Self::Extreme, RightCensored) => {
                if sign { MAX_HESSIAN } else { MIN_HESSIAN }
            }
            (Self::Extreme, LeftCensored) => MIN_HESSIAN,
            (Self::Extreme, IntervalCensored) => {
                if sign { MAX_HESSIAN } else { MIN_HESSIAN }
            }
        }
    }
}

/// Error function, needed by the normal distribution's CDF.
///
/// Abramowitz & Stegun 7.1.26, whose `1.5e-7` worst-case error is far below the
/// `f32` the gradients are stored in.
pub(crate) fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    const A: [f64; 5] = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429];
    const P: f64 = 0.3275911;
    let t = 1.0 / (1.0 + P * x);
    let poly = A.iter().rev().fold(0.0, |acc, a| (acc + a) * t);
    sign * (1.0 - poly * (-x * x).exp())
}

/// The AFT loss for one row: negative log likelihood, its gradient and its
/// hessian, sharing the censoring analysis.
struct AftTerms {
    grad: f64,
    hess: f64,
}

fn aft_terms(dist: AftDistribution, y_lower: f64, y_upper: f64, y_pred: f64, sigma: f64) -> AftTerms {
    let (grad_num, grad_den, hess_num, hess_den, censor, z_sign);

    if y_lower == y_upper {
        // Uncensored.
        let z = (y_lower.ln() - y_pred) / sigma;
        let pdf = dist.pdf(z);
        let grad_pdf = dist.grad_pdf(z);
        let hess_pdf = dist.hess_pdf(z);
        grad_num = grad_pdf;
        grad_den = sigma * pdf;
        hess_num = -(pdf * hess_pdf - grad_pdf * grad_pdf);
        hess_den = sigma * sigma * pdf * pdf;
        censor = Censoring::Uncensored;
        z_sign = z > 0.0;
    } else {
        let (mut z_u, mut z_l) = (0.0f64, 0.0f64);
        let (pdf_u, cdf_u, grad_pdf_u);
        let (pdf_l, cdf_l, grad_pdf_l);
        let mut kind = Censoring::IntervalCensored;

        if y_upper.is_infinite() {
            pdf_u = 0.0;
            cdf_u = 1.0;
            grad_pdf_u = 0.0;
            kind = Censoring::RightCensored;
        } else {
            z_u = (y_upper.ln() - y_pred) / sigma;
            pdf_u = dist.pdf(z_u);
            cdf_u = dist.cdf(z_u);
            grad_pdf_u = dist.grad_pdf(z_u);
        }
        if y_lower <= 0.0 {
            pdf_l = 0.0;
            cdf_l = 0.0;
            grad_pdf_l = 0.0;
            kind = Censoring::LeftCensored;
        } else {
            z_l = (y_lower.ln() - y_pred) / sigma;
            pdf_l = dist.pdf(z_l);
            cdf_l = dist.cdf(z_l);
            grad_pdf_l = dist.grad_pdf(z_l);
        }

        let cdf_diff = cdf_u - cdf_l;
        let pdf_diff = pdf_u - pdf_l;
        let grad_diff = grad_pdf_u - grad_pdf_l;
        let sqrt_den = sigma * cdf_diff;

        grad_num = pdf_diff;
        grad_den = sqrt_den;
        hess_num = -(cdf_diff * grad_diff - pdf_diff * pdf_diff);
        hess_den = sqrt_den * sqrt_den;
        censor = kind;
        z_sign = z_u > 0.0 || z_l > 0.0;
    }

    let mut grad = grad_num / grad_den;
    if grad_den < AFT_EPS && (grad.is_nan() || grad.is_infinite()) {
        grad = dist.limit_grad(censor, z_sign, sigma);
    }
    let mut hess = hess_num / hess_den;
    if hess_den < AFT_EPS && (hess.is_nan() || hess.is_infinite()) {
        hess = dist.limit_hess(censor, z_sign, sigma);
    }
    AftTerms {
        grad: grad.clamp(MIN_GRADIENT, MAX_GRADIENT),
        hess: hess.clamp(MIN_HESSIAN, MAX_HESSIAN),
    }
}

/// Negative log likelihood of one row, `AFTLoss::Loss`. Shared with the
/// `aft-nloglik` metric.
pub(crate) fn aft_loss(
    dist: AftDistribution,
    y_lower: f64,
    y_upper: f64,
    y_pred: f64,
    sigma: f64,
) -> f64 {
    if y_lower == y_upper {
        let z = (y_lower.ln() - y_pred) / sigma;
        let pdf = dist.pdf(z);
        -(pdf / (sigma * y_lower)).max(AFT_EPS).ln()
    } else {
        let cdf_u = if y_upper.is_infinite() { 1.0 } else { dist.cdf((y_upper.ln() - y_pred) / sigma) };
        let cdf_l = if y_lower <= 0.0 { 0.0 } else { dist.cdf((y_lower.ln() - y_pred) / sigma) };
        -(cdf_u - cdf_l).max(AFT_EPS).ln()
    }
}

/// `survival:aft` — the accelerated failure time model.
///
/// Reads the censoring interval from `label_lower_bound`/`label_upper_bound`
/// rather than from the label, which is what lets one row be uncensored,
/// another right-censored and a third interval-censored.
#[derive(Clone, Copy, Debug)]
pub struct AftSurvival {
    dist: AftDistribution,
    sigma: f32,
}

impl AftSurvival {
    pub fn new(dist: AftDistribution, sigma: f32) -> Self {
        Self { dist, sigma }
    }

}

impl Objective for AftSurvival {
    fn name(&self) -> &'static str {
        "survival:aft"
    }

    fn num_output_group(&self, _info: &MetaInfo) -> usize {
        1
    }

    fn get_gradient(&self, preds: &[f32], info: &MetaInfo, _iter: i32, out: &mut Vec<GradientPair>) {
        out.clear();
        out.reserve(preds.len());
        let sigma = self.sigma as f64;
        for i in 0..preds.len() {
            let terms = aft_terms(
                self.dist,
                info.lower_bound(i) as f64,
                info.upper_bound(i) as f64,
                preds[i] as f64,
                sigma,
            );
            let w = info.weight(i);
            out.push(GradientPair {
                grad: terms.grad as f32 * w,
                hess: terms.hess as f32 * w,
            });
        }
    }

    /// Trees predict in log time, so a prediction is exponentiated.
    fn pred_transform(&self, preds: &mut Vec<f32>) {
        for p in preds.iter_mut() {
            *p = p.exp();
        }
    }

    /// The AFT metric wants the untransformed margin.
    fn eval_transform(&self, _preds: &mut Vec<f32>) {}

    fn prob_to_margin(&self, base_score: &mut [f32]) -> Result<()> {
        for b in base_score.iter_mut() {
            if *b <= 0.0 {
                return Err(Error::invalid(
                    "base_score",
                    format!("`base_score` must be greater than 0 for `survival:aft`, got {b}"),
                ));
            }
            *b = b.ln();
        }
        Ok(())
    }

    fn init_estimation(&self, info: &MetaInfo) -> Vec<f32> {
        // The label mean, on the time scale the intercept is stored in. A
        // censored row contributes its finite bound.
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for i in 0..info.num_row {
            let (lo, hi) = (info.lower_bound(i), info.upper_bound(i));
            let y = if hi.is_finite() { (lo + hi) / 2.0 } else { lo };
            if y > 0.0 {
                let w = info.weight(i) as f64;
                num += y as f64 * w;
                den += w;
            }
        }
        if den == 0.0 {
            return fit_intercept_glm_like(info, 1);
        }
        vec![(num / den) as f32]
    }

    fn default_metric(&self) -> String {
        "aft-nloglik".to_owned()
    }

    /// The AFT metric needs the same distribution and scale the loss uses; the
    /// metric's name cannot carry them.
    fn make_metric(
        &self,
        spec: &crate::parameters::EvalMetric,
    ) -> Result<Box<dyn crate::metric::Metric>> {
        match spec {
            crate::parameters::EvalMetric::AftNegLogLik => {
                Ok(Box::new(crate::metric::survival::AftNegLogLik::new(self.dist, self.sigma)))
            }
            other => crate::metric::create_from(other),
        }
    }

    fn validate_data(&self, info: &MetaInfo) -> Result<()> {
        if info.label_lower_bound.is_none() || info.label_upper_bound.is_none() {
            return Err(Error::invalid(
                "objective",
                "`survival:aft` needs the censoring interval; set it with \
                 `DMatrix::set_label_bounds`",
            ));
        }
        for i in 0..info.num_row {
            if info.lower_bound(i) < 0.0 {
                return Err(Error::invalid(
                    "label_lower_bound",
                    format!("`survival:aft` needs non-negative times, row {i} has a negative one"),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn survival_info(lower: &[f32], upper: &[f32]) -> MetaInfo {
        MetaInfo {
            num_row: lower.len(),
            num_col: 1,
            labels: lower.to_vec(),
            num_target: 1,
            label_lower_bound: Some(lower.to_vec()),
            label_upper_bound: Some(upper.to_vec()),
            ..Default::default()
        }
    }

    /// The approximation's worst-case error is 1.5e-7, far below the `f32` the
    /// gradients end up in, so that is the tolerance asserted here.
    #[test]
    fn erf_matches_known_values() {
        assert!(erf(0.0).abs() < 1.5e-7, "{}", erf(0.0));
        assert!((erf(1.0) - 0.842_700_79).abs() < 1.5e-7, "{}", erf(1.0));
        assert!((erf(-1.0) + 0.842_700_79).abs() < 1.5e-7);
        assert!((erf(3.0) - 0.999_977_9).abs() < 1.5e-7);
        assert!(erf(-2.0) < 0.0 && erf(2.0) > 0.0, "erf is odd");
    }

    #[test]
    fn every_aft_distribution_has_a_normalised_cdf() {
        for dist in [AftDistribution::Normal, AftDistribution::Logistic, AftDistribution::Extreme] {
            assert!(dist.cdf(-20.0) < 1e-6, "{dist}");
            assert!(dist.cdf(20.0) > 1.0 - 1e-6, "{dist}");
            assert!(dist.pdf(0.0) > 0.0, "{dist}");
            // The CDF rises everywhere.
            assert!(dist.cdf(0.5) > dist.cdf(-0.5), "{dist}");
        }
    }

    #[test]
    fn aft_pushes_an_underestimate_up_for_every_distribution() {
        for dist in [AftDistribution::Normal, AftDistribution::Logistic, AftDistribution::Extreme] {
            let obj = AftSurvival::new(dist, 1.0);
            let mut out = Vec::new();
            // True time 100, predicted margin 0 (time 1): far too low.
            obj.get_gradient(&[0.0], &survival_info(&[100.0], &[100.0]), 0, &mut out);
            assert!(out[0].grad < 0.0, "{dist}: {:?}", out[0]);
            assert!(out[0].hess > 0.0, "{dist}");
        }
    }

    #[test]
    fn a_right_censored_row_only_pushes_upwards() {
        let obj = AftSurvival::new(AftDistribution::Normal, 1.0);
        let mut out = Vec::new();
        // Survived past 100: predictions below it are penalised, above are not.
        let info = survival_info(&[100.0], &[f32::INFINITY]);
        obj.get_gradient(&[0.0], &info, 0, &mut out);
        assert!(out[0].grad < 0.0, "{:?}", out[0]);
        obj.get_gradient(&[20.0], &info, 0, &mut out);
        assert!(out[0].grad.abs() < 1e-3, "a large enough prediction is free: {:?}", out[0]);
    }

    #[test]
    fn aft_needs_its_censoring_interval() {
        let obj = AftSurvival::new(AftDistribution::Normal, 1.0);
        let bare = MetaInfo { num_row: 1, num_col: 1, labels: vec![1.0], num_target: 1, ..Default::default() };
        assert!(obj.validate_data(&bare).is_err());
        assert!(obj.validate_data(&survival_info(&[1.0], &[2.0])).is_ok());
    }

    #[test]
    fn aft_reports_time_but_evaluates_on_the_margin() {
        let obj = AftSurvival::new(AftDistribution::Normal, 1.0);
        let mut preds = vec![0.0f32];
        obj.pred_transform(&mut preds);
        assert_eq!(preds, vec![1.0], "exp(0)");
        let mut preds = vec![0.0f32];
        obj.eval_transform(&mut preds);
        assert_eq!(preds, vec![0.0], "the metric sees the margin");
    }

    #[test]
    fn cox_separates_events_from_censored_rows() {
        let obj = CoxRegression;
        // Positive label: an observed event. Negative: censored at |label|.
        let info = MetaInfo {
            num_row: 4,
            num_col: 1,
            labels: vec![1.0, -2.0, 3.0, -4.0],
            num_target: 1,
            ..Default::default()
        };
        let mut out = Vec::new();
        obj.get_gradient(&[0.0; 4], &info, 0, &mut out);
        assert_eq!(out.len(), 4);
        // An event pulls its own risk down (negative gradient); a censored row
        // only ever contributes to others' risk sets.
        assert!(out[0].grad < 0.0, "{:?}", out[0]);
        assert!(out[1].grad > 0.0, "a censored row has no event term: {:?}", out[1]);
        assert!(out.iter().all(|p| p.grad.is_finite() && p.hess.is_finite()));
    }
}
