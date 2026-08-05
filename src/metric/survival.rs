//! The survival metrics: `cox-nloglik`, `aft-nloglik` and
//! `interval-regression-accuracy`.
//!
//! Ports of `src/metric/survival_metric.cu` and the Cox likelihood in
//! `src/objective/regression_obj.cu`.

use super::{Metric, mean};
use crate::data::MetaInfo;
use crate::objective::survival::aft_loss;
use crate::parameters::AftDistribution;

/// `cox-nloglik` — the negative log partial likelihood.
///
/// Predictions arrive as hazard ratios (`exp(margin)`), which is what the
/// objective's transform produces.
#[derive(Clone, Copy, Debug, Default)]
pub struct CoxNegLogLik;

impl Metric for CoxNegLogLik {
    fn name(&self) -> &str {
        "cox-nloglik"
    }

    fn eval(&self, preds: &[f32], info: &MetaInfo) -> f64 {
        if preds.is_empty() {
            return 0.0;
        }
        // Ascending event time; the risk set of an event is everything that has
        // not failed yet, so the sweep runs from the latest time backwards.
        let mut order: Vec<usize> = (0..preds.len()).collect();
        order.sort_by(|&a, &b| info.labels[b].abs().total_cmp(&info.labels[a].abs()));

        let mut risk_sum = 0.0f64;
        let mut nloglik = 0.0f64;
        let mut wsum = 0.0f64;
        for &i in &order {
            // Predictions are hazard ratios already, so no exponential here.
            risk_sum += preds[i].max(1e-16) as f64;
            if info.labels[i] > 0.0 {
                let w = info.weight(i) as f64;
                nloglik += w * (risk_sum.ln() - (preds[i].max(1e-16) as f64).ln());
                wsum += w;
            }
        }
        mean(nloglik, wsum)
    }
}

/// `aft-nloglik` — the negative log likelihood of the AFT model.
///
/// The scale and distribution are the objective's; a metric configured on its
/// own uses the same defaults `AFTParam` declares.
#[derive(Clone, Copy, Debug)]
pub struct AftNegLogLik {
    dist: AftDistribution,
    sigma: f32,
}

impl Default for AftNegLogLik {
    fn default() -> Self {
        Self { dist: AftDistribution::Normal, sigma: 1.0 }
    }
}

impl AftNegLogLik {
    pub fn new(dist: AftDistribution, sigma: f32) -> Self {
        Self { dist, sigma }
    }
}

impl Metric for AftNegLogLik {
    fn name(&self) -> &str {
        "aft-nloglik"
    }

    fn eval(&self, preds: &[f32], info: &MetaInfo) -> f64 {
        let (mut esum, mut wsum) = (0.0f64, 0.0f64);
        for i in 0..preds.len() {
            let w = info.weight(i) as f64;
            esum += aft_loss(
                self.dist,
                info.lower_bound(i) as f64,
                info.upper_bound(i) as f64,
                preds[i] as f64,
                self.sigma as f64,
            ) * w;
            wsum += w;
        }
        mean(esum, wsum)
    }
}

/// `interval-regression-accuracy` — the fraction of predictions that land
/// inside their censoring interval.
///
/// Predictions are in log time, so the metric exponentiates before comparing.
/// Unlike the rest of this module it is *maximised*.
#[derive(Clone, Copy, Debug, Default)]
pub struct IntervalRegressionAccuracy;

impl Metric for IntervalRegressionAccuracy {
    fn name(&self) -> &str {
        "interval-regression-accuracy"
    }

    fn eval(&self, preds: &[f32], info: &MetaInfo) -> f64 {
        let (mut esum, mut wsum) = (0.0f64, 0.0f64);
        for i in 0..preds.len() {
            let w = info.weight(i) as f64;
            let pred = preds[i].exp();
            let inside = pred >= info.lower_bound(i) && pred <= info.upper_bound(i);
            esum += f64::from(inside) * w;
            wsum += w;
        }
        mean(esum, wsum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn survival(lower: &[f32], upper: &[f32]) -> MetaInfo {
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

    #[test]
    fn interval_accuracy_counts_predictions_inside_the_interval() {
        let m = IntervalRegressionAccuracy;
        let d = survival(&[1.0, 10.0], &[2.0, 20.0]);
        // exp(0.4) ~ 1.5 (inside) and exp(0) = 1 (below 10).
        let got = m.eval(&[0.4, 0.0], &d);
        assert!((got - 0.5).abs() < 1e-9, "{got}");
        // Both inside.
        let got = m.eval(&[0.4, 2.7], &d);
        assert!((got - 1.0).abs() < 1e-9, "{got}");
    }

    #[test]
    fn aft_nloglik_is_lowest_near_the_true_time() {
        let m = AftNegLogLik::default();
        let d = survival(&[10.0, 10.0], &[10.0, 10.0]);
        let near = m.eval(&[10.0f32.ln(), 10.0f32.ln()], &d);
        let far = m.eval(&[1.0f32.ln(), 1.0f32.ln()], &d);
        assert!(near < far, "{near} vs {far}");
    }

    #[test]
    fn every_aft_distribution_scores_finitely() {
        let d = survival(&[1.0, 5.0], &[2.0, f32::INFINITY]);
        for dist in [AftDistribution::Normal, AftDistribution::Logistic, AftDistribution::Extreme] {
            let got = AftNegLogLik::new(dist, 1.0).eval(&[0.5, 1.5], &d);
            assert!(got.is_finite(), "{dist}: {got}");
        }
    }

    #[test]
    fn cox_nloglik_prefers_ranking_events_above_survivors() {
        let m = CoxNegLogLik;
        // Row 0 fails at time 1, row 1 is censored at time 2.
        let d = MetaInfo {
            num_row: 2,
            num_col: 1,
            labels: vec![1.0, -2.0],
            num_target: 1,
            ..Default::default()
        };
        // A high hazard for the row that actually failed scores better.
        let good = m.eval(&[4.0, 1.0], &d);
        let bad = m.eval(&[1.0, 4.0], &d);
        assert!(good < bad, "{good} vs {bad}");
    }
}
