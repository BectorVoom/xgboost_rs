//! Metrics that score one `(row, output)` at a time.
//!
//! A port of `src/metric/elementwise_metric.cu`. Each variant is one
//! `EvalRow` policy plus the `GetFinal` reduction that closes it.

use super::{Metric, elementwise_reduce, mean};
use crate::data::MetaInfo;

/// `kRtEps`, used by the gamma deviance to keep its logarithm finite.
const RT_EPS: f32 = 1e-6;

/// Which elementwise loss is being scored.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    /// `rmse`.
    Rmse,
    /// `rmsle`.
    Rmsle,
    /// `mae`.
    Mae,
    /// `mape`.
    Mape,
    /// `mphe`, the mean Pseudo-Huber error.
    Mphe { slope: f32 },
    /// `logloss`.
    Logloss,
    /// `error` / `error@t`.
    Error { threshold: f32 },
    /// `poisson-nloglik`.
    PoissonNegLogLik,
    /// `gamma-nloglik`.
    GammaNegLogLik,
    /// `gamma-deviance`.
    GammaDeviance,
    /// `tweedie-nloglik@rho`.
    TweedieNegLogLik { rho: f32 },
}

impl Kind {
    fn name(self) -> String {
        match self {
            Self::Rmse => "rmse".to_owned(),
            Self::Rmsle => "rmsle".to_owned(),
            Self::Mae => "mae".to_owned(),
            Self::Mape => "mape".to_owned(),
            Self::Mphe { .. } => "mphe".to_owned(),
            Self::Logloss => "logloss".to_owned(),
            // `EvalError::Name` drops the argument at the default threshold, so
            // `error@0.5` and `error` report the same name.
            Self::Error { threshold } => {
                if threshold == 0.5 { "error".to_owned() } else { format!("error@{threshold}") }
            }
            Self::PoissonNegLogLik => "poisson-nloglik".to_owned(),
            Self::GammaNegLogLik => "gamma-nloglik".to_owned(),
            Self::GammaDeviance => "gamma-deviance".to_owned(),
            Self::TweedieNegLogLik { rho } => format!("tweedie-nloglik@{rho}"),
        }
    }

    /// `EvalRow`: the unweighted score of one `(label, prediction)`.
    fn row(self, label: f32, pred: f32) -> f64 {
        match self {
            Self::Rmse => {
                let d = (label - pred) as f64;
                d * d
            }
            Self::Rmsle => {
                let d = (label.ln_1p() - pred.ln_1p()) as f64;
                d * d
            }
            Self::Mae => (label - pred).abs() as f64,
            Self::Mape => ((label - pred) / label).abs() as f64,
            Self::Mphe { slope } => {
                let a = label - pred;
                (slope * slope * ((1.0 + (a / slope) * (a / slope)).sqrt() - 1.0)) as f64
            }
            Self::Logloss => {
                // `xlogy(-y, p) + xlogy(-(1 - y), 1 - p)`: a zero coefficient
                // contributes nothing even where the logarithm would diverge.
                let xlogy = |x: f32, y: f32| -> f64 {
                    if x == 0.0 { 0.0 } else { (x * y.max(1e-16).ln()) as f64 }
                };
                xlogy(-label, pred) + xlogy(-(1.0 - label), 1.0 - pred)
            }
            // Labels are assumed to be in [0, 1]: a prediction above the
            // threshold is wrong by `1 - label`, below it by `label`.
            Self::Error { threshold } => {
                if pred > threshold { (1.0 - label) as f64 } else { label as f64 }
            }
            Self::PoissonNegLogLik => {
                let py = pred.max(1e-16) as f64;
                log_gamma(label as f64 + 1.0) + py - py.ln() * label as f64
            }
            Self::GammaNegLogLik => {
                // The exponential-family form at a unit dispersion, where the
                // remaining term is identically zero.
                let py = pred.max(1e-6) as f64;
                let theta = -1.0 / py;
                let b = -(-theta).ln();
                -(label as f64 * theta - b)
            }
            Self::GammaDeviance => {
                let predt = (pred + RT_EPS) as f64;
                let label = (label + RT_EPS) as f64;
                (predt / label).ln() + label / predt - 1.0
            }
            Self::TweedieNegLogLik { rho } => {
                let (y, p, rho) = (label as f64, pred as f64, rho as f64);
                let a = y * ((1.0 - rho) * p.ln()).exp() / (1.0 - rho);
                let b = ((2.0 - rho) * p.ln()).exp() / (2.0 - rho);
                -a + b
            }
        }
    }

    /// `GetFinal`.
    fn finalize(self, esum: f64, wsum: f64) -> f64 {
        match self {
            // The two squared losses take a root.
            Self::Rmse | Self::Rmsle => {
                if wsum == 0.0 { esum.sqrt() } else { (esum / wsum).sqrt() }
            }
            // The deviance is twice the mean.
            Self::GammaDeviance => 2.0 * esum / if wsum <= 0.0 { RT_EPS as f64 } else { wsum },
            _ => mean(esum, wsum),
        }
    }
}

/// `log(Gamma(x))` — Lanczos approximation, accurate well past `f32`.
fn log_gamma(x: f64) -> f64 {
    const G: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection, so the series is only evaluated where it converges well.
        (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - log_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = G[0];
        let t = x + 7.5;
        for (i, g) in G.iter().enumerate().skip(1) {
            a += g / (x + i as f64);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

/// One of the elementwise metrics.
#[derive(Clone, Debug)]
pub struct Elementwise {
    kind: Kind,
    name: String,
}

impl Elementwise {
    pub fn new(kind: Kind) -> Self {
        Self { kind, name: kind.name() }
    }
}

impl Metric for Elementwise {
    fn name(&self) -> &str {
        &self.name
    }

    fn eval(&self, preds: &[f32], info: &MetaInfo) -> f64 {
        let (esum, wsum) = elementwise_reduce(preds, info, |y, p| self.kind.row(y, p));
        self.kind.finalize(esum, wsum)
    }
}

/// `quantile` and `expectile` — the two metrics whose per-row loss depends on
/// *which* output it is scoring.
///
/// The alphas are not part of the metric's name upstream either; the metric
/// weights every output equally, which is what makes a bare `quantile` a valid
/// `eval_metric` for any `quantile_alpha`.
#[derive(Clone, Debug)]
pub struct PinballLoss {
    /// Square the residual (expectile) rather than take it as it is (quantile).
    squared: bool,
    alpha: Vec<f32>,
    name: &'static str,
}

impl PinballLoss {
    pub fn quantile() -> Self {
        Self { squared: false, alpha: Vec::new(), name: "quantile" }
    }

    pub fn expectile() -> Self {
        Self { squared: true, alpha: Vec::new(), name: "expectile" }
    }

    /// Pin the alphas the fit is using. Without them the metric spreads the
    /// outputs evenly over `(0, 1)`, which is the best it can do from the
    /// prediction shape alone.
    pub fn with_alpha(mut self, alpha: Vec<f32>) -> Self {
        self.alpha = alpha;
        self
    }

    fn alpha_at(&self, t: usize, n: usize) -> f32 {
        match self.alpha.get(t) {
            Some(a) => *a,
            None if n == 1 => 0.5,
            None => (t + 1) as f32 / (n + 1) as f32,
        }
    }
}

impl Metric for PinballLoss {
    fn name(&self) -> &str {
        self.name
    }

    fn eval(&self, preds: &[f32], info: &MetaInfo) -> f64 {
        if info.num_row == 0 {
            return 0.0;
        }
        let n_groups = (preds.len() / info.num_row).max(1);
        let (mut esum, mut wsum) = (0.0f64, 0.0f64);
        for i in 0..info.num_row {
            let w = info.weight(i) as f64;
            let y = info.label(i, 0);
            for t in 0..n_groups {
                let alpha = self.alpha_at(t, n_groups);
                let p = preds[i * n_groups + t];
                let loss = if self.squared {
                    let diff = p - y;
                    let scale = if diff >= 0.0 { 1.0 - alpha } else { alpha };
                    (scale * diff * diff) as f64
                } else {
                    let d = y - p;
                    let sign = if d >= 0.0 { 1.0f32 } else { 0.0 };
                    ((alpha * sign * d) - (1.0 - alpha) * (1.0 - sign) * d) as f64
                };
                esum += loss * w;
                wsum += w;
            }
        }
        mean(esum, wsum)
    }
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

    #[test]
    fn rmse_is_the_root_mean_square_of_residuals() {
        let m = Elementwise::new(Kind::Rmse);
        assert_eq!(m.eval(&[1.0, 2.0], &info(&[1.0, 2.0], None)), 0.0);
        let got = m.eval(&[3.0, 4.0], &info(&[0.0, 0.0], None));
        assert!((got - 12.5f64.sqrt()).abs() < 1e-12, "{got}");
    }

    #[test]
    fn weights_bias_the_mean() {
        let m = Elementwise::new(Kind::Rmse);
        let got = m.eval(&[2.0, 0.0], &info(&[0.0, 0.0], Some(&[3.0, 1.0])));
        assert!((got - (12.0f64 / 4.0).sqrt()).abs() < 1e-12, "{got}");
    }

    #[test]
    fn mae_and_mape_measure_absolute_and_relative_error() {
        // Residuals of 2 and 1.
        assert_eq!(Elementwise::new(Kind::Mae).eval(&[3.0, 0.0], &info(&[1.0, 1.0], None)), 1.5);
        // |(10 - 5)/10| = 0.5 and |(2 - 1)/2| = 0.5.
        assert_eq!(Elementwise::new(Kind::Mape).eval(&[5.0, 1.0], &info(&[10.0, 2.0], None)), 0.5);
    }

    #[test]
    fn logloss_rewards_confident_correct_predictions() {
        let m = Elementwise::new(Kind::Logloss);
        let d = info(&[1.0, 0.0], None);
        let good = m.eval(&[0.99, 0.01], &d);
        let bad = m.eval(&[0.01, 0.99], &d);
        assert!(good < bad, "{good} vs {bad}");
        assert!(good > 0.0);
        // A perfectly confident correct prediction is finite, not -inf.
        assert!(m.eval(&[1.0, 0.0], &d).abs() < 1e-9);
    }

    #[test]
    fn the_error_threshold_moves_the_decision_boundary() {
        let d = info(&[1.0, 1.0], None);
        let preds = [0.6f32, 0.4];
        assert_eq!(Elementwise::new(Kind::Error { threshold: 0.5 }).eval(&preds, &d), 0.5);
        // Raising the threshold rejects both predictions.
        assert_eq!(Elementwise::new(Kind::Error { threshold: 0.7 }).eval(&preds, &d), 1.0);
        // Lowering it accepts both.
        assert_eq!(Elementwise::new(Kind::Error { threshold: 0.3 }).eval(&preds, &d), 0.0);
    }

    #[test]
    fn the_error_metric_only_names_a_non_default_threshold() {
        assert_eq!(Elementwise::new(Kind::Error { threshold: 0.5 }).name(), "error");
        assert_eq!(Elementwise::new(Kind::Error { threshold: 0.7 }).name(), "error@0.7");
    }

    #[test]
    fn log_gamma_matches_factorials() {
        // log(Gamma(n + 1)) = log(n!).
        for (n, fact) in [(0u32, 1.0f64), (1, 1.0), (4, 24.0), (6, 720.0)] {
            let got = log_gamma(n as f64 + 1.0);
            assert!((got - fact.ln()).abs() < 1e-9, "n = {n}: {got} vs {}", fact.ln());
        }
    }

    #[test]
    fn poisson_nloglik_is_minimised_at_the_label() {
        let m = Elementwise::new(Kind::PoissonNegLogLik);
        let d = info(&[3.0], None);
        let at_label = m.eval(&[3.0], &d);
        assert!(at_label < m.eval(&[1.0], &d), "{at_label}");
        assert!(at_label < m.eval(&[8.0], &d));
    }

    #[test]
    fn gamma_deviance_is_zero_at_a_perfect_fit() {
        let m = Elementwise::new(Kind::GammaDeviance);
        assert!(m.eval(&[2.0, 5.0], &info(&[2.0, 5.0], None)).abs() < 1e-6);
        assert!(m.eval(&[1.0, 9.0], &info(&[2.0, 5.0], None)) > 0.0);
    }

    #[test]
    fn tweedie_names_and_scores_by_its_variance_power() {
        let m = Elementwise::new(Kind::TweedieNegLogLik { rho: 1.5 });
        assert_eq!(m.name(), "tweedie-nloglik@1.5");
        let d = info(&[2.0], None);
        // The loss is minimised near the label for the log-link mean.
        assert!(m.eval(&[2.0], &d) < m.eval(&[20.0], &d));
    }

    #[test]
    fn the_quantile_metric_is_the_pinball_loss() {
        // One output, so alpha is 0.5 and the loss is half the absolute error.
        let m = PinballLoss::quantile();
        let got = m.eval(&[0.0, 0.0], &info(&[2.0, -2.0], None));
        assert!((got - 1.0).abs() < 1e-6, "{got}");

        // A high alpha punishes under-prediction harder than over-prediction.
        let m = PinballLoss::quantile().with_alpha(vec![0.9]);
        let under = m.eval(&[0.0], &info(&[1.0], None));
        let over = m.eval(&[1.0], &info(&[0.0], None));
        assert!(under > over, "{under} vs {over}");
    }

    #[test]
    fn the_expectile_metric_squares_the_residual() {
        let m = PinballLoss::expectile().with_alpha(vec![0.5]);
        let got = m.eval(&[0.0], &info(&[2.0], None));
        assert!((got - 2.0).abs() < 1e-6, "0.5 * 2^2, got {got}");
    }

    #[test]
    fn multi_output_predictions_score_against_their_own_target() {
        // Two targets, two outputs: the second target is predicted perfectly.
        let d = MetaInfo {
            num_row: 2,
            num_col: 1,
            labels: vec![0.0, 5.0, 0.0, 5.0],
            num_target: 2,
            ..Default::default()
        };
        let m = Elementwise::new(Kind::Mae);
        assert_eq!(m.eval(&[1.0, 5.0, 1.0, 5.0], &d), 0.5, "half the outputs are off by one");
    }
}
