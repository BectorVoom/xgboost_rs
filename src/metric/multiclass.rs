//! The multiclass metrics, `merror` and `mlogloss`.
//!
//! A port of `src/metric/multiclass_metric.cu`. Both read one row of class
//! probabilities at a time, which is what the objective's `EvalTransform`
//! hands them even for `multi:softmax`.

use super::{Metric, mean};
use crate::data::MetaInfo;

/// Which multiclass score is being computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// `merror` — the fraction of rows whose top class is wrong.
    Error,
    /// `mlogloss` — the negative log likelihood of the true class.
    Logloss,
}

/// `merror` or `mlogloss`.
#[derive(Clone, Copy, Debug)]
pub struct MultiClass {
    kind: Kind,
}

impl MultiClass {
    pub fn error() -> Self {
        Self { kind: Kind::Error }
    }

    pub fn logloss() -> Self {
        Self { kind: Kind::Logloss }
    }
}

impl Metric for MultiClass {
    fn name(&self) -> &str {
        match self.kind {
            Kind::Error => "merror",
            Kind::Logloss => "mlogloss",
        }
    }

    fn eval(&self, preds: &[f32], info: &MetaInfo) -> f64 {
        if info.num_row == 0 {
            return 0.0;
        }
        let n_class = (preds.len() / info.num_row).max(1);
        let (mut esum, mut wsum) = (0.0f64, 0.0f64);
        for i in 0..info.num_row {
            let w = info.weight(i) as f64;
            let row = &preds[i * n_class..(i + 1) * n_class];
            let label = info.label(i, 0) as usize;
            let score = match self.kind {
                Kind::Error => {
                    let best = row
                        .iter()
                        .enumerate()
                        .fold((0usize, f32::MIN), |acc, (c, &v)| {
                            if v > acc.1 { (c, v) } else { acc }
                        })
                        .0;
                    f64::from(best != label)
                }
                Kind::Logloss => {
                    // A label outside the class range would be rejected by the
                    // objective; scoring it as maximally wrong keeps the metric
                    // total for callers that evaluate a foreign model.
                    let p = row.get(label).copied().unwrap_or(0.0);
                    -(p.max(1e-16) as f64).ln()
                }
            };
            esum += score * w;
            wsum += w;
        }
        mean(esum, wsum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn merror_counts_wrong_top_classes() {
        let m = MultiClass::error();
        // Row 0 predicts class 1 (right), row 1 predicts class 0 (wrong).
        let preds = [0.1f32, 0.8, 0.1, 0.7, 0.2, 0.1];
        assert_eq!(m.eval(&preds, &info(&[1.0, 2.0])), 0.5);
        assert_eq!(m.eval(&preds, &info(&[1.0, 0.0])), 0.0);
    }

    #[test]
    fn mlogloss_scores_the_probability_of_the_true_class() {
        let m = MultiClass::logloss();
        let preds = [0.2f32, 0.8];
        // -log(0.8) for a row labelled 1.
        let got = m.eval(&preds, &info(&[1.0]));
        assert!((got + 0.8f64.ln()).abs() < 1e-6, "{got}");
        // A confident wrong prediction scores far worse.
        assert!(m.eval(&preds, &info(&[0.0])) > got);
    }

    #[test]
    fn a_zero_probability_stays_finite() {
        let m = MultiClass::logloss();
        let got = m.eval(&[0.0f32, 1.0], &info(&[0.0]));
        assert!(got.is_finite() && got > 30.0, "{got}");
    }

    #[test]
    fn weights_reach_the_multiclass_metrics() {
        let mut d = info(&[1.0, 0.0]);
        d.weights = Some(vec![3.0, 1.0]);
        // Row 0 right, row 1 wrong: weighting row 0 more lowers the error.
        let preds = [0.1f32, 0.9, 0.1, 0.9];
        assert_eq!(MultiClass::error().eval(&preds, &d), 0.25);
    }
}
