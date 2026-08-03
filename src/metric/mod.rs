//! Evaluation metrics.

use crate::data::MetaInfo;

/// A metric evaluated on transformed predictions.
pub trait Metric {
    fn name(&self) -> &'static str;
    fn eval(&self, preds: &[f32], info: &MetaInfo) -> f64;
}

/// Root mean squared error, weighted when the matrix carries weights.
#[derive(Clone, Copy, Debug, Default)]
pub struct Rmse;

impl Metric for Rmse {
    fn name(&self) -> &'static str {
        "rmse"
    }

    fn eval(&self, preds: &[f32], info: &MetaInfo) -> f64 {
        let mut sum = 0.0f64;
        let mut wsum = 0.0f64;
        for i in 0..preds.len() {
            let diff = (preds[i] - info.labels[i]) as f64;
            let w = info.weight(i) as f64;
            sum += diff * diff * w;
            wsum += w;
        }
        if wsum == 0.0 { 0.0 } else { (sum / wsum).sqrt() }
    }
}

/// Construct a metric by its XGBoost name.
pub fn create(name: &str) -> crate::Result<Box<dyn Metric>> {
    match name {
        "rmse" => Ok(Box::new(Rmse)),
        other => Err(crate::Error::invalid(
            "eval_metric",
            format!("`{other}` is not implemented; Phase 1 supports `rmse`"),
        )),
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
            weights: weights.map(|w| w.to_vec()),
            base_margin: None,
        }
    }

    #[test]
    fn perfect_predictions_score_zero() {
        assert_eq!(Rmse.eval(&[1.0, 2.0], &info(&[1.0, 2.0], None)), 0.0);
    }

    #[test]
    fn rmse_is_the_root_mean_square_of_residuals() {
        // Residuals 3 and 4 -> sqrt((9 + 16) / 2).
        let got = Rmse.eval(&[3.0, 4.0], &info(&[0.0, 0.0], None));
        assert!((got - (12.5f64).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn weights_bias_the_mean() {
        // Residual 2 with weight 3, residual 0 with weight 1.
        let got = Rmse.eval(&[2.0, 0.0], &info(&[0.0, 0.0], Some(&[3.0, 1.0])));
        assert!((got - (12.0f64 / 4.0).sqrt()).abs() < 1e-12);
    }
}
