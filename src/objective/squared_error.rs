//! `reg:squarederror` — one-half squared error.
//!
//! `grad = (p - y) * w`, `hess = 1 * w`, identity transform. The intercept is
//! the (weighted) sample mean of the labels, matching
//! `FitInterceptGlmLike::InitEstimation` for objectives whose
//! `scale_pos_weight` is 1.

use super::{GradientPair, Objective};
use crate::data::MetaInfo;

/// The squared-error regression objective.
#[derive(Clone, Copy, Debug, Default)]
pub struct SquaredError;

impl Objective for SquaredError {
    fn name(&self) -> &'static str {
        "reg:squarederror"
    }

    fn get_gradient(&self, preds: &[f32], info: &MetaInfo, _iter: i32, out: &mut Vec<GradientPair>) {
        out.clear();
        out.reserve(preds.len());
        match &info.weights {
            Some(w) => {
                for i in 0..preds.len() {
                    out.push(GradientPair {
                        grad: (preds[i] - info.labels[i]) * w[i],
                        hess: w[i],
                    });
                }
            }
            None => {
                for i in 0..preds.len() {
                    out.push(GradientPair { grad: preds[i] - info.labels[i], hess: 1.0 });
                }
            }
        }
    }

    fn pred_transform(&self, _preds: &mut [f32]) {}

    fn init_estimation(&self, info: &MetaInfo) -> f32 {
        if info.labels.is_empty() {
            return 0.0;
        }
        match &info.weights {
            Some(w) => {
                let mut num = 0.0f64;
                let mut den = 0.0f64;
                for (y, w) in info.labels.iter().zip(w) {
                    num += (*y as f64) * (*w as f64);
                    den += *w as f64;
                }
                if den == 0.0 { 0.0 } else { (num / den) as f32 }
            }
            None => {
                let sum: f64 = info.labels.iter().map(|&y| y as f64).sum();
                (sum / info.labels.len() as f64) as f32
            }
        }
    }

    fn default_metric(&self) -> &'static str {
        "rmse"
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
    fn gradient_is_the_residual_and_hessian_is_one() {
        let mut out = Vec::new();
        SquaredError.get_gradient(&[1.0, 2.0], &info(&[0.5, 3.0], None), 0, &mut out);
        assert_eq!(out, vec![
            GradientPair { grad: 0.5, hess: 1.0 },
            GradientPair { grad: -1.0, hess: 1.0 },
        ]);
    }

    #[test]
    fn weights_scale_both_derivatives() {
        let mut out = Vec::new();
        SquaredError.get_gradient(&[1.0], &info(&[0.0], Some(&[2.5])), 0, &mut out);
        assert_eq!(out, vec![GradientPair { grad: 2.5, hess: 2.5 }]);
    }

    #[test]
    fn intercept_is_the_label_mean() {
        assert_eq!(SquaredError.init_estimation(&info(&[1.0, 2.0, 3.0], None)), 2.0);
    }

    #[test]
    fn intercept_is_the_weighted_label_mean() {
        let got = SquaredError.init_estimation(&info(&[0.0, 4.0], Some(&[3.0, 1.0])));
        assert_eq!(got, 1.0);
    }

    #[test]
    fn empty_labels_give_a_zero_intercept() {
        assert_eq!(SquaredError.init_estimation(&info(&[], None)), 0.0);
    }
}
