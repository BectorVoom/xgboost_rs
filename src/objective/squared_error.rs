//! `reg:squarederror` — one-half squared error.
//!
//! `grad = (p - y) * w`, `hess = 1 * w`, identity transform, where `w` is the
//! row weight multiplied by `scale_pos_weight` when the label is exactly 1.
//! `RegLossObj::GetGradient` applies that scaling for every regression loss,
//! not only the classification ones, so it is part of this objective too.

use super::{GradientPair, Objective};
use crate::data::MetaInfo;

/// The squared-error regression objective.
#[derive(Clone, Copy, Debug)]
pub struct SquaredError {
    /// `scale_pos_weight`: an extra factor on rows whose label is 1.
    scale_pos_weight: f32,
}

impl Default for SquaredError {
    fn default() -> Self {
        Self { scale_pos_weight: 1.0 }
    }
}

impl SquaredError {
    pub fn new(scale_pos_weight: f32) -> Self {
        Self { scale_pos_weight }
    }

    /// Effective weight of row `i`.
    #[inline]
    fn weight(&self, info: &MetaInfo, i: usize) -> f32 {
        let w = info.weight(i);
        if info.labels[i] == 1.0 { w * self.scale_pos_weight } else { w }
    }
}

impl Objective for SquaredError {
    fn name(&self) -> &'static str {
        "reg:squarederror"
    }

    fn get_gradient(&self, preds: &[f32], info: &MetaInfo, _iter: i32, out: &mut Vec<GradientPair>) {
        out.clear();
        out.reserve(preds.len());
        for i in 0..preds.len() {
            let w = self.weight(info, i);
            out.push(GradientPair { grad: (preds[i] - info.labels[i]) * w, hess: w });
        }
    }

    fn pred_transform(&self, _preds: &mut [f32]) {}

    /// The (weighted) label mean.
    ///
    /// Upstream switches from `FitInterceptGlmLike` to a Newton step when
    /// `scale_pos_weight` is set; for squared error the Newton step is
    /// `-sum(grad) / sum(hess)` at a zero margin, which is the same weighted
    /// mean with the scaled weights — so one expression covers both.
    fn init_estimation(&self, info: &MetaInfo) -> f32 {
        if info.labels.is_empty() {
            return 0.0;
        }
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for i in 0..info.labels.len() {
            let w = self.weight(info, i) as f64;
            num += info.labels[i] as f64 * w;
            den += w;
        }
        if den == 0.0 { 0.0 } else { (num / den) as f32 }
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
        SquaredError::default().get_gradient(&[1.0, 2.0], &info(&[0.5, 3.0], None), 0, &mut out);
        assert_eq!(out, vec![
            GradientPair { grad: 0.5, hess: 1.0 },
            GradientPair { grad: -1.0, hess: 1.0 },
        ]);
    }

    #[test]
    fn weights_scale_both_derivatives() {
        let mut out = Vec::new();
        SquaredError::default().get_gradient(&[1.0], &info(&[0.0], Some(&[2.5])), 0, &mut out);
        assert_eq!(out, vec![GradientPair { grad: 2.5, hess: 2.5 }]);
    }

    #[test]
    fn scale_pos_weight_only_touches_rows_labelled_one() {
        let mut out = Vec::new();
        SquaredError::new(3.0).get_gradient(&[0.0, 0.0], &info(&[1.0, 2.0], None), 0, &mut out);
        assert_eq!(out, vec![
            GradientPair { grad: -3.0, hess: 3.0 },
            GradientPair { grad: -2.0, hess: 1.0 },
        ]);
    }

    #[test]
    fn scale_pos_weight_compounds_with_the_row_weight() {
        let mut out = Vec::new();
        SquaredError::new(2.0).get_gradient(&[0.0], &info(&[1.0], Some(&[5.0])), 0, &mut out);
        assert_eq!(out, vec![GradientPair { grad: -10.0, hess: 10.0 }]);
    }

    #[test]
    fn intercept_is_the_label_mean() {
        assert_eq!(SquaredError::default().init_estimation(&info(&[1.0, 2.0, 3.0], None)), 2.0);
    }

    #[test]
    fn intercept_is_the_weighted_label_mean() {
        let got = SquaredError::default().init_estimation(&info(&[0.0, 4.0], Some(&[3.0, 1.0])));
        assert_eq!(got, 1.0);
    }

    #[test]
    fn scale_pos_weight_pulls_the_intercept_towards_one() {
        // Labels 0 and 1: unweighted the intercept is 0.5, and tripling the
        // positive row's weight moves it to 3/4.
        let plain = SquaredError::default().init_estimation(&info(&[0.0, 1.0], None));
        assert_eq!(plain, 0.5);
        let scaled = SquaredError::new(3.0).init_estimation(&info(&[0.0, 1.0], None));
        assert_eq!(scaled, 0.75);
    }

    #[test]
    fn empty_labels_give_a_zero_intercept() {
        assert_eq!(SquaredError::default().init_estimation(&info(&[], None)), 0.0);
    }
}
