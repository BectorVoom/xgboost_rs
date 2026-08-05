//! The classification objectives.
//!
//! Ports of `src/objective/hinge.cu` and `src/objective/multiclass_obj.cu`.

use super::{GradientPair, Objective, check_labels, fit_intercept, sigmoid};
use crate::data::MetaInfo;
use crate::{Error, Result};

/// `binary:logitraw` — the logistic loss reported as an untransformed margin.
///
/// It cannot ride on `RegLossObj` because the loss and the transform disagree:
/// the gradient is taken at `sigmoid(margin)` while the prediction stays the
/// margin itself.
#[derive(Clone, Copy, Debug)]
pub struct LogitRaw {
    scale_pos_weight: f32,
}

impl LogitRaw {
    pub fn new(scale_pos_weight: f32) -> Self {
        Self { scale_pos_weight }
    }
}

impl Objective for LogitRaw {
    fn name(&self) -> &'static str {
        "binary:logitraw"
    }

    fn get_gradient(&self, preds: &[f32], info: &MetaInfo, _iter: i32, out: &mut Vec<GradientPair>) {
        let n_targets = info.n_targets();
        out.clear();
        out.reserve(preds.len());
        for idx in 0..preds.len() {
            let label = info.labels[idx];
            let mut w = info.weight(idx / n_targets);
            if label == 1.0 {
                w *= self.scale_pos_weight;
            }
            let p = sigmoid(preds[idx]);
            out.push(GradientPair {
                grad: (p - label) * w,
                hess: (p * (1.0 - p)).max(1e-16) * w,
            });
        }
    }

    /// No transform: the margin *is* the prediction.
    fn pred_transform(&self, _preds: &mut Vec<f32>) {}

    fn init_estimation(&self, info: &MetaInfo) -> Vec<f32> {
        fit_intercept(self, info)
    }

    fn default_metric(&self) -> String {
        "logloss".to_owned()
    }

    fn validate_data(&self, info: &MetaInfo) -> Result<()> {
        check_labels(
            self.name(),
            info,
            |y| (0.0..=1.0).contains(&y),
            "label must be in [0, 1] for logistic regression",
        )
    }
}

/// `binary:hinge` — hinge loss, predicting `0` or `1`.
///
/// The hessian is `f32::MIN_POSITIVE` rather than `0` outside the margin so a
/// leaf never divides by zero; that is upstream's choice too.
#[derive(Clone, Copy, Debug, Default)]
pub struct BinaryHinge;

impl Objective for BinaryHinge {
    fn name(&self) -> &'static str {
        "binary:hinge"
    }

    fn get_gradient(&self, preds: &[f32], info: &MetaInfo, _iter: i32, out: &mut Vec<GradientPair>) {
        let n_targets = info.n_targets();
        out.clear();
        out.reserve(preds.len());
        for idx in 0..preds.len() {
            let w = info.weight(idx / n_targets);
            let p = preds[idx];
            // Labels are 0/1; the hinge works on -1/+1.
            let y = info.labels[idx] * 2.0 - 1.0;
            let pair = if p * y < 1.0 {
                GradientPair { grad: -y * w, hess: w }
            } else {
                GradientPair { grad: 0.0, hess: f32::MIN_POSITIVE }
            };
            out.push(pair);
        }
    }

    fn pred_transform(&self, preds: &mut Vec<f32>) {
        for p in preds.iter_mut() {
            *p = if *p > 0.0 { 1.0 } else { 0.0 };
        }
    }

    fn init_estimation(&self, info: &MetaInfo) -> Vec<f32> {
        fit_intercept(self, info)
    }

    fn default_metric(&self) -> String {
        "error".to_owned()
    }

    fn validate_data(&self, info: &MetaInfo) -> Result<()> {
        check_labels(
            self.name(),
            info,
            |y| (0.0..=1.0).contains(&y),
            "label must be in [0, 1] for hinge loss",
        )
    }
}

/// `multi:softmax` and `multi:softprob` — softmax over `num_class` outputs.
///
/// The two differ only in what a prediction reports: `softprob` gives the class
/// probabilities, `softmax` the index of the largest. Metrics always see the
/// probabilities, which is what [`eval_transform`](Objective::eval_transform)
/// is for.
#[derive(Clone, Copy, Debug)]
pub struct SoftmaxMultiClass {
    num_class: usize,
    output_prob: bool,
}

impl SoftmaxMultiClass {
    pub fn new(num_class: usize, output_prob: bool) -> Self {
        Self { num_class, output_prob }
    }

    /// `common::Softmax` over one row, in place.
    fn softmax(row: &mut [f32]) {
        let wmax = row.iter().copied().fold(f32::MIN, f32::max);
        let mut wsum = 0.0f64;
        for v in row.iter() {
            wsum += (v - wmax).exp() as f64;
        }
        for v in row.iter_mut() {
            *v = ((*v - wmax).exp() as f64 / wsum) as f32;
        }
    }
}

impl Objective for SoftmaxMultiClass {
    fn name(&self) -> &'static str {
        if self.output_prob { "multi:softprob" } else { "multi:softmax" }
    }

    fn num_output_group(&self, _info: &MetaInfo) -> usize {
        self.num_class
    }

    fn get_gradient(&self, preds: &[f32], info: &MetaInfo, _iter: i32, out: &mut Vec<GradientPair>) {
        let k = self.num_class;
        out.clear();
        out.resize(preds.len(), GradientPair::default());
        let mut point = vec![0.0f32; k];
        for i in 0..info.num_row {
            point.copy_from_slice(&preds[i * k..(i + 1) * k]);
            Self::softmax(&mut point);
            let label = info.label(i, 0);
            let wt = info.weight(i);
            for (c, &p) in point.iter().enumerate() {
                let h = (2.0 * p * (1.0 - p) * wt).max(1e-16);
                let g = if label == c as f32 { p - 1.0 } else { p };
                out[i * k + c] = GradientPair { grad: g * wt, hess: h };
            }
        }
    }

    fn pred_transform(&self, preds: &mut Vec<f32>) {
        let k = self.num_class;
        if self.output_prob {
            for row in preds.chunks_mut(k) {
                Self::softmax(row);
            }
            return;
        }
        // `softmax` reports the class index, so the buffer shrinks by `k`.
        let n = preds.len() / k;
        for i in 0..n {
            let row = &preds[i * k..(i + 1) * k];
            let best = row
                .iter()
                .enumerate()
                .fold((0usize, f32::MIN), |acc, (c, &v)| if v > acc.1 { (c, v) } else { acc })
                .0;
            preds[i] = best as f32;
        }
        preds.truncate(n);
    }

    /// Metrics always want the probabilities, even for `multi:softmax`.
    fn eval_transform(&self, preds: &mut Vec<f32>) {
        for row in preds.chunks_mut(self.num_class) {
            Self::softmax(row);
        }
    }

    fn init_estimation(&self, info: &MetaInfo) -> Vec<f32> {
        // `SoftmaxMultiClassObj::InitEstimation`: the class frequencies,
        // centred in log space so the margins sum to zero.
        let k = self.num_class;
        let mut counts = vec![0.0f64; k];
        let mut sum_weight = 0.0f64;
        for i in 0..info.num_row {
            let w = info.weight(i) as f64;
            let c = info.label(i, 0) as usize;
            if c < k {
                counts[c] += w;
            }
            sum_weight += w;
        }
        if sum_weight <= 0.0 {
            return vec![0.0; k];
        }
        let logs: Vec<f64> = counts.iter().map(|c| (c / sum_weight).max(1e-6).ln()).collect();
        let mean = logs.iter().sum::<f64>() / k as f64;
        logs.iter().map(|v| (v - mean) as f32).collect()
    }

    fn default_metric(&self) -> String {
        "mlogloss".to_owned()
    }

    fn validate_data(&self, info: &MetaInfo) -> Result<()> {
        if info.n_targets() != 1 {
            return Err(Error::invalid(
                "num_target",
                "multi-class multi-label is not supported; give one label per row",
            ));
        }
        let k = self.num_class as f32;
        check_labels(
            self.name(),
            info,
            |y| y >= 0.0 && y < k && y.floor() == y,
            "label must be a whole number in [0, num_class)",
        )
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
    fn hinge_only_penalises_inside_the_margin() {
        let mut out = Vec::new();
        // Label 1 (y = +1): a margin above 1 is already right.
        BinaryHinge.get_gradient(&[2.0, 0.5], &info(&[1.0, 1.0]), 0, &mut out);
        assert_eq!(out[0], GradientPair { grad: 0.0, hess: f32::MIN_POSITIVE });
        assert_eq!(out[1], GradientPair { grad: -1.0, hess: 1.0 });
    }

    #[test]
    fn hinge_predicts_zero_or_one() {
        let mut preds = vec![-3.0f32, 0.0, 0.25];
        BinaryHinge.pred_transform(&mut preds);
        assert_eq!(preds, vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn logitraw_reports_the_margin_but_learns_the_probability() {
        let obj = LogitRaw::new(1.0);
        let mut preds = vec![2.5f32];
        obj.pred_transform(&mut preds);
        assert_eq!(preds, vec![2.5], "logitraw does not transform");

        let mut out = Vec::new();
        obj.get_gradient(&[0.0], &info(&[1.0]), 0, &mut out);
        assert_eq!(out[0].grad, -0.5, "sigmoid(0) - 1");
    }

    #[test]
    fn softmax_reduces_to_a_class_index_and_softprob_does_not() {
        let softmax = SoftmaxMultiClass::new(3, false);
        let mut preds = vec![0.1f32, 5.0, 0.2, 4.0, 0.0, 0.0];
        softmax.pred_transform(&mut preds);
        assert_eq!(preds, vec![1.0, 0.0], "the largest margin per row");

        let softprob = SoftmaxMultiClass::new(3, true);
        let mut preds = vec![0.0f32, 0.0, 0.0];
        softprob.pred_transform(&mut preds);
        for p in &preds {
            assert!((p - 1.0 / 3.0).abs() < 1e-6, "{preds:?}");
        }
    }

    #[test]
    fn softmax_metrics_always_see_probabilities() {
        let softmax = SoftmaxMultiClass::new(2, false);
        let mut preds = vec![0.0f32, 0.0];
        softmax.eval_transform(&mut preds);
        assert_eq!(preds, vec![0.5, 0.5]);
    }

    #[test]
    fn softmax_gradient_pushes_the_true_class_up() {
        let obj = SoftmaxMultiClass::new(3, true);
        let mut out = Vec::new();
        obj.get_gradient(&[0.0, 0.0, 0.0], &info(&[2.0]), 0, &mut out);
        assert!(out[2].grad < 0.0, "the true class gets a negative gradient");
        assert!(out[0].grad > 0.0 && out[1].grad > 0.0);
        // The gradients of one row sum to zero: probabilities sum to one.
        let sum: f32 = out.iter().map(|p| p.grad).sum();
        assert!(sum.abs() < 1e-6, "{sum}");
    }

    #[test]
    fn softmax_intercept_is_the_centred_log_frequency() {
        let obj = SoftmaxMultiClass::new(2, true);
        // Balanced classes give equal, zero-centred margins.
        let intercept = obj.init_estimation(&info(&[0.0, 1.0]));
        assert_eq!(intercept.len(), 2);
        for v in &intercept {
            assert!(v.abs() < 1e-6, "{intercept:?}");
        }
        // An imbalance moves the frequent class up and the rare one down.
        let skewed = obj.init_estimation(&info(&[0.0, 0.0, 0.0, 1.0]));
        assert!(skewed[0] > 0.0 && skewed[1] < 0.0, "{skewed:?}");
    }

    #[test]
    fn multiclass_labels_must_be_class_indices() {
        let obj = SoftmaxMultiClass::new(3, false);
        assert!(obj.validate_data(&info(&[0.0, 2.0])).is_ok());
        assert!(obj.validate_data(&info(&[3.0])).is_err(), "out of range");
        assert!(obj.validate_data(&info(&[1.5])).is_err(), "not a whole number");
    }
}
