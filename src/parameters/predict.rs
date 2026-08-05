//! The XGBoost prediction ("predict") parameter surface, for CPU and GPU.
//!
//! These are the arguments of `Booster.predict` and `Booster.inplace_predict`
//! (`python-package/xgboost/core.py`). Unlike every other group in this module
//! they are *not* `Learner` parameters, so they are deliberately outside
//! [`ToConfig`](super::ToConfig): they reach XGBoost as the JSON configuration
//! string of `XGBoosterPredictFromDMatrix` / `XGBoosterPredictFrom*`, whose
//! `RequiredArg<Integer>` / `RequiredArg<Boolean>` lookups demand real JSON
//! integers and booleans rather than the strings `Learner::SetParam` takes.
//! [`PredictParameters::to_predict_config`] and
//! [`InplacePredictParameters::to_inplace_config`] emit exactly that JSON.
//!
//! # CPU and GPU
//!
//! Two prediction paths differ by device, and both are checked by
//! [`PredictParameters::validate_with`] rather than left to a `LOG(FATAL)`
//! mid-prediction:
//!
//! * Approximated contributions and interactions
//!   ([`PredictionType::ApproxContribution`], [`PredictionType::ApproxInteraction`])
//!   exist only in the CPU predictor; `GPUPredictor::PredictContribution` in
//!   `src/predictor/gpu_predictor.cu` aborts on them.
//! * The `gblinear` booster has no leaves and no prediction range, so
//!   [`PredictionType::Leaf`] and a non-zero [`IterationRange::begin`] are
//!   rejected for it (`LinearCheckLayer` in `src/gbm/gblinear.cc`).

use std::fmt;

use serde::{Deserialize, Serialize};

use super::device::Device;
use super::str_enum::str_enum;
use super::{BoosterParameters, BoosterType};
use crate::error::{Error, Result};

/// The XGBoost name used in errors about the round range.
const ITERATION_RANGE: &str = "iteration_range";
/// The XGBoost name used in errors about the `training` flag.
const TRAINING: &str = "training";
/// The XGBoost name used in errors about the missing-value marker.
const MISSING: &str = "missing";

str_enum! {
    /// What a prediction call returns.
    ///
    /// This is `xgboost::PredictionType` (`include/xgboost/learner.h`), which
    /// crosses the C API as the integer `type` field — see
    /// [`PredictionType::as_code`]. Python's `Booster.predict` instead takes
    /// one boolean per kind and folds them into the same value; the
    /// [`output_margin`](Self::output_margin), [`pred_leaf`](Self::pred_leaf),
    /// [`pred_contribs`](Self::pred_contribs),
    /// [`pred_interactions`](Self::pred_interactions) and
    /// [`approx_contribs`](Self::approx_contribs) accessors reproduce that
    /// decomposition, so exactly one kind of prediction can ever be requested
    /// (`CHECK_LE(multiple_predictions, 1)` in `src/learner.cc`) by
    /// construction.
    ///
    /// ```
    /// use xgboost_rs::parameters::PredictionType;
    ///
    /// assert_eq!(PredictionType::Leaf.as_code(), 6);
    /// assert_eq!(PredictionType::from_code(3).unwrap(), PredictionType::ApproxContribution);
    /// assert!(PredictionType::ApproxInteraction.pred_interactions());
    /// assert!(PredictionType::ApproxInteraction.approx_contribs());
    /// ```
    pub enum PredictionType : "predict_type" {
        /// The model's prediction with the objective's transform applied
        /// (probabilities for the logistic objectives, class labels for
        /// `multi:softmax`, …). XGBoost's default.
        Value = "value",
        /// The raw, untransformed margin.
        Margin = "margin",
        /// Exact SHAP values: one contribution per feature plus a bias column.
        Contribution = "contribution",
        /// Approximated SHAP values. CPU only.
        ApproxContribution = "approx_contribution",
        /// Exact SHAP interaction values, one matrix per sample.
        Interaction = "interaction",
        /// Approximated SHAP interaction values. CPU only.
        ApproxInteraction = "approx_interaction",
        /// The leaf index each sample reaches in each tree.
        Leaf = "leaf",
    }
    default = Value;
}

impl PredictionType {
    /// The integer the C API `type` field carries, matching the `kValue = 0` …
    /// `kLeaf = 6` numbering of `xgboost::PredictionType`.
    pub const fn as_code(self) -> u8 {
        match self {
            Self::Value => 0,
            Self::Margin => 1,
            Self::Contribution => 2,
            Self::ApproxContribution => 3,
            Self::Interaction => 4,
            Self::ApproxInteraction => 5,
            Self::Leaf => 6,
        }
    }

    /// Parse a C API `type` integer back into a typed value.
    pub fn from_code(code: u8) -> Result<Self> {
        Self::ALL.iter().copied().find(|kind| kind.as_code() == code).ok_or_else(|| {
            Error::parse(Self::parameter_name(), code.to_string(), "expected an integer in [0, 6]")
        })
    }

    /// `output_margin` as `Learner::Predict` takes it.
    pub const fn output_margin(self) -> bool {
        matches!(self, Self::Margin)
    }

    /// `pred_contribs` as `Learner::Predict` takes it.
    pub const fn pred_contribs(self) -> bool {
        matches!(self, Self::Contribution | Self::ApproxContribution)
    }

    /// `pred_interactions` as `Learner::Predict` takes it.
    pub const fn pred_interactions(self) -> bool {
        matches!(self, Self::Interaction | Self::ApproxInteraction)
    }

    /// `pred_leaf` as `Learner::Predict` takes it.
    pub const fn pred_leaf(self) -> bool {
        matches!(self, Self::Leaf)
    }

    /// `approx_contribs` as `Learner::Predict` takes it. Meaningful only
    /// alongside [`pred_contribs`](Self::pred_contribs) or
    /// [`pred_interactions`](Self::pred_interactions), which is why it cannot
    /// be set independently here.
    pub const fn approx_contribs(self) -> bool {
        matches!(self, Self::ApproxContribution | Self::ApproxInteraction)
    }

    /// Whether this asks for SHAP values or SHAP interaction values.
    pub const fn is_shap(self) -> bool {
        self.pred_contribs() || self.pred_interactions()
    }

    /// The exact counterpart of an approximated kind; every other kind is its
    /// own counterpart. Used to point at the alternative when the approximated
    /// kind is unavailable on the requested device.
    pub const fn exact(self) -> Self {
        match self {
            Self::ApproxContribution => Self::Contribution,
            Self::ApproxInteraction => Self::Interaction,
            other => other,
        }
    }

    /// Whether the `training` flag reaches this kind of prediction.
    ///
    /// `Learner::Predict` consults it only on the branch that neither SHAP nor
    /// leaf output takes, so `training` is silently ignored for those.
    pub const fn honours_training(self) -> bool {
        !self.is_shap() && !self.pred_leaf()
    }

    /// Whether in-place prediction can produce this kind.
    ///
    /// `Learner::InplacePredict` transforms for [`Value`](Self::Value), passes
    /// [`Margin`](Self::Margin) through, and aborts on everything else.
    pub const fn supports_inplace(self) -> bool {
        matches!(self, Self::Value | Self::Margin)
    }

    /// Whether this kind runs on `device`.
    ///
    /// Only the approximated SHAP kinds are device-restricted: the CUDA
    /// predictor has no implementation of them.
    pub fn supports_device(self, device: Device) -> bool {
        !(self.approx_contribs() && device.is_cuda())
    }
}

/// The `iteration_range` argument: which boosting rounds ("layers") to predict
/// with, as a half-open `[begin, end)` range of zero-based round indices.
///
/// `end == 0` means "through the last round the model has", exactly as
/// `detail::LayerToTree` in `src/gbm/gbtree.h` reads it, so [`Self::all`] and
/// `(0, 0)` are the same request. Restricting the range is how a caller
/// predicts with the best iteration found by early stopping.
///
/// ```
/// use xgboost_rs::parameters::IterationRange;
///
/// // The whole model, whatever its size.
/// assert_eq!(IterationRange::all().resolve(120).unwrap(), (0, 120));
/// // The first 42 rounds, as early stopping would ask for.
/// assert_eq!(IterationRange::new(0, 42).unwrap().resolve(120).unwrap(), (0, 42));
/// // Asking for more rounds than the model has is an error, not a silent clamp.
/// assert!(IterationRange::new(0, 200).unwrap().resolve(120).is_err());
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IterationRange {
    /// First round used, inclusive.
    pub begin: u32,
    /// One past the last round used; `0` means "through the last round".
    pub end: u32,
}

impl IterationRange {
    /// Every round the model has — XGBoost's `(0, 0)` default.
    pub const fn all() -> Self {
        Self { begin: 0, end: 0 }
    }

    /// A validated `[begin, end)` range. Pass `end = 0` for "to the end".
    pub fn new(begin: u32, end: u32) -> Result<Self> {
        let range = Self { begin, end };
        range.validate()?;
        Ok(range)
    }

    /// Whether this is the whole model.
    pub const fn is_all(self) -> bool {
        self.begin == 0 && self.end == 0
    }

    /// Check the range on its own, without a model to size it against.
    pub fn validate(self) -> Result<()> {
        if self.end != 0 && self.end < self.begin {
            return Err(Error::invalid(
                ITERATION_RANGE,
                format!("end must be >= begin or 0 for \"to the end\", got {self}"),
            ));
        }
        Ok(())
    }

    /// Resolve against a model with `boosted_rounds` rounds, returning the
    /// concrete `[begin, end)` layer range.
    ///
    /// This is `detail::LayerToTree`'s layer arithmetic: `end == 0` becomes
    /// `boosted_rounds`, and a range reaching past the model is rejected
    /// (`CHECK_LE(end, model.BoostedRounds())`, "Out of range for tree
    /// layers"). Unlike upstream, a `begin` past the resolved `end` is also
    /// rejected rather than indexing `iteration_indptr` out of bounds.
    pub fn resolve(self, boosted_rounds: u32) -> Result<(u32, u32)> {
        self.validate()?;
        let end = if self.end == 0 { boosted_rounds } else { self.end };
        if end > boosted_rounds {
            return Err(Error::invalid(
                ITERATION_RANGE,
                format!("{self} is out of range for a model with {boosted_rounds} boosted rounds"),
            ));
        }
        if self.begin > end {
            return Err(Error::invalid(
                ITERATION_RANGE,
                format!(
                    "begin {} is past the last of {boosted_rounds} boosted rounds",
                    self.begin
                ),
            ));
        }
        Ok((self.begin, end))
    }
}

impl fmt::Display for IterationRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.begin, self.end)
    }
}

/// Everything `Booster.predict` is configured with: what to predict, from which
/// rounds, and how to shape the result.
///
/// ```
/// use xgboost_rs::parameters::{
///     BoosterParameters, Device, GeneralParameters, IterationRange, PredictParameters,
///     PredictionType,
/// };
///
/// let params = PredictParameters::builder()
///     .predict_type(PredictionType::Contribution)
///     .iteration_range(IterationRange::new(0, 42)?)
///     .build()?;
///
/// assert_eq!(
///     params.to_predict_config(),
///     r#"{"type":2,"training":false,"iteration_begin":0,"iteration_end":42,"strict_shape":false}"#
/// );
///
/// // Exact SHAP values run on GPU; the approximated ones do not.
/// let gpu = BoosterParameters::builder()
///     .general(GeneralParameters::builder().device(Device::cuda(0)).build()?)
///     .build()?;
/// params.validate_with(&gpu)?;
/// assert!(
///     PredictParameters::builder()
///         .predict_type(PredictionType::ApproxContribution)
///         .build()?
///         .validate_with(&gpu)
///         .is_err()
/// );
/// # Ok::<(), xgboost_rs::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PredictParameters {
    /// What the call returns.
    pub predict_type: PredictionType,
    /// Which boosting rounds to use.
    pub iteration_range: IterationRange,
    /// Whether this prediction is part of a training loop — the gradient
    /// prediction a custom objective consumes, which for DART applies dropout
    /// and so differs from an inference-time prediction.
    pub training: bool,
    /// Give the output a shape that does not depend on whether the model is
    /// multi-class or multi-target.
    pub strict_shape: bool,
    /// Check that the data's feature names match the model's before predicting.
    /// A Python-side check, so it does not appear in the C API config.
    pub validate_features: bool,
}

impl Default for PredictParameters {
    fn default() -> Self {
        Self {
            predict_type: PredictionType::Value,
            iteration_range: IterationRange::all(),
            training: false,
            strict_shape: false,
            // `Booster.predict`'s own default.
            validate_features: true,
        }
    }
}

impl PredictParameters {
    /// Start from XGBoost's defaults: transformed values from the whole model.
    pub fn builder() -> PredictParametersBuilder {
        PredictParametersBuilder::default()
    }

    /// Check every field that can be checked without a model or a booster.
    pub fn validate(&self) -> Result<()> {
        self.iteration_range.validate()?;

        // `GBTree::PredictLeaf` insists on `tree_begin == 0`: leaf prediction
        // can truncate the model but not skip its first rounds.
        if self.predict_type.pred_leaf() && self.iteration_range.begin != 0 {
            return Err(Error::invalid(
                ITERATION_RANGE,
                format!(
                    "leaf prediction supports only ranges starting at 0, got {}; \
                     slice the model instead",
                    self.iteration_range
                ),
            ));
        }
        Ok(())
    }

    /// Check this prediction against the model it will run on, including every
    /// device-dependent rule.
    ///
    /// Also validates `booster` itself, so one call covers the whole
    /// configuration a prediction needs.
    pub fn validate_with(&self, booster: &BoosterParameters) -> Result<()> {
        self.validate()?;
        booster.validate()?;

        let device = booster.general.device;
        if !self.predict_type.supports_device(device) {
            return Err(Error::invalid(
                PredictionType::parameter_name(),
                format!(
                    "`{}` is implemented in the CPU predictor only; got device `{device}`. \
                     Use `{}` on this device, or predict with `device` set to `cpu`",
                    self.predict_type,
                    self.predict_type.exact()
                ),
            ));
        }

        if matches!(booster.booster, BoosterType::Gblinear(_)) {
            if self.predict_type.pred_leaf() {
                return Err(Error::invalid(
                    PredictionType::parameter_name(),
                    "the `gblinear` booster has no leaves, so leaf prediction is unavailable",
                ));
            }
            if self.iteration_range.begin != 0 {
                return Err(Error::invalid(
                    ITERATION_RANGE,
                    format!(
                        "the `gblinear` booster does not support a prediction range, so begin \
                         must be 0; got {}",
                        self.iteration_range
                    ),
                ));
            }
        }
        Ok(())
    }

    /// Caveats XGBoost would not report until it was too late. Nothing here is
    /// fatal on its own; each depends on a model this crate cannot see.
    pub fn warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        if self.training && !self.predict_type.honours_training() {
            warnings.push(
                "`training` affects value and margin prediction only; it is ignored for SHAP \
                 and leaf output"
                    .to_owned(),
            );
        }
        if self.strict_shape && self.predict_type.pred_leaf() {
            warnings.push(
                "`strict_shape` with leaf prediction is rejected for models with vector leaves \
                 (`multi_strategy = multi_output_tree`), because the strict leaf shape cannot \
                 express them"
                    .to_owned(),
            );
        }
        warnings
    }

    /// The JSON configuration `XGBoosterPredictFromDMatrix` expects.
    ///
    /// [`validate_features`](Self::validate_features) is deliberately absent:
    /// it is a check on the DMatrix's feature names that the Python package
    /// performs before calling the C API, not a field the C API reads.
    pub fn to_predict_config(&self) -> String {
        json_object(&[
            ("type", self.predict_type.as_code().to_string()),
            (TRAINING, json_bool(self.training)),
            ("iteration_begin", self.iteration_range.begin.to_string()),
            ("iteration_end", self.iteration_range.end.to_string()),
            ("strict_shape", json_bool(self.strict_shape)),
        ])
    }
}

/// Consuming builder for [`PredictParameters`].
#[derive(Clone, Copy, Debug, Default)]
pub struct PredictParametersBuilder {
    inner: PredictParameters,
}

impl PredictParametersBuilder {
    /// What the call returns.
    pub fn predict_type(mut self, predict_type: PredictionType) -> Self {
        self.inner.predict_type = predict_type;
        self
    }

    /// Which boosting rounds to use.
    pub fn iteration_range(mut self, iteration_range: IterationRange) -> Self {
        self.inner.iteration_range = iteration_range;
        self
    }

    /// Predict as part of a training loop (DART applies dropout).
    pub fn training(mut self, training: bool) -> Self {
        self.inner.training = training;
        self
    }

    /// Shape the output independently of the model's class/target count.
    pub fn strict_shape(mut self, strict_shape: bool) -> Self {
        self.inner.strict_shape = strict_shape;
        self
    }

    /// Check the data's feature names against the model's.
    pub fn validate_features(mut self, validate_features: bool) -> Self {
        self.inner.validate_features = validate_features;
        self
    }

    /// Validate and produce the parameters.
    pub fn build(self) -> Result<PredictParameters> {
        self.inner.validate()?;
        Ok(self.inner)
    }
}

/// Everything `Booster.inplace_predict` is configured with: the shared
/// prediction parameters plus the missing-value marker of the raw input, which
/// is not wrapped in a DMatrix and so carries no such marker of its own.
///
/// In-place prediction is the lock-free path that skips the DMatrix
/// construction and the prediction cache. It supports only
/// [`PredictionType::Value`] and [`PredictionType::Margin`]; the remaining
/// kinds need [`PredictParameters`] and a DMatrix.
///
/// ```
/// use xgboost_rs::parameters::{InplacePredictParameters, PredictionType};
///
/// let params = InplacePredictParameters::builder()
///     .predict_type(PredictionType::Margin)
///     .missing(-999.0)
///     .build()?;
///
/// assert_eq!(
///     params.to_inplace_config(),
///     r#"{"type":1,"training":false,"iteration_begin":0,"iteration_end":0,"missing":-999,"strict_shape":false,"cache_id":0}"#
/// );
///
/// // SHAP output needs the DMatrix path.
/// assert!(
///     InplacePredictParameters::builder()
///         .predict_type(PredictionType::Contribution)
///         .build()
///         .is_err()
/// );
/// # Ok::<(), xgboost_rs::Error>(())
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct InplacePredictParameters {
    /// The parameters shared with DMatrix prediction. `training` has no meaning
    /// here and must stay unset.
    pub predict: PredictParameters,
    /// The value in the input that means "missing". `None` selects XGBoost's
    /// default, NaN, which is also the only way to request NaN: `Some(NaN)` is
    /// rejected, since NaN compares unequal to itself and would make two
    /// identical configurations look different.
    pub missing: Option<f32>,
}

impl InplacePredictParameters {
    /// Start from XGBoost's defaults: transformed values, NaN means missing.
    pub fn builder() -> InplacePredictParametersBuilder {
        InplacePredictParametersBuilder::default()
    }

    /// Check every field, including the ones in-place prediction restricts
    /// further than DMatrix prediction does.
    pub fn validate(&self) -> Result<()> {
        self.predict.validate()?;

        if !self.predict.predict_type.supports_inplace() {
            return Err(Error::invalid(
                PredictionType::parameter_name(),
                format!(
                    "in-place prediction produces only `{}` or `{}`, got `{}`; predict from a \
                     DMatrix for SHAP and leaf output",
                    PredictionType::Value,
                    PredictionType::Margin,
                    self.predict.predict_type
                ),
            ));
        }

        // `Learner::InplacePredict` never reads the flag — the C API documents
        // it as "Not used for inplace prediction" — so setting it can only mean
        // the caller expects an effect they will not get. Upstream ignores it
        // silently; this crate rejects it.
        if self.predict.training {
            return Err(Error::invalid(
                TRAINING,
                "in-place prediction ignores `training`; leave it unset, or predict from a \
                 DMatrix if you need the training-time prediction",
            ));
        }

        if let Some(missing) = self.missing
            && !missing.is_finite()
        {
            return Err(Error::invalid(
                MISSING,
                format!("must be finite, got {missing}; use `None` for XGBoost's NaN default"),
            ));
        }
        Ok(())
    }

    /// Check this prediction against the model it will run on.
    pub fn validate_with(&self, booster: &BoosterParameters) -> Result<()> {
        self.validate()?;
        self.predict.validate_with(booster)
    }

    /// Caveats worth surfacing; see [`PredictParameters::warnings`].
    pub fn warnings(&self) -> Vec<String> {
        self.predict.warnings()
    }

    /// The JSON configuration the `XGBoosterPredictFrom*` in-place entry points
    /// expect.
    pub fn to_inplace_config(&self) -> String {
        json_object(&[
            ("type", self.predict.predict_type.as_code().to_string()),
            // Always false: `validate` rejects the alternative.
            (TRAINING, json_bool(false)),
            ("iteration_begin", self.predict.iteration_range.begin.to_string()),
            ("iteration_end", self.predict.iteration_range.end.to_string()),
            (MISSING, json_f32(self.missing.unwrap_or(f32::NAN))),
            ("strict_shape", json_bool(self.predict.strict_shape)),
            // Reserved for a future in-place prediction cache. The C API does
            // not read it yet; the Python package always sends 0, so do the
            // same rather than produce a config that differs from upstream's.
            ("cache_id", "0".to_owned()),
        ])
    }
}

/// Consuming builder for [`InplacePredictParameters`].
#[derive(Clone, Copy, Debug, Default)]
pub struct InplacePredictParametersBuilder {
    inner: InplacePredictParameters,
}

impl InplacePredictParametersBuilder {
    /// Set the shared prediction parameters wholesale.
    pub fn predict(mut self, predict: PredictParameters) -> Self {
        self.inner.predict = predict;
        self
    }

    /// What the call returns; only `value` and `margin` are available.
    pub fn predict_type(mut self, predict_type: PredictionType) -> Self {
        self.inner.predict.predict_type = predict_type;
        self
    }

    /// Which boosting rounds to use.
    pub fn iteration_range(mut self, iteration_range: IterationRange) -> Self {
        self.inner.predict.iteration_range = iteration_range;
        self
    }

    /// Shape the output independently of the model's class/target count.
    pub fn strict_shape(mut self, strict_shape: bool) -> Self {
        self.inner.predict.strict_shape = strict_shape;
        self
    }

    /// Check the data's feature names against the model's.
    pub fn validate_features(mut self, validate_features: bool) -> Self {
        self.inner.predict.validate_features = validate_features;
        self
    }

    /// The value in the input that means "missing". Leave unset for NaN.
    pub fn missing(mut self, missing: f32) -> Self {
        self.inner.missing = Some(missing);
        self
    }

    /// Validate and produce the parameters.
    pub fn build(self) -> Result<InplacePredictParameters> {
        self.inner.validate()?;
        Ok(self.inner)
    }
}

/// Render `entries` as a JSON object, preserving order.
///
/// The values are already JSON literals. Hand-rendering rather than reaching
/// for `serde_json` is what lets [`json_f32`] emit `NaN`, which `serde_json`
/// turns into `null` and XGBoost then rejects.
fn json_object(entries: &[(&str, String)]) -> String {
    let body = entries
        .iter()
        .map(|(key, value)| format!("\"{key}\":{value}"))
        .collect::<Vec<_>>()
        .join(",");
    format!("{{{body}}}")
}

/// A JSON boolean literal.
fn json_bool(value: bool) -> String {
    if value { "true" } else { "false" }.to_owned()
}

/// A JSON number literal, using the non-standard `NaN` / `Infinity` spellings
/// for the non-finite values.
///
/// `JsonReader::ParseNumber` in `src/common/json.cc` accepts exactly these —
/// `NaN` capitalised that way, because `nan` would collide with `null` in its
/// LR(1) parser — and they are what Python's `json.dumps` writes, so this is
/// byte-for-byte the config upstream sends.
fn json_f32(value: f32) -> String {
    if value.is_nan() {
        "NaN".to_owned()
    } else if value == f32::INFINITY {
        "Infinity".to_owned()
    } else if value == f32::NEG_INFINITY {
        "-Infinity".to_owned()
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameters::{GeneralParameters, LinearBoosterParameters};

    #[test]
    fn codes_match_upstream_prediction_type() {
        for (kind, code) in [
            (PredictionType::Value, 0),
            (PredictionType::Margin, 1),
            (PredictionType::Contribution, 2),
            (PredictionType::ApproxContribution, 3),
            (PredictionType::Interaction, 4),
            (PredictionType::ApproxInteraction, 5),
            (PredictionType::Leaf, 6),
        ] {
            assert_eq!(kind.as_code(), code);
            assert_eq!(PredictionType::from_code(code).unwrap(), kind);
        }
        assert!(PredictionType::from_code(7).is_err());
    }

    #[test]
    fn flags_request_exactly_one_kind_of_prediction() {
        // `Learner::Predict` aborts unless at most one of these is set.
        for kind in PredictionType::ALL.iter().copied() {
            let kinds = u8::from(kind.pred_leaf())
                + u8::from(kind.pred_contribs())
                + u8::from(kind.pred_interactions());
            assert!(kinds <= 1, "{kind} requests {kinds} kinds of prediction");
        }
        assert!(PredictionType::Margin.output_margin());
        assert!(!PredictionType::Value.output_margin());
        assert!(PredictionType::ApproxContribution.approx_contribs());
        assert!(!PredictionType::Contribution.approx_contribs());
        assert_eq!(PredictionType::ApproxInteraction.exact(), PredictionType::Interaction);
        assert_eq!(PredictionType::Leaf.exact(), PredictionType::Leaf);
    }

    #[test]
    fn iteration_range_resolves_the_way_layer_to_tree_does() {
        assert_eq!(IterationRange::all().resolve(10).unwrap(), (0, 10));
        assert_eq!(IterationRange::new(3, 0).unwrap().resolve(10).unwrap(), (3, 10));
        assert_eq!(IterationRange::new(3, 7).unwrap().resolve(10).unwrap(), (3, 7));
        // An empty range is legal; XGBoost only requires begin <= end.
        assert_eq!(IterationRange::new(4, 4).unwrap().resolve(10).unwrap(), (4, 4));

        assert!(IterationRange::new(7, 3).is_err());
        assert!(IterationRange::new(0, 11).unwrap().resolve(10).is_err());
        assert!(IterationRange::new(11, 0).unwrap().resolve(10).is_err());
        assert!(IterationRange::all().is_all());
        assert!(!IterationRange::new(0, 1).unwrap().is_all());
    }

    #[test]
    fn approximated_shap_is_cpu_only() {
        let cpu = BoosterParameters::default();
        let gpu = BoosterParameters::builder()
            .general(GeneralParameters::builder().device(Device::cuda(0)).build().unwrap())
            .build()
            .unwrap();

        for kind in [PredictionType::ApproxContribution, PredictionType::ApproxInteraction] {
            let params = PredictParameters::builder().predict_type(kind).build().unwrap();
            params.validate_with(&cpu).unwrap();
            let err = params.validate_with(&gpu).unwrap_err().to_string();
            assert!(err.contains("cuda:0"), "{err}");
            assert!(err.contains(kind.exact().as_str()), "{err}");
        }

        // The exact kinds run on both.
        for kind in [PredictionType::Contribution, PredictionType::Interaction] {
            let params = PredictParameters::builder().predict_type(kind).build().unwrap();
            params.validate_with(&cpu).unwrap();
            params.validate_with(&gpu).unwrap();
        }
    }

    #[test]
    fn leaf_prediction_cannot_skip_leading_rounds() {
        let err = PredictParameters::builder()
            .predict_type(PredictionType::Leaf)
            .iteration_range(IterationRange::new(2, 8).unwrap())
            .build()
            .unwrap_err()
            .to_string();
        assert!(err.contains(ITERATION_RANGE), "{err}");

        // Truncating is fine, it is only a non-zero begin that is rejected.
        PredictParameters::builder()
            .predict_type(PredictionType::Leaf)
            .iteration_range(IterationRange::new(0, 8).unwrap())
            .build()
            .unwrap();
    }

    #[test]
    fn gblinear_has_neither_leaves_nor_a_prediction_range() {
        let linear = BoosterParameters::builder()
            .linear(LinearBoosterParameters::default())
            .build()
            .unwrap();

        assert!(
            PredictParameters::builder()
                .predict_type(PredictionType::Leaf)
                .build()
                .unwrap()
                .validate_with(&linear)
                .is_err()
        );
        assert!(
            PredictParameters::builder()
                .iteration_range(IterationRange::new(1, 0).unwrap())
                .build()
                .unwrap()
                .validate_with(&linear)
                .is_err()
        );
        // Truncating to the first n rounds is still allowed.
        PredictParameters::builder()
            .iteration_range(IterationRange::new(0, 5).unwrap())
            .build()
            .unwrap()
            .validate_with(&linear)
            .unwrap();
    }

    #[test]
    fn inplace_prediction_restricts_the_kind_and_the_training_flag() {
        for kind in PredictionType::ALL.iter().copied() {
            let built =
                InplacePredictParameters::builder().predict_type(kind).build();
            assert_eq!(built.is_ok(), kind.supports_inplace(), "{kind}");
        }

        let training = InplacePredictParameters {
            predict: PredictParameters { training: true, ..PredictParameters::default() },
            missing: None,
        };
        let err = training.validate().unwrap_err().to_string();
        assert!(err.contains(TRAINING), "{err}");

        let nan = InplacePredictParameters { missing: Some(f32::NAN), ..Default::default() };
        let err = nan.validate().unwrap_err().to_string();
        assert!(err.contains(MISSING), "{err}");
    }

    #[test]
    fn configs_match_the_json_the_python_package_sends() {
        assert_eq!(
            PredictParameters::default().to_predict_config(),
            r#"{"type":0,"training":false,"iteration_begin":0,"iteration_end":0,"strict_shape":false}"#
        );
        assert_eq!(
            PredictParameters::builder()
                .predict_type(PredictionType::ApproxInteraction)
                .iteration_range(IterationRange::new(2, 9).unwrap())
                .training(true)
                .strict_shape(true)
                .build()
                .unwrap()
                .to_predict_config(),
            r#"{"type":5,"training":true,"iteration_begin":2,"iteration_end":9,"strict_shape":true}"#
        );

        // `missing` defaults to NaN, which JSON cannot spell but XGBoost reads.
        assert_eq!(
            InplacePredictParameters::default().to_inplace_config(),
            r#"{"type":0,"training":false,"iteration_begin":0,"iteration_end":0,"missing":NaN,"strict_shape":false,"cache_id":0}"#
        );
    }

    #[test]
    fn warnings_flag_what_only_a_model_could_reject() {
        let ignored_training = PredictParameters::builder()
            .predict_type(PredictionType::Leaf)
            .training(true)
            .build()
            .unwrap();
        assert!(ignored_training.warnings().iter().any(|w| w.contains("training")));

        let strict_leaf = PredictParameters::builder()
            .predict_type(PredictionType::Leaf)
            .strict_shape(true)
            .build()
            .unwrap();
        assert!(strict_leaf.warnings().iter().any(|w| w.contains("vector leaves")));

        assert!(PredictParameters::default().warnings().is_empty());
    }
}
