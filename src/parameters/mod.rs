//! The XGBoost fit ("training") and prediction parameter surface, for CPU and
//! GPU.
//!
//! Everything XGBoost's `Learner::Configure` accepts is represented here as a
//! typed value with a consuming builder, validated up front, and convertible
//! back to the `(name, value)` strings XGBoost itself consumes. The prediction
//! arguments are covered too, in the JSON form the C API's predict entry points
//! take rather than as `Learner` parameters.
//!
//! # Layout
//!
//! The grouping follows the upstream documentation, so a parameter is where a
//! reader of the XGBoost docs expects it:
//!
//! | Group | Type | Upstream owner |
//! |---|---|---|
//! | General | [`GeneralParameters`] | `Context`, `GlobalConfiguration` |
//! | Tree booster | [`TreeBoosterParameters`] | `tree::TrainParam`, `GBTreeTrainParam`, `HistMakerTrainParam`, `ColMakerTrainParam` |
//! | DART booster | [`DartParameters`] | `DartTrainParam` |
//! | Linear booster | [`LinearBoosterParameters`] | `LinearTrainParam`, `GBLinearTrainParam`, `CoordinateParam` |
//! | Learning task | [`LearningTaskParameters`] | `LearnerTrainParam`, `LearnerModelParamLegacy`, per-objective params |
//! | Training loop | [`TrainingParameters`] | `xgboost.train` arguments |
//! | Prediction | [`PredictParameters`], [`InplacePredictParameters`] | `Booster.predict` / `Booster.inplace_predict` arguments |
//!
//! [`BoosterParameters`] composes the first five; [`TrainingParameters`] wraps
//! that with the loop controls. Prediction stands apart: it configures a call
//! against an already-trained model, not the `Learner`, so it is emitted as
//! JSON by [`PredictParameters::to_predict_config`] instead of through
//! [`ToConfig`].
//!
//! # CPU and GPU
//!
//! The device is one parameter, [`Device`], and it changes what the rest mean.
//! Three methods make that explicit rather than leaving it to a later surprise:
//!
//! * [`TreeBoosterParameters::resolved_updaters`] reproduces
//!   `MapTreeMethodToUpdaters`, so `hist` becomes `grow_quantile_histmaker` on
//!   CPU and `grow_gpu_hist` on CUDA, and `exact` is rejected outright on GPU.
//! * [`TreeBoosterParameters::max_cached_hist_nodes`] resolves the histogram
//!   cache size to its device-dependent default (65536 on CPU, 4096 on CUDA).
//! * [`PredictionType::supports_device`] reports that approximated SHAP values
//!   and interactions exist only in the CPU predictor.
//!
//! [`BoosterParameters::validate`] runs every device-dependent fit check and
//! [`PredictParameters::validate_with`] every device-dependent prediction
//! check, so an invalid CPU/GPU combination fails at build time rather than
//! mid-fit or mid-prediction.
//!
//! # Example
//!
//! ```
//! use xgboost_rs::parameters::{
//!     BoosterParameters, Device, EvalMetric, GeneralParameters, LearningTaskParameters,
//!     Objective, SamplingMethod, ToConfig, TreeBoosterParameters, TreeMethod,
//! };
//!
//! let params = BoosterParameters::builder()
//!     .general(
//!         GeneralParameters::builder()
//!             .device(Device::cuda(0))
//!             .nthread(8)
//!             .build()?,
//!     )
//!     .tree(
//!         TreeBoosterParameters::builder()
//!             .tree_method(TreeMethod::Hist)
//!             .eta(0.1)
//!             .max_depth(8)
//!             .subsample(0.8)
//!             .colsample_bytree(0.8)
//!             .sampling_method(SamplingMethod::GradientBased)
//!             .build()?,
//!     )
//!     .learning(
//!         LearningTaskParameters::builder()
//!             .objective(Objective::BinaryLogistic)
//!             .eval_metric([EvalMetric::Logloss, EvalMetric::Auc])
//!             .seed(0)
//!             .build()?,
//!     )
//!     .build()?;
//!
//! let config = params.to_config_map();
//! assert_eq!(config["device"], "cuda:0");
//! assert_eq!(config["booster"], "gbtree");
//! assert_eq!(config["eval_metric"], "logloss,auc");
//! # Ok::<(), xgboost_rs::Error>(())
//! ```

mod config;
mod device;
mod general;
mod learning;
mod linear;
mod predict;
mod str_enum;
mod training;
mod tree;
mod validate;

use serde::{Deserialize, Serialize};

pub use config::{ConfigEntry, ToConfig};
pub use device::{Device, SyclKind};
pub use general::{GeneralParameters, GeneralParametersBuilder, Verbosity};
pub use learning::{
    AftDistribution, EvalMetric, LambdaRankPairMethod, LambdaRankParameters,
    LearningTaskParameters, LearningTaskParametersBuilder, Objective,
};
pub use linear::{
    FeatureSelector, LinearBoosterParameters, LinearBoosterParametersBuilder, LinearUpdater,
};
pub use predict::{
    InplacePredictParameters, InplacePredictParametersBuilder, IterationRange, PredictParameters,
    PredictParametersBuilder, PredictionType,
};
pub use training::{TrainingParameters, TrainingParametersBuilder, VerboseEval};
pub use tree::{
    DartNormalizeType, DartParameters, DartParametersBuilder, DartSampleType, DefaultDirection,
    GrowPolicy, MonotoneConstraint, MultiStrategy, ProcessType, SamplingMethod,
    TreeBoosterParameters, TreeBoosterParametersBuilder, TreeMethod, TreeUpdaterName,
};

use config::{ConfigEntry as Entry, push};
use crate::error::{Error, Result};

/// Which booster the fit uses, along with that booster's parameters.
///
/// The `booster` parameter and the booster's own parameters cannot disagree,
/// because they are the same value.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoosterType {
    /// `gbtree` — gradient boosted trees. XGBoost's default.
    Gbtree(TreeBoosterParameters),
    /// `dart` — gradient boosted trees with dropout.
    Dart(DartParameters),
    /// `gblinear` — a regularised linear model.
    Gblinear(LinearBoosterParameters),
}

impl Default for BoosterType {
    fn default() -> Self {
        Self::Gbtree(TreeBoosterParameters::default())
    }
}

impl BoosterType {
    /// The XGBoost `booster` string.
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Gbtree(_) => "gbtree",
            Self::Dart(_) => "dart",
            Self::Gblinear(_) => "gblinear",
        }
    }

    /// The tree parameters, for the two tree boosters.
    pub const fn tree(&self) -> Option<&TreeBoosterParameters> {
        match self {
            Self::Gbtree(tree) => Some(tree),
            Self::Dart(dart) => Some(&dart.tree),
            Self::Gblinear(_) => None,
        }
    }

    /// Validate this booster's own parameters.
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Gbtree(tree) => tree.validate(),
            Self::Dart(dart) => dart.validate(),
            Self::Gblinear(linear) => linear.validate(),
        }
    }
}

impl ToConfig for BoosterType {
    fn collect_config(&self, out: &mut Vec<Entry>) {
        push(out, "booster", self.name());
        match self {
            Self::Gbtree(tree) => tree.collect_config(out),
            Self::Dart(dart) => dart.collect_config(out),
            Self::Gblinear(linear) => linear.collect_config(out),
        }
    }
}

/// Everything XGBoost's `Learner` is configured with: general parameters, the
/// booster and its parameters, and the learning task.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BoosterParameters {
    /// Parameters that apply whatever the booster and objective.
    pub general: GeneralParameters,
    /// The booster and its own parameters.
    pub booster: BoosterType,
    /// The objective, its knobs, and the evaluation metrics.
    pub learning: LearningTaskParameters,
}

impl BoosterParameters {
    /// Start from XGBoost's defaults: CPU `gbtree`, `reg:squarederror`.
    pub fn builder() -> BoosterParametersBuilder {
        BoosterParametersBuilder::default()
    }

    /// The updater pipeline the fit will run, or an error if the requested
    /// tree method cannot run on the requested device.
    ///
    /// Returns `None` for `gblinear`, which has its own solver rather than a
    /// tree updater sequence.
    pub fn resolved_updaters(&self) -> Result<Option<Vec<TreeUpdaterName>>> {
        match self.booster.tree() {
            Some(tree) => tree.resolved_updaters(self.general.device).map(Some),
            None => Ok(None),
        }
    }

    /// Validate every group and every cross-group rule, including the
    /// device-dependent ones.
    pub fn validate(&self) -> Result<()> {
        self.general.validate()?;
        self.booster.validate()?;
        self.learning.validate()?;

        // The only place a tree method and a device meet: resolving the
        // updater sequence is exactly the check XGBoost performs at configure
        // time, so reuse it rather than restating the rule.
        self.resolved_updaters()?;

        if matches!(self.booster, BoosterType::Gblinear(_)) && !self.general.device.is_cpu() {
            return Err(Error::invalid(
                "device",
                format!(
                    "the `gblinear` booster has no GPU implementation; got device `{}`",
                    self.general.device
                ),
            ));
        }
        Ok(())
    }

    /// Warnings XGBoost would print for this configuration. Nothing here is
    /// fatal; surfacing them lets a caller log the same advice upstream gives.
    pub fn warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        if let Some(tree) = self.booster.tree()
            && tree.updater.is_some()
        {
            warnings.push(
                "`updater` is set explicitly, so `tree_method` is ignored; prefer setting \
                 `tree_method` alone unless you need a custom pipeline"
                    .to_owned(),
            );
        }
        if self.learning.objective == Objective::RegLinear {
            warnings.push(
                "`reg:linear` is deprecated; use `reg:squarederror` instead".to_owned(),
            );
        }
        if let Some(tree) = self.booster.tree()
            && tree.subsample == 1.0
            && tree.sampling_method != SamplingMethod::Uniform
        {
            warnings.push(
                "`sampling_method` has no effect while `subsample` is 1".to_owned(),
            );
        }
        warnings
    }
}

impl ToConfig for BoosterParameters {
    fn collect_config(&self, out: &mut Vec<Entry>) {
        self.general.collect_config(out);
        self.booster.collect_config(out);
        self.learning.collect_config(out);
    }
}

/// Consuming builder for [`BoosterParameters`].
#[derive(Clone, Debug, Default)]
pub struct BoosterParametersBuilder {
    inner: BoosterParameters,
}

impl BoosterParametersBuilder {
    /// General parameters.
    pub fn general(mut self, general: GeneralParameters) -> Self {
        self.inner.general = general;
        self
    }

    /// Use the `gbtree` booster with these parameters.
    pub fn tree(mut self, tree: TreeBoosterParameters) -> Self {
        self.inner.booster = BoosterType::Gbtree(tree);
        self
    }

    /// Use the `dart` booster with these parameters.
    pub fn dart(mut self, dart: DartParameters) -> Self {
        self.inner.booster = BoosterType::Dart(dart);
        self
    }

    /// Use the `gblinear` booster with these parameters.
    pub fn linear(mut self, linear: LinearBoosterParameters) -> Self {
        self.inner.booster = BoosterType::Gblinear(linear);
        self
    }

    /// Set the booster directly.
    pub fn booster(mut self, booster: BoosterType) -> Self {
        self.inner.booster = booster;
        self
    }

    /// Learning-task parameters.
    pub fn learning(mut self, learning: LearningTaskParameters) -> Self {
        self.inner.learning = learning;
        self
    }

    /// Validate and produce the parameters.
    pub fn build(self) -> Result<BoosterParameters> {
        self.inner.validate()?;
        Ok(self.inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_cpu_gbtree_squared_error() {
        let params = BoosterParameters::default();
        assert_eq!(params.booster.name(), "gbtree");
        assert_eq!(params.general.device, Device::Cpu);
        assert_eq!(params.learning.objective, Objective::RegSquaredError);
        params.validate().unwrap();

        let config = params.to_config_map();
        assert_eq!(config["booster"], "gbtree");
        assert_eq!(config["device"], "cpu");
        assert_eq!(config["objective"], "reg:squarederror");
        assert_eq!(config["tree_method"], "auto");
    }

    #[test]
    fn rejects_exact_on_gpu_only_when_the_device_is_a_gpu() {
        let exact = TreeBoosterParameters::builder().tree_method(TreeMethod::Exact).build().unwrap();

        BoosterParameters::builder().tree(exact.clone()).build().unwrap();

        let err = BoosterParameters::builder()
            .general(GeneralParameters::builder().device(Device::cuda(0)).build().unwrap())
            .tree(exact)
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("exact"), "{err}");
    }

    #[test]
    fn rejects_gblinear_on_gpu() {
        let err = BoosterParameters::builder()
            .general(GeneralParameters::builder().device(Device::cuda(0)).build().unwrap())
            .linear(LinearBoosterParameters::default())
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("gblinear"), "{err}");

        BoosterParameters::builder().linear(LinearBoosterParameters::default()).build().unwrap();
    }

    #[test]
    fn resolves_updaters_for_both_devices() {
        let cpu = BoosterParameters::default();
        assert_eq!(
            cpu.resolved_updaters().unwrap(),
            Some(vec![TreeUpdaterName::GrowQuantileHistMaker])
        );

        let gpu = BoosterParameters::builder()
            .general(GeneralParameters::builder().device(Device::cuda(1)).build().unwrap())
            .build()
            .unwrap();
        assert_eq!(gpu.resolved_updaters().unwrap(), Some(vec![TreeUpdaterName::GrowGpuHist]));

        let linear =
            BoosterParameters::builder().linear(LinearBoosterParameters::default()).build().unwrap();
        assert_eq!(linear.resolved_updaters().unwrap(), None);
    }

    #[test]
    fn booster_type_drives_the_booster_parameter() {
        let dart = BoosterParameters::builder()
            .dart(DartParameters::builder().rate_drop(0.1).build().unwrap())
            .build()
            .unwrap();
        let config = dart.to_config_map();
        assert_eq!(config["booster"], "dart");
        assert_eq!(config["rate_drop"], "0.1");
        // DART still carries every tree parameter.
        assert_eq!(config["max_depth"], "6");

        let linear = BoosterParameters::builder()
            .linear(LinearBoosterParameters::default())
            .build()
            .unwrap();
        let config = linear.to_config_map();
        assert_eq!(config["booster"], "gblinear");
        assert_eq!(config["feature_selector"], "cyclic");
        // No tree parameters leak into a linear fit.
        assert!(!config.contains_key("max_depth"));
        assert!(!config.contains_key("tree_method"));
    }

    #[test]
    fn reports_non_fatal_warnings() {
        let explicit_updater = BoosterParameters::builder()
            .tree(
                TreeBoosterParameters::builder()
                    .updater([TreeUpdaterName::GrowQuantileHistMaker])
                    .build()
                    .unwrap(),
            )
            .build()
            .unwrap();
        assert!(explicit_updater.warnings().iter().any(|w| w.contains("updater")));

        let ineffective_sampling = BoosterParameters::builder()
            .tree(
                TreeBoosterParameters::builder()
                    .sampling_method(SamplingMethod::GradientBased)
                    .build()
                    .unwrap(),
            )
            .build()
            .unwrap();
        assert!(ineffective_sampling.warnings().iter().any(|w| w.contains("subsample")));

        assert!(BoosterParameters::default().warnings().is_empty());
    }
}
