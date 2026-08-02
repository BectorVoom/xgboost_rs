//! General parameters — the ones that apply whatever booster and objective the
//! fit uses. Upstream these live in `Context` (`include/xgboost/context.h`) and
//! `GlobalConfiguration` (`include/xgboost/global_config.h`).

use serde::{Deserialize, Serialize};

use super::config::{ConfigEntry, ToConfig, push, push_bool};
use super::device::Device;
use super::str_enum::str_enum;
use crate::error::Result;

str_enum! {
    /// How much XGBoost prints while fitting.
    pub enum Verbosity: "verbosity" {
        /// Nothing.
        Silent = "0",
        /// Warnings only (XGBoost's default).
        Warning = "1",
        /// Warnings plus progress information.
        Info = "2",
        /// Everything, including per-stage timings.
        Debug = "3",
    }
    default = Warning;
}

/// General parameters.
///
/// Defaults match XGBoost's C++ defaults. Note that `validate_parameters`
/// defaults to `false` here as it does in C++ — the Python package flips it to
/// `true`, but this crate validates eagerly in [`build`](GeneralParametersBuilder::build)
/// anyway, so the flag only affects a downstream real-XGBoost run fed from
/// [`ToConfig`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GeneralParameters {
    /// Which processor runs the fit: `cpu`, `cuda`, `cuda:<ordinal>`, …
    pub device: Device,
    /// Verbosity of XGBoost's own logging.
    pub verbosity: Verbosity,
    /// Number of threads. `0` means "all cores visible to the process".
    pub nthread: u32,
    /// Ask XGBoost to warn about configuration entries nothing consumed.
    pub validate_parameters: bool,
    /// Suppress the objective's default evaluation metric.
    pub disable_default_eval_metric: bool,
    /// Fail instead of silently falling back to CPU when the requested CUDA
    /// ordinal does not exist.
    pub fail_on_invalid_gpu_id: bool,
    /// Allocate GPU memory through the RAPIDS Memory Manager. Requires an
    /// XGBoost build with RMM support.
    pub use_rmm: bool,
}

impl Default for GeneralParameters {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            verbosity: Verbosity::Warning,
            nthread: 0,
            validate_parameters: false,
            disable_default_eval_metric: false,
            fail_on_invalid_gpu_id: false,
            use_rmm: false,
        }
    }
}

impl GeneralParameters {
    /// Start from XGBoost's defaults.
    pub fn builder() -> GeneralParametersBuilder {
        GeneralParametersBuilder::default()
    }

    /// Check every field. There is nothing to reject that the type system has
    /// not already ruled out, but the method exists so callers that mutate the
    /// public fields directly can re-check without going through the builder.
    pub fn validate(&self) -> Result<()> {
        Ok(())
    }
}

impl ToConfig for GeneralParameters {
    fn collect_config(&self, out: &mut Vec<ConfigEntry>) {
        push(out, "device", self.device);
        push(out, "verbosity", self.verbosity);
        push(out, "nthread", self.nthread);
        push_bool(out, "validate_parameters", self.validate_parameters);
        push_bool(out, "disable_default_eval_metric", self.disable_default_eval_metric);
        push_bool(out, "fail_on_invalid_gpu_id", self.fail_on_invalid_gpu_id);
        push_bool(out, "use_rmm", self.use_rmm);
    }
}

/// Consuming builder for [`GeneralParameters`].
#[derive(Clone, Debug, Default)]
pub struct GeneralParametersBuilder {
    inner: GeneralParameters,
}

impl GeneralParametersBuilder {
    /// Which processor runs the fit.
    pub fn device(mut self, device: Device) -> Self {
        self.inner.device = device;
        self
    }

    /// Verbosity of XGBoost's own logging.
    pub fn verbosity(mut self, verbosity: Verbosity) -> Self {
        self.inner.verbosity = verbosity;
        self
    }

    /// Number of threads; `0` means all available cores.
    pub fn nthread(mut self, nthread: u32) -> Self {
        self.inner.nthread = nthread;
        self
    }

    /// Ask XGBoost to warn about unused configuration entries.
    pub fn validate_parameters(mut self, validate: bool) -> Self {
        self.inner.validate_parameters = validate;
        self
    }

    /// Suppress the objective's default evaluation metric.
    pub fn disable_default_eval_metric(mut self, disable: bool) -> Self {
        self.inner.disable_default_eval_metric = disable;
        self
    }

    /// Fail rather than fall back to CPU on an invalid CUDA ordinal.
    pub fn fail_on_invalid_gpu_id(mut self, fail: bool) -> Self {
        self.inner.fail_on_invalid_gpu_id = fail;
        self
    }

    /// Allocate GPU memory through the RAPIDS Memory Manager.
    pub fn use_rmm(mut self, use_rmm: bool) -> Self {
        self.inner.use_rmm = use_rmm;
        self
    }

    /// Validate and produce the parameters.
    pub fn build(self) -> Result<GeneralParameters> {
        self.inner.validate()?;
        Ok(self.inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_xgboost() {
        let params = GeneralParameters::default();
        assert_eq!(params.device, Device::Cpu);
        assert_eq!(params.verbosity, Verbosity::Warning);
        assert_eq!(params.nthread, 0);
        assert!(!params.validate_parameters);
    }

    #[test]
    fn emits_upstream_parameter_names() {
        let config = GeneralParameters::builder()
            .device(Device::cuda(1))
            .verbosity(Verbosity::Debug)
            .nthread(8)
            .build()
            .unwrap()
            .to_config_map();
        assert_eq!(config["device"], "cuda:1");
        assert_eq!(config["verbosity"], "3");
        assert_eq!(config["nthread"], "8");
        assert_eq!(config["validate_parameters"], "0");
    }
}
