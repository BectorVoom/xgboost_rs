//! Parameters for the `gblinear` booster.
//!
//! Mirrors `LinearTrainParam` (`src/linear/param.h`), `GBLinearTrainParam`
//! (`src/gbm/gblinear.cc`) and `CoordinateParam`
//! (`src/linear/coordinate_common.h`).

use serde::{Deserialize, Serialize};

use super::config::{ConfigEntry, ToConfig, push, push_opt};
use super::str_enum::str_enum;
use super::validate;
use crate::error::{Error, Result};

str_enum! {
    /// Linear model solver.
    pub enum LinearUpdater: "updater" {
        /// Hogwild parallel coordinate descent. Non-deterministic, and limited
        /// to the `cyclic` and `shuffle` feature selectors.
        Shotgun = "shotgun",
        /// Ordinary coordinate descent; deterministic and supports every
        /// feature selector.
        CoordDescent = "coord_descent",
    }
    default = Shotgun;
}

str_enum! {
    /// Feature selection / ordering strategy for the linear solver.
    pub enum FeatureSelector: "feature_selector" {
        /// Cycle through features in order.
        Cyclic = "cyclic",
        /// Reshuffle the cycle each round.
        Shuffle = "shuffle",
        /// Pick features uniformly at random. `coord_descent` only.
        Random = "random",
        /// Pick the feature with the largest gradient magnitude each step.
        /// `coord_descent` only; expensive.
        Greedy = "greedy",
        /// Approximate greedy using a `top_k` shortlist. `coord_descent` only.
        Thrifty = "thrifty",
    }
    default = Cyclic;
}

/// Parameters for the `gblinear` booster.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LinearBoosterParameters {
    /// Learning rate (upstream alias `learning_rate`). Note the linear booster
    /// defaults to `0.5`, not the tree booster's `0.3`.
    pub eta: f32,
    /// L2 regularisation on weights (alias `reg_lambda`).
    pub lambda: f32,
    /// L1 regularisation on weights (alias `reg_alpha`).
    pub alpha: f32,
    /// Linear model solver.
    pub updater: LinearUpdater,
    /// Feature selection / ordering strategy.
    pub feature_selector: FeatureSelector,
    /// Shortlist size for the `greedy` and `thrifty` selectors. `0` uses every
    /// feature.
    pub top_k: u32,
    /// Stop early once the largest weight update falls below this.
    pub tolerance: f32,
    /// Rows per batch. `None` means unbounded, XGBoost's default.
    pub max_row_perbatch: Option<u64>,
}

impl Default for LinearBoosterParameters {
    fn default() -> Self {
        Self {
            eta: 0.5,
            lambda: 0.0,
            alpha: 0.0,
            updater: LinearUpdater::Shotgun,
            feature_selector: FeatureSelector::Cyclic,
            top_k: 0,
            tolerance: 0.0,
            max_row_perbatch: None,
        }
    }
}

impl LinearBoosterParameters {
    /// Start from XGBoost's defaults.
    pub fn builder() -> LinearBoosterParametersBuilder {
        LinearBoosterParametersBuilder::default()
    }

    /// Validate every field and the solver/selector pairing.
    pub fn validate(&self) -> Result<()> {
        validate::ge("eta", self.eta, 0.0)?;
        validate::ge("lambda", self.lambda, 0.0)?;
        validate::ge("alpha", self.alpha, 0.0)?;
        validate::ge("tolerance", self.tolerance, 0.0)?;
        if let Some(rows) = self.max_row_perbatch {
            validate::ge("max_row_perbatch", rows, 1)?;
        }

        // `ShotgunUpdater::Configure` rejects anything but cyclic/shuffle.
        if self.updater == LinearUpdater::Shotgun
            && !matches!(self.feature_selector, FeatureSelector::Cyclic | FeatureSelector::Shuffle)
        {
            return Err(Error::invalid(
                "feature_selector",
                format!(
                    "the `shotgun` updater supports only `cyclic` and `shuffle`, got `{}`; \
                     use updater `coord_descent`",
                    self.feature_selector
                ),
            ));
        }
        Ok(())
    }
}

impl ToConfig for LinearBoosterParameters {
    fn collect_config(&self, out: &mut Vec<ConfigEntry>) {
        push(out, "eta", self.eta);
        push(out, "lambda", self.lambda);
        push(out, "alpha", self.alpha);
        push(out, "updater", self.updater);
        push(out, "feature_selector", self.feature_selector);
        push(out, "top_k", self.top_k);
        push(out, "tolerance", self.tolerance);
        push_opt(out, "max_row_perbatch", self.max_row_perbatch);
    }
}

/// Consuming builder for [`LinearBoosterParameters`].
#[derive(Clone, Debug, Default)]
pub struct LinearBoosterParametersBuilder {
    inner: LinearBoosterParameters,
}

impl LinearBoosterParametersBuilder {
    /// Learning rate.
    pub fn eta(mut self, eta: f32) -> Self {
        self.inner.eta = eta;
        self
    }

    /// L2 regularisation on weights.
    pub fn lambda(mut self, lambda: f32) -> Self {
        self.inner.lambda = lambda;
        self
    }

    /// L1 regularisation on weights.
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.inner.alpha = alpha;
        self
    }

    /// Linear model solver.
    pub fn updater(mut self, updater: LinearUpdater) -> Self {
        self.inner.updater = updater;
        self
    }

    /// Feature selection / ordering strategy.
    pub fn feature_selector(mut self, selector: FeatureSelector) -> Self {
        self.inner.feature_selector = selector;
        self
    }

    /// Shortlist size for `greedy` / `thrifty`.
    pub fn top_k(mut self, top_k: u32) -> Self {
        self.inner.top_k = top_k;
        self
    }

    /// Convergence tolerance on the largest weight update.
    pub fn tolerance(mut self, tolerance: f32) -> Self {
        self.inner.tolerance = tolerance;
        self
    }

    /// Rows per batch; leave unset for unbounded.
    pub fn max_row_perbatch(mut self, rows: u64) -> Self {
        self.inner.max_row_perbatch = Some(rows);
        self
    }

    /// Validate and produce the parameters.
    pub fn build(self) -> Result<LinearBoosterParameters> {
        self.inner.validate()?;
        Ok(self.inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_xgboost() {
        let params = LinearBoosterParameters::default();
        assert_eq!(params.eta, 0.5);
        assert_eq!(params.lambda, 0.0);
        assert_eq!(params.alpha, 0.0);
        assert_eq!(params.updater, LinearUpdater::Shotgun);
        assert_eq!(params.feature_selector, FeatureSelector::Cyclic);
        assert_eq!(params.top_k, 0);
        params.validate().unwrap();
    }

    #[test]
    fn rejects_selectors_shotgun_cannot_run() {
        for selector in
            [FeatureSelector::Greedy, FeatureSelector::Thrifty, FeatureSelector::Random]
        {
            let err = LinearBoosterParameters::builder()
                .feature_selector(selector)
                .build()
                .unwrap_err();
            assert!(err.to_string().contains("feature_selector"), "{selector}");

            LinearBoosterParameters::builder()
                .updater(LinearUpdater::CoordDescent)
                .feature_selector(selector)
                .build()
                .unwrap();
        }
        LinearBoosterParameters::builder().feature_selector(FeatureSelector::Shuffle).build().unwrap();
    }

    #[test]
    fn emits_upstream_parameter_names() {
        let config = LinearBoosterParameters::builder()
            .updater(LinearUpdater::CoordDescent)
            .feature_selector(FeatureSelector::Thrifty)
            .top_k(10)
            .lambda(2.0)
            .build()
            .unwrap()
            .to_config_map();
        assert_eq!(config["updater"], "coord_descent");
        assert_eq!(config["feature_selector"], "thrifty");
        assert_eq!(config["top_k"], "10");
        assert_eq!(config["lambda"], "2");
        assert!(!config.contains_key("max_row_perbatch"));
    }
}
