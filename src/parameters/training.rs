//! Parameters that belong to the training loop rather than to the model:
//! how many rounds to boost, when to stop early, how loudly to report.
//!
//! These are `xgboost.train`'s own arguments
//! (`python-package/xgboost/training.py`), not DMLC parameters, so they are
//! *not* part of [`ToConfig`] output — XGBoost's `Learner` never sees them.

use serde::{Deserialize, Serialize};

use super::BoosterParameters;
use crate::error::{Error, Result};

/// How often the training loop reports evaluation results.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerboseEval {
    /// Report nothing.
    Silent,
    /// Report every `n`th round (and always the first and last).
    Every(u32),
}

impl Default for VerboseEval {
    fn default() -> Self {
        Self::Every(1)
    }
}

/// A complete fit specification: the model parameters plus the loop controls.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrainingParameters {
    /// Everything the `Learner` is configured with.
    pub booster: BoosterParameters,
    /// Number of boosting rounds. With `num_parallel_tree > 1` each round
    /// still counts once.
    pub num_boost_round: u32,
    /// Stop when the last watchlist metric has not improved for this many
    /// rounds. Requires at least one evaluation set at fit time.
    pub early_stopping_rounds: Option<u32>,
    /// How often to report evaluation results.
    pub verbose_eval: VerboseEval,
    /// Whether early stopping should maximise the metric. `None` lets the
    /// metric decide, which is what you want unless you supply a custom one.
    pub maximize: Option<bool>,
}

impl Default for TrainingParameters {
    fn default() -> Self {
        Self {
            booster: BoosterParameters::default(),
            // `xgboost.train`'s own default.
            num_boost_round: 10,
            early_stopping_rounds: None,
            verbose_eval: VerboseEval::default(),
            maximize: None,
        }
    }
}

impl TrainingParameters {
    /// Start from the defaults.
    pub fn builder() -> TrainingParametersBuilder {
        TrainingParametersBuilder::default()
    }

    /// Validate the loop controls and everything they wrap.
    pub fn validate(&self) -> Result<()> {
        self.booster.validate()?;
        if self.num_boost_round == 0 {
            return Err(Error::invalid("num_boost_round", "must be >= 1"));
        }
        if let Some(rounds) = self.early_stopping_rounds
            && rounds == 0
        {
            return Err(Error::invalid("early_stopping_rounds", "must be >= 1 when set"));
        }
        if let VerboseEval::Every(0) = self.verbose_eval {
            return Err(Error::invalid(
                "verbose_eval",
                "reporting period must be >= 1; use VerboseEval::Silent to report nothing",
            ));
        }
        Ok(())
    }
}

/// Consuming builder for [`TrainingParameters`].
#[derive(Clone, Debug, Default)]
pub struct TrainingParametersBuilder {
    inner: TrainingParameters,
}

impl TrainingParametersBuilder {
    /// The model parameters.
    pub fn booster(mut self, booster: BoosterParameters) -> Self {
        self.inner.booster = booster;
        self
    }

    /// Number of boosting rounds.
    pub fn num_boost_round(mut self, num_boost_round: u32) -> Self {
        self.inner.num_boost_round = num_boost_round;
        self
    }

    /// Stop after this many rounds without improvement.
    pub fn early_stopping_rounds(mut self, rounds: u32) -> Self {
        self.inner.early_stopping_rounds = Some(rounds);
        self
    }

    /// How often to report evaluation results.
    pub fn verbose_eval(mut self, verbose_eval: VerboseEval) -> Self {
        self.inner.verbose_eval = verbose_eval;
        self
    }

    /// Force early stopping to maximise (`true`) or minimise (`false`).
    pub fn maximize(mut self, maximize: bool) -> Self {
        self.inner.maximize = Some(maximize);
        self
    }

    /// Validate and produce the parameters.
    pub fn build(self) -> Result<TrainingParameters> {
        self.inner.validate()?;
        Ok(self.inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_xgboost_train() {
        let params = TrainingParameters::default();
        assert_eq!(params.num_boost_round, 10);
        assert_eq!(params.early_stopping_rounds, None);
        assert_eq!(params.verbose_eval, VerboseEval::Every(1));
        params.validate().unwrap();
    }

    #[test]
    fn rejects_degenerate_loop_controls() {
        assert!(TrainingParameters::builder().num_boost_round(0).build().is_err());
        assert!(TrainingParameters::builder().early_stopping_rounds(0).build().is_err());
        assert!(
            TrainingParameters::builder().verbose_eval(VerboseEval::Every(0)).build().is_err()
        );
        TrainingParameters::builder().verbose_eval(VerboseEval::Silent).build().unwrap();
    }
}
