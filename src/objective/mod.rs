//! Objectives: the loss being boosted.
//!
//! Phase 1 covers `reg:squarederror`. An objective supplies first and second
//! derivatives per row, the transform applied to raw margins at prediction
//! time, and the intercept (`base_score`) that boosting starts from.

pub mod squared_error;

use crate::data::MetaInfo;

pub use squared_error::SquaredError;

/// First and second derivative of the loss at one row.
///
/// Mirrors `xgboost::GradientPair`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GradientPair {
    pub grad: f32,
    pub hess: f32,
}

/// A boosting objective.
pub trait Objective {
    /// The XGBoost parameter value that selects this objective.
    fn name(&self) -> &'static str;

    /// Fill `out` with per-row gradients for the current margins.
    fn get_gradient(&self, preds: &[f32], info: &MetaInfo, iter: i32, out: &mut Vec<GradientPair>);

    /// Map raw margins to the reported prediction scale.
    fn pred_transform(&self, preds: &mut [f32]);

    /// Estimate the intercept boosting starts from, in margin space.
    fn init_estimation(&self, info: &MetaInfo) -> f32;

    /// Metric reported when the user did not choose one.
    fn default_metric(&self) -> &'static str;
}

/// Construct an objective by its XGBoost name, at its default settings.
pub fn create(name: &str) -> crate::Result<Box<dyn Objective>> {
    create_with(name, 1.0)
}

/// Construct an objective by its XGBoost name, with the learning-task
/// parameters that objectives read.
pub fn create_with(name: &str, scale_pos_weight: f32) -> crate::Result<Box<dyn Objective>> {
    match name {
        "reg:squarederror" | "reg:linear" => Ok(Box::new(SquaredError::new(scale_pos_weight))),
        other => Err(crate::Error::invalid(
            "objective",
            format!("`{other}` is not implemented; Phase 1 supports `reg:squarederror`"),
        )),
    }
}
