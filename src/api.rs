//! The high-level API: [`train`] and [`Booster`].
//!
//! This is the Rust-native counterpart of XGBoost's Python layer. Parameters
//! come from the typed builders in [`crate::parameters`], so a configuration
//! that trains here can also be emitted verbatim for a real XGBoost build.

use crate::data::DMatrix;
use crate::learner::Learner;
use crate::parameters::{BoosterType, TrainingParameters, TreeBoosterParameters};
use crate::tree::param::{GrowPolicy, TrainParam};
use crate::{Error, Result};
use std::collections::BTreeMap;

/// A trained model.
pub struct Booster {
    learner: Learner,
}

impl Booster {
    /// Run one more boosting round on `dtrain`.
    pub fn update(&mut self, iter: i32, dtrain: &DMatrix) -> Result<()> {
        self.learner.update_one_iter(iter, dtrain)
    }

    /// Predictions on the objective's output scale.
    pub fn predict(&self, dmat: &DMatrix) -> Vec<f32> {
        self.learner.predict(dmat)
    }

    /// Raw margins, before any objective transform.
    pub fn predict_margin(&self, dmat: &DMatrix) -> Vec<f32> {
        self.learner.predict_margin(dmat)
    }

    /// Leaf index reached in every tree, one row per input row.
    pub fn predict_leaf(&self, dmat: &DMatrix) -> Vec<Vec<u32>> {
        crate::predictor::predict_leaf(&self.learner.gbm().model, dmat)
    }

    /// The configured metric evaluated on `dmat`.
    pub fn eval(&self, dmat: &DMatrix) -> f64 {
        self.learner.eval(dmat)
    }

    pub fn boosted_rounds(&self) -> usize {
        self.learner.boosted_rounds()
    }

    pub fn num_features(&self) -> usize {
        self.learner.gbm().model.num_feature
    }

    pub fn base_score(&self) -> f32 {
        self.learner.base_score()
    }

    pub(crate) fn learner(&self) -> &Learner {
        &self.learner
    }

    pub(crate) fn from_learner(learner: Learner) -> Self {
        Self { learner }
    }

    /// Feature importance, keyed `"f{index}"` as XGBoost reports it.
    ///
    /// `"weight"` counts splits per feature; `"gain"` averages the loss
    /// reduction of those splits.
    pub fn get_score(&self, importance_type: &str) -> Result<BTreeMap<String, f64>> {
        let mut counts: BTreeMap<u32, f64> = BTreeMap::new();
        let mut gains: BTreeMap<u32, f64> = BTreeMap::new();
        for tree in &self.learner.gbm().model.trees {
            for (nid, node) in tree.nodes.iter().enumerate() {
                if node.is_leaf() {
                    continue;
                }
                *counts.entry(node.split_index).or_default() += 1.0;
                *gains.entry(node.split_index).or_default() += tree.stats[nid].loss_chg as f64;
            }
        }
        let out = match importance_type {
            "weight" => counts.iter().map(|(f, c)| (format!("f{f}"), *c)).collect(),
            "gain" => gains
                .iter()
                .map(|(f, g)| (format!("f{f}"), g / counts[f]))
                .collect(),
            "total_gain" => gains.iter().map(|(f, g)| (format!("f{f}"), *g)).collect(),
            other => {
                return Err(Error::invalid(
                    "importance_type",
                    format!("`{other}` is not supported; use weight, gain or total_gain"),
                ));
            }
        };
        Ok(out)
    }

    /// Serialise to XGBoost's JSON model format.
    pub fn save_model(&self) -> String {
        crate::model_io::save_model(self)
    }

    /// Load a model written by [`Booster::save_model`] or by XGBoost itself.
    pub fn load_model(json: &str) -> Result<Self> {
        crate::model_io::load_model(json)
    }
}

/// One `(name, metric value)` pair per evaluation matrix, per round.
pub type EvalHistory = Vec<Vec<(String, f64)>>;

/// Train a model, mirroring `xgboost.train`.
///
/// `evals` are evaluated after every round; the returned history has one entry
/// per round.
pub fn train(
    params: &TrainingParameters,
    dtrain: &DMatrix,
    evals: &[(&DMatrix, &str)],
) -> Result<(Booster, EvalHistory)> {
    params.validate()?;
    let tree = match &params.booster.booster {
        BoosterType::Gbtree(tree) => tree,
        other => {
            return Err(Error::invalid(
                "booster",
                format!(
                    "`{}` is not implemented; Phase 1 supports `gbtree`",
                    other.name()
                ),
            ));
        }
    };
    let param = train_param(tree)?;

    let learning = &params.booster.learning;
    let objective = learning.objective.name();
    // An unset `eval_metric` falls back to the objective's own default, as
    // XGBoost's `Learner::Configure` does.
    let eval_metric = match learning.eval_metric.first() {
        Some(m) => m.to_string(),
        None => crate::objective::create(objective)?.default_metric().to_string(),
    };

    let mut learner = Learner::new(
        objective,
        &eval_metric,
        dtrain.num_col(),
        param,
        learning.base_score,
    )?;

    let mut history: EvalHistory = Vec::new();
    for iter in 0..params.num_boost_round {
        learner.update_one_iter(iter as i32, dtrain)?;
        let mut round = Vec::new();
        for (dmat, name) in evals {
            let value = if std::ptr::eq(*dmat, dtrain) {
                learner.eval_train(dtrain)
            } else {
                learner.eval(dmat)
            };
            round.push((format!("{name}-{}", learner.metric().name()), value));
        }
        history.push(round);
    }

    Ok((Booster { learner }, history))
}

/// Translate the public tree parameters into the internal training parameters,
/// rejecting anything the Phase 1 CPU `hist` path does not implement.
fn train_param(tree: &TreeBoosterParameters) -> Result<TrainParam> {
    if tree.subsample != 1.0 {
        return Err(Error::invalid("subsample", "row sampling is not implemented"));
    }
    for (name, value) in [
        ("colsample_bytree", tree.colsample_bytree),
        ("colsample_bylevel", tree.colsample_bylevel),
        ("colsample_bynode", tree.colsample_bynode),
    ] {
        if value != 1.0 {
            return Err(Error::invalid(name, "column sampling is not implemented"));
        }
    }
    if tree.num_parallel_tree != 1 {
        return Err(Error::invalid("num_parallel_tree", "forests are not implemented"));
    }
    if !tree.monotone_constraints.is_empty() {
        return Err(Error::invalid("monotone_constraints", "constraints are not implemented"));
    }
    if tree.interaction_constraints.is_some() {
        return Err(Error::invalid("interaction_constraints", "constraints are not implemented"));
    }

    Ok(TrainParam {
        learning_rate: tree.eta,
        min_split_loss: tree.gamma,
        max_depth: tree.max_depth as i32,
        max_leaves: tree.max_leaves as i32,
        max_bin: tree.max_bin,
        grow_policy: match tree.grow_policy.as_str() {
            "lossguide" => GrowPolicy::LossGuide,
            _ => GrowPolicy::DepthWise,
        },
        min_child_weight: tree.min_child_weight,
        reg_lambda: tree.lambda,
        reg_alpha: tree.alpha,
        max_delta_step: tree.max_delta_step,
    })
}
