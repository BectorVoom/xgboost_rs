//! The high-level API: [`train`] and [`Booster`].
//!
//! This is the Rust-native counterpart of XGBoost's Python layer. Parameters
//! come from the typed builders in [`crate::parameters`], so a configuration
//! that trains here can also be emitted verbatim for a real XGBoost build.

use crate::context::Context;
use crate::data::DMatrix;
use crate::learner::Learner;
use crate::parameters::{
    BoosterType, Device, EvalMetric, TrainingParameters, TreeBoosterParameters, TreeUpdaterName,
    VerboseEval, Verbosity,
};
use crate::tree::param::{GrowPolicy, TrainParam};
use crate::{Error, Result};
use std::collections::BTreeMap;

/// A trained model.
pub struct Booster {
    learner: Learner,
    /// Round with the best watchlist score, when early stopping was enabled.
    best_iteration: Option<usize>,
    /// The score at [`Booster::best_iteration`].
    best_score: Option<f64>,
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

    /// Every configured metric evaluated on `dmat`, in configuration order.
    pub fn eval(&self, dmat: &DMatrix) -> Vec<(&'static str, f64)> {
        self.learner.eval(dmat)
    }

    /// Boosting rounds run. With `num_parallel_tree > 1` a round grows several
    /// trees but still counts once, as it does upstream.
    pub fn boosted_rounds(&self) -> usize {
        self.learner.boosted_rounds()
    }

    /// Trees in the ensemble: `boosted_rounds() * num_parallel_tree`.
    pub fn num_trees(&self) -> usize {
        self.learner.gbm().model.num_trees()
    }

    pub fn num_features(&self) -> usize {
        self.learner.gbm().model.num_feature
    }

    pub fn base_score(&self) -> f32 {
        self.learner.base_score()
    }

    /// The round with the best watchlist score, if early stopping was enabled.
    ///
    /// Rounds after it are still part of the model: like `xgboost.train` with
    /// its default `save_best=False`, early stopping records the best round
    /// rather than truncating the ensemble.
    pub fn best_iteration(&self) -> Option<usize> {
        self.best_iteration
    }

    /// The watchlist score at [`Booster::best_iteration`].
    pub fn best_score(&self) -> Option<f64> {
        self.best_score
    }

    pub(crate) fn learner(&self) -> &Learner {
        &self.learner
    }

    pub(crate) fn from_learner(learner: Learner) -> Self {
        Self { learner, best_iteration: None, best_score: None }
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

/// One `(name, metric value)` pair per evaluation matrix and metric, per round.
///
/// Names are `"{eval set}-{metric}"`, as XGBoost's `evals_result` keys them.
pub type EvalHistory = Vec<Vec<(String, f64)>>;

/// Train a model, mirroring `xgboost.train`.
///
/// `evals` are evaluated after every round; the returned history has one entry
/// per round actually run, which is fewer than `num_boost_round` when early
/// stopping fires.
pub fn train(
    params: &TrainingParameters,
    dtrain: &DMatrix,
    evals: &[(&DMatrix, &str)],
) -> Result<(Booster, EvalHistory)> {
    params.validate()?;
    let general = &params.booster.general;
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
    check_supported_updater(tree, general.device)?;
    check_feature_constraints(tree, dtrain.num_col())?;
    let param = train_param(tree)?;

    let learning = &params.booster.learning;
    let objective = learning.objective.name();
    let obj = crate::objective::create_with(objective, learning.scale_pos_weight)?;

    // An unset `eval_metric` falls back to the objective's own default, as
    // XGBoost's `Learner::Configure` does — unless that default is disabled,
    // which leaves the fit with no metric at all.
    let metric_names: Vec<String> = if learning.eval_metric.is_empty() {
        if general.disable_default_eval_metric {
            Vec::new()
        } else {
            vec![obj.default_metric().to_owned()]
        }
    } else {
        learning.eval_metric.iter().map(EvalMetric::to_string).collect()
    };
    let metrics = metric_names
        .iter()
        .map(|name| crate::metric::create(name))
        .collect::<Result<Vec<_>>>()?;

    let ctx = Context::new(general, learning);
    if ctx.logs(Verbosity::Warning) {
        for warning in params.booster.warnings() {
            eprintln!("[xgboost_rs] WARNING: {warning}");
        }
    }

    let mut learner = Learner::new(
        ctx,
        obj,
        metrics,
        dtrain.num_col(),
        param,
        learning.base_score,
    );

    let mut stopper = match params.early_stopping_rounds {
        Some(rounds) => Some(EarlyStopping::new(rounds, params.maximize, evals, &metric_names)?),
        None => None,
    };

    let mut history: EvalHistory = Vec::new();
    for iter in 0..params.num_boost_round {
        learner.update_one_iter(iter as i32, dtrain)?;

        let mut round = Vec::new();
        for (dmat, name) in evals {
            let scores = if std::ptr::eq(*dmat, dtrain) {
                learner.eval_train(dtrain)
            } else {
                learner.eval(dmat)
            };
            for (metric, value) in scores {
                round.push((format!("{name}-{metric}"), value));
            }
        }
        report(params.verbose_eval, iter, params.num_boost_round, &round);
        history.push(round);

        if let Some(stopper) = stopper.as_mut()
            && stopper.update(iter as usize, history.last().expect("just pushed"))
        {
            break;
        }
    }

    let (best_iteration, best_score) = match &stopper {
        Some(s) => (Some(s.best_iteration), Some(s.best_score)),
        None => (None, None),
    };
    Ok((Booster { learner, best_iteration, best_score }, history))
}

/// Print one round's evaluation results, in `xgboost.train`'s format.
///
/// Output goes to stderr so a caller can keep using stdout for data.
fn report(verbose: VerboseEval, iter: u32, total: u32, round: &[(String, f64)]) {
    let period = match verbose {
        VerboseEval::Silent => return,
        VerboseEval::Every(n) => n,
    };
    // The first and last rounds always print, as XGBoost's monitor does.
    let show = iter.is_multiple_of(period) || iter + 1 == total;
    if !show || round.is_empty() {
        return;
    }
    let mut line = format!("[{iter}]");
    for (name, value) in round {
        line.push_str(&format!("\t{name}:{value:.5}"));
    }
    eprintln!("{line}");
}

/// Tracks the watchlist score and decides when to stop.
///
/// Mirrors `xgboost.callback.EarlyStopping`: the metric watched is the **last**
/// metric of the **last** evaluation set, and a round counts as an improvement
/// only if it beats the best score seen so far.
struct EarlyStopping {
    rounds: u32,
    maximize: bool,
    /// Index into a round's results of the metric being watched.
    watched: usize,
    best_score: f64,
    best_iteration: usize,
    since_improvement: u32,
    started: bool,
}

impl EarlyStopping {
    fn new(
        rounds: u32,
        maximize: Option<bool>,
        evals: &[(&DMatrix, &str)],
        metrics: &[String],
    ) -> Result<Self> {
        if evals.is_empty() || metrics.is_empty() {
            return Err(Error::invalid(
                "early_stopping_rounds",
                "early stopping needs at least one evaluation set and one metric",
            ));
        }
        // Results are laid out set-major, so the last set's last metric is the
        // final entry of a round.
        let watched = evals.len() * metrics.len() - 1;
        let metric = metrics.last().expect("checked above");
        Ok(Self {
            rounds,
            maximize: maximize.unwrap_or_else(|| metric_is_maximised(metric)),
            watched,
            best_score: 0.0,
            best_iteration: 0,
            since_improvement: 0,
            started: false,
        })
    }

    /// Record a round; returns `true` when training should stop.
    fn update(&mut self, iter: usize, round: &[(String, f64)]) -> bool {
        let score = round[self.watched].1;
        let improved = if !self.started {
            self.started = true;
            true
        } else if self.maximize {
            score > self.best_score
        } else {
            score < self.best_score
        };

        if improved {
            self.best_score = score;
            self.best_iteration = iter;
            self.since_improvement = 0;
        } else {
            self.since_improvement += 1;
        }
        self.since_improvement >= self.rounds
    }
}

/// Whether a metric is better when larger, following the list
/// `xgboost.callback.EarlyStopping` uses when `maximize` is unset.
fn metric_is_maximised(metric: &str) -> bool {
    const MAXIMISED: &[&str] = &["auc", "aucpr", "pre", "map", "ndcg"];
    if metric == "mape" {
        return false;
    }
    MAXIMISED.iter().any(|m| metric.starts_with(m))
}

/// Check the per-feature constraints against the matrix they will be applied
/// to. A constraint naming a column that does not exist would otherwise
/// silently constrain nothing.
fn check_feature_constraints(tree: &TreeBoosterParameters, num_col: usize) -> Result<()> {
    if tree.monotone_constraints.len() > num_col {
        return Err(Error::invalid(
            "monotone_constraints",
            format!(
                "{} constraints given for a matrix with {num_col} features",
                tree.monotone_constraints.len()
            ),
        ));
    }
    if let Some(groups) = &tree.interaction_constraints {
        for group in groups {
            for &feature in group {
                if feature as usize >= num_col {
                    return Err(Error::invalid(
                        "interaction_constraints",
                        format!(
                            "feature index {feature} is out of range for a matrix with \
                             {num_col} features"
                        ),
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Reject configurations whose updater pipeline this crate does not implement,
/// rather than silently training something else.
fn check_supported_updater(tree: &TreeBoosterParameters, device: Device) -> Result<()> {
    if !device.is_cpu() {
        return Err(Error::invalid(
            "device",
            format!("training runs on the CPU; got device `{device}`"),
        ));
    }
    let updaters = tree.resolved_updaters(device)?;
    if updaters != [TreeUpdaterName::GrowQuantileHistMaker] {
        let names: Vec<&str> = updaters.iter().map(|u| u.as_str()).collect();
        return Err(Error::invalid(
            "tree_method",
            format!(
                "only the `hist` updater `grow_quantile_histmaker` is implemented; \
                 this configuration resolves to `{}`",
                names.join(",")
            ),
        ));
    }
    Ok(())
}

/// Translate the public tree parameters into the internal training parameters,
/// rejecting anything the CPU `hist` path does not implement.
fn train_param(tree: &TreeBoosterParameters) -> Result<TrainParam> {
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
        subsample: tree.subsample,
        sampling_method: tree.sampling_method,
        colsample_bytree: tree.colsample_bytree,
        colsample_bylevel: tree.colsample_bylevel,
        colsample_bynode: tree.colsample_bynode,
        num_parallel_tree: tree.num_parallel_tree,
        monotone_constraints: tree.monotone_constraints.clone(),
        interaction_constraints: tree.interaction_constraints.clone(),
    })
}
