//! The high-level API: [`train`] and [`Booster`].
//!
//! This is the Rust-native counterpart of XGBoost's Python layer. Parameters
//! come from the typed builders in [`crate::parameters`], so a configuration
//! that trains here can also be emitted verbatim for a real XGBoost build.

use crate::context::Context;
use crate::gbm::DartConfig;
use crate::data::DMatrix;
use crate::learner::Learner;
use crate::parameters::{
    BoosterType, Device, EvalMetric, IterationRange, MultiStrategy, PredictParameters,
    PredictionType, TrainingParameters, TreeBoosterParameters, TreeUpdaterName, VerboseEval,
    Verbosity,
};
use crate::predictor::TreeRange;
use crate::tree::param::{GrowPolicy, TrainParam};
use crate::{Error, Result};
use std::collections::BTreeMap;

/// A prediction plus the shape it should be read with.
///
/// The shape is what `strict_shape` controls: with it set, every prediction
/// kind reports the full rank XGBoost documents (`(n_rows, n_groups)` for
/// values, `(n_rows, n_groups, n_features + 1)` for contributions), so a caller
/// does not have to know whether the model happened to be multi-class.
#[derive(Clone, Debug, PartialEq)]
pub struct Prediction {
    /// The values, row-major in `shape` order.
    pub values: Vec<f32>,
    /// Dimensions of `values`.
    pub shape: Vec<usize>,
}

impl Prediction {
    /// Values per row, i.e. the product of every dimension after the first.
    pub fn row_stride(&self) -> usize {
        self.shape.iter().skip(1).product::<usize>().max(1)
    }
}

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
        self.learner.predict(dmat, self.all_trees())
    }

    /// Raw margins, before any objective transform.
    pub fn predict_margin(&self, dmat: &DMatrix) -> Vec<f32> {
        self.learner.predict_margin(dmat, self.all_trees())
    }

    /// Leaf index reached in every tree, one row per input row.
    ///
    /// An error for `gblinear`, which has no leaves — the same refusal
    /// upstream's `GBLinear::PredictLeaf` makes.
    pub fn predict_leaf(&self, dmat: &DMatrix) -> Result<Vec<Vec<u32>>> {
        let model = &self.tree_model()?.model;
        Ok(crate::predictor::predict_leaf(model, dmat, self.all_trees()))
    }

    /// Apply DART's dropout to a margin when the caller asked for a
    /// training-time prediction.
    ///
    /// This is the `training` flag: a DART model predicts from a *thinned*
    /// ensemble while it is being fitted, and a caller computing its own
    /// gradients needs the same thinned prediction the training loop uses. For
    /// `gbtree` there is nothing to drop and the flag changes nothing, which is
    /// also what upstream does.
    ///
    /// The draw is seeded from the configured `seed` and the ensemble size, not
    /// from the session engine: predicting must be repeatable and must not
    /// consume randomness a later round depends on. That makes the dropped set
    /// a deterministic function of the model rather than the same sequence
    /// upstream's engine would produce.
    fn apply_training_dropout(
        &self,
        params: &PredictParameters,
        dmat: &DMatrix,
        values: &mut [f32],
    ) {
        if !params.training {
            return;
        }
        let Some(tree) = self.learner.booster().tree() else { return };
        let seed = self
            .learner
            .seed()
            .wrapping_mul(crate::context::RAND_SEED_MAGIC)
            .wrapping_add(self.boosted_rounds() as i64) as u32;
        tree.apply_training_dropout(seed, dmat, values);
    }

    /// The tree ensemble, or an error naming the booster that has none.
    fn tree_model(&self) -> Result<&crate::gbm::GBTree> {
        self.learner.gbm().ok_or_else(|| {
            Error::invalid(
                "booster",
                "this operation needs a tree ensemble; the model is `gblinear`",
            )
        })
    }

    /// Predict with the full [`PredictParameters`] surface.
    ///
    /// This is `Booster.predict`: the prediction *kind*, the round range, the
    /// `training` flag and `strict_shape` all take effect here rather than
    /// being accepted and ignored.
    pub fn predict_with(&self, params: &PredictParameters, dmat: &DMatrix) -> Result<Prediction> {
        params.validate()?;
        if params.validate_features && dmat.num_col() != self.num_features() {
            return Err(Error::invalid(
                "validate_features",
                format!(
                    "the model has {} features, the matrix has {}",
                    self.num_features(),
                    dmat.num_col()
                ),
            ));
        }
        let n_rows = dmat.num_row();
        let n_groups = self.learner.num_output_group();
        let n_features = self.num_features();
        let base = self.learner.base_margin();

        if let Some(linear) = self.learner.booster().linear() {
            return self.predict_linear(linear, params, dmat, n_rows, n_groups, n_features);
        }

        let model = &self.tree_model()?.model;
        let (begin, end) = params.iteration_range.resolve(self.boosted_rounds() as u32)?;
        let trees = TreeRange::from_rounds(model, begin, end);

        Ok(match params.predict_type {
            PredictionType::Value => {
                let mut values = self.learner.predict_margin(dmat, trees);
                self.apply_training_dropout(params, dmat, &mut values);
                self.learner.objective().pred_transform(&mut values);
                // `multi:softmax` collapses its groups into one class index.
                let stride = values.len() / n_rows.max(1);
                Prediction { values, shape: shape_for_value(n_rows, stride, params.strict_shape) }
            }
            PredictionType::Margin => {
                let mut values = self.learner.predict_margin(dmat, trees);
                self.apply_training_dropout(params, dmat, &mut values);
                Prediction {
                    values,
                    shape: shape_for_value(n_rows, n_groups, params.strict_shape),
                }
            }
            PredictionType::Leaf => {
                let leaves = crate::predictor::predict_leaf(model, dmat, trees);
                let per_row = leaves.first().map_or(0, Vec::len);
                let values: Vec<f32> =
                    leaves.into_iter().flatten().map(|v| v as f32).collect();
                let shape = if params.strict_shape {
                    // (rows, rounds, groups, parallel trees)
                    vec![
                        n_rows,
                        (end - begin) as usize,
                        n_groups,
                        model.num_parallel_tree.max(1) as usize,
                    ]
                } else {
                    vec![n_rows, per_row]
                };
                Prediction { values, shape }
            }
            PredictionType::Contribution | PredictionType::ApproxContribution => {
                let approximate = params.predict_type.approx_contribs();
                let values =
                    crate::predictor::predict_contribution(model, dmat, base, trees, approximate);
                let shape = if params.strict_shape || n_groups > 1 {
                    vec![n_rows, n_groups, n_features + 1]
                } else {
                    vec![n_rows, n_features + 1]
                };
                Prediction { values, shape }
            }
            PredictionType::Interaction | PredictionType::ApproxInteraction => {
                let approximate = params.predict_type.approx_contribs();
                let values =
                    crate::predictor::predict_interaction(model, dmat, base, trees, approximate);
                let shape = if params.strict_shape || n_groups > 1 {
                    vec![n_rows, n_groups, n_features + 1, n_features + 1]
                } else {
                    vec![n_rows, n_features + 1, n_features + 1]
                };
                Prediction { values, shape }
            }
        })
    }

    /// Predict with a `gblinear` model.
    ///
    /// Split out because almost every prediction kind means something
    /// different — or nothing at all — for a model with no trees.
    fn predict_linear(
        &self,
        linear: &crate::linear::GBLinear,
        params: &PredictParameters,
        dmat: &DMatrix,
        n_rows: usize,
        n_groups: usize,
        n_features: usize,
    ) -> Result<Prediction> {
        // `GBLinear::PredictBatch` refuses a layer range outright: there are no
        // rounds to slice, only one set of weights.
        if params.iteration_range != IterationRange::default() {
            return Err(Error::invalid(
                "iteration_range",
                "the `gblinear` booster has no per-round trees, so it cannot predict a \
                 round range",
            ));
        }
        let base = self.learner.base_margin();
        let width = n_features + 1;
        Ok(match params.predict_type {
            PredictionType::Value => {
                let mut values = linear.predict_margin(dmat, base);
                self.learner.objective().pred_transform(&mut values);
                let stride = values.len() / n_rows.max(1);
                Prediction { values, shape: shape_for_value(n_rows, stride, params.strict_shape) }
            }
            PredictionType::Margin => Prediction {
                values: linear.predict_margin(dmat, base),
                shape: shape_for_value(n_rows, n_groups, params.strict_shape),
            },
            PredictionType::Leaf => {
                return Err(Error::invalid(
                    "predict_type",
                    "the `gblinear` booster has no leaves to report",
                ));
            }
            PredictionType::Contribution | PredictionType::ApproxContribution => {
                let values = linear.predict_contribution(dmat, base);
                let shape = if params.strict_shape || n_groups > 1 {
                    vec![n_rows, n_groups, width]
                } else {
                    vec![n_rows, width]
                };
                Prediction { values, shape }
            }
            PredictionType::Interaction | PredictionType::ApproxInteraction => {
                // A linear model has no interaction effects, so the whole of a
                // feature's contribution is its main effect and the matrix is
                // diagonal. That keeps the invariant the tree path guarantees —
                // each row sums to the feature's total contribution — which
                // upstream's zero-filled buffer does not.
                let contribs = linear.predict_contribution(dmat, base);
                let mut values = vec![0.0f32; n_rows * n_groups * width * width];
                for slot in 0..n_rows * n_groups {
                    for i in 0..width {
                        values[slot * width * width + i * width + i] = contribs[slot * width + i];
                    }
                }
                let shape = if params.strict_shape || n_groups > 1 {
                    vec![n_rows, n_groups, width, width]
                } else {
                    vec![n_rows, width, width]
                };
                Prediction { values, shape }
            }
        })
    }

    fn all_trees(&self) -> TreeRange {
        self.learner.all_trees()
    }

    /// Every configured metric evaluated on `dmat`, in configuration order.
    pub fn eval(&self, dmat: &DMatrix) -> Vec<(String, f64)> {
        self.learner.eval(dmat)
    }

    /// Boosting rounds run. With `num_parallel_tree > 1` or several output
    /// groups a round grows many trees but still counts once, as upstream.
    pub fn boosted_rounds(&self) -> usize {
        self.learner.boosted_rounds()
    }

    /// Trees in the ensemble: `boosted_rounds() * num_parallel_tree *
    /// num_output_group`.
    pub fn num_trees(&self) -> usize {
        self.learner.gbm().map_or(0, |g| g.model.num_trees())
    }

    pub fn num_features(&self) -> usize {
        self.learner.booster().num_feature()
    }

    /// Outputs this model predicts per row.
    pub fn num_output_group(&self) -> usize {
        self.learner.num_output_group()
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
        // `GBLinear::FeatureScore` defines only `weight`, and defines it as the
        // model's own coefficients rather than a split count.
        if let Some(linear) = self.learner.booster().linear() {
            if importance_type != "weight" {
                return Err(Error::invalid(
                    "importance_type",
                    format!(
                        "`gblinear` defines only `weight` for feature importance, got \
                         `{importance_type}`"
                    ),
                ));
            }
            let n_groups = linear.model.num_output_group;
            return Ok((0..linear.model.num_feature)
                .flat_map(|f| {
                    (0..n_groups).map(move |g| {
                        let key =
                            if n_groups == 1 { format!("f{f}") } else { format!("f{f}-{g}") };
                        (key, f, g)
                    })
                })
                .map(|(key, f, g)| (key, linear.model.weight_of(f, g) as f64))
                .collect());
        }
        let mut counts: BTreeMap<u32, f64> = BTreeMap::new();
        let mut gains: BTreeMap<u32, f64> = BTreeMap::new();
        for tree in &self.tree_model()?.model.trees {
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
            "gain" => gains.iter().map(|(f, g)| (format!("f{f}"), g / counts[f])).collect(),
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

/// The shape of a value/margin prediction.
fn shape_for_value(n_rows: usize, stride: usize, strict: bool) -> Vec<usize> {
    if strict || stride > 1 { vec![n_rows, stride] } else { vec![n_rows] }
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
    train_from(params, dtrain, evals, None)
}

/// Train, optionally continuing from an already-trained model.
///
/// This is `xgboost.train`'s `xgb_model` argument. Passing a base model makes
/// the rounds *add to* that ensemble instead of starting from nothing, and it
/// is what gives `process_type=update` something to work on: under that
/// setting a round rewrites one round of the base model rather than growing a
/// new one.
///
/// The base model supplies the ensemble and the intercept; every other
/// parameter comes from `params`, so a continued fit may legitimately use a
/// different learning rate, depth or updater than the fit that produced it.
pub fn train_from(
    params: &TrainingParameters,
    dtrain: &DMatrix,
    evals: &[(&DMatrix, &str)],
    base: Option<&Booster>,
) -> Result<(Booster, EvalHistory)> {
    params.validate()?;
    let general = &params.booster.general;
    if general.device.is_sycl() {
        return Err(Error::invalid(
            "device",
            format!("there is no SYCL updater in this build; got device `{}`", general.device),
        ));
    }
    #[cfg(not(feature = "gpu"))]
    if !general.device.is_cpu() {
        return Err(Error::invalid(
            "device",
            format!(
                "this build has no GPU kernels; rebuild with the `gpu` feature to use                  device `{}`",
                general.device
            ),
        ));
    }
    let (tree, dart) = match &params.booster.booster {
        BoosterType::Gbtree(tree) => (Some(tree), None),
        BoosterType::Dart(dart) => (Some(&dart.tree), Some(DartConfig::from_parameters(dart))),
        BoosterType::Gblinear(_) => (None, None),
    };
    if let Some(tree) = tree {
        check_supported_updater(tree, general.device)?;
        check_supported_tree_options(tree, dtrain, general.device)?;
        check_feature_constraints(tree, dtrain.num_col())?;
    }

    let learning = &params.booster.learning;
    let obj = crate::objective::create(&learning.objective, learning.scale_pos_weight)?;

    // An unset `eval_metric` falls back to the objective's own default, as
    // XGBoost's `Learner::Configure` does — unless that default is disabled,
    // which leaves the fit with no metric at all.
    let metrics = if learning.eval_metric.is_empty() {
        if general.disable_default_eval_metric {
            Vec::new()
        } else {
            // The objective builds its own default so a parameterised metric
            // (`tweedie-nloglik@1.5`, `aft-nloglik`) gets the objective's
            // settings rather than the metric defaults.
            let spec: EvalMetric = obj.default_metric().parse()?;
            vec![obj.make_metric(&spec)?]
        }
    } else {
        learning
            .eval_metric
            .iter()
            .map(|m| obj.make_metric(m))
            .collect::<Result<Vec<_>>>()?
    };
    let metric_names: Vec<String> = metrics.iter().map(|m| m.name().to_owned()).collect();

    let ctx = Context::new(general, learning);
    if ctx.logs(Verbosity::Warning) {
        for warning in params.booster.warnings() {
            eprintln!("[xgboost_rs] WARNING: {warning}");
        }
        if general.validate_parameters {
            for name in unused_parameters(params, tree, dtrain.info().has_categorical()) {
                eprintln!(
                    "[xgboost_rs] WARNING: parameter `{name}` was set but nothing in this \
                     fit consumed it"
                );
            }
        }
    }

    let mut learner = match &params.booster.booster {
        BoosterType::Gblinear(linear) => Learner::with_booster(
            ctx,
            obj,
            metrics,
            crate::gbm::Booster::Linear(Box::new(crate::linear::GBLinear::new(
                dtrain.num_col(),
                linear.clone(),
            ))),
            learning.base_score,
        ),
        _ => {
            let param = train_param(tree.expect("a tree booster"), general.device)?;
            Learner::new(ctx, obj, metrics, dtrain.num_col(), param, learning.base_score)
        }
    };
    learner.set_boost_from_average(learning.boost_from_average);
    if let Some(dart) = dart {
        learner.set_dart(dart);
    }
    if let Some(base) = base {
        learner.continue_from(base.learner())?;
    }
    // `process_type=update` rewrites trees the model already holds. Without a
    // base model there are none, so every round would be a silent no-op.
    if let Some(tree) = tree
        && tree.process_type == crate::parameters::ProcessType::Update
        && !learner.has_trees()
    {
        return Err(Error::invalid(
            "process_type",
            "`update` revisits the trees an existing model holds; pass one to \
             `api::train_from`, or use `process_type=default` to grow new trees",
        ));
    }

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
    const MAXIMISED: &[&str] =
        &["auc", "aucpr", "pre", "map", "ndcg", "ams", "interval-regression-accuracy"];
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
    let updaters = tree.resolved_updaters(device)?;
    for updater in &updaters {
        let implemented = matches!(
            updater,
            TreeUpdaterName::GrowQuantileHistMaker
                | TreeUpdaterName::GrowHistMaker
                | TreeUpdaterName::GrowColMaker
                | TreeUpdaterName::Prune
                | TreeUpdaterName::Refresh
        ) || (cfg!(feature = "gpu") && *updater == TreeUpdaterName::GrowGpuHist);
        if !implemented {
            return Err(Error::invalid(
                "updater",
                format!("the `{updater}` updater is not implemented on this device"),
            ));
        }
    }
    // `ColMaker::Builder::Update` refuses an unbounded depth outright: it grows
    // level by level and has no other stopping rule.
    if updaters.contains(&TreeUpdaterName::GrowColMaker) && tree.max_depth == 0 {
        return Err(Error::invalid(
            "max_depth",
            "the `exact` tree method grows level by level and cannot run with an \
             unlimited depth; set max_depth",
        ));
    }
    Ok(())
}

/// Reject the tree-booster options this build cannot act on, so none of them is
/// accepted and then silently ignored.
fn check_supported_tree_options(
    tree: &TreeBoosterParameters,
    dtrain: &DMatrix,
    device: Device,
) -> Result<()> {
    if tree.multi_strategy == MultiStrategy::MultiOutputTree {
        // A vector leaf is grown from one histogram per target, which only the
        // `hist` updaters build here — on either device. The others would
        // silently fall back to one tree per target, which is the opposite of
        // what was asked for.
        let updaters = tree.resolved_updaters(device)?;
        let hist = updaters == [TreeUpdaterName::GrowQuantileHistMaker]
            || (cfg!(feature = "gpu") && updaters == [TreeUpdaterName::GrowGpuHist]);
        if !hist {
            return Err(Error::invalid(
                "multi_strategy",
                "`multi_output_tree` is implemented for the `hist` tree method only; \
                 use `tree_method=hist` with the default updater pipeline",
            ));
        }
        if dtrain.info().has_categorical() {
            return Err(Error::invalid(
                "multi_strategy",
                "`multi_output_tree` has no categorical split; one-hot encode the \
                 categories, or use `one_output_per_tree`",
            ));
        }
    }
    // Only the histogram updaters bin by category. `exact` enumerates a
    // column's values in ascending order, which would read the category codes
    // as an ordering they do not have, so it is refused rather than quietly
    // fitting something else.
    if dtrain.info().has_categorical()
        && tree.resolved_updaters(device)?.contains(&TreeUpdaterName::GrowColMaker)
    {
        return Err(Error::invalid(
            "tree_method",
            "the `exact` tree method has no categorical split; use `hist` or `approx`, \
             or one-hot encode the categories yourself",
        ));
    }
    Ok(())
}

/// Warn about parameters this build accepts but cannot act on.
///
/// This is what `validate_parameters` asks for: XGBoost prints a warning for
/// every configuration entry nothing consumed, and so does this — otherwise a
/// caller has no way to tell a knob that did nothing from one that did.
fn unused_parameters(
    params: &TrainingParameters,
    tree: Option<&TreeBoosterParameters>,
    has_categorical: bool,
) -> Vec<String> {
    let mut unused = Vec::new();
    let default = TreeBoosterParameters::default();

    if let Some(tree) = tree {
        let updaters =
            tree.resolved_updaters(params.booster.general.device).unwrap_or_default();
        let uses = |u: TreeUpdaterName| updaters.contains(&u);

        // The two `exact`-only knobs, on a fit that runs something else.
        if !uses(TreeUpdaterName::GrowColMaker) {
            if tree.default_direction != default.default_direction {
                unused.push("default_direction".to_owned());
            }
            if tree.opt_dense_col != default.opt_dense_col {
                unused.push("opt_dense_col".to_owned());
            }
        }
        // `exact` enumerates every distinct value, so there are no bins.
        if uses(TreeUpdaterName::GrowColMaker) && tree.max_bin != default.max_bin {
            unused.push("max_bin".to_owned());
        }
        if !uses(TreeUpdaterName::Refresh) && tree.refresh_leaf != default.refresh_leaf {
            unused.push("refresh_leaf".to_owned());
        }
        // Both categorical knobs only mean something once some column is
        // marked categorical.
        if !has_categorical {
            if tree.max_cat_to_onehot != default.max_cat_to_onehot {
                unused.push("max_cat_to_onehot".to_owned());
            }
            if tree.max_cat_threshold != default.max_cat_threshold {
                unused.push("max_cat_threshold".to_owned());
            }
        }
        // A single-process fit has no workers to synchronise.
        if tree.debug_synchronize {
            unused.push("debug_synchronize".to_owned());
        }
        // `sparse_threshold` picks the column layout the row partitioner
        // reads, which only the histogram updaters use.
        if !uses(TreeUpdaterName::GrowQuantileHistMaker)
            && !uses(TreeUpdaterName::GrowHistMaker)
            && tree.sparse_threshold != default.sparse_threshold
        {
            unused.push("sparse_threshold".to_owned());
        }
    }
    // GPU-only settings, on a CPU fit.
    let general = &params.booster.general;
    if general.use_rmm {
        unused.push("use_rmm".to_owned());
    }
    if general.fail_on_invalid_gpu_id {
        unused.push("fail_on_invalid_gpu_id".to_owned());
    }
    unused
}

/// Translate the public tree parameters into the internal training parameters.
fn train_param(tree: &TreeBoosterParameters, device: Device) -> Result<TrainParam> {
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
        max_cached_hist_node: tree.max_cached_hist_nodes(device),
        multi_output_tree: tree.multi_strategy == MultiStrategy::MultiOutputTree,
        sparse_threshold: tree.sparse_threshold,
        max_cat_to_onehot: tree.max_cat_to_onehot,
        max_cat_threshold: tree.max_cat_threshold,
        default_direction: tree.default_direction,
        opt_dense_col: tree.opt_dense_col,
        updaters: tree.resolved_updaters(device)?,
        process_type: tree.process_type,
        refresh_leaf: tree.refresh_leaf,
        device,
    })
}

/// Metrics named in a fit's `eval_metric`, for callers that want the resolved
/// list without running a round.
pub fn resolved_metric_names(params: &TrainingParameters) -> Result<Vec<String>> {
    let learning = &params.booster.learning;
    if learning.eval_metric.is_empty() {
        let obj = crate::objective::create(&learning.objective, learning.scale_pos_weight)?;
        if params.booster.general.disable_default_eval_metric {
            return Ok(Vec::new());
        }
        return Ok(vec![obj.default_metric()]);
    }
    Ok(learning.eval_metric.iter().map(EvalMetric::to_string).collect())
}
