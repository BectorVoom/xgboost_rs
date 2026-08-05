//! The `Learner`: objective → booster → metrics for one training run.
//!
//! It owns the two pieces of state a fit carries between rounds: the intercept
//! (`base_score`, estimated once from the labels) and the margin cache for the
//! training matrix, which every round updates in place instead of re-predicting.

use crate::context::Context;
use crate::data::DMatrix;
use crate::gbm::{Booster, GBTree};
use crate::metric::Metric;
use crate::objective::{GradientPair, Objective};
use crate::predictor::TreeRange;
use crate::tree::param::TrainParam;
use crate::{Error, Result};

/// Orchestrates boosting rounds over a training matrix.
pub struct Learner {
    ctx: Context,
    obj: Box<dyn Objective>,
    /// Every configured metric, evaluated in order. May be empty when the
    /// objective's default metric is disabled and none was requested.
    metrics: Vec<Box<dyn Metric>>,
    booster: Booster,
    /// The intercept in *prediction* space, one value per output group.
    base_score: Vec<f32>,
    /// The same intercept in margin space; what predictions actually start
    /// from.
    base_margin: Vec<f32>,
    /// User-supplied or loaded `base_score`, which suppresses intercept
    /// estimation. A single value is broadcast over the output groups; a whole
    /// vector — which is what a loaded multi-output model carries — is used as
    /// it is.
    base_score_override: Option<Vec<f32>>,
    /// Whether the intercept may be estimated from the labels at all.
    boost_from_average: bool,
    configured: bool,
    /// Margin cache for the training matrix, updated in place each round.
    train_cache: Vec<f32>,
}

impl Learner {
    pub fn new(
        ctx: Context,
        obj: Box<dyn Objective>,
        metrics: Vec<Box<dyn Metric>>,
        num_feature: usize,
        param: TrainParam,
        base_score: Option<f32>,
    ) -> Self {
        Self {
            ctx,
            obj,
            metrics,
            booster: Booster::Tree(GBTree::new(num_feature, param)),
            base_score: vec![0.5],
            base_margin: vec![0.5],
            base_score_override: base_score.map(|b| vec![b]),
            boost_from_average: true,
            configured: false,
            train_cache: Vec::new(),
        }
    }

    /// Build a learner around an arbitrary booster, which is how `gblinear`
    /// fits are constructed.
    pub fn with_booster(
        ctx: Context,
        obj: Box<dyn Objective>,
        metrics: Vec<Box<dyn Metric>>,
        booster: Booster,
        base_score: Option<f32>,
    ) -> Self {
        Self {
            ctx,
            obj,
            metrics,
            booster,
            base_score: vec![0.5],
            base_margin: vec![0.5],
            base_score_override: base_score.map(|b| vec![b]),
            boost_from_average: true,
            configured: false,
            train_cache: Vec::new(),
        }
    }

    /// Turn the booster into `dart`: dropout before each round's gradients.
    ///
    /// Only a tree booster has dropout; calling this on `gblinear` is a
    /// programming error the API layer prevents.
    pub fn set_dart(&mut self, dart: crate::gbm::DartConfig) {
        match &mut self.booster {
            Booster::Tree(tree) => tree.set_dart(dart),
            Booster::Linear(_) => debug_assert!(false, "gblinear has no dropout"),
        }
    }

    /// Whether the intercept is estimated from the labels (`true`, the
    /// default) or left at XGBoost's `0.5`.
    pub fn set_boost_from_average(&mut self, boost_from_average: bool) {
        self.boost_from_average = boost_from_average;
    }

    /// The intercept in prediction space. Multi-output fits have one per
    /// output; the first is reported for the scalar accessor.
    pub fn base_score(&self) -> f32 {
        self.base_score[0]
    }

    /// The whole intercept vector.
    pub fn base_scores(&self) -> &[f32] {
        &self.base_score
    }

    /// The intercept in margin space, which is what predictions start from.
    pub fn base_margin(&self) -> &[f32] {
        &self.base_margin
    }

    /// The booster this learner trains.
    pub fn booster(&self) -> &Booster {
        &self.booster
    }

    /// The tree ensemble, for the callers that only exist for tree models.
    pub fn gbm(&self) -> Option<&GBTree> {
        self.booster.tree()
    }

    pub fn objective(&self) -> &dyn Objective {
        self.obj.as_ref()
    }

    /// The configured `seed`, for the paths that need a reproducible draw
    /// without advancing the session engine.
    pub fn seed(&self) -> i64 {
        self.ctx.seed
    }

    /// The configured metrics, in evaluation order.
    pub fn metrics(&self) -> &[Box<dyn Metric>] {
        &self.metrics
    }

    pub fn boosted_rounds(&self) -> usize {
        self.booster.boosted_rounds()
    }

    /// Outputs per row this fit produces.
    pub fn num_output_group(&self) -> usize {
        self.booster.num_output_group()
    }

    /// Estimate the intercept and seed the margin cache. Idempotent.
    fn configure(&mut self, dtrain: &DMatrix) -> Result<()> {
        if self.configured {
            return Ok(());
        }
        let info = dtrain.info();
        if info.labels.len() != dtrain.num_row() * info.n_targets() {
            return Err(Error::DataShape {
                expected: dtrain.num_row() * info.n_targets(),
                got: info.labels.len(),
            });
        }
        self.obj.validate_data(info)?;

        let n_groups = self.obj.num_output_group(info).max(1);
        self.booster.set_num_output_group(n_groups);

        self.base_score = match &self.base_score_override {
            Some(b) if b.len() == n_groups => b.clone(),
            Some(b) => vec![b[0]; n_groups],
            None if self.boost_from_average => {
                let mut estimated = self.obj.init_estimation(info);
                let last = *estimated.last().unwrap_or(&0.5);
                estimated.resize(n_groups, last);
                estimated
            }
            // `boost_from_average=false` keeps XGBoost's untrained default.
            None => vec![0.5; n_groups],
        };
        self.base_margin = self.base_score.clone();
        self.obj.prob_to_margin(&mut self.base_margin)?;

        // `approx` needs to know whether re-sketching per round could ever
        // change the cuts, which only the objective can answer.
        self.booster.set_constant_hessian(self.obj.has_constant_hessian());
        // Configure first: `process_type=update` empties the model here, and
        // the starting margin has to reflect what the booster holds *after*
        // that, not before.
        self.booster.configure(dtrain)?;

        // Whatever the booster still holds is part of the starting margin. For
        // a fresh fit that is nothing and this is exactly `init_margin`; for a
        // continued fit it is the inherited ensemble's prediction.
        self.train_cache = self.booster.predict_margin(dtrain, &self.base_margin, self.all_trees());
        self.configured = true;
        Ok(())
    }

    /// Run one boosting round on `dtrain`.
    pub fn update_one_iter(&mut self, iter: i32, dtrain: &DMatrix) -> Result<()> {
        self.configure(dtrain)?;
        // `seed_per_iteration` reseeds before the round's sampling draws.
        let rounds = self.boosted_rounds();
        self.ctx.seed_for_iteration(rounds);
        // DART drops trees *before* the gradients are taken, so the round fits
        // the residuals of a thinned ensemble.
        self.booster.pre_boost(&mut self.ctx, dtrain, &mut self.train_cache);
        let mut gpair: Vec<GradientPair> = Vec::new();
        self.obj.get_gradient(&self.train_cache, dtrain.info(), iter, &mut gpair);
        self.booster.do_boost(&mut self.ctx, dtrain, &gpair, &mut self.train_cache)
    }

    /// Every metric evaluated on `dmat`, as `(metric name, value)`.
    pub fn eval(&self, dmat: &DMatrix) -> Vec<(String, f64)> {
        let mut preds = self.predict_margin(dmat, self.all_trees());
        self.obj.eval_transform(&mut preds);
        self.eval_preds(&preds, dmat)
    }

    /// Every metric on the training matrix, reusing the margin cache.
    pub fn eval_train(&self, dtrain: &DMatrix) -> Vec<(String, f64)> {
        let mut preds = self.train_cache.clone();
        self.obj.eval_transform(&mut preds);
        self.eval_preds(&preds, dtrain)
    }

    fn eval_preds(&self, preds: &[f32], dmat: &DMatrix) -> Vec<(String, f64)> {
        self.metrics.iter().map(|m| (m.name().to_owned(), m.eval(preds, dmat.info()))).collect()
    }

    /// Every tree in the model, or an empty range for a linear booster.
    pub fn all_trees(&self) -> TreeRange {
        match self.booster.tree() {
            Some(tree) => TreeRange::all(&tree.model),
            None => TreeRange { begin: 0, end: 0 },
        }
    }

    pub fn predict_margin(&self, dmat: &DMatrix, trees: TreeRange) -> Vec<f32> {
        self.booster.predict_margin(dmat, &self.base_margin, trees)
    }

    pub fn predict(&self, dmat: &DMatrix, trees: TreeRange) -> Vec<f32> {
        let mut preds = self.predict_margin(dmat, trees);
        self.obj.pred_transform(&mut preds);
        preds
    }

    /// Continue from an already-trained model, as `xgboost.train`'s
    /// `xgb_model` argument does.
    ///
    /// The ensemble and the intercept are inherited; the parameters are not,
    /// so a continued fit may use different ones — which is the point of
    /// `process_type=update`, where the new parameters describe how to rewrite
    /// the old trees rather than how to grow new ones.
    pub fn continue_from(&mut self, base: &Self) -> Result<()> {
        match (&mut self.booster, base.booster()) {
            (Booster::Tree(dst), Booster::Tree(src)) => {
                dst.model = src.model.clone();
                if src.is_dart() {
                    return Err(Error::invalid(
                        "booster",
                        "continuing from a `dart` model is not supported: dropout rescales \
                         every existing tree, so the weights a continued fit would inherit \
                         are not the ones it would have produced",
                    ));
                }
            }
            (Booster::Linear(dst), Booster::Linear(src)) => {
                dst.model = src.model.clone();
            }
            _ => {
                return Err(Error::invalid(
                    "booster",
                    format!(
                        "cannot continue a `{}` fit from a `{}` model",
                        self.booster.name(),
                        base.booster.name()
                    ),
                ));
            }
        }
        // The intercept was estimated once, by the fit that produced the base
        // model; re-estimating it here would shift every existing tree.
        self.base_score_override = Some(base.base_score.clone());
        Ok(())
    }

    /// Trees the booster already holds, for callers deciding whether a
    /// continued fit has anything to work with.
    pub fn has_trees(&self) -> bool {
        self.booster.tree().is_some_and(|t| t.model.num_trees() > 0)
    }

    /// Boosting rounds the base model of a continued fit can supply, which
    /// bounds how many rounds `process_type=update` has work for.
    pub fn base_rounds(&self) -> usize {
        self.booster.tree().map_or(0, |t| t.model.num_rounds())
    }

    /// Rebuild a learner from a deserialised model (no training state).
    pub(crate) fn from_model(
        obj: Box<dyn Objective>,
        metrics: Vec<Box<dyn Metric>>,
        booster: Booster,
        base_score: Vec<f32>,
    ) -> Result<Self> {
        let mut base_margin = base_score.clone();
        obj.prob_to_margin(&mut base_margin)?;
        Ok(Self {
            ctx: Context::default(),
            obj,
            metrics,
            booster,
            base_score_override: Some(base_score.clone()),
            base_score,
            base_margin,
            boost_from_average: true,
            configured: false,
            train_cache: Vec::new(),
        })
    }
}
