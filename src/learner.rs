//! The `Learner`: objective → booster → metrics for one training run.
//!
//! It owns the two pieces of state a fit carries between rounds: the intercept
//! (`base_score`, estimated once from the labels) and the margin cache for the
//! training matrix, which every round updates in place instead of re-predicting.

use crate::context::Context;
use crate::data::DMatrix;
use crate::gbm::GBTree;
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
    gbm: GBTree,
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
            gbm: GBTree::new(num_feature, param),
            base_score: vec![0.5],
            base_margin: vec![0.5],
            base_score_override: base_score.map(|b| vec![b]),
            boost_from_average: true,
            configured: false,
            train_cache: Vec::new(),
        }
    }

    /// Turn the booster into `dart`: dropout before each round's gradients.
    pub fn set_dart(&mut self, dart: crate::gbm::DartConfig) {
        self.gbm.set_dart(dart);
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

    pub fn gbm(&self) -> &GBTree {
        &self.gbm
    }

    pub fn objective(&self) -> &dyn Objective {
        self.obj.as_ref()
    }

    /// The configured metrics, in evaluation order.
    pub fn metrics(&self) -> &[Box<dyn Metric>] {
        &self.metrics
    }

    pub fn boosted_rounds(&self) -> usize {
        self.gbm.model.num_rounds()
    }

    /// Outputs per row this fit produces.
    pub fn num_output_group(&self) -> usize {
        self.gbm.model.num_output_group.max(1)
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
        self.gbm.set_num_output_group(n_groups);

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

        self.train_cache =
            crate::predictor::init_margin(dtrain, &self.base_margin, dtrain.num_row(), n_groups);
        self.gbm.configure(dtrain)?;
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
        self.gbm.pre_boost(&mut self.ctx, dtrain, &mut self.train_cache);
        let mut gpair: Vec<GradientPair> = Vec::new();
        self.obj.get_gradient(&self.train_cache, dtrain.info(), iter, &mut gpair);
        self.gbm.do_boost(&mut self.ctx, dtrain, &gpair, &mut self.train_cache)
    }

    /// Every metric evaluated on `dmat`, as `(metric name, value)`.
    pub fn eval(&self, dmat: &DMatrix) -> Vec<(String, f64)> {
        let mut preds = self.predict_margin(dmat, TreeRange::all(&self.gbm.model));
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

    pub fn predict_margin(&self, dmat: &DMatrix, trees: TreeRange) -> Vec<f32> {
        crate::predictor::predict_margin(&self.gbm.model, dmat, &self.base_margin, trees)
    }

    pub fn predict(&self, dmat: &DMatrix, trees: TreeRange) -> Vec<f32> {
        let mut preds = self.predict_margin(dmat, trees);
        self.obj.pred_transform(&mut preds);
        preds
    }

    /// Rebuild a learner from a deserialised model (no training state).
    pub(crate) fn from_model(
        obj: Box<dyn Objective>,
        metrics: Vec<Box<dyn Metric>>,
        gbm: GBTree,
        base_score: Vec<f32>,
    ) -> Result<Self> {
        let mut base_margin = base_score.clone();
        obj.prob_to_margin(&mut base_margin)?;
        Ok(Self {
            ctx: Context::default(),
            obj,
            metrics,
            gbm,
            base_score_override: Some(base_score.clone()),
            base_score,
            base_margin,
            boost_from_average: true,
            configured: false,
            train_cache: Vec::new(),
        })
    }
}
