//! The `Learner`: objective → booster → metrics for one training run.

use crate::context::Context;
use crate::data::DMatrix;
use crate::gbm::GBTree;
use crate::metric::Metric;
use crate::objective::{GradientPair, Objective};
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
    base_score: f32,
    /// User-supplied `base_score`, which suppresses intercept estimation.
    base_score_override: Option<f32>,
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
            base_score: 0.5,
            base_score_override: base_score,
            configured: false,
            train_cache: Vec::new(),
        }
    }

    pub fn base_score(&self) -> f32 {
        self.base_score
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

    /// Estimate the intercept and seed the margin cache. Idempotent.
    fn configure(&mut self, dtrain: &DMatrix) -> Result<()> {
        if self.configured {
            return Ok(());
        }
        if dtrain.info().labels.len() != dtrain.num_row() {
            return Err(Error::DataShape {
                expected: dtrain.num_row(),
                got: dtrain.info().labels.len(),
            });
        }
        self.base_score = match self.base_score_override {
            Some(b) => b,
            None => self.obj.init_estimation(dtrain.info()),
        };
        self.train_cache = match &dtrain.info().base_margin {
            Some(m) => m.clone(),
            None => vec![self.base_score; dtrain.num_row()],
        };
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
        let mut gpair: Vec<GradientPair> = Vec::new();
        self.obj.get_gradient(&self.train_cache, dtrain.info(), iter, &mut gpair);
        self.gbm.do_boost(&mut self.ctx, dtrain, &gpair, &mut self.train_cache)
    }

    /// Every metric evaluated on `dmat`, as `(metric name, value)`.
    pub fn eval(&self, dmat: &DMatrix) -> Vec<(&'static str, f64)> {
        let mut preds = self.predict_margin(dmat);
        self.obj.pred_transform(&mut preds);
        self.eval_preds(&preds, dmat)
    }

    /// Every metric on the training matrix, reusing the margin cache.
    pub fn eval_train(&self, dtrain: &DMatrix) -> Vec<(&'static str, f64)> {
        let mut preds = self.train_cache.clone();
        self.obj.pred_transform(&mut preds);
        self.eval_preds(&preds, dtrain)
    }

    fn eval_preds(&self, preds: &[f32], dmat: &DMatrix) -> Vec<(&'static str, f64)> {
        self.metrics.iter().map(|m| (m.name(), m.eval(preds, dmat.info()))).collect()
    }

    pub fn predict_margin(&self, dmat: &DMatrix) -> Vec<f32> {
        crate::predictor::predict_margin(&self.gbm.model, dmat, self.base_score)
    }

    pub fn predict(&self, dmat: &DMatrix) -> Vec<f32> {
        let mut preds = self.predict_margin(dmat);
        self.obj.pred_transform(&mut preds);
        preds
    }

    /// Rebuild a learner from a deserialised model (no training state).
    pub(crate) fn from_model(
        obj: Box<dyn Objective>,
        metric: Box<dyn Metric>,
        gbm: GBTree,
        base_score: f32,
    ) -> Self {
        Self {
            ctx: Context::default(),
            obj,
            metrics: vec![metric],
            gbm,
            base_score,
            base_score_override: Some(base_score),
            configured: false,
            train_cache: Vec::new(),
        }
    }
}
