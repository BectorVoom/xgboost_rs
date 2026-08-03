//! The `Learner`: objective → booster → metric for one training run.

use crate::data::DMatrix;
use crate::gbm::GBTree;
use crate::metric::Metric;
use crate::objective::{GradientPair, Objective};
use crate::tree::param::TrainParam;
use crate::{Error, Result};

/// Orchestrates boosting rounds over a training matrix.
pub struct Learner {
    obj: Box<dyn Objective>,
    metric: Box<dyn Metric>,
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
        objective: &str,
        eval_metric: &str,
        num_feature: usize,
        param: TrainParam,
        base_score: Option<f32>,
    ) -> Result<Self> {
        Ok(Self {
            obj: crate::objective::create(objective)?,
            metric: crate::metric::create(eval_metric)?,
            gbm: GBTree::new(num_feature, param),
            base_score: 0.5,
            base_score_override: base_score,
            configured: false,
            train_cache: Vec::new(),
        })
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

    pub fn metric(&self) -> &dyn Metric {
        self.metric.as_ref()
    }

    pub fn boosted_rounds(&self) -> usize {
        self.gbm.model.num_trees()
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
        let mut gpair: Vec<GradientPair> = Vec::new();
        self.obj.get_gradient(&self.train_cache, dtrain.info(), iter, &mut gpair);
        self.gbm.do_boost(dtrain, &gpair, &mut self.train_cache)
    }

    /// Metric value on `dmat` for the model as it currently stands.
    pub fn eval(&self, dmat: &DMatrix) -> f64 {
        let mut preds = self.predict_margin(dmat);
        self.obj.pred_transform(&mut preds);
        self.metric.eval(&preds, dmat.info())
    }

    /// Metric on the training matrix, reusing the margin cache.
    pub fn eval_train(&self, dtrain: &DMatrix) -> f64 {
        let mut preds = self.train_cache.clone();
        self.obj.pred_transform(&mut preds);
        self.metric.eval(&preds, dtrain.info())
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
            obj,
            metric,
            gbm,
            base_score,
            base_score_override: Some(base_score),
            configured: false,
            train_cache: Vec::new(),
        }
    }
}
