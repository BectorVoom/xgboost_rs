//! The `gblinear` booster: a regularised linear model fitted by coordinate
//! descent.
//!
//! A port of `xgboost::gbm::GBLinear` (`src/gbm/gblinear.cc`) together with its
//! two updaters, `shotgun` (`src/linear/updater_shotgun.cc`) and
//! `coord_descent` (`src/linear/updater_coordinate.cc`).
//!
//! The model is one weight per `(feature, output group)` plus one intercept per
//! group, and a boosting round is one sweep of elastic-net coordinate descent
//! over those weights. Every step is a proximal Newton step on the objective's
//! current gradients, and the gradients are corrected in place after each
//! coordinate moves, so a sweep sees an increasingly up-to-date model.
//!
//! # What differs from upstream, and why
//!
//! Upstream's `shotgun` is Hogwild: it updates weights from several threads
//! with no synchronisation, so two runs on the same data can give different
//! models. This crate guarantees reproducibility, so `shotgun` is executed as
//! the sequential algorithm it approximates — features are swept in selector
//! order, one at a time — while the work *inside* one coordinate (the column
//! reduction) stays parallel and block-deterministic. The result is the model
//! Hogwild converges towards, without the race.
//!
//! `coord_descent` is already deterministic upstream and is ported as-is.

pub mod coordinate;
mod selector;

use crate::context::Context;
use crate::data::DMatrix;
use crate::data::csc::CscPages;
use crate::objective::GradientPair;
use crate::parameters::{FeatureSelector, LinearBoosterParameters, LinearUpdater};
use crate::{Error, Result};

use coordinate::{
    bias_gradient, column_gradient, coordinate_delta, coordinate_delta_bias, update_bias_residual,
    update_residual,
};
use selector::Selector;

/// The weights a `gblinear` fit produces, upstream's `GBLinearModel`.
///
/// `weight` is row-major `(feature, group)` with the intercepts appended as a
/// final "feature", which is the layout XGBoost's JSON model stores.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GBLinearModel {
    pub weight: Vec<f32>,
    pub num_feature: usize,
    pub num_output_group: usize,
    pub num_boosted_rounds: usize,
}

impl GBLinearModel {
    pub fn new(num_feature: usize, num_output_group: usize) -> Self {
        Self {
            weight: vec![0.0; (num_feature + 1) * num_output_group.max(1)],
            num_feature,
            num_output_group: num_output_group.max(1),
            num_boosted_rounds: 0,
        }
    }

    /// Weight of `(fidx, group)`.
    #[inline]
    pub fn weight_of(&self, fidx: usize, group: usize) -> f32 {
        self.weight[fidx * self.num_output_group + group]
    }

    /// The intercept of `group`.
    #[inline]
    pub fn bias(&self, group: usize) -> f32 {
        self.weight[self.num_feature * self.num_output_group + group]
    }

    /// Resize for a different output-group count, discarding any weights.
    fn resize(&mut self, num_feature: usize, num_output_group: usize) {
        let wanted = (num_feature + 1) * num_output_group.max(1);
        if self.weight.len() != wanted {
            self.weight = vec![0.0; wanted];
        }
        self.num_feature = num_feature;
        self.num_output_group = num_output_group.max(1);
    }
}

/// The `gblinear` booster.
pub struct GBLinear {
    pub model: GBLinearModel,
    param: LinearBoosterParameters,
    /// Column-major view of the training matrix, built once.
    pages: Option<CscPages>,
    /// `sum_instance_weight`, which denormalises the penalties.
    sum_instance_weight: f64,
    /// The weights as of the previous round, for the `tolerance` check.
    previous_weight: Vec<f32>,
    is_converged: bool,
    selector: Selector,
    /// Working copy of the round's gradients; coordinate descent rewrites them
    /// as it goes and the objective's buffer must survive.
    gpair: Vec<GradientPair>,
    /// The weights as of the start of the current round, so the prediction
    /// cache can be advanced by the difference instead of recomputed.
    round_start_weight: Vec<f32>,
    /// The transpose and the residual gradients, resident on the device.
    /// `Some` only for a `device=cuda` fit with the `coord_descent` updater.
    #[cfg(feature = "gpu")]
    gpu: Option<crate::gpu::linear::GpuLinear<crate::gpu::DefaultRuntime>>,
}

impl GBLinear {
    pub fn new(num_feature: usize, param: LinearBoosterParameters) -> Self {
        Self {
            model: GBLinearModel::new(num_feature, 1),
            selector: Selector::new(param.feature_selector),
            param,
            pages: None,
            sum_instance_weight: 0.0,
            previous_weight: Vec::new(),
            is_converged: false,
            gpair: Vec::new(),
            round_start_weight: Vec::new(),
            #[cfg(feature = "gpu")]
            gpu: None,
        }
    }

    /// Rebuild from a loaded model, with no training state.
    pub fn from_model(model: GBLinearModel) -> Self {
        Self {
            model,
            param: LinearBoosterParameters::default(),
            selector: Selector::new(FeatureSelector::Cyclic),
            pages: None,
            sum_instance_weight: 0.0,
            previous_weight: Vec::new(),
            is_converged: false,
            gpair: Vec::new(),
            round_start_weight: Vec::new(),
            #[cfg(feature = "gpu")]
            gpu: None,
        }
    }

    pub fn param(&self) -> &LinearBoosterParameters {
        &self.param
    }

    pub fn set_num_output_group(&mut self, n: usize) {
        if n.max(1) != self.model.num_output_group {
            self.model.resize(self.model.num_feature, n);
            self.previous_weight.clear();
            self.is_converged = false;
        }
    }

    /// Transpose the training matrix and total its row weights. Idempotent.
    pub fn configure(&mut self, dtrain: &DMatrix) -> Result<()> {
        if dtrain.info().has_categorical() {
            return Err(Error::invalid(
                "feature_types",
                "the `gblinear` booster has no categorical splits; mark every column \
                 numerical, or one-hot encode the categories yourself",
            ));
        }
        if self.pages.is_none() {
            let batch = self.param.max_row_perbatch.map(|r| r as usize);
            self.pages = Some(crate::threading::install(|| CscPages::build(dtrain, batch, false)));
            self.sum_instance_weight =
                (0..dtrain.num_row()).map(|r| dtrain.info().weight(r) as f64).sum();
        }
        Ok(())
    }

    /// Run one boosting round and fold the weight change into `preds`.
    ///
    /// `preds` is row-major `(row, group)` and already holds the intercept, so
    /// advancing it by the round's weight *delta* leaves it equal to a fresh
    /// prediction — without a second pass to recompute what has not moved.
    pub fn do_boost(
        &mut self,
        ctx: &mut Context,
        dtrain: &DMatrix,
        gpair: &[GradientPair],
        preds: &mut [f32],
    ) -> Result<()> {
        self.configure(dtrain)?;

        self.round_start_weight.clear();
        self.round_start_weight.extend_from_slice(&self.model.weight);

        if !self.check_convergence() {
            self.gpair.clear();
            self.gpair.extend_from_slice(gpair);
            let n_rows = dtrain.num_row();
            match self.param.updater {
                // `shotgun` is hogwild parallel coordinate descent and has no
                // device counterpart, upstream or here; it runs on the CPU
                // whatever `device` says, which is what
                // `BoosterParameters::warnings` warns about.
                LinearUpdater::Shotgun => self.shotgun(ctx, n_rows),
                #[cfg(feature = "gpu")]
                LinearUpdater::CoordDescent if ctx.device.is_cuda() => {
                    self.configure_device(ctx, n_rows);
                    self.coord_descent_gpu(ctx, n_rows);
                }
                LinearUpdater::CoordDescent => self.coord_descent(ctx, n_rows),
            }
        }
        self.model.num_boosted_rounds += 1;
        self.apply_delta(dtrain, preds);
        Ok(())
    }

    /// `GBLinear::CheckConvergence`: stop updating once the largest weight
    /// change between two consecutive rounds falls to `tolerance`.
    ///
    /// A `tolerance` of `0` — the default — disables the check entirely rather
    /// than stopping on an exactly-zero step.
    fn check_convergence(&mut self) -> bool {
        if self.param.tolerance == 0.0 {
            return false;
        }
        if self.is_converged {
            return true;
        }
        if self.previous_weight.len() != self.model.weight.len() {
            self.previous_weight.clone_from(&self.model.weight);
            return false;
        }
        let largest = self
            .model
            .weight
            .iter()
            .zip(&self.previous_weight)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        self.previous_weight.clone_from(&self.model.weight);
        self.is_converged = largest <= self.param.tolerance;
        self.is_converged
    }

    /// Penalties scaled by the total row weight, upstream's
    /// `DenormalizePenalties`. Without it `lambda` would mean something
    /// different on every dataset size.
    fn penalties(&self) -> (f64, f64) {
        (
            self.param.alpha as f64 * self.sum_instance_weight,
            self.param.lambda as f64 * self.sum_instance_weight,
        )
    }

    /// Fit every intercept, correcting the residuals as each one moves.
    fn update_bias(&mut self, n_rows: usize) {
        let n_groups = self.model.num_output_group;
        let lr = self.param.eta as f64;
        for gid in 0..n_groups {
            let (g, h) = bias_gradient(0..n_rows, gid, n_groups, &self.gpair);
            let dbias = (lr * coordinate_delta_bias(g, h)) as f32;
            self.model.weight[self.model.num_feature * n_groups + gid] += dbias;
            update_bias_residual(0..n_rows, gid, n_groups, dbias, &mut self.gpair);
        }
    }

    /// `ShotgunUpdater::Update`, executed sequentially (see the module docs).
    ///
    /// One sweep per row batch, features outer and groups inner — the opposite
    /// nesting to `coord_descent`, which is what makes the two updaters give
    /// different models from the same selector.
    fn shotgun(&mut self, ctx: &mut Context, n_rows: usize) {
        self.update_bias(n_rows);

        let (alpha, lambda) = self.penalties();
        let n_features = self.model.num_feature;
        let n_groups = self.model.num_output_group;
        let lr = self.param.eta as f64;
        let pages = self.pages.take().expect("configured");

        self.selector.setup(
            ctx,
            &self.model.weight,
            &self.gpair,
            &pages,
            n_features,
            n_groups,
            alpha,
            lambda,
            self.param.top_k,
        );

        for page in pages.iter() {
            for i in 0..n_features {
                let Some(fidx) = self.selector.next_feature(
                    ctx,
                    i,
                    &self.model.weight,
                    &self.gpair,
                    &pages,
                    n_features,
                    0,
                    n_groups,
                    alpha,
                    lambda,
                ) else {
                    break;
                };
                for gid in 0..n_groups {
                    let (g, h) = column_gradient(page, fidx, gid, n_groups, &self.gpair);
                    let w = self.model.weight[fidx * n_groups + gid];
                    let dw = (lr * coordinate_delta(g, h, w as f64, alpha, lambda)) as f32;
                    if dw == 0.0 {
                        continue;
                    }
                    self.model.weight[fidx * n_groups + gid] = w + dw;
                    update_residual(page, fidx, gid, n_groups, dw, &mut self.gpair);
                }
            }
        }
        self.pages = Some(pages);
    }

    /// `CoordinateUpdater::Update`: groups outer, features inner, and every
    /// feature's gradient summed over the whole matrix rather than one batch.
    fn coord_descent(&mut self, ctx: &mut Context, n_rows: usize) {
        self.update_bias(n_rows);

        let (alpha, lambda) = self.penalties();
        let n_features = self.model.num_feature;
        let n_groups = self.model.num_output_group;
        let lr = self.param.eta as f64;
        let pages = self.pages.take().expect("configured");

        self.selector.setup(
            ctx,
            &self.model.weight,
            &self.gpair,
            &pages,
            n_features,
            n_groups,
            alpha,
            lambda,
            self.param.top_k,
        );

        for gid in 0..n_groups {
            for i in 0..n_features {
                let Some(fidx) = self.selector.next_feature(
                    ctx,
                    i,
                    &self.model.weight,
                    &self.gpair,
                    &pages,
                    n_features,
                    gid,
                    n_groups,
                    alpha,
                    lambda,
                ) else {
                    break;
                };
                let (mut g, mut h) = (0.0f64, 0.0f64);
                for page in pages.iter() {
                    let (pg, ph) = column_gradient(page, fidx, gid, n_groups, &self.gpair);
                    g += pg;
                    h += ph;
                }
                let w = self.model.weight[fidx * n_groups + gid];
                let dw = (lr * coordinate_delta(g, h, w as f64, alpha, lambda)) as f32;
                self.model.weight[fidx * n_groups + gid] = w + dw;
                for page in pages.iter() {
                    update_residual(page, fidx, gid, n_groups, dw, &mut self.gpair);
                }
            }
        }
        self.pages = Some(pages);
    }

    /// Upload the transpose, once per fit. Idempotent.
    #[cfg(feature = "gpu")]
    fn configure_device(&mut self, ctx: &Context, n_rows: usize) {
        if self.gpu.is_some() {
            return;
        }
        let ordinal = ctx.device.ordinal().unwrap_or(0).max(0) as usize;
        self.gpu = Some(crate::gpu::linear::GpuLinear::new(
            crate::gpu::default_client(ordinal),
            self.pages.as_ref().expect("configured"),
            n_rows,
            self.model.num_output_group,
        ));
    }

    /// [`coord_descent`](Self::coord_descent) with both O(nnz) passes on the
    /// device: upstream's `GPUCoordinateUpdater`.
    ///
    /// Step for step the same loop, and deliberately so — the device sums a
    /// column in the same fixed block order the CPU does
    /// (`crate::gpu::linear`), so the two produce the *same* model rather than
    /// two close ones, and `tests/gpu_linear.rs` holds them to that.
    ///
    /// The gradients live on the device between features, because that is the
    /// point: each step corrects them in place and the next column's sum reads
    /// them back. They come home only when a feature selector wants to score
    /// features by them, which two of the five do.
    #[cfg(feature = "gpu")]
    fn coord_descent_gpu(&mut self, ctx: &mut Context, n_rows: usize) {
        let (alpha, lambda) = self.penalties();
        let n_features = self.model.num_feature;
        let n_groups = self.model.num_output_group;
        let lr = self.param.eta as f64;
        // Only `greedy` and `thrifty` score features by the residuals; for the
        // other three the host copy is never read and never needs fetching.
        let selector_reads_gradients = matches!(
            self.param.feature_selector,
            FeatureSelector::Greedy | FeatureSelector::Thrifty
        );

        let pages = self.pages.take().expect("configured");
        let gpu = self.gpu.as_mut().expect("configured above");
        gpu.upload_gpair(&self.gpair);

        // The intercepts first, exactly as `update_bias` does.
        for gid in 0..n_groups {
            let (g, h) = gpu.bias_gradient(gid);
            let dbias = (lr * coordinate_delta_bias(g, h)) as f32;
            self.model.weight[n_features * n_groups + gid] += dbias;
            gpu.update_bias_residual(gid, dbias);
        }

        if selector_reads_gradients {
            gpu.download_gpair(&mut self.gpair);
        }
        self.selector.setup(
            ctx,
            &self.model.weight,
            &self.gpair,
            &pages,
            n_features,
            n_groups,
            alpha,
            lambda,
            self.param.top_k,
        );

        for gid in 0..n_groups {
            for i in 0..n_features {
                if selector_reads_gradients {
                    gpu.download_gpair(&mut self.gpair);
                }
                let Some(fidx) = self.selector.next_feature(
                    ctx,
                    i,
                    &self.model.weight,
                    &self.gpair,
                    &pages,
                    n_features,
                    gid,
                    n_groups,
                    alpha,
                    lambda,
                ) else {
                    break;
                };
                let (mut g, mut h) = (0.0f64, 0.0f64);
                for page in 0..gpu.num_pages() {
                    let (pg, ph) = gpu.column_gradient(page, fidx, gid);
                    g += pg;
                    h += ph;
                }
                let w = self.model.weight[fidx * n_groups + gid];
                let dw = (lr * coordinate_delta(g, h, w as f64, alpha, lambda)) as f32;
                self.model.weight[fidx * n_groups + gid] = w + dw;
                for page in 0..gpu.num_pages() {
                    gpu.update_residual(page, fidx, gid, dw);
                }
            }
        }

        // The round's residuals are the next round's starting point only
        // through the model, but `check_convergence` and the tests read the
        // host copy, so it is left in step with the device.
        gpu.download_gpair(&mut self.gpair);
        let _ = n_rows;
        self.pages = Some(pages);
    }

    /// Advance the prediction cache by this round's weight change.
    fn apply_delta(&self, dtrain: &DMatrix, preds: &mut [f32]) {
        let n_groups = self.model.num_output_group;
        let n_features = self.model.num_feature;
        let delta: Vec<f32> = self
            .model
            .weight
            .iter()
            .zip(&self.round_start_weight)
            .map(|(new, old)| new - old)
            .collect();
        if delta.iter().all(|d| *d == 0.0) {
            return;
        }
        for r in 0..dtrain.num_row() {
            let (idx, val) = dtrain.row(r);
            for gid in 0..n_groups {
                let mut acc = delta[n_features * n_groups + gid];
                for (&c, &v) in idx.iter().zip(val) {
                    acc += v * delta[c as usize * n_groups + gid];
                }
                preds[r * n_groups + gid] += acc;
            }
        }
    }

    /// Raw margins for `dmat`: the intercept plus the linear terms.
    pub fn predict_margin(&self, dmat: &DMatrix, base_margin: &[f32]) -> Vec<f32> {
        let n_groups = self.model.num_output_group;
        let mut preds =
            crate::predictor::init_margin(dmat, base_margin, dmat.num_row(), n_groups);
        for r in 0..dmat.num_row() {
            let (idx, val) = dmat.row(r);
            for gid in 0..n_groups {
                let mut acc = self.model.bias(gid);
                for (&c, &v) in idx.iter().zip(val) {
                    acc += v * self.model.weight_of(c as usize, gid);
                }
                preds[r * n_groups + gid] += acc;
            }
        }
        preds
    }

    /// Feature contributions: `num_feature + 1` values per `(row, group)`, the
    /// last being the bias.
    ///
    /// A linear model's attribution is exact and needs no path enumeration —
    /// feature `f` contributes `x_f * w_f` and nothing else.
    pub fn predict_contribution(&self, dmat: &DMatrix, base_margin: &[f32]) -> Vec<f32> {
        let n_groups = self.model.num_output_group;
        let width = self.model.num_feature + 1;
        let margins =
            crate::predictor::init_margin(dmat, base_margin, dmat.num_row(), n_groups);
        let mut out = vec![0.0f32; dmat.num_row() * n_groups * width];
        for r in 0..dmat.num_row() {
            let (idx, val) = dmat.row(r);
            for gid in 0..n_groups {
                let base = (r * n_groups + gid) * width;
                for (&c, &v) in idx.iter().zip(val) {
                    out[base + c as usize] = v * self.model.weight_of(c as usize, gid);
                }
                out[base + width - 1] = self.model.bias(gid) + margins[r * n_groups + gid];
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameters::LinearBoosterParametersBuilder;

    /// `y = 2 * x0 - 3 * x1 + 1`, which a linear model can fit exactly.
    fn linear_data(n: usize) -> (DMatrix, Vec<f32>) {
        let mut state = 0x1234_5678u64;
        let mut next = || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            ((state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32) / ((1u32 << 24) as f32)
        };
        let x: Vec<f32> = (0..n * 2).map(|_| next()).collect();
        let y: Vec<f32> =
            (0..n).map(|r| 2.0 * x[r * 2] - 3.0 * x[r * 2 + 1] + 1.0).collect();
        let mut d = DMatrix::from_dense(&x, n, 2, f32::NAN).unwrap();
        d.set_labels(&y).unwrap();
        (d, y)
    }

    /// Squared-error gradients against a current margin.
    fn gradients(preds: &[f32], labels: &[f32]) -> Vec<GradientPair> {
        preds
            .iter()
            .zip(labels)
            .map(|(p, y)| GradientPair { grad: p - y, hess: 1.0 })
            .collect()
    }

    fn fit(param: LinearBoosterParametersBuilder, rounds: usize) -> (GBLinear, Vec<f32>, Vec<f32>) {
        let (d, y) = linear_data(200);
        let mut booster = GBLinear::new(2, param.build().unwrap());
        let mut ctx = Context::default();
        let mut preds = vec![0.0f32; d.num_row()];
        for _ in 0..rounds {
            let gpair = gradients(&preds, &y);
            booster.do_boost(&mut ctx, &d, &gpair, &mut preds).unwrap();
        }
        (booster, preds, y)
    }

    fn rmse(preds: &[f32], y: &[f32]) -> f32 {
        (preds.iter().zip(y).map(|(p, t)| (p - t).powi(2)).sum::<f32>() / y.len() as f32).sqrt()
    }

    #[test]
    fn coordinate_descent_recovers_the_generating_weights() {
        let (booster, preds, y) = fit(
            LinearBoosterParameters::builder().updater(LinearUpdater::CoordDescent).eta(0.5),
            200,
        );
        assert!(rmse(&preds, &y) < 1e-2, "rmse {}", rmse(&preds, &y));
        assert!((booster.model.weight_of(0, 0) - 2.0).abs() < 0.05);
        assert!((booster.model.weight_of(1, 0) + 3.0).abs() < 0.05);
        assert!((booster.model.bias(0) - 1.0).abs() < 0.05);
    }

    #[test]
    fn shotgun_fits_the_same_model_as_coordinate_descent() {
        let (a, pa, y) = fit(LinearBoosterParameters::builder().eta(0.5), 200);
        let (b, pb, _) = fit(
            LinearBoosterParameters::builder().updater(LinearUpdater::CoordDescent).eta(0.5),
            200,
        );
        assert!(rmse(&pa, &y) < 1e-2);
        // Different sweep order, same optimum.
        assert!((a.model.weight_of(0, 0) - b.model.weight_of(0, 0)).abs() < 0.05);
        assert!((rmse(&pa, &y) - rmse(&pb, &y)).abs() < 1e-3);
    }

    #[test]
    fn every_feature_selector_fits() {
        for (updater, selector) in [
            (LinearUpdater::Shotgun, FeatureSelector::Cyclic),
            (LinearUpdater::Shotgun, FeatureSelector::Shuffle),
            (LinearUpdater::CoordDescent, FeatureSelector::Cyclic),
            (LinearUpdater::CoordDescent, FeatureSelector::Shuffle),
            (LinearUpdater::CoordDescent, FeatureSelector::Random),
            (LinearUpdater::CoordDescent, FeatureSelector::Greedy),
            (LinearUpdater::CoordDescent, FeatureSelector::Thrifty),
        ] {
            let (_, preds, y) = fit(
                LinearBoosterParameters::builder()
                    .updater(updater)
                    .feature_selector(selector)
                    .eta(0.5),
                300,
            );
            assert!(rmse(&preds, &y) < 0.2, "{updater}/{selector}: rmse {}", rmse(&preds, &y));
        }
    }

    #[test]
    fn l1_drops_a_weak_feature_and_keeps_a_strong_one() {
        // A third column with a real but tiny coefficient, so `alpha = 0` keeps
        // it and the only question is what the penalty does.
        let (base, base_y) = linear_data(200);
        let mut x = Vec::new();
        let mut y = Vec::new();
        for r in 0..base.num_row() {
            let (_, val) = base.row(r);
            let weak = ((r * 37) % 11) as f32 / 11.0;
            x.extend_from_slice(val);
            x.push(weak);
            y.push(base_y[r] + 0.02 * weak);
        }
        let d = DMatrix::from_dense(&x, y.len(), 3, f32::NAN).unwrap();

        // `eta = 1` takes the whole proximal step, so the L1 clamp at `-w`
        // lands a dropped weight exactly on zero. Any smaller learning rate
        // only ever halves it, and it decays towards zero without arriving.
        let fit_with = |alpha: f32| {
            let param = LinearBoosterParameters::builder()
                .updater(LinearUpdater::CoordDescent)
                .alpha(alpha)
                .eta(1.0)
                .build()
                .unwrap();
            let mut booster = GBLinear::new(3, param);
            let mut ctx = Context::default();
            let mut preds = vec![0.0f32; d.num_row()];
            for _ in 0..200 {
                let gpair = gradients(&preds, &y);
                booster.do_boost(&mut ctx, &d, &gpair, &mut preds).unwrap();
            }
            booster
        };

        let unpenalised = fit_with(0.0);
        assert_ne!(unpenalised.model.weight_of(2, 0), 0.0, "without L1 the weak column is kept");

        // `alpha` is denormalised by the total row weight, so 0.05 over 200
        // rows is a threshold of 10 — above the weak column's gradient and far
        // below the strong one's.
        let penalised = fit_with(0.05);
        assert_eq!(penalised.model.weight_of(2, 0), 0.0, "L1 zeroes the weak column outright");
        assert!(penalised.model.weight_of(0, 0) > 1.0, "and leaves the strong one standing");
    }

    #[test]
    fn tolerance_stops_updating_once_the_weights_settle() {
        let param = LinearBoosterParameters::builder()
            .updater(LinearUpdater::CoordDescent)
            .tolerance(1e9)
            .build()
            .unwrap();
        let (d, y) = linear_data(50);
        let mut booster = GBLinear::new(2, param);
        let mut ctx = Context::default();
        let mut preds = vec![0.0f32; d.num_row()];
        for _ in 0..5 {
            let gpair = gradients(&preds, &y);
            booster.do_boost(&mut ctx, &d, &gpair, &mut preds).unwrap();
        }
        // Round 1 fits, round 2 sees a change below the tolerance and latches.
        assert!(booster.is_converged);
        assert_eq!(booster.model.num_boosted_rounds, 5, "rounds still count");
    }

    #[test]
    fn the_prediction_cache_matches_a_fresh_prediction() {
        let (d, y) = linear_data(120);
        let mut booster =
            GBLinear::new(2, LinearBoosterParameters::builder().eta(0.4).build().unwrap());
        let mut ctx = Context::default();
        let mut preds = vec![0.0f32; d.num_row()];
        for _ in 0..30 {
            let gpair = gradients(&preds, &y);
            booster.do_boost(&mut ctx, &d, &gpair, &mut preds).unwrap();
        }
        let fresh = booster.predict_margin(&d, &[0.0]);
        for (a, b) in preds.iter().zip(&fresh) {
            assert!((a - b).abs() < 1e-4, "cache {a} vs fresh {b}");
        }
    }

    #[test]
    fn contributions_sum_to_the_prediction() {
        let (d, y) = linear_data(60);
        let mut booster =
            GBLinear::new(2, LinearBoosterParameters::builder().eta(0.5).build().unwrap());
        let mut ctx = Context::default();
        let mut preds = vec![0.0f32; d.num_row()];
        for _ in 0..40 {
            let gpair = gradients(&preds, &y);
            booster.do_boost(&mut ctx, &d, &gpair, &mut preds).unwrap();
        }
        let contribs = booster.predict_contribution(&d, &[0.0]);
        let margins = booster.predict_margin(&d, &[0.0]);
        for r in 0..d.num_row() {
            let sum: f32 = contribs[r * 3..(r + 1) * 3].iter().sum();
            assert!((sum - margins[r]).abs() < 1e-4, "row {r}: {sum} vs {}", margins[r]);
        }
    }

    #[test]
    fn batching_rows_changes_shotgun_but_not_coordinate_descent() {
        let unbatched = |updater| {
            fit(LinearBoosterParameters::builder().updater(updater).eta(0.5), 20).0.model.weight
        };
        let batched = |updater| {
            fit(
                LinearBoosterParameters::builder()
                    .updater(updater)
                    .eta(0.5)
                    .max_row_perbatch(50),
                20,
            )
            .0
            .model
            .weight
        };
        assert_eq!(
            unbatched(LinearUpdater::CoordDescent),
            batched(LinearUpdater::CoordDescent),
            "coord_descent sums over every batch, so the split cannot matter"
        );
        assert_ne!(
            unbatched(LinearUpdater::Shotgun),
            batched(LinearUpdater::Shotgun),
            "shotgun sweeps once per batch, so the split is part of the model"
        );
    }

    #[test]
    fn categorical_columns_are_rejected_rather_than_treated_as_numbers() {
        let (mut d, _) = linear_data(10);
        d.set_feature_types(&[crate::FeatureType::Categorical, crate::FeatureType::Numerical])
            .unwrap();
        let mut booster =
            GBLinear::new(2, LinearBoosterParameters::default());
        let err = booster.configure(&d).unwrap_err();
        assert!(err.to_string().contains("categorical"), "{err}");
    }

    #[test]
    fn multiple_output_groups_get_their_own_weights() {
        let (d, y) = linear_data(100);
        // Two groups: the second sees the negated label.
        let mut booster = GBLinear::new(2, LinearBoosterParameters::default());
        booster.set_num_output_group(2);
        let mut ctx = Context::default();
        let mut preds = vec![0.0f32; d.num_row() * 2];
        for _ in 0..100 {
            let gpair: Vec<GradientPair> = (0..d.num_row())
                .flat_map(|r| {
                    [
                        GradientPair { grad: preds[r * 2] - y[r], hess: 1.0 },
                        GradientPair { grad: preds[r * 2 + 1] + y[r], hess: 1.0 },
                    ]
                })
                .collect();
            booster.do_boost(&mut ctx, &d, &gpair, &mut preds).unwrap();
        }
        assert!((booster.model.weight_of(0, 0) + booster.model.weight_of(0, 1)).abs() < 0.05);
        assert!(booster.model.weight_of(0, 0) > 0.5);
    }
}
