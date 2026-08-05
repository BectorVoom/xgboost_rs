//! Rust port of XGBoost.
//!
//! # Training
//!
//! The CPU `hist` path trains `gbtree` and `dart` models end to end, for every
//! XGBoost objective and every evaluation metric:
//!
//! ```no_run
//! use xgboost_rs::{DMatrix, api, parameters::TrainingParameters};
//!
//! let mut dtrain = DMatrix::from_dense(&[0.0, 1.0, 2.0, 3.0], 2, 2, f32::NAN)?;
//! dtrain.set_labels(&[0.5, 1.5])?;
//!
//! let params = TrainingParameters { num_boost_round: 10, ..Default::default() };
//! let (booster, history) = api::train(&params, &dtrain, &[(&dtrain, "train")])?;
//!
//! let preds = booster.predict(&dtrain);
//! let model = booster.save_model(); // XGBoost's JSON format
//! # Ok::<(), xgboost_rs::Error>(())
//! ```
//!
//! ## Layers
//!
//! ```text
//! api            train() / Booster: predict, eval, importance, model IO
//!   learner      objective -> booster -> metrics for one run
//!     gbm        the tree ensemble, one boosting round, and DART dropout
//!       tree     RegTree, the split arithmetic, the hist updater, row and
//!                column sampling, the two constraint systems, TreeSHAP
//!     objective  gradients, the base_score intercept, and the prediction
//!                transform, for every XGBoost objective
//!     metric     every XGBoost eval_metric
//!   context      threads, seed, and the session random engine
//!   rng          the C++ generators sampling decisions are drawn from
//!   data         DMatrix, quantile cuts, the binned feature matrix
//! ```
//!
//! ## What the fit reads
//!
//! **Tree booster** — everything a CPU `hist` fit can act on: `eta`, `gamma`,
//! `max_depth`, `max_leaves`, `max_bin`, `grow_policy`, `min_child_weight`,
//! `lambda`, `alpha`, `max_delta_step`, `subsample` with either
//! `sampling_method`, all three `colsample_*` ratios, `num_parallel_tree`,
//! `monotone_constraints`, `interaction_constraints` and
//! `max_cached_hist_node`.
//!
//! **DART booster** — `sample_type`, `normalize_type`, `rate_drop`, `one_drop`
//! and `skip_drop`, with per-tree weights carried through prediction and model
//! IO.
//!
//! **Learning task** — every `objective` (all 21, from `reg:squarederror`
//! through `survival:aft` and the three `rank:*` losses) and every
//! `eval_metric`, including the parameterised spellings `error@t`,
//! `tweedie-nloglik@rho`, `ndcg@n-`, `map@n-`, `pre@n` and `ams@t`. Also
//! `base_score`, `boost_from_average`, `seed`, `seed_per_iteration`,
//! `scale_pos_weight`, `num_class`, `num_target` and the per-objective knobs
//! (`huber_slope`, `quantile_alpha`, `expectile_alpha`,
//! `tweedie_variance_power`, `aft_loss_distribution*`, the `lambdarank_*`
//! family).
//!
//! **General** — `device` (CPU), `nthread`, `verbosity`,
//! `disable_default_eval_metric` and `validate_parameters`.
//!
//! **Prediction** — `predict_type` in all seven kinds (value, margin, leaf,
//! exact and approximate SHAP contributions and interactions),
//! `iteration_range`, `strict_shape` and `validate_features`.
//!
//! **Training loop** — `num_boost_round`, `early_stopping_rounds`,
//! `verbose_eval` and `maximize`.
//!
//! ## What is rejected rather than ignored
//!
//! A parameter that selects an algorithm this crate does not have is an error
//! from [`train`], never a silent fallback: the `exact` and `approx` tree
//! methods and their updaters, `process_type = update` with the `prune` and
//! `refresh` updaters, `multi_strategy = multi_output_tree` (vector leaves),
//! the `gblinear` booster, categorical feature types, and any non-CPU `device`.
//! Parameters that only steer one of those — `default_direction`,
//! `opt_dense_col`, `refresh_leaf`, `max_cat_to_onehot`, `max_cat_threshold`,
//! `use_rmm`, `fail_on_invalid_gpu_id` — are accepted but inert, and setting
//! `validate_parameters` makes the fit say so.
//!
//! Two further modules stand beside them:
//!
//! * [`parameters`] — the full XGBoost training ("fit") and prediction
//!   parameter surface for both CPU and GPU: typed enums, consuming builders,
//!   validation, and XGBoost-compatible key/value + JSON config emission. Pure
//!   CPU code, no GPU toolchain required.
//! * [`gpu`] — XGBoost's `gpu_hist` CUDA device kernels rewritten with CubeCL.
//!   The kernels are runtime-generic: they run on any CubeCL runtime
//!   (Vulkan/wgpu by default, CUDA with the `cuda` cargo feature). Gated behind
//!   the default-on `gpu` feature, because building CubeCL needs a backend
//!   toolchain (on macOS, the Vulkan SDK) that CPU-only consumers should not
//!   have to install. Build with `--no-default-features` to skip it; the
//!   training path above never needs it.
//!
//! # Agreement with XGBoost
//!
//! Training is a port, not a re-derivation: quantile cuts, the order of
//! histogram accumulation, which child is built versus subtracted, the
//! float/double split arithmetic and the tie-breaking rule all follow
//! xgboost 3.0.5. `tests/oracle.rs` and `tests/oracle_train.rs` assert this
//! against committed fixtures from that pinned version — cut points bit for
//! bit, tree structure and leaf assignment exactly, predictions and metrics
//! within `1e-5`.
//!
//! # Parallelism
//!
//! Training uses every core by default; set `nthread` on the parameters for a
//! single fit, or [`set_num_threads`] for the process. It is deterministic:
//! rows are cut into fixed-size blocks, blocks are dealt to a fixed number of
//! lanes, and partial results are reduced in lane order, none of which depends
//! on the thread count. Row sampling is deterministic the same way — row `i`'s
//! draw is a closed-form function of `i` and the seed. The same data and
//! parameters give a bit-identical model on one core or on many, which
//! `tests/determinism.rs` checks.

pub mod api;
pub mod context;
pub mod data;
pub mod error;
pub mod gbm;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod learner;
pub mod metric;
pub mod model_io;
pub mod objective;
pub mod parameters;
pub mod predictor;
#[cfg(feature = "gpu")]
pub mod reference;
pub mod rng;
pub mod threading;
pub mod tree;

pub use api::{Booster, Prediction, train};
pub use context::Context;
pub use data::{DMatrix, FeatureType};
pub use error::{Error, Result};
pub use threading::{num_threads, set_num_threads};
