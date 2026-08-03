//! Rust port of XGBoost.
//!
//! # Training
//!
//! The CPU `hist` path trains `reg:squarederror` gbtree models end to end:
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
//!   learner      objective -> booster -> metric for one run
//!     gbm        the tree ensemble and one boosting round
//!       tree     RegTree, the split arithmetic, and the hist updater
//!     objective  gradients and the base_score intercept
//!     metric     rmse
//!   data         DMatrix, quantile cuts, the binned feature matrix
//! ```
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
//! Training uses every core by default; see [`set_num_threads`]. It is
//! deterministic: rows are cut into fixed-size blocks, blocks are dealt to a
//! fixed number of lanes, and partial results are reduced in lane order, none
//! of which depends on the thread count. The same data and parameters give a
//! bit-identical model on one core or on many, which `tests/determinism.rs`
//! checks.

pub mod api;
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
pub mod threading;
pub mod tree;

pub use api::{Booster, train};
pub use data::DMatrix;
pub use error::{Error, Result};
pub use threading::{num_threads, set_num_threads};
