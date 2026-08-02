//! Rust port of XGBoost.
//!
//! Two layers exist today:
//!
//! * [`parameters`] — the full XGBoost training ("fit") parameter surface for
//!   both CPU and GPU: typed enums, consuming builders, validation, and
//!   XGBoost-compatible key/value + JSON config emission. Pure CPU code, no GPU
//!   toolchain required.
//! * [`gpu`] — XGBoost's `gpu_hist` CUDA device kernels rewritten with CubeCL.
//!   The kernels are runtime-generic: they run on any CubeCL runtime
//!   (Vulkan/wgpu by default, CUDA with the `cuda` cargo feature). Gated behind
//!   the default-on `gpu` feature, because building CubeCL needs a backend
//!   toolchain (on macOS, the Vulkan SDK) that CPU-only consumers should not
//!   have to install.
//!
//! # Overview
//!
//! ```text
//! GradientQuantiser          f32 gradients -> deterministic fixed-point i64
//! HistogramBuilder (builder) configures + uploads an EllpackMatrix
//!   -> HistogramEngine       builds per-node gradient histograms on device
//! ```

pub mod error;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod parameters;
#[cfg(feature = "gpu")]
pub mod reference;

pub use error::{Error, Result};
