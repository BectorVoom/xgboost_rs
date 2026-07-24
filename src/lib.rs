//! Rust port of XGBoost's `gpu_hist` CUDA device kernels, rewritten with CubeCL.
//!
//! The kernels are runtime-generic: they run on any CubeCL runtime (Vulkan/wgpu
//! by default, CUDA with the `cuda` cargo feature).
//!
//! # Overview
//!
//! ```text
//! GradientQuantiser          f32 gradients -> deterministic fixed-point i64
//! HistogramBuilder (builder) configures + uploads an EllpackMatrix
//!   -> HistogramEngine       builds per-node gradient histograms on device
//! ```

pub mod error;
pub mod gpu;
pub mod reference;

pub use error::{Error, Result};
