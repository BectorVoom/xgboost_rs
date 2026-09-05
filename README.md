# xgboost_rs

Fast, pure Rust implementation of XGBoost with CPU and GPU acceleration.

## Features

- **Pure Rust Engine:** High-performance histogram-based decision tree growth (`hist`).
- **Support for Grow Policies:** `depthwise` (depth-first level-by-level) and `lossguide` (leaf-wise best-first).
- **One kernel source for CPU and GPU:** the `gpu_hist` kernels are written once
  over CubeCL's `Runtime` and run on the CubeCL CPU runtime, on Vulkan, or on
  CUDA. The default build needs no GPU toolchain at all, so the device code is
  buildable and testable on any machine — and produces the same model.
- **Python Bindings:** Ready for PyPI installation via `maturin`.

## Installation

### Rust Crate

Add to your `Cargo.toml`:

```toml
[dependencies]
xgboost_rs = "0.1.0"
```

The default build (`gpu` + `cpu`) compiles the kernels against the CubeCL CPU
runtime and needs nothing installed. Pick an accelerator backend to actually
reach a GPU:

```toml
# NVIDIA, via CUDA — needs the CUDA toolkit.
xgboost_rs = { version = "0.1.0", features = ["cuda"] }

# Anything with Vulkan, via wgpu + SPIR-V — needs the Vulkan SDK.
xgboost_rs = { version = "0.1.0", features = ["vulkan"] }

# Apple GPUs, via wgpu + Metal Shading Language — needs nothing installed.
# Runs the integer kernels only; see the note below.
xgboost_rs = { version = "0.1.0", features = ["metal"] }

# No kernels at all: parameters, data and the CPU trainer only.
xgboost_rs = { version = "0.1.0", default-features = false }
```

`cuda` wins over `vulkan` over `metal` over `cpu`; `xgboost_rs::gpu::BACKEND`
names whichever the build resolved to.

**Metal cannot run a fit.** Metal Shading Language has no `double`, and this
crate's split gain arithmetic, gradient quantiser and `gblinear` solver are
ports of XGBoost's `double` — demoting them to `f32` would be a different model,
not the same one computed differently. A `metal` build runs the ELLPACK,
histogram, subtraction-trick and row-partitioning kernels (all `i64`), and
refuses a fit with `Error::NoF64Support`. It is useful as *coverage*: it is the
only backend that needs no toolchain and still executes cubes concurrently, so
it is what checks the shared-memory kernels against a real GPU.

### Python Package (PyPI)

```bash
pip install xgboost_rs
```

## Quick Start (Rust)

```rust
use xgboost_rs::parameters::{TrainParam, GrowPolicy};

let param = TrainParam {
    max_depth: 6,
    grow_policy: GrowPolicy::DepthWise,
    ..Default::default()
};
```

## License

Licensed under Apache License, Version 2.0 ([LICENSE](LICENSE)).
