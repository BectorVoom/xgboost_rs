# xgboost_rs

Fast, pure Rust implementation of XGBoost with CPU and GPU acceleration.

## Features

- **Pure Rust Engine:** High-performance histogram-based decision tree growth (`hist`).
- **Support for Grow Policies:** `depthwise` (depth-first level-by-level) and `lossguide` (leaf-wise best-first).
- **GPU Acceleration:** Optional GPU backend via `cubecl` / Vulkan / CUDA.
- **Python Bindings:** Ready for PyPI installation via `maturin`.

## Installation

### Rust Crate

Add to your `Cargo.toml`:

```toml
[dependencies]
xgboost_rs = { version = "0.1.0", default-features = false }
```

To enable GPU acceleration:

```toml
[dependencies]
xgboost_rs = { version = "0.1.0", features = ["gpu"] }
```

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
