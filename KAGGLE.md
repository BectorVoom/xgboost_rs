# Running the oracle + speed benchmark on a Kaggle CUDA GPU

The `bench` binary tests every kernel path against a CPU oracle (exact `i64`
equality) and then times histogram builds. Built with `--features cuda` it runs
on CubeCL's CUDA runtime (NVRTC-compiled kernels); Kaggle's GPU images ship the
CUDA driver and toolkit it needs.

## Setup

1. Create a Kaggle notebook, **Settings → Accelerator → GPU** (T4/P100/L4).
2. Upload `xgboost_rs_kaggle.tar.gz` (created at the workspace root) as a
   Kaggle *dataset*, e.g. named `xgboost-rs`.
3. Run these cells:

```bash
# Cell 1 — install Rust (~1 min)
!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal

# Cell 2 — unpack the crate
!mkdir -p /kaggle/working/xgboost_rs && tar xzf /kaggle/input/xgboost-rs/xgboost_rs_kaggle.tar.gz -C /kaggle/working/xgboost_rs

# Cell 3 — build + run oracle and benchmark on CUDA (~3-5 min first build)
!cd /kaggle/working/xgboost_rs && ~/.cargo/bin/cargo run --release --features cuda --bin bench
```

## Expected output

```
runtime: cuda / device: ...

== oracle (cuda runtime) ==
quantise                          .. ok
native i64 atomics: supported
hist dense/shared                 .. ok
hist dense/global/u32-split       .. ok
hist dense-compressed/shared      .. ok
hist sparse/shared                .. ok
hist sparse/global/u32-split      .. ok
hist dense/global/i64-native      .. ok
hist sparse/global/i64-native     .. ok
hist sparse/shared/i64-flush      .. ok
subtraction trick                 .. ok
oracle: ALL PASS

== speed (cuda runtime) ==
rows=1048576 features=32 bins/feature=256 total-bins=8192 iters=20
native i64 atomics: supported
dense/shared           ...  ms/build  ...  Gentry/s
dense/global/u32       ...
dense/global/i64       ...
sparse0.5/shared       ...
sparse0.5/global/u32   ...
sparse0.5/global/i64   ...
```

The `global/u32` vs `global/i64` pair isolates the u32-carry emulation against
native 64-bit atomicAdd (what XGBoost's `AtomicAddGpairGlobal` uses); on CUDA
the engine auto-selects native atomics unless overridden with
`HistogramBuilder::native_i64_atomics(false)`.

If any oracle case fails the binary exits non-zero with the failing case name.

## Knobs

Benchmark size via environment variables (defaults in parentheses):

```bash
!cd /kaggle/working/xgboost_rs && \
  BENCH_ROWS=4194304 BENCH_FEATURES=64 BENCH_BINS=256 BENCH_ITERS=50 \
  ~/.cargo/bin/cargo run --release --features cuda --bin bench
```

- `BENCH_ROWS` (1048576), `BENCH_FEATURES` (32), `BENCH_BINS` per feature
  (256), `BENCH_ITERS` timed launches per case (20).
- Memory: the ELLPACK matrix is `4 * ROWS * FEATURES` bytes on device.

## Notes for interpreting numbers

- `Gentry/s` = matrix entries visited per second (`rows * features / time`);
  XGBoost's own gpu_hist throughput scales the same way.
- `dense/shared` vs `dense/global` isolates the shared-memory privatised
  histogram against pure global atomics — on CUDA hardware expect the shared
  path to win clearly at 256 bins/feature.
- Timing excludes data upload/quantisation (device-resident inputs, one sync
  per timed batch), so it measures the kernel itself.

## Sanity check without a GPU

The same binary runs on any Vulkan device (including CPU lavapipe) when built
without `--features cuda`:

```bash
cargo run --release --bin bench
```
