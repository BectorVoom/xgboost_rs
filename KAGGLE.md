# Running the oracle + speed benchmark on a Kaggle CUDA GPU

## Scripted (preferred)

`tools/kaggle/push.sh [--wait] [metadata.json]` packages the crate, uploads
it as a dataset, waits for ingest, and pushes a kernel. It needs an
authenticated `kaggle` CLI. Three kernels exist, named by their metadata file:

| metadata | script | what it measures |
|---|---|---|
| `kernel-metadata.json` (default) | `run.py` | the kernel oracle, the GPU-gated fixtures, XGBoost CPU vs GPU |
| `param-bench-metadata.json` | `run_param_bench.py` | the parameter sweep on both devices |
| `gpu-compare-metadata.json` | `run_gpu_compare.py` | **`device=cuda` against XGBoost `gpu_hist`** (`docs/gpu-benchmarks.md`), with the CUDA test suites, per-phase timers, CubeCL's per-kernel profile, a transfer micro-benchmark and a launch-shape sweep |

### Skip the build: cross-compile here

A Kaggle session rebuilds the crate from source in about three and a half
minutes, plus rustup. `tools/kaggle/cross-build.sh` builds the same binaries —
`train_bench`, `bench`, and the `kernels` / `gpu_training` test binaries —
for Linux x86_64 on this machine in about ninety seconds and puts them in
`prebuilt/`, which `push.sh` ships and `run_gpu_compare.py` uses instead of
building. A change-measure loop on the T4 is then ~12 minutes end to end.

It needs `rustup target add x86_64-unknown-linux-gnu`, `cargo install
cargo-zigbuild`, and a `zig` on PATH (`pip install ziglang` provides one;
pass `ZIG=/path/to/zig`). No CUDA toolkit: `cudarc` loads the driver at run
time, and is told the version to bind (`CUDARC_CUDA_VERSION=13000`, Kaggle's
driver) rather than asked to find one.

Two things the first runs taught, both of which look like something else:

* `cubecl-cuda` 0.10 does not register `f64` as a supported scalar type (a
  matmul workaround of its own), so a type-table check refuses every CUDA
  fit with `NoF64Support` — and the training tests then pass by skipping.
  `gpu::supports_f64` answers CUDA by name for that reason.
* Kaggle VMs differ by up to ~1.7x on host-side work from one session to
  the next. `run_gpu_compare.py` prints a host canary; compare within a run.
* `kernels status` right after a push still reports the *previous* run's
  COMPLETE, and `datasets status` says ready for the previous version while
  the new one ingests; `push.sh` waits for both to move. A session that ran
  the wrong binaries prints an older `built` stamp.
* The head-to-head is measured three ways: cold (one fit per fresh process),
  cold with `--prewarm` (the crate's binary starts the driver's context
  creation on a thread before it generates its data) and warm
  (`compare_gpu.py --warm`, both sides fit once before the clock). The CUDA
  primary context alone is ~350 ms on these VMs, and XGBoost's Python
  process has paid it at import, before its timer, which `--prewarm` is the
  counterpart of.
* Read the phase log (`XGB_PHASES=1`, `phases_*.txt`) next to the per-kernel
  profile (`cubecl_profile_*.log`): where the two disagree is host work,
  an upload or a sync, never a kernel. The tree fit cannot be profiled on
  Metal (no `f64`), so per-phase work goes through the T4; both kernel
  slugs (`gpu-compare-metadata.json`, `-b`) can run at once.

**Ask for a T4, not the default.** Kaggle hands out a **P100 (SM 6.0)** unless
told otherwise, and XGBoost's own 3.x wheels are built for SM70+ — on a P100
every XGBoost CUDA path fails with `This program was not compiled for SM 60`,
so there is no `gpu_hist` baseline and the four GPU-gated oracle cases cannot
be pinned. The CubeCL kernels are unaffected (NVRTC compiles for the live
device), which is why the kernel oracle passes on a P100 that XGBoost refuses.

`enable_gpu` is deprecated and cannot choose a type. `machine_shape` can:

```json
{ "enable_gpu": true, "machine_shape": "NvidiaTeslaT4" }
```

Accepted values are `NvidiaTeslaT4`, `NvidiaTeslaP100` and `Tpu1VmV38`. This is
in `tools/kaggle/kernel-metadata.json`.

Two limits worth knowing before you queue anything, because both fail at push
time with a message that looks like a bug in the config:

* **two concurrent batch GPU sessions** per account;
* **30 GPU-hours per week**, and `kernels push` refuses outright once that is
  spent. Check it before packaging:

```python
from kagglesdk import KaggleClient
from kagglesdk.kernels.types.kernels_api_service import ApiGetAcceleratorQuotaStatisticsRequest
with KaggleClient() as c:
    q = c.kernels.kernels_api_client.get_accelerator_quota_statistics(
        ApiGetAcceleratorQuotaStatisticsRequest()).gpu_quota
    print(q.time_used, "of", q.total_time_allowed)
```

## Manual

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
  histogram against pure global atomics.

## Measured on a Kaggle Tesla P100 (SM 6.0), CUDA 13.0

Oracle: **ALL PASS** — all ten kernel cases at exact `i64` equality against the
CPU reference, with `native i64 atomics: supported`.

| case | 1M x 32 | 4.2M x 64 |
|---|---:|---:|
| `dense/shared`        |  8.72 Gentry/s |  9.89 Gentry/s |
| `dense/global/u32`    |  6.43 |  6.48 |
| `dense/global/i64`    | **13.95** | **12.43** |
| `sparse0.5/shared`    | **21.21** | **20.83** |
| `sparse0.5/global/u32`| 12.09 | 12.09 |
| `sparse0.5/global/i64`| 19.48 | 20.22 |

**This contradicts the expectation stated above**, which used to read "on CUDA
hardware expect the shared path to win clearly at 256 bins/feature". On dense
input it does not: native 64-bit global atomics beat the privatised shared
histogram by **1.6x** at 1M x 32 and 1.26x at 4.2M x 64. Shared only wins on
the sparse case.

That matters because `HistogramBuilder::build` picks the shared path whenever a
group's bins fit in the shared-memory budget, so on this hardware the dense
path auto-selects the slower of the two. Worth making the choice measured
rather than assumed before any of this is wired into training.

## What a P100 session cannot measure

XGBoost's own 3.x wheels are compiled for SM70 and up, so on a P100 every CUDA
path in XGBoost fails outright with `This program was not compiled for SM 60`.
Two consequences:

- there is no XGBoost `gpu_hist` baseline from a P100 session (the CPU numbers
  still come out: 200k x 32 in 1.19s, 1M x 32 in 4.28s, 1M x 64 in 7.93s, 20
  rounds at depth 8);
- the four GPU-gated oracle cases — `device=cuda`,
  `sampling_method=gradient_based`, `updater=grow_gpu_hist` and
  `grow_gpu_approx` — still cannot be pinned, and `tools/gen_string_param_fixtures.py`
  correctly reports `cuda device usable: False` there.

Those need an SM70+ accelerator (T4, L4 or V100). The CubeCL kernels are
unaffected either way: they compile through NVRTC for whatever device is
present, which is why the oracle above passes on the same P100 that XGBoost
refuses to run on.
- Timing excludes data upload/quantisation (device-resident inputs, one sync
  per timed batch), so it measures the kernel itself.

## Sanity check without a GPU

The same binary runs without a GPU at all. The default build puts the kernels
on the CubeCL CPU runtime, which needs no toolchain:

```bash
cargo run --release --bin bench          # CubeCL CPU runtime
cargo run --release --features vulkan --bin bench   # any Vulkan device, incl. lavapipe
```

It prints which backend it resolved to, so there is never a question of what a
number was measured on.
