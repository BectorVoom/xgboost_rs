#!/usr/bin/env python3
"""xgboost_rs on a Kaggle CUDA GPU.

Three jobs, each independent so a failure in one still reports the others:

  1. Build the crate with --features cuda and run the CubeCL kernel oracle +
     speed benchmark. This is the only place the GPU kernels meet real CUDA
     (native i64 atomics, NVRTC compilation); locally they only ever run on
     lavapipe.
  2. Regenerate the string-parameter oracle fixtures with a GPU actually
     present, which is the only way to pin the four GPU-gated cases —
     device=cuda, sampling_method=gradient_based, updater=grow_gpu_hist and
     updater=grow_gpu_approx. On a CPU box device=cuda silently trains on the
     CPU, so those fixtures cannot be generated anywhere else.
  3. Time XGBoost's own hist (CPU) against gpu_hist on the same data, which is
     the number the Rust GPU path has to beat.

Everything lands in /kaggle/working: logs as .txt, results as .json, and the
regenerated fixtures as a tarball.
"""

import json
import os
import pathlib
import subprocess
import sys
import time

WORK = pathlib.Path("/kaggle/working")
SRC = pathlib.Path("/kaggle/working/xgboost_rs")
CARGO = os.path.expanduser("~/.cargo/bin/cargo")
RESULTS: dict = {}


def run(cmd, cwd=None, timeout=3600, env=None, log=None):
    """Run a command, stream-capture it, and record the outcome."""
    print(f"\n$ {cmd}", flush=True)
    full_env = {**os.environ, **(env or {})}
    t0 = time.time()
    p = subprocess.run(
        cmd, shell=True, cwd=cwd, capture_output=True, text=True,
        timeout=timeout, env=full_env,
    )
    out = (p.stdout or "") + (p.stderr or "")
    print(out[-8000:], flush=True)
    if log:
        (WORK / log).write_text(out)
    return {"cmd": cmd, "returncode": p.returncode,
            "seconds": round(time.time() - t0, 1), "tail": out[-4000:]}


def section(name):
    print("\n" + "=" * 70 + f"\n== {name}\n" + "=" * 70, flush=True)


# ---------------------------------------------------------------- setup ----
section("environment")
RESULTS["nvidia_smi"] = run("nvidia-smi")
RESULTS["input_listing"] = run("ls -R /kaggle/input | head -40")

# Two things about /kaggle/input that a hardcoded path gets wrong:
#   * Kaggle expands an uploaded archive on ingest, so the dataset holds the
#     crate tree directly, not the tarball that was pushed;
#   * the mount point is not always /kaggle/input/<slug> — this dataset lands
#     under /kaggle/input/datasets/<owner>/<slug>.
# Find the crate root by looking for its Cargo.toml instead of assuming either.
found = subprocess.run(
    "find /kaggle/input -maxdepth 5 -name Cargo.toml -print -quit",
    shell=True, capture_output=True, text=True,
).stdout.strip()
if not found:
    raise SystemExit("no Cargo.toml anywhere under /kaggle/input — is the dataset attached?")
CRATE_IN = pathlib.Path(found).parent
print(f"crate found at {CRATE_IN}", flush=True)
# Copy out: /kaggle/input is read-only and cargo needs a writable target/.
RESULTS["unpack"] = run(f"mkdir -p {SRC} && cp -r {CRATE_IN}/. {SRC}/ && ls {SRC}")

# The GPU decides what can be measured. XGBoost's 3.x wheels are built for
# SM70+, so on Kaggle's P100 (SM 6.0) its CUDA path cannot run at all — while
# the CubeCL kernels compile through NVRTC for whatever device is present and
# are unaffected.
CC = run("nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader")
RESULTS["gpu_query"] = CC
gpu_name = CC["tail"].strip()
try:
    cap = float(gpu_name.split(",")[-1].strip())
except (ValueError, IndexError):
    cap = 0.0
XGB_GPU_SUPPORTED = cap >= 7.0
print(f"gpu: {gpu_name!r}  compute capability {cap}  "
      f"xgboost CUDA usable: {XGB_GPU_SUPPORTED}", flush=True)
RESULTS["xgb_gpu_supported"] = XGB_GPU_SUPPORTED

section("install rust")
RESULTS["rustup"] = run(
    "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
    "sh -s -- -y --profile minimal --default-toolchain nightly",
    timeout=1800,
)

# ------------------------------------------------- 1. CubeCL CUDA oracle ----
section("1. CubeCL kernel oracle + speed on CUDA")
RESULTS["cuda_bench"] = run(
    f"{CARGO} run --release --features cuda --bin bench",
    cwd=SRC, timeout=3600, log="cuda_bench.txt",
)

# A larger sweep, only if the first one worked.
if RESULTS["cuda_bench"]["returncode"] == 0:
    RESULTS["cuda_bench_large"] = run(
        f"{CARGO} run --release --features cuda --bin bench",
        cwd=SRC, timeout=3600, log="cuda_bench_large.txt",
        env={"BENCH_ROWS": "4194304", "BENCH_FEATURES": "64",
             "BENCH_BINS": "256", "BENCH_ITERS": "50"},
    )

# ------------------------------------- 2. GPU-gated oracle fixture regen ----
section("2. regenerate string-parameter fixtures with a real GPU")
RESULTS["pip_xgb"] = run(f"{sys.executable} -m pip install -q 'xgboost==3.4.0'", timeout=1800)
RESULTS["fixture_regen"] = run(
    f"{sys.executable} tools/gen_string_param_fixtures.py",
    cwd=SRC, timeout=3600, log="fixture_regen.txt",
)
RESULTS["fixture_tar"] = run(
    f"cd {SRC} && tar czf {WORK}/strparam_gpu_fixtures.tar.gz tests/fixtures/strparam"
)
# Surface the GPU-gated cases directly in the results, so they are readable
# without unpacking the tarball.
gpu_cases = {}
for name in ["index", "device_cuda", "sampling_method_gradient_based",
             "updater_grow_gpu_hist", "updater_grow_gpu_approx"]:
    p = SRC / "tests" / "fixtures" / "strparam" / f"{name}.json"
    if p.exists():
        d = json.loads(p.read_text())
        if name == "index":
            gpu_cases[name] = {"gpu_present": d.get("gpu_present"),
                               "n_cases": len(d.get("cases", []))}
        else:
            gpu_cases[name] = {
                k: d.get(k) for k in
                ("outcome", "error", "gpu_present", "requires_gpu",
                 "base_score", "metric", "metric_history", "objective_name")
            }
RESULTS["gpu_gated_cases"] = gpu_cases

# ------------------------------------------- 3. XGBoost CPU vs GPU timing ----
section("3. XGBoost hist (CPU) vs gpu_hist — the target to beat")
timing_script = r'''
import json, os, time
import numpy as np, xgboost as xgb

GPU_OK = os.environ.get("XGB_GPU_SUPPORTED") == "1"
rng = np.random.default_rng(0)
out = {"xgboost_version": xgb.__version__, "gpu_usable": GPU_OK, "runs": []}
devices = ["cpu", "cuda"] if GPU_OK else ["cpu"]
if not GPU_OK:
    print("skipping the cuda runs: this GPU is below the SM70 XGBoost wheels need",
          flush=True)

for n_rows, n_feat in [(200_000, 32), (1_000_000, 32), (1_000_000, 64)]:
    X = rng.normal(size=(n_rows, n_feat)).astype(np.float32)
    y = (X[:, 0] * 2.0 - X[:, 3] + rng.normal(scale=0.1, size=n_rows)).astype(np.float32)
    for device in devices:
        params = {"objective": "reg:squarederror", "tree_method": "hist",
                  "device": device, "max_depth": 8, "max_bin": 256,
                  "eta": 0.3, "verbosity": 0, "seed": 0}
        d = xgb.DMatrix(X, label=y)
        t0 = time.perf_counter()
        bst = xgb.train(params, d, num_boost_round=20)
        t = time.perf_counter() - t0
        pred = bst.predict(d)
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        out["runs"].append({"rows": n_rows, "features": n_feat, "device": device,
                            "seconds": round(t, 3), "rmse": rmse})
        print(f"{device:5} rows={n_rows:>9} feat={n_feat:>3}  {t:7.3f}s  rmse={rmse:.6f}", flush=True)

json.dump(out, open("/kaggle/working/xgboost_timing.json", "w"), indent=1)
'''
(WORK / "xgb_timing.py").write_text(timing_script)
RESULTS["xgboost_timing"] = run(
    f"{sys.executable} {WORK}/xgb_timing.py", timeout=3600, log="xgboost_timing.txt",
    env={"XGB_GPU_SUPPORTED": "1" if XGB_GPU_SUPPORTED else "0"},
)
p = WORK / "xgboost_timing.json"
if p.exists():
    RESULTS["xgboost_timing_json"] = json.loads(p.read_text())

# --------------------------------------------------------------- report ----
section("summary")
for k, v in RESULTS.items():
    if isinstance(v, dict) and "returncode" in v:
        status = "ok " if v["returncode"] == 0 else "FAIL"
        print(f"  {status} {k:24} {v['seconds']:>8.1f}s  rc={v['returncode']}")
(WORK / "results.json").write_text(json.dumps(RESULTS, indent=1, default=str))
print("\nwrote /kaggle/working/results.json")
