#!/usr/bin/env python3
"""xgboost_rs `device=cuda` against XGBoost `gpu_hist`, on a Kaggle T4.

The number this crate's GPU path has to beat is XGBoost's own, on the same
card, the same data and the same configuration — `tools/compare_gpu.py`'s
table, which `docs/gpu-benchmarks.md` records. This kernel produces that
table, and around it everything needed to act on it:

  1. the CUDA kernel oracle and the two GPU test suites, so a speed number is
     never reported for a build that gives a different answer;
  2. the head-to-head table (`compare.txt`);
  3. per-phase wall clock of a device fit (`XGB_PHASES`) and CubeCL's own
     per-kernel profile, on the baseline case and the two shapes that
     stress it (deep, and many rows), which is what says *where* a gap is;
  4. a sweep over the histogram launch shape (`XGB_GPU_TUNE`), so the next
     change is chosen from a measurement rather than a guess.

Binaries come from `prebuilt/` when `tools/kaggle/cross-build.sh` put them in
the upload, and are built here otherwise. Everything lands in /kaggle/working.
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

WORK = pathlib.Path("/kaggle/working")
SRC = WORK / "xgboost_rs"
CARGO = os.path.expanduser("~/.cargo/bin/cargo")
SUMMARY: list = []


def run(cmd, cwd=None, timeout=3600, env=None, log=None, tail=6000):
    print(f"\n$ {cmd}", flush=True)
    full_env = {**os.environ, **(env or {})}
    t0 = time.time()
    try:
        p = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True,
                           timeout=timeout, env=full_env)
        out = (p.stdout or "") + (p.stderr or "")
        rc = p.returncode
    except subprocess.TimeoutExpired as exc:
        out = f"TIMEOUT after {timeout}s\n" + (exc.stdout or "") + (exc.stderr or "")
        rc = -1
    dt = time.time() - t0
    print(out[-tail:], flush=True)
    print(f"[{dt:.0f}s, rc={rc}]", flush=True)
    if log:
        (WORK / log).write_text(out)
    SUMMARY.append((log or cmd[:60], rc, round(dt, 1)))
    return rc, out


def section(name):
    print("\n" + "=" * 70 + f"\n== {name}\n" + "=" * 70, flush=True)


# ---------------------------------------------------------------- setup ----
section("environment")
run("nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv")
run("nproc && free -g | head -2 && grep -m1 'model name' /proc/cpuinfo")
# Kaggle VMs differ run to run by the better part of 2x on host-side work
# (run #3 of 2026-09-05 measured every host phase at 1.7x run #2's), so a
# fixed host loop is timed here and every host-side number below is read
# against it. Ratios within one run are the comparison; absolutes are not.
run(f"{sys.executable} -c \"import time; t=time.perf_counter(); s=0\nfor i in range(20_000_000): s+=i*i\nprint('host canary: %.3fs for 2e7 iterations' % (time.perf_counter()-t))\"")

found = subprocess.run("find /kaggle/input -maxdepth 6 -name Cargo.toml -print -quit",
                       shell=True, capture_output=True, text=True).stdout.strip()
if not found:
    raise SystemExit("no Cargo.toml under /kaggle/input — is the dataset attached?")
crate_in = pathlib.Path(found).parent
print(f"crate found at {crate_in}", flush=True)
run(f"mkdir -p {SRC} && cp -r {crate_in}/. {SRC}/ && ls {SRC}")

# ------------------------------------------------------------ binaries ----
section("binaries")
PRE = SRC / "prebuilt"
BINS = {}
if PRE.is_dir() and (PRE / "train_bench").exists():
    rev = (PRE / "GIT_REV").read_text().strip() if (PRE / "GIT_REV").exists() else "?"
    stamp = (PRE / "BUILD_STAMP").read_text().strip() if (PRE / "BUILD_STAMP").exists() else "?"
    print(f"using prebuilt binaries ({rev}, built {stamp})", flush=True)
    for name in ("train_bench", "bench", "kernels", "gpu_training"):
        p = PRE / name
        if p.exists():
            p.chmod(0o755)
            BINS[name] = p
else:
    run("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
        "sh -s -- -y --profile minimal", timeout=1800)
    rc, _ = run(f"{CARGO} build --release --no-default-features --features cuda "
                "--bin train_bench --bin bench", cwd=SRC, timeout=3600, log="build.txt")
    if rc != 0:
        raise SystemExit("build failed")
    for name in ("train_bench", "bench"):
        BINS[name] = SRC / "target" / "release" / name
    rc, out = run(f"{CARGO} test --release --no-default-features --features cuda "
                  "--test kernels --test gpu_training --no-run --message-format=json",
                  cwd=SRC, timeout=3600, log="build_tests.txt", tail=500)
    for line in out.splitlines():
        if line.startswith("{"):
            try:
                m = json.loads(line)
            except ValueError:
                continue
            if m.get("reason") == "compiler-artifact" and m.get("executable") \
                    and m["target"]["kind"] == ["test"]:
                BINS[m["target"]["name"]] = pathlib.Path(m["executable"])
print("binaries:", {k: str(v) for k, v in BINS.items()}, flush=True)

# ------------------------------------------------------------- xgboost ----
section("xgboost")
rc, out = run(f"{sys.executable} -c 'import xgboost; print(xgboost.__version__)'")
if rc != 0 or out.strip().splitlines()[-1] != "3.4.0":
    run(f"{sys.executable} -m pip install -q 'xgboost==3.4.0'", timeout=1800)
run(f"{sys.executable} -c 'import xgboost; print(xgboost.__version__)'")

# ------------------------------------------------- 1. correctness first ----
section("0a. the driver's JIT cache")
# The runtime loads PTX, which the driver compiles to SASS per process unless
# its own cache (`~/.nv/ComputeCache`, `CUDA_CACHE_*`) holds it.
run("env | grep -i cuda_cache; echo HOME=$HOME; ls -la ~/.nv/ComputeCache 2>&1 | head -5; du -sh ~/.nv/ComputeCache 2>&1")

section("0. host-to-device bandwidth, independent of CubeCL")
# `bench` measures uploads through CubeCL at ~0.37 GB/s on this VM; a copy
# that owes nothing to the runtime says whether that is the VM.
run(f"{sys.executable} tools/kaggle/h2d_probe.py", cwd=SRC, timeout=600, log="torch_bandwidth.txt")

section("1. CUDA oracle + kernel and training test suites")
run(f"{BINS['bench']}", cwd=SRC, timeout=1800, log="bench.txt")
# The kernel oracle and throughput of the histogram's contiguous-items form.
run(f"{BINS['bench']}", cwd=SRC, timeout=1800, log="bench_contig.txt",
    env={"XGB_GPU_TUNE": "hist_contig=1", "BENCH_ITERS": "10"})
# The native-i64 shared accumulator sums wrongly on CUDA (see
# `HistogramBuilder`); this dumps the CUDA source it compiles to, so the
# reason can be read rather than guessed. The oracle fails here by design.
run(f"{BINS['bench']} || true", cwd=SRC, timeout=1800, log="bench_smem64.txt", tail=1200,
    env={"XGB_GPU_TUNE": "hist_smem64=1", "BENCH_ITERS": "10"})
# Where the histogram kernel's time goes: without its atomics, and without
# its bin decode as well (the oracle fails there by design; the speed table
# is the point).
for probe in ("1", "2"):
    run(f"{BINS['bench']} || true", cwd=SRC, timeout=1800, log=f"bench_probe{probe}.txt",
        env={"XGB_GPU_TUNE": f"hist_probe={probe}", "BENCH_ITERS": "10", "BENCH_SKIP_ORACLE": "1"})
for name in ("kernels", "gpu_training"):
    if name in BINS:
        run(f"{BINS[name]} --test-threads=1", cwd=SRC, timeout=1800, log=f"test_{name}.txt")
# An illegal address in one kernel poisons the CUDA context for every launch
# after it, so a failure above names the first victim, not the culprit. The
# sanitizer names the kernel and the access; it is slow, so only the device
# binning/sketch tests and one small fit run under it.
SAN = shutil.which("compute-sanitizer") or "/usr/local/cuda/bin/compute-sanitizer"
DUMP = {"CUBECL_DEBUG_LOG": str(WORK / "cuda_kernels.log"), "CUBECL_DEBUG_OPTION": "debug"}
if pathlib.Path(SAN).exists() and "kernels" in BINS:
    run(f"{SAN} --print-limit 3 {BINS['kernels']} --test-threads=1 device_", cwd=SRC,
        timeout=1800, log="sanitizer_kernels.txt", tail=5000, env=DUMP)
    run(f"{SAN} --print-limit 3 {BINS['train_bench']} --rows 3000 --features 5 --rounds 1 "
        "--depth 3 --device cuda", cwd=SRC, timeout=1800, log="sanitizer_train.txt", tail=5000)
else:
    print("no compute-sanitizer; dumping the kernel sources instead", flush=True)
    run(f"{BINS['kernels']} --test-threads=1 device_", cwd=SRC, timeout=1800,
        log="device_tests.txt", tail=3000, env=DUMP)

# ----------------------------------------------------- 2. head to head ----
section("2. xgboost_rs device=cuda vs XGBoost gpu_hist")
BASE = "--rows 500000 --features 50 --rounds 20 --depth 6 --max-bin 256"
# The first process on a machine compiles every kernel through NVRTC and
# fills the on-disk PTX cache; every later process loads it. Both are worth
# a number — the cold fit is what a one-off script pays, the warm one is the
# like-for-like against XGBoost's precompiled kernels — and the table below
# is measured warm, as XGBoost is.
run(f"{BINS['train_bench']} {BASE} --device cuda", cwd=SRC, timeout=1800,
    env={"XGB_NO_KERNEL_CACHE": "1"}, log="cold_jit.txt", tail=400)
run(f"{BINS['train_bench']} {BASE} --device cuda", cwd=SRC, timeout=1800,
    log="warm_first.txt", tail=400)
run(f"{BINS['train_bench']} {BASE} --device cuda", cwd=SRC, timeout=1800,
    log="warm_second.txt", tail=400)
# The primary context built while the data is generated (`--prewarm`),
# which is the accounting a Python process gets from `import`.
run(f"{BINS['train_bench']} {BASE} --device cuda --prewarm", cwd=SRC, timeout=1800,
    log="warm_first_prewarm.txt", tail=400)
run(f"{sys.executable} tools/compare_gpu.py --binary {BINS['train_bench']}",
    cwd=SRC, timeout=3600, log="compare.txt", tail=12000)
run(f"{sys.executable} tools/compare_gpu.py --prewarm --binary {BINS['train_bench']}",
    cwd=SRC, timeout=3600, log="compare_prewarm.txt", tail=12000)
# Both sides warm: the CUDA primary context alone is ~350 ms on this VM and
# XGBoost's kernel modules load lazily too, so this is the table for a
# process that fits more than once.
run(f"{sys.executable} tools/compare_gpu.py --warm --binary {BINS['train_bench']}",
    cwd=SRC, timeout=3600, log="compare_warm.txt", tail=12000)

# --------------------------------------------------------- 3. profiles ----
section("2b. repeat stability")
# The 20-feature case once fitted to a different RMSE after a warm-up fit in
# the same process (run #22, 0.04400 against 0.04228); the same fit
# repeated in one process must be bit-stable.
# Run #23 reproduced it: 3 of 8 repeats differed with 20 features, none
# with 50. The cause was the direct upload returning before its DMA landed;
# the no-direct-upload run is the control.
for feats, reps, env in (("20", 16, ""), ("50", 8, ""), ("20", 16, "XGB_NO_DIRECT_UPLOAD=1 ")):
    tag = "_nodirect" if env else ""
    run(f"{env}{BINS['train_bench']} --rows 500000 --features {feats} --rounds 20 --depth 6 "
        f"--max-bin 256 --device cuda --repeats {reps}", cwd=SRC, timeout=1800,
        log=f"repeats_f{feats}{tag}.txt", tail=2000)

section("3. where the time goes")
SHAPES = {
    "baseline": BASE,
    "depth10": "--rows 500000 --features 50 --rounds 20 --depth 10 --max-bin 256",
    "rows1m": "--rows 1000000 --features 50 --rounds 20 --depth 6 --max-bin 256",
    "lossguide": "--rows 500000 --features 50 --rounds 20 --depth 0 --max-leaves 64 --lossguide --max-bin 256",
    "sparse": "--rows 500000 --features 50 --rounds 20 --depth 6 --max-bin 256 --sparsity 0.3",
}
for label, args in SHAPES.items():
    run(f"{BINS['train_bench']} {args} --device cuda --breakdown", cwd=SRC,
        timeout=1800, log=f"breakdown_{label}.txt")
    rc, out = run(f"{BINS['train_bench']} {args} --device cuda", cwd=SRC, timeout=1800,
                  env={"XGB_PHASES": "1"}, log=f"phases_{label}.txt", tail=2500)
    if label == "baseline":
        # The same, for the second fit of a process whose context was built
        # before the data: what a process pays once is the difference.
        run(f"{BINS['train_bench']} {args} --device cuda --prewarm --warmup", cwd=SRC,
            timeout=1800, env={"XGB_PHASES": "1"}, log="phases_warm.txt", tail=2500)
    prof = WORK / f"cubecl_profile_{label}.log"
    run(f"{BINS['train_bench']} {args} --device cuda", cwd=SRC, timeout=1800,
        env={"CUBECL_DEBUG_LOG": str(prof), "CUBECL_DEBUG_OPTION": "profile"},
        log=f"profile_run_{label}.txt", tail=800)
    if prof.exists():
        text = prof.read_text()
        print(text[-6000:], flush=True)

# ------------------------------------------------- 4. launch-shape sweep ----
section("4. histogram launch shape sweep (XGB_GPU_TUNE)")
VARIANTS = ["", "hist_smem64=1", "hist_global=1", "hist_smem64=1,hist_bps=2", "hist_block=768"]
rows = []
for v in VARIANTS:
    rc, out = run(f"{BINS['train_bench']} {BASE} --device cuda --repeats 2", cwd=SRC,
                  timeout=1800, env={"XGB_GPU_TUNE": v}, tail=300)
    t = [l for l in out.splitlines() if l.startswith("train:")]
    rows.append((v or "(default)", t[0] if t else f"rc={rc}"))
table = "\n".join(f"{v:<20} {t}" for v, t in rows)
print(table, flush=True)
(WORK / "tune_sweep.txt").write_text(table + "\n")

section("the driver's JIT cache, after the runs")
run("du -sh ~/.nv/ComputeCache 2>&1; ls ~/.nv/ComputeCache 2>&1 | head -3")

section("summary")
for name, rc, dt in SUMMARY:
    print(f"  {'ok ' if rc == 0 else 'FAIL'} {name:40} {dt:>7.1f}s rc={rc}")
