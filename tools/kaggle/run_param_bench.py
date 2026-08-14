#!/usr/bin/env python3
"""The parameter performance sweep, on a real CUDA GPU.

`docs/parameter-performance.md` compares what every performance-affecting
parameter costs on the CPU against what it costs on the device. Locally the
device half runs on lavapipe — a software Vulkan implementation on the same
CPU — which says what the device *path* does but nothing about GPU speed. This
runs the identical sweep on Kaggle hardware so that column is real.

Two configurations, small first so a timeout still leaves something:

  1. 50k x 20, 5 rounds — the size the committed lavapipe table uses, so the
     two are directly comparable row for row.
  2. 200k x 40, 10 rounds — the size the CPU table uses, and large enough that
     a GPU has something to do. At the small size a T4 is mostly launch
     overhead, so a device column measured only there would understate it.

Both are run on both devices *on the same VM*, back to back. That matters: a
Kaggle CPU is not this laptop's, so only a same-VM ratio means anything.

Everything lands in /kaggle/working as .txt beside the console log.
"""

import os
import pathlib
import subprocess
import time

WORK = pathlib.Path("/kaggle/working")
SRC = pathlib.Path("/kaggle/working/xgboost_rs")
CARGO = os.path.expanduser("~/.cargo/bin/cargo")
BIN = SRC / "target" / "release" / "param_bench"


def run(cmd, cwd=None, timeout=3600, log=None):
    print(f"\n$ {cmd}", flush=True)
    t0 = time.time()
    try:
        p = subprocess.run(
            cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
        out = (p.stdout or "") + (p.stderr or "")
        rc = p.returncode
    except subprocess.TimeoutExpired as exc:
        out = f"TIMEOUT after {timeout}s\n" + (exc.stdout or "") + (exc.stderr or "")
        rc = -1
    print(out[-12000:], flush=True)
    print(f"[{time.time() - t0:.0f}s, rc={rc}]", flush=True)
    if log:
        (WORK / log).write_text(out)
    return rc


def section(name):
    print("\n" + "=" * 70 + f"\n== {name}\n" + "=" * 70, flush=True)


section("environment")
run("nvidia-smi")
run("nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader")
run("nproc && free -g | head -2 && cat /proc/cpuinfo | grep -m1 'model name'")

# Kaggle expands an uploaded archive on ingest and does not always mount it at
# /kaggle/input/<slug>, so find the crate rather than assuming either.
found = subprocess.run(
    "find /kaggle/input -maxdepth 5 -name Cargo.toml -print -quit",
    shell=True, capture_output=True, text=True,
).stdout.strip()
if not found:
    raise SystemExit("no Cargo.toml under /kaggle/input — is the dataset attached?")
print(f"crate found at {pathlib.Path(found).parent}", flush=True)
# /kaggle/input is read-only and cargo needs a writable target/.
run(f"mkdir -p {SRC} && cp -r {pathlib.Path(found).parent}/. {SRC}/ && ls {SRC}")

section("install rust")
run(
    "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
    "sh -s -- -y --profile minimal --default-toolchain nightly",
    timeout=1800,
)

section("build param_bench --features cuda")
# `cuda` adds the CUDA backend to the Vulkan one rather than replacing it, so
# this builds both. It is the slowest step here by far.
if run(f"{CARGO} build --release --features cuda --bin param_bench",
       cwd=SRC, timeout=3600, log="build.txt") != 0:
    raise SystemExit("build failed — nothing to measure")

CONFIGS = [
    ("small", "--rows 50000 --features 20 --rounds 5 --repeats 3"),
    ("large", "--rows 200000 --features 40 --rounds 10 --repeats 2"),
]

for label, args in CONFIGS:
    for device in ("cpu", "cuda"):
        section(f"sweep: {label}, device={device}")
        run(
            f"{BIN} {args} --threads 0 --device {device}",
            cwd=SRC, timeout=7200, log=f"sweep_{label}_{device}.txt",
        )

section("done")
run(f"ls -la {WORK}")
