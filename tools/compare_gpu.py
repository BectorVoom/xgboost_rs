#!/usr/bin/env python3
"""Head-to-head: this crate against the reference XGBoost, on both devices.

The Rust benchmark writes the generated dataset once and the Python harness
reads the same bytes, so a row of the table compares two implementations on
identical data and an identical configuration.

    python tools/compare_gpu.py --binary target/release/train_bench

Every case is timed on `device=cuda` for both implementations; pass `--cpu` to
time `device=cpu` as well, which is what says whether a GPU win is real or just
a slow CPU baseline.

Each side trains exactly once per case by default. Repeating the fit would let
XGBoost reuse the binned matrix it caches on the DMatrix while this crate
rebuilt it, which is not a like-for-like comparison.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile

# name, rows, features, rounds, depth, max_leaves, max_bin, sparsity, policy
CASES = [
    ("baseline",      500_000,  50, 20,  6,  0, 256, 0.0, "depthwise"),
    ("rows 100k",     100_000,  50, 20,  6,  0, 256, 0.0, "depthwise"),
    ("rows 1M",     1_000_000,  50, 20,  6,  0, 256, 0.0, "depthwise"),
    ("features 20",   500_000,  20, 20,  6,  0, 256, 0.0, "depthwise"),
    ("features 200",  200_000, 200, 20,  6,  0, 256, 0.0, "depthwise"),
    ("depth 4",       500_000,  50, 20,  4,  0, 256, 0.0, "depthwise"),
    ("depth 10",      500_000,  50, 20, 10,  0, 256, 0.0, "depthwise"),
    ("max_bin 64",    500_000,  50, 20,  6,  0,  64, 0.0, "depthwise"),
    ("max_bin 512",   500_000,  50, 20,  6,  0, 512, 0.0, "depthwise"),
    ("sparse 0.3",    500_000,  50, 20,  6,  0, 256, 0.3, "depthwise"),
    ("lossguide 64",  500_000,  50, 20,  0, 64, 256, 0.0, "lossguide"),
    ("rounds 100",    200_000,  50, 100, 6,  0, 256, 0.0, "depthwise"),
]

TRAIN_RE = re.compile(r"^train:\s+([0-9.]+)s\s+final train-rmse:\s+([0-9.eE+-]+)", re.M)


def parse(out: str) -> tuple[float, float]:
    m = TRAIN_RE.search(out)
    if not m:
        raise SystemExit(f"could not parse timing from:\n{out}")
    return float(m.group(1)), float(m.group(2))


def run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"failed: {' '.join(cmd)}\n{p.stdout}\n{p.stderr}")
    return p.stdout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="target/release/train_bench")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--bench-xgb", default=os.path.join(os.path.dirname(__file__), "bench_xgb.py"))
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--cpu", action="store_true", help="also time both on device=cpu")
    ap.add_argument("--only", default=None, help="substring filter over case names")
    ap.add_argument(
        "--warm",
        action="store_true",
        help="both sides fit once before the clock, so a process's one-time costs "
        "(the CUDA context, kernel modules) are outside the timing",
    )
    ap.add_argument(
        "--prewarm",
        action="store_true",
        help="the Rust side starts the device's initialisation before its data "
        "build (`train_bench --prewarm`), which is when a Python process pays it",
    )
    args = ap.parse_args()

    tmp = tempfile.mkdtemp()
    data = os.path.join(tmp, "d.bin")

    header = f"{'case':<16} {'rows':>9} {'feat':>5} {'rnd':>4} "
    header += f"{'rs-gpu':>9} {'xgb-gpu':>9} {'speedup':>8}"
    if args.cpu:
        header += f" {'rs-cpu':>9} {'xgb-cpu':>9}"
    header += "  rmse"
    print(header)
    print("-" * len(header))

    for name, rows, feats, rounds, depth, leaves, mbin, sparsity, policy in CASES:
        if args.only and args.only not in name:
            continue

        common = [
            "--rows", str(rows), "--features", str(feats), "--rounds", str(rounds),
            "--depth", str(depth), "--max-bin", str(mbin), "--sparsity", str(sparsity),
            "--max-leaves", str(leaves), "--threads", str(args.threads), "--repeats", "1",
        ]
        rs = [args.binary, *common]
        if policy == "lossguide":
            rs.append("--lossguide")
        if args.warm:
            rs.append("--warmup")
        if args.prewarm:
            rs.append("--prewarm")

        # The Rust GPU run also writes the dataset the Python side reads.
        rs_gpu_t, rs_rmse = parse(run([*rs, "--device", "cuda", "--dump", data]))

        xgb = [
            args.python, args.bench_xgb, data,
            "--rows", str(rows), "--features", str(feats), "--rounds", str(rounds),
            "--depth", str(depth), "--max-leaves", str(leaves), "--grow-policy", policy,
            "--max-bin", str(mbin), "--threads", str(args.threads), "--repeats", "1",
        ]
        if args.warm:
            xgb.append("--warmup")
        xgb_gpu_t, xgb_rmse = parse(run([*xgb, "--device", "cuda"]))

        row = f"{name:<16} {rows:>9} {feats:>5} {rounds:>4} "
        row += f"{rs_gpu_t:>9.3f} {xgb_gpu_t:>9.3f} {xgb_gpu_t / rs_gpu_t:>7.2f}x"

        if args.cpu:
            rs_cpu_t, _ = parse(run([*rs, "--device", "cpu"]))
            xgb_cpu_t, _ = parse(run([*xgb, "--device", "cpu"]))
            row += f" {rs_cpu_t:>9.3f} {xgb_cpu_t:>9.3f}"

        # The two implementations bin and accumulate differently, so the RMSE
        # is reported rather than asserted equal; a large gap means the timings
        # are not measuring the same amount of work.
        rel = abs(rs_rmse - xgb_rmse) / max(abs(xgb_rmse), 1e-12)
        row += f"  {rs_rmse:.5f}/{xgb_rmse:.5f} ({rel:.1e})"
        print(row, flush=True)


if __name__ == "__main__":
    main()
