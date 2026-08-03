#!/usr/bin/env python3
"""Time the reference XGBoost on a dataset produced by `train_bench --dump`.

    cargo run --release --no-default-features --bin train_bench -- \
        --rows 100000 --features 50 --dump /tmp/bench.bin
    .venv-oracle/bin/python tools/bench_xgb.py /tmp/bench.bin \
        --rows 100000 --features 50 --rounds 20 --threads 8

Reads the exact bytes the Rust benchmark trained on, so the two timings are
comparable run for run.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import xgboost as xgb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--features", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--max-bin", type=int, default=256)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()

    raw = np.fromfile(args.path, dtype=np.float32)
    n = args.rows * args.features
    x = raw[:n].reshape(args.rows, args.features)
    y = raw[n : n + args.rows]

    params = {
        "tree_method": "hist",
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.3,
        "max_depth": args.depth,
        "max_bin": args.max_bin,
        "seed": 0,
    }
    if args.threads > 0:
        params["nthread"] = args.threads

    # DMatrix construction (binning included) is measured separately, matching
    # how the Rust side reports data build vs train.
    t = time.perf_counter()
    dtrain = xgb.DMatrix(x, label=y)
    build = time.perf_counter() - t

    best = float("inf")
    rmse = None
    for _ in range(args.repeats):
        evals_result: dict = {}
        t = time.perf_counter()
        xgb.train(
            params,
            dtrain,
            num_boost_round=args.rounds,
            evals=[(dtrain, "train")],
            evals_result=evals_result,
            verbose_eval=False,
        )
        best = min(best, time.perf_counter() - t)
        rmse = evals_result["train"]["rmse"][-1]

    print(f"xgboost {xgb.__version__}  threads={args.threads or 'auto'}")
    print(f"data build: {build:.3f}s")
    print(f"train: {best:.3f}s  final train-rmse: {rmse:.6f}")


if __name__ == "__main__":
    main()
