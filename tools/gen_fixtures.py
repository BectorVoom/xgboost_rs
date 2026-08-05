#!/usr/bin/env python3
"""Generate golden fixtures from a pinned XGBoost for the Rust oracle tests.

Run with the pinned interpreter:

    .venv-oracle/bin/python tools/gen_fixtures.py

Everything written under tests/fixtures/ is committed; Python is a fixture
generator only, never a runtime dependency of the crate.
"""

from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import scipy.sparse
import xgboost as xgb

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures"
AGARICUS = ROOT / "xgboost-master" / "demo" / "data" / "agaricus.txt.train"


def dense_small() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 5)).astype(np.float32)
    y = (x[:, 0] * 2.0 - x[:, 3] + rng.normal(scale=0.1, size=200)).astype(np.float32)
    return x, y


def dense_dup() -> tuple[np.ndarray, np.ndarray]:
    """Few distinct values per feature: exercises cut de-duplication."""
    rng = np.random.default_rng(1)
    x = rng.integers(0, 10, size=(300, 4)).astype(np.float32)
    y = (x[:, 1] - 0.5 * x[:, 2] + rng.normal(scale=0.5, size=300)).astype(np.float32)
    return x, y


def dense_missing() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(250, 6)).astype(np.float32)
    y = (x[:, 2] * 1.5 + rng.normal(scale=0.2, size=250)).astype(np.float32)
    x[rng.random(x.shape) < 0.25] = np.nan
    return x, y


def weighted() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(180, 4)).astype(np.float32)
    y = (x[:, 0] + x[:, 1] * x[:, 1] + rng.normal(scale=0.2, size=180)).astype(np.float32)
    w = rng.uniform(0.25, 4.0, size=180).astype(np.float32)
    return x, y, w


def libsvm_to_csr(path: pathlib.Path) -> tuple[list[int], list[int], list[float], list[float], int]:
    indptr, indices, values, labels = [0], [], [], []
    n_col = 0
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            labels.append(float(parts[0]))
            for tok in parts[1:]:
                k, v = tok.split(":")
                k = int(k)
                indices.append(k)
                values.append(float(v))
                n_col = max(n_col, k + 1)
            indptr.append(len(indices))
    return indptr, indices, values, labels, n_col


def dump_dense(name: str, x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> None:
    """Write a dataset as plain JSON so the Rust tests need no .npy reader.

    Values round-trip exactly: every float32 is representable as a float64.
    """
    data = {
        "n_row": int(x.shape[0]),
        "n_col": int(x.shape[1]),
        "layout": "dense",
        # NaN is the missing sentinel; JSON has no NaN literal, so encode it as null.
        "values": [None if np.isnan(v) else float(v) for v in x.reshape(-1)],
        "labels": [float(v) for v in y],
        "weights": None if w is None else [float(v) for v in w],
    }
    path = OUT / f"data_{name}.json"
    path.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {path.relative_to(ROOT)}  ({path.stat().st_size / 1024:.0f} KiB)")


def dump_csr(
    name: str,
    indptr: list[int],
    indices: list[int],
    values: list[float],
    labels: list[float],
    n_col: int,
) -> None:
    data = {
        "n_row": len(labels),
        "n_col": n_col,
        "layout": "csr",
        "indptr": indptr,
        "indices": indices,
        "values": [float(np.float32(v)) for v in values],
        "labels": labels,
        "weights": None,
    }
    path = OUT / f"data_{name}.json"
    path.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {path.relative_to(ROOT)}  ({path.stat().st_size / 1024:.0f} KiB)")


def dump_case(name: str, dmat: xgb.DMatrix, params: dict, num_round: int, extra: dict) -> None:
    params = dict(params)
    params.setdefault("nthread", 1)
    params.setdefault("tree_method", "hist")
    params.setdefault("objective", "reg:squarederror")
    params.setdefault("eval_metric", "rmse")
    params.setdefault("seed", 0)

    evals_result: dict = {}
    booster = xgb.train(
        params,
        dmat,
        num_boost_round=num_round,
        evals=[(dmat, "train")],
        evals_result=evals_result,
        verbose_eval=False,
    )

    ptrs, vals = dmat.get_quantile_cut()
    model = json.loads(booster.save_raw(raw_format="json").decode("utf-8"))
    preds = booster.predict(dmat)
    margins = booster.predict(dmat, output_margin=True)
    leaf = booster.predict(dmat, pred_leaf=True)

    case = {
        "xgboost_version": xgb.__version__,
        "params": params,
        "num_round": num_round,
        "cut_ptrs": [int(p) for p in ptrs],
        "cut_values": [float(v) for v in vals],
        "base_score": float(model["learner"]["learner_model_param"]["base_score"]),
        "rmse": [float(v) for v in evals_result["train"]["rmse"]],
        "predictions": [float(v) for v in preds],
        "margins": [float(v) for v in margins],
        "leaf": [[int(v) for v in row] for row in np.atleast_2d(leaf)],
        "trees": model["learner"]["gradient_booster"]["model"]["trees"],
        "score_gain": {k: float(v) for k, v in booster.get_score(importance_type="gain").items()},
        "score_weight": {k: float(v) for k, v in booster.get_score(importance_type="weight").items()},
        **extra,
    }
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(case, separators=(",", ":")))
    print(f"wrote {path.relative_to(ROOT)}  ({path.stat().st_size / 1024:.0f} KiB)")


def dump_method_case(
    name: str,
    dmat: xgb.DMatrix,
    params: dict,
    num_round: int,
    extra: dict,
    base: xgb.Booster | None = None,
) -> xgb.Booster:
    """A case for a tree method or booster other than plain `hist`.

    Kept separate from `dump_case` because these cases are compared on
    predictions, metrics and tree structure but not on quantile cuts: `approx`
    re-sketches per round, `exact` never sketches at all, and `gblinear` has no
    trees to compare.
    """
    params = dict(params)
    params.setdefault("nthread", 1)
    params.setdefault("objective", "reg:squarederror")
    params.setdefault("seed", 0)
    metric = params.setdefault("eval_metric", "rmse")

    evals_result: dict = {}
    booster = xgb.train(
        params,
        dmat,
        num_boost_round=num_round,
        evals=[(dmat, "train")],
        evals_result=evals_result,
        verbose_eval=False,
        xgb_model=base,
    )
    model = json.loads(booster.save_raw(raw_format="json").decode("utf-8"))
    gbm = model["learner"]["gradient_booster"]

    case = {
        "xgboost_version": xgb.__version__,
        "params": params,
        "num_round": num_round,
        "base_score": float(model["learner"]["learner_model_param"]["base_score"]),
        "metric": metric,
        "metric_history": [float(v) for v in evals_result["train"][metric]],
        "predictions": [float(v) for v in booster.predict(dmat)],
        "margins": [float(v) for v in booster.predict(dmat, output_margin=True)],
        "trees": gbm.get("model", {}).get("trees", []),
        "weights": gbm.get("model", {}).get("weights", []),
        **extra,
    }
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(case, separators=(",", ":")))
    print(f"wrote {path.relative_to(ROOT)}  ({path.stat().st_size / 1024:.0f} KiB)")
    return booster


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    x, y = dense_small()
    dump_dense("dense_small", x, y)
    dump_case(
        "dense_small_b256_d6",
        xgb.DMatrix(x, label=y),
        {"max_bin": 256, "max_depth": 6, "eta": 0.3},
        8,
        {"data": "dense_small", "n_row": int(x.shape[0]), "n_col": int(x.shape[1])},
    )
    dump_case(
        "dense_small_b16_d3",
        xgb.DMatrix(x, label=y),
        {"max_bin": 16, "max_depth": 3, "eta": 0.5, "lambda": 2.0, "min_child_weight": 3.0},
        6,
        {"data": "dense_small", "n_row": int(x.shape[0]), "n_col": int(x.shape[1])},
    )

    x, y = dense_dup()
    dump_dense("dense_dup", x, y)
    dump_case(
        "dense_dup_b8_d4",
        xgb.DMatrix(x, label=y),
        {"max_bin": 8, "max_depth": 4, "eta": 0.3, "gamma": 0.1},
        6,
        {"data": "dense_dup", "n_row": int(x.shape[0]), "n_col": int(x.shape[1])},
    )

    x, y = dense_missing()
    dump_dense("dense_missing", x, y)
    dump_case(
        "dense_missing_b64_d5",
        xgb.DMatrix(x, label=y),
        {"max_bin": 64, "max_depth": 5, "eta": 0.3, "alpha": 0.5},
        6,
        {"data": "dense_missing", "n_row": int(x.shape[0]), "n_col": int(x.shape[1])},
    )

    x, y, w = weighted()
    dump_dense("weighted", x, y, w)
    dump_case(
        "weighted_b32_d4",
        xgb.DMatrix(x, label=y, weight=w),
        {"max_bin": 32, "max_depth": 4, "eta": 0.3},
        6,
        {"data": "weighted", "n_row": int(x.shape[0]), "n_col": int(x.shape[1])},
    )

    indptr, indices, values, labels, n_col = libsvm_to_csr(AGARICUS)
    dump_csr("agaricus", indptr, indices, values, labels, n_col)
    csr = scipy.sparse.csr_matrix(
        (np.asarray(values, dtype=np.float32), np.asarray(indices), np.asarray(indptr)),
        shape=(len(labels), n_col),
    )
    dump_case(
        "agaricus_b256_d6",
        xgb.DMatrix(csr, label=np.asarray(labels, dtype=np.float32)),
        {"max_bin": 256, "max_depth": 6, "eta": 0.3},
        5,
        {"data": "agaricus", "n_row": len(labels), "n_col": n_col},
    )


def tree_method_cases() -> None:
    """Cases for the tree methods, boosters and updaters beyond CPU `hist`."""
    datasets: dict[str, xgb.DMatrix] = {}

    x, y = dense_small()
    datasets["dense_small"] = xgb.DMatrix(x, label=y)
    x, y = dense_dup()
    datasets["dense_dup"] = xgb.DMatrix(x, label=y)
    x, y = dense_missing()
    datasets["dense_missing"] = xgb.DMatrix(x, label=y)
    x, y, w = weighted()
    datasets["weighted"] = xgb.DMatrix(x, label=y, weight=w)
    indptr, indices, values, labels, n_col = libsvm_to_csr(AGARICUS)
    csr = scipy.sparse.csr_matrix(
        (np.asarray(values, dtype=np.float32), np.asarray(indices), np.asarray(indptr)),
        shape=(len(labels), n_col),
    )
    datasets["agaricus"] = xgb.DMatrix(csr, label=np.asarray(labels, dtype=np.float32))

    shape = {
        name: {"n_row": int(d.num_row()), "n_col": int(d.num_col())}
        for name, d in datasets.items()
    }

    def case(name, data, params, rounds):
        dump_method_case(
            name, datasets[data], params, rounds, {"data": data, **shape[data]}
        )

    # --- exact ---
    case("exact_dense_small", "dense_small", {"tree_method": "exact", "max_depth": 4, "eta": 0.3}, 5)
    case(
        "exact_missing",
        "dense_missing",
        {"tree_method": "exact", "max_depth": 5, "eta": 0.3, "alpha": 0.5},
        4,
    )
    # `gamma` is applied by the `prune` stage of the exact pipeline, so this
    # case pins the pruned tree — deleted node slots included.
    case(
        "exact_gamma",
        "dense_dup",
        {"tree_method": "exact", "max_depth": 4, "eta": 0.3, "gamma": 0.5},
        4,
    )
    case("exact_sparse", "agaricus", {"tree_method": "exact", "max_depth": 4, "eta": 0.3}, 3)
    case(
        "exact_logistic",
        "agaricus",
        {
            "tree_method": "exact",
            "max_depth": 4,
            "eta": 0.3,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        },
        4,
    )
    case(
        "exact_weighted",
        "weighted",
        {"tree_method": "exact", "max_depth": 4, "eta": 0.3, "min_child_weight": 2.0},
        4,
    )

    # --- approx ---
    case(
        "approx_dense_small",
        "dense_small",
        {"tree_method": "approx", "max_depth": 4, "eta": 0.3, "max_bin": 64},
        5,
    )
    case("approx_missing", "dense_missing", {"tree_method": "approx", "max_depth": 5, "eta": 0.3}, 4)
    case("approx_sparse", "agaricus", {"tree_method": "approx", "max_depth": 4, "eta": 0.3}, 3)
    # A varying hessian is what makes `approx` re-sketch every round, so this
    # is the case that actually exercises the weighted sketch.
    case(
        "approx_logistic",
        "agaricus",
        {
            "tree_method": "approx",
            "max_depth": 4,
            "eta": 0.3,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        },
        4,
    )
    case(
        "approx_weighted",
        "weighted",
        {"tree_method": "approx", "max_depth": 4, "eta": 0.3, "max_bin": 32},
        4,
    )

    # --- gblinear ---
    for updater in ("shotgun", "coord_descent"):
        case(
            f"linear_{updater}",
            "dense_small",
            {"booster": "gblinear", "updater": updater, "eta": 0.5, "lambda": 0.1},
            30,
        )

    # --- continued training and process_type=update ---
    base_params = {"tree_method": "hist", "max_depth": 4, "eta": 0.3}
    base = dump_method_case(
        "update_base",
        datasets["dense_small"],
        base_params,
        4,
        {"data": "dense_small", **shape["dense_small"]},
    )
    dump_method_case(
        "update_continued",
        datasets["dense_small"],
        base_params,
        3,
        {"data": "dense_small", "base": "update_base", **shape["dense_small"]},
        base=base,
    )
    for name, updater, gamma in [
        ("update_refresh", "refresh", 0.0),
        ("update_prune", "prune", 2.0),
        ("update_refresh_prune", "refresh,prune", 2.0),
    ]:
        dump_method_case(
            name,
            datasets["dense_small"],
            {**base_params, "process_type": "update", "updater": updater, "gamma": gamma},
            4,
            {"data": "dense_small", "base": "update_base", **shape["dense_small"]},
            base=base,
        )


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
    tree_method_cases()
