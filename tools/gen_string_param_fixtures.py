#!/usr/bin/env python3
"""Golden fixtures pinning *every string-valued parameter* against real XGBoost.

`tools/gen_fixtures.py` pins the numeric core of a `hist` fit. This script pins
the other axis: for each accepted spelling of each string-valued parameter, it
runs a fit configured with that spelling and records what XGBoost produced.

The Rust side (`tests/oracle_string_parameters.rs`) replays each case and holds
the result to the SPEC §1.5 bar. Cases XGBoost itself rejects are recorded as
`"error"` cases, so "this build refuses that combination" is pinned too rather
than silently skipped.

Run with the pinned interpreter (XGBoost **3.4.0**, the upstream line this crate
was written against — SPEC.md §"Upstream reference: vendored XGBoost 3.4.0-dev".
That is deliberately *newer* than the 3.0.5 used by `gen_fixtures.py`'s numeric
core: `reg:expectileerror` and the expectile metric do not exist at all in
3.0.5, so 3.0.5 cannot pin this crate's objective surface):

    <venv>/bin/python tools/gen_string_param_fixtures.py

Everything written under tests/fixtures/strparam/ is committed; Python is a
fixture generator only, never a runtime dependency of the crate.
"""

from __future__ import annotations

import json
import pathlib
import traceback

import numpy as np
import xgboost as xgb

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "strparam"

# Every fit is single-threaded and seeded: the fixtures have to be reproducible
# and the Rust side is compared value-for-value.
BASE = {"nthread": 1, "seed": 0, "verbosity": 0}

N_ROUND = 4

# How many rows of the SHAP contribution/interaction blocks to record. Those
# are rows x groups x (cols+1)^k floats; a prefix pins the behaviour just as
# well as the whole matrix and keeps the committed fixtures small.
HEAVY_PREDICT_ROWS = 48

# Parameters that only mean something on a CUDA device. On a CPU-only box
# XGBoost *silently falls back* for some of them (`device=cuda` trains on the
# CPU and produces byte-identical output), so a fixture generated here would
# look like a passing GPU test while proving nothing. Each such case records
# `requires_gpu` and the generator stamps whether a GPU was actually present.
GPU_ONLY_CASES = {
    "sampling_method_gradient_based",
    "updater_grow_gpu_hist",
    "updater_grow_gpu_approx",
    "device_cuda",
}


def detect_gpu() -> bool:
    """True only if XGBoost can really train on a CUDA device here.

    `device=cuda` is not a probe: it falls back to CPU without raising. The
    `grow_gpu_hist` updater does not fall back, so it is the honest test.
    """
    try:
        x = np.zeros((8, 2), dtype=np.float32)
        d = xgb.DMatrix(x, label=np.arange(8, dtype=np.float32))
        xgb.train(
            {"updater": "grow_gpu_hist", "device": "cuda", "verbosity": 0, "nthread": 1},
            d,
            num_boost_round=1,
        )
        return True
    except Exception:  # noqa: BLE001 - absence of a GPU is the answer
        return False


GPU_PRESENT = False  # set in main()


# --------------------------------------------------------------- datasets ---
#
# One dataset per label geometry. Each is dumped in the same JSON shape
# `tests/common/mod.rs` already reads, extended with the optional ranking and
# censoring metadata that the rank/survival objectives need.


def _features(rows: int, cols: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(rows, cols)).astype(np.float32)


def ds_regression() -> dict:
    """Continuous targets, both signs: the plain regression losses."""
    x = _features(240, 6, 10)
    rng = np.random.default_rng(11)
    y = (x[:, 0] * 2.0 - x[:, 3] + rng.normal(scale=0.1, size=240)).astype(np.float32)
    return {"x": x, "y": y}


def ds_positive() -> dict:
    """Strictly positive targets: the log-link losses (gamma, tweedie, poisson,
    squaredlogerror) reject anything <= 0."""
    x = _features(240, 6, 12)
    rng = np.random.default_rng(13)
    y = np.exp(0.5 * x[:, 0] + 0.25 * x[:, 2] + rng.normal(scale=0.1, size=240))
    return {"x": x, "y": y.astype(np.float32)}


def ds_unit() -> dict:
    """Targets in [0, 1]: reg:logistic and the probability-shaped losses."""
    x = _features(240, 6, 14)
    y = 1.0 / (1.0 + np.exp(-(x[:, 0] + 0.5 * x[:, 1])))
    return {"x": x, "y": y.astype(np.float32)}


def ds_binary() -> dict:
    x = _features(300, 6, 16)
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(np.float32)
    return {"x": x, "y": y}


def ds_binary_weighted() -> dict:
    """Binary with weights: `ams@t` and the weighted AUC paths read them."""
    d = ds_binary()
    rng = np.random.default_rng(17)
    d["w"] = rng.uniform(0.5, 2.0, size=d["y"].shape[0]).astype(np.float32)
    return d


def ds_multiclass() -> dict:
    x = _features(300, 6, 18)
    score = np.stack([x[:, 0], x[:, 1], x[:, 2]], axis=1)
    y = np.argmax(score, axis=1).astype(np.float32)
    return {"x": x, "y": y, "num_class": 3}


def ds_ranking() -> dict:
    """Graded relevance in query groups of 5: what NDCG is defined on."""
    rows, group_size = 300, 5
    x = _features(rows, 6, 20)
    rng = np.random.default_rng(21)
    y = rng.integers(0, 4, size=rows).astype(np.float32)
    return {"x": x, "y": y, "group": [group_size] * (rows // group_size)}


def ds_ranking_binary() -> dict:
    """Ranking with *binary* relevance. `map` and `pre` are only defined on
    binary labels — upstream fails the `is_binary` check on graded ones — so
    they get their own dataset rather than a graded one they would reject."""
    rows, group_size = 300, 5
    x = _features(rows, 6, 30)
    rng = np.random.default_rng(31)
    y = (rng.random(rows) < 0.4).astype(np.float32)
    return {"x": x, "y": y, "group": [group_size] * (rows // group_size)}


def ds_survival() -> dict:
    """Interval-censored targets for survival:aft and aft-nloglik."""
    x = _features(240, 6, 22)
    rng = np.random.default_rng(23)
    lower = np.exp(0.4 * x[:, 0] + rng.normal(scale=0.1, size=240)).astype(np.float32)
    upper = (lower * rng.uniform(1.1, 2.0, size=240)).astype(np.float32)
    # `label` is unused by AFT but DMatrix wants one; the bounds carry the target.
    return {"x": x, "y": lower, "label_lower_bound": lower, "label_upper_bound": upper}


def ds_cox() -> dict:
    """survival:cox reads the label's *sign* as the censoring indicator:
    positive means an observed event, negative means right-censored."""
    x = _features(240, 6, 24)
    rng = np.random.default_rng(25)
    t = np.exp(0.3 * x[:, 0] + rng.normal(scale=0.2, size=240))
    censored = rng.random(240) < 0.3
    y = np.where(censored, -t, t).astype(np.float32)
    return {"x": x, "y": y}


def ds_multitarget() -> dict:
    """A 2-column label matrix: the only shape `multi_strategy` changes."""
    x = _features(240, 6, 26)
    rng = np.random.default_rng(27)
    y = np.stack(
        [
            x[:, 0] * 2.0 + rng.normal(scale=0.1, size=240),
            -x[:, 1] + 0.5 * x[:, 2] + rng.normal(scale=0.1, size=240),
        ],
        axis=1,
    ).astype(np.float32)
    return {"x": x, "y": y}


def ds_missing() -> dict:
    """A quarter of the values missing: what `default_direction` steers."""
    x = _features(240, 6, 28)
    rng = np.random.default_rng(29)
    y = (x[:, 2] * 1.5 + rng.normal(scale=0.2, size=240)).astype(np.float32)
    x[rng.random(x.shape) < 0.25] = np.nan
    return {"x": x, "y": y}


DATASETS = {
    "regression": ds_regression,
    "positive": ds_positive,
    "unit": ds_unit,
    "binary": ds_binary,
    "binary_weighted": ds_binary_weighted,
    "multiclass": ds_multiclass,
    "ranking": ds_ranking,
    "ranking_binary": ds_ranking_binary,
    "survival": ds_survival,
    "cox": ds_cox,
    "multitarget": ds_multitarget,
    "missing": ds_missing,
}


def dump_dataset(name: str, d: dict) -> None:
    x, y = d["x"], d["y"]
    out = {
        "n_row": int(x.shape[0]),
        "n_col": int(x.shape[1]),
        "layout": "dense",
        # NaN is the missing sentinel; JSON has no NaN literal, so encode as null.
        "values": [None if np.isnan(v) else float(v) for v in x.reshape(-1)],
        # A multi-target label is dumped flattened, with `n_target` to shape it.
        "labels": [float(v) for v in np.asarray(y).reshape(-1)],
        "n_target": int(y.shape[1]) if np.asarray(y).ndim == 2 else 1,
        "weights": [float(v) for v in d["w"]] if "w" in d else None,
        "group": [int(g) for g in d["group"]] if "group" in d else None,
        "label_lower_bound": (
            [float(v) for v in d["label_lower_bound"]] if "label_lower_bound" in d else None
        ),
        "label_upper_bound": (
            [float(v) for v in d["label_upper_bound"]] if "label_upper_bound" in d else None
        ),
    }
    path = OUT / f"data_{name}.json"
    path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"  data {name:16} {x.shape[0]}x{x.shape[1]}  ({path.stat().st_size / 1024:.0f} KiB)")


def make_dmatrix(name: str) -> xgb.DMatrix:
    d = DATASETS[name]()
    kwargs = {"label": d["y"]}
    if "w" in d:
        kwargs["weight"] = d["w"]
    if "label_lower_bound" in d:
        kwargs["label_lower_bound"] = d["label_lower_bound"]
        kwargs["label_upper_bound"] = d["label_upper_bound"]
    dmat = xgb.DMatrix(d["x"], missing=np.nan, **kwargs)
    if "group" in d:
        dmat.set_group(d["group"])
    return dmat


# ------------------------------------------------------------ case table ---
#
# (case name, the parameter this case pins, its value, dataset, extra params).
# Extra params carry whatever the value needs to be *legal* and *effective* —
# an objective for a metric, a booster for a dart knob, and so on.

CASES: list[tuple[str, str, str, str, dict]] = []


def case(name: str, param: str, value: str, data: str, **params) -> None:
    CASES.append((name, param, value, data, params))


def objective_extras(obj: str) -> dict:
    """The companion parameters an objective needs before it will configure.

    Upstream treats these as required, not defaulted: `reg:quantileerror` and
    `reg:expectileerror` abort on an empty alpha list, and `multi:*` needs the
    class count.
    """
    if obj.startswith("multi:"):
        return {"num_class": 3}
    if obj == "reg:quantileerror":
        return {"quantile_alpha": 0.5}
    if obj == "reg:expectileerror":
        return {"expectile_alpha": 0.5}
    if obj == "reg:tweedie":
        return {"tweedie_variance_power": 1.5}
    return {}


# --- objective: every accepted spelling, on a dataset its labels suit -------
for obj, data in [
    ("reg:squarederror", "regression"),
    ("reg:squaredlogerror", "positive"),
    ("reg:logistic", "unit"),
    ("reg:pseudohubererror", "regression"),
    ("reg:absoluteerror", "regression"),
    ("reg:quantileerror", "regression"),
    ("reg:expectileerror", "regression"),
    ("reg:gamma", "positive"),
    ("reg:tweedie", "positive"),
    ("reg:linear", "regression"),  # deprecated alias for reg:squarederror
    ("count:poisson", "positive"),
    ("survival:cox", "cox"),
    ("survival:aft", "survival"),
    ("binary:logistic", "binary"),
    ("binary:logitraw", "binary"),
    ("binary:hinge", "binary"),
    ("multi:softmax", "multiclass"),
    ("multi:softprob", "multiclass"),
    ("rank:pairwise", "ranking"),
    ("rank:ndcg", "ranking"),
    # `rank:map` optimises MAP, which is only defined on binary relevance.
    ("rank:map", "ranking_binary"),
]:
    slug = obj.replace(":", "_")
    case(f"objective_{slug}", "objective", obj, data, objective=obj, **objective_extras(obj))

# --- eval_metric: every accepted spelling, with an objective that fits ------
for metric, data, obj in [
    ("rmse", "regression", "reg:squarederror"),
    ("rmsle", "positive", "reg:squarederror"),
    ("mae", "regression", "reg:squarederror"),
    ("mape", "positive", "reg:squarederror"),
    ("mphe", "regression", "reg:pseudohubererror"),
    ("logloss", "binary", "binary:logistic"),
    ("error", "binary", "binary:logistic"),
    ("error@0.7", "binary", "binary:logistic"),
    ("merror", "multiclass", "multi:softprob"),
    ("mlogloss", "multiclass", "multi:softprob"),
    ("auc", "binary", "binary:logistic"),
    ("aucpr", "binary", "binary:logistic"),
    # `pre` and `map` are binary-relevance metrics; `ndcg` takes graded labels.
    ("pre", "ranking_binary", "rank:ndcg"),
    ("pre@3", "ranking_binary", "rank:ndcg"),
    ("ndcg", "ranking", "rank:ndcg"),
    ("ndcg@3", "ranking", "rank:ndcg"),
    ("ndcg@3-", "ranking", "rank:ndcg"),
    ("map", "ranking_binary", "rank:map"),
    ("map@3", "ranking_binary", "rank:map"),
    ("map@3-", "ranking_binary", "rank:map"),
    ("poisson-nloglik", "positive", "count:poisson"),
    ("gamma-nloglik", "positive", "reg:gamma"),
    ("cox-nloglik", "cox", "survival:cox"),
    ("gamma-deviance", "positive", "reg:gamma"),
    ("tweedie-nloglik@1.5", "positive", "reg:tweedie"),
    ("aft-nloglik", "survival", "survival:aft"),
    ("interval-regression-accuracy", "survival", "survival:aft"),
    ("quantile", "regression", "reg:quantileerror"),
    ("expectile", "regression", "reg:expectileerror"),
    ("ams@0.15", "binary_weighted", "binary:logistic"),
]:
    slug = metric.replace(":", "_").replace("@", "_at_").replace("-", "_").replace(".", "p")
    case(f"metric_{slug}", "eval_metric", metric, data,
         objective=obj, eval_metric=metric, **objective_extras(obj))

# --- tree_method -----------------------------------------------------------
for tm in ["auto", "exact", "approx", "hist"]:
    case(f"tree_method_{tm}", "tree_method", tm, "regression", tree_method=tm)

# --- grow_policy (hist and approx are the two that honour it) --------------
for gp in ["depthwise", "lossguide"]:
    case(f"grow_policy_{gp}", "grow_policy", gp, "regression",
         tree_method="hist", grow_policy=gp, max_depth=0, max_leaves=8)

# --- sampling_method -------------------------------------------------------
# `gradient_based` is a CUDA-only sampler upstream; the case is still recorded
# so the CPU build's refusal is pinned, and the GPU run replaces it.
for sm in ["uniform", "gradient_based"]:
    case(f"sampling_method_{sm}", "sampling_method", sm, "regression",
         tree_method="hist", subsample=0.6, sampling_method=sm)

# --- process_type / updater pipelines --------------------------------------
case("process_type_default", "process_type", "default", "regression",
     tree_method="hist", process_type="default")
# `process_type=update` needs an existing model to update, handled specially.
case("process_type_update", "process_type", "update", "regression",
     tree_method="hist", process_type="update", updater="refresh")

for up in ["grow_colmaker", "grow_histmaker", "grow_quantile_histmaker",
           "grow_quantile_histmaker_sycl", "grow_gpu_hist", "grow_gpu_approx"]:
    case(f"updater_{up}", "updater", up, "regression", updater=up)
case("updater_prune", "updater", "prune", "regression",
     updater="grow_colmaker,prune", gamma=0.5)
case("updater_refresh", "updater", "refresh", "regression",
     updater="grow_quantile_histmaker,refresh", refresh_leaf=1)

# --- multi_strategy --------------------------------------------------------
for ms in ["one_output_per_tree", "multi_output_tree"]:
    case(f"multi_strategy_{ms}", "multi_strategy", ms, "multitarget",
         tree_method="hist", multi_strategy=ms)

# --- default_direction (an `exact`-only knob, and only missing values move) -
for dd in ["learn", "left", "right"]:
    case(f"default_direction_{dd}", "default_direction", dd, "missing",
         tree_method="exact", default_direction=dd)

# --- monotone_constraints --------------------------------------------------
for mc in ["-1", "0", "1"]:
    case(f"monotone_constraints_{mc.replace('-', 'neg')}", "monotone_constraints", mc,
         "regression", tree_method="hist",
         monotone_constraints="(" + ",".join([mc] + ["0"] * 5) + ")")

# --- dart: sample_type x normalize_type ------------------------------------
for st in ["uniform", "weighted"]:
    case(f"sample_type_{st}", "sample_type", st, "regression",
         booster="dart", sample_type=st, rate_drop=0.2, skip_drop=0.0)
for nt in ["tree", "forest"]:
    case(f"normalize_type_{nt}", "normalize_type", nt, "regression",
         booster="dart", normalize_type=nt, rate_drop=0.2, skip_drop=0.0)

# --- gblinear: feature_selector x updater ----------------------------------
for fs in ["cyclic", "shuffle", "random", "greedy", "thrifty"]:
    # `greedy`/`thrifty` are coord_descent-only selectors upstream.
    upd = "coord_descent" if fs in ("greedy", "thrifty", "cyclic", "random") else "shotgun"
    case(f"feature_selector_{fs}", "feature_selector", fs, "regression",
         booster="gblinear", feature_selector=fs, updater=upd, top_k=3)
for lu in ["shotgun", "coord_descent"]:
    case(f"linear_updater_{lu}", "updater", lu, "regression",
         booster="gblinear", updater=lu,
         feature_selector="shuffle" if lu == "shotgun" else "cyclic")

# --- aft_loss_distribution -------------------------------------------------
for dist in ["normal", "logistic", "extreme"]:
    case(f"aft_loss_distribution_{dist}", "aft_loss_distribution", dist, "survival",
         objective="survival:aft", eval_metric="aft-nloglik",
         aft_loss_distribution=dist, aft_loss_distribution_scale=1.0)

# --- lambdarank_pair_method ------------------------------------------------
for pm in ["mean", "topk"]:
    case(f"lambdarank_pair_method_{pm}", "lambdarank_pair_method", pm, "ranking",
         objective="rank:ndcg", eval_metric="ndcg", lambdarank_pair_method=pm,
         lambdarank_num_pair_per_sample=2)

# --- verbosity: must not change a single output value ----------------------
for v in ["0", "1", "2", "3"]:
    case(f"verbosity_{v}", "verbosity", v, "regression", tree_method="hist", verbosity=int(v))

# --- device ----------------------------------------------------------------
case("device_cpu", "device", "cpu", "regression", tree_method="hist", device="cpu")
case("device_cuda", "device", "cuda", "regression", tree_method="hist", device="cuda")


# ---------------------------------------------------------------- runner ---


def prediction_block(booster: xgb.Booster, dmat: xgb.DMatrix) -> dict:
    """Every `predict_type` spelling, recorded off the same model.

    This is what pins the `PredictionType` enum: each variant is a distinct
    `predict` call upstream, so one model exercises all seven.
    """
    def block(p, rows: int | None) -> dict:
        p = np.asarray(p)
        full_shape = [int(s) for s in p.shape]
        # SHAP contributions and interactions are rows x groups x cols(+1)^k:
        # the whole matrix is megabytes and adds no coverage over a prefix, so
        # the heavy blocks are truncated along the row axis. `shape` stays the
        # full shape and `rows` says how much of it is here.
        kept = p if rows is None else p[:rows]
        return {
            "shape": full_shape,
            "rows": int(kept.shape[0]),
            "values": [float(v) for v in kept.astype(np.float64).reshape(-1)],
        }

    out = {}
    for key, kwargs, rows in [
        ("value", {}, None),
        ("margin", {"output_margin": True}, None),
        ("contribution", {"pred_contribs": True}, HEAVY_PREDICT_ROWS),
        ("approx_contribution", {"pred_contribs": True, "approx_contribs": True}, HEAVY_PREDICT_ROWS),
        ("interaction", {"pred_interactions": True}, HEAVY_PREDICT_ROWS),
        ("approx_interaction", {"pred_interactions": True, "approx_contribs": True}, HEAVY_PREDICT_ROWS),
        ("leaf", {"pred_leaf": True}, None),
    ]:
        try:
            out[key] = block(booster.predict(dmat, **kwargs), rows)
        except Exception as exc:  # noqa: BLE001 - recorded, not raised
            out[key] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def parse_base_score(raw: str) -> list[float]:
    """`base_score` is a scalar string on a single-output model and a
    bracketed vector (`"[-3.5e-2]"`) once the model has several outputs."""
    text = str(raw).strip()
    if text.startswith("["):
        inner = text.strip("[]").strip()
        return [float(v) for v in inner.split(",")] if inner else []
    return [float(text)]


def run_case(name: str, param: str, value: str, data: str, params: dict) -> dict:
    """Train one case and record everything the Rust side compares.

    A configuration XGBoost refuses is recorded as an `error` case rather than
    dropped: "this spelling is rejected here" is itself behaviour worth pinning.
    """
    full = dict(BASE)
    full.setdefault("objective", "reg:squarederror")
    full.update(params)
    full.setdefault("max_depth", 4)
    full.setdefault("eta", 0.3)

    dmat = make_dmatrix(data)
    record: dict = {
        "xgboost_version": xgb.__version__,
        "param": param,
        "value": value,
        "data": data,
        "params": {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in full.items()},
        "num_round": N_ROUND,
    }
    if name in GPU_ONLY_CASES:
        # Recorded on every run, so a fixture can never be mistaken for GPU
        # evidence it is not: on a CPU box `device=cuda` trains on the CPU and
        # still reports `ok`.
        record["requires_gpu"] = True
        record["gpu_present"] = GPU_PRESENT

    try:
        evals_result: dict = {}
        base = None
        if full.get("process_type") == "update":
            # An `update` run needs a model to update; build it with the same
            # configuration minus the update itself.
            seed_params = {k: v for k, v in full.items()
                           if k not in ("process_type", "updater")}
            base = xgb.train(seed_params, dmat, num_boost_round=N_ROUND, verbose_eval=False)
            record["updates_existing_model"] = True

        booster = xgb.train(
            full,
            dmat,
            num_boost_round=N_ROUND,
            evals=[(dmat, "train")],
            evals_result=evals_result,
            verbose_eval=False,
            xgb_model=base,
        )
    except Exception as exc:  # noqa: BLE001 - refusal is the recorded outcome
        record["outcome"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["error_detail"] = traceback.format_exc(limit=1).strip().splitlines()[-1]
        return record

    model = json.loads(booster.save_raw(raw_format="json").decode("utf-8"))
    learner = model["learner"]
    gbm = learner["gradient_booster"]
    # dart nests its trees one level deeper than gbtree.
    gbm_model = gbm.get("model", gbm.get("gbtree", {}).get("model", {}))

    metric_name = next(iter(evals_result["train"]))
    record.update(
        {
            "outcome": "ok",
            "objective_name": learner["objective"]["name"],
            "base_score": parse_base_score(learner["learner_model_param"]["base_score"]),
            "num_class": int(learner["learner_model_param"].get("num_class", 0)),
            "num_target": int(learner["learner_model_param"].get("num_target", 1)),
            "metric": metric_name,
            "metric_history": [float(v) for v in evals_result["train"][metric_name]],
            "trees": gbm_model.get("trees", []),
            "weights": gbm_model.get("weights", []),
            "predict": prediction_block(booster, dmat),
        }
    )
    return record


def main() -> None:
    global GPU_PRESENT
    OUT.mkdir(parents=True, exist_ok=True)
    GPU_PRESENT = detect_gpu()
    print(f"xgboost {xgb.__version__}   cuda device usable: {GPU_PRESENT}")

    print("datasets:")
    used = sorted({data for _, _, _, data, _ in CASES})
    for name in used:
        dump_dataset(name, DATASETS[name]())

    print(f"cases: {len(CASES)}")
    index = []
    n_ok = n_err = 0
    for name, param, value, data, params in CASES:
        record = run_case(name, param, value, data, params)
        (OUT / f"{name}.json").write_text(json.dumps(record, separators=(",", ":")))
        if record["outcome"] == "ok":
            n_ok += 1
            print(f"  ok    {param:24} {value}")
        else:
            n_err += 1
            print(f"  ERROR {param:24} {value}  -> {record['error'][:90]}")
        entry = {"case": name, "param": param, "value": value,
                 "data": data, "outcome": record["outcome"]}
        if name in GPU_ONLY_CASES:
            entry["requires_gpu"] = True
        index.append(entry)

    (OUT / "index.json").write_text(
        json.dumps(
            {"xgboost_version": xgb.__version__, "gpu_present": GPU_PRESENT, "cases": index},
            indent=1,
        )
    )
    print(f"\n{n_ok} trained, {n_err} refused, {len(CASES)} total -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
