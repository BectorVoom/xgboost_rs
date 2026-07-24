# Phase Research: Rewrite XGBoost in Rust (same/all API, builder pattern, module design, oracle API test, oracle speed test)

Evidence tags used: `[VERIFIED: CODEGRAPH ...]`, `[VERIFIED: LOCAL path:line]`, `[VERIFIED: WEB url]`, `[INFERRED: ...]`, `[UNVERIFIED: ...]`. Access date for all web sources: 2026-07-21.

---

## Summary

- **Goal.** Reimplement XGBoost in Rust with the *same and complete public API*, an idiomatic *builder pattern* for parameters, a *module design* mirroring XGBoost's core, plus an *oracle API test* (Rust output vs real XGBoost) and an *oracle speed test* (Rust vs C++ timing).
- **Reality of the existing Rust port.** The repo is **not** a full XGBoost. It is a focused CubeCL port of only the GPU histogram path (`gpu_hist`): a gradient quantiser, an ELLPACK matrix, a histogram builder/engine with a builder pattern, plus a CPU oracle and a bench binary. There is **no** `DMatrix`, `Booster`, `Learner`, objective, metric, tree updater, predictor, or C API. `[VERIFIED: LOCAL src/lib.rs:14-18]` `[VERIFIED: LOCAL src/gpu/mod.rs:1-6]`
- **Upstream target.** Vendored XGBoost source tree at `xgboost/`, version **3.4.0-dev**, CodeGraph-indexed. `[VERIFIED: LOCAL xgboost/CMakeLists.txt:project VERSION 3.4.0]`
- **Authoritative API surface.** The stable ABI is the **C API** in `include/xgboost/c_api.h` (~90 `XGB_DLL` functions). The high-level Python (`DMatrix`, `Booster`, `train`, sklearn `XGBModel`/`XGBClassifier`/`XGBRegressor`) and R (`xgb.DMatrix`, `xgb.train`) APIs are thin layers over that C ABI. `[VERIFIED: LOCAL xgboost/include/xgboost/c_api.h]` `[VERIFIED: CODEGRAPH XGBoosterUpdateOneIter → UpdateOneIter]`
- **Biggest risk.** "Same and all API" plus "bit-for-bit oracle match" against a mature C++/CUDA codebase is a multi-quarter effort. Exact-match blockers: floating-point summation order, quantile sketching, histogram binning, missing-value routing, base_score/intercept estimation, and regularized split gain. Determinism is achievable *only* under a constrained config (single thread, fixed seed, `hist`, fixed `base_score`).
- **Highest-impact tooling gap.** No reference XGBoost is available locally to act as an oracle: **no** installed `xgboost` Python package, **no** built `libxgboost.so`, **no** R, **no** `cmake`/`nvcc`. Only `g++` and Rust exist. Producing golden outputs requires installing/building XGBoost (network access) — this is currently an unresolved blocker for the oracle tests. `[VERIFIED: LOCAL bash: pip/import xgboost fails, which R/cmake/nvcc empty]`

---

## Existing Rust Port Inventory

**Crate:** `xgboost_rs` v0.1.0, edition 2024. `[VERIFIED: LOCAL Cargo.toml:1-4]`

**Dependencies** `[VERIFIED: LOCAL Cargo.toml:6-17]`:
- `anyhow = "1.0.104"`, `bytemuck = "1"` (locked 1.25.1), `thiserror = "2.0.19"`
- `cubecl = { version = "0.10.0", features = ["vulkan"] }` — `vulkan` is mandatory because kernels use `i64`/`f64`, which the WGSL compiler cannot express (comment). `[VERIFIED: LOCAL Cargo.toml:9-11]`
- Features: `default = []`; `cuda = ["cubecl/cuda"]`. `[VERIFIED: LOCAL Cargo.toml:14-17]`
- Transitive (locked): `wgpu 29.0.4`, `naga 29.0.4`, `ash 0.38.0`. `[VERIFIED: LOCAL Cargo.lock]`

**Source tree** `[VERIFIED: LOCAL bash find src]`:
| File | Contents |
|---|---|
| `src/lib.rs` | Crate root; re-exports `error`, `gpu`, `reference`, `Error`, `Result`. Docstring frames the whole crate as "Rust port of XGBoost's `gpu_hist` CUDA device kernels". `[VERIFIED: LOCAL src/lib.rs:1-18]` |
| `src/error.rs` | `Error` enum via `thiserror` (`MatrixShape`, `InvalidCuts`, `GpairCount`, `HistogramLen`, `HistogramBins`, `Sync`) + `Result<T>` alias. `[VERIFIED: LOCAL src/error.rs:5-26]` |
| `src/gpu/mod.rs` | Types `GradientPair{f32}`, `GradientPairPrecise{f64}`, `GradientPairInt64{i64,i64}` (Pod, repr(C)), `DeviceGpairs`, `DeviceRows`, `DeviceHistogram`. `[VERIFIED: LOCAL src/gpu/mod.rs:10-96]` |
| `src/gpu/ellpack.rs` | `EllpackLayout{Dense,DenseCompressed,Sparse}`, `EllpackMatrix` (gidx, row_stride, cut_ptrs, null_value…). `[VERIFIED: LOCAL src/gpu/ellpack.rs:15-57]` |
| `src/gpu/quantiser.rs` | `GradientQuantiser` (deterministic fixed-point), `create_rounding_factor`, `quantise_gpair_kernel` (`#[cube(launch)]`), `quantise`/`quantise_to_device`. Ports `deterministic.cuh` + `quantiser.cu`. `[VERIFIED: LOCAL src/gpu/quantiser.rs:41-165]` |
| `src/gpu/histogram.rs` | `HistogramBuilder` (**builder pattern**: `.force_global()`, `.native_i64_atomics()`, `.build()`), `HistogramEngine` (`.build()`, `.build_to_device()`, `.subtract()`, `.upload_rows()`), CubeCL kernels porting `histogram.cu/.cuh` incl. `AtomicAdd64As32`, shared/global paths, subtraction trick. 636 lines. `[VERIFIED: LOCAL src/gpu/histogram.rs:1-120]` |
| `src/reference.rs` | CPU oracle `cpu_histogram`, deterministic `Rng` (xorshift), `random_gpairs`, `random_matrix`. `[VERIFIED: LOCAL src/reference.rs:30-97]` |
| `src/main.rs` | Tiny wgpu demo of the histogram path. `[VERIFIED: LOCAL src/main.rs]` |
| `src/bin/bench.rs` | `run_oracle` (GPU vs CPU exact i64 equality across every kernel path) + `run_bench` (ms/build, Gentry/s). Env knobs `BENCH_ROWS/FEATURES/BINS/ITERS`. `[VERIFIED: LOCAL src/bin/bench.rs:50-194]` |
| `tests/kernels.rs` | 247 lines; kernel tests vs CPU oracle, mirroring `xgboost/tests/cpp/tree/gpu_hist/test_histogram.cu`. `[VERIFIED: LOCAL tests/kernels.rs:1-40]` |

**Build/test state.** `cargo build --offline --lib` succeeds instantly from cache (target/ already populated: `libxgboost_rs.rlib`, `bench`, `xgboost_rs` binaries present). `[VERIFIED: LOCAL bash: cargo build --offline --lib → Finished]` Running `bench`/tests needs a working CubeCL runtime (Vulkan device such as lavapipe, or CUDA with `--features cuda`); not exercised here. `[INFERRED from src/bin/bench.rs runtime selection]` A memory note claims a "gpu_hist CubeCL 0.10 port tested on lavapipe via `vulkan` feature" — **confirmed** by the code, not by a live run here.

**Distribution artifact.** `xgboost_rs_kaggle.tar.gz` + `KAGGLE.md` document running the oracle/speed bench on a Kaggle CUDA GPU. `[VERIFIED: LOCAL KAGGLE.md:1-95]`

**Bottom line:** the existing work is ~5% of a full XGBoost — the single most performance-critical GPU kernel and its test harness. Everything else (data, boosting loop, objectives, metrics, tree/linear learners, predictors, serialization, C/Python/R API) is greenfield.

---

## XGBoost Public API Inventory

### C API — the stable ABI to mirror (`include/xgboost/c_api.h`, ~90 `XGB_DLL` fns) `[VERIFIED: LOCAL xgboost/include/xgboost/c_api.h grep XGB_DLL]`

**Global / meta:** `XGBoostVersion`, `XGBuildInfo`, `XGBGetLastError`, `XGBRegisterLogCallback`, `XGBSetGlobalConfig`, `XGBGetGlobalConfig`.

**DMatrix — construction:** `XGDMatrixCreateFromURI`, `...FromFile` (legacy), `...FromMat` / `...FromMat_omp` (dense C array), `...FromCSR`, `...FromCSC`, `...FromDense`, `...FromColumnar`, `...FromCudaColumnar`, `...FromCudaArrayInterface`, `...FromDataIter`, `XGProxyDMatrixCreate`, `XGDMatrixCreateFromCallback`, `XGQuantileDMatrixCreateFromCallback`, `XGExtMemQuantileDMatrixCreateFromCallback`, proxy setters (`XGProxyDMatrixSetData*`).
**DMatrix — ops/info:** `XGDMatrixSliceDMatrix(Ex)`, `XGDMatrixFree`, `XGDMatrixSaveBinary`, `XGDMatrixSet/GetFloatInfo`, `...UIntInfo`, `...StrFeatureInfo`, `...DenseInfo`, `...InfoFromInterface`, `XGDMatrixNumRow/NumCol/NumNonMissing/DataSplitMode`, `XGDMatrixGetDataAsCSR`, `XGDMatrixGetQuantileCut`, categories APIs. `[VERIFIED: LOCAL c_api.h:148-986]`

**Booster — lifecycle & params:** `XGBoosterCreate`, `XGBoosterFree`, `XGBoosterReset`, `XGBoosterSlice`, `XGBoosterBoostedRounds`, `XGBoosterSetParam`, `XGBoosterGetNumFeature`.
**Booster — training loop:** `XGBoosterUpdateOneIter` (built-in objective), `XGBoosterBoostOneIter`/`XGBoosterTrainOneIter` (custom grad/hess), `XGBoosterEvalOneIter`. `[VERIFIED: LOCAL c_api.h:1067-1106]` `[VERIFIED: CODEGRAPH XGBoosterUpdateOneIter (c_api.cc:1114) → Learner::UpdateOneIter]`
**Booster — prediction:** `XGBoosterPredict`, `XGBoosterPredictFromDMatrix`, `...FromDense`, `...FromColumnar`, `...FromCSR`, `...FromCudaArray`, `...FromCudaColumnar`. Option-mask bits select margin/leaf/contrib/interaction. `[VERIFIED: LOCAL c_api.cc:1268-1287]`
**Booster — persistence:** `XGBoosterLoadModel`, `XGBoosterSaveModel` (json/ubj by extension), `XGBoosterLoadModelFromBuffer`, `XGBoosterSaveModelToBuffer`, `XGBoosterSerializeToBuffer`, `XGBoosterUnserializeFromBuffer`, `XGBoosterSaveJsonConfig`, `XGBoosterLoadJsonConfig`. `[VERIFIED: LOCAL c_api.cc:1530-1556; c_api.h:1380-1475]`
**Booster — introspection:** `XGBoosterDumpModel(Ex)(WithFeatures)`, `XGBoosterGet/SetAttr`, `XGBoosterGetAttrNames`, `XGBoosterGet/SetStrFeatureInfo`, `XGBoosterFeatureScore` (feature importance), categories.
**Distributed (out of scope for a first rewrite):** `XGTracker*`, `XGCommunicator*`.

### Learner C++ interface (what the C API dispatches to) `[VERIFIED: LOCAL xgboost/include/xgboost/learner.h:73-176]`
`Configure()`, `UpdateOneIter(iter, dtrain)`, `BoostOneIter(iter, dtrain, grad, hess)`, `EvalOneIter(iter, data_sets, names)`, `Predict(...)`, `InplacePredict(...)`, `GetNumFeature()`, plus save/load. `UpdateOneIter` internally calls `GetGradient`, `BoostedRounds`, `ValidateDMatrix`, etc. `[VERIFIED: CODEGRAPH UpdateOneIter → GetGradient/BoostedRounds/ValidateDMatrix]`

### Python high-level API (thin over C ABI) `[VERIFIED: LOCAL xgboost/python-package/xgboost/core.py, training.py, sklearn.py]`
- `core.DMatrix` (core.py:652), `core.Booster` (core.py:1731). Booster public methods incl. `update` (2176), `boost` (2218), `eval` (2382), `predict` (2406), `inplace_predict` (2543), `save_model` (2770), `load_model` (2828), `get_dump` (2965), `get_score` (3017, feature importance).
- `training.train(...)` (training.py:53) and `training.cv(...)` (training.py:435) — the canonical training + cross-validation entry points.
- sklearn wrappers: `XGBModel` (sklearn.py:806), `XGBClassifier` (1698), `XGBRegressor` (1998), `XGBRanker` (2146), `XGBRFClassifier`/`XGBRFRegressor` (random forest variants). These expose `.fit()/.predict()/.predict_proba()/.feature_importances_`.

### R API `[VERIFIED: LOCAL xgboost/demo/kaggle-higgs/speedtest.R:32-45]`
`xgb.DMatrix(data, label=, weight=, missing=)`, `xgb.train(param, dmat, nrounds, watchlist)`, `xgb.save(bst, path)`; param list keys like `objective`, `eta`, `max_depth`, `eval_metric`, `nthread`, `scale_pos_weight`. (Modern R also exposes `xgboost()` sklearn-like front end.) `[INFERRED from R-package]`

### Registered algorithm names (the string-keyed extension points) `[VERIFIED: LOCAL bash grep XGBOOST_REGISTER_*]`
- **Objectives:** `reg:linear`, `reg:pseudohubererror`, `reg:absoluteerror`, `reg:expectileerror`, `reg:tweedie`, `count:poisson`, `binary:hinge`, `multi:softmax`, `multi:softprob`, `survival:aft`, `survival:cox`, `rank:pairwise`, `rank:ndcg`, `rank:map` (and `reg:squarederror`/`binary:logistic`/`binary:logitraw` registered via templated macros not caught by the simple grep — treat objective list as representative, verify the full set from `regression_obj.cc`). `[VERIFIED: LOCAL grep objective/]` `[UNVERIFIED: exact full objective set]`
- **Metrics:** `rmse`, `rmsle`, `mae`, `mape`, `mphe`, `logloss`, `error`, `merror`, `mlogloss`, `auc`, `aucpr`, `pre`, `map`, `ndcg`, `poisson-nloglik`, `gamma-nloglik`, `gamma-deviance`, `tweedie-nloglik`, `cox-nloglik`, `aft-nloglik`, `interval-regression-accuracy`, `quantile`, `expectile`, `ams`. `[VERIFIED: LOCAL grep metric/]`
- **Tree updaters:** `grow_colmaker` (exact), `grow_histmaker`, `grow_quantile_histmaker` (hist), `grow_gpu_hist`, `grow_gpu_approx`, `prune`, `refresh`, `sync`. `[VERIFIED: LOCAL grep tree/]`
- **Boosters:** `gbtree`, `gblinear`, `dart`. `[VERIFIED: CODEGRAPH GBTree/GBLinear/Dart extends GradientBooster]`

---

## Reference Rust Binding (rust-xgboost) design

`davechallis/rust-xgboost` (crate `xgboost` on crates.io) is a **FFI binding** to libxgboost, but its parameter layer is the idiomatic Rust builder API to mirror. `[VERIFIED: WEB https://github.com/davechallis/rust-xgboost]` `[VERIFIED: WEB https://docs.rs/xgboost/latest/xgboost/parameters/index.html]`

Module layout of its `parameters` module:
- Submodules: `parameters::tree`, `parameters::linear`, `parameters::dart`, `parameters::learning`.
- Top-level: `BoosterParameters` + `BoosterParametersBuilder`; `TrainingParameters` + `TrainingParametersBuilder`; enum `BoosterType`.
- `tree`: `TreeBoosterParameters` + `TreeBoosterParametersBuilder`, enums `TreeMethod`, `GrowPolicy`, `Predictor`, `ProcessType`.
- `linear`: `LinearBoosterParameters(+Builder)`; `dart`: `DartBoosterParameters(+Builder)`; `learning`: `LearningTaskParameters(+Builder)` (objective, eval metrics, base_score).
- Usage pattern: `TreeBoosterParametersBuilder::default().max_depth(2).eta(1.0).build()` → passed via `.booster_type(BoosterType::Tree(...))` into `BoosterParametersBuilder`. `[VERIFIED: WEB docs.rs/xgboost/parameters]`

Note: the existing `xgboost_rs::gpu::histogram::HistogramBuilder` already follows the same idiomatic consuming-builder style (`.force_global(...).native_i64_atomics(...).build(&matrix)`). `[VERIFIED: LOCAL src/bin/bench.rs:88-92]` The rewrite should keep this convention and layer the `parameters` module hierarchy above it.

---

## Core Module Map (C++ → proposed Rust modules)

Class hierarchy verified via CodeGraph (`extends`): `GBTree`, `GBLinear`, `Dart` → `GradientBooster` → `Model` + `Configurable`; `TreeUpdater` → `Configurable`; `Predictor` base with cpu/gpu/sycl impls. `[VERIFIED: CODEGRAPH GradientBooster extends Model/Configurable; TreeUpdater extends Configurable]`

| C++ subsystem | Key files/classes | Role | Proposed Rust module |
|---|---|---|---|
| **Data** | `src/data/*`, `include/xgboost/data.h`; `DMatrix`, `MetaInfo`, adapters (`DenseAdapter`, CSR/CSC), `GHistIndexMatrix`, `EllpackPage` | Data ingestion, quantile cuts, gradient index | `data::{dmatrix, meta, adapters, gradient_index, ellpack}` (ellpack already exists) |
| **Learner** | `src/learner.cc`, `learner.h`; `Learner` | Orchestrates objective→gbm→metric; owns model params, base_score | `learner` |
| **GBM** | `src/gbm/{gbtree,gblinear,gbm}.cc`, `gbtree_model.*`; `GradientBooster`, `GBTree`, `GBLinear`, `Dart`, `GBTreeModel` | Boosting strategy, tree ensemble container | `gbm::{gbtree, gblinear, dart, model}` |
| **Tree** | `src/tree/updater_*.cc`, `tree/hist/*`, `tree/gpu_hist/*`, `tree_model.*`, `param.*`, `split_evaluator.h`, `driver.h`; `TreeUpdater`, `RegTree` | Tree growth (hist/approx/exact/gpu_hist), split finding, pruning | `tree::{updater_hist, updater_approx, updater_exact, gpu_hist(exists), model, param, split_eval}` |
| **Objective** | `src/objective/*`; `ObjFunction`, `regression_loss.h`, `init_estimation.*` | grad/hess, `PredTransform`, `InitEstimation` (base_score) | `objective::{regression, binary, multiclass, ranking, survival, init_estimation}` |
| **Metric** | `src/metric/*`; `Metric` | Eval metrics | `metric::{elementwise, multiclass, rank, auc, survival}` |
| **Linear** | `src/linear/updater_*.cc`; coordinate/shotgun updaters | gblinear weight updates | `linear::updater` |
| **Predictor** | `src/predictor/*`; `Predictor`, `PredictionCacheEntry`, `PredictionContainer` | Batch/inplace/leaf/contrib prediction | `predictor::{cpu, gpu}` |
| **Common** | `src/common/*`; `HistogramCuts`, `deterministic.cuh`, quantile sketch, `Span`, `HostDeviceVector`, threading | Shared numerics/util | `common::{hist_util, quantile, deterministic(exists in quantiser), math}` |
| **C API** | `src/c_api/c_api.cc` | FFI surface | `c_api` (cdylib) |
| **Serialization** | JSON/UBJSON model IO via `SaveModel/LoadModel` | Model persistence & config | `model_io` (serde_json + a UBJSON impl) |

Registration is via `dmlc::Registry` string keys (`XGBOOST_REGISTER_OBJECTIVE`, `_METRIC`, `_TREE_UPDATER`, `_GBM`, `_PREDICTOR`). `[VERIFIED: CODEGRAPH GradientBoosterReg / PredictorReg registry macros]` In Rust this maps to a trait-object registry keyed by `&str` (matching the parameter strings) rather than macros.

Core flow: `Learner::UpdateOneIter` → objective `GetGradient` → `GradientBooster::DoBoost` → `TreeUpdater::Update` (builds histograms → split → grow `RegTree`) → prediction cache update. `[VERIFIED: CODEGRAPH UpdateOneIter → GetGradient; GBLinear::DoBoost → updater_->Update]`

---

## Oracle API Test Feasibility

**What is installed (verified):** `g++`, `cargo 1.97.0`, Rust. `[VERIFIED: LOCAL bash which g++; cargo --version]`
**What is NOT installed (verified):**
- Python `xgboost` package — `import xgboost` finds only the local source dir (no `__version__`); no compiled extension. `pip3 list` shows no xgboost/numpy/scipy/scikit. `[VERIFIED: LOCAL bash import/pip]`
- No built `libxgboost.so`/`.dylib` under `xgboost/lib` or `xgboost/build`. `[VERIFIED: LOCAL bash find]`
- No `R`/`Rscript`. `[VERIFIED: LOCAL bash which R]`
- No `cmake`, no `nvcc` (cannot build libxgboost or CUDA from source here). `[VERIFIED: LOCAL bash which cmake nvcc]`

**Consequence:** There is currently **no runnable reference XGBoost** to generate golden outputs. To create an oracle the planner must either (a) `pip install xgboost` (needs network + numpy; CPU wheel is sufficient for `hist`), or (b) build libxgboost from source (needs `cmake` install + `g++`; no GPU). Recommended: pin a specific upstream release (e.g. install the wheel matching the vendored `3.4.0` line) and record its exact version in the golden fixtures. `[INFERRED]`

**Datasets available in-repo** `[VERIFIED: LOCAL bash ls demo]`:
- `xgboost/demo/data/agaricus.txt.train` / `agaricus.txt.test` — the canonical LIBSVM binary-classification smoke dataset (small, deterministic). Best first oracle target.
- `xgboost/demo/data/regression/`, `veterans_lung_cancer.csv` (survival), `demo/kaggle-higgs/` scripts (Higgs — data file `data/training.csv` **not** vendored; must be downloaded), `demo/kaggle-otto/`, `demo/multiclass_classification/`.

**Golden-output approach (recommended):**
1. Freeze a reference XGBoost version; for each `(dataset, objective, params)` case, in real XGBoost: build DMatrix → `train` with fixed config → dump (i) predictions (`predict`, margin + prob), (ii) model as JSON (`save_model`), (iii) `get_dump`/text trees, (iv) `get_score` importances, (v) eval metric per round. Serialize to fixture files committed under `tests/fixtures/`.
2. Rust test loads the same dataset + params, runs its own train/predict, and asserts equality against the fixtures.
3. Comparison tolerance: exact for tree structure and integer counts; near-exact float tolerance for predictions (see determinism below). Prefer asserting the **quantised/rounded** intermediate (the repo already relies on exact i64 histogram equality — extend that philosophy).

**Determinism controls required for a fair/repeatable oracle** `[INFERRED from XGBoost params; VERIFIED base_score mechanism below]`:
- `nthread = 1` (parallel float reduction order differs otherwise).
- `tree_method = "hist"` (deterministic, CPU; avoids GPU nondeterminism and gpu-only availability).
- fixed `seed`, `subsample = 1`, `colsample_* = 1` (no sampling RNG divergence).
- fixed `base_score` (else XGBoost auto-estimates the intercept via `FitIntercept::InitEstimation` → `FitStump` + `PredTransform`, which the Rust port must replicate exactly to match). `[VERIFIED: LOCAL xgboost/src/objective/init_estimation.cc:18-45]`
- identical missing value handling and `max_bin`.

---

## Oracle Speed Test Feasibility

**Existing references:**
- The repo's own `bench` binary already does GPU histogram timing (ms/build, Gentry/s) and is the model for a Rust-side benchmark harness. `[VERIFIED: LOCAL src/bin/bench.rs:122-183]`
- Upstream `demo/kaggle-higgs/speedtest.R` times `xgb.train` for `threads = c(1,2,4,8,16)`, `nrounds=120`, `max_depth=6`, `eta=0.1`, `objective=binary:logitraw` on Higgs (350k rows). CodeGraph earlier flagged `xgboost.time` at speedtest.R:27. `[VERIFIED: LOCAL xgboost/demo/kaggle-higgs/speedtest.R:27-48]`
- `demo/kaggle-higgs/speedtest.py` is the Python analogue. `[VERIFIED: LOCAL bash ls demo/kaggle-higgs]`

**Fair comparison requirements:** identical dataset + identical params (`objective`, `eta`, `max_depth`, `nrounds`, `max_bin`, `tree_method=hist`); measure both single-thread and matched multi-thread; exclude data-load/quantile-cut time from the timed region or measure it separately (the Rust bench already separates upload/quantise from kernel time); warm up to exclude JIT/kernel compilation; report throughput per boosting round. Because the local machine has no reference XGBoost and no R, speed comparisons must run wherever a reference XGBoost is installed (e.g. the documented Kaggle GPU notebook, or a CPU box with `pip install xgboost`). `[VERIFIED: LOCAL KAGGLE.md]` `[INFERRED]`

---

## Dependencies & Versions

**Rust (current, locked)** `[VERIFIED: LOCAL Cargo.toml, Cargo.lock]`:
- `cubecl 0.10.0` (features `vulkan`; `cuda` optional) — GPU kernel authoring; `wgpu 29.0.4` / `naga 29.0.4` / `ash 0.38.0` pulled transitively.
- `anyhow 1.0.104`, `thiserror 2.0.19`, `bytemuck 1.25.1`.
- Rust edition 2024, toolchain `cargo 1.97.0`.

**Likely additions for a full rewrite** `[INFERRED / UNVERIFIED — planner to confirm]`:
- `serde` + `serde_json` for JSON model/config IO; a UBJSON encoder for `.ubj` parity (XGBoost defaults to UBJSON). `[VERIFIED: LOCAL c_api.cc:1546-1553 ext json/ubj]`
- `rayon` for CPU multithreading parity with XGBoost's OpenMP (note: threading changes float reduction order → determinism risk).
- `ndarray` (not currently a dep despite the task hint — **not present** in Cargo.lock) if a matrix abstraction is wanted; XGBoost uses its own `linalg`/`HostDeviceVector`.
- A CSV/LIBSVM parser for DMatrix-from-file (agaricus is LIBSVM format).

**Upstream:** vendored XGBoost `3.4.0-dev` (`CMakeLists.txt` VERSION 3.4.0). Build deps for producing an oracle: `cmake ≥ 3.18` (per CMakeLists min), C++ compiler; CUDA optional. `[VERIFIED: LOCAL xgboost/CMakeLists.txt:1]`

**Context7:** not queried in this pass (network-optional). For CubeCL 0.10 API specifics the authoritative source is the CubeCL docs/repo; recommend `npx ctx7@latest library "CubeCL"` then `docs` before writing new kernels. `[UNVERIFIED: Context7 not run]`

---

## Risks & Scope

| Risk | Trigger | Consequence | Prevention / Verification |
|---|---|---|---|
| **Scope explosion** | "same and ALL API" = ~90 C fns + Python/sklearn + R + 20+ objectives, 24 metrics, 8 updaters, 3 boosters, JSON/UBJSON IO, distributed | Multi-quarter, unbounded | Phase it: start with `hist` gbtree + `reg:squarederror`/`binary:logistic` + core DMatrix/Booster/train/predict/save; treat GPU/dart/ranking/survival/distributed as later phases. **Planner must define an MVP API subset.** |
| **Float bit-exactness** | Parallel/SIMD summation reorders adds | Predictions differ from oracle | Use the existing deterministic fixed-point quantiser philosophy; force `nthread=1`; compare with tolerance; assert on quantised intermediates. `[VERIFIED: LOCAL src/gpu/quantiser.rs]` |
| **Quantile sketch / histogram binning** | `max_bin` cut computation (WQ sketch) differs | Different splits → divergent trees | Port `common/quantile` + `HistogramCuts` faithfully; oracle-test cut points directly. `[VERIFIED: LOCAL tree updater includes gradient_index/hist_util]` |
| **Missing-value routing** | Default direction learning at splits | Wrong leaf assignment | Mirror XGBoost's default-direction + sparsity-aware split exactly; test with sparse agaricus. `[VERIFIED: LOCAL ellpack null_value handling]` |
| **base_score / intercept** | Auto-estimated when not set | Constant offset on all predictions | Replicate `FitIntercept::InitEstimation`→`FitStump`→`PredTransform`, or pin `base_score` in tests. `[VERIFIED: LOCAL init_estimation.cc:18-45]` |
| **Regularized split gain** | `lambda`, `alpha`, `gamma`, `min_child_weight` formula | Off-by-epsilon gain → different split chosen | Port `split_evaluator.h`/`param.h` exactly; unit-test gain against C++ values. `[VERIFIED: LOCAL tree/split_evaluator.h, param.h]` |
| **No local oracle** | No xgboost/R/cmake installed | Cannot generate/validate golden outputs | Install reference XGBoost (network) and pin version; commit fixtures. **Blocking.** `[VERIFIED: LOCAL bash]` |
| **Model format parity** | UBJSON default, JSON alt | Saved models not interoperable with real XGBoost | Implement JSON first (verifiable via `save_model('*.json')`), UBJSON second. `[VERIFIED: LOCAL c_api.cc:1546]` |
| **CubeCL API drift / GPU-only kernels** | Existing kernels assume Vulkan/CUDA i64/f64 | CPU-only path missing for portable training | Provide a CPU `hist` updater independent of CubeCL for the API/oracle tests; keep GPU as acceleration. `[VERIFIED: LOCAL Cargo.toml comment on WGSL i64/f64]` |

**Realistic scope statement:** A faithful, fully-API-complete, bit-for-bit XGBoost rewrite is a large multi-person, multi-quarter effort. A credible first deliverable is a CPU `hist` gbtree with a small objective/metric set, the core DMatrix/Booster/train/predict/save API, an idiomatic builder-pattern parameter module, and oracle API+speed tests against a pinned reference XGBoost on the agaricus dataset. `[INFERRED]`

---

## Open Questions (top blockers for planning)

1. **MVP API boundary.** Does "same and all API" mean the full C ABI + Python + R immediately, or a phased subset first? This single decision determines whether the plan is weeks or quarters. `[UNVERIFIED — needs user decision]`
2. **Oracle source.** May we `pip install xgboost` / `cmake`-build libxgboost (network access) to generate golden fixtures, and which exact reference version do we pin (match vendored 3.4.0 or a released 3.x wheel)? Without this the oracle tests cannot run. `[VERIFIED: no local oracle; decision needed]`
3. **CPU vs GPU as the primary training path.** The existing code is GPU-only (CubeCL). Should the rewrite implement a portable CPU `hist` updater as the primary/oracle path, with the existing GPU histogram as optional acceleration? `[INFERRED — needs confirmation]`
4. **Exactness bar.** Is the acceptance criterion bit-for-bit equality with XGBoost, or "within tolerance"? This dictates threading model, summation strategy, and how much of the deterministic-quantiser machinery must extend beyond the GPU path. `[UNVERIFIED]`
5. **Objective/metric/booster coverage for v1.** Which of the 20+ objectives / 24 metrics / {gbtree,gblinear,dart} must ship first? (Suggest `reg:squarederror`, `binary:logistic`, `multi:softprob` + `rmse`/`logloss`/`error`/`mlogloss`, gbtree/hist only.) `[UNVERIFIED]`

---

## Sources

- **Local files:** `Cargo.toml`, `Cargo.lock`, `KAGGLE.md`, `src/lib.rs`, `src/error.rs`, `src/gpu/{mod,ellpack,quantiser,histogram}.rs`, `src/reference.rs`, `src/main.rs`, `src/bin/bench.rs`, `tests/kernels.rs`.
- **Vendored upstream (xgboost 3.4.0-dev):** `include/xgboost/{c_api.h,learner.h,gbm.h,tree_updater.h,predictor.h}`, `src/c_api/c_api.cc`, `src/gbm/{gbtree.h,gblinear.cc}`, `src/objective/init_estimation.cc`, `src/tree/updater_*`, `python-package/xgboost/{core.py,training.py,sklearn.py}`, `demo/kaggle-higgs/speedtest.R`, `demo/data/`, `CMakeLists.txt`.
- **CodeGraph queries** (projectPath `xgboost/`): C API training/predict/save flow (`XGBoosterUpdateOneIter → Learner::UpdateOneIter`); class hierarchy (`GBTree/GBLinear/Dart → GradientBooster → Model/Configurable`, `TreeUpdater`, `Predictor`); registry macros.
- **Bash verification:** `which g++/R/cmake/nvcc`, `pip3 list`, `python3 -c import xgboost`, `cargo --version`, `cargo build --offline --lib`, `grep XGB_DLL`, `grep XGBOOST_REGISTER_*`.
- **Web:** [davechallis/rust-xgboost](https://github.com/davechallis/rust-xgboost), [docs.rs xgboost::parameters](https://docs.rs/xgboost/latest/xgboost/parameters/index.html).

---

## Confidence Assessment

- **HIGH:** Existing Rust port inventory and its GPU-only scope; C API surface and function list; core C++ module/class hierarchy; registered objective/metric/updater names; absence of local reference XGBoost/R/cmake/nvcc; Cargo dependency versions; rust-xgboost builder module layout; base_score auto-estimation mechanism. (Directly verified by files, CodeGraph, command output, or official docs.)
- **MEDIUM:** Exact-match determinism requirements (nthread/seed/tree_method/base_score) — grounded in XGBoost design and the repo's determinism philosophy but not exercised against a live oracle here; likely new dependencies (serde/rayon/UBJSON).
- **LOW:** Complete/exact set of registered objectives (grep missed templated registrations); Context7 CubeCL specifics (not queried); MVP scope boundary and exactness bar (require user decisions); feasibility timeline.
