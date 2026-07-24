---
title: XGBoost-in-Rust Rewrite — Phase 1 MVP (Rust-native high-level API, reg:squarederror + rmse, CPU hist gbtree, oracle tests)
status: draft
format: markdown
spec_version: 1
updated_at: 2026-07-21T00:00:00Z
source_requirements:
  - "User: rewrite xgboost in rust (same and all api, builder pattern, module design, oracle api test, oracle speed test)"
  - "User decision: no C API, no R — Rust-native high-level API only"
  - "User decision: phased MVP first"
  - "User decision: oracle via pinned pip install xgboost, committed golden fixtures"
  - "User decision: acceptance = predictions within ~1e-5 + exact tree structure/integers"
  - "User decision: v1 coverage = reg:squarederror + rmse only"
  - ".planning/plans/xgboost-rust-rewrite/research.md"
---

# XGBoost-in-Rust Rewrite — Phase 1 MVP Specification (draft)

Evidence labels: `[VERIFIED: CODEGRAPH ...]`, `[VERIFIED: LOCAL path:line]`, `[VERIFIED: WEB url]`, `[INFERRED: ...]`, `[UNVERIFIED: ...]`. Access date for all sources: 2026-07-21. Upstream reference: vendored XGBoost **3.4.0-dev** at `xgboost/`.

---

## 1. Context

The `xgboost_rs` crate today is a CubeCL GPU port of only XGBoost's `gpu_hist` histogram path (~1,137 LOC): a gradient quantiser, ELLPACK matrix, `HistogramBuilder`/engine, a CPU oracle, and a bench binary. There is **no** `DMatrix`, `Booster`, `Learner`, objective, metric, tree updater, predictor, or model IO. `[VERIFIED: LOCAL src/lib.rs:1-18]` `[VERIFIED: LOCAL src/gpu/mod.rs:1-6]`

The user wants a Rust rewrite of XGBoost with the **same public API** (surfaced as an idiomatic Rust-native API — **explicitly not** the C ABI and **not** R), an idiomatic **builder pattern** for parameters, a **module design** mirroring XGBoost core, an **oracle API test** (Rust output vs real XGBoost), and an **oracle speed test** (Rust vs C++ timing).

This document specifies **Phase 1 (MVP)** only: a working vertical slice that trains and predicts one objective end-to-end and is validated against a pinned real-XGBoost oracle. Later phases (additional objectives/metrics/boosters, approx/exact/GPU tree methods, UBJSON, sklearn wrappers, `cv`, ranking/survival) are out of scope here and listed under §2.

### Locked decisions (non-negotiable inputs)
1. **API surface:** Rust-native high-level API modeled on XGBoost's Python layer (`DMatrix`, `Booster`, `train`, `predict`, `save_model`/`load_model`, feature importance) + builder-pattern params. No C ABI, no R. `[VERIFIED: user decision]`
2. **Phasing:** MVP vertical slice first. `[VERIFIED: user decision]`
3. **Primary training path:** portable **CPU `hist`** gbtree updater is the oracle/primary path; the existing CubeCL GPU histogram is optional acceleration, not required for Phase 1. `[VERIFIED: user decision — assumption accepted]`
4. **Oracle source:** pinned `pip install xgboost` generates committed golden fixtures under `tests/fixtures/`. Python is a fixture generator only, never a shipped runtime dependency. `[VERIFIED: user decision]`
5. **Exactness bar:** predictions within absolute/relative tolerance `≤ 1e-5`; tree structure (node count, split feature, split condition bin, default direction), leaf counts, and all integer outputs must match the oracle **exactly**. `[VERIFIED: user decision]`
6. **v1 coverage:** objective `reg:squarederror` + metric `rmse` only; booster `gbtree`; tree method `hist`; single-thread deterministic config. `[VERIFIED: user decision]`

---

## 2. Scope and non-goals

### In scope (Phase 1)
- Data: `DMatrix` from dense row-major `f32` + `MetaInfo` (labels, optional weights, missing sentinel); `DMatrix` from LIBSVM file (agaricus) / CSR sparse; quantile-cut computation (`max_bin`) and a binned gradient index for the `hist` path.
- Objective: `reg:squarederror` (`grad = pred − label`, `hess = 1`), identity `PredTransform`, default metric `rmse`. `[VERIFIED: LOCAL src/objective/regression_obj.cu:184-197,249-251]`
- Intercept: `base_score` auto-estimation = (weighted) sample mean of labels. `[VERIFIED: LOCAL src/objective/regression_obj.cu:212-220]`
- Tree: `RegTree` model; CPU `hist` updater (histogram build → regularized split evaluation → depthwise growth with missing-value default direction, `gamma`/`min_child_weight` pruning); leaf weight `= eta · CalcWeight`. `[VERIFIED: LOCAL src/tree/param.h:224-265; src/tree/split_evaluator.h:74-99]`
- GBM: `GBTreeModel` ensemble container; `GBTree::do_boost` (one round); `Learner` orchestration (`configure`, `update_one_iter`, `eval_one_iter`, base_score init on iter 0).
- Predictor: CPU tree traversal producing margin and transformed value (identity for squared error); `predict` on a `DMatrix`.
- Metric: `rmse`.
- Parameters: builder-pattern `parameters` module mirroring `rust-xgboost` (`BoosterParameters`, `TreeBoosterParameters`, `LearningTaskParameters`, `TrainingParameters` + `*Builder`). `[VERIFIED: WEB docs.rs/xgboost/parameters]`
- High-level API: `train(params, dtrain, num_round, evals) -> Booster`; `Booster::{update, predict, eval, save_model, load_model, get_score, boosted_rounds, num_features}`.
- Model IO: JSON `save_model`/`load_model` round-trip (and interop-read of a real-XGBoost JSON model to the extent needed for the oracle test). UBJSON deferred.
- Oracle API test: Rust train/predict vs committed fixtures from pinned XGBoost, per §1.5 tolerance.
- Oracle speed test: Rust-vs-XGBoost timing harness on identical data/params (executes wherever a reference XGBoost is installed; Rust side always runnable).

### Non-goals (explicitly deferred to later phases)
- C ABI (`c_api`) and R bindings — **removed from the design entirely per user**.
- Objectives/metrics beyond `reg:squarederror`/`rmse`; `binary:logistic`, `multi:softprob`, ranking, survival, tweedie, etc.
- Boosters `gblinear`, `dart`; tree methods `approx`, `exact`, `gpu_hist`, `gpu_approx`.
- UBJSON model format; `cv`; sklearn-style wrappers; distributed/federated (`XGTracker*`, `XGCommunicator*`); categorical features; multi-target; sampling (`subsample`/`colsample_*` < 1); multi-threading determinism (Phase 1 is `nthread=1`).
- Bit-for-bit equality (user chose tolerance + exact structure).

---

## 3. Dependencies

**Current crate deps (locked):** `cubecl 0.10.0` (feature `vulkan`; `cuda` optional), `anyhow 1.0.104`, `thiserror 2.0.19`, `bytemuck 1.25.1`; edition 2024; toolchain `cargo 1.97.0`. `[VERIFIED: LOCAL Cargo.toml:1-17; Cargo.lock]`

**Proposed additions (planner to confirm exact versions via Context7/crates.io):**
- `serde` + `serde_json` — JSON model/config IO. `[INFERRED]`
- A LIBSVM/CSV parser (hand-rolled or a small crate) for `DMatrix` from agaricus. `[INFERRED]`
- Optional later: `rayon` for CPU parallelism (deferred — Phase 1 is single-thread for determinism). `[INFERRED]`
- **No** new GPU deps for Phase 1; CPU `hist` path must not require a CubeCL runtime. `[VERIFIED: LOCAL Cargo.toml:9-11 WGSL i64/f64 constraint]`

**External (test-time only, not a crate dep):** a pinned `xgboost` Python wheel to generate fixtures; exact version pinned in the fixture generator and recorded in fixture metadata. `[VERIFIED: user decision]`

**Upstream reference:** vendored XGBoost `3.4.0-dev`. `[VERIFIED: LOCAL xgboost/CMakeLists.txt VERSION 3.4.0]`

---

## 4. Typed contracts

Rust-flavored signatures (planner may refine names to Rust conventions; semantics are binding). All Phase-1 compute is `f64` accumulation over `f32` inputs unless stated, single-threaded.

```rust
// ---- data ----
pub struct MetaInfo { pub num_row: usize, pub num_col: usize,
    pub labels: Vec<f32>, pub weights: Option<Vec<f32>>, pub base_margin: Option<Vec<f32>> }
pub struct DMatrix { /* opaque: raw values + MetaInfo + optional cached gradient index */ }
impl DMatrix {
    pub fn from_dense(data: &[f32], nrow: usize, ncol: usize, missing: f32) -> Result<DMatrix>;
    pub fn from_libsvm(path: &Path) -> Result<DMatrix>;          // agaricus format
    pub fn set_labels(&mut self, y: &[f32]) -> Result<()>;
    pub fn num_row(&self) -> usize; pub fn num_col(&self) -> usize;
}

// ---- quantile cuts / gradient index (hist) ----
pub struct HistogramCuts { pub cut_ptrs: Vec<u32>, pub cut_values: Vec<f32>, pub min_values: Vec<f32> }
pub fn build_cuts(dmat: &DMatrix, max_bin: u32) -> Result<HistogramCuts>;
pub struct GHistIndex { pub index: Vec<u32>, pub row_ptr: Vec<usize>, pub cuts: HistogramCuts }
pub fn build_gradient_index(dmat: &DMatrix, cuts: &HistogramCuts) -> Result<GHistIndex>;

// ---- objective ----
pub struct GradientPair { pub grad: f32, pub hess: f32 }
pub trait Objective {
    fn get_gradient(&self, preds: &[f32], info: &MetaInfo, iter: i32) -> Vec<GradientPair>;
    fn pred_transform(&self, preds: &mut [f32]);
    fn init_estimation(&self, info: &MetaInfo) -> f32;   // base_score (margin space)
    fn default_metric(&self) -> &'static str;
}
// SquaredError: grad=(p-y)*w, hess=1*w; pred_transform = identity; init = mean(labels); default_metric="rmse"

// ---- tree ----
pub struct TrainParam { pub eta: f32, pub max_depth: u32, pub max_leaves: u32,
    pub min_child_weight: f32, pub reg_lambda: f32, pub reg_alpha: f32, pub gamma: f32,
    pub max_delta_step: f32, pub max_bin: u32, pub grow_policy: GrowPolicy, pub base_score: Option<f32> }
pub struct RegTree { /* nodes: split_feature:i32(-1=leaf), split_cond:f32, default_left:bool,
                        left:i32, right:i32, leaf_value:f32, sum_hess:f32 */ }
pub trait TreeUpdater { fn update(&mut self, gpair: &[GradientPair], dmat: &DMatrix, tree: &mut RegTree); }
// CalcWeight(G,H) = if H<=0 {0} else clamp(-thresholdL1(G,alpha)/(H+lambda), max_delta_step)
// CalcGain(G,H)   = if H<=0 {0} else Sqr(thresholdL1(G,alpha))/(H+lambda)  // == node root_gain
// CalcSplitGain   = CalcGain(GL,HL)+CalcGain(GR,HR); -inf unless HL,HR>0 AND >=min_child_weight (hess)
// loss_chg        = CalcSplitGain(GL,HL, GR,HR) - parent.root_gain   // THE decision & stored/importance "gain"
//   [VERIFIED: LOCAL xgboost/src/tree/hist/evaluate_splits.h:290-293,298-301; ApplyTreeSplit:416-420]
// split validity  = a node splits iff best loss_chg > kRtEps (=1e-6f) AND loss_chg >= gamma(min_split_loss)
//   [VERIFIED: LOCAL xgboost/src/tree/hist/expand_entry.h:124-128; base.h:309 kRtEps; param.h:181 gamma alias]
// split_pt / dir  = forward pass (missing->right): split_pt=cut_val[i], default_left=false, ALWAYS runs;
//                   backward pass (missing->left): split_pt=NumericBinLowerBound(cut_ptr,cut_val,fidx,i),
//                   default_left=true, runs ONLY IF the feature has missing (SplitContainsMissingValues).
//                   Missing is IMPLICIT: parent - Σ(non-missing bins); there is NO dedicated missing bin in the split scan.
//   [VERIFIED: LOCAL xgboost/src/tree/hist/evaluate_splits.h:281-309,369-372]
// leaf weight     = eta * CalcWeight(Gnode,Hnode)  [VERIFIED: LOCAL evaluate_splits.h:405-418]

// ---- gbm / learner ----
pub struct GBTreeModel { pub trees: Vec<RegTree>, pub base_score: f32, pub num_feature: usize }
pub struct Learner { /* objective + gbtree model + params + metric */ }
impl Learner {
    pub fn configure(&mut self) -> Result<()>;
    pub fn update_one_iter(&mut self, iter: i32, dtrain: &DMatrix) -> Result<()>;
    pub fn eval_one_iter(&mut self, iter: i32, evals: &[(&DMatrix, &str)]) -> Result<String>;
    pub fn predict(&self, dmat: &DMatrix, output_margin: bool) -> Result<Vec<f32>>;
}

// ---- metric ----
pub trait Metric { fn name(&self) -> &str; fn eval(&self, preds:&[f32], info:&MetaInfo) -> f64; }
// Rmse = sqrt( sum_i w_i*(p_i - y_i)^2 / sum_i w_i )

// ---- parameters (builder pattern) ----
pub struct TreeBoosterParametersBuilder { /* eta,max_depth,lambda,alpha,gamma,min_child_weight,max_bin,... */ }
// TreeBoosterParametersBuilder::default().max_depth(6).eta(0.3).build() -> TreeBoosterParameters
pub struct BoosterParametersBuilder { /* booster_type, learning_params, ... */ }
pub struct TrainingParametersBuilder { /* dtrain, num_round(boost_rounds), evals, ... */ }

// ---- high-level API ----
pub fn train(params: &BoosterParameters, dtrain: &DMatrix, num_round: u32,
             evals: &[(&DMatrix, &str)]) -> Result<Booster>;
pub struct Booster { /* wraps Learner */ }
impl Booster {
    pub fn update(&mut self, iter: i32, dtrain: &DMatrix) -> Result<()>;
    pub fn predict(&self, dmat: &DMatrix) -> Result<Vec<f32>>;
    pub fn eval(&self, dmat: &DMatrix, name: &str) -> Result<String>;
    pub fn save_model(&self, path: &Path) -> Result<()>;       // JSON
    pub fn load_model(path: &Path) -> Result<Booster>;         // JSON
    pub fn get_score(&self, importance_type: &str) -> Result<HashMap<String, f64>>;
    pub fn boosted_rounds(&self) -> usize;
    pub fn num_features(&self) -> usize;
}
```

Error type extends the existing `xgboost_rs::Error` (`thiserror`). `[VERIFIED: LOCAL src/error.rs:5-26]`

---

## 5. Failure-isolated behavioral specifications

Each spec has one behavioral responsibility, one trigger, an explicit dependency boundary, exact input/output or typed error, a deterministic observable result, and independently testable acceptance criteria. Given/When/Then acceptance tests live in §6.

> **Status of every spec below: `draft`.**

### Data
- **SPEC-D01 — DMatrix from dense.** Build `DMatrix` + `MetaInfo` from row-major `&[f32]` with a `missing` sentinel; NaN and `missing` become structural missing. In: `(&[f32], nrow, ncol, missing)`. Out: `DMatrix` or a **new dedicated** `Error::DMatrixShape` (do **not** reuse the ELLPACK-worded `Error::MatrixShape`). Dep: none. Source req: §2 data. `[VERIFIED: LOCAL src/error.rs:7-8 MatrixShape is ELLPACK-specific]`
- **SPEC-D02 — DMatrix from LIBSVM.** Parse agaricus LIBSVM (`label idx:val ...`, 1-based indices) into CSR + labels. In: `&Path`. Out: `DMatrix` (sparse) or `Error`. Dep: file IO. `[VERIFIED: LOCAL xgboost/demo/data/agaricus.txt.train]`
- **SPEC-D03 — Quantile cuts.** Compute per-feature cut points for `max_bin` reproducing XGBoost's `hist` `HistogramCuts`. **Two paths, both required for parity:** (a) low-cardinality/dense path — when a feature's distinct-value count ≤ `max_bin`, cuts are the sorted distinct values (upper bounds); (b) **weighted-quantile-sketch path** (`common::WQuantileSketch`/`HostSketchContainer`) — when distinct count > `max_bin` (the continuous regression fixture), cuts are the sketch's `max_bin` quantile boundaries. The MVP MUST implement path (b), not only (a), or the continuous fixture will diverge. Also emit `min_values` per feature. In: `(&DMatrix, max_bin)`. Out: `HistogramCuts`. Dep: SPEC-D01/D02. **Isolation:** cut computation is the single most likely source of tree divergence; it is its own spec and its own oracle assertion (AC-D03) that MUST pass before any training-parity test is trusted. `[VERIFIED: research Risks — quantile sketch; LOCAL xgboost/src/common/quantile*]`
- **SPEC-D04 — Gradient index (binning).** Map each **non-missing** feature value to its bin via `HistogramCuts` (upper-bound binary search within the feature's cut slice); missing/NaN cells are simply **absent** from the row's entries (CSR-style), so histogram accumulation naturally skips them and the split scan reconstructs missing implicitly (SPEC-T03). Produce `GHistIndex { index, row_ptr, cuts }`. In: `(&DMatrix, &HistogramCuts)`. Out: `GHistIndex`. Dep: SPEC-D03.

### Objective & intercept
- **SPEC-O01 — SquaredError gradient.** `grad_i = (pred_i − label_i)·w_i`, `hess_i = 1·w_i`; `w_i=1` if no weights. In: `(&[f32] preds, &MetaInfo, iter)`. Out: `Vec<GradientPair>`. Dep: none. `[VERIFIED: LOCAL regression_obj.cu:184-197]`
- **SPEC-O02 — SquaredError pred transform & default metric.** `pred_transform` = identity; `default_metric()` = `"rmse"`. `[VERIFIED: LOCAL regression_obj.cu:202-210,200]`
- **SPEC-O03 — base_score init estimation.** With no explicit `base_score`, estimate intercept = (weighted) sample mean of labels (identity link ⇒ margin = mean). In: `&MetaInfo`. Out: `f32`. Dep: none. **Isolation:** a wrong intercept shifts every prediction by a constant; separately tested. `[VERIFIED: LOCAL regression_obj.cu:212-220]`

### Tree
- **SPEC-T01 — TrainParam + validation.** Hold and validate `eta,max_depth,max_leaves,min_child_weight,reg_lambda,reg_alpha,gamma,max_delta_step,max_bin,grow_policy,base_score`; enforce ranges. Dep: none. `[VERIFIED: LOCAL src/tree/param.h]`
- **SPEC-T02 — RegTree model.** Node array with split feature/condition/default-direction/children/leaf value/sum_hess; add-node, mark-leaf, `predict_one(row)` traversal with missing routing. Dep: none. `[VERIFIED: CODEGRAPH RegTree]`
- **SPEC-T03 — Node histogram build.** For a node's row set, accumulate per-(feature,bin) `(Σgrad, Σhess)` in `f64` from `GHistIndex` over **non-missing** cells only, AND compute the node's total stats `parent.stats = Σ gpair over all rows in the node` (including missing rows). The per-feature missing stats used by the split scan are **implicit**: `missing = parent.stats − Σ(that feature's bins)` — there is NO dedicated missing bin in the scan. In: `(&GHistIndex, rows, &[GradientPair])`. Out: `NodeHist { bins, parent_stats }`. Dep: SPEC-D04. Mirrors existing CPU oracle philosophy. `[VERIFIED: LOCAL src/reference.rs cpu_histogram; xgboost/src/tree/hist/evaluate_splits.h:281-309]`
- **SPEC-T04 — Split evaluation.** Given a node histogram + `parent.root_gain = CalcGain(parent.stats)`, scan each feature's bins, computing `loss_chg = CalcSplitGain(GL,HL, GR,HR) − parent.root_gain` where `CalcSplitGain = CalcGain(GL,HL)+CalcGain(GR,HR)` and returns −∞ unless `HL,HR>0 ∧ ≥min_child_weight` (hess, not weight). Track best `(feature, split_pt, default_left, loss_chg, left_sum, right_sum)`. A node is **valid to split** iff best `loss_chg > kRtEps (=1e-6)` **AND** `loss_chg ≥ gamma`; else the node becomes a leaf (this kRtEps gate is separate from gamma and MUST be applied, or spurious tiny-gain splits diverge from XGBoost). In: `(&NodeHist, parent_gain, &TrainParam)`. Out: `Option<SplitCandidate>`. Dep: SPEC-T03. `[VERIFIED: LOCAL param.h:224-265; split_evaluator.h:74-99; evaluate_splits.h:281-309; expand_entry.h:124-128; base.h:309]`
- **SPEC-T05 — Split-point convention & missing default direction.** Enumerate each numeric feature with XGBoost's two-pass sparsity-aware scan: **forward** (`left_sum += bin i`, `right = parent − left`, missing→right): `split_pt = cut_val[i]`, `default_left = false`; **runs always**. **backward** (missing→left): `split_pt = NumericBinLowerBound(cut_ptr, cut_val, fidx, i)`, `default_left = true`; **runs only if the feature has missing** (`parent.stats ≠ Σ non-missing bins`). Keep whichever pass yields the higher `loss_chg`; tie-break matches XGBoost (`SplitEntry::Update` keeps the first/existing on non-strict improvement ⇒ forward/`default_left=false` wins ties). Store `split_pt` and `default_left` on the node. In: node hist + parent stats. Out: `default_left: bool`, `split_pt: f32`. Dep: SPEC-T04. **Isolation:** sparse-data routing + split-point convention is a distinct divergence source (agaricus is sparse). `[VERIFIED: LOCAL xgboost/src/tree/hist/evaluate_splits.h:281-309,369-372]`
- **SPEC-T06 — Tree growth driver (depthwise).** Grow a `RegTree` up to `max_depth` (grow_policy=`depthwise` for MVP): from root, per level, build hists (parent−child subtraction optimization allowed but must stay within the ≤1e-5 bar, else build directly), evaluate splits, and apply a node's best split **only when it is valid per SPEC-T04 (`loss_chg > kRtEps AND loss_chg ≥ gamma`)** — otherwise the node is finalized as a leaf. Store on each internal node the split `loss_chg` (parent-subtracted gain) and node `sum_hess` (cover) for SPEC-A04. Assign leaf weight `= eta·CalcWeight(G,H)` to terminal nodes. In: `(&[GradientPair], &DMatrix/&GHistIndex, &TrainParam)`. Out: `RegTree`. Dep: SPEC-T02..T05. `[VERIFIED: CODEGRAPH driver.h; UpdateOneIter flow; evaluate_splits.h ApplyTreeSplit:416-420]`

### GBM / Learner
- **SPEC-G01 — GBTreeModel container.** Append trees; hold `base_score`, `num_feature`; iterate trees for prediction. Dep: SPEC-T02.
- **SPEC-G02 — GBTree do_boost (one round).** iter n: get gradients from current margin preds, grow one `RegTree`, append, update cached margin preds by `+leaf_value` per row. In: `(&DMatrix, iter, &mut GBTreeModel, &Objective, &TrainParam)`. Dep: SPEC-O01, SPEC-T06, SPEC-G01. `[VERIFIED: CODEGRAPH DoBoost → updater Update]`
- **SPEC-L01 — Learner orchestration.** `configure`; on iter 0 initialize `base_score` (SPEC-O03) unless user-set and seed margin cache; `update_one_iter` = one `do_boost`; `eval_one_iter` formats `"[iter] name-metric:value"`. In/out per §4. Dep: SPEC-G02, SPEC-O03, SPEC-M01. `[VERIFIED: LOCAL xgboost/include/xgboost/learner.h:73-176]`

### Predictor & metric
- **SPEC-P01 — CPU predictor.** For each row, `margin = base_score + Σ_tree leaf_value(row)`; `value = pred_transform(margin)` (identity). Support `output_margin`. In: `(&GBTreeModel, &DMatrix, output_margin)`. Out: `Vec<f32>`. Dep: SPEC-T02, SPEC-G01, SPEC-O02. `[VERIFIED: CODEGRAPH Predictor]`
- **SPEC-M01 — RMSE metric.** `sqrt(Σ w_i (p_i−y_i)^2 / Σ w_i)`, `w_i=1` default. In: `(&[f32], &MetaInfo)`. Out: `f64`. Dep: none. `[VERIFIED: LOCAL grep metric/ rmse]`

### Parameters (builder pattern)
- **SPEC-B01 — TreeBoosterParameters(+Builder).** Consuming builder for tree params with XGBoost defaults (`eta=0.3,max_depth=6,lambda=1,alpha=0,gamma=0,min_child_weight=1,max_bin=256`), `.build()` validates → `TreeBoosterParameters`. Mirrors `rust-xgboost`. Dep: SPEC-T01. `[VERIFIED: WEB docs.rs/xgboost/parameters]`
- **SPEC-B02 — LearningTaskParameters(+Builder).** objective (`reg:squarederror`), eval metric(s), optional `base_score`, `seed`. Dep: none. `[VERIFIED: WEB docs.rs/xgboost/parameters::learning]`
- **SPEC-B03 — BoosterParameters + TrainingParameters(+Builder).** Compose booster type (Tree) + learning params; training params carry `num_round`, `evals`. `.build()` → typed structs consumable by `train`. Dep: SPEC-B01, SPEC-B02. `[VERIFIED: WEB docs.rs/xgboost/parameters]`

### High-level API & model IO
- **SPEC-A01 — `train` entry point.** `train(params, dtrain, num_round, evals)` builds a `Learner`, runs `num_round` `update_one_iter`, evaluating `evals` each round, returns `Booster`. Dep: SPEC-L01, SPEC-B03. `[VERIFIED: LOCAL xgboost/python-package/xgboost/training.py:53]`
- **SPEC-A02 — Booster methods.** `update/predict/eval/boosted_rounds/num_features` delegate to `Learner`. Dep: SPEC-L01, SPEC-P01.
- **SPEC-A03 — JSON model save/load round-trip.** `save_model(*.json)` then `load_model` yields byte-identical re-save and identical predictions; JSON layout compatible enough to load a real-XGBoost `reg:squarederror` JSON model for the oracle read test. Dep: SPEC-G01, serde_json. **Isolation:** serialization defects are independent of training. `[VERIFIED: LOCAL xgboost/src/c_api/c_api.cc:1546 json ext]`
- **SPEC-A04 — Feature importance (`get_score`).** Per feature (keyed `f{idx}`), compute: `weight` = number of splits on that feature; `gain`/`total_gain` = Σ (and mean) of the split **`loss_chg`** stored at each split node (the parent-subtracted gain, SPEC-T04 — NOT the raw child-sum `CalcSplitGain`); `cover`/`total_cover` = Σ (and mean) of node `sum_hess` at each split. This matches XGBoost's stored `RegTree` split gain (`ApplyTreeSplit` passes `candidate.split.loss_chg`). Dep: SPEC-G01, SPEC-T06 (nodes must carry `loss_chg`+`sum_hess`). `[VERIFIED: LOCAL xgboost/python-package/xgboost/core.py:3017 get_score; xgboost/src/tree/hist/evaluate_splits.h:416-420]`

### Oracle tests
- **SPEC-X01 — Oracle fixture generator.** A pinned Python script (`tests/fixtures/gen.py`) using a fixed `xgboost==<pinned>` version trains `reg:squarederror` on committed datasets under a fixed deterministic config (`nthread=1, tree_method=hist, seed=0, subsample=1, colsample_*=1, base_score=0.5, max_bin, eta, max_depth, num_round`) and emits: per-round `rmse`, final predictions (margin), model JSON, text tree dump, and `get_score` importances. Metadata records the exact xgboost version + config. Dep: external xgboost (test-time). `[VERIFIED: user decision; research Determinism controls]`
- **SPEC-X02 — Oracle API test (Rust vs fixtures).** Rust trains/predicts with the identical config and asserts: predictions within `≤1e-5` (abs or rel); tree structure (node count, split feature, split bin/condition, default direction) and integer outputs **exact**; per-round `rmse` within `≤1e-5`; importance keys/counts exact. In: committed fixtures. Out: pass/fail. Dep: SPEC-A01, SPEC-P01, SPEC-M01, SPEC-A04, SPEC-D03. `[VERIFIED: user decision — exactness bar]`
- **SPEC-X03 — Oracle speed test.** A bench (extend `src/bin/bench.rs` style) times Rust `train` vs XGBoost `train` on identical data/params (single-thread `hist`), excludes data-load/cut time from the timed region, warms up, reports ms/round + rows/s. Rust side always runs; the XGBoost side runs where a reference install exists (documented, e.g. Kaggle/CPU box). In: dataset+params. Out: timing report. Dep: SPEC-A01. `[VERIFIED: LOCAL src/bin/bench.rs:122-183; xgboost/demo/kaggle-higgs/speedtest.R:27-48]`

---

## 6. Acceptance scenarios (Given/When/Then — Red-test seeds)

- **AC-D01** *(SPEC-D01)*: Given a 3×2 f32 array with one NaN; When `from_dense(..., missing=NaN)`; Then `num_row=3,num_col=2` and the NaN cell is structurally missing.
- **AC-D03** *(SPEC-D03)*: Given a committed small dataset; When `build_cuts(max_bin=16)`; Then `cut_ptrs`/`cut_values` equal the XGBoost `get_quantile_cut` fixture exactly (integer ptrs exact, values ≤1e-5).
- **AC-O01** *(SPEC-O01)*: Given preds `[0.5,0.5]`, labels `[1.0,0.0]`, no weights; When `get_gradient`; Then gpairs `[(-0.5,1.0),(0.5,1.0)]`.
- **AC-O03** *(SPEC-O03)*: Given labels `[1,0,0,1]`; When `init_estimation`; Then `0.5` (±1e-6).
- **AC-T04** *(SPEC-T04)*: Given a node hist with parent stats `(G,H)` and a known best split; When `evaluate_split(lambda=1,gamma=0)`; Then the returned `loss_chg = [GL²/(HL+1)+GR²/(HR+1)] − G²/(H+1)` (±1e-6) and the chosen `(feature, split_pt)` match hand-computed values. Also: a hist whose best `loss_chg ≤ 1e-6` returns `None` (no split — kRtEps gate), and with `gamma=5` a split whose `loss_chg=3` returns `None`.
- **AC-T05** *(SPEC-T05)*: Given a feature with missing rows where routing missing left yields strictly higher `loss_chg`; When enumerating; Then `default_left==true` and `split_pt==NumericBinLowerBound(...)` (backward pass); the symmetric no-benefit case gives `default_left==false`, `split_pt==cut_val[i]` (forward), and a feature with **no** missing never runs the backward pass. Ties resolve to `default_left=false`.
- **AC-T06** *(SPEC-T06)*: Given tiny data, `max_depth=1,eta=1`; When grow one tree; Then node count, split feature, `split_pt`, and `default_left` match the XGBoost single-tree dump **exactly** (no spurious tiny-gain nodes — kRtEps/gamma gate) and leaf weights match ≤1e-5.
- **AC-A04** *(SPEC-A04)*: Given a 2-tree model with known per-split `loss_chg` and node `sum_hess`; When `get_score("gain")`/`get_score("weight")`/`get_score("cover")`; Then `weight["f0"]` = split count, `gain["f0"]` = Σ stored `loss_chg` (not raw child-sum), `cover["f0"]` = Σ `sum_hess`; keys are `f{idx}`.
- **AC-P01** *(SPEC-P01)*: Given a 1-tree model; When `predict(output_margin=true)`; Then `margin = base_score + leaf` per row.
- **AC-M01** *(SPEC-M01)*: Given preds/labels; When `rmse`; Then equals `sqrt(mean((p−y)^2))` (±1e-9).
- **AC-B01** *(SPEC-B01)*: Given `TreeBoosterParametersBuilder::default().max_depth(6).eta(0.3).build()`; Then defaults `lambda=1,gamma=0,min_child_weight=1,max_bin=256` are present.
- **AC-A03** *(SPEC-A03)*: Given a trained Booster; When `save_model("m.json")` then `load_model`; Then re-save is byte-identical and predictions match exactly.
- **AC-X02** *(SPEC-X02, headline)*: Given committed XGBoost fixtures for agaricus + a small regression set (`num_round=10,max_depth=6,eta=0.3,max_bin=256`); When Rust trains/predicts the same config; Then predictions ≤1e-5, tree structure exact, per-round rmse ≤1e-5, importances exact.
- **AC-X03** *(SPEC-X03)*: Given identical data/params; When the speed harness runs; Then it emits comparable ms/round for Rust and (where available) XGBoost without errors.

---

## 7. Impact scope

All Phase-1 work is **greenfield additive** — no existing public symbol changes; the GPU `histogram`/`quantiser`/`ellpack` modules are untouched and remain optional. `[VERIFIED: LOCAL src/lib.rs:14-18]`

| Area | New Rust module | Classification |
|---|---|---|
| Data | `data::{dmatrix, meta, adapters, cuts, gradient_index}` | local (new) |
| Objective | `objective::{squared_error, init_estimation}` | local (new) |
| Tree | `tree::{param, model, hist_updater, split_eval}` | local (new) |
| GBM/Learner | `gbm::{gbtree, model}`, `learner` | local (new) |
| Predictor/Metric | `predictor::cpu`, `metric::rmse` | local (new) |
| Params | `parameters::{tree, learning, booster, training}` | external/public API (new) |
| High-level API | `api::{train, booster}`, `model_io::json` | external/public API (new) |
| Tests | `tests/oracle_api.rs`, `tests/fixtures/*`, `src/bin/oracle_speed.rs` | operational |

Existing `tests/kernels.rs` and `src/bin/bench.rs` remain valid (GPU path). `lib.rs` gains new `pub mod` exports only. `[VERIFIED: LOCAL src/lib.rs]`

---

## 8. Compatibility and migration

- No migration of existing code; only additive `pub mod` declarations in `src/lib.rs`. `[VERIFIED: LOCAL src/lib.rs:14-18]`
- New `serde`/`serde_json` and a LIBSVM parser are additive deps; the CPU `hist` path must build and unit-test **without** a GPU/CubeCL runtime (no GPU confirmed here). `[VERIFIED: research Oracle feasibility]`
- **Test execution environments (do not overstate what runs here):** (i) all Rust unit/logic tests (`cargo test --lib`) and the Rust-only speed harness run offline on this box; (ii) the **oracle fixture tests** (`oracle_cuts`, `oracle_tree`, `oracle_api` → AC-D03/AC-T06/AC-X02) require the committed golden fixtures, which are produced **once** by a network `pip install xgboost==<pinned>` fixture generator — they are **environment-gated until those fixtures are committed**, not runnable from a bare checkout on this box; (iii) the **XGBoost side of the speed test** (AC-X03) runs only where a reference XGBoost is installed. `[VERIFIED: research.md:126-133 no local oracle]`
- `GradientPair` is reused as an identical `#[derive(Clone,Copy,Debug,Default,PartialEq)]` plain struct (`grad:f32,hess:f32`); it compiles and is usable by CPU code without any GPU device. (The existing `gpu::GradientPair` is **not** `#[repr(C)]` and **not** `bytemuck::Pod`; do not assume either for CPU use.) `[VERIFIED: LOCAL src/gpu/mod.rs:10-15]`
- JSON model format targets read-compat with real XGBoost `reg:squarederror` models for the oracle read test; full cross-version model compatibility (UBJSON, all objectives) is deferred. `[INFERRED]`
- Builder-pattern parameter names/defaults follow `rust-xgboost` conventions for familiarity. `[VERIFIED: WEB docs.rs/xgboost/parameters]`

---

## 9. Risks and open questions

| Risk | Consequence | Mitigation / test |
|---|---|---|
| Quantile-cut mismatch (SPEC-D03) | Different bins → divergent trees | Oracle-assert cut points directly (AC-D03) before training tests. `[VERIFIED: research]` |
| Missing-value routing (SPEC-T05) | Wrong leaves on sparse agaricus | Dedicated spec + sparse-data acceptance test (AC-T05). `[VERIFIED: research]` |
| base_score estimation (SPEC-O03) | Constant prediction offset | Pin `base_score=0.5` in fixtures **and** test estimator separately (AC-O03). `[VERIFIED: LOCAL regression_obj.cu:212-220]` |
| Float summation order | >1e-5 drift vs oracle | `nthread=1`, `f64` histogram accumulation, tolerance bar (not bit-exact). `[VERIFIED: user decision]` |
| Parent−child hist subtraction | Small numeric drift | Optional; if used must stay within tolerance, else build each node directly. `[INFERRED]` |
| JSON schema drift vs XGBoost | Oracle read test fails | Phase 1 only needs `reg:squarederror` gbtree JSON subset; pin the xgboost version. `[INFERRED]` |

**Open questions (need resolution during planning/impl):**
1. Exact XGBoost `hist` cut algorithm for the committed datasets — sorted-distinct vs weighted-quantile sketch threshold (`max_bin` vs cardinality). Resolve by inspecting `common/quantile` + comparing to `get_quantile_cut` fixture. `[UNVERIFIED]`
2. Pinned xgboost version for fixtures (match vendored 3.4.0 line vs latest 3.x wheel). `[UNVERIFIED — planner/impl to pin]`
3. Minimal JSON model schema fields required for round-trip + real-model read. `[UNVERIFIED]`
4. Whether `grow_policy=depthwise` alone suffices for Phase 1 oracle parity (lossguide deferred). `[INFERRED yes]`

---

## 10. Traceability and sources

**Spec → evidence:**
- SPEC-O01/O02/O03 ← `xgboost/src/objective/regression_obj.cu:184-220,249-251` `[VERIFIED: CODEGRAPH]`
- SPEC-T04 ← `xgboost/src/tree/param.h:224-265`, `xgboost/src/tree/split_evaluator.h:74-99` `[VERIFIED: CODEGRAPH]`
- SPEC-L01 ← `xgboost/include/xgboost/learner.h:73-176` `[VERIFIED: LOCAL]`
- SPEC-A01 ← `xgboost/python-package/xgboost/training.py:53` `[VERIFIED: LOCAL]`
- SPEC-A04 ← `xgboost/python-package/xgboost/core.py:3017` `[VERIFIED: LOCAL]`
- SPEC-B01..B03 ← `docs.rs/xgboost/parameters` `[VERIFIED: WEB]`
- SPEC-T03 ← `src/reference.rs cpu_histogram` (existing CPU oracle) `[VERIFIED: LOCAL]`
- SPEC-X03 ← `src/bin/bench.rs:122-183`, `xgboost/demo/kaggle-higgs/speedtest.R:27-48` `[VERIFIED: LOCAL]`

**Research report:** `.planning/plans/xgboost-rust-rewrite/research.md`.

**Pending TreeFinder update:** this document is authored locally; if the TreeFinder MCP index is writable it should be upserted as collection/document `xgboost-rust-rewrite/SPEC` (draft). Until then treat this file as the authoritative draft. `[UNVERIFIED: TreeFinder target not yet confirmed]`
