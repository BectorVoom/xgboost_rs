---
title: XGBoost-in-Rust Rewrite — Phase 1 MVP Implementation Plan (goal-backward, strict TDD)
status: draft
plan_version: 1
updated_at: 2026-07-21T00:00:00Z
source_spec: .planning/plans/xgboost-rust-rewrite/SPEC.md
source_research: .planning/plans/xgboost-rust-rewrite/research.md
upstream_reference: xgboost/ (vendored XGBoost 3.4.0-dev)
toolchain: cargo 1.97.0, edition 2024
scope: Phase 1 MVP only (reg:squarederror + rmse, CPU hist gbtree, oracle tests)
---

# Phase 1 MVP Implementation Plan

This plan is derived **goal-backward** from SPEC §6 acceptance scenarios (headline
**AC-X02**) and decomposed into 28 independently executable, strictly-TDD tasks.
Every task is `Red → Green → Refactor` and maps to one primary SPEC ID (a few
tightly-coupled companions are named where a contract is inseparable). No
production code is written here; this document is the executable instruction set
for the implementer.

All work is **greenfield additive** (SPEC §7): only new `pub mod` lines are added
to `src/lib.rs`, new files under `src/…`, `tests/…`, `tests/fixtures/…`, and new
`serde`/`serde_json`/LIBSVM-parser deps in `Cargo.toml`. Existing GPU modules
(`gpu::*`, `reference.rs`, `bench.rs`, `tests/kernels.rs`) are untouched. The CPU
`hist` path must **build and test without any CubeCL/GPU runtime** — verified: the
crate already builds offline (`cargo build --offline --lib → Finished`,
research.md:42). `gpu::GradientPair` is a plain
`#[derive(Clone,Copy,Debug,Default,PartialEq)]` struct (verified `src/gpu/mod.rs:10-15`
— NOT `#[repr(C)]`, NOT `bytemuck::Pod`) that *compiles* and is usable by CPU code
without a device; only kernel *execution* needs Vulkan/CUDA.

---

## 0. Goal-backward derivation (why these tasks, in this order)

The headline observable success condition is **AC-X02**: Rust train/predict on
agaricus + a small regression set, with the identical pinned config, matches
committed XGBoost fixtures — predictions ≤ 1e-5, per-round `rmse` ≤ 1e-5, tree
structure/integers exact, importances exact.

Working backward from AC-X02, the test needs, in reverse dependency order:

1. A committed **oracle fixture set** from a pinned real XGBoost (SPEC-X01) — which
   requires a **`pip install xgboost==<pinned>` prerequisite** (no local oracle
   exists; research.md:126-133).
2. A Rust **`train` entry point** returning a `Booster` (SPEC-A01/A02) that runs
   `num_round` boosting iterations and predicts.
3. `train` needs a **`Learner`** (SPEC-L01) orchestrating objective → gbm →
   metric, plus **base_score init** (SPEC-O03) and per-round **rmse** (SPEC-M01).
4. The `Learner` needs a **`GBTree` boosting round** (SPEC-G02) over a
   **`GBTreeModel`** (SPEC-G01), each round growing one tree via the **hist tree
   driver** (SPEC-T06).
5. The driver needs the **split-gain evaluator** (SPEC-T04) + **missing-value
   default direction** (SPEC-T05) over **node histograms** (SPEC-T03) built from a
   **gradient index** (SPEC-D04) computed from **quantile cuts** (SPEC-D03) —
   *the single most likely divergence source, isolated and oracle-asserted first*.
6. All of that needs the **`RegTree` model** (SPEC-T02), **`TrainParam`**
   (SPEC-T01), the **SquaredError objective** (SPEC-O01/O02), and the **`DMatrix`**
   (SPEC-D01/D02) as the data substrate — the root of the graph.
7. Prediction (SPEC-P01), the **builder-pattern parameters** (SPEC-B01/B02/B03),
   **JSON model IO** (SPEC-A03), and **feature importance** (SPEC-A04) complete the
   public API surface the oracle test exercises.
8. Finally the **oracle API test** (SPEC-X02) and **oracle speed test** (SPEC-X03).

Therefore the build order is: data → cuts/index → objective/base_score → tree
(model → hist → split → missing → driver) → gbm/learner → predictor/metric →
params → high-level api/model-io → oracle fixtures → oracle api test → oracle
speed test. This matches the ordering constraint in the task brief.

---

## 1. Module layout (SPEC §7) and shared conventions

New modules to be added to `src/lib.rs` (`pub mod` only; additive):

```
data::{dmatrix, meta, adapters, cuts, gradient_index}
objective::{mod, squared_error, init_estimation}
tree::{param, model, split_eval, hist_updater}
gbm::{model, gbtree}
learner
predictor::cpu
metric::{mod, rmse}
parameters::{tree, learning, booster, training}
api::{train, booster}
model_io::json
```

Conventions to follow (from the existing crate):
- Errors extend `src/error.rs`'s `thiserror` enum (`Error`, `Result<T>`); binaries
  wrap with `anyhow` (error.rs:1-26).
- Builders are **consuming builders** matching `HistogramBuilder`
  (`.field(v)…​.build()`), per rust-xgboost conventions (research.md:87-96).
- Registration of objective/metric/updater by `&str` key = a **trait-object
  registry** keyed on the XGBoost param strings (`"reg:squarederror"`, `"rmse"`,
  `"grow_quantile_histmaker"`), per research.md:118.
- `GradientPair { grad: f32, hess: f32 }`: reuse the existing structurally
  identical `gpu::GradientPair` via a re-export/type alias in `objective` to avoid
  a duplicate type. Verified (`src/gpu/mod.rs:10-15`): it is a plain struct
  `#[derive(Clone, Copy, Debug, Default, PartialEq)] pub struct GradientPair {
  grad: f32, hess: f32 }` — it is **not** `#[repr(C)]` and **not** `bytemuck::Pod`
  (only `GradientPairInt64` is Pod). It is nonetheless fully usable by CPU code
  without any GPU device. If the implementer prefers decoupling, define an
  identical `objective::GradientPair`; either satisfies §4.
- **All histogram/statistic accumulation is `f64`** over `f32` inputs;
  single-threaded (`nthread=1`) for determinism (SPEC §4, §9 float-summation risk).
- Unit tests live in `#[cfg(test)] mod tests` inside each source file, run with
  `cargo test --lib <filter>`. Integration/oracle tests live under `tests/`, run
  with `cargo test --test <name>`.

Validation commands (research.md:126-133, verified `cargo 1.97.0`). **Do not
overstate what runs from a bare checkout on this box** — the fixture-based oracle
tests are environment-gated:
- **Offline / always runnable here:**
  - `cargo build --offline` — compile the crate + bins offline (already succeeds).
  - `cargo test --lib <filter>` — run in-crate unit tests (no GPU, no fixtures).
  - `cargo run --bin oracle_speed` — Rust-only speed harness (Rust side always runs).
- **Environment-gated (require committed fixtures produced once by a network
  `pip install xgboost==<pinned>` — TASK-26; NOT runnable from a bare checkout,
  research.md:126-133 confirms no local xgboost):**
  - `cargo test --test oracle_cuts` (AC-D03), `cargo test --test oracle_tree`
    (AC-T06), `cargo test --test oracle_api` (AC-X02).
  - The **XGBoost side** of the speed test (`tests/fixtures/speedtest.py`, AC-X03)
    runs only where a reference XGBoost is installed.
  These gated tests are marked `#[ignore]` (or skip-with-clear-message when the
  fixture directory is absent) so a bare `cargo test` stays green.

---

## 2. Execution waves (parallelization map)

Tasks in the same wave touch disjoint files with no unmet contract dependency and
may run in parallel. `‖` = parallelizable.

- **Wave 0 (kickoff, ‖):** TASK-26 (SPEC-X01 fixture generator — Python-only,
  network `pip install`, no Rust dependency; start immediately so fixtures exist
  by Wave 8). TASK-01 (SPEC-D01 DMatrix — Rust root).
- **Wave 1 (after D01, ‖):** TASK-02 (D02), TASK-05 (O01), TASK-06 (O02),
  TASK-07 (O03), TASK-08 (T01), TASK-09 (T02), TASK-16 (M01). All separate files.
- **Wave 2 (‖):** TASK-03 (D03, needs D01), TASK-19 (B01, needs T01),
  TASK-20 (B02).
- **Wave 3:** TASK-04 (D04, needs D03), TASK-21 (B03, needs B01+B02, ‖ with D04).
- **Wave 4:** TASK-10 (T03, needs D04+O01).
- **Wave 5:** TASK-11 (T04, needs T03).
- **Wave 6:** TASK-12 (T05, needs T04).
- **Wave 7:** TASK-13 (T06, needs T01+T02+T03+T05).
- **Wave 8 (‖):** TASK-14 (G01, needs T02), TASK-18 (P01, needs T02+G01+O02).
- **Wave 9:** TASK-15 (G02, needs T06+G01+O01).
- **Wave 10:** TASK-17 (L01, needs G02+O03+M01).
- **Wave 11 (‖):** TASK-22 (A01, needs L01+B03), TASK-24 (A03 JSON, needs G01),
  TASK-25 (A04 importance, needs G01+T06 for stored per-split loss_chg+sum_hess).
- **Wave 12:** TASK-23 (A02, needs A01+P01).
- **Wave 13:** TASK-27 (X02 oracle API, needs A01/A02/P01/M01/A04/D03 + fixtures
  from TASK-26).
- **Wave 14:** TASK-28 (X03 oracle speed, needs A01).

Dependency graph (edges = "must precede"):

```
D01 -> D02, O01, O03, M01, T01, T02, D03
D03 -> D04
O01, D04 -> T03 -> T04 -> T05 -> T06
T01, T02, T03, T05 -> T06        # T06 needs param, model, node-hist, split+missing
T02 -> G01, P01
O02, G01 -> P01
T06, G01, O01 -> G02 -> L01
O03, M01 -> L01
T01 -> B01 ;  B01, B02 -> B03
L01, B03 -> A01 -> A02
L01, P01 -> A02
G01 -> A03 ; G01, T06 -> A04   # A04 reads stored per-split loss_chg + sum_hess
A01, A02, P01, M01, A04, D03 -> X02
A01 -> X03
# Fixture edges (X01 = TASK-26 emits the committed golden fixtures):
X01 -> D03   # oracle_cuts / AC-D03 fixture assertion
X01 -> T06   # oracle_tree / AC-T06 single-tree-dump fixture assertion
X01 -> A03   # real-XGBoost model.json read test
X01 -> X02   # oracle_api / AC-X02 full-parity fixtures
```

---

## 3. SPEC-ID → task coverage map

| SPEC ID | Task | SPEC ID | Task |
|---|---|---|---|
| SPEC-D01 | TASK-01 | SPEC-L01 | TASK-17 |
| SPEC-D02 | TASK-02 | SPEC-P01 | TASK-18 |
| SPEC-D03 | TASK-03 | SPEC-M01 | TASK-16 |
| SPEC-D04 | TASK-04 | SPEC-B01 | TASK-19 |
| SPEC-O01 | TASK-05 | SPEC-B02 | TASK-20 |
| SPEC-O02 | TASK-06 | SPEC-B03 | TASK-21 |
| SPEC-O03 | TASK-07 | SPEC-A01 | TASK-22 |
| SPEC-T01 | TASK-08 | SPEC-A02 | TASK-23 |
| SPEC-T02 | TASK-09 | SPEC-A03 | TASK-24 |
| SPEC-T03 | TASK-10 | SPEC-A04 | TASK-25 |
| SPEC-T04 | TASK-11 | SPEC-X01 | TASK-26 |
| SPEC-T05 | TASK-12 | SPEC-X02 | TASK-27 |
| SPEC-T06 | TASK-13 | SPEC-X03 | TASK-28 |
| SPEC-G01 | TASK-14 |  |  |
| SPEC-G02 | TASK-15 |  |  |

All 28 SPEC IDs covered. Reverse check: every task references ≥ 1 SPEC ID (below).

---

## 4. Tasks

Each task: **ID + SPEC IDs · Goal + observable completion · Prerequisites/blockers ·
Files/symbols · Red · Green · Refactor · Validation · Completion evidence ·
Parallelization**.

---

### TASK-01 — DMatrix from dense (SPEC-D01)
- **Goal / completion:** `DMatrix::from_dense(&[f32], nrow, ncol, missing)` builds a
  `DMatrix` + `MetaInfo`; NaN and the `missing` sentinel become structural missing;
  shape mismatch returns a typed error. Done when AC-D01 passes.
- **Prerequisites:** none (Rust root). **Blockers:** none.
- **Files/symbols:**
  - Create `src/data/mod.rs` (`pub mod dmatrix; pub mod meta; pub mod adapters;`).
  - Create `src/data/meta.rs`: `struct MetaInfo { num_row, num_col, labels: Vec<f32>, weights: Option<Vec<f32>>, base_margin: Option<Vec<f32>> }`.
  - Create `src/data/dmatrix.rs`: opaque `struct DMatrix`, `from_dense`, `set_labels`, `num_row`, `num_col`, internal missing-mask/value store.
  - Modify `src/error.rs`: **do NOT reuse `Error::MatrixShape`** — its message is
    ELLPACK-specific (`"ELLPACK shape mismatch: gidx has {got} entries, expected
    n_rows * row_stride = {expected}"`, error.rs:7-8) and would be misleading for a
    dense-matrix shape error. Add a dedicated `Error::DMatrixShape { expected:
    usize, got: usize }` with a neutral message (e.g. `"dense matrix length {got}
    does not match nrow*ncol = {expected}"`), plus `Error::Data(String)` for
    parse/label errors.
  - Modify `src/lib.rs`: add `pub mod data;`.
- **Red:** `#[test] fn from_dense_marks_missing()` in `src/data/dmatrix.rs`. Input:
  `data=[1.0, f32::NAN, 3.0, 4.0, 99.0, 6.0]`, `nrow=3, ncol=2, missing=99.0`.
  Expect `num_row()==3`, `num_col()==2`, cell (0,1) missing (NaN) and cell (2,0)
  missing (sentinel), others present. Also a shape case `data.len()=5, nrow=3,
  ncol=2` expects `Err(Error::DMatrixShape{..})`. Initial failure: `DMatrix`/method
  do not exist → compile error, then assertion failure.
- **Green:** minimal `MetaInfo`, `DMatrix` with row-major store + missing predicate
  (`v.is_nan() || v == missing`); `from_dense` validates `data.len()==nrow*ncol`.
- **Refactor:** extract `is_missing(v, missing)` helper; document the opaque layout.
  Regression scope: `cargo test --lib data::`.
- **Validation:** `cargo build --offline` · `cargo test --lib from_dense`.
- **Completion evidence:** both dense assertions + the `DMatrixShape` error case pass.
- **Parallelization:** Wave 0. Root; nothing else may start its Rust deps until this
  compiles.

---

### TASK-02 — DMatrix from LIBSVM (SPEC-D02)
- **Goal / completion:** `DMatrix::from_libsvm(&Path)` parses agaricus LIBSVM
  (`label idx:val …`, **1-based** indices) into a CSR sparse `DMatrix` + labels.
  Done when a 2-row fixture parses to the expected CSR + labels.
- **Prerequisites:** TASK-01 (DMatrix/MetaInfo types). **Blockers:** none (dataset
  vendored at `xgboost/demo/data/agaricus.txt.train`, verified).
- **Files/symbols:**
  - Create `src/data/adapters.rs`: `parse_libsvm(reader) -> (Csr, Vec<f32>)`,
    `struct Csr { indptr: Vec<usize>, indices: Vec<u32>, values: Vec<f32>, ncol: usize }`.
  - Modify `src/data/dmatrix.rs`: `from_libsvm`, sparse internal representation,
    `num_col` = max index seen (0-based after decrement).
  - Decide LIBSVM parser: hand-rolled (no new dep) — recommended to keep deps minimal.
- **Red:** `#[test] fn libsvm_two_rows_1based()` in `src/data/adapters.rs`. Input
  string `"1 1:1 3:1\n0 2:1\n"`. Expect labels `[1.0, 0.0]`, `indptr=[0,2,3]`,
  `indices=[0,2,1]` (1-based → 0-based), `values=[1.0,1.0,1.0]`, `ncol==3`. Initial
  failure: `parse_libsvm` missing.
- **Green:** line parser splitting on whitespace, `idx:val` split on `:`, decrement
  index, build CSR; blank/comment lines skipped.
- **Refactor:** stream via `BufReader` for the real file; add `Error::Data` on
  malformed token. Regression: `cargo test --lib data::adapters`.
- **Validation:** `cargo test --lib libsvm_two_rows_1based` · then a smoke test
  loading `xgboost/demo/data/agaricus.txt.train` asserting `num_row()>0`.
- **Completion evidence:** CSR/label assertions pass; real agaricus file loads.
- **Parallelization:** Wave 1 (‖ with O01/O02/O03/T01/T02/M01; different files).

---

### TASK-03 — Quantile cuts (SPEC-D03) — *isolation-critical*
- **Goal / completion:** `build_cuts(&DMatrix, max_bin) -> HistogramCuts` reproduces
  XGBoost `hist` `HistogramCuts` (`cut_ptrs` integer-exact, `cut_values` ≤ 1e-5)
  for the committed small dataset. Done when AC-D03 passes against the
  `get_quantile_cut` fixture.
- **Prerequisites:** TASK-01. **Blockers:** *Open question §9.1* — resolve the two
  cut branches by reading `xgboost/src/common/quantile.{h,cc}` (`WQuantileSketch`,
  `HostSketchContainer`, `SketchContainer::MakeCuts`) + `hist_util.cc` and diffing
  against the `get_quantile_cut` fixture (TASK-26 emits it). **Both branches are
  required for AC-X02** (SPEC-D03, updated): (a) low-cardinality — distinct-value
  count ≤ `max_bin` → cuts are the sorted distinct values (upper bounds); (b)
  **weighted-quantile-sketch** — distinct count > `max_bin` (the continuous
  regression fixture, cardinality > `max_bin=256`) → cuts are the sketch's
  `max_bin` quantile boundaries. Path (b) is on the AC-X02 critical path and MUST
  ship, not just (a).
- **Files/symbols:**
  - Create `src/data/cuts.rs`: `struct HistogramCuts { cut_ptrs: Vec<u32>, cut_values: Vec<f32>, min_values: Vec<f32> }`, `build_cuts`, a `sketch` submodule implementing the weighted-quantile sketch (`WQuantileSketch`-equivalent).
  - Modify `src/data/mod.rs`: `pub mod cuts;`.
- **Red (unit, offline):** `#[test] fn cuts_low_cardinality_sorted_distinct()` in
  `src/data/cuts.rs`: values `[1,2,3]`, `max_bin>=4` → distinct ≤ max_bin branch →
  cuts at the upper bounds XGBoost uses; `min_values[0]` = first value's lower
  bound. `#[test] fn cuts_high_cardinality_uses_sketch()`: a single feature with
  1000 distinct values, `max_bin=16` → distinct > max_bin → the WQ-sketch branch
  runs and yields exactly `max_bin` (16) cut boundaries at the expected quantile
  positions. Initial failure: `build_cuts` / sketch missing.
- **Red (fixture, ENVIRONMENT-GATED):** `#[test] fn cuts_match_fixture()` in
  `tests/oracle_cuts.rs`, marked `#[ignore]` / skip-with-message when
  `tests/fixtures/<case>/quantile_cut.json` is absent. Given the committed
  continuous regression matrix; when `build_cuts(max_bin=16)`; then `cut_ptrs`
  exact and each `cut_value` within 1e-5 of the `get_quantile_cut` fixture — this
  exercises the sketch branch against the real oracle.
- **Green:** implement **both** branches: sorted-distinct for ≤ max_bin, and the
  weighted-quantile sketch (`HostSketchContainer`-equivalent with `WQuantileSketch`
  merge/prune → `MakeCuts`) for > max_bin; `f64` internal accumulation; emit
  `min_values` per feature.
- **Refactor:** dedupe the per-feature dispatch; document the cardinality threshold.
- **Validation:** `cargo test --lib cuts_` (offline units, both branches) ·
  (gated) `cargo test --test oracle_cuts` after TASK-26 fixtures exist.
- **Completion evidence:** both unit branches pass offline; the gated fixture
  equality holds where fixtures exist. This must pass **before** any training-parity
  test is trusted (SPEC §9 risk #1).
- **Parallelization:** Wave 2. Isolated file; only D01 precedes it.

---

### TASK-04 — Gradient index / binning (SPEC-D04)
- **Goal / completion:** `build_gradient_index(&DMatrix, &HistogramCuts) ->
  GHistIndex` maps each **non-missing** feature value to its bin (upper-bound binary
  search within the feature's cut slice). Missing/NaN cells are simply **absent**
  from the row's entries (CSR-style) — there is **NO dedicated missing bin/slot**;
  the split scan reconstructs missing implicitly (SPEC-T03/T05). Done when a known
  matrix bins non-missing cells correctly and omits missing cells.
- **Prerequisites:** TASK-03. **Blockers:** none.
- **Files/symbols:**
  - Create `src/data/gradient_index.rs`: `struct GHistIndex { index: Vec<u32>, row_ptr: Vec<usize>, cuts: HistogramCuts }` (CSR-shaped: `row_ptr` delimits each row's present entries), `build_gradient_index`.
  - Modify `src/data/mod.rs`: `pub mod gradient_index;`.
- **Red:** `#[test] fn binning_omits_missing_cells()` in `src/data/gradient_index.rs`.
  Given a 2-row, 2-feature matrix where row 0 = `[1.0, missing]` and row 1 =
  `[3.0, 5.0]`, cuts `cut_ptrs=[0,2,4]`, feature-0 `cut_values=[2.0,4.0]`,
  feature-1 `cut_values=[4.0,6.0]`; when binned; then row 0 has exactly ONE entry
  (feature 0 → bin 0) and feature 1 is **absent** (no reserved-slot bin appended);
  row 1 has TWO entries (feature 0 → bin 1, feature 1 → global bin 3); `row_ptr ==
  [0, 1, 3]`. Initial failure: function missing. (Explicitly assert there is no
  extra "missing" bin index for the absent cell.)
- **Green:** for each present (non-missing) value, upper-bound binary-search within
  the feature's `[cut_ptrs[f], cut_ptrs[f+1])` slice of `cut_values`; skip missing
  cells entirely; build `row_ptr` over present entries only.
- **Refactor:** share the search with `RegTree` traversal's bin logic if convenient;
  document that missing is implicit (`parent − Σbins`), consumed by SPEC-T03/T05.
- **Validation:** `cargo test --lib binning_omits_missing_cells`.
- **Completion evidence:** non-missing bins correct; missing cells absent; `row_ptr`
  matches; no dedicated missing slot exists.
- **Parallelization:** Wave 3. Depends only on D03.

---

### TASK-05 — SquaredError gradient (SPEC-O01)
- **Goal / completion:** `SquaredError::get_gradient` yields `grad_i =
  (pred_i − label_i)·w_i`, `hess_i = 1·w_i` (`w_i=1` when no weights). Done when
  AC-O01 passes.
- **Prerequisites:** TASK-01 (MetaInfo). **Blockers:** none.
- **Files/symbols:**
  - Create `src/objective/mod.rs`: `trait Objective { get_gradient, pred_transform, init_estimation, default_metric }`, `type GradientPair = crate::gpu::GradientPair;` (re-export) or an identical local struct; a `&str` registry `objective_by_name(&str) -> Box<dyn Objective>`.
  - Create `src/objective/squared_error.rs`: `struct SquaredError; impl Objective`.
  - Modify `src/lib.rs`: `pub mod objective;`.
- **Red:** `#[test] fn squared_error_gradient()` in `src/objective/squared_error.rs`.
  Input `preds=[0.5,0.5]`, `MetaInfo{labels:[1.0,0.0], weights:None,…}`, `iter=0`.
  Expect gpairs `[(-0.5,1.0),(0.5,1.0)]`. Initial failure: `Objective`/`SquaredError`
  missing. Evidence for formula: `regression_obj.cu:184-197`.
- **Green:** implement `get_gradient` only (leave `pred_transform`/`init_estimation`
  `todo!()` for TASK-06/07 or minimal stubs that later tasks flesh out).
- **Refactor:** apply weights branch once; keep `f32` I/O. Regression:
  `cargo test --lib objective::`.
- **Validation:** `cargo test --lib squared_error_gradient`.
- **Completion evidence:** AC-O01 exact tuple match.
- **Parallelization:** Wave 1 (‖). Separate file from D02/T01/T02/M01.

---

### TASK-06 — SquaredError pred_transform & default metric (SPEC-O02)
- **Goal / completion:** `pred_transform` = identity (no mutation);
  `default_metric()` == `"rmse"`. Done when a transform round-trip is a no-op and
  the metric key matches.
- **Prerequisites:** TASK-05 (trait + struct exist). **Blockers:** none.
- **Files/symbols:** Modify `src/objective/squared_error.rs` (`pred_transform`,
  `default_metric`).
- **Red:** `#[test] fn identity_transform_and_metric()`. Given `preds=[−1.0,0.0,2.5]`,
  clone; when `pred_transform(&mut preds)`; then `preds` unchanged and
  `default_metric()=="rmse"`. Initial failure: stub differs / returns wrong key.
  Evidence: `regression_obj.cu:200,202-210`.
- **Green:** empty-body `pred_transform`; return `"rmse"`.
- **Refactor:** document identity link. Regression: `cargo test --lib objective::`.
- **Validation:** `cargo test --lib identity_transform_and_metric`.
- **Completion evidence:** no-op + `"rmse"` assertions pass.
- **Parallelization:** Wave 1 (‖, same file as O01 — sequence O05→O06 within the
  file; not parallel with TASK-05, parallel with all non-objective Wave-1 tasks).

---

### TASK-07 — base_score init estimation (SPEC-O03) — *isolation-critical*
- **Goal / completion:** `SquaredError::init_estimation(&MetaInfo) -> f32` returns
  the (weighted) sample mean of labels (identity link ⇒ margin = mean). Done when
  AC-O03 passes.
- **Prerequisites:** TASK-05, TASK-01. **Blockers:** none.
- **Files/symbols:**
  - Create `src/objective/init_estimation.rs`: `fn weighted_mean(labels, weights) -> f32`.
  - Modify `src/objective/squared_error.rs`: `init_estimation` delegates.
- **Red:** `#[test] fn init_estimation_is_mean()`. Given `labels=[1,0,0,1]`,
  `weights:None`; when `init_estimation`; then `0.5` within 1e-6. Add a weighted
  case `labels=[1,0], weights=[3,1]` → `0.75`. Initial failure: function missing.
  Evidence: `regression_obj.cu:212-220`.
- **Green:** `f64` accumulation of `Σ w·y / Σ w` (unweighted → `n`), cast to `f32`.
- **Refactor:** guard `Σw==0`. Regression: `cargo test --lib objective::`.
- **Validation:** `cargo test --lib init_estimation`.
- **Completion evidence:** unweighted 0.5 and weighted 0.75 within 1e-6.
- **Oracle-validation note:** the main AC-X02 fixtures pin `base_score=0.5`
  (SPEC-X01), so end-to-end parity does **not** depend on this estimator. But when
  `base_score` is UNSET, XGBoost's estimated intercept is exactly `mean(labels)`
  (identity link), which is what a base_score-unset fixture would check; the
  estimator is exercised directly by AC-O03 here. Keep both: pinned 0.5 for AC-X02,
  estimator unit-tested here.
- **Parallelization:** Wave 1 (‖ with non-objective tasks). *Isolated because a
  wrong intercept shifts every prediction by a constant (SPEC §9 risk #3).*

---

### TASK-08 — TrainParam + validation (SPEC-T01)
- **Goal / completion:** `TrainParam` holds `eta, max_depth, max_leaves,
  min_child_weight, reg_lambda, reg_alpha, gamma, max_delta_step, max_bin,
  grow_policy, base_score` and validates ranges. Done when a valid build succeeds
  and an out-of-range field errors.
- **Prerequisites:** none (uses only primitives). **Blockers:** none.
- **Files/symbols:**
  - Create `src/tree/mod.rs` (`pub mod param; pub mod model; pub mod split_eval; pub mod hist_updater;`).
  - Create `src/tree/param.rs`: `struct TrainParam { … }`, `enum GrowPolicy { Depthwise, LossGuide }`, `fn validate(&self) -> Result<()>`.
  - Modify `src/lib.rs`: `pub mod tree;`. Modify `src/error.rs`: `Error::Param(String)`.
- **Red:** `#[test] fn trainparam_validates_ranges()` in `src/tree/param.rs`. A valid
  `TrainParam{eta:0.3,max_depth:6,…}` validates `Ok`; `eta:-1.0` → `Err(Error::Param)`;
  `max_bin:0` → `Err`. Initial failure: struct/validate missing. Evidence:
  `xgboost/src/tree/param.h`.
- **Green:** struct + `validate` enforcing `eta∈(0,1], max_depth≥0, max_bin≥2,
  min_child_weight≥0, lambda≥0, gamma≥0`.
- **Refactor:** derive `Clone, Debug`; default `Depthwise`. Regression:
  `cargo test --lib tree::param`.
- **Validation:** `cargo test --lib trainparam_validates_ranges`.
- **Completion evidence:** valid Ok + two Err cases pass.
- **Parallelization:** Wave 1 (‖). New `tree` module, disjoint from data/objective.

---

### TASK-09 — RegTree model (SPEC-T02)
- **Goal / completion:** `RegTree` node array (split_feature `i32` `-1`=leaf,
  split_cond `f32`, default_left `bool`, left/right `i32`, leaf_value `f32`,
  sum_hess `f32`) with add-node, mark-leaf, and `predict_one(row)` traversal that
  routes missing per `default_left`. Done when a hand-built 3-node tree predicts
  correctly incl. the missing route.
- **Prerequisites:** TASK-01 (row access). **Blockers:** none.
- **Files/symbols:** Create `src/tree/model.rs`: `struct RegTree`, `struct Node`,
  `add_node`, `set_leaf`, `predict_one(&self, row) -> f32`, `num_nodes`.
- **Red:** `#[test] fn regtree_predicts_with_missing_route()`. Build root split on
  feature 0 at `cond=2.0`, `default_left=true`, left leaf `10.0`, right leaf `20.0`.
  Row `[1.0]` → `10.0`; row `[3.0]` → `20.0`; row with feature 0 **missing** → left
  (`10.0`, honoring `default_left`). Initial failure: types/traversal missing.
  Evidence: `[CODEGRAPH RegTree]`.
- **Green:** node vector + traversal; XGBoost split test `value < split_cond → left`.
- **Refactor:** `#[inline]` traversal; document `-1` leaf sentinel. Regression:
  `cargo test --lib tree::model`.
- **Validation:** `cargo test --lib regtree_predicts_with_missing_route`.
- **Completion evidence:** three routing assertions pass.
- **Parallelization:** Wave 1 (‖). Separate file.

---

### TASK-10 — Node histogram build (SPEC-T03)
- **Goal / completion:** For a node's row set, accumulate per-`(feature,bin)`
  `(Σgrad, Σhess)` in `f64` over **non-missing** cells from a `GHistIndex`, AND
  compute the node total `parent_stats = Σ gpair over ALL rows in the node`
  (including rows whose value for a given feature is missing). The per-feature
  missing stats consumed by the split scan are **implicit**: `missing =
  parent_stats − Σ(that feature's bins)`; there is NO dedicated missing bin. Done
  when a known index + gpairs yields the expected `NodeHist { bins, parent_stats }`.
- **Prerequisites:** TASK-04 (GHistIndex), TASK-05 (GradientPair). **Blockers:** none.
- **Files/symbols:** Create `src/tree/hist_updater.rs` (start): `struct NodeHist {
  bins: Vec<(f64,f64)>, parent_stats: (f64,f64) }`, `fn build_node_hist(&GHistIndex,
  rows: &[usize], &[GradientPair]) -> NodeHist`. (Mirrors CPU oracle philosophy,
  reference.rs:79-97; parent-total per evaluate_splits.h:281-309.)
- **Red:** `#[test] fn node_hist_and_parent_stats()`. GHistIndex over 2 features
  where row 0 = `{f0→bin0}` (f1 missing), row 1 = `{f0→bin1, f1→bin(global)}`;
  gpairs `[(1.0,1.0),(2.0,1.0)]`; rows `[0,1]`. Expect f0 bins `[(1.0,1.0),
  (2.0,1.0)]`, f1's single present bin `(2.0,1.0)`, and **`parent_stats ==
  (3.0,2.0)`** (both rows counted). Assert the implicit f1 missing =
  `parent_stats − Σ f1 bins == (1.0,1.0)` (row 0). Initial failure: function/field
  missing.
- **Green:** loop present entries → bins in `f64`; separately sum every row's gpair
  into `parent_stats` (independent of which features are present). No missing bin.
- **Refactor:** reuse the `cpu_histogram` iteration pattern from `reference.rs`;
  keep parent−child subtraction *optional* and gated (SPEC §9 risk #5). Regression:
  `cargo test --lib tree::hist_updater`.
- **Validation:** `cargo test --lib node_hist_and_parent_stats`.
- **Completion evidence:** exact `f64` bin sums + correct `parent_stats`; implicit
  missing reconstructs correctly.
- **Parallelization:** Wave 4. Needs D04 + O01.

---

### TASK-11 — Split evaluation (SPEC-T04)
- **Goal / completion:** Given a `NodeHist` + `parent_gain = CalcGain(parent_stats)`,
  scan each feature's bins, computing the **parent-subtracted** gain
  `loss_chg = CalcSplitGain(GL,HL,GR,HR) − parent_gain`, where `CalcSplitGain =
  CalcGain(GL,HL) + CalcGain(GR,HR)` returns `-inf` unless `HL,HR>0 ∧ ≥
  min_child_weight` (comparing the child **hessian**, not weight). Track the best
  `(feature, split_pt, default_left, loss_chg, left_sum, right_sum)`. **Two-part
  validity gate (both required):** a node splits iff best `loss_chg > kRtEps`
  (`= 1e-6f`, strict `>`) **AND** `loss_chg ≥ gamma` (non-strict `≥`, `gamma =
  min_split_loss`); otherwise return `None` (node becomes a leaf). The kRtEps gate
  is **separate from gamma** and MUST be applied — with `gamma=0`, omitting it would
  accept tiny-gain splits XGBoost rejects → extra nodes → AC-X02 structure failure.
  Done when AC-T04 (incl. the two rejection cases) passes.
- **Prerequisites:** TASK-10, TASK-08. **Blockers:** none.
- **Files/symbols:** Create `src/tree/split_eval.rs`: `const K_RT_EPS: f32 = 1e-6;`
  (mirrors `base.h:309`), `fn threshold_l1(g,alpha)`, `fn calc_gain(g,h,param)`
  (`== root_gain`), `fn calc_weight(g,h,param)` (clamp to `max_delta_step`),
  `fn calc_split_gain(gl,hl,gr,hr,param)`, `struct SplitCandidate { feature,
  split_pt: f32, default_left: bool, loss_chg: f32, left_sum, right_sum }`,
  `fn evaluate_split(&NodeHist, parent_gain: f64, &TrainParam) -> Option<SplitCandidate>`.
- **Red:** `#[test] fn split_loss_chg_matches_hand_value()`. NodeHist with parent
  `(G,H)` and a known best split (feature 0: left `(GL,HL)`, right `(GR,HR)`),
  `lambda=1, alpha=0, gamma=0`; expect chosen `(feature, split_pt)` and
  **`loss_chg == [GL²/(HL+1)+GR²/(HR+1)] − G²/(H+1)`** within 1e-6.
  `#[test] fn rejects_below_krteps()`: construct so the best `loss_chg ≤ 1e-6`
  (e.g. a perfectly balanced split with near-zero improvement) → `evaluate_split`
  returns `None`.
  `#[test] fn rejects_below_gamma()`: a split with `loss_chg == 3.0`, `gamma == 5.0`
  → `None` (and with `gamma == 3.0`, non-strict `≥`, the same `loss_chg==3.0` split
  is **accepted**). Initial failure: functions missing. Evidence: `param.h:224-265`,
  `split_evaluator.h:74-99`, `evaluate_splits.h:281-309,290-293,298-301`,
  `expand_entry.h:124-128`, `base.h:309`.
- **Green:** implement `threshold_l1`, `calc_gain`, `calc_weight`, `calc_split_gain`;
  prefix-sum scan producing `loss_chg = split_gain − parent_gain`; child
  `min_child_weight`/`H>0` gate → `-inf`; apply the `loss_chg > kRtEps AND loss_chg
  ≥ gamma` gate before returning `Some`.
- **Refactor:** share prefix-sum accumulation; document that `loss_chg` (not the raw
  child-sum `CalcSplitGain`) is THE decision quantity and the stored/importance
  "gain" (consumed by TASK-13/TASK-25). Regression: `cargo test --lib tree::split_eval`.
- **Validation:** `cargo test --lib split_loss_chg_matches_hand_value` ·
  `cargo test --lib rejects_below_krteps` · `cargo test --lib rejects_below_gamma`.
- **Completion evidence:** AC-T04 `loss_chg` + chosen split match; both kRtEps and
  gamma rejection cases return `None`; the `gamma==loss_chg` boundary accepts.
- **Parallelization:** Wave 5. Needs T03.

---

### TASK-12 — Split-point convention & missing default direction (SPEC-T05) — *isolation-critical*
- **Goal / completion:** Implement XGBoost's exact two-pass sparsity-aware numeric
  scan, with the correct `split_pt` value per pass and the correct condition for
  running the backward pass:
  - **Forward pass** (`left_sum += bin i`, `right = parent − left`, missing→right):
    `split_pt = cut_val[i]`, `default_left = false`; **ALWAYS runs**.
  - **Backward pass** (missing→left): `split_pt = NumericBinLowerBound(cut_ptr,
    cut_val, fidx, i)`, `default_left = true`; **runs ONLY IF the feature has
    missing** — i.e. `parent_stats(feature) ≠ Σ(that feature's non-missing bins)`
    (`SplitContainsMissingValues`). Missing is **implicit** (`parent − Σbins`);
    there is no dedicated missing bin.
  Keep whichever pass yields the higher `loss_chg`; ties resolve to the
  forward/existing candidate ⇒ `default_left = false` (matches XGBoost
  `SplitEntry::Update`'s non-strict improvement rule). Store `split_pt` and
  `default_left` on the candidate. Done when AC-T05 passes.
- **Prerequisites:** TASK-11. **Blockers:** none.
- **Files/symbols:** Modify `src/tree/split_eval.rs`: extend `evaluate_split` with
  the forward+backward enumeration; add `fn numeric_bin_lower_bound(cuts, fidx, i)
  -> f32` mirroring XGBoost's `NumericBinLowerBound` (the lower boundary of bin `i`
  for the feature). Add a `feature_has_missing(&NodeHist, fidx) -> bool` helper
  using `parent_stats` vs the feature's bin sum.
- **Red:** `#[test] fn split_point_and_direction_convention()`.
  (a) Feature WITH missing, constructed so routing missing **left** yields strictly
  higher `loss_chg`: expect `default_left == true` AND
  `split_pt == numeric_bin_lower_bound(cuts, fidx, i)` (backward value).
  (b) Symmetric no-benefit case (missing→right better): `default_left == false` AND
  `split_pt == cut_val[i]` (forward value).
  (c) Feature with **no missing** (`parent_stats == Σ bins`): assert the backward
  pass never runs (best candidate is always `default_left == false`, forward
  `split_pt`), even if a hypothetical missing-left routing would have scored higher.
  (d) Exact-tie case: `default_left == false` (forward wins ties).
  Initial failure: `split_pt`/backward pass not implemented. Evidence:
  `evaluate_splits.h:281-309,369-372`.
- **Green:** two-pass enumeration; gate the backward pass on `feature_has_missing`;
  set `split_pt` per pass; tie-break to forward.
- **Refactor:** unify the two passes behind one helper. Regression:
  `cargo test --lib tree::split_eval`.
- **Validation:** `cargo test --lib split_point_and_direction_convention`.
- **Completion evidence:** all four sub-cases pass — including that a no-missing
  feature never runs the backward pass and ties give `default_left=false`.
- **Parallelization:** Wave 6. Needs T04. *Distinct divergence source, isolated.*

---

### TASK-13 — Tree growth driver, depthwise (SPEC-T06)
- **Goal / completion:** Grow a `RegTree` to `max_depth` (`grow_policy=Depthwise`):
  per level build node hists (with `parent_stats`), evaluate splits (incl. split_pt
  + missing dir), and apply a node's best split **only when it is valid per
  SPEC-T04 — `loss_chg > kRtEps (=1e-6) AND loss_chg ≥ gamma`** — otherwise finalize
  the node as a leaf. Store on each internal node the split `loss_chg`
  (parent-subtracted gain) and the node `sum_hess` (cover) for SPEC-A04. Assign leaf
  weight `= eta·CalcWeight(G,H)` to terminals. Done when AC-T06 matches an XGBoost
  single-tree dump **exactly** (node count/feature/split_pt/default_left; no spurious
  tiny-gain nodes) with leaves ≤ 1e-5.
- **Prerequisites:** TASK-08 (T01 param), TASK-09 (T02 model), TASK-10 (T03
  node-hist + parent_stats), TASK-12 (T04+T05 split eval + missing dir). **Blockers:**
  the exact-match assertion requires the single-tree fixture from TASK-26 (X01) —
  gated; a hand-computed depth-1 case precedes fixtures.
- **Files/symbols:** Modify `src/tree/hist_updater.rs`: `struct HistUpdater; impl
  TreeUpdater`; `trait TreeUpdater { fn update(&mut self, &[GradientPair], &DMatrix/
  &GHistIndex, &mut RegTree) }`; depthwise queue/driver, row-partitioning per split.
  Ensure `RegTree` internal nodes carry `split_gain: f32` (the stored `loss_chg`)
  and `sum_hess: f32` (extend `src/tree/model.rs` `Node` if not already present).
- **Red (unit, offline):** `#[cfg(test)] fn grow_depth1_hand_case()`: 4 rows with
  an obvious split, `max_depth=1, eta=1` → known leaf weights + exactly 3 nodes;
  additionally a **kRtEps/gamma case**: data whose only candidate split has
  `loss_chg ≤ 1e-6` (or `< gamma`) → the tree stays a single leaf (1 node), proving
  no spurious split. Assert internal node stores `split_gain == loss_chg` and
  `sum_hess`.
- **Red (fixture, ENVIRONMENT-GATED):** `#[test] fn grow_one_tree_matches_dump()` in
  `tests/oracle_tree.rs`, `#[ignore]`/skip when `tests/fixtures/<case>/tree_dump.txt`
  absent. `max_depth=1, eta=1`; node count, split feature, `split_pt`, `default_left`
  match the XGBoost dump exactly; leaves ≤ 1e-5. Initial failure: updater missing.
- **Green:** partitioning + depthwise growth using T03/T04/T05; leaf `= eta *
  calc_weight(G,H)`; apply the full `loss_chg > kRtEps AND loss_chg ≥ gamma` gate
  (not just `gamma`); persist `split_gain`+`sum_hess` on internal nodes.
- **Refactor:** optional parent−child hist subtraction *only if within tolerance*
  (SPEC §9 risk #5); else build each node directly. Regression:
  `cargo test --lib tree::`.
- **Validation:** `cargo test --lib grow_depth1_hand_case` (offline) · (gated)
  `cargo test --test oracle_tree` after TASK-26 fixtures exist.
- **Completion evidence:** hand depth-1 leaves + node count correct; kRtEps/gamma
  case yields a single leaf; internal nodes carry `loss_chg`+`sum_hess`; the gated
  fixture structure is exact with leaves ≤ 1e-5.
- **Parallelization:** Wave 7. Sequential apex of the tree subsystem.

---

### TASK-14 — GBTreeModel container (SPEC-G01)
- **Goal / completion:** `GBTreeModel` appends trees, holds `base_score`,
  `num_feature`, and iterates trees for prediction. Done when append + iterate +
  base_score round-trip in-memory.
- **Prerequisites:** TASK-09 (RegTree). **Blockers:** none.
- **Files/symbols:**
  - Create `src/gbm/mod.rs` (`pub mod model; pub mod gbtree;`).
  - Create `src/gbm/model.rs`: `struct GBTreeModel { trees: Vec<RegTree>, base_score: f32, num_feature: usize }`, `push_tree`, `num_trees`, `predict_margin_row`.
  - Modify `src/lib.rs`: `pub mod gbm;`.
- **Red:** `#[test] fn gbtree_model_appends_and_sums()`. Push two 1-node leaf trees
  (`3.0`, `4.0`), `base_score=0.5`; `predict_margin_row(any) == 0.5+3.0+4.0 == 7.5`.
  Initial failure: struct/methods missing.
- **Green:** vector push + margin sum over `predict_one`.
- **Refactor:** derive `Clone`. Regression: `cargo test --lib gbm::model`.
- **Validation:** `cargo test --lib gbtree_model_appends_and_sums`.
- **Completion evidence:** 7.5 margin assertion passes.
- **Parallelization:** Wave 8 (‖ with P01). Needs T02.

---

### TASK-15 — GBTree do_boost, one round (SPEC-G02)
- **Goal / completion:** One round: get gradients from current margin preds, grow
  one `RegTree` via the hist updater, append to model, update cached margin preds
  by `+leaf_value` per row. Done when one round on tiny data reduces training rmse
  and appends exactly one tree.
- **Prerequisites:** TASK-13 (T06), TASK-14 (G01), TASK-05 (O01). **Blockers:** none.
- **Files/symbols:** Create `src/gbm/gbtree.rs`: `struct GBTree { model: GBTreeModel,
  updater: HistUpdater, cuts/index cache }`, `fn do_boost(&mut self, &DMatrix, iter,
  &dyn Objective, &TrainParam, margin_cache: &mut [f32])`.
- **Red:** `#[test] fn do_boost_appends_one_tree_and_updates_margin()`. Tiny 4-row
  regression data, `base_score` set, `eta=1, max_depth=1`; capture margin before;
  when `do_boost(iter=0)`; then `model.num_trees()==1` and the new margins equal
  `old_margin + new_tree.predict_one(row)` per row. Initial failure: `do_boost`
  missing. Evidence: `[CODEGRAPH DoBoost → updater Update]`.
- **Green:** call objective `get_gradient(margin_cache)`, build/reuse `GHistIndex`,
  `updater.update`, push tree, add leaf outputs into `margin_cache`.
- **Refactor:** cache cuts/index across rounds (build once). Regression:
  `cargo test --lib gbm::`.
- **Validation:** `cargo test --lib do_boost_appends_one_tree`.
- **Completion evidence:** exactly one tree appended; margin-update identity holds.
- **Parallelization:** Wave 9. Needs T06+G01+O01.

---

### TASK-16 — RMSE metric (SPEC-M01)
- **Goal / completion:** `Rmse::eval(&[f32], &MetaInfo) -> f64 =
  sqrt(Σ w_i(p_i−y_i)² / Σ w_i)` (`w_i=1` default). Done when AC-M01 matches
  `sqrt(mean((p−y)²))` within 1e-9.
- **Prerequisites:** TASK-01 (MetaInfo). **Blockers:** none.
- **Files/symbols:**
  - Create `src/metric/mod.rs`: `trait Metric { fn name(&self)->&str; fn eval(&self,&[f32],&MetaInfo)->f64 }`, `metric_by_name(&str)`.
  - Create `src/metric/rmse.rs`: `struct Rmse; impl Metric`.
  - Modify `src/lib.rs`: `pub mod metric;`.
- **Red:** `#[test] fn rmse_matches_closed_form()`. `preds=[1.0,2.0,3.0]`,
  `labels=[1.0,2.0,0.0]`, no weights; expect `sqrt(9/3)==sqrt(3.0)` within 1e-9;
  `name()=="rmse"`. Initial failure: struct/eval missing. Evidence: `grep metric/ rmse`.
- **Green:** `f64` accumulation; unweighted `Σw = n`.
- **Refactor:** weighted branch; guard `Σw==0`. Regression: `cargo test --lib metric::`.
- **Validation:** `cargo test --lib rmse_matches_closed_form`.
- **Completion evidence:** value within 1e-9, name matches.
- **Parallelization:** Wave 1 (‖). Independent file, only D01 precedes.

---

### TASK-17 — Learner orchestration (SPEC-L01)
- **Goal / completion:** `Learner::{configure, update_one_iter, eval_one_iter,
  predict}`: on iter 0 initialize `base_score` (via SPEC-O03) unless user-set and
  seed the margin cache; `update_one_iter` = one `do_boost`; `eval_one_iter`
  formats `"[<iter>]\t<name>-<metric>:<value>"`. Done when a 2-round train on tiny
  data produces a monotonically improving formatted rmse string and initializes
  base_score from labels.
- **Prerequisites:** TASK-15 (G02), TASK-07 (O03), TASK-16 (M01). **Blockers:** none.
- **Files/symbols:** Create `src/learner.rs`: `struct Learner { objective: Box<dyn
  Objective>, gbm: GBTree, param: TrainParam, metric: Box<dyn Metric>, base_score,
  margin_cache }`; methods per SPEC §4. Modify `src/lib.rs`: `pub mod learner;`.
- **Red:** `#[test] fn learner_inits_base_score_and_formats_eval()`. Build a Learner
  over tiny regression data with `base_score=None`; `configure`; `update_one_iter(0,
  dtrain)`; assert internal `base_score == mean(labels)` (via a test accessor) and
  `eval_one_iter(0,&[(&dtrain,"train")])` returns a string starting `"[0]\ttrain-rmse:"`.
  Initial failure: Learner missing. Evidence: `learner.h:73-176`.
- **Green:** wire objective/gbm/metric; iter-0 base_score init from O03 seeding
  margin cache; eval formatting.
- **Refactor:** extract the format helper; ensure user-set base_score is respected.
  Regression: `cargo test --lib learner`.
- **Validation:** `cargo test --lib learner_inits_base_score`.
- **Completion evidence:** base_score == mean(labels); eval string format exact.
- **Parallelization:** Wave 10. Central integration point.

---

### TASK-18 — CPU predictor (SPEC-P01)
- **Goal / completion:** `predict(&GBTreeModel, &DMatrix, output_margin) ->
  Vec<f32>`: `margin = base_score + Σ_tree leaf_value(row)`;
  `value = pred_transform(margin)` (identity). Done when AC-P01 holds for a 1-tree
  model with `output_margin=true`.
- **Prerequisites:** TASK-14 (G01), TASK-09 (T02), TASK-06 (O02). **Blockers:** none.
- **Files/symbols:**
  - Create `src/predictor/mod.rs` (`pub mod cpu;`).
  - Create `src/predictor/cpu.rs`: `fn predict(&GBTreeModel, &DMatrix, output_margin, &dyn Objective) -> Vec<f32>`.
  - Modify `src/lib.rs`: `pub mod predictor;`.
- **Red:** `#[test] fn predict_margin_is_base_plus_leaf()`. 1-tree model
  (`base_score=0.5`, single leaf `2.0`), 2-row DMatrix; `predict(output_margin=true)`
  → `[2.5, 2.5]`; with `output_margin=false` and identity transform, identical.
  Initial failure: function missing. Evidence: `[CODEGRAPH Predictor]`.
- **Green:** row loop calling `GBTreeModel::predict_margin_row`; apply
  `pred_transform` unless margin requested.
- **Refactor:** batch over rows once. Regression: `cargo test --lib predictor::`.
- **Validation:** `cargo test --lib predict_margin_is_base_plus_leaf`.
- **Completion evidence:** `[2.5,2.5]` both modes.
- **Parallelization:** Wave 8 (‖ with G01→ actually needs G01, so after G01; ‖ with
  nothing that writes predictor). Runs alongside TASK-14 completion.

---

### TASK-19 — TreeBoosterParameters(+Builder) (SPEC-B01)
- **Goal / completion:** Consuming builder with XGBoost defaults (`eta=0.3,
  max_depth=6, lambda=1, alpha=0, gamma=0, min_child_weight=1, max_bin=256`);
  `.build()` validates → `TreeBoosterParameters`. Done when AC-B01 shows defaults
  after overriding `max_depth`/`eta`.
- **Prerequisites:** TASK-08 (TrainParam for validation reuse). **Blockers:** none.
- **Files/symbols:**
  - Create `src/parameters/mod.rs` (`pub mod tree; pub mod learning; pub mod booster; pub mod training;`).
  - Create `src/parameters/tree.rs`: `struct TreeBoosterParameters`, `struct TreeBoosterParametersBuilder`, `enum GrowPolicy`/`TreeMethod`, `Default`, `build() -> Result<TreeBoosterParameters>`.
  - Modify `src/lib.rs`: `pub mod parameters;`.
- **Red:** `#[test] fn tree_builder_defaults()`.
  `TreeBoosterParametersBuilder::default().max_depth(6).eta(0.3).build().unwrap()`;
  assert `lambda==1.0, gamma==0.0, min_child_weight==1.0, max_bin==256, eta==0.3,
  max_depth==6`. Initial failure: builder missing. Evidence: `docs.rs/xgboost/parameters`.
- **Green:** builder with defaults matching XGBoost; `build` maps to `TrainParam`
  and calls its `validate`.
- **Refactor:** derive `Clone, Debug`; consuming setters returning `Self`.
  Regression: `cargo test --lib parameters::tree`.
- **Validation:** `cargo test --lib tree_builder_defaults`.
- **Completion evidence:** all default assertions pass.
- **Parallelization:** Wave 2 (‖ with D03/B02). Needs T01.

---

### TASK-20 — LearningTaskParameters(+Builder) (SPEC-B02)
- **Goal / completion:** Builder for `objective` (`"reg:squarederror"`), eval
  metric(s), optional `base_score`, `seed`. Done when a built struct exposes those
  fields with correct defaults.
- **Prerequisites:** none. **Blockers:** none.
- **Files/symbols:** Create `src/parameters/learning.rs`:
  `struct LearningTaskParameters`, `LearningTaskParametersBuilder`, `Default`, `build`.
- **Red:** `#[test] fn learning_builder_defaults()`.
  `LearningTaskParametersBuilder::default().build().unwrap()`; assert
  `objective=="reg:squarederror"`, `eval_metric` contains `"rmse"` (or None →
  objective default), `base_score==None`, `seed==0`. Initial failure: builder missing.
  Evidence: `docs.rs/xgboost/parameters::learning`.
- **Green:** struct + builder + defaults.
- **Refactor:** validate objective string against the registry (TASK-05).
  Regression: `cargo test --lib parameters::learning`.
- **Validation:** `cargo test --lib learning_builder_defaults`.
- **Completion evidence:** default assertions pass.
- **Parallelization:** Wave 2 (‖). No deps.

---

### TASK-21 — BoosterParameters + TrainingParameters(+Builder) (SPEC-B03)
- **Goal / completion:** Compose booster type (Tree) + learning params;
  `TrainingParameters` carries `num_round` and `evals`; `.build()` → typed structs
  consumable by `train`. Done when a composed build yields the expected nested config.
- **Prerequisites:** TASK-19 (B01), TASK-20 (B02). **Blockers:** none.
- **Files/symbols:**
  - Create `src/parameters/booster.rs`: `struct BoosterParameters`, `BoosterParametersBuilder`, `enum BoosterType { Tree(TreeBoosterParameters) }`.
  - Create `src/parameters/training.rs`: `struct TrainingParameters`, `TrainingParametersBuilder` (`num_round`, `evals`).
- **Red:** `#[test] fn booster_composes_tree_and_learning()`. Build
  `BoosterParameters` from a Tree booster + learning params; assert the nested
  `max_depth`/`objective` are reachable and `num_round` set on TrainingParameters.
  Initial failure: types missing.
- **Green:** compose structs; `build` validates children.
- **Refactor:** ensure `evals` borrows are lifetime-clean for `train`. Regression:
  `cargo test --lib parameters::`.
- **Validation:** `cargo test --lib booster_composes_tree`.
- **Completion evidence:** nested config reachable.
- **Parallelization:** Wave 3 (‖ with D04). Needs B01+B02.

---

### TASK-22 — `train` entry point (SPEC-A01)
- **Goal / completion:** `train(params, dtrain, num_round, evals) -> Result<Booster>`
  builds a `Learner`, runs `num_round` `update_one_iter`, evaluates `evals` each
  round, returns a `Booster`. Done when a 3-round train on tiny data returns a
  Booster with 3 boosted rounds and prints/collects per-round eval lines.
- **Prerequisites:** TASK-17 (L01), TASK-21 (B03). **Blockers:** none.
- **Files/symbols:**
  - Create `src/api/mod.rs` (`pub mod train; pub mod booster;`), `src/api/train.rs`: `fn train(...)`.
  - Modify `src/lib.rs`: `pub mod api;` and re-export `api::train::train`, `api::booster::Booster`.
- **Red:** `#[test] fn train_runs_num_round_and_returns_booster()`. Tiny regression
  DMatrix; params `num_round=3, max_depth=2, eta=0.3`; when `train`; then
  `booster.boosted_rounds()==3` and training rmse of round 2 < round 0. Initial
  failure: `train` missing. Evidence: `training.py:53`.
- **Green:** construct Learner from params, loop `update_one_iter` + `eval_one_iter`,
  wrap in Booster.
- **Refactor:** collect eval strings into a returned/loggable structure. Regression:
  `cargo test --lib api::train`.
- **Validation:** `cargo test --lib train_runs_num_round`.
- **Completion evidence:** 3 rounds; rmse decreases.
- **Parallelization:** Wave 11 (‖ with A03/A04). Needs L01+B03.

---

### TASK-23 — Booster methods (SPEC-A02)
- **Goal / completion:** `Booster::{update, predict, eval, boosted_rounds,
  num_features}` delegate to `Learner`. Done when predict on train data matches the
  Learner's predict and `num_features()` equals `dtrain.num_col()`.
- **Prerequisites:** TASK-22 (A01), TASK-18 (P01). **Blockers:** none.
- **Files/symbols:** Create `src/api/booster.rs`: `struct Booster { learner: Learner }`,
  delegating methods per SPEC §4 (`save_model`/`load_model`/`get_score` land in
  TASK-24/25).
- **Red:** `#[test] fn booster_delegates_predict_and_shape()`. Train a Booster;
  `booster.predict(&dtrain)` equals `learner.predict(&dtrain,false)` element-wise;
  `booster.num_features()==dtrain.num_col()`; `boosted_rounds()==num_round`. Initial
  failure: methods missing.
- **Green:** thin delegations.
- **Refactor:** hide `Learner` behind the `Booster` API. Regression:
  `cargo test --lib api::booster`.
- **Validation:** `cargo test --lib booster_delegates_predict`.
- **Completion evidence:** predict parity + shape assertions.
- **Parallelization:** Wave 12. Needs A01+P01.

---

### TASK-24 — JSON model save/load round-trip (SPEC-A03) — *separate serialization task*
- **Goal / completion:** `Booster::save_model("m.json")` then `load_model` yields a
  byte-identical re-save and identical predictions; the JSON layout is compatible
  enough to load a real-XGBoost `reg:squarederror` gbtree JSON for the oracle read
  test. Done when AC-A03 passes.
- **Prerequisites:** TASK-14 (G01) + `serde`/`serde_json` dep. **Blockers:** *Open
  question §9.3* — minimal JSON schema fields for round-trip + real-model read;
  resolve by reading a fixture model JSON emitted by TASK-26 and mapping only the
  `learner.gradient_booster.model.trees[*]` + `base_score`/`num_feature` subset.
- **Files/symbols:**
  - Modify `Cargo.toml`: add `serde = { version = "1", features = ["derive"] }`, `serde_json = "1"` (planner to confirm exact latest 1.x via crates.io at impl time).
  - Create `src/model_io/mod.rs` (`pub mod json;`), `src/model_io/json.rs`: `save_json`, `load_json`, serde models mirroring the XGBoost JSON subset.
  - Modify `src/api/booster.rs`: `save_model`, `load_model`.
  - Modify `src/lib.rs`: `pub mod model_io;`.
- **Red (unit, offline):** `#[test] fn json_round_trip_is_byte_identical()` in
  `src/model_io/json.rs`. Train a tiny Booster; `save_model(tmp1)`;
  `load_model(tmp1)`; `save_model(tmp2)`; assert `read(tmp1)==read(tmp2)` bytes and
  predictions equal exactly. This is a pure Rust round-trip — no fixture, runs
  offline on this box.
- **Red (fixture, ENVIRONMENT-GATED):** `#[test] fn reads_real_xgboost_model_json()`,
  marked `#[ignore]`/skip-with-message when `tests/fixtures/<case>/model.json` is
  absent (produced by TASK-26's network `pip install xgboost`). Load the real
  XGBoost model JSON and predict within 1e-5 of the fixture predictions. Initial
  failure: serde models missing. Evidence: `c_api.cc:1546`.
- **Green:** serde structs for the tree/model subset; deterministic field ordering
  for byte-identity; loader tolerant of extra XGBoost fields (`#[serde(default)]`).
- **Refactor:** share the tree model between save and load; UBJSON deferred.
  Regression: `cargo test --lib model_io::json`.
- **Validation:** `cargo test --lib json_round_trip` (offline) · (gated)
  `cargo test --lib reads_real_xgboost_model_json -- --ignored` after TASK-26.
- **Completion evidence:** byte-identical re-save + prediction equality (offline);
  real-model read within 1e-5 where fixtures exist.
- **Parallelization:** Wave 11 (‖ with A01/A04). Needs G01 only.

---

### TASK-25 — Feature importance `get_score` (SPEC-A04)
- **Goal / completion:** `Booster::get_score(importance_type) -> HashMap<String,f64>`
  per feature (keyed `f{idx}`): `weight` = split count; `gain`/`total_gain` = Σ (and
  mean) of the **stored `loss_chg`** (the parent-subtracted split gain persisted by
  TASK-13 on each split node — **NOT** the raw child-sum `CalcSplitGain`);
  `cover`/`total_cover` = Σ (and mean) of node `sum_hess` at each split. This matches
  XGBoost's stored `RegTree` split gain (`ApplyTreeSplit` stores
  `candidate.split.loss_chg`). Done when a hand-built 2-tree ensemble with known
  per-split `loss_chg` and `sum_hess` yields the expected maps.
- **Prerequisites:** TASK-14 (G01), TASK-13 (T06 — nodes must already carry
  `split_gain`(=loss_chg) + `sum_hess`). **Blockers:** none.
- **Files/symbols:** Modify `src/api/booster.rs`: `get_score`; consume the per-split
  `split_gain`(=`loss_chg`) + `sum_hess` on `RegTree` internal nodes (added in
  TASK-13; do not re-derive a child-sum gain here).
- **Red:** `#[test] fn get_score_weight_gain_cover()`. Build a model with two trees,
  each one split on feature 0 with known stored `loss_chg` (e.g. `2.0` and `4.0`)
  and node `sum_hess` (e.g. `5.0` and `7.0`); assert `get_score("weight")["f0"]==2.0`;
  `get_score("gain")["f0"] == 2.0+4.0 == 6.0` (**sum of stored `loss_chg`, not the
  raw child-sum `CalcSplitGain`**); `get_score("cover")["f0"] == 5.0+7.0 == 12.0`;
  keys are `f{idx}`. Initial failure: method missing. Evidence: `core.py:3017`;
  `evaluate_splits.h:416-420`.
- **Green:** traverse trees, aggregate per feature by importance type from the stored
  node fields; `f{idx}` keys.
- **Refactor:** single traversal filling all four maps. Regression:
  `cargo test --lib api::booster`.
- **Validation:** `cargo test --lib get_score_weight_gain_cover`.
- **Completion evidence:** weight/gain(=Σ stored loss_chg)/cover maps match hand
  values; keys `f{idx}`.
- **Parallelization:** Wave 11 (‖ with A01/A03). Needs G01 + T06 (stored node fields).

---

### TASK-26 — Oracle fixture generator (SPEC-X01) — *includes pip-install prerequisite*
- **Goal / completion:** A pinned Python script generates committed golden fixtures
  from a fixed `xgboost==<pinned>` version. Done when running it produces, per case,
  the fixture files listed below with recorded version/config metadata.
- **Prerequisites:** none in Rust. **Blockers / prerequisite step (explicit):**
  **No reference XGBoost is installed** (research.md:126-133: no pip xgboost, no
  libxgboost.so, no R/cmake/nvcc). The generator's first documented step is
  `python3 -m pip install "xgboost==<PINNED>" numpy scipy` over the network.
  *Open question §9.2:* pin the exact version — recommend the released 3.x CPU wheel
  closest to the vendored `3.4.0-dev` line; **confirm at impl time** and record the
  resolved version string in fixture metadata. This is a **network-dependent
  prerequisite task** that must complete before TASK-27 (and the fixture halves of
  TASK-03/T06/A03).
- **Files/symbols:**
  - Create `tests/fixtures/gen.py`: builds `DMatrix` from agaricus + a small
    regression set (from `xgboost/demo/data/regression/machine.data` or a committed
    synthetic CSV), trains `reg:squarederror` with the fixed deterministic config
    (`nthread=1, tree_method=hist, seed=0, subsample=1, colsample_*=1,
    base_score=0.5, max_bin=256, eta=0.3, max_depth=6, num_round=10`), and emits:
    per-round `rmse`, final margin predictions, model JSON (`save_model('*.json')`),
    text tree dump (`get_dump`), `get_score` importances, and
    `get_quantile_cut` output.
  - Create `tests/fixtures/<case>/{meta.json, predictions.json, rmse_per_round.json,
    model.json, tree_dump.txt, importance.json, quantile_cut.json}` (committed).
  - Create `tests/fixtures/README.md`: exact `pip install` command + pinned version +
    regeneration instructions.
- **Red:** N/A production Red (this is a fixture generator, not crate code). The
  "Red" analogue is a `tests/fixtures/gen.py --self-check` assertion inside the
  script: after training, assert the model has 10 trees and reload→predict matches
  in-memory predict; the script exits non-zero if the pinned xgboost is missing.
- **Green:** implement the generator; commit the produced fixtures; record the
  resolved `xgboost.__version__` in `meta.json`.
- **Refactor:** parameterize cases in a small config list; keep outputs stable/sorted
  for reproducibility.
- **Validation:** `python3 tests/fixtures/gen.py` (where xgboost is installed);
  verify the committed fixture files exist and `meta.json` records the version/config.
- **Completion evidence:** committed fixture set present with version+config metadata;
  self-check passes.
- **Parallelization:** Wave 0 (‖ with TASK-01). No Rust dependency; start
  immediately so fixtures exist for Wave 2+ (D03), Wave 7 (T06), Wave 11 (A03),
  Wave 13 (X02). *The one network-dependent task; flag early.*

---

### TASK-27 — Oracle API test, Rust vs fixtures (SPEC-X02) — *headline AC-X02*
- **Goal / completion:** Rust trains/predicts with the identical config and asserts:
  predictions ≤ 1e-5 (abs or rel); tree structure (node count, split feature, split
  bin/condition, default direction) and integer outputs **exact**; per-round `rmse`
  ≤ 1e-5; importance keys/counts exact. Done when AC-X02 passes for agaricus + the
  small regression set.
- **Prerequisites:** TASK-22 (A01), TASK-23 (A02), TASK-18 (P01), TASK-16 (M01),
  TASK-25 (A04), TASK-03 (D03); **committed fixtures from TASK-26**. **Blockers /
  environment gate:** this whole test is **ENVIRONMENT-GATED** — it requires the
  committed golden fixtures produced once by TASK-26's network `pip install
  xgboost==<pinned>` (research.md:126-133: no local xgboost). It is **NOT runnable
  from a bare checkout on this box**; mark it `#[ignore]`/skip-with-clear-message
  when `tests/fixtures/` is absent, exactly like the XGBoost side of the speed test.
  If TASK-03 §9.1 cut algorithm is unresolved, this test surfaces it (fail early on
  `quantile_cut` before training assertions).
- **Files/symbols:** Create `tests/oracle_api.rs`: helpers to load fixtures + a small
  `assert_close(a,b,1e-5)`; test fns `agaricus_matches_fixture`,
  `regression_matches_fixture`, each loading data + config from `meta.json`.
- **Red:** `#[test] fn agaricus_matches_fixture()`. Load agaricus + `meta.json`
  config; `train`; assert (a) `build_cuts` == `quantile_cut.json`; (b) per-round rmse
  == `rmse_per_round.json` within 1e-5; (c) predictions == `predictions.json` within
  1e-5; (d) tree structure == `tree_dump.txt` **exactly** (node count, split
  feature, `split_pt`, `default_left`) — the kRtEps/gamma gate (TASK-11/13) must
  prevent any spurious tiny-gain nodes; (e) `get_score` == `importance.json`
  (keys/counts exact, `gain` = Σ stored `loss_chg`). Initial failure: any divergence
  (or missing fixture → skip/fail with a clear message). Evidence: SPEC §1.5.
- **Green:** N/A production code — this is the acceptance gate. Any failure routes
  back to the responsible task (D03 cuts, T04 kRtEps/gamma gate + loss_chg, T05
  split_pt/missing routing, O03 base_score, A04 importance). Fix there, not here.
- **Refactor:** dedupe fixture-loading across the two cases; add the regression case.
  Regression: full `cargo test --test oracle_api`.
- **Validation (gated):** `cargo test --test oracle_api` — requires committed
  fixtures; skips cleanly on a bare checkout.
- **Completion evidence:** both cases pass all five assertion groups within the
  stated tolerances where fixtures exist — this is the plan's headline success
  condition (AC-X02).
- **Parallelization:** Wave 13. Sequential gate; depends on the full training stack +
  fixtures.

---

### TASK-28 — Oracle speed test (SPEC-X03)
- **Goal / completion:** A bench (modeled on `src/bin/bench.rs`) times Rust `train`
  vs XGBoost `train` on identical data/params (single-thread `hist`), excludes
  data-load/cut time from the timed region, warms up, and reports ms/round + rows/s.
  Rust side always runs; the XGBoost side runs where a reference install exists.
  Done when the Rust harness emits a timing report without error.
- **Prerequisites:** TASK-22 (A01). **Blockers:** the XGBoost comparison half needs a
  reference install (documented, e.g. Kaggle/CPU box) — Rust half is always runnable.
- **Files/symbols:**
  - Create `src/bin/oracle_speed.rs`: builds the DMatrix + `GHistIndex` once
    (outside the timed region), warms up one round, times `num_round` of Rust
    `train`, reports `ms/round` and `rows/s`. Env knobs mirroring bench
    (`SPEED_ROUNDS`, `SPEED_MAXDEPTH`, `SPEED_MAXBIN`) via a small `env_usize` helper
    (pattern from `bench.rs:45-47`).
  - Create `tests/fixtures/speedtest.py`: the XGBoost-side timer (same data/params,
    same excluded regions) for the reference box, referencing
    `xgboost/demo/kaggle-higgs/speedtest.py` conventions.
  - Update `KAGGLE.md` (docs) noting how to run both sides where xgboost is installed.
- **Red:** `#[test] fn speed_harness_reports_positive_throughput()` in
  `src/bin/oracle_speed.rs` (`#[cfg(test)]`): run a tiny timed loop (2 rounds, 100
  rows) and assert the reported `rows/s > 0.0` and `ms/round.is_finite()`. Initial
  failure: harness missing.
- **Green:** implement the timed loop with warmup + excluded setup; print the report.
- **Refactor:** share the report struct; document that the XGBoost side is
  environment-gated. Regression: `cargo test --lib speed_harness` /
  `cargo run --bin oracle_speed`.
- **Validation:** `cargo build --offline` (compiles the bin) · `cargo run --bin
  oracle_speed` (Rust timing) · `python3 tests/fixtures/speedtest.py` (only where
  xgboost installed).
- **Completion evidence:** Rust report emits finite ms/round + positive rows/s; the
  XGBoost comparison runs on a reference box (documented, not required here).
- **Parallelization:** Wave 14. Needs A01; independent of X02.

---

## 5. Cross-cutting notes, risks, and unresolved blockers

**Determinism (SPEC §9 float-summation risk):** every task that accumulates uses
`f64` over `f32` and single-thread order. No `rayon` in Phase 1. Parent−child
histogram subtraction (TASK-13) is *optional* and only permitted if it stays within
the 1e-5 bar; otherwise build each node directly.

**GPU independence:** no task adds a GPU/CubeCL runtime requirement. `gpu::*` is
imported only for the `GradientPair` type (a plain derive struct, not Pod; usable
without a device); the crate already builds offline. All `cargo test --lib`
(offline unit) commands run without Vulkan/CUDA. The fixture-based `--test
oracle_cuts`/`oracle_tree`/`oracle_api` are separately environment-gated (need
TASK-26 fixtures), not GPU-gated.

**Unresolved blockers / assumptions carried from SPEC §9 (surfaced, not resolved):**
1. **§9.1 — exact `hist` cut algorithm.** Resolved in scope to **both** branches
   (sorted-distinct for distinct ≤ max_bin; weighted-quantile sketch for distinct >
   max_bin — the continuous fixture); the remaining unknown is exact cut/boundary
   values, resolved by reading `xgboost/src/common/quantile*`/`hist_util*` and
   diffing the `quantile_cut` fixture. Owned by **TASK-03** (both branches shipped).
   Highest-risk parity item; asserted first (AC-D03) before any training test is
   trusted.
2. **§9.2 — pinned xgboost version** for fixtures. Owned by **TASK-26**; must be
   confirmed at impl time and recorded in `meta.json`. Network `pip install` is a
   hard prerequisite for all oracle tests.
3. **§9.3 — minimal JSON model schema** for round-trip + real-model read. Owned by
   **TASK-24**; resolve against a TASK-26 fixture `model.json`.
4. **§9.4 — depthwise-only sufficiency** for Phase-1 parity (lossguide deferred).
   Assumed **yes**; validated by AC-X02 in **TASK-27**. If parity fails only under
   lossguide-shaped trees, escalate as a new task (out of Phase-1 scope).

**Dependency-version confirmation (deferred to impl, per SPEC §3):** exact `serde`
/ `serde_json` 1.x versions and the LIBSVM-parser choice (recommended hand-rolled,
no new dep) to be pinned when TASK-02/TASK-24 land. No CubeCL/GPU deps added.

**TreeFinder:** the `tree_finder` MCP is a corpus of Markdown *documents*, not the
crate's spec store configured by a `planning/settings.json` (none exists in this
repo). Per the task brief, this PLAN.md is authored to the specified path and is the
authoritative planning artifact; no TreeFinder upsert was requested or performed.

---

## 6. Definition of done (phase)

- All 28 tasks complete; `cargo build --offline` and `cargo test --lib` green.
- `cargo test --test oracle_cuts`, `--test oracle_tree`, `--test oracle_api` green
  against committed fixtures (AC-D03, AC-T06, **AC-X02** headline).
- `cargo run --bin oracle_speed` emits a Rust timing report; XGBoost-side comparison
  documented for a reference box.
- No existing public symbol changed; only additive `pub mod` in `src/lib.rs` and
  additive deps in `Cargo.toml` (SPEC §7/§8).
</content>
</invoke>
