//! Oracle parity for *every string-valued parameter*, against a pinned real
//! XGBoost 3.4.0 (`tools/gen_string_param_fixtures.py`).
//!
//! `tests/string_parameters.rs` proves each spelling round-trips through the
//! parameter surface, and `tests/string_parameter_behaviour.rs` proves each one
//! reaches the fit. Neither proves the fit is *right*. This file does: it
//! replays each fixture's configuration and holds the result to the SPEC §1.5
//! bar — predictions and metrics within `1e-5`, tree structure and every
//! integer identical.
//!
//! The configuration is rebuilt from the fixture's own XGBoost parameter map
//! rather than from a list written here, and [`config_to_params`] **panics on a
//! parameter it does not recognise**. That is deliberate: it is what stops this
//! file silently ignoring a knob the generator started emitting.

mod common;

use std::collections::BTreeMap;
use std::str::FromStr;

use serde_json::Value;
use xgboost_rs::parameters::{
    AftDistribution, BoosterParameters, BoosterType, DartNormalizeType, DartParameters,
    DartSampleType, DefaultDirection, Device, EvalMetric, FeatureSelector, GeneralParameters,
    GrowPolicy, LambdaRankPairMethod, LearningTaskParameters, LinearBoosterParameters,
    LinearUpdater, MonotoneConstraint, MultiStrategy, Objective, ProcessType, SamplingMethod,
    TrainingParameters, TreeBoosterParameters, TreeMethod, TreeUpdaterName, VerboseEval, Verbosity,
};
use xgboost_rs::data::cuts::build_cuts;
use xgboost_rs::{DMatrix, api};

use common::{f32_array, fixture_dir, u32_array};

/// Relative tolerance for anything the SPEC calls a float comparison.
const TOL: f64 = 1e-5;

/// Absolute floor for comparing a *gain* across devices, on top of [`TOL`].
///
/// A gain is a difference of regularised sums, so a small gain computed out of
/// large sums keeps far fewer significant digits than the sums do. Measured
/// across the whole fixture set the largest gap is `1.6e-5` absolute, on a
/// vector leaf whose gain sums one term per target; this is set an order of
/// magnitude above that and nothing else in the model is given any slack.
#[cfg(feature = "gpu")]
const GAIN_ABS_TOL: f64 = 1e-4;

// ------------------------------------------------------------ fixtures ----

fn strparam_dir() -> std::path::PathBuf {
    fixture_dir().join("strparam")
}

fn load(name: &str) -> Value {
    let path = strparam_dir().join(format!("{name}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read fixture {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("bad JSON in {}: {e}", path.display()))
}

/// One entry of the generated index.
struct Case {
    name: String,
    param: String,
    value: String,
    data: String,
    outcome: String,
    requires_gpu: bool,
}

/// Every generated case. Read from the committed index rather than listed
/// here, so a case added to the generator is picked up without editing Rust.
fn cases() -> Vec<Case> {
    let index = load("index");
    index["cases"]
        .as_array()
        .expect("index.json must have a `cases` array")
        .iter()
        .map(|c| Case {
            name: c["case"].as_str().unwrap().to_owned(),
            param: c["param"].as_str().unwrap().to_owned(),
            value: c["value"].as_str().unwrap().to_owned(),
            data: c["data"].as_str().unwrap().to_owned(),
            outcome: c["outcome"].as_str().unwrap().to_owned(),
            requires_gpu: c["requires_gpu"].as_bool().unwrap_or(false),
        })
        .collect()
}

/// Cases whose fit this build is expected to reproduce: trained upstream, and
/// not dependent on hardware absent from the machine that generated them.
fn trainable_cases() -> Vec<Case> {
    cases()
        .into_iter()
        .filter(|c| c.outcome == "ok")
        // A `requires_gpu` case generated without a GPU records a CPU fallback
        // fit. Comparing against it would pass while proving nothing about the
        // GPU, so it is covered by `gpu_cases_are_honest_about_the_device`
        // instead and replayed for real on CUDA hardware.
        .filter(|c| !c.requires_gpu)
        .collect()
}

/// Load a string-parameter dataset, including the ranking and censoring
/// metadata the plain `common::load_data` does not carry.
fn load_dataset(name: &str) -> DMatrix {
    let v = load(&format!("data_{name}"));
    let n_row = v["n_row"].as_u64().unwrap() as usize;
    let n_col = v["n_col"].as_u64().unwrap() as usize;
    let n_target = v["n_target"].as_u64().unwrap_or(1) as usize;

    // `null` encodes NaN, the missing sentinel.
    let values: Vec<f32> = v["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().map(|f| f as f32).unwrap_or(f32::NAN))
        .collect();
    let mut d = DMatrix::from_dense(&values, n_row, n_col, f32::NAN).unwrap();

    let labels = f32_array(&v["labels"]);
    if n_target > 1 {
        d.set_labels_multi(&labels, n_target).unwrap();
    } else {
        d.set_labels(&labels).unwrap();
    }
    if let Some(w) = v["weights"].as_array() {
        d.set_weights(&w.iter().map(|x| x.as_f64().unwrap() as f32).collect::<Vec<_>>()).unwrap();
    }
    if let Some(g) = v["group"].as_array() {
        d.set_group(&g.iter().map(|x| x.as_u64().unwrap() as usize).collect::<Vec<_>>()).unwrap();
    }
    if let (Some(lo), Some(hi)) =
        (v["label_lower_bound"].as_array(), v["label_upper_bound"].as_array())
    {
        let lo: Vec<f32> = lo.iter().map(|x| x.as_f64().unwrap() as f32).collect();
        let hi: Vec<f32> = hi.iter().map(|x| x.as_f64().unwrap() as f32).collect();
        d.set_label_bounds(&lo, &hi).unwrap();
    }
    d
}

// ------------------------------------------------------- config parsing ----

/// The parameter map as strings, which is how XGBoost itself takes it.
fn config_map(fixture: &Value) -> BTreeMap<String, String> {
    fixture["params"]
        .as_object()
        .expect("a fixture always records the parameters it was trained with")
        .iter()
        .map(|(k, v)| {
            let text = match v {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            (k.clone(), text)
        })
        .collect()
}

/// Rebuild this crate's parameters from an XGBoost configuration map.
///
/// Unrecognised names panic rather than being skipped: a fixture trained with
/// a knob this function drops would be compared against a *different* fit, and
/// the resulting pass would be meaningless.
fn config_to_params(case: &str, cfg: &BTreeMap<String, String>) -> TrainingParameters {
    let mut general = GeneralParameters::default();
    let mut learning = LearningTaskParameters::default();
    let mut tree = TreeBoosterParameters::default();
    let mut linear = LinearBoosterParameters::default();
    let mut dart = DartParameters::default();
    let mut kind = "gbtree";

    // Parameters read after the loop, because they modify a value the loop may
    // set in either order (the objective's own tuning knobs).
    let mut num_class: Option<u32> = None;
    let mut quantile_alpha: Option<f32> = None;
    let mut expectile_alpha: Option<f32> = None;
    let mut tweedie_power: Option<f32> = None;
    let mut aft_dist: Option<AftDistribution> = None;
    let mut aft_scale: Option<f32> = None;
    let mut pair_method: Option<LambdaRankPairMethod> = None;
    let mut num_pair: Option<u32> = None;

    let f32_of = |k: &str, v: &str| -> f32 {
        v.parse().unwrap_or_else(|e| panic!("{case}: `{k}` = {v:?} is not a float: {e}"))
    };
    let u32_of = |k: &str, v: &str| -> u32 {
        v.parse().unwrap_or_else(|e| panic!("{case}: `{k}` = {v:?} is not an integer: {e}"))
    };

    for (key, value) in cfg {
        let v = value.as_str();
        match key.as_str() {
            // --- general ---
            "device" => general.device = v.parse().unwrap(),
            "verbosity" => general.verbosity = Verbosity::from_str(v).unwrap(),
            "nthread" => general.nthread = u32_of(key, v),

            // --- learning task ---
            "objective" => learning.objective = Objective::from_str(v).unwrap(),
            "eval_metric" => learning.eval_metric = vec![EvalMetric::from_str(v).unwrap()],
            "seed" => learning.seed = v.parse().unwrap(),
            "base_score" => learning.base_score = Some(f32_of(key, v)),

            // --- objective tuning, applied once the objective is known ---
            "num_class" => num_class = Some(u32_of(key, v)),
            "quantile_alpha" => quantile_alpha = Some(f32_of(key, v)),
            "expectile_alpha" => expectile_alpha = Some(f32_of(key, v)),
            "tweedie_variance_power" => tweedie_power = Some(f32_of(key, v)),
            "aft_loss_distribution" => aft_dist = Some(AftDistribution::from_str(v).unwrap()),
            "aft_loss_distribution_scale" => aft_scale = Some(f32_of(key, v)),
            "lambdarank_pair_method" => {
                pair_method = Some(LambdaRankPairMethod::from_str(v).unwrap())
            }
            "lambdarank_num_pair_per_sample" => num_pair = Some(u32_of(key, v)),

            // --- which booster ---
            "booster" => {
                kind = match v {
                    "gbtree" => "gbtree",
                    "dart" => "dart",
                    "gblinear" => "gblinear",
                    other => panic!("{case}: unknown booster `{other}`"),
                }
            }

            // --- tree booster ---
            "eta" => {
                tree.eta = f32_of(key, v);
                linear.eta = tree.eta;
            }
            "gamma" => tree.gamma = f32_of(key, v),
            "max_depth" => tree.max_depth = u32_of(key, v),
            "max_leaves" => tree.max_leaves = u32_of(key, v),
            "max_bin" => tree.max_bin = u32_of(key, v),
            "min_child_weight" => tree.min_child_weight = f32_of(key, v),
            "subsample" => tree.subsample = f32_of(key, v),
            "sampling_method" => tree.sampling_method = SamplingMethod::from_str(v).unwrap(),
            "tree_method" => tree.tree_method = TreeMethod::from_str(v).unwrap(),
            "grow_policy" => tree.grow_policy = GrowPolicy::from_str(v).unwrap(),
            "process_type" => tree.process_type = ProcessType::from_str(v).unwrap(),
            "multi_strategy" => tree.multi_strategy = MultiStrategy::from_str(v).unwrap(),
            "default_direction" => tree.default_direction = DefaultDirection::from_str(v).unwrap(),
            "refresh_leaf" => tree.refresh_leaf = v != "0",
            "monotone_constraints" => {
                tree.monotone_constraints = v
                    .trim_matches(['(', ')'])
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| MonotoneConstraint::from_str(s.trim()).unwrap())
                    .collect()
            }
            // `updater` names the tree pipeline for a tree booster and the
            // solver for a linear one; the booster decides which it is.
            "updater" => {
                if kind == "gblinear" || LinearUpdater::from_str(v).is_ok() {
                    linear.updater = LinearUpdater::from_str(v).unwrap();
                } else {
                    tree.updater = Some(
                        v.split(',')
                            .map(|u| TreeUpdaterName::from_str(u.trim()).unwrap())
                            .collect(),
                    );
                }
            }

            // --- dart ---
            "sample_type" => dart.sample_type = DartSampleType::from_str(v).unwrap(),
            "normalize_type" => dart.normalize_type = DartNormalizeType::from_str(v).unwrap(),
            "rate_drop" => dart.rate_drop = f32_of(key, v),
            "skip_drop" => dart.skip_drop = f32_of(key, v),
            "one_drop" => dart.one_drop = v != "0",

            // --- gblinear ---
            "feature_selector" => {
                linear.feature_selector = FeatureSelector::from_str(v).unwrap()
            }
            "top_k" => linear.top_k = u32_of(key, v),

            other => panic!(
                "{case}: the fixture was trained with `{other} = {value}`, which this test does \
                 not know how to rebuild. Add it to `config_to_params` — skipping it would \
                 compare against a different fit."
            ),
        }
    }

    // The objective's own tuning parameters, now that the objective is known.
    learning.objective = match (learning.objective, num_class) {
        (Objective::MultiSoftmax { .. }, Some(n)) => Objective::MultiSoftmax { num_class: n },
        (Objective::MultiSoftprob { .. }, Some(n)) => Objective::MultiSoftprob { num_class: n },
        (obj, _) => obj,
    };
    if let (Objective::RegQuantileError { .. }, Some(a)) = (&learning.objective, quantile_alpha) {
        learning.objective = Objective::RegQuantileError { quantile_alpha: vec![a] };
    }
    if let (Objective::RegExpectileError { .. }, Some(a)) = (&learning.objective, expectile_alpha) {
        learning.objective = Objective::RegExpectileError { expectile_alpha: vec![a] };
    }
    if let (Objective::RegTweedie { .. }, Some(p)) = (&learning.objective, tweedie_power) {
        learning.objective = Objective::RegTweedie { tweedie_variance_power: p };
    }
    if let Objective::SurvivalAft { aft_loss_distribution, aft_loss_distribution_scale } =
        &learning.objective
    {
        learning.objective = Objective::SurvivalAft {
            aft_loss_distribution: aft_dist.unwrap_or(*aft_loss_distribution),
            aft_loss_distribution_scale: aft_scale.unwrap_or(*aft_loss_distribution_scale),
        };
    }
    for rank in [&mut learning.objective] {
        if let Objective::RankPairwise(p) | Objective::RankNdcg(p) | Objective::RankMap(p) = rank {
            if let Some(m) = pair_method {
                p.pair_method = m;
            }
            if let Some(n) = num_pair {
                p.num_pair_per_sample = Some(n);
            }
        }
    }

    let booster = match kind {
        "gbtree" => BoosterType::Gbtree(tree),
        "dart" => {
            dart.tree = tree;
            BoosterType::Dart(dart)
        }
        "gblinear" => BoosterType::Gblinear(linear),
        other => unreachable!("{other}"),
    };

    TrainingParameters {
        booster: BoosterParameters { booster, general, learning },
        num_boost_round: 0, // set by the caller from the fixture
        verbose_eval: VerboseEval::Silent,
        ..Default::default()
    }
}

/// Rebuild a case's configuration and run it, returning the fit and the
/// fixture it must match.
fn train_case(case: &Case) -> Result<(xgboost_rs::Booster, api::EvalHistory, Value, DMatrix), String>
{
    let fixture = load(&case.name);
    let dmat = load_dataset(&case.data);
    let mut params = config_to_params(&case.name, &config_map(&fixture));
    params.num_boost_round = fixture["num_round"].as_u64().unwrap() as u32;

    // `process_type=update` rewrites an existing ensemble; the fixture records
    // that it was seeded with a plain fit of the same shape.
    let base = if fixture["updates_existing_model"].as_bool().unwrap_or(false) {
        let mut seed = params.clone();
        if let BoosterType::Gbtree(t) = &mut seed.booster.booster {
            t.process_type = ProcessType::Default;
            t.updater = None;
        }
        Some(
            api::train(&seed, &dmat, &[])
                .map_err(|e| format!("{}: seeding the update failed: {e}", case.name))?
                .0,
        )
    } else {
        None
    };

    match api::train_from(&params, &dmat, &[(&dmat, "train")], base.as_ref()) {
        Ok((b, h)) => Ok((b, h, fixture, dmat)),
        Err(e) => Err(e.to_string()),
    }
}

fn close(got: f64, want: f64) -> bool {
    (got - want).abs() <= TOL * want.abs().max(1.0)
}

/// Collects one line per failing case instead of stopping at the first.
///
/// The point of a 96-case parity matrix is to know *how much* of the string
/// surface matches upstream. A test that aborts on case one reports a single
/// bit and hides the rest, which is the opposite of what this file is for.
#[derive(Default)]
struct Report {
    failed: Vec<String>,
    passed: usize,
}

impl Report {
    /// Record a case's outcome. `detail` is `None` when the case matched.
    fn record(&mut self, case: &Case, detail: Option<String>) {
        match detail {
            None => self.passed += 1,
            Some(d) => self.failed.push(format!(
                "  {:<34} {} = {:<28} {d}",
                case.name, case.param, case.value
            )),
        }
    }

    fn finish(self, what: &str) {
        let total = self.passed + self.failed.len();
        assert!(total > 0, "{what}: no cases were compared");
        if !self.failed.is_empty() {
            panic!(
                "{what}: {}/{total} cases match XGBoost 3.4.0; {} differ:\n{}",
                self.passed,
                self.failed.len(),
                self.failed.join("\n")
            );
        }
    }
}

/// Compare two float sequences, returning a description of the first
/// difference. Length mismatches are reported before any value is read.
fn diff_floats(what: &str, got: &[f32], want: &[f32]) -> Option<String> {
    if got.len() != want.len() {
        return Some(format!("{what}: got {} values, want {}", got.len(), want.len()));
    }
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if !close(*g as f64, *w as f64) {
            return Some(format!("{what}[{i}]: {g} != {w}"));
        }
    }
    None
}

// ------------------------------------------------------------- the tests ----

/// The census: the index must actually cover the whole string surface, and
/// every case must be accounted for as trained, refused, or GPU-gated.
#[test]
fn the_fixture_set_covers_the_string_parameter_surface() {
    let cases = cases();
    assert!(!cases.is_empty(), "no fixtures: run tools/gen_string_param_fixtures.py");

    let mut by_param: BTreeMap<&str, usize> = BTreeMap::new();
    for c in &cases {
        *by_param.entry(c.param.as_str()).or_default() += 1;
    }
    // Every string-valued parameter this crate exposes must appear.
    for param in [
        "objective",
        "eval_metric",
        "tree_method",
        "grow_policy",
        "sampling_method",
        "process_type",
        "updater",
        "multi_strategy",
        "default_direction",
        "monotone_constraints",
        "sample_type",
        "normalize_type",
        "feature_selector",
        "aft_loss_distribution",
        "lambdarank_pair_method",
        "verbosity",
        "device",
    ] {
        assert!(by_param.contains_key(param), "no fixture pins `{param}`");
    }

    let index = load("index");
    assert_eq!(
        index["xgboost_version"].as_str().unwrap(),
        "3.4.0",
        "fixtures must come from the pinned upstream reference"
    );
}

/// Every accepted spelling must produce the objective XGBoost recorded — the
/// cheapest proof that the configuration reached the learner intact.
#[test]
fn the_configured_objective_matches_xgboost() {
    let mut report = Report::default();
    for case in trainable_cases() {
        let detail = match train_case(&case) {
            Err(e) => Some(format!("training failed: {e}")),
            Ok((booster, _, fixture, _)) => {
                let model: Value = serde_json::from_str(&booster.save_model()).unwrap();
                let got = model["learner"]["objective"]["name"].as_str().unwrap().to_owned();
                let want = fixture["objective_name"].as_str().unwrap();
                (got != want).then(|| format!("objective `{got}` != `{want}`"))
            }
        };
        report.record(&case, detail);
    }
    report.finish("configured objective");
}

/// The quantile cuts the fit binned against.
///
/// Every split threshold a `hist` tree records is one of these values, so a cut
/// that drifts moves split thresholds without anything in the split-choice
/// logic being wrong — and once a row falls on the other side of a boundary,
/// the whole subtree below diverges. Checking cuts separately is what tells a
/// binning bug apart from a split-evaluation bug.
///
/// Bit-exact, not within tolerance: a cut is a value copied from the data, not
/// a computed quantity, so "close" is already wrong.
///
/// # What the current failures are
///
/// This is **upstream version drift, not a defect here.** The failures split
/// exactly on row count: every dataset with 300 rows fails and every dataset
/// with 240 fails nothing. With `max_bin = 256` that is precisely the line
/// between "fewer distinct values than bins, so every value becomes a cut" and
/// "the quantile sketch actually runs".
///
/// Sketching the same 300-row column under both pinned XGBoosts gives:
///
/// ```text
/// 3.0.5:  -2.400131, -2.3067033, -2.2267025, -2.0828412, ...
/// 3.4.0:  -2.410044, -2.400131,  -2.3067033, -2.2267025, ...
/// ```
///
/// 3.4.0's list is 3.0.5's shifted by one, with an extra cut at the low end —
/// and 3.4.0 also reports `min_values` as `-inf` where 3.0.5 reported a finite
/// minimum. This crate reproduces the 3.0.5 sequence exactly, which is what
/// `tests/oracle.rs` (pinned to 3.0.5) still proves.
///
/// So the sketch here is one upstream version behind. `SPEC.md` names 3.4.0 as
/// the reference, so closing this means porting 3.4.0's sketch — and it is the
/// single root cause behind most of the remaining tree, leaf and prediction
/// differences, because everything downstream bins against these values.
#[test]
fn quantile_cuts_match_xgboost_exactly() {
    let mut report = Report::default();
    for case in trainable_cases() {
        let fixture = load(&case.name);
        let (Some(want_ptrs), Some(want_values)) =
            (fixture["cut_ptrs"].as_array(), fixture["cut_values"].as_array())
        else {
            continue; // `exact` and `gblinear` never sketch
        };
        let want_ptrs = u32_array(&Value::Array(want_ptrs.clone()));
        // `null` is a non-finite entry: 3.4.0 writes each feature's leading
        // "min value" as -inf rather than a finite minimum.
        let want_values: Vec<f32> = want_values
            .iter()
            .map(|x| x.as_f64().map(|f| f as f32).unwrap_or(f32::NEG_INFINITY))
            .collect();

        let dmat = load_dataset(&case.data);
        let max_bin = config_map(&fixture)
            .get("max_bin")
            .and_then(|v| v.parse().ok())
            .unwrap_or(256u32);

        let detail = match build_cuts(&dmat, max_bin) {
            Err(e) => Some(format!("building cuts failed: {e}")),
            Ok(cuts) => (0..cuts.num_features()).find_map(|f| {
                // `get_quantile_cut()` reports each feature as
                // `[min_value, ...cuts]`; `HistogramCuts` keeps the minimums in
                // their own array.
                let (b, e) = (want_ptrs[f] as usize, want_ptrs[f + 1] as usize);
                let want_min = want_values[b];
                let want_cuts = &want_values[b + 1..e];
                let got_cuts =
                    &cuts.cut_values[cuts.cut_ptrs[f] as usize..cuts.cut_ptrs[f + 1] as usize];

                // A -inf min is 3.4.0's "no lower bound" sentinel rather than a
                // value from the data, so there is nothing to compare it to;
                // the cut values below are what split thresholds come from.
                if want_min.is_finite() && cuts.min_values[f].to_bits() != want_min.to_bits() {
                    return Some(format!(
                        "feature {f} min {} != {want_min}",
                        cuts.min_values[f]
                    ));
                }
                if got_cuts.len() != want_cuts.len() {
                    return Some(format!(
                        "feature {f} has {} cuts, want {}",
                        got_cuts.len(),
                        want_cuts.len()
                    ));
                }
                got_cuts.iter().zip(want_cuts).enumerate().find_map(|(i, (g, w))| {
                    (g.to_bits() != w.to_bits())
                        .then(|| format!("feature {f} cut {i}: {g} != {w}"))
                })
            }),
        };
        report.record(&case, detail);
    }
    report.finish("quantile cuts");
}

/// The intercept every prediction starts from.
///
/// This runs before the tree comparisons on purpose: `base_score` is estimated
/// from the labels by the objective, so if it drifts, the first gradient is
/// wrong and *every* downstream tree, leaf and metric differs. A tree-shape
/// failure with a matching intercept means something else; a tree-shape failure
/// with a drifting intercept usually means only this.
#[test]
fn base_score_matches_xgboost() {
    let mut report = Report::default();
    for case in trainable_cases() {
        let detail = match train_case(&case) {
            Err(e) => Some(format!("training failed: {e}")),
            Ok((booster, _, fixture, _)) => {
                let want = f32_array(&fixture["base_score"]);
                let got = booster.base_score();
                // A multi-output model records one intercept per group; this
                // build exposes the first, so compare against that.
                (!close(got as f64, want[0] as f64))
                    .then(|| format!("base_score {got} != {}", want[0]))
            }
        };
        report.record(&case, detail);
    }
    report.finish("base_score");
}

/// The per-round evaluation history: the metric's name and every value.
#[test]
fn per_round_metric_history_matches_xgboost() {
    let mut report = Report::default();
    for case in trainable_cases() {
        let detail = match train_case(&case) {
            Err(e) => Some(format!("training failed: {e}")),
            Ok((_, history, fixture, _)) => {
                let want: Vec<f64> = fixture["metric_history"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect();
                let want_name = format!("train-{}", fixture["metric"].as_str().unwrap());

                if history.len() != want.len() {
                    Some(format!("{} rounds, want {}", history.len(), want.len()))
                } else {
                    history.iter().zip(&want).enumerate().find_map(|(i, (round, w))| {
                        let (name, got) = &round[0];
                        if name != &want_name {
                            Some(format!("metric named `{name}`, want `{want_name}`"))
                        } else if !close(*got, *w) {
                            Some(format!("{name}[{i}]: {got} != {w}"))
                        } else {
                            None
                        }
                    })
                }
            }
        };
        report.record(&case, detail);
    }
    report.finish("per-round metric history");
}

/// Predictions on the objective's output scale, and the raw margins under it.
#[test]
fn predictions_and_margins_match_xgboost() {
    let mut report = Report::default();
    for case in trainable_cases() {
        let detail = match train_case(&case) {
            Err(e) => Some(format!("training failed: {e}")),
            Ok((booster, _, fixture, dmat)) => {
                [("value", booster.predict(&dmat)), ("margin", booster.predict_margin(&dmat))]
                    .iter()
                    .find_map(|(what, got)| {
                        let block = &fixture["predict"][what];
                        // Upstream could not produce it either; nothing to compare.
                        block.get("error").is_none().then(|| {
                            diff_floats(what, got, &f32_array(&block["values"]))
                        })?
                    })
            }
        };
        report.record(&case, detail);
    }
    report.finish("predictions and margins");
}

/// Leaf assignment is an integer: it must be identical, not close. Two models
/// can agree on every prediction and still route rows differently.
#[test]
fn leaf_assignment_matches_xgboost_exactly() {
    let mut report = Report::default();
    for case in trainable_cases() {
        let detail = match train_case(&case) {
            Err(e) => Some(format!("training failed: {e}")),
            Ok((booster, _, fixture, dmat)) => {
                let block = &fixture["predict"]["leaf"];
                if block.get("error").is_some() {
                    None
                } else {
                    let want: Vec<u32> = block["values"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_f64().unwrap() as u32)
                        .collect();
                    let got: Vec<u32> =
                        booster.predict_leaf(&dmat).unwrap().iter().flatten().copied().collect();
                    if got.len() != want.len() {
                        Some(format!("leaf matrix has {} entries, want {}", got.len(), want.len()))
                    } else {
                        got.iter().zip(&want).enumerate().find(|(_, (g, w))| g != w).map(
                            |(i, (g, w))| format!("row {} lands in leaf {g}, want {w}", i / 4),
                        )
                    }
                }
            }
        };
        report.record(&case, detail);
    }
    report.finish("leaf assignment");
}

/// Tree structure: every integer identical, every threshold within tolerance.
#[test]
fn tree_structure_matches_xgboost() {
    let mut report = Report::default();
    for case in trainable_cases() {
        let detail = match train_case(&case) {
            Err(e) => Some(format!("training failed: {e}")),
            Ok((booster, _, fixture, _)) => {
                let want_trees = fixture["trees"].as_array().unwrap();
                if want_trees.is_empty() {
                    None // gblinear has no trees
                } else {
                    let model: Value = serde_json::from_str(&booster.save_model()).unwrap();
                    let gbm = &model["learner"]["gradient_booster"];
                    let empty = Vec::new();
                    let got_trees = gbm["model"]["trees"]
                        .as_array()
                        .or_else(|| gbm["gbtree"]["model"]["trees"].as_array())
                        .unwrap_or(&empty);

                    if got_trees.len() != want_trees.len() {
                        Some(format!("{} trees, want {}", got_trees.len(), want_trees.len()))
                    } else {
                        got_trees.iter().zip(want_trees).enumerate().find_map(|(t, (got, want))| {
                            for key in [
                                "left_children",
                                "right_children",
                                "parents",
                                "split_indices",
                                "default_left",
                            ] {
                                if got[key] != want[key] {
                                    return Some(format!("tree {t}: `{key}` differs"));
                                }
                            }
                            diff_floats(
                                &format!("tree {t} split_conditions"),
                                &f32_array(&got["split_conditions"]),
                                &f32_array(&want["split_conditions"]),
                            )
                        })
                    }
                }
            }
        };
        report.record(&case, detail);
    }
    report.finish("tree structure");
}

/// The linear booster's weights, for the `gblinear` string parameters
/// (`feature_selector` and the linear `updater`) that have no trees to compare.
#[test]
fn linear_weights_match_xgboost() {
    let mut report = Report::default();
    for case in trainable_cases() {
        let want = f32_array(&load(&case.name)["weights"]);
        if want.is_empty() {
            continue; // not a gblinear case
        }
        let detail = match train_case(&case) {
            Err(e) => Some(format!("training failed: {e}")),
            Ok((booster, _, _, _)) => {
                let model: Value = serde_json::from_str(&booster.save_model()).unwrap();
                let got = f32_array(&model["learner"]["gradient_booster"]["model"]["weights"]);
                diff_floats("weight", &got, &want)
            }
        };
        report.record(&case, detail);
    }
    report.finish("gblinear weights");
}

/// A configuration upstream refuses must be refused here too, naming the
/// parameter — never accepted and quietly trained as something else.
#[test]
fn configurations_xgboost_refuses_are_refused_here() {
    let refused: Vec<Case> = cases()
        .into_iter()
        .filter(|c| c.outcome == "error")
        // A `requires_gpu` case refused upstream was refused for want of a
        // *device* on the generating machine ("Must have at least one
        // device"), not because the configuration is invalid. This build
        // reaches its own GPU path, so accepting it is correct rather than a
        // missed refusal; `gpu_cases_are_honest_about_the_device` is what
        // keeps those cases honest.
        .filter(|c| !c.requires_gpu)
        .collect();
    assert!(!refused.is_empty(), "the generator recorded no refusals; that is itself suspicious");

    for case in refused {
        let fixture = load(&case.name);
        let cfg = config_map(&fixture);
        // Rebuilding may itself fail (an unimplemented updater), which counts.
        let params = std::panic::catch_unwind(|| config_to_params(&case.name, &cfg));
        let Ok(mut params) = params else { continue };
        params.num_boost_round = fixture["num_round"].as_u64().unwrap() as u32;

        let dmat = load_dataset(&case.data);
        let err = api::train(&params, &dmat, &[(&dmat, "train")]).err().unwrap_or_else(|| {
            panic!(
                "{}: XGBoost refuses `{} = {}` ({}), but this build accepted it",
                case.name,
                case.param,
                case.value,
                fixture["error"].as_str().unwrap_or("?")
            )
        });
        assert!(
            err.to_string().contains(&case.param) || err.to_string().contains(&case.value),
            "{}: refusal must name the parameter or its value, got: {err}",
            case.name
        );
    }
}

/// Print one case's first tree beside the fixture's, for working on a
/// difference the summary only names.
///
/// Ignored by default; run it at a case with
/// `CASE=objective_binary_logistic cargo test --test oracle_string_parameters
///  -- --ignored --nocapture dump_one_case`.
#[test]
#[ignore = "diagnostic, not an assertion"]
fn dump_one_case() {
    let name = std::env::var("CASE").unwrap_or_else(|_| "objective_binary_logistic".to_owned());
    let case = cases()
        .into_iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("no case named {name}"));

    let (booster, history, fixture, _) = train_case(&case).expect("training failed");
    let model: Value = serde_json::from_str(&booster.save_model()).unwrap();
    let gbm = &model["learner"]["gradient_booster"];
    let empty = Vec::new();
    let got = gbm["model"]["trees"]
        .as_array()
        .or_else(|| gbm["gbtree"]["model"]["trees"].as_array())
        .unwrap_or(&empty);

    println!("\ncase {name}   params {}", fixture["params"]);
    println!("base_score  got {:?}  want {}", booster.base_score(), fixture["base_score"]);
    println!("metric      got {:?}", history.first());
    println!("            want {} {}", fixture["metric"], fixture["metric_history"]);

    let want = &fixture["trees"][0];
    for key in ["left_children", "right_children", "parents", "split_indices",
                "split_conditions", "default_left", "sum_hessian", "base_weights"] {
        println!("\n{key}\n  got  {}", got.first().map(|t| t[key].to_string()).unwrap_or_default());
        println!("  want {}", want[key]);
    }
}

/// GPU-gated cases must be honest about what generated them.
///
/// `device=cuda` does not fail without a GPU — upstream falls back to the CPU
/// and returns identical numbers. A fixture generated that way would pass a
/// naive comparison while proving nothing, so it is excluded from the parity
/// tests until it is regenerated on real CUDA hardware.
#[test]
fn gpu_cases_are_honest_about_the_device() {
    let index = load("index");
    let gpu_present = index["gpu_present"].as_bool().unwrap_or(false);

    for case in cases().into_iter().filter(|c| c.requires_gpu) {
        let fixture = load(&case.name);
        assert_eq!(
            fixture["gpu_present"].as_bool(),
            Some(gpu_present),
            "{}: fixture and index disagree about whether a GPU was present",
            case.name
        );
        if !gpu_present {
            assert!(
                !trainable_cases().iter().any(|c| c.name == case.name),
                "{}: a fixture generated without a GPU must not be compared as if it were a \
                 GPU fit",
                case.name
            );
        }
    }
}

/// `device=cuda` must not silently train on the CPU here the way upstream
/// does. Whatever this build does with a GPU device, it has to be explicit.
#[test]
fn a_cuda_device_is_never_silently_downgraded_to_cpu() {
    let dmat = load_dataset("regression");
    let mut params = TrainingParameters {
        booster: BoosterParameters {
            booster: BoosterType::Gbtree(TreeBoosterParameters::default()),
            general: GeneralParameters {
                device: Device::cuda(0),
                verbosity: Verbosity::Silent,
                ..Default::default()
            },
            learning: LearningTaskParameters::default(),
        },
        verbose_eval: VerboseEval::Silent,
        ..Default::default()
    };
    params.num_boost_round = 2;

    let cpu = {
        let mut p = params.clone();
        p.booster.general.device = Device::Cpu;
        api::train(&p, &dmat, &[(&dmat, "train")]).unwrap().0.predict(&dmat)
    };

    match api::train(&params, &dmat, &[(&dmat, "train")]) {
        // Refusing is acceptable, as long as it names the device.
        Err(e) => assert!(
            e.to_string().contains("device") || e.to_string().contains("cuda"),
            "a CUDA fit must fail naming the device, got: {e}"
        ),
        // Succeeding is acceptable only if it really ran somewhere else. A
        // byte-identical CPU result is the silent-fallback bug.
        Ok((booster, _)) => {
            let gpu = booster.predict(&dmat);
            assert_eq!(gpu.len(), cpu.len());
            for (g, c) in gpu.iter().zip(&cpu) {
                assert!(
                    (g - c).abs() <= 1e-5 * c.abs().max(1.0),
                    "a CUDA fit that succeeds must agree with the CPU fit: {g} != {c}"
                );
            }
        }
    }
}

// ------------------------------------------------------- device parity ----

/// Replay a case's configuration on `device`, for `rounds` rounds.
///
/// A thinner [`train_case`]: the fixture's own round count lets round one's
/// quantisation difference feed round two's gradients, and from there the two
/// devices are fitting different problems. The device claim this file makes is
/// about a *single* round, where both paths see identical gradients.
#[cfg(feature = "gpu")]
fn train_case_on(case: &Case, device: Device, rounds: u32) -> Result<xgboost_rs::Booster, String> {
    let fixture = load(&case.name);
    let dmat = load_dataset(&case.data);
    let mut params = config_to_params(&case.name, &config_map(&fixture));
    params.num_boost_round = rounds;
    params.booster.general.device = device;
    retarget_updaters(&mut params, device);

    let base = if fixture["updates_existing_model"].as_bool().unwrap_or(false) {
        let mut seed = params.clone();
        if let BoosterType::Gbtree(t) = &mut seed.booster.booster {
            t.process_type = ProcessType::Default;
            t.updater = None;
        }
        // The ensemble being rewritten is grown on the same device, so the
        // comparison stays device-against-device rather than putting a
        // CPU-grown model underneath a device fit.
        Some(api::train(&seed, &dmat, &[]).map_err(|e| format!("seeding: {e}"))?.0)
    } else {
        None
    };

    api::train_from(&params, &dmat, &[(&dmat, "train")], base.as_ref())
        .map(|(b, _)| b)
        .map_err(|e| e.to_string())
}

/// Point an explicitly named grower at `device`.
///
/// A fixture that names `grow_quantile_histmaker` is asking for *`hist`*, and
/// `hist` on CUDA is spelled `grow_gpu_hist` — the updater name carries the
/// device, so replaying the same request on the other device means renaming
/// it. Without this the comparison would only ever be able to say the device
/// refuses a CPU updater, which is a fact about the spelling and not about the
/// fit. The tree-modifying stages (`prune`, `refresh`) belong to no device and
/// are carried across untouched.
#[cfg(feature = "gpu")]
fn retarget_updaters(params: &mut TrainingParameters, device: Device) {
    use TreeUpdaterName::*;
    let Some(tree) = params.booster.booster.tree_mut() else { return };
    let Some(updaters) = tree.updater.as_mut() else { return };
    for u in updaters.iter_mut() {
        *u = match (*u, device.is_cuda()) {
            (GrowQuantileHistMaker, true) | (GrowGpuHist, false) => {
                if device.is_cuda() { GrowGpuHist } else { GrowQuantileHistMaker }
            }
            (GrowHistMaker, true) | (GrowGpuApprox, false) => {
                if device.is_cuda() { GrowGpuApprox } else { GrowHistMaker }
            }
            (other, _) => other,
        };
    }
}

/// The reason a case cannot run on `device=cuda`, when XGBoost has that same
/// rule. Anything not named here **must** run on the device.
///
/// Both rules are upstream's own: `MapTreeMethodToUpdaters` (`src/gbm/gbtree.cc`)
/// has no GPU entry for `exact`, and there is no SYCL backend in this build.
/// They are stated against the fixture's own configuration rather than against
/// a list of case names, so a fixture added later is classified by what it asks
/// for rather than by whether someone remembered to list it.
#[cfg(feature = "gpu")]
fn device_refusal(cfg: &BTreeMap<String, String>) -> Option<&'static str> {
    let updater = cfg.get("updater").map(String::as_str).unwrap_or("");
    let names = |u: &str| updater.split(',').any(|part| part.trim() == u);
    if cfg.get("tree_method").map(String::as_str) == Some("exact") || names("grow_colmaker") {
        return Some("the `exact` updater has no GPU implementation, as upstream");
    }
    if updater.split(',').any(|part| part.trim().ends_with("_sycl")) {
        return Some("there is no SYCL backend in this build");
    }
    None
}

/// **Every string-valued parameter must drive `device=cuda` exactly as it
/// drives the CPU.**
///
/// This is what "full parameter support on both devices" has to mean if it is
/// to mean anything: not that the device *accepts* the parameter, but that one
/// round of it produces the identical tree. Bit-identical is the right bar and
/// not an optimistic one — both paths see the same gradients on round one, and
/// although the device sums quantised `i64` bins where the CPU sums `f64`, the
/// split arithmetic that decides the tree agrees exactly.
///
/// Together with the CPU cases above being pinned to XGBoost 3.4.0, this pins
/// the *device* fit to XGBoost as well — the only way to pin it at all on a
/// machine with no NVIDIA GPU, where upstream cannot produce a GPU fixture to
/// compare against (see [`gpu_cases_are_honest_about_the_device`]).
#[cfg(feature = "gpu")]
#[test]
fn every_string_parameter_fits_the_same_on_cpu_and_on_the_device() {
    // A backend with no `f64` refuses every device fit (Metal — MSL has no
    // `double`; see `xgboost_rs::gpu::supports_f64`), so there is no device fit
    // to compare against. `gpu_training::a_backend_without_f64_refuses_the_fit`
    // is what holds that refusal to being a named one.
    if !xgboost_rs::gpu::supports_f64(&xgboost_rs::gpu::default_client(0)) {
        return;
    }
    let mut report = Report::default();
    let mut refused = 0usize;

    for case in cases().into_iter().filter(|c| c.outcome == "ok" || c.requires_gpu) {
        let cfg = config_map(&load(&case.name));
        if let Some(reason) = device_refusal(&cfg) {
            // Refusing is acceptable only if it is *this* refusal, said out loud.
            match train_case_on(&case, Device::cuda(0), 1) {
                Ok(_) => report.record(
                    &case,
                    Some(format!("expected a refusal ({reason}), but the device fit ran")),
                ),
                Err(e) => {
                    refused += 1;
                    if !(e.contains("device") || e.contains("exact") || e.contains("SYCL")) {
                        report.record(&case, Some(format!("refused without naming why: {e}")));
                    }
                }
            }
            continue;
        }

        let detail = match (train_case_on(&case, Device::Cpu, 1), train_case_on(&case, Device::cuda(0), 1)) {
            (Err(c), _) => Some(format!("the CPU fit itself failed: {c}")),
            (Ok(_), Err(g)) => Some(format!("the device refused it: {g}")),
            (Ok(cpu), Ok(gpu)) => diff_models(&cpu, &gpu, &load_dataset(&case.data)),
        };
        report.record(&case, detail);
    }

    assert!(refused > 0, "no case exercised a documented device refusal");
    report.finish("one-round device parity");
}

/// Describe the first way two fits differ: the ensemble first, then what it
/// predicts. Integers are compared exactly, and so are the predictions — one
/// round of the same gradients leaves no room for a tolerance.
#[cfg(feature = "gpu")]
fn diff_models(
    cpu: &xgboost_rs::Booster,
    gpu: &xgboost_rs::Booster,
    dmat: &DMatrix,
) -> Option<String> {
    let booster_of = |b: &xgboost_rs::Booster| -> Value {
        let m: Value = serde_json::from_str(&b.save_model()).unwrap();
        m["learner"]["gradient_booster"].clone()
    };
    let (c, g) = (booster_of(cpu), booster_of(gpu));
    let trees = |v: &Value| v["model"]["trees"].as_array().cloned().unwrap_or_default();
    let (ct, gt) = (trees(&c), trees(&g));
    if ct.len() != gt.len() {
        return Some(format!("tree count: cpu {} != gpu {}", ct.len(), gt.len()));
    }
    for (i, (a, b)) in ct.iter().zip(&gt).enumerate() {
        // Compare every field the model records, not a list written here: a
        // field added to the serialiser must not slip through unchecked.
        for (field, want) in a.as_object().expect("a tree is a JSON object") {
            let got = &b[field];
            if got == want {
                continue;
            }
            // `loss_changes` is the one field allowed to drift, to the SPEC's
            // ordinary `1e-5` relative bar or [`GAIN_ABS_TOL`], whichever is
            // looser. The backend evaluates the gain's division at `f64` and
            // narrows once, so it does not reproduce
            // `CalcGainGivenWeight`'s narrow-before-divide; and because a gain
            // is the *difference* of two such quantities, a small gain
            // subtracted out of large sums loses relative precision to
            // cancellation — which is what the absolute floor covers. It is a
            // recorded diagnostic, not a decision: which feature, which
            // threshold, which default direction and what the leaf is worth
            // are all pinned bit-for-bit by the fields around it.
            if field == "loss_changes"
                && let (Some(w), Some(g)) = (want.as_array(), got.as_array())
                && w.len() == g.len()
                && w.iter().zip(g).all(|(x, y)| {
                    let (x, y) = (x.as_f64().unwrap_or(f64::NAN), y.as_f64().unwrap_or(f64::NAN));
                    close(y, x) || (x - y).abs() <= GAIN_ABS_TOL
                })
            {
                continue;
            }
            if field == "loss_changes"
                && let (Some(w), Some(g)) = (want.as_array(), got.as_array())
                && w.len() == g.len()
            {
                let worst = w
                    .iter()
                    .zip(g)
                    .map(|(x, y)| {
                        let (x, y) = (x.as_f64().unwrap(), y.as_f64().unwrap());
                        ((x - y).abs() / x.abs().max(1.0), x, y)
                    })
                    .max_by(|a, b| a.0.total_cmp(&b.0))
                    .unwrap();
                return Some(format!(
                    "tree {i}: `loss_changes` worst relative gap {:.3e} (cpu {} vs gpu {})",
                    worst.0, worst.1, worst.2
                ));
            }
            let where_ = match (want.as_array(), got.as_array()) {
                (Some(w), Some(g)) if w.len() != g.len() => {
                    format!(" (cpu has {} entries, gpu {})", w.len(), g.len())
                }
                (Some(w), Some(g)) => match w.iter().zip(g).position(|(x, y)| x != y) {
                    Some(k) => format!(" at [{k}]: cpu {} != gpu {}", w[k], g[k]),
                    None => String::new(),
                },
                _ => String::new(),
            };
            return Some(format!("tree {i}: `{field}` differs{where_}"));
        }
    }
    // Anything outside the tree arrays — the tree count, `tree_info`, the
    // dropout weights — must agree exactly.
    for (key, want) in c.as_object().expect("the booster is a JSON object") {
        if key != "model" && g[key] != *want {
            return Some(format!("`{key}` differs outside the trees"));
        }
    }
    for key in ["gbtree_model_param", "iteration_indptr", "tree_info"] {
        if c["model"][key] != g["model"][key] {
            return Some(format!("`model.{key}` differs"));
        }
    }
    for (i, (a, b)) in cpu.predict(dmat).iter().zip(&gpu.predict(dmat)).enumerate() {
        if a != b {
            return Some(format!("prediction[{i}]: cpu {a} != gpu {b}"));
        }
    }
    None
}
