//! Shared fixture loading for the oracle tests.
//!
//! Fixtures under `tests/fixtures/` are produced by `tools/gen_fixtures.py`
//! against a pinned XGBoost (see `xgboost_version` in each case file). They are
//! committed, so the tests never need Python at run time.

#![allow(dead_code)] // each test binary compiles this module and uses a subset

use serde_json::Value;
use std::path::PathBuf;
use xgboost_rs::data::DMatrix;

pub fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

pub fn load_json(name: &str) -> Value {
    let path = fixture_dir().join(format!("{name}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read fixture {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("bad JSON in {}: {e}", path.display()))
}

/// Load the dataset a case refers to, as a `DMatrix` with labels and weights.
pub fn load_data(name: &str) -> DMatrix {
    let v = load_json(&format!("data_{name}"));
    let n_row = v["n_row"].as_u64().unwrap() as usize;
    let n_col = v["n_col"].as_u64().unwrap() as usize;

    let mut d = match v["layout"].as_str().unwrap() {
        "dense" => {
            // `null` encodes NaN, the missing sentinel.
            let values: Vec<f32> = v["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().map(|f| f as f32).unwrap_or(f32::NAN))
                .collect();
            DMatrix::from_dense(&values, n_row, n_col, f32::NAN).unwrap()
        }
        "csr" => {
            let indptr: Vec<usize> =
                v["indptr"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
            let indices: Vec<u32> =
                v["indices"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
            let values: Vec<f32> =
                v["values"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
            DMatrix::from_csr(&indptr, &indices, &values, n_col, f32::NAN).unwrap()
        }
        other => panic!("unknown layout {other}"),
    };

    let labels: Vec<f32> =
        v["labels"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
    d.set_labels(&labels).unwrap();
    if let Some(w) = v["weights"].as_array() {
        let weights: Vec<f32> = w.iter().map(|x| x.as_f64().unwrap() as f32).collect();
        d.set_weights(&weights).unwrap();
    }
    d
}

pub fn f32_array(v: &Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}

pub fn u32_array(v: &Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}

/// Every committed case, as `(case name, dataset name)`.
pub const CASES: &[(&str, &str)] = &[
    ("dense_small_b256_d6", "dense_small"),
    ("dense_small_b16_d3", "dense_small"),
    ("dense_dup_b8_d4", "dense_dup"),
    ("dense_missing_b64_d5", "dense_missing"),
    ("weighted_b32_d4", "weighted"),
    ("agaricus_b256_d6", "agaricus"),
];
