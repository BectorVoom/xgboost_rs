//! Oracle tests: the Rust implementation against golden fixtures from a pinned
//! real XGBoost (see `tools/gen_fixtures.py`).

mod common;

use common::{CASES, f32_array, load_data, load_json, u32_array};
use xgboost_rs::data::cuts::build_cuts;

/// Quantile cuts must match upstream exactly — every later stage is defined in
/// terms of these bin boundaries, so any drift here compounds.
#[test]
fn quantile_cuts_match_xgboost_exactly() {
    for (case, data) in CASES {
        let fixture = load_json(case);
        let dmat = load_data(data);
        let max_bin = fixture["params"]["max_bin"].as_u64().unwrap() as u32;

        let cuts = build_cuts(&dmat, max_bin).unwrap();
        // `get_quantile_cut()` reports each feature as `[min_value, ...cuts]`,
        // whereas `HistogramCuts` keeps the min values in their own array.
        //
        // From 3.4.0 that leading slot is `-inf`: upstream dropped
        // `HistogramCuts::min_vals_` and a first bin's lower bound is now
        // unbounded. JSON has no infinity literal, so the generator writes
        // `null` for it.
        let want_ptrs = u32_array(&fixture["cut_ptrs"]);
        let want_values: Vec<f32> = fixture["cut_values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().map(|f| f as f32).unwrap_or(f32::NEG_INFINITY))
            .collect();

        assert_eq!(
            cuts.num_features(),
            want_ptrs.len() - 1,
            "{case}: feature count differs"
        );
        for f in 0..cuts.num_features() {
            let (b, e) = (want_ptrs[f] as usize, want_ptrs[f + 1] as usize);
            let want_min = want_values[b];
            let want_cuts = &want_values[b + 1..e];
            let got_cuts = &cuts.cut_values
                [cuts.cut_ptrs[f] as usize..cuts.cut_ptrs[f + 1] as usize];

            assert_eq!(
                cuts.min_values[f].to_bits(),
                want_min.to_bits(),
                "{case}: feature {f} min value differs: got {}, want {want_min}",
                cuts.min_values[f]
            );
            assert_eq!(
                got_cuts.len(),
                want_cuts.len(),
                "{case}: feature {f} has {} bins, want {}",
                got_cuts.len(),
                want_cuts.len()
            );
            for (i, (got, want)) in got_cuts.iter().zip(want_cuts).enumerate() {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{case}: feature {f} cut {i} differs: got {got}, want {want}"
                );
            }
        }
    }
}
