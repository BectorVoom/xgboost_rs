//! `grow_policy` model pins.
//!
//! The split search is free to group its work however it likes, because the
//! per-node merge is a total order on `(loss change, smallest feature index)`
//! and so cannot see the grouping. "Free to" is a claim about the code, though,
//! and the way to keep it true is to pin the models: these digests were taken
//! before the split search was chunked for speed, and any change to how the
//! work is scheduled must leave them alone.
//!
//! `lossguide` is pinned as carefully as `depthwise` because it is the policy
//! the chunking was written for — it expands one node at a time, so it has the
//! fewest tasks to spread and the most to gain.

use xgboost_rs::parameters::{
    BoosterParameters, BoosterType, GeneralParameters, GrowPolicy, LearningTaskParameters,
    TrainingParameters, TreeBoosterParameters, VerboseEval, Verbosity,
};
use xgboost_rs::{DMatrix, api};

/// FNV-1a over the serialised model, so a digest is stable across machines and
/// a mismatch names a single number rather than a page of JSON.
fn digest(text: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

/// Deterministic data with a signal in every column and some missing values,
/// so both the forward and backward split enumerations are exercised.
fn data(rows: usize, cols: usize) -> DMatrix {
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut next = || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u32 << 24) as f32
    };
    let mut x: Vec<f32> = (0..rows * cols).map(|_| next()).collect();
    let y: Vec<f32> = (0..rows)
        .map(|r| {
            let row = &x[r * cols..(r + 1) * cols];
            row.iter().enumerate().map(|(c, v)| v / (c + 1) as f32).sum::<f32>()
                + row[0] * row[cols - 1]
        })
        .collect();
    // Punch holes after the labels are formed, so the missing values carry
    // signal the tree has to route.
    for (i, v) in x.iter_mut().enumerate() {
        if i % 11 == 0 {
            *v = f32::NAN;
        }
    }

    let mut d = DMatrix::from_dense(&x, rows, cols, f32::NAN).unwrap();
    d.set_labels(&y).unwrap();
    d
}

fn model(grow_policy: GrowPolicy, max_depth: u32, max_leaves: u32) -> String {
    let d = data(4000, 12);
    let params = TrainingParameters {
        booster: BoosterParameters {
            booster: BoosterType::Gbtree(TreeBoosterParameters {
                grow_policy,
                max_depth,
                max_leaves,
                ..Default::default()
            }),
            general: GeneralParameters { verbosity: Verbosity::Silent, ..Default::default() },
            learning: LearningTaskParameters::default(),
        },
        num_boost_round: 6,
        verbose_eval: VerboseEval::Silent,
        ..Default::default()
    };
    api::train(&params, &d, &[]).expect("training failed").0.save_model()
}

#[test]
fn depthwise_models_are_unchanged() {
    for (max_depth, max_leaves, expected) in [
        (6u32, 0u32, "f3dd2c94663b81b8"),
        (0, 32, "20c52842a9329002"),
        (8, 64, "7d88de5acacc1246"),
    ] {
        let got = digest(&model(GrowPolicy::DepthWise, max_depth, max_leaves));
        assert_eq!(got, expected, "depthwise depth={max_depth} leaves={max_leaves}");
    }
}

#[test]
fn lossguide_models_are_unchanged() {
    for (max_depth, max_leaves, expected) in [
        (0u32, 16u32, "68aaf5131424dc52"),
        (0, 64, "211d6c73e74dc0d4"),
        (0, 256, "66c9527d1b53c2e4"),
        (6, 64, "e3dd3764cba26a52"),
    ] {
        let got = digest(&model(GrowPolicy::LossGuide, max_depth, max_leaves));
        assert_eq!(got, expected, "lossguide depth={max_depth} leaves={max_leaves}");
    }
}

/// The chunking reads the thread count to size its jobs, so the thread count
/// must still be invisible to the model — for `lossguide` most of all, which
/// has the fewest tasks and so the most chunk-size variation.
#[test]
fn both_policies_are_thread_independent() {
    for policy in [GrowPolicy::DepthWise, GrowPolicy::LossGuide] {
        // The pool is process-global and set once, so this compares the two
        // policies under whatever pool the harness already installed rather
        // than re-installing one. The cross-thread-count comparison proper is
        // `tests/determinism.rs`, which runs a fresh process per thread count.
        let a = model(policy, 0, 64);
        let b = model(policy, 0, 64);
        assert_eq!(digest(&a), digest(&b), "{policy} is not reproducible in one process");
    }
}
