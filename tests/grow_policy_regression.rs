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

/// Compare a policy's digests, reporting every mismatch at once.
///
/// These pins move whenever the *binning* changes, because different cut values
/// put rows in different bins — and also whenever the model *file* changes
/// shape, since the digest is taken over the saved JSON. Both are legitimate
/// and both move every pin at once, so a wholesale mismatch here wants
/// re-pinning while a single one moving means the scheduling claim above has
/// been broken. Reporting all of them is what tells those two apart.
///
/// Last re-pinned when the vector-leaf model format was aligned with
/// upstream's, which dropped the always-empty leaf array from a scalar tree.
/// The fits themselves did not move: the oracle files are what say so.
fn assert_digests(policy: GrowPolicy, cases: &[(u32, u32, &str)]) {
    let mut wrong = Vec::new();
    for &(max_depth, max_leaves, expected) in cases {
        let got = digest(&model(policy, max_depth, max_leaves));
        if got != expected {
            wrong.push(format!(
                "  depth={max_depth:<3} leaves={max_leaves:<4} got {got}, pinned {expected}"
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "{policy} models changed ({} of {} pins):\n{}",
        wrong.len(),
        cases.len(),
        wrong.join("\n")
    );
}

#[test]
fn depthwise_models_are_unchanged() {
    assert_digests(
        GrowPolicy::DepthWise,
        &[
            (6, 0, "3238676d5c7b571f"),
            (0, 32, "58ff1165bcd6b610"),
            (8, 64, "d5b5ba2ace6476d1"),
        ],
    );
}

#[test]
fn lossguide_models_are_unchanged() {
    assert_digests(
        GrowPolicy::LossGuide,
        &[
            (0, 16, "951d30ca423d879d"),
            (0, 64, "ab47ac96ae4a0063"),
            (0, 256, "f15fab677e57bf06"),
            (6, 64, "629857552727447d"),
        ],
    );
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
