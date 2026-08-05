//! Training must be reproducible: the same data and configuration produce a
//! bit-identical model regardless of how many threads run the fit.
//!
//! The thread pool is process-global, so each thread count is exercised in its
//! own run of the benchmark binary, which prints a checksum of the serialised
//! model.

use std::process::Command;

fn model_hash(threads: usize, sparsity: &str, subsample: &str, colsample: &str) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_train_bench"))
        .args([
            "--rows", "20000",
            "--features", "15",
            "--rounds", "4",
            "--sparsity", sparsity,
            "--subsample", subsample,
            "--colsample", colsample,
            "--threads", &threads.to_string(),
            "--hash",
        ])
        .output()
        .expect("failed to run train_bench");
    assert!(
        out.status.success(),
        "train_bench failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .find_map(|l| l.strip_prefix("model-hash: ").map(str::to_owned))
        .expect("benchmark did not print a model hash")
}

#[test]
fn model_is_identical_across_thread_counts() {
    for sparsity in ["0.0", "0.35"] {
        let baseline = model_hash(1, sparsity, "1.0", "1.0");
        for threads in [2, 3, 8] {
            assert_eq!(
                model_hash(threads, sparsity, "1.0", "1.0"),
                baseline,
                "sparsity {sparsity}: {threads} threads produced a different model than 1 thread"
            );
        }
    }
}

/// Sampling is the one part of a fit that is not a pure function of the data,
/// so it is the one most easily made thread-dependent. Row draws come from a
/// closed form in the row index and column draws happen before any parallel
/// work, so neither may move with the thread count.
#[test]
fn a_sampled_model_is_identical_across_thread_counts() {
    let baseline = model_hash(1, "0.0", "0.5", "0.6");
    for threads in [2, 3, 8] {
        assert_eq!(
            model_hash(threads, "0.0", "0.5", "0.6"),
            baseline,
            "{threads} threads produced a different sampled model than 1 thread"
        );
    }
}
