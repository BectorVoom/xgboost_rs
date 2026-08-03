//! Training must be reproducible: the same data and configuration produce a
//! bit-identical model regardless of how many threads run the fit.
//!
//! The thread pool is process-global, so each thread count is exercised in its
//! own run of the benchmark binary, which prints a checksum of the serialised
//! model.

use std::process::Command;

fn model_hash(threads: usize, sparsity: &str) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_train_bench"))
        .args([
            "--rows", "20000",
            "--features", "15",
            "--rounds", "4",
            "--sparsity", sparsity,
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
        let baseline = model_hash(1, sparsity);
        for threads in [2, 3, 8] {
            assert_eq!(
                model_hash(threads, sparsity),
                baseline,
                "sparsity {sparsity}: {threads} threads produced a different model than 1 thread"
            );
        }
    }
}
