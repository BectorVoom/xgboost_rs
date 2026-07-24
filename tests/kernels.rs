//! Kernel tests against the CPU oracle in `xgboost_rs::reference`, mirroring
//! the checks in `xgboost/tests/cpp/tree/gpu_hist/test_histogram.cu`.
//!
//! GPU histograms must match the sequential CPU sums *exactly* (i64 equality).

use cubecl::Runtime;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

use xgboost_rs::gpu::ellpack::EllpackLayout;
use xgboost_rs::gpu::histogram::{HistogramBuilder, supports_native_i64_atomics};
use xgboost_rs::gpu::quantiser::{GradientQuantiser, quantise};
use xgboost_rs::gpu::{GradientPair, GradientPairInt64};
use xgboost_rs::reference::{cpu_histogram, random_gpairs, random_matrix};

type R = WgpuRuntime;

fn client() -> cubecl::prelude::ComputeClient<R> {
    R::client(&WgpuDevice::default())
}

#[test]
fn quantise_matches_host_reference() {
    let client = client();
    let gpairs = random_gpairs(4096, 7);
    let quantiser = GradientQuantiser::new(&gpairs, gpairs.len() as u64);

    let device = quantise::<R>(&client, &gpairs, &quantiser);
    let host: Vec<GradientPairInt64> =
        gpairs.iter().map(|g| quantiser.to_fixed_point(*g)).collect();

    assert_eq!(device, host);
}

#[test]
fn quantise_preserves_positive_curvature() {
    let client = client();
    // A hessian small enough to truncate to 0 must be bumped to 1.
    let mut gpairs = random_gpairs(64, 11);
    gpairs[3].hess = 1e-30;
    let quantiser = GradientQuantiser::new(&gpairs, gpairs.len() as u64);

    let device = quantise::<R>(&client, &gpairs, &quantiser);
    assert!(device[3].hess >= 1);
}

fn run_histogram_case(layout: EllpackLayout, sparsity: f32, force_global: bool) {
    let client = client();
    let (n_rows, n_features, bins) = (2048, 8, 24);
    let matrix = random_matrix(n_rows, n_features, bins, sparsity, layout, 42);
    let gpairs = random_gpairs(n_rows, 13);
    let quantiser = GradientQuantiser::new(&gpairs, n_rows as u64);
    let quantised = quantise::<R>(&client, &gpairs, &quantiser);

    // A node containing every third row, as row partitioning would produce.
    let ridx: Vec<u32> = (0..n_rows as u32).step_by(3).collect();

    let engine = HistogramBuilder::new(&client)
        .force_global(force_global)
        .build(&matrix)
        .unwrap();
    assert_eq!(engine.uses_shared_memory(), !force_global);

    let gpu = engine.build(&quantised, &ridx).unwrap();
    let cpu = cpu_histogram(&matrix, &quantised, &ridx);

    assert_eq!(gpu, cpu);
}

#[test]
fn histogram_dense_shared() {
    run_histogram_case(EllpackLayout::Dense, 0.0, false);
}

#[test]
fn histogram_dense_global() {
    run_histogram_case(EllpackLayout::Dense, 0.0, true);
}

#[test]
fn histogram_dense_compressed_shared() {
    run_histogram_case(EllpackLayout::DenseCompressed, 0.3, false);
}

#[test]
fn histogram_sparse_shared() {
    run_histogram_case(EllpackLayout::Sparse, 0.5, false);
}

#[test]
fn histogram_sparse_global() {
    run_histogram_case(EllpackLayout::Sparse, 0.5, true);
}

/// Forces multiple feature groups: a tiny shared-memory budget still fits one
/// feature (24 bins * 16 B = 384 B) but not all of them.
#[test]
fn histogram_many_feature_groups() {
    let client = client();
    let (n_rows, n_features, bins) = (512, 8, 24);
    let matrix = random_matrix(n_rows, n_features, bins, 0.0, EllpackLayout::Dense, 3);
    let gpairs = random_gpairs(n_rows, 17);
    let quantiser = GradientQuantiser::new(&gpairs, n_rows as u64);
    let quantised = quantise::<R>(&client, &gpairs, &quantiser);
    let ridx: Vec<u32> = (0..n_rows as u32).collect();

    let engine = HistogramBuilder::new(&client).shmem_bytes(1024).build(&matrix).unwrap();
    assert!(engine.uses_shared_memory());

    let gpu = engine.build(&quantised, &ridx).unwrap();
    let cpu = cpu_histogram(&matrix, &quantised, &ridx);
    assert_eq!(gpu, cpu);
}

/// Negative gradient sums exercise the carry propagation in AtomicAdd64As32:
/// adding a negative i64 to a positive accumulator wraps the low word.
#[test]
fn histogram_carry_propagation() {
    let client = client();
    let n_rows = 1024usize;
    let matrix = random_matrix(n_rows, 2, 4, 0.0, EllpackLayout::Dense, 5);
    // All-negative gradients concentrate sign flips in every bin.
    let gpairs: Vec<GradientPair> = random_gpairs(n_rows, 23)
        .into_iter()
        .map(|g| GradientPair { grad: -g.grad.abs(), hess: g.hess })
        .collect();
    let quantiser = GradientQuantiser::new(&gpairs, n_rows as u64);
    let quantised = quantise::<R>(&client, &gpairs, &quantiser);
    let ridx: Vec<u32> = (0..n_rows as u32).collect();

    let engine = HistogramBuilder::new(&client).build(&matrix).unwrap();
    let gpu = engine.build(&quantised, &ridx).unwrap();
    let cpu = cpu_histogram(&matrix, &quantised, &ridx);
    assert_eq!(gpu, cpu);
}

#[test]
fn subtraction_trick() {
    let client = client();
    let n_rows = 1024usize;
    let matrix = random_matrix(n_rows, 4, 16, 0.0, EllpackLayout::Dense, 9);
    let gpairs = random_gpairs(n_rows, 29);
    let quantiser = GradientQuantiser::new(&gpairs, n_rows as u64);
    let quantised = quantise::<R>(&client, &gpairs, &quantiser);

    let all: Vec<u32> = (0..n_rows as u32).collect();
    let (left, right): (Vec<u32>, Vec<u32>) = all.iter().partition(|r| *r % 3 == 0);

    let engine = HistogramBuilder::new(&client).build(&matrix).unwrap();
    let parent = engine.build(&quantised, &all).unwrap();
    let built = engine.build(&quantised, &left).unwrap();
    let sibling = engine.subtract(&parent, &built).unwrap();

    let expected = cpu_histogram(&matrix, &quantised, &right);
    assert_eq!(sibling, expected);
}

/// Regression for the CUDA `illegal address` crash: a sparse matrix whose
/// total bins exceed the shared-memory budget must NOT be split into multiple
/// shared groups (global bins make per-group shared histograms invalid). The
/// builder must fall back to the global path and still match the oracle.
#[test]
fn histogram_sparse_forces_global_when_bins_exceed_shmem() {
    let client = client();
    // 32 features x 256 bins = 8192 total bins; 4 KiB shmem fits only 256
    // bins, far fewer than the whole row -> must fall back to global.
    let (n_rows, n_features, bins) = (4096, 32, 256);
    let matrix = random_matrix(n_rows, n_features, bins, 0.5, EllpackLayout::Sparse, 42);
    let gpairs = random_gpairs(n_rows, 13);
    let quantiser = GradientQuantiser::new(&gpairs, n_rows as u64);
    let quantised = quantise::<R>(&client, &gpairs, &quantiser);
    let ridx: Vec<u32> = (0..n_rows as u32).step_by(3).collect();

    let engine = HistogramBuilder::new(&client).shmem_bytes(4096).build(&matrix).unwrap();
    assert!(!engine.uses_shared_memory(), "sparse+overflow must use global path");

    let gpu = engine.build(&quantised, &ridx).unwrap();
    let cpu = cpu_histogram(&matrix, &quantised, &ridx);
    assert_eq!(gpu, cpu);
}

/// A sparse matrix whose bins DO fit uses a single shared group (never split).
#[test]
fn histogram_sparse_single_shared_group() {
    let client = client();
    let (n_rows, n_features, bins) = (2048, 8, 24); // 192 bins, fits 48 KiB
    let matrix = random_matrix(n_rows, n_features, bins, 0.5, EllpackLayout::Sparse, 7);
    let gpairs = random_gpairs(n_rows, 13);
    let quantiser = GradientQuantiser::new(&gpairs, n_rows as u64);
    let quantised = quantise::<R>(&client, &gpairs, &quantiser);
    let ridx: Vec<u32> = (0..n_rows as u32).collect();

    let engine = HistogramBuilder::new(&client).build(&matrix).unwrap();
    assert!(engine.uses_shared_memory());
    assert_eq!(
        engine.build(&quantised, &ridx).unwrap(),
        cpu_histogram(&matrix, &quantised, &ridx)
    );
}

/// Native i64-atomic global accumulation must match the u32-carry scheme and
/// the CPU oracle exactly. Skipped when the device lacks 64-bit atomics.
#[test]
fn histogram_native_i64_atomics() {
    let client = client();
    if !supports_native_i64_atomics(&client) {
        eprintln!("skipping: runtime has no native i64 atomics");
        return;
    }

    let (n_rows, n_features, bins) = (2048, 8, 24);
    let matrix = random_matrix(n_rows, n_features, bins, 0.5, EllpackLayout::Sparse, 42);
    let gpairs = random_gpairs(n_rows, 13);
    let quantiser = GradientQuantiser::new(&gpairs, n_rows as u64);
    let quantised = quantise::<R>(&client, &gpairs, &quantiser);
    let ridx: Vec<u32> = (0..n_rows as u32).step_by(3).collect();
    let cpu = cpu_histogram(&matrix, &quantised, &ridx);

    // Global path (pure native atomics) and shared path (native flush).
    for force_global in [true, false] {
        let engine = HistogramBuilder::new(&client)
            .force_global(force_global)
            .native_i64_atomics(true)
            .build(&matrix)
            .unwrap();
        assert!(engine.uses_native_i64_atomics());
        assert_eq!(engine.build(&quantised, &ridx).unwrap(), cpu);
    }
}

#[test]
fn builder_rejects_bad_shapes() {
    let client = client();
    let mut matrix = random_matrix(16, 2, 4, 0.0, EllpackLayout::Dense, 1);
    matrix.gidx.pop();
    assert!(matches!(
        HistogramBuilder::new(&client).build(&matrix),
        Err(xgboost_rs::Error::MatrixShape { .. })
    ));

    let matrix = random_matrix(16, 2, 4, 0.0, EllpackLayout::Dense, 1);
    let engine = HistogramBuilder::new(&client).build(&matrix).unwrap();
    let too_few = vec![GradientPairInt64::default(); 3];
    assert!(matches!(
        engine.upload_gpairs(&too_few),
        Err(xgboost_rs::Error::GpairCount { expected: 16, got: 3 })
    ));
}
