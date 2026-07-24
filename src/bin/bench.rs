//! Oracle (correctness) and speed benchmark for the CubeCL histogram kernels.
//!
//! Runs on CUDA when built with `--features cuda` (e.g. a Kaggle GPU
//! notebook), otherwise on the wgpu/Vulkan runtime.
//!
//! Environment overrides:
//!   BENCH_ROWS      rows in the benchmark matrix   (default 1048576)
//!   BENCH_FEATURES  features                       (default 32)
//!   BENCH_BINS      bins per feature               (default 256)
//!   BENCH_ITERS     timed iterations per case      (default 20)

use std::time::Instant;

use anyhow::{Context, bail};
use cubecl::Runtime;
use cubecl::prelude::ComputeClient;

use xgboost_rs::gpu::ellpack::EllpackLayout;
use xgboost_rs::gpu::histogram::{HistogramBuilder, supports_native_i64_atomics};
use xgboost_rs::gpu::quantiser::{GradientQuantiser, quantise, quantise_to_device};
use xgboost_rs::gpu::{GradientPair, GradientPairInt64};
use xgboost_rs::reference::{cpu_histogram, random_gpairs, random_matrix};

#[cfg(feature = "cuda")]
mod runtime {
    pub type R = cubecl::cuda::CudaRuntime;
    pub type Device = cubecl::cuda::CudaDevice;
    pub const NAME: &str = "cuda";
}

#[cfg(not(feature = "cuda"))]
mod runtime {
    pub type R = cubecl::wgpu::WgpuRuntime;
    pub type Device = cubecl::wgpu::WgpuDevice;
    pub const NAME: &str = "wgpu";
}

use runtime::R;

fn sync(client: &ComputeClient<R>) -> anyhow::Result<()> {
    cubecl::future::block_on(client.sync())
        .map_err(|e| anyhow::anyhow!("device sync failed: {e:?}"))
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// GPU vs CPU oracle over every kernel path; exact i64 equality required.
fn run_oracle(client: &ComputeClient<R>) -> anyhow::Result<()> {
    println!("== oracle ({} runtime) ==", runtime::NAME);

    // Quantiser.
    let gpairs = random_gpairs(4096, 7);
    let quantiser = GradientQuantiser::new(&gpairs, gpairs.len() as u64);
    let device = quantise::<R>(client, &gpairs, &quantiser);
    let host: Vec<GradientPairInt64> =
        gpairs.iter().map(|g| quantiser.to_fixed_point(*g)).collect();
    if device != host {
        bail!("quantiser mismatch vs host oracle");
    }
    println!("quantise                          .. ok");

    // Histogram paths. `atomics`: None = auto-detect, Some(x) = forced.
    let native = supports_native_i64_atomics(client);
    println!("native i64 atomics: {}", if native { "supported" } else { "not supported" });

    let mut cases: Vec<(&str, EllpackLayout, f32, bool, Option<bool>)> = vec![
        ("hist dense/shared", EllpackLayout::Dense, 0.0, false, None),
        ("hist dense/global/u32-split", EllpackLayout::Dense, 0.0, true, Some(false)),
        ("hist dense-compressed/shared", EllpackLayout::DenseCompressed, 0.3, false, None),
        ("hist sparse/shared", EllpackLayout::Sparse, 0.5, false, None),
        ("hist sparse/global/u32-split", EllpackLayout::Sparse, 0.5, true, Some(false)),
    ];
    if native {
        cases.push(("hist dense/global/i64-native", EllpackLayout::Dense, 0.0, true, Some(true)));
        cases.push(("hist sparse/global/i64-native", EllpackLayout::Sparse, 0.5, true, Some(true)));
        cases.push(("hist sparse/shared/i64-flush", EllpackLayout::Sparse, 0.5, false, Some(true)));
    }
    for &(name, layout, sparsity, force_global, atomics) in &cases {
        let (n_rows, n_features, bins) = (4096, 8, 24);
        let matrix = random_matrix(n_rows, n_features, bins, sparsity, layout, 42);
        let gpairs = random_gpairs(n_rows, 13);
        let quantiser = GradientQuantiser::new(&gpairs, n_rows as u64);
        let quantised = quantise::<R>(client, &gpairs, &quantiser);
        let ridx: Vec<u32> = (0..n_rows as u32).step_by(3).collect();

        let mut builder = HistogramBuilder::new(client).force_global(force_global);
        if let Some(native) = atomics {
            builder = builder.native_i64_atomics(native);
        }
        let engine = builder.build(&matrix)?;
        let gpu = engine.build(&quantised, &ridx)?;
        let cpu = cpu_histogram(&matrix, &quantised, &ridx);
        if gpu != cpu {
            bail!("{name}: GPU histogram differs from CPU oracle");
        }
        println!("{name:33} .. ok");
    }

    // Subtraction trick.
    let n_rows = 4096usize;
    let matrix = random_matrix(n_rows, 4, 16, 0.0, EllpackLayout::Dense, 9);
    let gpairs = random_gpairs(n_rows, 29);
    let quantiser = GradientQuantiser::new(&gpairs, n_rows as u64);
    let quantised = quantise::<R>(client, &gpairs, &quantiser);
    let all: Vec<u32> = (0..n_rows as u32).collect();
    let (left, right): (Vec<u32>, Vec<u32>) = all.iter().partition(|r| *r % 3 == 0);
    let engine = HistogramBuilder::new(client).build(&matrix)?;
    let parent = engine.build(&quantised, &all)?;
    let built = engine.build(&quantised, &left)?;
    let sibling = engine.subtract(&parent, &built)?;
    if sibling != cpu_histogram(&matrix, &quantised, &right) {
        bail!("subtraction trick differs from CPU oracle");
    }
    println!("subtraction trick                 .. ok");

    println!("oracle: ALL PASS\n");
    Ok(())
}

fn run_bench(client: &ComputeClient<R>) -> anyhow::Result<()> {
    let n_rows = env_usize("BENCH_ROWS", 1 << 20);
    let n_features = env_usize("BENCH_FEATURES", 32);
    let bins = env_usize("BENCH_BINS", 256) as u32;
    let iters = env_usize("BENCH_ITERS", 20);

    println!("== speed ({} runtime) ==", runtime::NAME);
    println!(
        "rows={n_rows} features={n_features} bins/feature={bins} total-bins={} iters={iters}",
        n_features as u32 * bins
    );

    let native = supports_native_i64_atomics(client);
    println!("native i64 atomics: {}", if native { "supported" } else { "not supported" });

    // `atomics`: Some(x) forces the global-accumulation mode; the shared-path
    // cases use auto-detection (mode only affects the flush there).
    let mut cases: Vec<(&str, EllpackLayout, f32, bool, Option<bool>)> = vec![
        ("dense/shared", EllpackLayout::Dense, 0.0, false, None),
        ("dense/global/u32", EllpackLayout::Dense, 0.0, true, Some(false)),
        ("sparse0.5/shared", EllpackLayout::Sparse, 0.5, false, None),
        ("sparse0.5/global/u32", EllpackLayout::Sparse, 0.5, true, Some(false)),
    ];
    if native {
        cases.insert(2, ("dense/global/i64", EllpackLayout::Dense, 0.0, true, Some(true)));
        cases.push(("sparse0.5/global/i64", EllpackLayout::Sparse, 0.5, true, Some(true)));
    }

    for &(name, layout, sparsity, force_global, atomics) in &cases {
        let matrix = random_matrix(n_rows, n_features, bins, sparsity, layout, 42);
        let gpairs: Vec<GradientPair> = random_gpairs(n_rows, 13);
        let quantiser = GradientQuantiser::new(&gpairs, n_rows as u64);

        let mut builder = HistogramBuilder::new(client).force_global(force_global);
        if let Some(native) = atomics {
            builder = builder.native_i64_atomics(native);
        }
        let engine = builder.build(&matrix)?;
        let gpairs_dev = quantise_to_device::<R>(client, &gpairs, &quantiser);
        let all_rows: Vec<u32> = (0..n_rows as u32).collect();
        let rows_dev = engine.upload_rows(&all_rows);

        // Warmup (includes kernel compilation).
        for _ in 0..3 {
            engine.build_to_device(&gpairs_dev, &rows_dev);
        }
        sync(client)?;

        let start = Instant::now();
        for _ in 0..iters {
            engine.build_to_device(&gpairs_dev, &rows_dev);
        }
        sync(client)?;
        let elapsed = start.elapsed();

        let ms = elapsed.as_secs_f64() * 1e3 / iters as f64;
        // Matrix entries visited per build.
        let entries = (n_rows * n_features) as f64;
        let geps = entries / (elapsed.as_secs_f64() / iters as f64) / 1e9;
        println!("{name:22} {ms:9.3} ms/build   {geps:7.2} Gentry/s");
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let device = runtime::Device::default();
    let client = R::client(&device);
    println!("runtime: {} / device: {:?}\n", runtime::NAME, device);

    run_oracle(&client).context("oracle failed")?;
    run_bench(&client).context("benchmark failed")?;
    Ok(())
}
