use anyhow::Context;

use xgboost_rs::gpu::GradientPair;
use xgboost_rs::gpu::ellpack::{EllpackLayout, EllpackMatrix};
use xgboost_rs::gpu::histogram::HistogramBuilder;
use xgboost_rs::gpu::quantiser::{GradientQuantiser, quantise};
use xgboost_rs::gpu::{BACKEND, DefaultRuntime, default_client};

fn main() -> anyhow::Result<()> {
    let client = default_client(0);
    println!("backend: {BACKEND}");

    // Tiny dense demo: 6 rows x 2 features, 3 bins per feature.
    let matrix = EllpackMatrix {
        gidx: vec![0, 1, 1, 2, 2, 0, 0, 0, 1, 1, 2, 2],
        row_stride: 2,
        base_rowid: 0,
        n_rows: 6,
        cut_ptrs: vec![0, 3, 6],
        null_value: u32::MAX,
        layout: EllpackLayout::Dense,
    };
    let gpairs: Vec<GradientPair> = (0..6)
        .map(|i| GradientPair { grad: i as f32 - 2.5, hess: 1.0 })
        .collect();

    let quantiser = GradientQuantiser::new(&gpairs, gpairs.len() as u64);
    let quantised = quantise::<DefaultRuntime>(&client, &gpairs, &quantiser);
    let ridx: Vec<u32> = (0..6).collect();

    let engine = HistogramBuilder::new(&client)
        .build(&matrix)
        .context("failed to set up histogram engine")?;
    let hist = engine.build(&quantised, &ridx).context("histogram build failed")?;

    println!("bin |        grad |  hess");
    for (bin, pair) in hist.iter().enumerate() {
        let fp = quantiser.to_floating_point(*pair);
        println!("{bin:3} | {:11.4} | {:5.2}", fp.grad, fp.hess);
    }
    Ok(())
}
