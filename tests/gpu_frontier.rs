//! The two pieces the tree driver stands on: binning a `DMatrix` into an
//! ELLPACK, and accumulating a whole frontier of node histograms into one
//! allocation.
//!
//! A frontier needs to share one buffer because the split evaluator reads a
//! level in a single launch, indexing each node by a bin offset.

use cubecl::Runtime;
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

use xgboost_rs::DMatrix;
use xgboost_rs::data::cuts::build_cuts;
use xgboost_rs::data::gradient_index::build_gradient_index;
use xgboost_rs::gpu::{DeviceRows, GradientPairInt64};
use xgboost_rs::gpu::ellpack::{EllpackLayout, build_ellpack};
use xgboost_rs::gpu::histogram::{HistogramBuilder, NodeHistJob};
use xgboost_rs::reference::{Rng, cpu_histogram, random_matrix};

type R = WgpuRuntime;

fn client() -> ComputeClient<R> {
    R::client(&WgpuDevice::default())
}

/// A matrix with `missing_rate` of its entries absent.
fn matrix(rows: usize, cols: usize, missing_rate: f32, seed: u64) -> DMatrix {
    let mut rng = Rng(seed);
    let values: Vec<f32> = (0..rows * cols)
        .map(|_| {
            if rng.next_f32() < missing_rate {
                f32::NAN
            } else {
                rng.next_f32() * 10.0 - 5.0
            }
        })
        .collect();
    let mut d = DMatrix::from_dense(&values, rows, cols, f32::NAN).unwrap();
    let labels: Vec<f32> = (0..rows).map(|_| rng.next_f32()).collect();
    d.set_labels(&labels).unwrap();
    d
}

/// The ELLPACK must bin exactly as the CPU gradient index does — same cuts,
/// same bin per stored value, same notion of "missing".
#[test]
fn ellpack_bins_match_the_cpu_gradient_index() {
    for (rows, cols, missing, max_bin) in
        [(500usize, 6usize, 0.0f32, 32u32), (700, 5, 0.3, 64), (300, 9, 0.6, 16)]
    {
        let d = matrix(rows, cols, missing, 41 + rows as u64);
        let cuts = build_cuts(&d, max_bin).unwrap();
        let gi = build_gradient_index(&d, &cuts).unwrap();
        let ell = build_ellpack(&d, &cuts);

        assert_eq!(ell.n_rows, rows);
        assert_eq!(ell.row_stride, cols);
        // A `DMatrix` elides zeros as well as NaNs, so only a matrix that
        // ended up with every entry binned is `Dense`.
        let binned: usize = (0..rows).map(|r| gi.row_global_bins(r).len()).sum();
        let holes = binned < rows * cols;
        assert_eq!(
            ell.layout,
            if holes { EllpackLayout::DenseCompressed } else { EllpackLayout::Dense },
            "layout for missing rate {missing}"
        );

        for r in 0..rows {
            // The CPU index reports a row's *global* bins, present values only.
            let mut want = gi.row_global_bins(r);
            want.sort_unstable();

            let mut got: Vec<u32> = (0..cols)
                .filter_map(|f| {
                    let v = ell.gidx[r * cols + f];
                    (ell.is_dense() || v != ell.null_value).then(|| v + cuts.cut_ptrs[f])
                })
                .collect();
            got.sort_unstable();

            assert_eq!(got, want, "row {r} of {rows}x{cols} missing={missing}");
        }
    }
}

/// Every node of a frontier writes into its own slot of one buffer, and the
/// slots do not touch each other.
#[test]
fn frontier_slots_accumulate_independently() {
    let client = client();
    let m = random_matrix(2000, 6, 24, 0.2, EllpackLayout::DenseCompressed, 43);
    let engine = HistogramBuilder::new(&client).build(&m).unwrap();

    let mut rng = Rng(47);
    let gpairs: Vec<GradientPairInt64> = (0..m.n_rows)
        .map(|_| GradientPairInt64 {
            grad: rng.next_u32() as i64 - i64::from(u32::MAX / 2),
            hess: i64::from(rng.next_u32() % 1000) + 1,
        })
        .collect();
    let dev_gpairs = engine.upload_gpairs(&gpairs).unwrap();

    // Three disjoint row sets, one per frontier node.
    let sets: Vec<Vec<u32>> = vec![
        (0..600u32).collect(),
        (600..1500u32).collect(),
        (1500..2000u32).collect(),
    ];

    let n_bins = m.n_bins() as usize;
    let frontier = client.create_from_slice(bytemuck::cast_slice(&vec![
        0u32;
        n_bins * 4 * sets.len()
    ]));

    for (slot, rows) in sets.iter().enumerate() {
        let dev_rows = engine.upload_rows(rows);
        engine.build_into(
            &dev_gpairs,
            &dev_rows,
            &frontier,
            n_bins * sets.len(),
            (slot * n_bins) as u32,
        );
    }

    let bytes = client.read_one_unchecked(frontier);
    let words: &[u32] = bytemuck::cast_slice(&bytes);

    for (slot, rows) in sets.iter().enumerate() {
        let want = cpu_histogram(&m, &gpairs, rows);
        let got: Vec<GradientPairInt64> = words[slot * n_bins * 4..(slot + 1) * n_bins * 4]
            .chunks_exact(4)
            .map(|w| GradientPairInt64 {
                grad: (w[0] as i64) | ((w[1] as i64) << 32),
                hess: (w[2] as i64) | ((w[3] as i64) << 32),
            })
            .collect();
        assert_eq!(got, want, "slot {slot}");
    }
}

/// The grower's exact shape: a parent histogram in one buffer, one child built
/// from a *slice* of a shared row index into a second buffer, and the sibling
/// subtracted into another slot of that same second buffer.
#[test]
fn builds_one_child_and_subtracts_the_sibling() {
    let client = client();
    let m = random_matrix(3000, 5, 32, 0.15, EllpackLayout::DenseCompressed, 61);
    let engine = HistogramBuilder::new(&client).build(&m).unwrap();

    let mut rng = Rng(67);
    let gpairs: Vec<GradientPairInt64> = (0..m.n_rows)
        .map(|_| GradientPairInt64 {
            grad: rng.next_u32() as i64 - i64::from(u32::MAX / 2),
            hess: i64::from(rng.next_u32() % 700) + 1,
        })
        .collect();
    let dev_gpairs = engine.upload_gpairs(&gpairs).unwrap();
    let n_bins = m.n_bins() as usize;

    // The partitioner's layout: one buffer, the node's rows a contiguous range.
    let all: Vec<u32> = (0..m.n_rows as u32).collect();
    let ridx = client.create_from_slice(bytemuck::cast_slice(&all));
    let split = 1100usize; // left child is [0, split), right is [split, n)

    let parent_buf = client.create_from_slice(bytemuck::cast_slice(&vec![0u32; n_bins * 4]));
    engine.build_into(
        &dev_gpairs,
        &DeviceRows::slice(ridx.clone(), 0, m.n_rows),
        &parent_buf,
        n_bins,
        0,
    );

    // Build the right child into slot 0, subtract the left into slot 1.
    let frontier = client.create_from_slice(bytemuck::cast_slice(&vec![0u32; n_bins * 4 * 2]));
    engine.build_into(
        &dev_gpairs,
        &DeviceRows::slice(ridx.clone(), split, m.n_rows - split),
        &frontier,
        n_bins * 2,
        0,
    );
    engine.subtract_into(&parent_buf, n_bins, 0, &frontier, n_bins * 2, 0, n_bins as u32);

    let bytes = client.read_one_unchecked(frontier);
    let words: &[u32] = bytemuck::cast_slice(&bytes);
    let read_slot = |slot: usize| -> Vec<GradientPairInt64> {
        words[slot * n_bins * 4..(slot + 1) * n_bins * 4]
            .chunks_exact(4)
            .map(|w| GradientPairInt64 {
                grad: (w[0] as i64) | ((w[1] as i64) << 32),
                hess: (w[2] as i64) | ((w[3] as i64) << 32),
            })
            .collect()
    };

    assert_eq!(read_slot(0), cpu_histogram(&m, &gpairs, &all[split..]), "built child");
    assert_eq!(read_slot(1), cpu_histogram(&m, &gpairs, &all[..split]), "subtracted sibling");
}

/// The batched path: a whole level's histograms in one launch, and a whole
/// level's siblings subtracted in one more.
///
/// This is what the grower actually issues, and it is the case a per-slot loop
/// does not exercise — a batched kernel indexes the node from a second grid
/// dimension, which is easy to get wrong in a way one node never shows.
#[test]
fn a_whole_level_builds_and_subtracts_in_one_launch_each() {
    let client = client();
    let m = random_matrix(4000, 5, 32, 0.1, EllpackLayout::DenseCompressed, 71);
    let engine = HistogramBuilder::new(&client).build(&m).unwrap();

    let mut rng = Rng(73);
    let gpairs: Vec<GradientPairInt64> = (0..m.n_rows)
        .map(|_| GradientPairInt64 {
            grad: rng.next_u32() as i64 - i64::from(u32::MAX / 2),
            hess: i64::from(rng.next_u32() % 900) + 1,
        })
        .collect();
    let dev_gpairs = engine.upload_gpairs(&gpairs).unwrap();
    let n_bins = m.n_bins() as usize;

    let all: Vec<u32> = (0..m.n_rows as u32).collect();
    let ridx = client.create_from_slice(bytemuck::cast_slice(&all));

    // Three parents, each a contiguous segment, each split somewhere.
    let parents = [(0usize, 1500usize), (1500, 1300), (2800, 1200)];
    let splits = [600usize, 400, 900]; // rows going to the built child

    let parent_buf = engine.zeroed(n_bins * 4 * parents.len());
    engine.build_into_batch(
        &dev_gpairs,
        &ridx,
        all.len(),
        &parents
            .iter()
            .enumerate()
            .map(|(i, &(b, l))| NodeHistJob {
                ridx_base: b as u32,
                n_ridx: l as u32,
                slot: (i * n_bins) as u32,
            })
            .collect::<Vec<_>>(),
        &parent_buf,
        n_bins * parents.len(),
    );

    // Build the second child of each parent, subtract the first.
    let frontier = engine.zeroed(n_bins * 4 * parents.len() * 2);
    let frontier_bins = n_bins * parents.len() * 2;
    let jobs: Vec<NodeHistJob> = parents
        .iter()
        .zip(splits)
        .enumerate()
        .map(|(i, (&(b, l), s))| NodeHistJob {
            ridx_base: (b + s) as u32,
            n_ridx: (l - s) as u32,
            slot: (i * 2 * n_bins) as u32,
        })
        .collect();
    engine.build_into_batch(&dev_gpairs, &ridx, all.len(), &jobs, &frontier, frontier_bins);
    engine.subtract_batch(
        &parent_buf,
        n_bins * parents.len(),
        &frontier,
        frontier_bins,
        &(0..parents.len())
            .map(|i| {
                ((i * n_bins) as u32, (i * 2 * n_bins) as u32, ((i * 2 + 1) * n_bins) as u32)
            })
            .collect::<Vec<_>>(),
    );

    let bytes = client.read_one_unchecked(frontier);
    let words: &[u32] = bytemuck::cast_slice(&bytes);
    let read_slot = |slot: usize| -> Vec<GradientPairInt64> {
        words[slot * n_bins * 4..(slot + 1) * n_bins * 4]
            .chunks_exact(4)
            .map(|w| GradientPairInt64 {
                grad: (w[0] as i64) | ((w[1] as i64) << 32),
                hess: (w[2] as i64) | ((w[3] as i64) << 32),
            })
            .collect()
    };

    for (i, (&(b, l), s)) in parents.iter().zip(splits).enumerate() {
        assert_eq!(
            read_slot(i * 2),
            cpu_histogram(&m, &gpairs, &all[b + s..b + l]),
            "built child of parent {i}"
        );
        assert_eq!(
            read_slot(i * 2 + 1),
            cpu_histogram(&m, &gpairs, &all[b..b + s]),
            "subtracted sibling of parent {i}"
        );
    }
}

/// A slot-0 build must still equal the standalone one, so the offset did not
/// change the existing path.
#[test]
fn slot_zero_matches_a_standalone_build() {
    let client = client();
    let m = random_matrix(1500, 5, 32, 0.0, EllpackLayout::Dense, 53);
    let engine = HistogramBuilder::new(&client).build(&m).unwrap();

    let mut rng = Rng(59);
    let gpairs: Vec<GradientPairInt64> = (0..m.n_rows)
        .map(|_| GradientPairInt64 {
            grad: rng.next_u32() as i64 - i64::from(u32::MAX / 2),
            hess: i64::from(rng.next_u32() % 500) + 1,
        })
        .collect();
    let rows: Vec<u32> = (0..m.n_rows as u32).collect();

    let standalone = engine.build(&gpairs, &rows).unwrap();
    assert_eq!(standalone, cpu_histogram(&m, &gpairs, &rows));
}
