//! Kernel tests against the CPU oracle in `xgboost_rs::reference`, mirroring
//! the checks in `xgboost/tests/cpp/tree/gpu_hist/test_histogram.cu`.
//!
//! GPU histograms must match the sequential CPU sums *exactly* (i64 equality).

use xgboost_rs::gpu::DefaultRuntime as R;
use xgboost_rs::gpu::ellpack::{DeviceEllpack, EllpackLayout, natural_bits};
use xgboost_rs::gpu::histogram::{
    HistogramBuilder, supports_atomic_add_u32, supports_native_i64_atomics,
};
use xgboost_rs::gpu::quantiser::{GradientQuantiser, quantise};
use xgboost_rs::gpu::{GradientPair, GradientPairInt64};
use xgboost_rs::reference::{cpu_histogram, random_gpairs, random_matrix};

/// A client on whichever backend this build resolved to: CUDA, wgpu/Vulkan
/// or the CubeCL CPU runtime. The kernels under test are the same either way.
fn client() -> cubecl::prelude::ComputeClient<R> {
    xgboost_rs::gpu::default_client(0)
}

/// The split gain arithmetic, the quantiser and the linear solver are ports of
/// XGBoost's `double`, so a backend with no `f64` cannot run them and refuses
/// at construction with `Error::NoF64Support`. Metal is that backend: MSL has
/// no `double` at all. Nothing to compare there — see
/// `xgboost_rs::gpu::supports_f64`.
fn has_f64() -> bool {
    xgboost_rs::gpu::supports_f64(&xgboost_rs::gpu::default_client(0))
}

#[test]
fn quantise_matches_host_reference() {
    if !has_f64() {
        return;
    }
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
    if !has_f64() {
        return;
    }
    let client = client();
    // A hessian small enough to truncate to 0 must be bumped to 1.
    let mut gpairs = random_gpairs(64, 11);
    gpairs[3].hess = 1e-30;
    let quantiser = GradientQuantiser::new(&gpairs, gpairs.len() as u64);

    let device = quantise::<R>(&client, &gpairs, &quantiser);
    assert!(device[3].hess >= 1);
}

/// A vector-leaf fit adds the targets' fixed-point sums together — a node's
/// cover, and the summed child sums a candidate carries — so one shared scale
/// has to leave room for all of them at once.
///
/// Bounding by the heaviest single target instead would fill the 62 bits below
/// the sign per target and overflow from three targets up, silently on device
/// and as a panic on the host.
#[test]
fn multi_target_quantiser_leaves_room_for_the_summed_sums() {
    let n_rows = 4096;
    // Same magnitude in every column, which is the worst case for the bound.
    let columns: Vec<Vec<GradientPair>> =
        (0..8).map(|t| random_gpairs(n_rows, 17 + t as u64)).collect();
    let views: Vec<&[GradientPair]> = columns.iter().map(Vec::as_slice).collect();
    let quantiser = GradientQuantiser::new_multi(&views, n_rows as u64);

    let mut total = GradientPairInt64::default();
    for column in &columns {
        let sum = column
            .iter()
            .map(|g| quantiser.to_fixed_point(*g))
            .fold(GradientPairInt64::default(), |a, b| a + b);
        // Checked, because the point of the bound is that this cannot wrap.
        total = GradientPairInt64 {
            grad: total.grad.checked_add(sum.grad).expect("summed gradient overflowed i64"),
            hess: total.hess.checked_add(sum.hess).expect("summed hessian overflowed i64"),
        };
    }
    assert!(total.hess > 0, "the summed hessian must survive quantisation");

    // One column is exactly the scalar quantiser, so a single-target fit keeps
    // every bit it had.
    let single = GradientQuantiser::new(&columns[0], n_rows as u64);
    let from_multi = GradientQuantiser::new_multi(&views[..1], n_rows as u64);
    assert_eq!(single.to_floating_point.grad, from_multi.to_floating_point.grad);
    assert_eq!(single.to_floating_point.hess, from_multi.to_floating_point.hess);
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

    let built = HistogramBuilder::new(&client).force_global(force_global).build(&matrix);

    // Global-memory accumulation is built on atomic adds. A runtime without
    // them has no such path, and says so rather than quietly building the
    // histogram some other way.
    if force_global && !supports_atomic_add_u32(&client) {
        assert!(
            matches!(built, Err(xgboost_rs::Error::NoGlobalHistogramPath)),
            "a runtime with no atomics must refuse force_global, got {}",
            built.err().map_or("Ok(engine)".to_owned(), |e| e.to_string())
        );
        return;
    }

    let engine = built.unwrap();
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

/// The packed device matrix, at every width `load_bin` has a shape for: 8 bits
/// (256-bin data with no hole), 9 bits (the same with the missing sentinel,
/// so an entry straddles word boundaries), 16, and the unpacked 32. The width
/// is forced rather than left to `device_bits`, so every shape runs on this
/// runtime whichever it would pick for itself, and the sparse layout's global
/// bins take the two-word path at 13 bits.
#[test]
fn histogram_reads_every_packed_width() {
    let client = client();
    let (n_rows, n_features, bins) = (3000, 6, 256);
    let cases = [
        (0.0, EllpackLayout::Dense, 8),
        (0.25, EllpackLayout::DenseCompressed, 9),
        (0.25, EllpackLayout::DenseCompressed, 16),
        (0.25, EllpackLayout::DenseCompressed, 32),
        (0.5, EllpackLayout::Sparse, 13),
    ];
    for (sparsity, layout, bits) in cases {
        let matrix = random_matrix(n_rows, n_features, bins, sparsity, layout, 33);
        assert!(natural_bits(&matrix) <= bits, "{layout:?} needs {} bits", natural_bits(&matrix));
        let gpairs = random_gpairs(n_rows, 35);
        let quantiser = GradientQuantiser::new(&gpairs, n_rows as u64);
        let quantised = quantise::<R>(&client, &gpairs, &quantiser);
        let ridx: Vec<u32> = (0..n_rows as u32).filter(|r| r % 5 != 1).collect();

        let ell = DeviceEllpack::upload_with_bits(&client, &matrix, bits);
        assert_eq!(ell.bits, bits);
        let engine = HistogramBuilder::new(&client).build_shared(&matrix, &ell).unwrap();
        assert_eq!(
            engine.build(&quantised, &ridx).unwrap(),
            cpu_histogram(&matrix, &quantised, &ridx),
            "layout {layout:?} at {bits} bits"
        );
    }
}

/// Enough rows that the atomic-free path cuts a node into several row chunks,
/// each accumulated by its own unit and merged afterwards; the atomic path's
/// grid-strided tile loop walks the same rows. Both have to give the oracle.
#[test]
fn histogram_many_row_chunks() {
    let client = client();
    let (n_rows, n_features, bins) = (50_000, 8, 24);
    let matrix = random_matrix(n_rows, n_features, bins, 0.0, EllpackLayout::Dense, 21);
    let gpairs = random_gpairs(n_rows, 19);
    let quantiser = GradientQuantiser::new(&gpairs, n_rows as u64);
    let quantised = quantise::<R>(&client, &gpairs, &quantiser);
    let ridx: Vec<u32> = (0..n_rows as u32).filter(|r| r % 7 != 0).collect();

    let engine = HistogramBuilder::new(&client).build(&matrix).unwrap();
    assert_eq!(
        engine.build(&quantised, &ridx).unwrap(),
        cpu_histogram(&matrix, &quantised, &ridx)
    );
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

    let built = HistogramBuilder::new(&client).shmem_bytes(4096).build(&matrix);

    // A runtime with no atomics privatises the whole histogram in global
    // memory, one copy per work item, so there is no budget to overflow: the
    // engine builds, and has to match the oracle like any other.
    if !supports_atomic_add_u32(&client) {
        let engine = built.unwrap();
        assert!(engine.uses_shared_memory(), "private partials are the privatised path");
        assert_eq!(
            engine.build(&quantised, &ridx).unwrap(),
            cpu_histogram(&matrix, &quantised, &ridx)
        );
        return;
    }

    let engine = built.unwrap();
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

/// A depth-10 frontier is ~26M `u32` words. Sizing that grid with a `div_ceil`
/// into X alone asks for more than 65,535 cubes, which wgpu rejects with a
/// validation error rather than clamping — so the clear used to abort the
/// process on a deep enough tree. `gpu::launch::elementwise` spreads the count
/// across the other axes instead.
#[test]
fn zeroed_clears_a_buffer_too_large_for_one_grid_axis() {
    let client = client();
    let matrix = random_matrix(64, 4, 8, 0.0, EllpackLayout::Dense, 1);
    let engine = HistogramBuilder::new(&client).build(&matrix).unwrap();

    // Comfortably past 65_535 * 256, the largest a single-axis grid of the
    // old fixed 256-unit block could address.
    let words = 70_000usize * 256;
    let handle = engine.zeroed(words);

    let got: Vec<u32> = bytemuck::cast_slice(&client.read_one_unchecked(handle)).to_vec();
    assert_eq!(got.len(), words);
    assert!(got.iter().all(|w| *w == 0), "buffer was not fully cleared");
}

/// The vectorised clear may only use a width that divides the buffer exactly;
/// a partial trailing vector would run off the end of the allocation.
#[test]
fn zeroed_handles_lengths_no_vector_width_divides() {
    let client = client();
    let matrix = random_matrix(64, 4, 8, 0.0, EllpackLayout::Dense, 1);
    let engine = HistogramBuilder::new(&client).build(&matrix).unwrap();

    for words in [1usize, 2, 3, 5, 7, 255, 257, 1023] {
        let handle = engine.zeroed(words);
        let got: Vec<u32> = bytemuck::cast_slice(&client.read_one_unchecked(handle)).to_vec();
        assert_eq!(got.len(), words, "words={words}");
        assert!(got.iter().all(|w| *w == 0), "words={words} not fully cleared");
    }
}

/// The device binning route — `bin_csr_kernel` and `pack_bins_kernel` — has
/// to produce the words `build_ellpack` + `pack_bins` produce, both layouts,
/// with and without holes, and at a width the runtime would not pick on its
/// own. Word equality rather than histogram equality: a bin that lands in the
/// wrong cell could still sum right by luck, a word cannot.
#[test]
fn device_binning_matches_the_host_ellpack() {
    use xgboost_rs::DMatrix;
    use xgboost_rs::data::cuts::build_cuts;
    use xgboost_rs::gpu::ellpack::{build_ellpack, pack_bins};

    let client = client();
    let (n_rows, n_cols) = (2011usize, 7usize);
    for sparsity in [0.0f32, 0.3] {
        // Values over a wide range so several features hit every bin,
        // including ties on a cut, which is where `search_bin` has to agree.
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let x: Vec<f32> = (0..n_rows * n_cols)
            .map(|i| {
                let r = next();
                if sparsity > 0.0 && (r % 1000) as f32 / 1000.0 < sparsity {
                    f32::NAN
                } else {
                    // Quantised to a coarse grid so ties are common.
                    ((r >> 20) % 97) as f32 * 0.25 - 12.0 + (i % 3) as f32 * 1e-3
                }
            })
            .collect();
        let dmat = DMatrix::from_dense(&x, n_rows, n_cols, f32::NAN).unwrap();
        let cuts = build_cuts(&dmat, 32).unwrap();

        let host = build_ellpack(&dmat, &cuts);
        let values = xgboost_rs::gpu::sketch::DeviceValues::upload(&client, &dmat);
        let (dev, shape) = DeviceEllpack::bin_on_device(&client, &values, &cuts);
        assert_eq!(shape.layout, host.layout, "sparsity {sparsity}");
        assert_eq!(shape.null_value, host.null_value);
        assert_eq!(dev.dense, host.is_dense());

        let expect = pack_bins(&host.gidx, dev.bits);
        let got: Vec<u32> = bytemuck::cast_slice(&client.read_one_unchecked(dev.gidx.clone())).to_vec();
        // Whole chunks on the device, entries plus a pad on the host: the
        // data words are the comparison, not the tails.
        let n = expect.len() - 1;
        assert!(got.len() > n);
        assert_eq!(&got[..n], &expect[..n], "row-major words at {} bits, sparsity {sparsity}", dev.bits);

        let mut transposed = vec![0u32; host.gidx.len()];
        for r in 0..n_rows {
            for f in 0..n_cols {
                transposed[f * n_rows + r] = host.gidx[r * n_cols + f];
            }
        }
        let expect_t = pack_bins(&transposed, dev.bits);
        let got_t: Vec<u32> = bytemuck::cast_slice(&client.read_one_unchecked(dev.gidx_t.clone())).to_vec();
        assert_eq!(&got_t[..n], &expect_t[..n], "feature-major words at {} bits, sparsity {sparsity}", dev.bits);
    }
}

/// The device sketch against the exact reference: every column sorted on the
/// host and read off by `exact_cuts_from_sorted`, which is `query_cut_values`
/// over an exact summary. Continuous columns, a column with heavy ties so the
/// rank queries repeat, a low-cardinality column under the distinct-value
/// branch, an all-missing column, and holes throughout.
#[test]
fn device_sketch_matches_the_exact_cuts() {
    let client = client();
    // Under one sort tile, several tiles, and enough rows that the
    // `(digit, tile)` table spans many chunks of the scan.
    for (n_rows, n_cols) in [(200usize, 4usize), (5003, 6), (70_001, 3)] {
        device_sketch_case(&client, n_rows, n_cols);
    }
}

fn device_sketch_case(client: &cubecl::prelude::ComputeClient<R>, n_rows: usize, n_cols: usize) {
    use xgboost_rs::DMatrix;
    use xgboost_rs::data::cuts::exact_cuts_from_sorted;
    use xgboost_rs::gpu::sketch::{DeviceValues, applies, device_cuts};

    let mut seed = 0x2545_F491_4F6C_DD1Du64 ^ (n_rows as u64);
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let x: Vec<f32> = (0..n_rows * n_cols)
        .map(|i| {
            let r = next();
            let f = i % n_cols;
            match f {
                // Continuous, both signs, with a few holes.
                0 => if r % 50 == 0 { f32::NAN } else { ((r >> 16) % 100_000) as f32 / 977.0 - 51.0 },
                // Continuous, positive.
                1 => ((r >> 8) % 1_000_003) as f32 * 1e-4,
                // Heavy ties: a few dozen distinct values, more than max_bin.
                2 => ((r >> 12) % 40) as f32 * 0.5,
                // Low cardinality: three values, under max_bin.
                3 => ((r >> 20) % 3) as f32,
                // All missing.
                4 => f32::NAN,
                // Mostly one value, so the rank grid repeats and the chain
                // has to step to the next distinct value.
                _ => if r % 1000 < 3 { ((r >> 24) % 5) as f32 + 1.0 } else { 0.0 },
            }
        })
        .collect();
    let dmat = DMatrix::from_dense(&x, n_rows, n_cols, f32::NAN).unwrap();
    let max_bin = 16u32;
    if !applies(client, &dmat, max_bin) {
        return;
    }
    let values = DeviceValues::upload(client, &dmat);
    let got = device_cuts(client, &values, max_bin);

    for f in 0..n_cols {
        let mut column: Vec<f32> =
            (0..n_rows).map(|r| x[r * n_cols + f]).filter(|v| !v.is_nan()).collect();
        column.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let expect = exact_cuts_from_sorted(&column, max_bin as usize);
        let (b, e) = (got.cut_ptrs[f] as usize, got.cut_ptrs[f + 1] as usize);
        assert_eq!(&got.cut_values[b..e], &expect[..], "{n_rows}x{n_cols} feature {f}");
        assert_eq!(got.min_values[f], f32::NEG_INFINITY);
    }
}

/// The device prediction update against the host's: rows spread over the
/// partitioner's two buffers, segments in arbitrary order, leaf values
/// added with a tree weight. Exact, as the host arithmetic is the same.
#[test]
fn device_prediction_update_matches_the_host() {
    use xgboost_rs::gpu::objective::DeviceRound;
    use xgboost_rs::gpu::tables::upload_vec;

    let client = client();
    let n = 5000usize;
    let info = xgboost_rs::data::MetaInfo {
        num_row: n,
        num_col: 1,
        labels: (0..n).map(|i| (i % 7) as f32).collect(),
        ..Default::default()
    };
    let mut preds: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
    let mut round = DeviceRound::new(&client, &info, &preds);

    // Two row buffers, as the partitioner's: the segments taken from each
    // together cover every row exactly once — positions `0..1000` are read
    // from the first (rows 4999 down to 4000) and `1000..5000` from the
    // second (a permutation of rows 0..4000).
    let perm0: Vec<u32> = (0..n as u32).rev().collect();
    let perm1: Vec<u32> = (0..n as u32)
        .map(|i| if i < 1000 { 4000 + i } else { ((i - 1000) * 7919) % 4000 })
        .collect();
    let b0 = upload_vec(&client, perm0.clone());
    let b1 = upload_vec(&client, perm1.clone());
    let segments = [(1u32, 3000u32, 2000u32, 0.5f32), (0, 0, 1000, -1.25), (1, 1000, 2000, 0.125)];
    let weight = 0.3f32;
    round.update_predictions(&client, [&b0, &b1], &segments, weight);

    for &(side, begin, len, value) in &segments {
        let buf = if side == 0 { &perm0 } else { &perm1 };
        for &row in &buf[begin as usize..(begin + len) as usize] {
            preds[row as usize] += value * weight;
        }
    }
    let mut got = vec![0f32; n];
    round.download(&client, &mut got);
    assert_eq!(got, preds);
}
