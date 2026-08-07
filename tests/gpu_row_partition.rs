//! The device row partitioner against a sequential CPU reference.
//!
//! The reference is `HistGrower::goes_left` + `partition_block`
//! (`src/tree/hist.rs`): a stable partition of each node segment, left rows
//! first. Row order is not something the model depends on — histograms sum
//! integers — but keeping it lets the two sides be compared element for
//! element, which is a far sharper test than comparing sets.

use cubecl::Runtime;
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

use xgboost_rs::gpu::ellpack::{DeviceEllpack, EllpackLayout, EllpackMatrix};
use xgboost_rs::gpu::row_partitioner::{RowPartitioner, SegmentSplit};
use xgboost_rs::reference::random_matrix;

type R = WgpuRuntime;

fn client() -> ComputeClient<R> {
    R::client(&WgpuDevice::default())
}

/// Port of `HistGrower::goes_left`, reading the ELLPACK the way the kernel does.
fn cpu_goes_left(m: &EllpackMatrix, row: u32, split: &SegmentSplit) -> bool {
    let f = split.fidx as usize;
    let row_begin = (row - m.base_rowid) as usize * m.row_stride;
    let (fb, fe) = (m.cut_ptrs[f], m.cut_ptrs[f + 1]);

    let raw = if m.is_compressed() {
        m.gidx[row_begin + f]
    } else {
        (0..m.row_stride)
            .map(|k| m.gidx[row_begin + k])
            .find(|&v| v != m.null_value && v >= fb && v < fe)
            .unwrap_or(m.null_value)
    };

    if !m.is_dense() && raw == m.null_value {
        return split.default_left;
    }
    let local = if m.is_compressed() { raw } else { raw - fb };
    if !split.cat_bits.is_empty() {
        // The bit set names the categories going *right*.
        (split.cat_bits[(local / 32) as usize] >> (local % 32)) & 1 == 0
    } else {
        (local as i64) <= split.cond
    }
}

/// `partition_block`: stable on both sides, left rows first.
fn cpu_partition(m: &EllpackMatrix, rows: &[u32], split: &SegmentSplit) -> (Vec<u32>, u32) {
    let (mut left, mut right) = (Vec::new(), Vec::new());
    for &r in rows {
        if cpu_goes_left(m, r, split) {
            left.push(r);
        } else {
            right.push(r);
        }
    }
    let n_left = left.len() as u32;
    left.extend(right);
    (left, n_left)
}

/// Partition `splits` on both sides and assert the whole row index agrees.
fn check(m: &EllpackMatrix, initial: &[u32], splits: &[SegmentSplit]) {
    let client = client();
    let ell = DeviceEllpack::upload(&client, m);
    let mut part = RowPartitioner::<R>::new(client, initial);

    let got_counts = part.partition(&ell, splits).unwrap();
    let got_rows = part.read();

    let mut want_rows = initial.to_vec();
    let mut want_counts = Vec::new();
    for sp in splits {
        let (b, l) = (sp.begin as usize, sp.len as usize);
        let (partitioned, n_left) = cpu_partition(m, &want_rows[b..b + l], sp);
        want_rows[b..b + l].copy_from_slice(&partitioned);
        want_counts.push(n_left);
    }

    assert_eq!(got_counts, want_counts, "left counts");
    assert_eq!(got_rows, want_rows, "row index");
}

fn numeric(begin: u32, len: u32, fidx: u32, cond: i64, default_left: bool) -> SegmentSplit {
    SegmentSplit { begin, len, fidx, cond, default_left, cat_bits: Vec::new() }
}

#[test]
fn dense_root_split() {
    let m = random_matrix(5000, 8, 32, 0.0, EllpackLayout::Dense, 3);
    let rows: Vec<u32> = (0..5000).collect();
    check(&m, &rows, &[numeric(0, 5000, 3, 15, false)]);
}

/// More rows than one tile, so the tile scan has to carry.
#[test]
fn spans_many_tiles() {
    let m = random_matrix(100_000, 4, 64, 0.0, EllpackLayout::Dense, 5);
    let rows: Vec<u32> = (0..100_000).collect();
    check(&m, &rows, &[numeric(0, 100_000, 1, 31, false)]);
}

/// A whole level at once: several segments, each with its own split.
#[test]
fn partitions_a_level_of_segments() {
    let m = random_matrix(20_000, 6, 32, 0.0, EllpackLayout::Dense, 7);
    let rows: Vec<u32> = (0..20_000).collect();
    let splits = vec![
        numeric(0, 5_000, 0, 10, false),
        numeric(5_000, 3_000, 2, 20, true),
        numeric(8_000, 9_000, 5, 4, false),
        numeric(17_000, 3_000, 1, 28, true),
    ];
    check(&m, &rows, &splits);
}

/// Segments not named in the batch must keep their rows untouched.
#[test]
fn leaves_other_segments_alone() {
    let m = random_matrix(10_000, 4, 16, 0.0, EllpackLayout::Dense, 11);
    let rows: Vec<u32> = (0..10_000).rev().collect(); // deliberately not sorted
    check(&m, &rows, &[numeric(2_000, 3_000, 1, 8, false)]);
}

#[test]
fn dense_compressed_missing_values_follow_the_default() {
    let m = random_matrix(30_000, 5, 32, 0.35, EllpackLayout::DenseCompressed, 13);
    let rows: Vec<u32> = (0..30_000).collect();
    for default_left in [false, true] {
        check(&m, &rows, &[numeric(0, 30_000, 2, 15, default_left)]);
    }
}

#[test]
fn sparse_layout_finds_the_features_bin() {
    let m = random_matrix(20_000, 6, 24, 0.4, EllpackLayout::Sparse, 17);
    let rows: Vec<u32> = (0..20_000).collect();
    for default_left in [false, true] {
        check(&m, &rows, &[numeric(0, 20_000, 4, 11, default_left)]);
    }
}

/// A categorical split routes by membership of the right-hand category set.
#[test]
fn categorical_split_tests_the_bit_set() {
    let m = random_matrix(20_000, 4, 40, 0.2, EllpackLayout::DenseCompressed, 19);
    let rows: Vec<u32> = (0..20_000).collect();
    // Categories 0, 5, 6, 7 and 33 go right.
    let mut cat_bits = vec![0u32; 2];
    for c in [0u32, 5, 6, 7, 33] {
        cat_bits[(c / 32) as usize] |= 1 << (c % 32);
    }
    check(
        &m,
        &rows,
        &[SegmentSplit { begin: 0, len: 20_000, fidx: 1, cond: 0, default_left: true, cat_bits }],
    );
}

/// Every row on one side is the degenerate case the scan has to survive.
#[test]
fn handles_all_left_and_all_right() {
    let m = random_matrix(4_000, 4, 16, 0.0, EllpackLayout::Dense, 23);
    let rows: Vec<u32> = (0..4_000).collect();
    check(&m, &rows, &[numeric(0, 4_000, 0, 15, false)]); // every bin <= 15
    check(&m, &rows, &[numeric(0, 4_000, 0, -1, false)]); // no bin <= -1
}

/// Segments shorter than a tile, and a segment of one row.
#[test]
fn handles_tiny_segments() {
    let m = random_matrix(1_000, 4, 16, 0.0, EllpackLayout::Dense, 29);
    let rows: Vec<u32> = (0..1_000).collect();
    check(
        &m,
        &rows,
        &[numeric(0, 1, 0, 7, false), numeric(1, 2, 1, 7, false), numeric(3, 300, 2, 7, false)],
    );
}

/// Two levels in sequence: the children of a split must themselves partition.
#[test]
fn successive_levels_compose() {
    let m = random_matrix(50_000, 5, 32, 0.1, EllpackLayout::DenseCompressed, 31);
    let rows: Vec<u32> = (0..50_000).collect();
    let client = client();
    let ell = DeviceEllpack::upload(&client, &m);
    let mut part = RowPartitioner::<R>::new(client, &rows);

    let root = numeric(0, 50_000, 2, 15, false);
    let n_left = part.partition(&ell, std::slice::from_ref(&root)).unwrap()[0];

    let level2 = vec![
        numeric(0, n_left, 0, 10, true),
        numeric(n_left, 50_000 - n_left, 3, 20, false),
    ];
    let counts2 = part.partition(&ell, &level2).unwrap();
    let got = part.read();

    // Same thing on the host.
    let (mut want, want_left) = cpu_partition(&m, &rows, &root);
    assert_eq!(want_left, n_left);
    let mut want_counts = Vec::new();
    for sp in &level2 {
        let (b, l) = (sp.begin as usize, sp.len as usize);
        let (p, nl) = cpu_partition(&m, &want[b..b + l], sp);
        want[b..b + l].copy_from_slice(&p);
        want_counts.push(nl);
    }

    assert_eq!(counts2, want_counts);
    assert_eq!(got, want);
}
