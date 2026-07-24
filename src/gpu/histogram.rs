//! Gradient histogram construction, ported from
//! `xgboost/src/tree/gpu_hist/histogram.{cu,cuh}` (single-target path:
//! `StHistKernel` / `HistKernelOneNodeTarget` / `AtomicAddGpairShared` /
//! `AtomicAddGpairGlobal` / `AtomicAdd64As32`) and the subtraction trick from
//! `DeviceHistogramBuilder::SubtractionTrick`.
//!
//! # Host API
//!
//! [`HistogramBuilder`] configures the engine and uploads the ELLPACK matrix
//! once; the resulting [`HistogramEngine`] then builds any number of node
//! histograms against device-resident data — mirroring how XGBoost's
//! `DeviceHistogramBuilder` is `Reset` once per iteration and dispatched per
//! node.
//!
//! # Accumulator representation
//!
//! A histogram bin is a `GradientPairInt64` (two `i64`s). On device it is
//! stored as four `u32` words `[grad_lo, grad_hi, hess_lo, hess_hi]` and
//! updated with two 32-bit atomics plus manual carry propagation — the exact
//! scheme of `AtomicAdd64As32` in histogram.cuh. XGBoost uses it for shared
//! memory because 64-bit shared atomics are slow on NVIDIA hardware; here it
//! is also used for global memory so the kernel runs on runtimes without
//! 64-bit atomics (WebGPU, older Vulkan devices). The end result is bitwise
//! identical either way: two's-complement addition is associative and
//! commutative, so the quantised sums are deterministic.

use cubecl::features::AtomicUsage;
use cubecl::ir::{StorageType, Type};
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::ellpack::EllpackMatrix;
use super::{DeviceGpairs, DeviceHistogram, DeviceRows, GradientPairInt64};
use crate::error::{Error, Result};

/// `kItemsPerThread` in histogram.cu.
pub const ITEMS_PER_THREAD: u32 = 8;
/// Block size. The CUDA code tunes this per SM architecture (768–1024);
/// 256 is a portable choice within every backend's workgroup limits.
pub const BLOCK_THREADS: u32 = 256;
/// Default shared-memory budget per workgroup (the CUDA code queries
/// `MaxSharedMemoryOptin`; 48 KiB is the portable baseline).
pub const DEFAULT_SHMEM_BYTES: usize = 48 * 1024;
/// Default upper bound on the grid size, standing in for the occupancy-based
/// `n_blocks_per_mp * n_mps` limit computed by `HistKernel::SetCfg`.
pub const DEFAULT_MAX_BLOCKS_PER_GROUP: u32 = 1024;

/// Port of `AtomicAdd64As32` (histogram.cuh): add a signed 64-bit value into
/// two consecutive `u32` words `[lo, hi]` using 32-bit atomics with carry.
#[cube]
fn atomic_add_i64_as_u32_shared(dst: &SharedMemory<Atomic<u32>>, word: u32, val: i64) {
    let lo = u32::cast_from(val & 0xFFFF_FFFFi64);
    let hi = u32::cast_from((val >> 32) & 0xFFFF_FFFFi64);
    let old = dst[word as usize].fetch_add(lo);
    let carry = u32::cast_from(old > 0xFFFF_FFFFu32 - lo);
    dst[(word + 1) as usize].fetch_add(hi + carry);
}

/// Same carry scheme against a global-memory histogram, used when the runtime
/// has no native 64-bit atomics; yields bit-identical results.
#[cube]
fn atomic_add_i64_as_u32_global(dst: &Array<Atomic<u32>>, word: u32, val: i64) {
    let lo = u32::cast_from(val & 0xFFFF_FFFFi64);
    let hi = u32::cast_from((val >> 32) & 0xFFFF_FFFFi64);
    let old = dst[word as usize].fetch_add(lo);
    let carry = u32::cast_from(old > 0xFFFF_FFFFu32 - lo);
    dst[(word + 1) as usize].fetch_add(hi + carry);
}

/// Global-memory gradient-pair add: the native branch is the port of
/// `AtomicAddGpairGlobal` (histogram.cu), which relies on 64-bit `atomicAdd`
/// ("global 64 bit integer atomics at the time of writing do not benefit from
/// being separated into two 32 bit atomics"); the fallback is the u32-carry
/// scheme. Exactly one of `hist32` / `hist64` is real — the other is a 1-word
/// dummy, and the comptime branch removes every access to it.
#[cube]
fn add_gpair_global(
    hist32: &Array<Atomic<u32>>,
    hist64: &Array<Atomic<i64>>,
    bin: u32,
    grad: i64,
    hess: i64,
    #[comptime] native_i64: bool,
) {
    if native_i64 {
        hist64[(bin * 2) as usize].fetch_add(grad);
        hist64[(bin * 2 + 1) as usize].fetch_add(hess);
    } else {
        atomic_add_i64_as_u32_global(hist32, bin * 4, grad);
        atomic_add_i64_as_u32_global(hist32, bin * 4 + 2, hess);
    }
}

/// Port of `StHistKernel` + `HistKernelOneNodeTarget` (histogram.cu): build
/// the gradient histogram for one node, one target.
///
/// Grid: `CUBE_POS_X` indexes the grid-strided tile loop, `CUBE_POS_Y` the
/// feature group (as in the CUDA `dim3 conf(n_blocks, n_groups)` launch).
///
/// * `gidx` — ELLPACK bin matrix, `n_rows * row_stride` entries.
/// * `cut_ptrs` — per-feature bin offsets (`feature_segments`).
/// * `groups` — feature groups, 4 words each:
///   `[start_feature, num_features, start_bin, num_bins]`.
/// * `ridx` — row indices belonging to the node.
/// * `gpair` — quantised gradients, interleaved `[grad, hess]` per row.
/// * `hist` — global histogram as 4 `u32` words per bin (used when
///   `native_i64` is false; 1-word dummy otherwise).
/// * `hist_i64` — global histogram as 2 `i64` words per bin (used when
///   `native_i64` is true; 1-word dummy otherwise). Same byte layout.
/// * `dense` / `compressed` — comptime layout flags (`kDense`/`kCompressed`).
/// * `use_shared` — comptime `kSharedMem`: privatise the group's bins in
///   shared memory, then flush to global.
/// * `native_i64` — comptime: use native 64-bit atomics for global-memory
///   accumulation (`AtomicAddGpairGlobal`) instead of the u32-carry scheme.
/// * `smem_words` — comptime shared buffer size; must be at least
///   `4 * max(group num_bins)` when `use_shared` (4 = `SMEM_MIN_WORDS` filler
///   otherwise, since the declaration cannot be elided).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn hist_kernel(
    gidx: &Array<u32>,
    cut_ptrs: &Array<u32>,
    groups: &Array<u32>,
    ridx: &Array<u32>,
    gpair: &Array<i64>,
    hist: &mut Array<Atomic<u32>>,
    hist_i64: &mut Array<Atomic<i64>>,
    row_stride: u32,
    base_rowid: u32,
    null_value: u32,
    n_ridx: u32,
    #[comptime] dense: bool,
    #[comptime] compressed: bool,
    #[comptime] use_shared: bool,
    #[comptime] native_i64: bool,
    #[comptime] smem_words: usize,
) {
    let group_base = (CUBE_POS_Y * 4) as usize;
    let start_feature = groups[group_base];
    let num_features = groups[group_base + 1];
    let start_bin = groups[group_base + 2];
    let num_bins = groups[group_base + 3];

    let smem = SharedMemory::<Atomic<u32>>::new(smem_words);

    if use_shared {
        // dh::BlockFill of the privatised histogram.
        let mut i = UNIT_POS_X;
        while i < num_bins * 4 {
            smem[i as usize].store(0u32);
            i += CUBE_DIM_X;
        }
        sync_cube();
    }

    // `feature_stride` in HistKernelOneNodeTarget: for compressed layouts each
    // block only walks its group's features, otherwise the full row stride.
    let feature_stride = if compressed { num_features } else { row_stride };
    let n_elements = n_ridx * feature_stride;

    let tile_size = CUBE_DIM_X * ITEMS_PER_THREAD;
    // Grid-strided tile loop over (row, feature) pairs.
    let mut offset = CUBE_POS_X * tile_size;
    let stride = tile_size * CUBE_COUNT_X;

    while offset < n_elements {
        #[unroll]
        for j in 0..ITEMS_PER_THREAD {
            let idx = offset + j * CUBE_DIM_X + UNIT_POS_X;
            if idx < n_elements {
                // process_valid_tile: unravel idx -> (row-in-set, feature-in-group).
                let ridx_in_set = idx / feature_stride;
                let fidx_in_set = idx - ridx_in_set * feature_stride;

                let row = ridx[ridx_in_set as usize];
                let fidx = fidx_in_set + start_feature;

                // IterIdx: entry for (row, fidx) in the ELLPACK matrix.
                let entry = (row - base_rowid) * row_stride + fidx;
                let bin = gidx[entry as usize];

                if dense || bin != null_value {
                    let grad = gpair[(2 * row) as usize];
                    let hess = gpair[(2 * row + 1) as usize];

                    let mut global_bin = bin;
                    if compressed {
                        global_bin += cut_ptrs[fidx as usize];
                    }

                    if use_shared {
                        let local = global_bin - start_bin;
                        atomic_add_i64_as_u32_shared(&smem, local * 4, grad);
                        atomic_add_i64_as_u32_shared(&smem, local * 4 + 2, hess);
                    } else {
                        add_gpair_global(hist, hist_i64, global_bin, grad, hess, native_i64);
                    }
                }
            }
        }
        offset += stride;
    }

    if use_shared {
        // Flush the privatised histogram back to global memory.
        sync_cube();
        let mut bin = UNIT_POS_X;
        while bin < num_bins {
            let src = (bin * 4) as usize;
            // The shared accumulator words already hold a finished i64 pair;
            // re-assemble and push it through the global add.
            let grad = i64::cast_from(smem[src].load())
                | (i64::cast_from(smem[src + 1].load()) << 32);
            let hess = i64::cast_from(smem[src + 2].load())
                | (i64::cast_from(smem[src + 3].load()) << 32);
            add_gpair_global(hist, hist_i64, start_bin + bin, grad, hess, native_i64);
            bin += CUBE_DIM_X;
        }
    }
}

/// Port of the `SubtractionTrick` device lambda (histogram.cuh):
/// `sibling = parent - built`, elementwise over interleaved i64 words.
#[cube(launch)]
pub fn subtract_hist_kernel(
    parent: &Array<i64>,
    built: &Array<i64>,
    out: &mut Array<i64>,
    n_words: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i < n_words {
        out[i as usize] = parent[i as usize] - built[i as usize];
    }
}

/// A contiguous run of features sharing one shared-memory histogram, the
/// host-side analogue of `FeatureGroup` (feature_groups.cuh).
#[derive(Clone, Copy, Debug)]
pub struct FeatureGroup {
    pub start_feature: u32,
    pub num_features: u32,
    pub start_bin: u32,
    pub num_bins: u32,
}

/// Port of the grouping logic in `FeatureGroups` (feature_groups.cu): pack
/// consecutive features into groups whose bins fit in `shmem_bytes` of shared
/// memory. Returns `None` if some single feature does not fit, in which case
/// the caller must fall back to one all-features group in global memory.
///
/// Only valid for **compressed** (feature-local bin) matrices: the multi-group
/// shared path relies on each block walking exactly its group's features. For
/// sparse (global-bin) matrices the caller must not split into multiple groups
/// (see `HistogramBuilder::build`).
pub fn build_feature_groups(cut_ptrs: &[u32], shmem_bytes: usize) -> Option<Vec<FeatureGroup>> {
    // 4 u32 accumulator words per bin.
    let max_bins = (shmem_bytes / (4 * core::mem::size_of::<u32>())) as u32;
    let n_features = cut_ptrs.len() - 1;

    let mut groups = Vec::new();
    let mut start = 0usize;
    while start < n_features {
        let mut end = start;
        while end < n_features && cut_ptrs[end + 1] - cut_ptrs[start] <= max_bins {
            end += 1;
        }
        if end == start {
            return None; // A single feature exceeds the shared-memory budget.
        }
        groups.push(FeatureGroup {
            start_feature: start as u32,
            num_features: (end - start) as u32,
            start_bin: cut_ptrs[start],
            num_bins: cut_ptrs[end] - cut_ptrs[start],
        });
        start = end;
    }
    Some(groups)
}

/// Filler size for the (mandatory) shared-memory declaration on the
/// global-memory path.
const SMEM_MIN_WORDS: usize = 4;

/// Whether the runtime supports native `i64` atomic adds (e.g. CUDA, or
/// Vulkan devices with `VK_KHR_shader_atomic_int64`).
pub fn supports_native_i64_atomics<R: Runtime>(client: &ComputeClient<R>) -> bool {
    let ty = StorageType::Atomic(i64::as_type_native_unchecked().elem_type());
    client.properties().atomic_type_usage(Type::new(ty)).contains(AtomicUsage::Add)
}

/// Configures a [`HistogramEngine`]; the analogue of
/// `DeviceHistogramBuilder::Reset` + `HistKernel`'s constructor.
///
/// ```no_run
/// # use xgboost_rs::gpu::histogram::HistogramBuilder;
/// # use cubecl::Runtime;
/// # let client = cubecl::wgpu::WgpuRuntime::client(&Default::default());
/// # let matrix: xgboost_rs::gpu::ellpack::EllpackMatrix = unimplemented!();
/// let engine = HistogramBuilder::new(&client)
///     .shmem_bytes(48 * 1024)
///     .force_global(false)
///     .build(&matrix)?;
/// # Ok::<(), xgboost_rs::Error>(())
/// ```
pub struct HistogramBuilder<'a, R: Runtime> {
    client: &'a ComputeClient<R>,
    shmem_bytes: usize,
    force_global: bool,
    max_blocks_per_group: u32,
    native_i64_atomics: Option<bool>,
}

impl<'a, R: Runtime> HistogramBuilder<'a, R> {
    pub fn new(client: &'a ComputeClient<R>) -> Self {
        Self {
            client,
            shmem_bytes: DEFAULT_SHMEM_BYTES,
            force_global: false,
            max_blocks_per_group: DEFAULT_MAX_BLOCKS_PER_GROUP,
            native_i64_atomics: None,
        }
    }

    /// Per-workgroup shared-memory budget for the privatised histogram path.
    pub fn shmem_bytes(mut self, bytes: usize) -> Self {
        self.shmem_bytes = bytes;
        self
    }

    /// Force the global-memory path (the `force_global_memory` test hook of
    /// `DeviceHistogramBuilder::Reset`).
    pub fn force_global(mut self, force: bool) -> Self {
        self.force_global = force;
        self
    }

    /// Cap on the grid size per feature group.
    pub fn max_blocks_per_group(mut self, n: u32) -> Self {
        self.max_blocks_per_group = n.max(1);
        self
    }

    /// Force native 64-bit atomics on (`true`) or off (`false`) for
    /// global-memory accumulation. Default: auto-detect from the runtime
    /// ([`supports_native_i64_atomics`]). Forcing `true` on a runtime without
    /// support fails at kernel compilation.
    pub fn native_i64_atomics(mut self, enable: bool) -> Self {
        self.native_i64_atomics = Some(enable);
        self
    }

    /// Validate the matrix, decide the shared/global dispatch, and upload the
    /// matrix to the device.
    pub fn build(self, matrix: &EllpackMatrix) -> Result<HistogramEngine<R>> {
        if matrix.cut_ptrs.len() < 2 {
            return Err(Error::InvalidCuts { got: matrix.cut_ptrs.len() });
        }
        let expected = matrix.n_rows * matrix.row_stride;
        if matrix.gidx.len() != expected {
            return Err(Error::MatrixShape { expected, got: matrix.gidx.len() });
        }

        let groups = if self.force_global {
            None
        } else if matrix.is_compressed() {
            // Feature-local bins: a block can walk just its group's features
            // (feature_stride = num_features), so multi-group shared is valid.
            build_feature_groups(&matrix.cut_ptrs, self.shmem_bytes)
        } else {
            // Sparse bins are global; the kernel walks the full row per block
            // (feature_stride = row_stride), so only a single all-features
            // group is valid for shared memory. Use it if every bin fits,
            // otherwise fall back to global.
            let n_bins = matrix.n_bins() as usize;
            if n_bins * 4 * core::mem::size_of::<u32>() <= self.shmem_bytes {
                Some(vec![FeatureGroup {
                    start_feature: 0,
                    num_features: matrix.n_features() as u32,
                    start_bin: 0,
                    num_bins: matrix.n_bins(),
                }])
            } else {
                None
            }
        };
        let use_shared = groups.is_some();
        let groups = groups.unwrap_or_else(|| {
            vec![FeatureGroup {
                start_feature: 0,
                num_features: matrix.n_features() as u32,
                start_bin: 0,
                num_bins: matrix.n_bins(),
            }]
        });

        let max_group_bins = groups.iter().map(|g| g.num_bins).max().unwrap() as usize;
        let smem_words =
            if use_shared { (max_group_bins * 4).max(SMEM_MIN_WORDS) } else { SMEM_MIN_WORDS };

        let groups_flat: Vec<u32> = groups
            .iter()
            .flat_map(|g| [g.start_feature, g.num_features, g.start_bin, g.num_bins])
            .collect();

        let native_i64 =
            self.native_i64_atomics.unwrap_or_else(|| supports_native_i64_atomics(self.client));

        let client = self.client.clone();
        let gidx = client.create_from_slice(bytemuck::cast_slice(&matrix.gidx));
        let cut_ptrs = client.create_from_slice(bytemuck::cast_slice(&matrix.cut_ptrs));
        let groups_dev = client.create_from_slice(bytemuck::cast_slice(&groups_flat));
        // 1-word placeholders for whichever histogram view the comptime
        // `native_i64` flag leaves unused.
        let dummy_u32 = client.create_from_slice(bytemuck::cast_slice(&[0u32]));
        let dummy_i64 = client.create_from_slice(bytemuck::cast_slice(&[0i64]));

        Ok(HistogramEngine {
            client,
            gidx,
            cut_ptrs,
            groups_dev,
            dummy_u32,
            dummy_i64,
            native_i64,
            n_groups: groups.len() as u32,
            n_bins: matrix.n_bins() as usize,
            n_rows: matrix.n_rows,
            n_cuts: matrix.cut_ptrs.len(),
            gidx_len: matrix.gidx.len(),
            row_stride: matrix.row_stride as u32,
            base_rowid: matrix.base_rowid,
            null_value: matrix.null_value,
            dense: matrix.is_dense(),
            compressed: matrix.is_compressed(),
            use_shared,
            smem_words,
            max_blocks_per_group: self.max_blocks_per_group,
        })
    }
}

/// Builds gradient histograms against a device-resident ELLPACK matrix; the
/// analogue of `DeviceHistogramBuilder` + `HistKernel::Dispatch`.
///
/// Created by [`HistogramBuilder::build`].
pub struct HistogramEngine<R: Runtime> {
    client: ComputeClient<R>,
    gidx: Handle,
    cut_ptrs: Handle,
    groups_dev: Handle,
    dummy_u32: Handle,
    dummy_i64: Handle,
    native_i64: bool,
    n_groups: u32,
    n_bins: usize,
    n_rows: usize,
    n_cuts: usize,
    gidx_len: usize,
    row_stride: u32,
    base_rowid: u32,
    null_value: u32,
    dense: bool,
    compressed: bool,
    use_shared: bool,
    smem_words: usize,
    max_blocks_per_group: u32,
}

impl<R: Runtime> HistogramEngine<R> {
    pub fn client(&self) -> &ComputeClient<R> {
        &self.client
    }

    pub fn n_bins(&self) -> usize {
        self.n_bins
    }

    /// Whether the shared-memory (privatised histogram) path is active.
    pub fn uses_shared_memory(&self) -> bool {
        self.use_shared
    }

    /// Whether global-memory accumulation uses native 64-bit atomics
    /// (`AtomicAddGpairGlobal`) instead of the u32-carry scheme.
    pub fn uses_native_i64_atomics(&self) -> bool {
        self.native_i64
    }

    /// Upload quantised gradient pairs (one per matrix row).
    pub fn upload_gpairs(&self, gpairs: &[GradientPairInt64]) -> Result<DeviceGpairs> {
        if gpairs.len() != self.n_rows {
            return Err(Error::GpairCount { expected: self.n_rows, got: gpairs.len() });
        }
        Ok(DeviceGpairs {
            handle: self.client.create_from_slice(bytemuck::cast_slice(gpairs)),
            n: gpairs.len(),
        })
    }

    /// Upload the row indices of one tree node.
    pub fn upload_rows(&self, ridx: &[u32]) -> DeviceRows {
        DeviceRows {
            handle: self.client.create_from_slice(bytemuck::cast_slice(ridx)),
            n: ridx.len(),
        }
    }

    /// Launch the histogram kernel, leaving the result on device.
    ///
    /// This is the hot path used per tree node; it performs no host round
    /// trips beyond the launch itself.
    pub fn build_to_device(&self, gpairs: &DeviceGpairs, rows: &DeviceRows) -> DeviceHistogram {
        // Zero-initialised accumulator: 4 u32 words per bin.
        let hist_words = vec![0u32; self.n_bins * 4];
        let hist = self.client.create_from_slice(bytemuck::cast_slice(&hist_words));

        // Grid sizing, mirroring the `launch` lambda in DispatchHistShmem:
        // enough tiles for (rows x features-per-group) items, occupancy-capped.
        let columns_per_group = self.row_stride.div_ceil(self.n_groups);
        let items_per_group = rows.n as u32 * columns_per_group;
        let tile = BLOCK_THREADS * ITEMS_PER_THREAD;
        let n_blocks = items_per_group.div_ceil(tile).clamp(1, self.max_blocks_per_group);

        // Exactly one of the two histogram views is the real buffer; the
        // other is a 1-word dummy never touched by the comptime-elided branch.
        let (hist32, hist32_len, hist64, hist64_len) = if self.native_i64 {
            (self.dummy_u32.clone(), 1, hist.clone(), self.n_bins * 2)
        } else {
            (hist.clone(), hist_words.len(), self.dummy_i64.clone(), 1)
        };

        hist_kernel::launch::<R>(
            &self.client,
            CubeCount::Static(n_blocks, self.n_groups, 1),
            CubeDim::new_1d(BLOCK_THREADS),
            unsafe { ArrayArg::from_raw_parts(self.gidx.clone(), self.gidx_len) },
            unsafe { ArrayArg::from_raw_parts(self.cut_ptrs.clone(), self.n_cuts) },
            unsafe { ArrayArg::from_raw_parts(self.groups_dev.clone(), self.n_groups as usize * 4) },
            unsafe { ArrayArg::from_raw_parts(rows.handle.clone(), rows.n) },
            unsafe { ArrayArg::from_raw_parts(gpairs.handle.clone(), gpairs.n * 2) },
            unsafe { ArrayArg::from_raw_parts(hist32, hist32_len) },
            unsafe { ArrayArg::from_raw_parts(hist64, hist64_len) },
            self.row_stride,
            self.base_rowid,
            self.null_value,
            rows.n as u32,
            self.dense,
            self.compressed,
            self.use_shared,
            self.native_i64,
            self.smem_words,
        );

        DeviceHistogram { handle: hist, n_bins: self.n_bins }
    }

    /// Subtraction trick (`sibling = parent - built`), on device.
    pub fn subtract_to_device(
        &self,
        parent: &DeviceHistogram,
        built: &DeviceHistogram,
    ) -> Result<DeviceHistogram> {
        if parent.n_bins != built.n_bins {
            return Err(Error::HistogramLen { parent: parent.n_bins, built: built.n_bins });
        }
        // Word views: subtraction runs over interleaved i64 pairs.
        let n_words = parent.n_bins * 2;
        let out = self.client.empty(n_words * core::mem::size_of::<i64>());

        subtract_hist_kernel::launch::<R>(
            &self.client,
            CubeCount::Static((n_words as u32).div_ceil(BLOCK_THREADS).max(1), 1, 1),
            CubeDim::new_1d(BLOCK_THREADS),
            unsafe { ArrayArg::from_raw_parts(parent.handle.clone(), n_words) },
            unsafe { ArrayArg::from_raw_parts(built.handle.clone(), n_words) },
            unsafe { ArrayArg::from_raw_parts(out.clone(), n_words) },
            n_words as u32,
        );

        Ok(DeviceHistogram { handle: out, n_bins: parent.n_bins })
    }

    /// Read a histogram back as `i64` gradient pairs.
    ///
    /// Accumulators (from [`Self::build_to_device`]) hold 4 `u32` words per
    /// bin; subtraction outputs hold plain i64 pairs. Both are the same bytes
    /// in little-endian layout, so one reader serves both.
    pub fn read(&self, hist: &DeviceHistogram) -> Result<Vec<GradientPairInt64>> {
        let bytes = self.client.read_one_unchecked(hist.handle.clone());
        let words: &[u32] = bytemuck::cast_slice(&bytes);
        if words.len() != hist.n_bins * 4 {
            return Err(Error::HistogramBins { expected: hist.n_bins, got: words.len() / 4 });
        }
        Ok(words
            .chunks_exact(4)
            .map(|w| GradientPairInt64 {
                grad: (w[0] as i64) | ((w[1] as i64) << 32),
                hess: (w[2] as i64) | ((w[3] as i64) << 32),
            })
            .collect())
    }

    /// Convenience: upload, build, and read back one node histogram.
    pub fn build(
        &self,
        gpairs: &[GradientPairInt64],
        ridx: &[u32],
    ) -> Result<Vec<GradientPairInt64>> {
        let gpairs = self.upload_gpairs(gpairs)?;
        let rows = self.upload_rows(ridx);
        self.read(&self.build_to_device(&gpairs, &rows))
    }

    /// Convenience: subtraction trick over host-side histograms.
    pub fn subtract(
        &self,
        parent: &[GradientPairInt64],
        built: &[GradientPairInt64],
    ) -> Result<Vec<GradientPairInt64>> {
        if parent.len() != built.len() {
            return Err(Error::HistogramLen { parent: parent.len(), built: built.len() });
        }
        let parent_dev = DeviceHistogram {
            handle: self.client.create_from_slice(bytemuck::cast_slice(parent)),
            n_bins: parent.len(),
        };
        let built_dev = DeviceHistogram {
            handle: self.client.create_from_slice(bytemuck::cast_slice(built)),
            n_bins: built.len(),
        };
        let out = self.subtract_to_device(&parent_dev, &built_dev)?;
        self.read(&out)
    }
}
