//! Device kernels for gradient-boosted tree training, ported from
//! `xgboost/src/tree/gpu_hist/*.cu` to CubeCL.
//!
//! # One kernel body, every runtime
//!
//! Every kernel here is generic over `R: Runtime` and every launch takes its
//! geometry from [`launch`], so the same source runs unchanged on CUDA, on
//! wgpu (Vulkan or Metal), and on the CubeCL CPU runtime. Which one a build
//! gets is the ladder in [`DefaultRuntime`]; [`BACKEND`] names it at run time,
//! so a fit never has to guess what it just ran on.
//!
//! They do not execute a kernel the same way, and two differences reach into
//! the kernel bodies rather than staying in the launcher:
//!
//! * **Cubes are not concurrent on every runtime.** On a GPU the hardware gives
//!   each resident workgroup its own slice of on-chip shared storage. The CPU
//!   runtime instead compiles the kernel body inside `for` loops over
//!   `CUBE_POS_{Z,Y,X}`, runs one OS thread per *unit*, and allocates
//!   `SharedMemory` **once for the whole launch** — so every cube reuses the
//!   same buffer. A unit that finishes cube `i` and loops on to cube `i + 1`
//!   will overwrite shared memory that a slower unit is still reading. Every
//!   kernel below that declares `SharedMemory` therefore ends with a trailing
//!   [`sync_cube`], which pins all units to the current cube before any of them
//!   advances. It costs one barrier per cube on a GPU, where it is redundant,
//!   and it is what makes the CPU runtime give the same answer at all.
//!
//! * **A "unit" is an OS thread when the runtime has no planes.** 256 units per
//!   cube is eight warps on NVIDIA and 256 spinning threads on the CPU runtime,
//!   and a `sync_cube` there is a spin barrier that costs a scheduler round
//!   once the cube is as wide as the machine. So every kernel that cooperates
//!   through shared memory has a second, *serial* shape behind a comptime
//!   `coop` flag ([`launch::cooperative`]): one unit owns a whole work item —
//!   a `(node, feature)` pair, a tile of rows, a row chunk of a node — and
//!   nothing it writes is another unit's to read, so there is no barrier and
//!   the launch can be as wide as the core count. The two shapes agree bit for
//!   bit because every accumulator is exact `i64` and every winner is chosen
//!   by a total order, so no result depends on how the work was divided.
//!
//! # Unchecked launches
//!
//! The training kernels are `#[cube(launch_unchecked)]`. CubeCL's checked mode
//! wraps every array read and write in a bounds test against the binding's
//! length — a clamp on reads, a skip on writes — and on the CPU runtime, whose
//! JIT does not optimise, that is a compare and a branch on every access of
//! every inner loop; measured on the histogram kernel it is 2.2×. The kernels
//! do not need it: each one derives its indices from the lengths and tables it
//! is handed and guards the one index that can run past the end (a
//! flattened unit or cube index against the work-item count), and the
//! oracle tests pin every path against a host reference. The contract that
//! makes a launch sound is therefore stated per kernel, in its own bounds
//! guards, and the `SAFETY` note at each launch site points here.
//!
//! [`sync_cube`]: fn@cubecl::prelude::sync_cube

// A build with the kernels but no backend has nothing to compile them for.
// Saying so here beats several hundred lines of "cannot find type
// `DefaultRuntime`" from every module downstream.
#[cfg(not(any(feature = "cpu", feature = "vulkan", feature = "metal", feature = "cuda")))]
compile_error!(
    "feature `gpu` needs a backend: enable one of `cpu`, `metal`, `vulkan` or \
     `cuda` (the default feature set is `gpu` + `cpu`)"
);

mod categorical;
pub mod ellpack;
pub mod evaluate_splits;
pub mod grower;
pub mod histogram;
pub mod launch;
pub mod linear;
pub mod quantiser;
pub mod row_partitioner;

use cubecl::server::Handle;

/// The CubeCL runtime this build's kernels run on.
///
/// A ladder over the backend features, most specialised first, so that adding a
/// backend to a build can only ever upgrade what it runs on:
///
/// | features enabled | `DefaultRuntime` | [`BACKEND`] |
/// | --- | --- | --- |
/// | `cuda` (with or without the others) | `CudaRuntime` | `"cuda"` |
/// | `vulkan`, no `cuda` | `WgpuRuntime` | `"vulkan"` |
/// | `metal`, no `vulkan`/`cuda` | `WgpuRuntime` | `"metal"` |
/// | `cpu` only (the default) | `CpuRuntime` | `"cpu"` |
///
/// `vulkan` and `metal` are the same runtime — wgpu — differing only in which
/// shader compiler is built in, which wgpu then picks from the adapter. They
/// are separate features because their build requirements are not the same:
/// SPIR-V passthrough needs the Vulkan SDK installed on macOS, MSL needs
/// nothing.
///
/// The bottom rung is what makes the kernels buildable and testable on a
/// machine with no GPU toolchain at all: `cubecl/cpu` is a pure-Rust MLIR JIT
/// with no system dependency, where `cargo test` exercises the same kernel
/// source a CUDA fit runs. It is the slow rung, not a second implementation.
#[cfg(feature = "cuda")]
pub type DefaultRuntime = cubecl::cuda::CudaRuntime;
#[cfg(all(any(feature = "vulkan", feature = "metal"), not(feature = "cuda")))]
pub type DefaultRuntime = cubecl::wgpu::WgpuRuntime;
#[cfg(all(
    feature = "cpu",
    not(feature = "vulkan"),
    not(feature = "metal"),
    not(feature = "cuda")
))]
pub type DefaultRuntime = cubecl::cpu::CpuRuntime;

/// The name of the backend [`DefaultRuntime`] resolved to.
///
/// A fit that asked for `device=cuda` on a build without the `cuda` feature is
/// running its kernels somewhere else, and this is what lets it say so rather
/// than quietly imply an NVIDIA GPU was involved.
pub const BACKEND: &str = if cfg!(feature = "cuda") {
    "cuda"
} else if cfg!(feature = "vulkan") {
    "vulkan"
} else if cfg!(feature = "metal") {
    "metal"
} else {
    "cpu"
};

/// Whether the runtime can express `f64` at all.
///
/// False on Metal, and not as an oversight: Metal Shading Language has no
/// `double`, so `cubecl-cpp`'s Metal dialect emits `#error type double not
/// supported!` where one is asked for and the shader fails to build. The
/// backend says so up front — `register_types` in `cubecl-wgpu`'s Metal backend
/// registers `I64` but not `F64` — which is what lets this be a refusal at
/// construction rather than a wgpu validation panic several launches later.
///
/// It matters because the `f64` in these kernels is not incidental. The split
/// gain arithmetic (`consider_split` and everything under it) and the
/// quantiser's fixed-point conversion are ports of XGBoost's `double`
/// arithmetic, and it is what makes the device fit agree with the CPU fit to
/// the 1e-5 the oracle tests demand. Demoting them to `f32` would be a
/// different model, not the same one computed differently.
pub fn supports_f64<R: cubecl::prelude::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
) -> bool {
    use cubecl::ir::{ElemType, FloatKind, StorageType};
    client.properties().supports_type(StorageType::Scalar(ElemType::Float(FloatKind::F64)))
}

/// A compute client for the requested device ordinal.
///
/// The ordinal selects a CUDA device; neither the wgpu runtime nor the CPU
/// runtime has an equivalent notion here, and both take their default device.
pub fn default_client(ordinal: usize) -> cubecl::prelude::ComputeClient<DefaultRuntime> {
    use cubecl::prelude::Runtime;
    #[cfg(feature = "cuda")]
    {
        DefaultRuntime::client(&cubecl::cuda::CudaDevice::new(ordinal))
    }
    #[cfg(all(any(feature = "vulkan", feature = "metal"), not(feature = "cuda")))]
    {
        let _ = ordinal;
        DefaultRuntime::client(&cubecl::wgpu::WgpuDevice::default())
    }
    #[cfg(all(
        feature = "cpu",
        not(feature = "vulkan"),
        not(feature = "metal"),
        not(feature = "cuda")
    ))]
    {
        let _ = ordinal;
        DefaultRuntime::client(&cubecl::cpu::CpuDevice)
    }
}

/// Single-precision gradient pair, mirrors `xgboost::GradientPair`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GradientPair {
    pub grad: f32,
    pub hess: f32,
}

/// Double-precision gradient pair, mirrors `xgboost::GradientPairPrecise`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GradientPairPrecise {
    pub grad: f64,
    pub hess: f64,
}

/// Fixed-point gradient pair, mirrors `xgboost::GradientPairInt64`.
///
/// Layout matches the CUDA type: two little-endian `i64` words, so a device
/// buffer of interleaved `[grad, hess]` `i64` values can be reinterpreted as a
/// slice of this struct.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct GradientPairInt64 {
    pub grad: i64,
    pub hess: i64,
}

impl core::ops::Add for GradientPairInt64 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self { grad: self.grad + rhs.grad, hess: self.hess + rhs.hess }
    }
}

impl core::ops::Sub for GradientPairInt64 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self { grad: self.grad - rhs.grad, hess: self.hess - rhs.hess }
    }
}

/// Device buffer of quantised gradient pairs (interleaved `[grad, hess]` i64).
#[derive(Clone, Debug)]
pub struct DeviceGpairs {
    pub(crate) handle: Handle,
    pub(crate) n: usize,
}

impl DeviceGpairs {
    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
}

/// Device buffer of row indices belonging to one tree node.
#[derive(Clone, Debug)]
pub struct DeviceRows {
    pub(crate) handle: Handle,
    /// First index of this node's slice of the buffer. The partitioner keeps
    /// every node's rows in one `ridx` allocation, so a node is a range of it
    /// rather than a buffer of its own.
    pub(crate) base: usize,
    pub(crate) n: usize,
}

impl DeviceRows {
    /// A node's slice of a partitioned row index.
    pub fn slice(handle: Handle, base: usize, n: usize) -> Self {
        Self { handle, base, n }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
}

/// Device-resident histogram: 4 `u32` accumulator words per bin
/// (see `gpu::histogram` module docs).
#[derive(Clone, Debug)]
pub struct DeviceHistogram {
    pub(crate) handle: Handle,
    pub(crate) n_bins: usize,
}

impl DeviceHistogram {
    pub fn n_bins(&self) -> usize {
        self.n_bins
    }
}
