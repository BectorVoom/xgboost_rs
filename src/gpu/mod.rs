//! GPU kernels for gradient-boosted tree training, ported from
//! `xgboost/src/tree/gpu_hist/*.cu` to CubeCL.

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

/// The CubeCL runtime a GPU fit runs on.
///
/// CUDA when the crate is built with the `cuda` feature — what a real GPU fit
/// wants — and wgpu/Vulkan otherwise, which is what makes the kernels testable
/// on any machine, including a CPU Vulkan implementation like lavapipe.
#[cfg(feature = "cuda")]
pub type DefaultRuntime = cubecl::cuda::CudaRuntime;
#[cfg(not(feature = "cuda"))]
pub type DefaultRuntime = cubecl::wgpu::WgpuRuntime;

/// A compute client for the requested device ordinal.
///
/// The ordinal selects a CUDA device; the wgpu runtime has no equivalent
/// notion here and always takes the default adapter.
pub fn default_client(ordinal: usize) -> cubecl::prelude::ComputeClient<DefaultRuntime> {
    use cubecl::prelude::Runtime;
    #[cfg(feature = "cuda")]
    {
        DefaultRuntime::client(&cubecl::cuda::CudaDevice::new(ordinal))
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = ordinal;
        DefaultRuntime::client(&cubecl::wgpu::WgpuDevice::default())
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
