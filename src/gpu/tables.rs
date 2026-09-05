//! Several small per-launch tables in one upload.
//!
//! A level's launches take a dozen host-built tables between them — the
//! partitioner's tile and segment columns, the histogram's job and cube
//! tables, the evaluator's per-node inputs — and each one used to be its own
//! `create_from_slice`: two host copies, a device allocation and a pageable
//! `memcpy` per table, twenty-odd times a level. Packed into one buffer they
//! are one allocation and one copy, and each table is bound as a sub-range of
//! it, which every backend's bindings can express.

use cubecl::bytes::Bytes;
use cubecl::prelude::*;
use cubecl::server::Handle;

/// Upload an owned vector, staged through the runtime's pinned pool on a
/// device with one.
///
/// Measured on a Kaggle T4: the client's slice route and its plain owned
/// route both moved 100 MB at 0.36 GB/s — a pageable `memcpy` the driver
/// chunks through its own staging — and 4 MB in 5–10 ms. Staged into pinned
/// memory first, the transfer is a DMA at the bus's speed, and the pinned
/// block goes back to the pool for the next upload, so the page-locking is
/// paid once per size. A plane-less runtime has no pool and no bus, and
/// keeps the plain route.
///
/// Not for a large one-off buffer: page-locking is paid per byte the first
/// time a size is seen, and for the 100 MB value matrix that was 460 ms on
/// the same VM — more than the pageable copy it would have saved. Above
/// [`PINNED_LIMIT`] the buffer goes up pageable.
pub fn upload_vec<R: Runtime, T: bytemuck::NoUninit + Send + Sync>(
    client: &ComputeClient<R>,
    data: Vec<T>,
) -> Handle {
    #[cfg(feature = "cuda")]
    if let Some(handle) = cuda_direct::upload(client, bytemuck::cast_slice(&data)) {
        return handle;
    }
    #[cfg(feature = "cuda")]
    if cuda_direct::applies(client, size_of_val(data.as_slice())) {
        // The runtime's write is a stream-ordered `memcpy_htod_async`: the
        // driver stages a pageable source before returning, so the copy
        // neither waits for the queue nor needs the queue drained first.
        return client.create_from_slice(bytemuck::cast_slice(&data));
    }
    let mut bytes = Bytes::from_elems(data);
    if super::launch::has_planes(client) && bytes.len() <= PINNED_LIMIT {
        client.staging(core::iter::once(&mut bytes), false);
    }
    client.create(bytes)
}

/// The CUDA driver's own copy into a CubeCL buffer.
///
/// CubeCL's runtime moves host memory at 0.36 GB/s on a Kaggle T4 by every
/// route it offers, on a bus PyTorch drives at 4.4 GB/s pageable and 12 GB/s
/// pinned (`docs/gpu-benchmarks.md`). The runtime executes its server on the
/// calling thread and binds its CUDA context there for every command, so
/// after a command the context is current here and a synchronous
/// `cuMemcpyHtoD` into the buffer's device pointer is an ordinary driver
/// call. Synchronous, and after a `sync`, because the runtime's streams are
/// non-blocking: the pool may hand out memory a queued kernel is still
/// reading, and the copy must not overtake it.
#[cfg(feature = "cuda")]
pub(crate) mod cuda_direct {
    //! See [`super::upload_vec`].
    use std::any::Any;
    use std::sync::atomic::{AtomicBool, Ordering};

    use cubecl::cuda::CudaRuntime;
    use cubecl::prelude::*;
    use cubecl::server::Handle;

    /// Set once the driver refused, after which the runtime's route is used.
    static FAILED: AtomicBool = AtomicBool::new(false);
    /// The device the client was made for; see [`super::super::default_client`].
    pub static ORDINAL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    /// Make the device's primary context current on this thread if none is.
    ///
    /// The runtime binds its context inside its commands, but the thread was
    /// found without one at the copy (`CUDA_ERROR_INVALID_CONTEXT`, run #11).
    /// The primary context is unique per device, so retaining it here binds
    /// the very context the runtime's allocations live in.
    fn bind_context() -> Result<(), cudarc::driver::DriverError> {
        use cudarc::driver::result::{ctx, device, primary_ctx};
        if ctx::get_current()?.is_some() {
            return Ok(());
        }
        let dev = device::get(ORDINAL.load(Ordering::Relaxed) as i32)?;
        // SAFETY: a valid device; the primary context outlives the process's
        // use of it, and setting it current is what the runtime itself does.
        unsafe {
            let ctx = primary_ctx::retain(dev)?;
            ctx::set_current(ctx)
        }
    }

    /// Smallest upload the direct route takes. Below it the runtime's own
    /// stream-ordered write is the cheaper one: the direct copy has to drain
    /// the queue first (the pool may have handed out memory a queued kernel
    /// still reads) and wait for its DMA after, and a level's tables are a
    /// few kilobytes launched between kernels — two idle gaps per table, four
    /// tables a level, where the write itself is microseconds.
    pub const DIRECT_LIMIT: usize = 1 << 20;

    /// Whether `client` is the CUDA runtime and the direct route is open —
    /// the size decision is the caller's.
    pub fn applies<R: Runtime>(client: &ComputeClient<R>, _bytes: usize) -> bool {
        !FAILED.load(Ordering::Relaxed)
            && (client as &dyn Any).downcast_ref::<ComputeClient<CudaRuntime>>().is_some()
    }

    /// The device's primary context, retained: the driver builds it on the
    /// first call (≈350 ms on a cloud VM) and every later retain — the
    /// runtime's own included — finds it built.
    pub fn retain_primary(ordinal: usize) -> Result<(), cudarc::driver::DriverError> {
        use cudarc::driver::result::{device, init, primary_ctx};
        init()?;
        let dev = device::get(ordinal as i32)?;
        // SAFETY: a valid device; the context is kept for the process.
        unsafe { primary_ctx::retain(dev) }.map(|_| ())
    }

    pub fn upload<R: Runtime>(client: &ComputeClient<R>, bytes: &[u8]) -> Option<Handle> {
        if FAILED.load(Ordering::Relaxed)
            || bytes.len() < DIRECT_LIMIT
            || std::env::var_os("XGB_NO_DIRECT_UPLOAD").is_some()
        {
            return None;
        }
        let client = (client as &dyn Any).downcast_ref::<ComputeClient<CudaRuntime>>()?;
        let handle = client.empty(bytes.len().max(1));
        let resource = client.get_resource(handle.clone()).ok()?;
        // Drains the queue: the pool may have handed out memory a queued
        // kernel still reads.
        if cubecl::future::block_on(client.sync()).is_err() {
            return None;
        }
        if let Err(e) = bind_context() {
            eprintln!("xgboost_rs: direct CUDA upload unavailable (no context: {e}); using the runtime's");
            FAILED.store(true, Ordering::Relaxed);
            return None;
        }
        let ptr = resource.resource().ptr;
        // SAFETY: `ptr` is the device address of a live allocation of at
        // least `bytes.len()` bytes, kept alive by `resource`; the queue is
        // drained; the copy is synchronous.
        let result = unsafe { cudarc::driver::result::memcpy_htod_sync(ptr, bytes) }
            // `cuMemcpyHtoD` from pageable memory returns once the bytes are
            // in the driver's staging buffer, not once the DMA has landed,
            // and the runtime's stream is non-blocking (it does not wait on
            // the null stream), so the next kernel could read the table
            // before the copy finished — which it did, once every few fits
            // (run #23: 3 of 8 identical fits ended at a different model).
            .and_then(|()| cudarc::driver::result::ctx::synchronize());
        match result {
            Ok(()) => Some(handle),
            Err(e) => {
                eprintln!("xgboost_rs: direct CUDA upload unavailable ({e}); using the runtime's");
                FAILED.store(true, Ordering::Relaxed);
                None
            }
        }
    }
}

/// Largest upload staged through pinned memory; see [`upload_vec`].
pub const PINNED_LIMIT: usize = 16 << 20;

/// [`upload_vec`] from a borrowed slice: the direct route needs no owned
/// buffer, so the caller's is copied to the device as it is, and only the
/// runtime's route pays for a copy of it.
pub fn upload_slice<R: Runtime, T: bytemuck::NoUninit + bytemuck::AnyBitPattern + Send + Sync>(
    client: &ComputeClient<R>,
    data: &[T],
) -> Handle {
    #[cfg(feature = "cuda")]
    if let Some(handle) = cuda_direct::upload(client, bytemuck::cast_slice(data)) {
        return handle;
    }
    #[cfg(feature = "cuda")]
    if cuda_direct::applies(client, size_of_val(data)) {
        return client.create_from_slice(bytemuck::cast_slice(data));
    }
    upload_vec(client, data.to_vec())
}

/// Alignment of every table within the buffer: enough for any element type
/// and for a backend's stricter sub-range alignment.
const ALIGN: usize = 256;

/// Tables being packed, before the upload.
#[derive(Default)]
pub struct TableBuilder {
    bytes: Vec<u8>,
    ranges: Vec<(usize, usize)>,
}

impl TableBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add `data`; the index names it to [`Tables::arg`]. An empty table is
    /// stored as one zero element, so that its binding is never empty.
    pub fn push<T: bytemuck::Pod>(&mut self, data: &[T]) -> usize {
        let start = self.bytes.len().div_ceil(ALIGN) * ALIGN;
        self.bytes.resize(start, 0);
        if data.is_empty() {
            self.bytes.extend(core::iter::repeat_n(0u8, size_of::<T>().max(1)));
        } else {
            self.bytes.extend_from_slice(bytemuck::cast_slice(data));
        }
        self.ranges.push((start, self.bytes.len()));
        self.ranges.len() - 1
    }

    /// Upload as one buffer.
    pub fn upload<R: Runtime>(self, client: &ComputeClient<R>) -> Tables {
        let ranges = self.ranges;
        let total = self.bytes.len();
        let handle = upload_vec(client, self.bytes);
        Tables { handle, ranges, total }
    }
}

/// The packed tables on the device.
pub struct Tables {
    handle: Handle,
    ranges: Vec<(usize, usize)>,
    total: usize,
}

impl Tables {
    /// Table `i` as an array argument of `len` elements.
    ///
    /// # Safety
    ///
    /// As `ArrayArg::from_raw_parts`: `len` is what the kernel bounds its
    /// accesses by, so it must not exceed the elements pushed for table `i`.
    pub unsafe fn arg<R: Runtime>(&self, i: usize, len: usize) -> ArrayArg<R> {
        let (start, end) = self.ranges[i];
        // `offset_end` trims from the end of the buffer, not an absolute end.
        let handle = self
            .handle
            .clone()
            .offset_start(start as u64)
            .offset_end((self.total - end) as u64);
        // SAFETY: the caller's contract, restated above.
        unsafe { ArrayArg::from_raw_parts(handle, len) }
    }
}
