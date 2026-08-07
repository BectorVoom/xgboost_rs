# What the GPU path still does not do

`device=cuda` trains and matches the CPU fit bit-for-bit on a single round
(`tests/gpu_training.rs`). Two things are still refused or absent. Each is
refused by name at the API boundary rather than silently mis-fitted, so nothing
here is a correctness risk — only missing capability.

## 1. Categorical splits (implemented)

The device evaluator scans a feature's bins in ascending order, which is
meaningless for category codes, so a categorical feature is masked out of
`evaluate_feature_kernel` (`GpuHistGrower::evaluate`, `src/gpu/grower.rs`) the
same way column sampling masks out a feature the node may not use. It is
scored instead by `src/gpu/categorical.rs`, the host-side readback route this
doc used to describe as the cheap alternative to a device-side sort:

- The node's histogram is read back once per batch — only when the batch has
  at least one categorical candidate — and decoded into the same
  `GradientPairInt64` pairs `HistogramEngine::read` produces.
- `categorical::enumerate` dispatches `common::UseOneHot` exactly as
  `HistGrower::evaluate_one` does (`n_bins_feature < max_cat_to_onehot`,
  default 4) to `enumerate_one_hot` or `enumerate_partition` — line-for-line
  ports of the CPU functions of the same name (`src/tree/hist.rs`), except the
  running sums stay in the histogram's native quantised `i64` form and are
  decoded only where the gain formula needs a float, matching how the numeric
  kernel path is already exact.
- The winning categorical candidate is merged against the device's
  numeric-only winner with the same tie-break `SplitEntry::need_replace` uses
  (`categorical::replaces`), then carried through `ExpandEntry::cat_bits` to
  `RegTree::expand_categorical` and to the row partitioner's `SegmentSplit`.

The row partitioner already handled categorical splits before this
(`goes_left` tests the bit set, `tests/gpu_row_partition.rs` covers it); the
gap was only the evaluator and the bit-set plumbing feeding it, both now
covered by `tests/gpu_training.rs`'s `categorical_splits_match_the_cpu_fit_*`
tests, which check bit-for-bit parity with the CPU fit the same way every
other GPU training test does.

## 2. `multi_strategy=multi_output_tree` (refused)

Refused for every path except CPU `hist`, GPU included. A vector leaf grows one
histogram per target; the device histogram kernel is scalar, so it would need a
target dimension — the same shape the node dimension took in `27ab94d`.

## 3. SYCL (refused, and out of scope)

`Device::Sycl` is rejected outright. CubeCL has no SYCL backend here, so there
is nothing to run it on.

## Not a feature gap, but outstanding

`docs/gpu-benchmarks.md` records that the device path is 5–14× slower than
XGBoost's `gpu_hist`, with the measured evidence for where the time goes.
