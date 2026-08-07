# What the GPU path still does not do

`device=cuda` trains and matches the CPU fit bit-for-bit on a single round
(`tests/gpu_training.rs`). One thing is still absent, and it is out of scope. It
is refused by name at the API boundary rather than silently mis-fitted, so
nothing here is a correctness risk — only missing capability.

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

## 2. `multi_strategy=multi_output_tree` (implemented)

A vector leaf grows one histogram per target. The device histogram kernel stays
scalar and is simply run once per target, over the same row sets and into the
target's own slice of the node's frontier slot — the shape `HistGrower` already
used on the CPU, where `build_hists` runs its scalar kernel once per gradient
column. The target dimension this doc used to ask the kernel for turned out to
be unnecessary; only the frontier layout learned about targets
(`GpuHistGrower::frontier_bins`, `src/gpu/grower.rs`).

The split search is where the shape genuinely changes, because one candidate is
now scored from `n_targets` running sums at once:

- The scalar evaluator accumulates as it scans, so its running sum *is* the
  candidate. Holding `n_targets` of those in shared memory would blow the 16 KiB
  workgroup budget the portable backends allow, so the scan is split out:
  `prefix_scan_kernel` materialises the inclusive prefix sums once per
  `(node, feature, target)` and `evaluate_feature_multi_kernel` reads whichever
  sums a candidate bin needs. Both are exact `i64`, so — as for the scalar
  path — the result does not depend on how the work was split across threads.
  The backward scan needs no second pass: the suffix sum at bin `i` is
  `total - prefix[i - 1]` in fixed point.
- The gain is a line-for-line port of `HistGrower::multi_split_gain`: each
  target's own regularised gain at its own bounded weight, summed in `f32` in
  target order, rejected as a whole when the children's *mean* hessian fails
  `IsValidSplit`. The monotone direction check is absent for the same reason it
  is a no-op on the CPU — it is applied to mean-hessian children whose gradient
  sum is zero, so both weights come out equal. The bounds still shape the gain,
  through each target's real sums.
- A candidate carries one `(grad, hess)` pair summed over targets, as
  `SplitEntry` does on the CPU, so the winner's per-target child sums are read
  back out of the same prefix sums by `multi_child_sums_kernel` — the device
  counterpart of `HistGrower::multi_child_sums`, down to how a `-1` condition
  (a threshold matching no cut) is interpreted.

One thing the CPU does not have to worry about: the fixed point has to stay
addable *across* targets, because the node's cover and a candidate's summed
child sums add the targets together in `i64` and each target alone already fills
the 62 bits below the sign. `GradientQuantiser::new_multi` therefore bounds by
the sum of the per-target bounds rather than the largest of them, at a cost of
`log2(n_targets)` bits. Bounding by the largest overflows from three targets up,
which is what the `matches_across_depth_and_grow_policy` case caught.

Categorical features are refused, on the GPU exactly as on the CPU: a vector
leaf has no categorical split there either.

## 3. SYCL (refused, and out of scope)

`Device::Sycl` is rejected outright. CubeCL has no SYCL backend here, so there
is nothing to run it on.

## Not a feature gap, but outstanding

`docs/gpu-benchmarks.md` records that the device path is 5–14× slower than
XGBoost's `gpu_hist`, with the measured evidence for where the time goes.
