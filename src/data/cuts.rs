//! Quantile cuts — a port of XGBoost's weighted-quantile sketch.
//!
//! This is a faithful port of `WQSummary` + `QuantileSketchTemplate` +
//! `HostSketchContainer` (xgboost 3.0.5, `src/common/quantile.{h,cc}`). The cut
//! points it produces are the single largest divergence risk against the
//! reference implementation, so the structure is kept deliberately close to the
//! original — same pruning arithmetic, same `f32` rank type, same push order —
//! and is asserted bit-for-bit against `DMatrix.get_quantile_cut()` fixtures in
//! `tests/oracle.rs`.

use super::DMatrix;
use crate::Result;
use rayon::prelude::*;
use std::cmp::Ordering;

/// Sketch capacity factor: `WQSketch::kFactor` upstream.
const K_FACTOR: f32 = 8.0;

/// One summary element: a value with its rank interval `[rmin, rmax]`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Entry {
    rmin: f32,
    rmax: f32,
    wmin: f32,
    value: f32,
}

impl Entry {
    /// Rank lower bound for values strictly greater than `value`.
    #[inline]
    fn rmin_next(&self) -> f32 {
        self.rmin + self.wmin
    }

    /// Rank upper bound for values strictly smaller than `value`.
    #[inline]
    fn rmax_prev(&self) -> f32 {
        self.rmax - self.wmin
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct QEntry {
    value: f32,
    weight: f32,
}

/// Order-preserving `f32` -> `u32` mapping, so floats can be radix sorted.
///
/// Flipping the sign bit orders the positives above the negatives; inverting
/// negatives reverses their descending bit pattern.
#[inline]
fn float_key(f: f32) -> u32 {
    let b = f.to_bits();
    if (b as i32) < 0 { !b } else { b ^ 0x8000_0000 }
}

/// Sort the staging queue by value.
///
/// The queue holds tens of thousands of entries and is re-sorted every time it
/// fills, which makes it the dominant cost of sketching. A least-significant
/// digit radix sort beats the comparison sort here; passes whose digit is
/// constant across the queue are skipped, which is common when a feature's
/// values share an exponent range.
fn sort_queue(queue: &mut [QEntry], scratch: &mut Vec<QEntry>) {
    let n = queue.len();
    // Radix bookkeeping does not pay for itself on short queues.
    if n < 512 {
        queue.sort_unstable_by(|a, b| a.value.partial_cmp(&b.value).unwrap_or(Ordering::Equal));
        return;
    }

    let mut counts = [[0u32; 256]; 4];
    for e in queue.iter() {
        let k = float_key(e.value);
        for (p, c) in counts.iter_mut().enumerate() {
            c[((k >> (8 * p)) & 0xff) as usize] += 1;
        }
    }

    scratch.clear();
    scratch.resize(n, QEntry::default());
    let mut in_scratch = false;
    for p in 0..4 {
        // Every entry shares this digit: the pass would be a plain copy.
        if counts[p].iter().any(|&c| c as usize == n) {
            continue;
        }
        let mut offsets = [0u32; 256];
        let mut acc = 0u32;
        for (o, c) in offsets.iter_mut().zip(&counts[p]) {
            *o = acc;
            acc += c;
        }
        let (src, dst): (&[QEntry], &mut [QEntry]) = if in_scratch {
            (scratch, queue)
        } else {
            (queue, scratch)
        };
        for e in src {
            let bucket = ((float_key(e.value) >> (8 * p)) & 0xff) as usize;
            dst[offsets[bucket] as usize] = *e;
            offsets[bucket] += 1;
        }
        in_scratch = !in_scratch;
    }
    if in_scratch {
        queue.copy_from_slice(scratch);
    }
}

/// `WQSummary::Queue::MakeSummary` — sort the staging queue and fold equal
/// values into one entry each.
fn make_summary(queue: &mut [QEntry], scratch: &mut Vec<QEntry>, out: &mut Vec<Entry>) {
    sort_queue(queue, scratch);
    out.clear();
    let mut wsum = 0.0f32;
    let mut i = 0usize;
    while i < queue.len() {
        let mut j = i + 1;
        let mut w = queue[i].weight;
        while j < queue.len() && queue[j].value == queue[i].value {
            w += queue[j].weight;
            j += 1;
        }
        out.push(Entry { rmin: wsum, rmax: wsum + w, wmin: w, value: queue[i].value });
        wsum += w;
        i = j;
    }
}

/// `WQSummary::SetPrune` — keep at most `maxsize` entries, spaced evenly in
/// rank space.
fn set_prune(src: &[Entry], maxsize: usize, out: &mut Vec<Entry>) {
    out.clear();
    if src.len() <= maxsize {
        out.extend_from_slice(src);
        return;
    }
    let begin = src[0].rmax;
    let range = src[src.len() - 1].rmin - src[0].rmax;
    let n = maxsize - 1;
    out.push(src[0]);
    // `lastidx` avoids emitting the same source entry twice.
    let mut i = 1usize;
    let mut lastidx = 0usize;
    for k in 1..n {
        let dx2 = 2.0 * ((k as f32 * range) / n as f32 + begin);
        while i < src.len() - 1 && dx2 >= src[i + 1].rmax + src[i + 1].rmin {
            i += 1;
        }
        if i == src.len() - 1 {
            break;
        }
        if dx2 < src[i].rmin_next() + src[i + 1].rmax_prev() {
            if i != lastidx {
                out.push(src[i]);
                lastidx = i;
            }
        } else if i + 1 != lastidx {
            out.push(src[i + 1]);
            lastidx = i + 1;
        }
    }
    if lastidx != src.len() - 1 {
        out.push(src[src.len() - 1]);
    }
}

/// `WQSummary::SetCombine` — merge two summaries, then repair rank ordering
/// that rounding may have broken.
fn set_combine(sa: &[Entry], sb: &[Entry], out: &mut Vec<Entry>) {
    out.clear();
    if sa.is_empty() {
        out.extend_from_slice(sb);
        return;
    }
    if sb.is_empty() {
        out.extend_from_slice(sa);
        return;
    }
    let (mut ai, mut bi) = (0usize, 0usize);
    let (mut aprev_rmin, mut bprev_rmin) = (0.0f32, 0.0f32);
    while ai < sa.len() && bi < sb.len() {
        let (a, b) = (sa[ai], sb[bi]);
        if a.value == b.value {
            out.push(Entry {
                rmin: a.rmin + b.rmin,
                rmax: a.rmax + b.rmax,
                wmin: a.wmin + b.wmin,
                value: a.value,
            });
            aprev_rmin = a.rmin_next();
            bprev_rmin = b.rmin_next();
            ai += 1;
            bi += 1;
        } else if a.value < b.value {
            out.push(Entry {
                rmin: a.rmin + bprev_rmin,
                rmax: a.rmax + b.rmax_prev(),
                wmin: a.wmin,
                value: a.value,
            });
            aprev_rmin = a.rmin_next();
            ai += 1;
        } else {
            out.push(Entry {
                rmin: b.rmin + aprev_rmin,
                rmax: b.rmax + a.rmax_prev(),
                wmin: b.wmin,
                value: b.value,
            });
            bprev_rmin = b.rmin_next();
            bi += 1;
        }
    }
    if ai < sa.len() {
        let brmax = sb[sb.len() - 1].rmax;
        while ai < sa.len() {
            let a = sa[ai];
            out.push(Entry { rmin: a.rmin + bprev_rmin, rmax: a.rmax + brmax, ..a });
            ai += 1;
        }
    }
    if bi < sb.len() {
        let armax = sa[sa.len() - 1].rmax;
        while bi < sb.len() {
            let b = sb[bi];
            out.push(Entry { rmin: b.rmin + aprev_rmin, rmax: b.rmax + armax, ..b });
            bi += 1;
        }
    }
    fix_error(out);
}

/// `WQSummary::FixError` — re-establish monotone ranks after rounding.
fn fix_error(data: &mut [Entry]) {
    let (mut prev_rmin, mut prev_rmax) = (0.0f32, 0.0f32);
    for e in data.iter_mut() {
        if e.rmin < prev_rmin {
            e.rmin = prev_rmin;
        } else {
            prev_rmin = e.rmin;
        }
        if e.rmax < prev_rmax {
            e.rmax = prev_rmax;
        }
        let rmin_next = e.rmin_next();
        if e.rmax < rmin_next {
            e.rmax = rmin_next;
        }
        prev_rmax = e.rmax;
    }
}

/// `QuantileSketchTemplate::LimitSizeLevel`.
fn limit_size_level(maxn: usize, eps: f64) -> (usize, usize) {
    let mut nlevel = 1usize;
    loop {
        let limit_size = ((nlevel as f64 / eps).ceil() as usize + 1).min(maxn);
        let n = 1usize.checked_shl(nlevel as u32).unwrap_or(usize::MAX);
        if n.saturating_mul(limit_size) >= maxn {
            return (nlevel, limit_size);
        }
        nlevel += 1;
    }
}

/// A single feature's sketch: `WQuantileSketch<float, float>`.
#[derive(Debug, Default)]
struct QuantileSketch {
    queue: Vec<QEntry>,
    qtail: usize,
    limit_size: usize,
    /// `level[0]` is scratch space, matching upstream.
    level: Vec<Vec<Entry>>,
    temp: Vec<Entry>,
    /// Double buffer for the radix sort of `queue`.
    sort_scratch: Vec<QEntry>,
}

impl QuantileSketch {
    fn init(&mut self, maxn: usize, eps: f64) {
        let (_nlevel, limit_size) = limit_size_level(maxn, eps);
        self.limit_size = limit_size;
        // `HostSketchContainer` immediately grows the queue to its full size,
        // so the lazy one-element path upstream never triggers here.
        self.queue = vec![QEntry::default(); limit_size * 2];
        self.qtail = 0;
        self.level.clear();
        self.temp.clear();
    }

    #[inline]
    fn push(&mut self, x: f32, w: f32) {
        if w == 0.0 {
            return;
        }
        if self.qtail == self.queue.len() && self.queue[self.qtail - 1].value != x {
            if self.queue.len() == 1 {
                self.queue.resize(self.limit_size * 2, QEntry::default());
            } else {
                let mut temp = std::mem::take(&mut self.temp);
                let mut scratch = std::mem::take(&mut self.sort_scratch);
                make_summary(&mut self.queue[..self.qtail], &mut scratch, &mut temp);
                self.sort_scratch = scratch;
                self.temp = temp;
                self.qtail = 0;
                self.push_temp();
            }
        }
        if self.qtail == 0 || self.queue[self.qtail - 1].value != x {
            self.queue[self.qtail] = QEntry { value: x, weight: w };
            self.qtail += 1;
        } else {
            self.queue[self.qtail - 1].weight += w;
        }
    }

    fn init_level(&mut self, nlevel: usize) {
        while self.level.len() < nlevel {
            self.level.push(Vec::new());
        }
    }

    /// `QuantileSketchTemplate::PushTemp` — cascade `temp` up the level stack.
    fn push_temp(&mut self) {
        let limit_size = self.limit_size;
        let mut l = 1usize;
        loop {
            self.init_level(l + 1);
            if self.level[l].is_empty() {
                let mut dst = std::mem::take(&mut self.level[l]);
                set_prune(&self.temp, limit_size, &mut dst);
                self.level[l] = dst;
                return;
            }
            let mut scratch = std::mem::take(&mut self.level[0]);
            set_prune(&self.temp, limit_size, &mut scratch);
            self.level[0] = scratch;

            let mut temp = std::mem::take(&mut self.temp);
            set_combine(&self.level[0], &self.level[l], &mut temp);
            self.temp = temp;

            if self.temp.len() > limit_size {
                self.level[l].clear();
            } else {
                self.level[l].clear();
                self.level[l].extend_from_slice(&self.temp);
                return;
            }
            l += 1;
        }
    }

    /// `QuantileSketchTemplate::GetSummary`.
    fn get_summary(&mut self, out: &mut Vec<Entry>) {
        let mut scratch = std::mem::take(&mut self.sort_scratch);
        make_summary(&mut self.queue[..self.qtail], &mut scratch, out);
        self.sort_scratch = scratch;
        if self.level.is_empty() {
            if out.len() > self.limit_size {
                let mut temp = std::mem::take(&mut self.temp);
                set_prune(out, self.limit_size, &mut temp);
                out.clear();
                out.extend_from_slice(&temp);
                self.temp = temp;
            }
            return;
        }

        let mut scratch = std::mem::take(&mut self.level[0]);
        set_prune(out, self.limit_size, &mut scratch);
        self.level[0] = scratch;

        for l in 1..self.level.len() {
            if self.level[l].is_empty() {
                continue;
            }
            if self.level[0].is_empty() {
                let src = std::mem::take(&mut self.level[l]);
                self.level[0].extend_from_slice(&src);
                self.level[l] = src;
            } else {
                set_combine(&self.level[0], &self.level[l], out);
                let mut scratch = std::mem::take(&mut self.level[0]);
                set_prune(out, self.limit_size, &mut scratch);
                self.level[0] = scratch;
            }
        }
        out.clear();
        out.extend_from_slice(&self.level[0]);
    }
}

/// Per-feature bin boundaries, the analogue of `common::HistogramCuts`.
///
/// `cut_values[cut_ptrs[f]..cut_ptrs[f + 1]]` are feature `f`'s upper bin
/// bounds; `min_values[f]` is a value below every observed one, used as the
/// split point when the backward scan splits on the first bin.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HistogramCuts {
    pub cut_ptrs: Vec<u32>,
    pub cut_values: Vec<f32>,
    pub min_values: Vec<f32>,
}

impl HistogramCuts {
    pub fn num_features(&self) -> usize {
        self.cut_ptrs.len().saturating_sub(1)
    }

    /// Total bins across all features.
    pub fn total_bins(&self) -> usize {
        self.cut_values.len()
    }

    pub fn feature_bins(&self, fidx: usize) -> usize {
        (self.cut_ptrs[fidx + 1] - self.cut_ptrs[fidx]) as usize
    }

    /// Global bin index for `value` in feature `fidx`: the first cut strictly
    /// greater than `value`, clamped to the feature's last bin.
    ///
    /// The search is branchless. Binning is a hot setup loop — one call per
    /// stored value — and a mispredicting binary search over a couple of
    /// hundred cuts costs far more than the compares themselves.
    #[inline]
    pub fn search_bin(&self, value: f32, fidx: usize) -> u32 {
        let beg = self.cut_ptrs[fidx] as usize;
        let end = self.cut_ptrs[fidx + 1] as usize;
        let vals = &self.cut_values[beg..end];

        // Invariant: the answer lies in `vals[base..base + n]`.
        let mut base = 0usize;
        let mut n = vals.len();
        while n > 1 {
            let half = n / 2;
            // Conditional move, not a branch: pick the upper half when its
            // first element is still <= value.
            base = if vals[base + half] <= value { base + half } else { base };
            n -= half;
        }
        let lo = base + usize::from(vals[base] <= value);

        let idx = beg + lo;
        // A value above every cut belongs to the last bin.
        (if idx == end { idx - 1 } else { idx }) as u32
    }

    /// Split point used by the backward scan when splitting at global bin
    /// `bin_idx` of feature `fidx`.
    #[inline]
    pub fn backward_split_point(&self, fidx: usize, bin_idx: usize) -> f32 {
        if bin_idx == self.cut_ptrs[fidx] as usize {
            self.min_values[fidx]
        } else {
            self.cut_values[bin_idx - 1]
        }
    }
}

/// Compute quantile cuts, mirroring `SketchOnDMatrix` for the CPU `hist` path.
///
/// Features are sketched independently, so the work is split across threads by
/// feature. Each feature still sees its values in row order, exactly as the
/// serial path does, so the cuts do not depend on the thread count.
pub fn build_cuts(dmat: &DMatrix, max_bin: u32) -> Result<HistogramCuts> {
    build_cuts_weighted(dmat, max_bin, None)
}

/// The same sketch, with an explicit per-row weight.
///
/// This is the `hessian` argument of upstream's `SketchOnDMatrix`, and it is
/// what makes `approx` a different tree method from `hist`: `hist` sketches
/// once with the matrix's own row weights, `approx` re-sketches every round
/// with `hessian * row weight`, so the bin boundaries follow wherever the
/// current model is least certain. A row whose weight is zero — which is what
/// row sampling produces — contributes nothing, exactly as upstream's
/// zero-weight skip does.
pub fn build_cuts_weighted(
    dmat: &DMatrix,
    max_bin: u32,
    row_weights: Option<&[f32]>,
) -> Result<HistogramCuts> {
    if let Some(w) = row_weights
        && w.len() != dmat.num_row()
    {
        return Err(crate::Error::DataShape { expected: dmat.num_row(), got: w.len() });
    }
    build_cuts_impl(dmat, max_bin, row_weights)
}

fn build_cuts_impl(
    dmat: &DMatrix,
    max_bin: u32,
    row_weights: Option<&[f32]>,
) -> Result<HistogramCuts> {
    let n_features = dmat.num_col();
    let column_sizes = dmat.column_sizes();
    let max_bins = max_bin as usize;

    let mut sketches: Vec<QuantileSketch> = (0..n_features)
        .map(|f| {
            let mut s = QuantileSketch::default();
            if column_sizes[f] > 0 {
                let n_bins = max_bins.min(column_sizes[f]).max(1);
                let eps = 1.0f64 / ((n_bins as f32 * K_FACTOR) as f64);
                s.init(column_sizes[f], eps);
            }
            s
        })
        .collect();

    // Group features so each task makes one pass over the rows; a task per
    // feature would re-read the whole matrix once per column.
    let chunk = n_features.div_ceil(crate::threading::num_threads() * 2).max(1);
    crate::threading::install(|| {
        sketches.par_chunks_mut(chunk).enumerate().for_each(|(c, group)| {
            let first = c * chunk;
            let last = first + group.len();
            for r in 0..dmat.num_row() {
                let w = match row_weights {
                    Some(weights) => weights[r],
                    None => dmat.info.weight(r),
                };
                let (idx, val) = dmat.row(r);
                for (&col, &v) in idx.iter().zip(val) {
                    let col = col as usize;
                    if col >= first && col < last {
                        group[col - first].push(v, w);
                    }
                }
            }
        });
    });

    // `AllReduce`: prune each sketch to the intermediate cut count.
    let mut reduced: Vec<Vec<Entry>> = vec![Vec::new(); n_features];
    let mut num_cuts = vec![0usize; n_features];
    crate::threading::install(|| {
        sketches
            .par_iter_mut()
            .zip(reduced.par_iter_mut())
            .zip(num_cuts.par_iter_mut())
            .enumerate()
            .for_each(|(f, ((sketch, reduced), num_cuts))| {
                if column_sizes[f] == 0 {
                    return;
                }
                let intermediate = column_sizes[f].min(max_bins * K_FACTOR as usize);
                let mut summary = Vec::new();
                sketch.get_summary(&mut summary);
                set_prune(&summary, intermediate, reduced);
                *num_cuts = intermediate;
            });
    });

    // `MakeCuts`.
    let mut cuts = HistogramCuts {
        cut_ptrs: vec![0u32],
        cut_values: Vec::new(),
        min_values: vec![0.0f32; n_features],
    };
    let mut final_summaries: Vec<Vec<Entry>> = vec![Vec::new(); n_features];
    for f in 0..n_features {
        let max_num_bins = num_cuts[f].min(max_bins);
        if num_cuts[f] != 0 {
            set_prune(&reduced[f], max_num_bins + 1, &mut final_summaries[f]);
            let mval = final_summaries[f][0].value;
            cuts.min_values[f] = mval - mval.abs() - 1e-5;
        } else {
            cuts.min_values[f] = 1e-5;
        }
    }

    for f in 0..n_features {
        let max_num_bins = num_cuts[f].min(max_bins);
        let a = &final_summaries[f];
        // `AddCutPoint`: element 0 is represented by `min_values`, so start at 1.
        let required = a.len().min(max_num_bins);
        for i in 1..required {
            let cpt = a[i].value;
            if i == 1 || cpt > *cuts.cut_values.last().unwrap() {
                cuts.cut_values.push(cpt);
            }
        }
        // A final cut strictly greater than every observed value.
        let cpt = if !a.is_empty() { a[a.len() - 1].value } else { cuts.min_values[f] };
        cuts.cut_values.push(cpt + (cpt.abs() + 1e-5));
        cuts.cut_ptrs.push(cuts.cut_values.len() as u32);
    }

    Ok(cuts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radix_sort_matches_the_comparison_sort() {
        // Mixed signs, zeros, duplicates and a wide exponent range.
        let values: Vec<f32> = (0..5000)
            .map(|i| {
                let x = ((i * 2654435761u64 as usize) % 9973) as f32;
                match i % 5 {
                    0 => -x,
                    1 => x * 1e-8,
                    2 => x * 1e8,
                    3 => 0.0,
                    _ => x,
                }
            })
            .collect();
        let mut a: Vec<QEntry> =
            values.iter().map(|&v| QEntry { value: v, weight: 1.0 }).collect();
        let mut b = a.clone();

        sort_queue(&mut a, &mut Vec::new());
        b.sort_unstable_by(|x, y| x.value.partial_cmp(&y.value).unwrap());

        let a: Vec<f32> = a.iter().map(|e| e.value).collect();
        let b: Vec<f32> = b.iter().map(|e| e.value).collect();
        assert_eq!(a, b);
    }

    #[test]
    fn single_distinct_value_yields_one_bin() {
        let d = DMatrix::from_dense(&[1.0, 1.0, 1.0, 1.0], 4, 1, f32::NAN).unwrap();
        let cuts = build_cuts(&d, 16).unwrap();
        assert_eq!(cuts.cut_ptrs, vec![0, 1]);
        assert!(cuts.cut_values[0] > 1.0);
        assert!(cuts.min_values[0] < 1.0);
        assert_eq!(cuts.search_bin(1.0, 0), 0);
    }

    #[test]
    fn few_distinct_values_are_kept_exactly() {
        let vals: Vec<f32> = (0..40).map(|i| (i % 4) as f32).collect();
        let d = DMatrix::from_dense(&vals, 40, 1, f32::NAN).unwrap();
        let cuts = build_cuts(&d, 16).unwrap();
        // Values 0..3: cuts are the upper bounds 1, 2, 3 plus a final sentinel.
        assert_eq!(cuts.cut_values[..3], [1.0, 2.0, 3.0]);
        assert_eq!(cuts.search_bin(0.0, 0), 0);
        assert_eq!(cuts.search_bin(1.0, 0), 1);
        assert_eq!(cuts.search_bin(3.0, 0), 3);
    }

    #[test]
    fn bins_never_exceed_max_bin() {
        let vals: Vec<f32> = (0..1000).map(|i| i as f32 * 0.5).collect();
        let d = DMatrix::from_dense(&vals, 1000, 1, f32::NAN).unwrap();
        for max_bin in [2u32, 8, 37, 256] {
            let cuts = build_cuts(&d, max_bin).unwrap();
            assert!(
                cuts.feature_bins(0) <= max_bin as usize,
                "max_bin={max_bin} produced {} bins",
                cuts.feature_bins(0)
            );
        }
    }

    #[test]
    fn empty_column_still_gets_one_bin() {
        let d = DMatrix::from_dense(&[1.0, f32::NAN, 2.0, f32::NAN], 2, 2, f32::NAN).unwrap();
        let cuts = build_cuts(&d, 16).unwrap();
        assert_eq!(cuts.feature_bins(1), 1);
    }
}
