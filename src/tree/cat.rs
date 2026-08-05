//! Category bit sets — how a categorical split records the categories it sends
//! right.
//!
//! A port of `common::CatBitField` (`src/common/bitfield.h`) and the free
//! functions in `src/common/categorical.h`, restricted to what a tree needs:
//! build a set while enumerating splits, then ask which way a row's category
//! goes. Category `c` is bit `c % 32` of word `c / 32`, which is the layout
//! upstream serialises, so a saved model's `categories` array is interchangeable
//! with the reference implementation's.

/// Bits per storage word. Upstream's `CatBitField` is `uint32_t`-based, and the
/// width is part of the on-disk format rather than an implementation choice.
const WORD_BITS: usize = 32;

/// One past the largest representable category, upstream's `OutOfRangeCat()`.
///
/// `f32` counts integers exactly up to 2^24; past that a category could not be
/// recovered from the value stored in the matrix, so upstream refuses the input
/// rather than silently rounding it.
pub const OUT_OF_RANGE_CAT: f32 = 16_777_216.0;

/// Words needed to hold `n_categories` bits.
pub const fn storage_size(n_categories: usize) -> usize {
    n_categories.div_ceil(WORD_BITS)
}

/// `common::InvalidCat` — whether `cat` is outside the representable range.
#[inline]
pub fn invalid_cat(cat: f32) -> bool {
    !(cat >= 0.0) || cat >= OUT_OF_RANGE_CAT
}

/// `common::AsCat` — the category code a stored value denotes, truncating
/// towards zero as upstream's `static_cast<bst_cat_t>` does.
#[inline]
pub fn as_cat(value: f32) -> u32 {
    value as u32
}

/// Add category `cat` to the set. Out-of-range bits are dropped, matching the
/// bounds check upstream's `Decision` performs on the reading side.
#[inline]
pub fn set_bit(bits: &mut [u32], cat: u32) {
    let (word, bit) = (cat as usize / WORD_BITS, cat as usize % WORD_BITS);
    if word < bits.len() {
        bits[word] |= 1 << bit;
    }
}

/// Whether category `cat` is in the set.
#[inline]
pub fn check_bit(bits: &[u32], cat: u32) -> bool {
    let (word, bit) = (cat as usize / WORD_BITS, cat as usize % WORD_BITS);
    matches!(bits.get(word), Some(w) if w & (1 << bit) != 0)
}

/// `common::Decision` — whether a row whose category is `value` takes the left
/// branch.
///
/// Left is "not one of the chosen categories", which is what makes a one-hot
/// split read as `feature == category → right`. A category the split never saw
/// — including one past the end of the stored bits, and any value outside the
/// representable range — is not chosen, so it goes left.
#[inline]
pub fn goes_left(bits: &[u32], value: f32) -> bool {
    if invalid_cat(value) {
        return true;
    }
    !check_bit(bits, as_cat(value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_set_category_goes_right_and_everything_else_goes_left() {
        let mut bits = vec![0u32; storage_size(70)];
        set_bit(&mut bits, 3);
        set_bit(&mut bits, 64);

        assert!(!goes_left(&bits, 3.0));
        assert!(!goes_left(&bits, 64.0));
        assert!(goes_left(&bits, 0.0));
        assert!(goes_left(&bits, 2.0));
        assert!(goes_left(&bits, 65.0));
    }

    #[test]
    fn storage_is_one_word_per_thirty_two_categories() {
        assert_eq!(storage_size(0), 0);
        assert_eq!(storage_size(1), 1);
        assert_eq!(storage_size(32), 1);
        assert_eq!(storage_size(33), 2);
    }

    #[test]
    fn a_category_past_the_stored_bits_goes_left() {
        let mut bits = vec![0u32; storage_size(4)];
        set_bit(&mut bits, 1);
        // Nothing was stored about category 100, so it cannot have been chosen.
        assert!(goes_left(&bits, 100.0));
        // And recording it was a no-op rather than a panic.
        set_bit(&mut bits, 100);
        assert_eq!(bits, vec![0b10]);
    }

    #[test]
    fn out_of_range_values_take_the_left_branch() {
        let bits = vec![u32::MAX; storage_size(64)];
        assert!(goes_left(&bits, -1.0));
        assert!(goes_left(&bits, f32::NAN));
        assert!(goes_left(&bits, OUT_OF_RANGE_CAT));
        assert!(!goes_left(&bits, 0.0), "an in-range category still reads its bit");
    }

    #[test]
    fn non_integral_values_truncate_towards_zero() {
        assert_eq!(as_cat(1.9), 1);
        assert_eq!(as_cat(0.5), 0);
    }

    #[test]
    fn invalid_cat_covers_the_whole_unrepresentable_range() {
        assert!(invalid_cat(-0.5));
        assert!(invalid_cat(f32::NAN));
        assert!(invalid_cat(f32::INFINITY));
        assert!(invalid_cat(OUT_OF_RANGE_CAT));
        assert!(!invalid_cat(0.0));
        assert!(!invalid_cat(OUT_OF_RANGE_CAT - 1.0));
    }
}
