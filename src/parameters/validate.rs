//! Range checks shared by every parameter struct.
//!
//! The bounds mirror the `set_lower_bound` / `set_range` clauses of XGBoost's
//! `DMLC_DECLARE_PARAMETER` blocks. Where this crate is deliberately stricter
//! than upstream the deviation is documented at the call site.
//!
//! Floats are additionally rejected when non-finite: XGBoost's `strtof`-based
//! parser accepts `nan`/`inf` and then misbehaves far away from the config, so
//! failing fast here is a strictly better error.

use crate::error::{Error, Result};

/// A parameter value that can be range-checked and reported back to the user.
pub(crate) trait Bounded: Copy + PartialOrd + std::fmt::Display {
    /// Floats override this to exclude NaN and infinities.
    fn is_valid_number(self) -> bool {
        true
    }
}

impl Bounded for u32 {}
impl Bounded for u64 {}
impl Bounded for i64 {}
impl Bounded for usize {}

impl Bounded for f32 {
    fn is_valid_number(self) -> bool {
        self.is_finite()
    }
}

impl Bounded for f64 {
    fn is_valid_number(self) -> bool {
        self.is_finite()
    }
}

/// Reject NaN and infinities.
pub(crate) fn finite<T: Bounded>(name: &'static str, value: T) -> Result<()> {
    if value.is_valid_number() {
        Ok(())
    } else {
        Err(Error::invalid(name, format!("must be a finite number, got {value}")))
    }
}

/// `value >= lower`.
pub(crate) fn ge<T: Bounded>(name: &'static str, value: T, lower: T) -> Result<()> {
    finite(name, value)?;
    if value < lower {
        return Err(Error::invalid(name, format!("must be >= {lower}, got {value}")));
    }
    Ok(())
}

/// `value > lower`.
pub(crate) fn gt<T: Bounded>(name: &'static str, value: T, lower: T) -> Result<()> {
    finite(name, value)?;
    if value <= lower {
        return Err(Error::invalid(name, format!("must be > {lower}, got {value}")));
    }
    Ok(())
}

/// `value <= upper`.
pub(crate) fn le<T: Bounded>(name: &'static str, value: T, upper: T) -> Result<()> {
    finite(name, value)?;
    if value > upper {
        return Err(Error::invalid(name, format!("must be <= {upper}, got {value}")));
    }
    Ok(())
}

/// `lower <= value <= upper`.
pub(crate) fn closed<T: Bounded>(name: &'static str, value: T, lower: T, upper: T) -> Result<()> {
    finite(name, value)?;
    if value < lower || value > upper {
        return Err(Error::invalid(
            name,
            format!("must be in [{lower}, {upper}], got {value}"),
        ));
    }
    Ok(())
}

/// `lower <= value < upper`.
pub(crate) fn half_open<T: Bounded>(name: &'static str, value: T, lower: T, upper: T) -> Result<()> {
    finite(name, value)?;
    if value < lower || value >= upper {
        return Err(Error::invalid(
            name,
            format!("must be in [{lower}, {upper}), got {value}"),
        ));
    }
    Ok(())
}

/// `0 < value <= 1` — the documented range of every `subsample`/`colsample_*`
/// ratio. XGBoost declares these as an inclusive `[0, 1]` range, but a ratio of
/// zero selects no rows/columns at all and makes training a no-op, so this
/// crate rejects it the way the XGBoost documentation says it should.
pub(crate) fn ratio(name: &'static str, value: f32) -> Result<()> {
    finite(name, value)?;
    if value <= 0.0 || value > 1.0 {
        return Err(Error::invalid(name, format!("must be in (0, 1], got {value}")));
    }
    Ok(())
}
