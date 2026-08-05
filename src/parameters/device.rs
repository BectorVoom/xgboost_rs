//! The `device` parameter: which processor runs the fit.
//!
//! Mirrors `MakeDeviceOrd` in `xgboost/src/context.cc`, whose accepted grammar
//! is `gpu(:[0-9]+)?|cuda(:[0-9]+)?|cpu|sycl(:cpu|:gpu)?(:-1|:[0-9]+)?`.
//! `gpu` is an alias for `cuda` and is normalised on parse, exactly as upstream
//! does, so [`Device::to_string`] always emits the canonical spelling.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::{Error, Result};

const NAME: &str = "device";

/// Which SYCL device class a [`Device::Sycl`] refers to.
///
/// SYCL is an XGBoost plugin; this crate parses and re-emits the spelling but
/// has no SYCL execution path of its own.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SyclKind {
    /// `sycl` — let the SYCL runtime choose.
    #[default]
    Default,
    /// `sycl:cpu`.
    Cpu,
    /// `sycl:gpu`.
    Gpu,
}

/// The `device` parameter.
///
/// ```
/// use xgboost_rs::parameters::Device;
///
/// assert_eq!("cuda:1".parse::<Device>().unwrap(), Device::cuda(1));
/// // `gpu` is an alias that normalises to `cuda`, as in XGBoost itself.
/// assert_eq!("gpu:1".parse::<Device>().unwrap().to_string(), "cuda:1");
/// assert!(Device::cuda(0).is_gpu());
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Device {
    /// `cpu`.
    #[default]
    Cpu,
    /// `cuda` (no ordinal, XGBoost picks device 0) or `cuda:<ordinal>`.
    Cuda(Option<u16>),
    /// `sycl`, `sycl:cpu`, `sycl:gpu`, optionally with an ordinal (`-1` means
    /// "any", which is why the ordinal is signed here and not elsewhere).
    Sycl(SyclKind, Option<i32>),
}

impl Device {
    /// `cuda:<ordinal>`.
    pub fn cuda(ordinal: u16) -> Self {
        Self::Cuda(Some(ordinal))
    }

    /// True for `cpu` only. Note that `sycl:cpu` is *not* the CPU device as far
    /// as XGBoost's `Context::IsCPU` is concerned, and it is not here either.
    pub fn is_cpu(self) -> bool {
        matches!(self, Self::Cpu)
    }

    /// True for `cuda` / `cuda:<n>`. This is the flag that selects the
    /// `grow_gpu_*` updaters.
    pub fn is_cuda(self) -> bool {
        matches!(self, Self::Cuda(_))
    }

    /// True for any accelerator device: CUDA, `sycl`, or `sycl:gpu`.
    pub fn is_gpu(self) -> bool {
        matches!(self, Self::Cuda(_) | Self::Sycl(SyclKind::Default | SyclKind::Gpu, _))
    }

    /// True for `sycl*`.
    pub fn is_sycl(self) -> bool {
        matches!(self, Self::Sycl(..))
    }

    /// The device ordinal, when one was given explicitly.
    pub fn ordinal(self) -> Option<i32> {
        match self {
            Self::Cpu => None,
            Self::Cuda(ordinal) => ordinal.map(i32::from),
            Self::Sycl(_, ordinal) => ordinal,
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => f.write_str("cpu"),
            Self::Cuda(None) => f.write_str("cuda"),
            Self::Cuda(Some(ordinal)) => write!(f, "cuda:{ordinal}"),
            Self::Sycl(kind, ordinal) => {
                f.write_str("sycl")?;
                match kind {
                    SyclKind::Default => {}
                    SyclKind::Cpu => f.write_str(":cpu")?,
                    SyclKind::Gpu => f.write_str(":gpu")?,
                }
                match ordinal {
                    Some(ordinal) => write!(f, ":{ordinal}"),
                    None => Ok(()),
                }
            }
        }
    }
}

impl FromStr for Device {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        let bad = |reason: &str| Error::parse(NAME, s, reason.to_owned());
        let expected = "expected one of: cpu, cuda, cuda:<ordinal>, gpu, gpu:<ordinal>, \
                        sycl, sycl:cpu, sycl:gpu, each optionally with `:<ordinal>`";

        // `gpu` is an alias for `cuda`; normalise before matching so the rest of
        // the crate never has to think about it again.
        if let Some(rest) = s.strip_prefix("gpu") {
            return format!("cuda{rest}").parse();
        }

        if s == "cpu" {
            return Ok(Self::Cpu);
        }
        if s == "cuda" {
            return Ok(Self::Cuda(None));
        }
        if let Some(ordinal) = s.strip_prefix("cuda:") {
            let ordinal = ordinal
                .parse::<u16>()
                .map_err(|_| bad("CUDA ordinal must be a non-negative integer"))?;
            return Ok(Self::Cuda(Some(ordinal)));
        }
        if let Some(rest) = s.strip_prefix("sycl") {
            return parse_sycl(s, rest);
        }
        Err(bad(expected))
    }
}

fn parse_sycl(whole: &str, rest: &str) -> Result<Device> {
    let bad = |reason: &str| Error::parse(NAME, whole, reason.to_owned());
    let ordinal = |text: &str| -> Result<i32> {
        let value = text
            .parse::<i32>()
            .map_err(|_| bad("SYCL ordinal must be an integer or -1"))?;
        if value < -1 {
            return Err(bad("SYCL ordinal must be >= -1"));
        }
        Ok(value)
    };

    // Longest prefixes first: `sycl:cpu:0` must not be read as kind-less.
    for (prefix, kind) in [(":cpu", SyclKind::Cpu), (":gpu", SyclKind::Gpu)] {
        if rest == prefix {
            return Ok(Device::Sycl(kind, None));
        }
        if let Some(tail) = rest.strip_prefix(prefix)
            && let Some(tail) = tail.strip_prefix(':')
        {
            return Ok(Device::Sycl(kind, Some(ordinal(tail)?)));
        }
    }
    if rest.is_empty() {
        return Ok(Device::Sycl(SyclKind::Default, None));
    }
    if let Some(tail) = rest.strip_prefix(':') {
        return Ok(Device::Sycl(SyclKind::Default, Some(ordinal(tail)?)));
    }
    Err(bad("unrecognised SYCL device spelling"))
}

impl Serialize for Device {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for Device {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        text.parse().map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_every_accepted_spelling() {
        for (input, canonical) in [
            ("cpu", "cpu"),
            ("cuda", "cuda"),
            ("cuda:0", "cuda:0"),
            ("cuda:7", "cuda:7"),
            ("gpu", "cuda"),
            ("gpu:3", "cuda:3"),
            ("sycl", "sycl"),
            ("sycl:cpu", "sycl:cpu"),
            ("sycl:gpu", "sycl:gpu"),
            ("sycl:gpu:1", "sycl:gpu:1"),
            ("sycl:cpu:-1", "sycl:cpu:-1"),
            ("sycl:2", "sycl:2"),
        ] {
            let device: Device = input.parse().unwrap();
            assert_eq!(device.to_string(), canonical, "input {input}");
            assert_eq!(canonical.parse::<Device>().unwrap(), device);
        }
    }

    #[test]
    fn rejects_invalid_spellings() {
        for input in ["", "CPU", "cuda:", "cuda:-1", "cuda:x", "gpu:-2", "sycl:tpu", "tpu"] {
            assert!(input.parse::<Device>().is_err(), "should reject {input:?}");
        }
    }

    #[test]
    fn classifies_devices() {
        assert!(Device::Cpu.is_cpu());
        assert!(!Device::Cpu.is_gpu());
        assert!(Device::cuda(0).is_cuda() && Device::cuda(0).is_gpu());
        assert!(!Device::Sycl(SyclKind::Cpu, None).is_gpu());
        assert!(Device::Sycl(SyclKind::Gpu, None).is_gpu());
        assert!(!Device::Sycl(SyclKind::Gpu, None).is_cuda());
        assert_eq!(Device::cuda(4).ordinal(), Some(4));
    }
}
