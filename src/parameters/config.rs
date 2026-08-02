//! Emission of XGBoost-compatible configuration key/value pairs.
//!
//! Every parameter struct in this module tree knows how to flatten itself into
//! the `(name, value)` string pairs XGBoost's own `Learner::Configure` accepts.
//! The names are the upstream ones (`eta`, `max_depth`, `device`, …) so a map
//! produced here can be handed verbatim to a real XGBoost build — that is what
//! makes the oracle harness possible.

use std::collections::BTreeMap;

/// One `(parameter name, parameter value)` pair, both already stringified the
/// way XGBoost's DMLC parameter parser expects.
pub type ConfigEntry = (String, String);

/// Flatten a parameter struct into XGBoost configuration entries.
pub trait ToConfig {
    /// Append this struct's entries to `out`.
    ///
    /// Implementors emit *only* the parameters that apply to the current
    /// configuration: a `gbtree` booster never emits `rate_drop`, and an
    /// objective never emits another objective's tuning knobs. Optional
    /// parameters that are `None` are omitted entirely so XGBoost applies its
    /// own default.
    fn collect_config(&self, out: &mut Vec<ConfigEntry>);

    /// Entries in declaration order (stable, may contain no duplicates).
    fn to_config(&self) -> Vec<ConfigEntry> {
        let mut out = Vec::new();
        self.collect_config(&mut out);
        out
    }

    /// Entries as a name-sorted map, the shape most callers want.
    ///
    /// XGBoost's own `Args` is a vector of pairs and genuinely allows a name to
    /// repeat — `eval_metric` is the one parameter that does. Repeats are
    /// joined with `,` here rather than silently dropped, so
    /// `eval_metric = "rmse,mae"` round-trips back to a two-element list.
    fn to_config_map(&self) -> BTreeMap<String, String> {
        let mut map: BTreeMap<String, String> = BTreeMap::new();
        for (key, value) in self.to_config() {
            map.entry(key)
                .and_modify(|existing| {
                    existing.push(',');
                    existing.push_str(&value);
                })
                .or_insert(value);
        }
        map
    }

    /// Entries as a flat JSON object of strings, ready to be written to disk
    /// and loaded by a fixture generator.
    fn to_config_json(&self) -> String {
        // Serialising a `BTreeMap<String, String>` cannot fail.
        serde_json::to_string_pretty(&self.to_config_map())
            .expect("BTreeMap<String, String> is always serialisable")
    }
}

/// Append `key = value`, stringifying via [`std::fmt::Display`].
///
/// Floats go through Rust's shortest-round-trip formatting, which XGBoost's
/// `strtof`-based parser reads back exactly.
pub(crate) fn push(out: &mut Vec<ConfigEntry>, key: &str, value: impl std::fmt::Display) {
    out.push((key.to_owned(), value.to_string()));
}

/// Append a boolean as `"0"` / `"1"`, matching XGBoost's own config output.
pub(crate) fn push_bool(out: &mut Vec<ConfigEntry>, key: &str, value: bool) {
    push(out, key, if value { "1" } else { "0" });
}

/// Append only when the option is set, so unset parameters keep XGBoost's
/// default instead of being pinned to ours.
pub(crate) fn push_opt(out: &mut Vec<ConfigEntry>, key: &str, value: Option<impl std::fmt::Display>) {
    if let Some(value) = value {
        push(out, key, value);
    }
}

/// Append a float list in XGBoost's `ParamArray` syntax, e.g. `(0.1,0.5,0.9)`.
pub(crate) fn push_f32_list(out: &mut Vec<ConfigEntry>, key: &str, values: &[f32]) {
    let joined =
        values.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",");
    push(out, key, format!("({joined})"));
}
