//! Per-round wall clock of the host side of a fit, printed when
//! `XGB_PHASES` is set: what surrounds a tree — the objective's gradients,
//! the prediction update, the metric — which the device grower's own phase
//! log (`gpu::PhaseLog`) does not see.

use std::time::{Duration, Instant};

/// Whether `XGB_PHASES` is set; read once.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("XGB_PHASES").is_some())
}

/// A round's host phases, accumulated and printed as one line.
#[derive(Default)]
pub struct RoundLog {
    last: Option<Instant>,
    items: Vec<(&'static str, Duration)>,
}

impl RoundLog {
    pub fn start() -> Self {
        if !enabled() {
            return Self::default();
        }
        Self { last: Some(Instant::now()), items: Vec::new() }
    }

    /// Close the phase that began at the previous mark under `name`.
    pub fn mark(&mut self, name: &'static str) {
        if let Some(last) = &mut self.last {
            let now = Instant::now();
            self.items.push((name, now - *last));
            *last = now;
        }
    }

    pub fn report(&self, what: &str) {
        if self.last.is_none() {
            return;
        }
        let total: Duration = self.items.iter().map(|(_, d)| *d).sum();
        let mut line = format!("PHASES {what} {:.2}ms:", total.as_secs_f64() * 1e3);
        for (name, d) in &self.items {
            line.push_str(&format!(" {name}={:.2}ms", d.as_secs_f64() * 1e3));
        }
        eprintln!("{line}");
    }
}
