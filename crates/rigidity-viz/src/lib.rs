//! Debug visualisation through Rerun, behind the `rerun` feature.
//!
//! No viewer of our own is written. Rerun is used as a log sink through
//! its stable SDK; embedding `re_viewer` is deliberately avoided, as its
//! custom-view interface breaks on every release.
//!
//! The core knows nothing about this crate: the dependency runs one way,
//! and an ordinary build does not pull Rerun in.
//!
//! # Example
//!
//! ```no_run
//! # #[cfg(feature = "rerun")]
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use rigidity_viz::Recorder;
//! let recorder = Recorder::spawn("rigidity")?;
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "rerun"))]
//! # fn main() {}
//! ```

#[cfg(feature = "rerun")]
mod recorder;

#[cfg(feature = "rerun")]
pub use recorder::{Recorder, VizError};
