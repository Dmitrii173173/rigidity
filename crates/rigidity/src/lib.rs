//! Point-cloud registration that reports which degrees of freedom the
//! geometry actually determined.
//!
//! This crate contains no code of its own. It is the single dependency to
//! add when you want the library: the workspace is split into `rigidity-*`
//! crates so that the core can keep a fixed, audited dependency list, and
//! that split is an implementation concern rather than something every
//! caller should have to reassemble in its own `Cargo.toml`.
//!
//! ```no_run
//! use rigidity::pipeline::{PrepareParams, RegisterParams, prepare, register_pair};
//!
//! let params = PrepareParams::default();
//! let moving = prepare("source.ply".as_ref(), &params)?;
//! let fixed = prepare("target.ply".as_ref(), &params)?;
//!
//! let result = register_pair(&moving, &fixed, &RegisterParams::default());
//! println!("{:.5} m", result.rmse);
//! # Ok::<(), rigidity::pipeline::PipelineError>(())
//! ```
//!
//! # Features
//!
//! | feature | pulls in | for |
//! |---|---|---|
//! | `pipeline` *(default)* | [`pipeline`], [`io`], [`spatial`] | file → surface → registration → report |
//! | `io` | [`io`] | PLY, PCD, LAS/LAZ, E57, TXT/CSV |
//! | `spatial` | [`spatial`] | the kd-tree |
//! | `scenes` | [`scenes`] | synthetic scenes with known null spaces |
//! | `graph` | [`graph`] | pose graphs whose edges are weighted by their own conditioning |
//!
//! With `default-features = false` only the core remains: `nalgebra`,
//! `rayon`, `thiserror` and nothing else.
//!
//! `rigidity-viz` is deliberately absent. It is a debugging aid whose
//! weight sits behind its own `rerun` feature, and a crate that wants it
//! is better off depending on it directly than reaching it through a
//! feature two levels down.

// ── the core, flattened ─────────────────────────────────────────────────
//
// Its modules sit at the top level rather than under `rigidity::core`:
// a module of that name would shadow the `core` of the standard library
// inside any file that uses both.

#[doc(inline)]
pub use rigidity_core::{cloud, icp, lie, linalg, neighbors, normals, observability, voxel};

#[doc(inline)]
pub use rigidity_core::{
    Attribute, AttributeData, BruteForce, CloudError, Neighbor, NeighborSearch, PointCloud,
};

/// The linear algebra the public interface is written in.
///
/// Re-exported for the same reason `rigidity-core` re-exports it: a caller
/// that resolves a different `nalgebra` gets type errors that read as
/// nonsense.
pub use rigidity_core::nalgebra;

// ── everything above the core ───────────────────────────────────────────

#[cfg(feature = "spatial")]
#[doc(inline)]
pub use rigidity_spatial as spatial;

#[cfg(feature = "io")]
#[doc(inline)]
pub use rigidity_io as io;

#[cfg(feature = "pipeline")]
#[doc(inline)]
pub use rigidity_pipeline as pipeline;

#[cfg(feature = "scenes")]
#[doc(inline)]
pub use rigidity_scenes as scenes;

#[cfg(feature = "graph")]
#[doc(inline)]
pub use rigidity_graph as graph;
