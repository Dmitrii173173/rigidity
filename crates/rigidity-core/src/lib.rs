//! The `rigidity` core.
//!
//! Dependencies are limited to `nalgebra`, `rayon` and `thiserror`: no file
//! I/O, no GPU, no UI. This crate must build and pass its tests on Linux,
//! Windows and macOS without a single system library.
//!
//! Three invariants hold throughout:
//! - points are stored as `f32` offsets from an `f64` origin, while all
//!   arithmetic is done in `f64`;
//! - `H = JᵀWJ` is never formed explicitly — the work goes through a
//!   tall-skinny QR of the weighted Jacobian;
//! - results are bit-for-bit reproducible at any thread count.

pub mod cloud;
pub mod icp;
pub mod lie;
pub mod linalg;
pub mod neighbors;
pub mod normals;
pub mod observability;
pub mod voxel;

pub use cloud::{Attribute, AttributeData, CloudError, PointCloud};
pub use neighbors::{BruteForce, Neighbor, NeighborSearch};
