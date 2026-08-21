//! Iterative closest point with a point-to-plane residual.
//!
//! Conventions: the pose update is left-multiplied, `T ← exp(Δξ)·T`, and
//! algebra coordinates are ordered `ξ = [ρ; φ]`. The synthetic scenes'
//! null spaces and the conditioning analysis assume the same conventions,
//! so neither can be changed in isolation.

mod kernel;
mod lm;
mod residual;

pub use kernel::Kernel;
pub use lm::{
    IcpConfig, IcpResult, IterationReport, Surface, register, register_observed, surface,
};
pub use residual::point_to_plane_row;
