//! The Lie groups `SO(3)` and `SE(3)`.
//!
//! Pose optimisation runs over the Lie algebra rather than over matrix
//! entries: six free parameters instead of sixteen constrained ones, and
//! the update `T ← exp(Δξ) · T` stays on the manifold by construction.
//!
//! Ordering convention: `ξ = [ρ; φ]` — translation first, then rotation.
//! It runs through the whole project: the same order indexes the rows of
//! the adjoint, the columns of the residual Jacobian and the degrees of
//! freedom in the conditioning report. It cannot be changed in one place
//! alone.

mod align;
mod jacobian;
mod se3;
mod series;
mod so3;

pub use align::absolute_orientation;
pub use jacobian::{
    inverse_left_jacobian_se3, inverse_left_jacobian_so3, inverse_right_jacobian_se3,
    left_jacobian_se3, left_jacobian_so3, right_jacobian_se3,
};
pub use se3::{Se3, hat_se3, vee_se3};
pub use so3::{So3, hat, vee};
