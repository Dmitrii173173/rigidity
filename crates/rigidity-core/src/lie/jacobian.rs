//! The left Jacobian of `SO(3)` and its inverse.
//!
//! The left Jacobian relates a perturbation of the algebra to the
//! corresponding perturbation of the group:
//!
//! ```text
//! exp((φ + δ)^) ≈ exp((J_l(φ)·δ)^) · exp(φ^)
//! ```
//!
//! It also appears in the `SE(3)` exponential, whose translational part is
//! `J_l(φ)·ρ`.

use nalgebra::Matrix3;
use nalgebra::Vector3;

use super::series;
use super::so3::hat;

/// Left Jacobian of `SO(3)`.
///
/// `J_l(φ) = I + ((1 − cos θ)/θ²) φ^ + ((θ − sin θ)/θ³) (φ^)²`
pub fn left_jacobian_so3(phi: &Vector3<f64>) -> Matrix3<f64> {
    let theta = phi.norm();
    let hat_phi = hat(phi);
    Matrix3::identity()
        + hat_phi * series::one_minus_cos_over_theta_sq(theta)
        + hat_phi * hat_phi * series::theta_minus_sin_over_theta_cubed(theta)
}

/// Inverse left Jacobian of `SO(3)`.
///
/// `J_l⁻¹(φ) = I − ½ φ^ + (1/θ² − (1 + cos θ)/(2 θ sin θ)) (φ^)²`
///
/// The expression degenerates as θ → 2π, where `J_l` itself is singular.
/// Within `|φ| < π`, which is the range of [`So3::log`](super::So3::log),
/// it is well behaved.
pub fn inverse_left_jacobian_so3(phi: &Vector3<f64>) -> Matrix3<f64> {
    let theta = phi.norm();
    let hat_phi = hat(phi);
    Matrix3::identity() - hat_phi * 0.5
        + hat_phi * hat_phi * series::inverse_left_jacobian_coef(theta)
}
