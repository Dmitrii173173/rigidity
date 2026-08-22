//! The left Jacobians of `SO(3)` and `SE(3)`, and their inverses.
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

use nalgebra::{Matrix3, Matrix6, Vector3, Vector6};

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

/// The `Q` block of the `SE(3)` left Jacobian.
///
/// Barfoot's equation 7.86. It is the whole difficulty of the `SE(3)`
/// Jacobian: the two diagonal blocks are just `J_l(φ)`, and everything that
/// makes the rotational and translational parts talk to each other is here.
///
/// The three coefficients lead with `θ⁰`, `θ⁴` and `θ⁵`; the second and
/// third are the reason `THETA_SMALL_Q` exists.
fn q_block(rho: &Vector3<f64>, phi: &Vector3<f64>) -> Matrix3<f64> {
    let theta = phi.norm();
    let (p, r) = (hat(phi), hat(rho));
    let (pr, rp) = (p * r, r * p);
    let prp = pr * p;

    0.5 * r
        + series::theta_minus_sin_over_theta_cubed(theta) * (pr + rp + prp)
        + series::q_second_coef(theta) * (p * pr + rp * p - 3.0 * prp)
        + series::q_third_coef(theta) * (prp * p + p * prp)
}

/// Left Jacobian of `SE(3)`, in the `ξ = [ρ; φ]` ordering.
///
/// ```text
/// J_l(ξ) = [[ J_l(φ),  Q(ξ) ],
///           [ 0,       J_l(φ) ]]
/// ```
///
/// Its defining property is the one the `SO(3)` version has:
/// `exp(ξ + δ) ≈ exp(J_l(ξ)·δ) · exp(ξ)` to first order in `δ`. That is
/// what `the_left_jacobian_is_the_derivative_of_exp` measures, against a
/// perturbation small enough that the neglected term is below the
/// tolerance and large enough that the difference is not all rounding.
pub fn left_jacobian_se3(xi: &Vector6<f64>) -> Matrix6<f64> {
    let rho = Vector3::new(xi[0], xi[1], xi[2]);
    let phi = Vector3::new(xi[3], xi[4], xi[5]);
    let j = left_jacobian_so3(&phi);

    let mut out = Matrix6::zeros();
    out.fixed_view_mut::<3, 3>(0, 0).copy_from(&j);
    out.fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&q_block(&rho, &phi));
    out.fixed_view_mut::<3, 3>(3, 3).copy_from(&j);
    out
}

/// Inverse left Jacobian of `SE(3)`.
///
/// ```text
/// J_l⁻¹(ξ) = [[ J_l⁻¹(φ),  −J_l⁻¹(φ)·Q·J_l⁻¹(φ) ],
///             [ 0,          J_l⁻¹(φ)            ]]
/// ```
///
/// Written out rather than obtained by inverting the 6×6: the block form is
/// exact, costs two 3×3 products instead of a decomposition, and cannot
/// fail. A general inverse of a matrix that is singular nowhere is a
/// `Result` nobody can act on.
pub fn inverse_left_jacobian_se3(xi: &Vector6<f64>) -> Matrix6<f64> {
    let rho = Vector3::new(xi[0], xi[1], xi[2]);
    let phi = Vector3::new(xi[3], xi[4], xi[5]);
    let inverse = inverse_left_jacobian_so3(&phi);

    let mut out = Matrix6::zeros();
    out.fixed_view_mut::<3, 3>(0, 0).copy_from(&inverse);
    out.fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-inverse * q_block(&rho, &phi) * inverse));
    out.fixed_view_mut::<3, 3>(3, 3).copy_from(&inverse);
    out
}

/// Right Jacobian of `SE(3)`: `J_r(ξ) = J_l(−ξ)`.
///
/// The right Jacobian is what a pose graph wants, because a graph
/// perturbs its nodes on the right — `T ← T·exp(δ)` keeps the increment in
/// the body frame, which is where the measurements were taken.
pub fn right_jacobian_se3(xi: &Vector6<f64>) -> Matrix6<f64> {
    left_jacobian_se3(&(-xi))
}

/// Inverse right Jacobian of `SE(3)`: `J_r⁻¹(ξ) = J_l⁻¹(−ξ)`.
pub fn inverse_right_jacobian_se3(xi: &Vector6<f64>) -> Matrix6<f64> {
    inverse_left_jacobian_se3(&(-xi))
}

#[cfg(test)]
mod se3_tests {
    use super::*;
    use crate::lie::Se3;

    /// A spread of algebra vectors: near zero, ordinary, and near π.
    ///
    /// Near π matters because the `SO(3)` logarithm changes branch there
    /// and the coefficients this file uses are at their least comfortable;
    /// near zero matters because that is where the Taylor branches take
    /// over. A test over the middle only would pass on a `Q` block that was
    /// wrong at both ends.
    fn samples() -> Vec<Vector6<f64>> {
        let mut out = Vec::new();
        // The last is just short of π, where the SO(3) logarithm changes
        // branch. Written as a subtraction rather than as 3.14 so that it
        // is unmistakably *near* π rather than a number that happens to
        // look like it.
        let near_pi = std::f64::consts::PI - 1e-3;
        for theta in [0.0, 1e-6, 1e-3, 0.1, 0.499, 0.501, 1.0, 2.0, 3.0, near_pi] {
            for (index, axis) in [
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
                Vector3::new(1.0, 2.0, -3.0).normalize(),
            ]
            .into_iter()
            .enumerate()
            {
                let phi = axis * theta;
                let rho = Vector3::new(0.3, -1.7, 2.5) * (1.0 + index as f64);
                out.push(Vector6::new(rho[0], rho[1], rho[2], phi[0], phi[1], phi[2]));
            }
        }
        out
    }

    /// The definition: `exp(ξ + δ) ≈ exp(J_l(ξ)·δ) · exp(ξ)`.
    ///
    /// This is the oracle. Everything else in this module is a consequence
    /// of it, and a `Q` block with a sign error would satisfy every
    /// structural test — symmetry, the block layout, agreeing with its own
    /// inverse — while failing this one.
    ///
    /// `δ` is 1e-6: the neglected term is second order, so about 1e-12,
    /// which is why the tolerance is what it is and not smaller.
    #[test]
    fn the_left_jacobian_is_the_derivative_of_exp() {
        const STEP: f64 = 1e-6;
        for xi in samples() {
            let jacobian = left_jacobian_se3(&xi);
            let base = Se3::exp(&xi);
            for axis in 0..6 {
                let mut delta = Vector6::zeros();
                delta[axis] = STEP;

                let moved = Se3::exp(&(xi + delta));
                let measured = (moved * base.inverse()).log();
                let predicted = jacobian * delta;

                let error = (measured - predicted).norm() / STEP;
                assert!(
                    error < 1e-5,
                    "ξ = {xi:?}, axis {axis}: the Jacobian is off by {error} per unit step"
                );
            }
        }
    }

    /// And the right Jacobian is the same statement perturbed on the right.
    #[test]
    fn the_right_jacobian_is_the_derivative_on_the_other_side() {
        const STEP: f64 = 1e-6;
        for xi in samples() {
            let jacobian = right_jacobian_se3(&xi);
            let base = Se3::exp(&xi);
            for axis in 0..6 {
                let mut delta = Vector6::zeros();
                delta[axis] = STEP;

                let moved = Se3::exp(&(xi + delta));
                let measured = (base.inverse() * moved).log();
                let predicted = jacobian * delta;

                let error = (measured - predicted).norm() / STEP;
                assert!(
                    error < 1e-5,
                    "ξ = {xi:?}, axis {axis}: off by {error} per unit step"
                );
            }
        }
    }

    #[test]
    fn the_inverse_undoes_the_jacobian() {
        for xi in samples() {
            let product = left_jacobian_se3(&xi) * inverse_left_jacobian_se3(&xi);
            let error = (product - Matrix6::identity()).abs().max();
            assert!(error < 1e-12, "ξ = {xi:?}: J·J⁻¹ is off by {error}");

            let product = inverse_right_jacobian_se3(&xi) * right_jacobian_se3(&xi);
            let error = (product - Matrix6::identity()).abs().max();
            assert!(
                error < 1e-12,
                "ξ = {xi:?}: the right pair is off by {error}"
            );
        }
    }

    /// At `φ = 0` the Jacobian is the identity and `Q` is `½ρ^`.
    ///
    /// The one case with an answer that can be written down, which is worth
    /// having when every other test is a comparison against a difference
    /// quotient.
    #[test]
    fn without_rotation_the_answer_is_known_exactly() {
        let rho = Vector3::new(0.4, -2.0, 7.5);
        let xi = Vector6::new(rho[0], rho[1], rho[2], 0.0, 0.0, 0.0);
        let jacobian = left_jacobian_se3(&xi);

        assert!(
            (jacobian.fixed_view::<3, 3>(0, 0) - Matrix3::identity())
                .abs()
                .max()
                < 1e-15
        );
        assert!(
            (jacobian.fixed_view::<3, 3>(3, 3) - Matrix3::identity())
                .abs()
                .max()
                < 1e-15
        );
        assert!(
            (jacobian.fixed_view::<3, 3>(0, 3) - 0.5 * hat(&rho))
                .abs()
                .max()
                < 1e-15
        );
        assert_eq!(jacobian.fixed_view::<3, 3>(3, 0), Matrix3::zeros());
    }

    /// Both branches of the two new coefficients agree where they meet.
    ///
    /// The companion of `series::branches_agree_at_threshold`, and the
    /// reason `THETA_SMALL_Q` is 0.5 rather than the 1e-2 the others use.
    #[test]
    fn q_branches_agree_at_threshold() {
        let theta = series::THETA_SMALL_Q;
        for (name, taylor, trig, scale) in [
            (
                "second",
                series::taylor_q_second_coef(theta),
                series::trig_q_second_coef(theta),
                1.0 / 24.0,
            ),
            (
                "third",
                series::taylor_q_third_coef(theta),
                series::trig_q_third_coef(theta),
                1.0 / 120.0,
            ),
        ] {
            let relative = (taylor - trig).abs() / scale;
            assert!(
                relative < 1e-11,
                "the {name} coefficient's branches differ by {relative} relative at the threshold"
            );
        }
    }
}
