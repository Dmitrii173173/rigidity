//! Property tests for the Lie groups.
//!
//! What is checked is not individual examples but identities that must
//! hold for every element of the group.

use std::f64::consts::PI;

use nalgebra::{Matrix3, Vector3, Vector6};
use proptest::prelude::*;
use rigidity_core::lie::{
    Se3, So3, hat, hat_se3, inverse_left_jacobian_so3, left_jacobian_so3, vee, vee_se3,
};

/// An `so(3)` algebra vector with norm in `[0, π)`.
///
/// The upper bound is excluded: at θ = π the logarithm is defined only up
/// to sign, and the identity `log(exp(φ)) = φ` stops being an equality.
fn so3_vector() -> impl Strategy<Value = Vector3<f64>> {
    (
        -1.0f64..1.0,
        -1.0f64..1.0,
        -1.0f64..1.0,
        0.0f64..(PI - 1e-9),
    )
        .prop_filter_map("degenerate axis", |(x, y, z, theta)| {
            let axis = Vector3::new(x, y, z);
            (axis.norm() > 1e-3).then(|| axis.normalize() * theta)
        })
}

/// An `se(3)` algebra vector in the `[ρ; φ]` ordering.
fn se3_vector() -> impl Strategy<Value = Vector6<f64>> {
    (so3_vector(), -10.0f64..10.0, -10.0f64..10.0, -10.0f64..10.0)
        .prop_map(|(phi, rx, ry, rz)| Vector6::new(rx, ry, rz, phi[0], phi[1], phi[2]))
}

/// The left Jacobian from its definition,
/// `exp((φ+δ)^) ≈ exp((J_l(φ)δ)^)·exp(φ^)`, by central differences in each
/// coordinate of δ.
fn numeric_left_jacobian(phi: Vector3<f64>) -> Matrix3<f64> {
    // h ≈ ε^(1/3): the balance between truncation O(h²) and rounding
    // O(ε/h).
    const H: f64 = 1e-5;
    let base_inverse = So3::exp(&phi).inverse();
    let mut jacobian = Matrix3::zeros();
    for i in 0..3 {
        let mut delta = Vector3::zeros();
        delta[i] = H;
        let forward = (So3::exp(&(phi + delta)) * base_inverse).log();
        let backward = (So3::exp(&(phi - delta)) * base_inverse).log();
        jacobian.set_column(i, &((forward - backward) / (2.0 * H)));
    }
    jacobian
}

proptest! {
    // The default 256 cases are not enough: the inverse-left-Jacobian bug
    // at θ → π only showed up at 4096. The neighbourhood of π occupies a
    // small share of a uniform sample in θ, and a rare branch needs
    // density.
    #![proptest_config(ProptestConfig { cases: 4096, ..ProptestConfig::default() })]

    /// `log(exp(φ)) = φ`
    #[test]
    fn so3_log_exp_roundtrip(phi in so3_vector()) {
        let recovered = So3::exp(&phi).log();
        prop_assert!((recovered - phi).norm() < 1e-12, "error {:.3e}", (recovered - phi).norm());
    }

    /// `exp(log(R)) = R`
    #[test]
    fn so3_exp_log_roundtrip(phi in so3_vector()) {
        let rotation = So3::exp(&phi);
        let recovered = So3::exp(&rotation.log());
        let err = (recovered.matrix() - rotation.matrix()).norm();
        prop_assert!(err < 1e-12, "error {err:.3e}");
    }

    /// `exp(φ)` belongs to `SO(3)`: orthogonal, with unit determinant.
    #[test]
    fn so3_exp_is_a_rotation(phi in so3_vector()) {
        let m = *So3::exp(&phi).matrix();
        let orthogonality = (m.transpose() * m - Matrix3::identity()).norm();
        prop_assert!(orthogonality < 1e-14, "RᵀR − I = {orthogonality:.3e}");
        prop_assert!((m.determinant() - 1.0).abs() < 1e-14);
    }

    /// Cross-check against the independent implementation in `nalgebra`.
    ///
    /// Catches convention mistakes — sign, ordering, transposition — that
    /// internally consistent tests let through.
    #[test]
    fn so3_exp_matches_nalgebra(phi in so3_vector()) {
        let ours = *So3::exp(&phi).matrix();
        let theirs = *nalgebra::Rotation3::from_scaled_axis(phi).matrix();
        prop_assert!((ours - theirs).norm() < 1e-14);
    }

    /// `Adj(R)·φ = vee(R·φ^·Rᵀ)`, which for `SO(3)` is simply `Rφ`.
    #[test]
    fn so3_adjoint_identity(phi in so3_vector(), psi in so3_vector()) {
        let rotation = So3::exp(&phi);
        let conjugated = vee(&(rotation.matrix() * hat(&psi) * rotation.matrix().transpose()));
        let via_adjoint = rotation.adjoint() * psi;
        prop_assert!((conjugated - via_adjoint).norm() < 1e-12);
    }

    /// `J_l⁻¹(φ)·J_l(φ) = I`
    #[test]
    fn left_jacobian_inverse_is_inverse(phi in so3_vector()) {
        let product = inverse_left_jacobian_so3(&phi) * left_jacobian_so3(&phi);
        let err = (product - Matrix3::identity()).norm();
        prop_assert!(err < 1e-12, "J⁻¹J − I = {err:.3e}");
    }

    /// The analytical left Jacobian against a central difference.
    #[test]
    fn left_jacobian_matches_numeric(phi in so3_vector()) {
        let analytic = left_jacobian_so3(&phi);
        let numeric = numeric_left_jacobian(phi);
        let err = (analytic - numeric).norm();
        prop_assert!(err < 1e-7, "analytic against numeric: {err:.3e}");
    }

    /// `log(exp(ξ)) = ξ`
    #[test]
    fn se3_log_exp_roundtrip(xi in se3_vector()) {
        let recovered = Se3::exp(&xi).log();
        prop_assert!((recovered - xi).norm() < 1e-12, "error {:.3e}", (recovered - xi).norm());
    }

    /// `exp(log(T)) = T`
    #[test]
    fn se3_exp_log_roundtrip(xi in se3_vector()) {
        let transform = Se3::exp(&xi);
        let recovered = Se3::exp(&transform.log());
        let err = (recovered.matrix() - transform.matrix()).norm();
        prop_assert!(err < 1e-12, "error {err:.3e}");
    }

    /// `Adj(T)·ξ = vee(T·ξ^·T⁻¹)`, the defining property of the adjoint.
    /// It also checks that the `[ρ; φ]` ordering agrees across `hat_se3`,
    /// `adjoint` and `exp`.
    #[test]
    fn se3_adjoint_identity(xi in se3_vector(), eta in se3_vector()) {
        let transform = Se3::exp(&xi);
        let conjugated = vee_se3(&(transform.matrix() * hat_se3(&eta) * transform.inverse().matrix()));
        let via_adjoint = transform.adjoint() * eta;
        let err = (conjugated - via_adjoint).norm();
        prop_assert!(err < 1e-12, "error {err:.3e}");
    }

    /// `Adj(T₁T₂) = Adj(T₁)·Adj(T₂)`: the adjoint really is a
    /// homomorphism.
    #[test]
    fn se3_adjoint_is_a_homomorphism(xi in se3_vector(), eta in se3_vector()) {
        let (a, b) = (Se3::exp(&xi), Se3::exp(&eta));
        let err = ((a * b).adjoint() - a.adjoint() * b.adjoint()).norm();
        prop_assert!(err < 1e-11, "error {err:.3e}");
    }

    /// `T⁻¹·T = I`
    #[test]
    fn se3_inverse_is_inverse(xi in se3_vector()) {
        let transform = Se3::exp(&xi);
        let product = transform.inverse() * transform;
        let err = (product.matrix() - Se3::identity().matrix()).norm();
        prop_assert!(err < 1e-12, "error {err:.3e}");
    }

    /// The action on a point agrees with homogeneous matrix multiplication.
    #[test]
    fn se3_transform_point_matches_matrix(xi in se3_vector(), px in -5.0f64..5.0, py in -5.0f64..5.0, pz in -5.0f64..5.0) {
        let transform = Se3::exp(&xi);
        let point = Vector3::new(px, py, pz);
        let direct = transform.transform_point(&point);
        let homogeneous = transform.matrix() * nalgebra::Vector4::new(px, py, pz, 1.0);
        let err = (direct - homogeneous.xyz()).norm();
        prop_assert!(err < 1e-12, "error {err:.3e}");
    }
}

/// The Taylor branch must stay continuous across every scale of θ.
///
/// This is where an implementation breaks quietly: as θ → 0 the Rodrigues
/// formula divides by θ, and without the series the result either blows up
/// or loses significant digits without a single runtime error.
#[test]
fn taylor_branch_is_continuous_at_small_angles() {
    let axis = Vector3::new(1.0, -2.0, 0.5).normalize();
    for theta in [0.0, 1e-12, 1e-8, 1e-4, 1e-2, 1e-1] {
        let phi = axis * theta;

        let rotation = So3::exp(&phi);
        let orthogonality =
            (rotation.matrix().transpose() * rotation.matrix() - Matrix3::identity()).norm();
        assert!(
            orthogonality < 1e-14,
            "θ = {theta:e}: RᵀR − I = {orthogonality:.3e}"
        );

        let err = (rotation.log() - phi).norm();
        assert!(err < 1e-10, "θ = {theta:e}: log∘exp error {err:.3e}");

        // ‖J_l(φ) − I‖ = O(θ): the leading term is ½‖φ^‖ = θ/√2. The
        // bound is non-strict, since at θ = 0 the Jacobian is exactly I.
        let deviation = (left_jacobian_so3(&phi) - Matrix3::identity()).norm();
        assert!(
            deviation <= theta,
            "θ = {theta:e}: J_l deviation {deviation:.3e}"
        );

        let product = inverse_left_jacobian_so3(&phi) * left_jacobian_so3(&phi);
        let inverse_err = (product - Matrix3::identity()).norm();
        assert!(
            inverse_err < 1e-13,
            "θ = {theta:e}: J⁻¹J − I = {inverse_err:.3e}"
        );
    }
}

/// A zero vector must not lead to a division by zero.
#[test]
fn exact_zero_is_identity() {
    let zero3 = Vector3::zeros();
    assert_eq!(*So3::exp(&zero3).matrix(), Matrix3::identity());
    assert_eq!(left_jacobian_so3(&zero3), Matrix3::identity());
    assert_eq!(inverse_left_jacobian_so3(&zero3), Matrix3::identity());
    assert_eq!(Se3::exp(&Vector6::zeros()), Se3::identity());
    assert_eq!(So3::identity().log(), zero3);
    assert_eq!(Se3::identity().log(), Vector6::zeros());
}

/// Stability of the inverse left Jacobian as θ → π.
///
/// The coefficient of `(φ^)²` is written through `tan(θ/2)`. The naive
/// form with `(1 + cos θ)` fails this test: at θ = π − 1e-6 it deviates by
/// about 7e-10 instead of 1e-15.
#[test]
fn inverse_left_jacobian_is_stable_near_pi() {
    let axis = Vector3::new(1.0, -2.0, 0.5).normalize();
    for delta in [1e-2, 1e-4, 1e-6, 1e-8, 0.0] {
        let phi = axis * (PI - delta);
        let product = inverse_left_jacobian_so3(&phi) * left_jacobian_so3(&phi);
        let err = (product - Matrix3::identity()).norm();
        assert!(err < 1e-13, "θ = π − {delta:e}: J⁻¹J − I = {err:.3e}");
    }
}

/// Regression: the counterexample proptest found at 4096 cases.
///
/// `|φ| = π − 2.9e-4` with a translational part of norm about 10 — the
/// coefficient's error is amplified by the length of the translation.
#[test]
fn se3_roundtrip_regression_near_pi() {
    let xi = Vector6::new(
        0.0,
        7.263_142_168_465_041,
        -6.727_543_297_705_218_5,
        0.0,
        -1.999_203_834_958_153_6,
        -2.423_031_111_602_081_7,
    );
    let err = (Se3::exp(&xi).log() - xi).norm();
    assert!(err < 1e-12, "error {err:.3e}");
}
