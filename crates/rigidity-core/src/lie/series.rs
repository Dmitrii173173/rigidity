//! Scalar coefficients of the Rodrigues formula and the Jacobians.
//!
//! Every coefficient has a trigonometric form and a Taylor series. Below
//! `THETA_SMALL` the series is used, because the trigonometric form suffers
//! catastrophic cancellation there. The textbook case is `(θ − sin θ)/θ³`:
//! at θ = 1e-3 the numerator is 1.67e-10 while the machine step near θ is
//! 2.2e-19, so the subtraction throws away nine significant digits.
//!
//! The threshold is chosen so that both branches are accurate at the
//! crossover: truncating the series at θ = 1e-2 costs at most 3e-16, and
//! the cancellation in the trigonometric form at the same point is about
//! 2e-12. This is what `branches_agree_at_threshold` checks.

/// Where the Taylor branch takes over.
pub(crate) const THETA_SMALL: f64 = 1e-2;

/// The same, for the two coefficients of the `SE(3)` Jacobian's `Q` block.
///
/// Fifty times larger, and it has to be. The coefficients above lead with
/// `θ²`; these lead with `θ⁴` and `θ⁵`, so their trigonometric forms
/// subtract quantities that agree to four or five more digits before the
/// answer appears. At θ = 1e-2 the third coefficient's numerator is 1.7e-12
/// built out of terms of size 3e-2 — eleven digits gone, and the branch
/// that is supposed to be the accurate one is the worse of the two.
///
/// At 0.5 both are around 1e-12 relative, which is the same crossover
/// quality `THETA_SMALL` buys for the others; `q_branches_agree_at_threshold`
/// is what holds that claim up.
pub(crate) const THETA_SMALL_Q: f64 = 0.5;

/// Width of the neighbourhood of θ = π in which `log` switches to
/// recovering the axis from the symmetric part of the matrix.
///
/// As θ → π, sin θ → 0 and the factor θ/sin θ in the standard formula
/// amplifies the error. At the boundary π − 1e-2 the relative error of the
/// recovered axis is still around 1e-14.
pub(crate) const THETA_NEAR_PI: f64 = 1e-2;

/// `sin θ / θ`, the coefficient of `φ^` in the Rodrigues formula.
pub(crate) fn sin_over_theta(theta: f64) -> f64 {
    if theta < THETA_SMALL {
        taylor_sin_over_theta(theta)
    } else {
        theta.sin() / theta
    }
}

/// `(1 − cos θ) / θ²`, the coefficient of `(φ^)²` in the Rodrigues formula.
pub(crate) fn one_minus_cos_over_theta_sq(theta: f64) -> f64 {
    if theta < THETA_SMALL {
        taylor_one_minus_cos_over_theta_sq(theta)
    } else {
        trig_one_minus_cos_over_theta_sq(theta)
    }
}

/// `(θ − sin θ) / θ³`, the coefficient of `(φ^)²` in the left Jacobian.
pub(crate) fn theta_minus_sin_over_theta_cubed(theta: f64) -> f64 {
    if theta < THETA_SMALL {
        taylor_theta_minus_sin_over_theta_cubed(theta)
    } else {
        (theta - theta.sin()) / (theta * theta * theta)
    }
}

/// `1/θ² − (1 + cos θ) / (2 θ sin θ)`, the coefficient of `(φ^)²` in the
/// inverse left Jacobian.
pub(crate) fn inverse_left_jacobian_coef(theta: f64) -> f64 {
    if theta < THETA_SMALL {
        taylor_inverse_left_jacobian_coef(theta)
    } else {
        trig_inverse_left_jacobian_coef(theta)
    }
}

/// `(θ² + 2 cos θ − 2) / (2 θ⁴)`.
///
/// The second coefficient of Barfoot's `Q`, multiplying
/// `φ^φ^ρ^ + ρ^φ^φ^ − 3 φ^ρ^φ^`.
pub(crate) fn q_second_coef(theta: f64) -> f64 {
    if theta < THETA_SMALL_Q {
        taylor_q_second_coef(theta)
    } else {
        trig_q_second_coef(theta)
    }
}

/// `(2θ − 3 sin θ + θ cos θ) / (2 θ⁵)`.
///
/// The third coefficient of Barfoot's `Q`, multiplying
/// `φ^ρ^φ^φ^ + φ^φ^ρ^φ^`.
pub(crate) fn q_third_coef(theta: f64) -> f64 {
    if theta < THETA_SMALL_Q {
        taylor_q_third_coef(theta)
    } else {
        trig_q_third_coef(theta)
    }
}

/// `θ / sin θ`, the factor appearing in the `SO(3)` logarithm.
pub(crate) fn theta_over_sin_theta(theta: f64) -> f64 {
    if theta < THETA_SMALL {
        taylor_theta_over_sin_theta(theta)
    } else {
        theta / theta.sin()
    }
}

// --- Taylor series ----------------------------------------------------------

pub(crate) fn taylor_sin_over_theta(theta: f64) -> f64 {
    let t2 = theta * theta;
    1.0 - t2 / 6.0 + t2 * t2 / 120.0
}

pub(crate) fn taylor_one_minus_cos_over_theta_sq(theta: f64) -> f64 {
    let t2 = theta * theta;
    0.5 - t2 / 24.0 + t2 * t2 / 720.0
}

pub(crate) fn taylor_theta_minus_sin_over_theta_cubed(theta: f64) -> f64 {
    let t2 = theta * theta;
    1.0 / 6.0 - t2 / 120.0 + t2 * t2 / 5040.0
}

pub(crate) fn taylor_inverse_left_jacobian_coef(theta: f64) -> f64 {
    let t2 = theta * theta;
    1.0 / 12.0 + t2 / 720.0 + t2 * t2 / 30240.0
}

pub(crate) fn taylor_theta_over_sin_theta(theta: f64) -> f64 {
    let t2 = theta * theta;
    1.0 + t2 / 6.0 + 7.0 * t2 * t2 / 360.0
}

/// Five terms rather than three, because the branch runs out to θ = 0.5
/// and three would leave 1e-9 there.
pub(crate) fn taylor_q_second_coef(theta: f64) -> f64 {
    let t2 = theta * theta;
    1.0 / 24.0 - t2 / 720.0 + t2 * t2 / 40_320.0 - t2 * t2 * t2 / 3_628_800.0
        + t2 * t2 * t2 * t2 / 479_001_600.0
}

pub(crate) fn taylor_q_third_coef(theta: f64) -> f64 {
    let t2 = theta * theta;
    1.0 / 120.0 - t2 / 2_520.0 + t2 * t2 / 120_960.0 - t2 * t2 * t2 / 9_979_200.0
        + t2 * t2 * t2 * t2 / 1_245_404_160.0
}

// --- trigonometric forms ----------------------------------------------------

/// Written through `1 − cos θ = 2 sin²(θ/2)`. The identity removes the
/// cancellation that the direct `(1.0 - theta.cos())` would suffer.
pub(crate) fn trig_one_minus_cos_over_theta_sq(theta: f64) -> f64 {
    let half_sin = (0.5 * theta).sin();
    2.0 * half_sin * half_sin / (theta * theta)
}

/// Written through the half-angle identity
/// `(1 + cos θ) / (2θ sin θ) = 1 / (2θ·tan(θ/2))`.
///
/// The direct form collapses as θ → π: there `cos θ ≈ −1`, and the sum
/// `1 + cos θ` loses its significant digits. At θ = π − 1e-6 the absolute
/// error of the coefficient reaches 3e-11 which, multiplied by `(φ^)²` of
/// norm about 14, pushes the `SE(3)` logarithm far outside a 1e-12
/// tolerance.
///
/// The tangent form has no cancellation at all: as θ → π, tan(θ/2) → ∞ and
/// the subtracted term goes smoothly to zero.
pub(crate) fn trig_inverse_left_jacobian_coef(theta: f64) -> f64 {
    1.0 / (theta * theta) - 1.0 / (2.0 * theta * (0.5 * theta).tan())
}

/// Written through `2 − 2 cos θ = 4 sin²(θ/2)`, for the same reason
/// `trig_one_minus_cos_over_theta_sq` is: the direct `2.0 * theta.cos() -
/// 2.0` loses its digits exactly where this coefficient is used.
pub(crate) fn trig_q_second_coef(theta: f64) -> f64 {
    let half_sin = (0.5 * theta).sin();
    let t2 = theta * theta;
    (t2 - 4.0 * half_sin * half_sin) / (2.0 * t2 * t2)
}

pub(crate) fn trig_q_third_coef(theta: f64) -> f64 {
    let t2 = theta * theta;
    (2.0 * theta - 3.0 * theta.sin() + theta * theta.cos()) / (2.0 * t2 * t2 * theta)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The naive form, kept only to check the algebra of the identity.
    fn naive_inverse_left_jacobian_coef(theta: f64) -> f64 {
        1.0 / (theta * theta) - (1.0 + theta.cos()) / (2.0 * theta * theta.sin())
    }

    /// Away from π both forms are accurate. A discrepancy would mean an
    /// error in the half-angle identity itself, not in conditioning.
    #[test]
    fn half_angle_identity_is_algebraically_correct() {
        for theta in [0.5, 1.0, 1.5, 2.0, 2.5] {
            let stable = trig_inverse_left_jacobian_coef(theta);
            let naive = naive_inverse_left_jacobian_coef(theta);
            assert!(
                (stable - naive).abs() < 1e-14,
                "θ = {theta}: stable {stable:.17e} vs naive {naive:.17e}"
            );
        }
    }

    /// At the crossover the two branches must agree; otherwise the
    /// threshold is wrong and the spectrum acquires a discontinuity.
    #[test]
    fn branches_agree_at_threshold() {
        let theta = THETA_SMALL;
        let pairs: [(&str, f64, f64); 5] = [
            (
                "sin θ / θ",
                taylor_sin_over_theta(theta),
                theta.sin() / theta,
            ),
            (
                "(1 − cos θ) / θ²",
                taylor_one_minus_cos_over_theta_sq(theta),
                trig_one_minus_cos_over_theta_sq(theta),
            ),
            (
                "(θ − sin θ) / θ³",
                taylor_theta_minus_sin_over_theta_cubed(theta),
                (theta - theta.sin()) / theta.powi(3),
            ),
            (
                "J⁻¹ coefficient",
                taylor_inverse_left_jacobian_coef(theta),
                trig_inverse_left_jacobian_coef(theta),
            ),
            (
                "θ / sin θ",
                taylor_theta_over_sin_theta(theta),
                theta / theta.sin(),
            ),
        ];
        for (name, taylor, trig) in pairs {
            assert!(
                (taylor - trig).abs() < 1e-10,
                "{name}: series {taylor:.17e} vs trigonometry {trig:.17e}, \
                 difference {:.3e}",
                (taylor - trig).abs()
            );
        }
    }

    /// The limits as θ → 0 are known analytically.
    #[test]
    fn limits_at_zero() {
        assert_eq!(sin_over_theta(0.0), 1.0);
        assert_eq!(one_minus_cos_over_theta_sq(0.0), 0.5);
        assert_eq!(theta_minus_sin_over_theta_cubed(0.0), 1.0 / 6.0);
        assert_eq!(inverse_left_jacobian_coef(0.0), 1.0 / 12.0);
        assert_eq!(theta_over_sin_theta(0.0), 1.0);
    }

    /// The coefficients stay continuous across every scale of θ, including
    /// subnormal ones.
    #[test]
    fn continuous_across_scales() {
        for theta in [0.0, 1e-12, 1e-8, 1e-4, 1e-3] {
            assert!((sin_over_theta(theta) - 1.0).abs() < 1e-6);
            assert!((one_minus_cos_over_theta_sq(theta) - 0.5).abs() < 1e-6);
            assert!((theta_minus_sin_over_theta_cubed(theta) - 1.0 / 6.0).abs() < 1e-6);
            assert!((inverse_left_jacobian_coef(theta) - 1.0 / 12.0).abs() < 1e-6);
            assert!((theta_over_sin_theta(theta) - 1.0).abs() < 1e-6);
        }
    }
}
