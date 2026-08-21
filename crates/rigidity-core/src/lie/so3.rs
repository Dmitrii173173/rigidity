//! The rotation group `SO(3)` and its Lie algebra `so(3)`.

use nalgebra::{Matrix3, Vector3};
use std::f64::consts::PI;

use super::series;

/// A rotation of three-dimensional space.
///
/// Stored as a 3×3 matrix. Every constructor except
/// [`from_matrix_unchecked`](So3::from_matrix_unchecked) guarantees
/// orthogonality to within rounding error.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct So3 {
    matrix: Matrix3<f64>,
}

impl So3 {
    /// The identity rotation.
    pub fn identity() -> Self {
        Self {
            matrix: Matrix3::identity(),
        }
    }

    /// Wraps a matrix without checking orthogonality.
    ///
    /// The caller is responsible for the matrix belonging to `SO(3)`.
    pub fn from_matrix_unchecked(matrix: Matrix3<f64>) -> Self {
        Self { matrix }
    }

    /// The rotation matrix.
    pub fn matrix(&self) -> &Matrix3<f64> {
        &self.matrix
    }

    /// The exponential map `so(3) → SO(3)`, i.e. the Rodrigues formula.
    ///
    /// `exp(φ) = I + (sin θ / θ) φ^ + ((1 − cos θ) / θ²) (φ^)²` with
    /// `θ = |φ|`. For small θ the coefficients come from Taylor series,
    /// because the trigonometric form suffers catastrophic cancellation
    /// there.
    pub fn exp(phi: &Vector3<f64>) -> Self {
        let theta = phi.norm();
        let hat_phi = hat(phi);
        let matrix = Matrix3::identity()
            + hat_phi * series::sin_over_theta(theta)
            + hat_phi * hat_phi * series::one_minus_cos_over_theta_sq(theta);
        Self { matrix }
    }

    /// The logarithm `SO(3) → so(3)`; the result has norm in `[0, π]`.
    ///
    /// Three branches:
    /// 1. θ < `THETA_SMALL` — the series for `θ / sin θ`;
    /// 2. the middle range — the direct formula via the skew part;
    /// 3. θ near π — the axis is recovered from the symmetric part, since
    ///    the skew part degenerates there: `R − Rᵀ → 0`.
    ///
    /// At θ = π the result is defined only up to sign: `exp(πa)` and
    /// `exp(−πa)` are the same matrix.
    pub fn log(&self) -> Vector3<f64> {
        let m = &self.matrix;
        // vee(R − Rᵀ)/2 = sin(θ)·a
        let skew = Vector3::new(
            m[(2, 1)] - m[(1, 2)],
            m[(0, 2)] - m[(2, 0)],
            m[(1, 0)] - m[(0, 1)],
        ) * 0.5;

        let sin_theta = skew.norm();
        let cos_theta = 0.5 * (m.trace() - 1.0);
        // atan2 is stable across the whole range, unlike acos near ±1.
        let theta = sin_theta.atan2(cos_theta);

        if theta < series::THETA_SMALL {
            skew * series::theta_over_sin_theta(theta)
        } else if theta < PI - series::THETA_NEAR_PI {
            skew * (theta / sin_theta)
        } else {
            self.log_near_pi(theta, cos_theta, &skew)
        }
    }

    /// The θ ≈ π branch: `R + Rᵀ = 2cos θ·I + 2(1 − cos θ)·a aᵀ` yields the
    /// outer product of the axis, and from it the axis itself.
    fn log_near_pi(&self, theta: f64, cos_theta: f64, skew: &Vector3<f64>) -> Vector3<f64> {
        let m = &self.matrix;
        let outer = (m + m.transpose() - Matrix3::identity() * (2.0 * cos_theta))
            / (2.0 * (1.0 - cos_theta));

        // The column with the largest diagonal entry: its norm is at least
        // 1/√3, so dividing by it is safe.
        let mut best = 0;
        for i in 1..3 {
            if outer[(i, i)] > outer[(best, best)] {
                best = i;
            }
        }
        let mut axis = outer.column(best) / outer[(best, best)].max(0.0).sqrt();

        // The sign comes from the skew part. At θ = π that part degenerates,
        // but there both signs produce the same matrix.
        if axis.dot(skew) < 0.0 {
            axis = -axis;
        }
        axis * theta
    }

    /// The inverse rotation.
    pub fn inverse(&self) -> Self {
        Self {
            matrix: self.matrix.transpose(),
        }
    }

    /// The adjoint representation. For `SO(3)` it is the matrix itself.
    pub fn adjoint(&self) -> Matrix3<f64> {
        self.matrix
    }
}

impl std::ops::Mul for So3 {
    type Output = So3;
    fn mul(self, rhs: So3) -> So3 {
        So3 {
            matrix: self.matrix * rhs.matrix,
        }
    }
}

impl std::ops::Mul<Vector3<f64>> for So3 {
    type Output = Vector3<f64>;
    fn mul(self, rhs: Vector3<f64>) -> Vector3<f64> {
        self.matrix * rhs
    }
}

impl Default for So3 {
    fn default() -> Self {
        Self::identity()
    }
}

/// The skew-symmetric matrix of a vector: `hat(v) w = v × w`.
pub fn hat(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// The inverse of [`hat`]: recovers the vector from a skew-symmetric matrix.
pub fn vee(m: &Matrix3<f64>) -> Vector3<f64> {
    Vector3::new(m[(2, 1)], m[(0, 2)], m[(1, 0)])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hat_is_cross_product() {
        let a = Vector3::new(1.0, 2.0, 3.0);
        let b = Vector3::new(-4.0, 5.0, 6.0);
        assert!((hat(&a) * b - a.cross(&b)).norm() < 1e-15);
    }

    #[test]
    fn vee_inverts_hat() {
        let v = Vector3::new(0.3, -1.7, 2.2);
        assert!((vee(&hat(&v)) - v).norm() < 1e-15);
    }

    /// A dedicated check of the θ ≈ π branch. That branch is the one most
    /// often implemented wrongly, and it fails by quietly returning an
    /// almost-correct answer.
    #[test]
    fn log_near_pi_is_exact() {
        for offset in [0.0, 1e-12, 1e-8, 1e-4, 5e-3] {
            let theta = PI - offset;
            let axis = Vector3::new(1.0, -2.0, 0.5).normalize();
            let phi = axis * theta;
            let recovered = So3::exp(&phi).log();
            // At θ = π the sign is undefined, so compare up to it.
            let err = (recovered - phi).norm().min((recovered + phi).norm());
            assert!(err < 1e-12, "θ = π − {offset:e}: error {err:.3e}");
        }
    }

    /// A rotation by exactly π.
    #[test]
    fn log_at_exactly_pi() {
        let axis = Vector3::new(0.0, 0.0, 1.0);
        let recovered = So3::exp(&(axis * PI)).log();
        assert!((recovered.norm() - PI).abs() < 1e-12);
        assert!(recovered.normalize().cross(&axis).norm() < 1e-12);
    }
}
