//! The group of rigid motions `SE(3)` and its Lie algebra `se(3)`.
//!
//! Algebra vectors are ordered `ξ = [ρ; φ]`: translational part first,
//! rotational part second. The same order indexes the rows and columns of
//! the adjoint and, further up the stack, the degrees of freedom in the
//! conditioning report.

use nalgebra::{Matrix3, Matrix4, Matrix6, Vector3, Vector6};

use super::jacobian::{inverse_left_jacobian_so3, left_jacobian_so3};
use super::so3::{So3, hat};

/// A rigid motion: a rotation and a translation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Se3 {
    rotation: So3,
    translation: Vector3<f64>,
}

impl Se3 {
    /// The identity transform.
    pub fn identity() -> Self {
        Self {
            rotation: So3::identity(),
            translation: Vector3::zeros(),
        }
    }

    /// Builds a transform from a rotation and a translation.
    pub fn from_parts(rotation: So3, translation: Vector3<f64>) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    /// Builds a transform from a homogeneous 4×4 matrix.
    ///
    /// The rotational block is not checked for orthogonality. This exists
    /// for reading poses out of files, where the matrix is already given
    /// and vouched for by the data source rather than by us.
    pub fn from_matrix_unchecked(matrix: Matrix4<f64>) -> Self {
        Self {
            rotation: So3::from_matrix_unchecked(matrix.fixed_view::<3, 3>(0, 0).into()),
            translation: Vector3::new(matrix[(0, 3)], matrix[(1, 3)], matrix[(2, 3)]),
        }
    }

    /// The rotational part.
    pub fn rotation(&self) -> &So3 {
        &self.rotation
    }

    /// The translational part.
    pub fn translation(&self) -> &Vector3<f64> {
        &self.translation
    }

    /// The homogeneous 4×4 matrix.
    pub fn matrix(&self) -> Matrix4<f64> {
        let mut m = Matrix4::identity();
        m.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(self.rotation.matrix());
        m.fixed_view_mut::<3, 1>(0, 3).copy_from(&self.translation);
        m
    }

    /// The exponential map `se(3) → SE(3)`.
    ///
    /// `exp([ρ; φ]) = (exp(φ), J_l(φ)·ρ)`. The translation passes through
    /// the left Jacobian; the naive `t = ρ` is only a first-order truth.
    pub fn exp(xi: &Vector6<f64>) -> Self {
        let rho = Vector3::new(xi[0], xi[1], xi[2]);
        let phi = Vector3::new(xi[3], xi[4], xi[5]);
        Self {
            rotation: So3::exp(&phi),
            translation: left_jacobian_so3(&phi) * rho,
        }
    }

    /// The logarithm `SE(3) → se(3)`.
    pub fn log(&self) -> Vector6<f64> {
        let phi = self.rotation.log();
        let rho = inverse_left_jacobian_so3(&phi) * self.translation;
        Vector6::new(rho[0], rho[1], rho[2], phi[0], phi[1], phi[2])
    }

    /// The inverse transform.
    pub fn inverse(&self) -> Self {
        let inv_rotation = self.rotation.inverse();
        Self {
            rotation: inv_rotation,
            translation: -(inv_rotation * self.translation),
        }
    }

    /// The 6×6 adjoint representation.
    ///
    /// `Adj(T) = [[R, t^ R], [0, R]]` in the `ξ = [ρ; φ]` ordering. Its
    /// defining property is `Adj(T)·ξ = vee(T · ξ^ · T⁻¹)`.
    pub fn adjoint(&self) -> Matrix6<f64> {
        let r = *self.rotation.matrix();
        let t_hat = hat(&self.translation);
        let mut adj = Matrix6::zeros();
        adj.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
        adj.fixed_view_mut::<3, 3>(0, 3).copy_from(&(t_hat * r));
        adj.fixed_view_mut::<3, 3>(3, 3).copy_from(&r);
        adj
    }

    /// Applies the transform to a point.
    pub fn transform_point(&self, p: &Vector3<f64>) -> Vector3<f64> {
        self.rotation * *p + self.translation
    }
}

impl std::ops::Mul for Se3 {
    type Output = Se3;
    fn mul(self, rhs: Se3) -> Se3 {
        Se3 {
            rotation: self.rotation * rhs.rotation,
            translation: self.rotation * rhs.translation + self.translation,
        }
    }
}

impl Default for Se3 {
    fn default() -> Self {
        Self::identity()
    }
}

/// The embedding of `se(3)` into 4×4 matrices: `ξ^ = [[φ^, ρ], [0, 0]]`.
pub fn hat_se3(xi: &Vector6<f64>) -> Matrix4<f64> {
    let phi = Vector3::new(xi[3], xi[4], xi[5]);
    let mut m = Matrix4::zeros();
    m.fixed_view_mut::<3, 3>(0, 0).copy_from(&hat(&phi));
    m[(0, 3)] = xi[0];
    m[(1, 3)] = xi[1];
    m[(2, 3)] = xi[2];
    m
}

/// The inverse of [`hat_se3`].
pub fn vee_se3(m: &Matrix4<f64>) -> Vector6<f64> {
    let rot: Matrix3<f64> = m.fixed_view::<3, 3>(0, 0).into();
    Vector6::new(
        m[(0, 3)],
        m[(1, 3)],
        m[(2, 3)],
        rot[(2, 1)],
        rot[(0, 2)],
        rot[(1, 0)],
    )
}
