//! The point-to-plane residual and its Jacobian.

use nalgebra::{Vector3, Vector6};

/// One row of the point-to-plane constraint Jacobian.
///
/// The residual is `e = nᵀ(R·p + t − q)`. Under the left perturbation
/// `T ← exp(ξ)·T` the point moves by `δp = ρ + φ × p`, hence
///
/// ```text
/// de/dξ = nᵀ(ρ + φ × p) = nᵀρ + (p × n)ᵀφ
/// ```
///
/// so the row is `[nᵀ | (p × n)ᵀ]` in the `ξ = [ρ; φ]` ordering. This uses
/// the identity `n·(φ × p) = φ·(p × n)`.
///
/// # Centre of rotation
///
/// The `p × n` block is taken about the coordinate origin. The convention
/// matters: a null space expressed in `[ρ; φ]` depends on which point the
/// rotation is taken about. Moving the scene by a transform `T` maps the
/// null space to itself under
/// `Adj(T)`.
pub fn point_to_plane_row(point: &Vector3<f64>, normal: &Vector3<f64>) -> Vector6<f64> {
    let moment = point.cross(normal);
    Vector6::new(normal.x, normal.y, normal.z, moment.x, moment.y, moment.z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lie::Se3;

    /// The row must match the numerical derivative of the residual with
    /// respect to a perturbation in the algebra.
    #[test]
    fn row_matches_numeric_derivative() {
        let point = Vector3::new(1.3, -0.7, 2.1);
        let normal = Vector3::new(0.3, 0.5, -0.8).normalize();
        let analytic = point_to_plane_row(&point, &normal);

        const H: f64 = 1e-6;
        for axis in 0..6 {
            let mut delta = Vector6::zeros();
            delta[axis] = H;
            let forward = normal.dot(&(Se3::exp(&delta).transform_point(&point) - point));
            let backward = normal.dot(&(Se3::exp(&(-delta)).transform_point(&point) - point));
            let numeric = (forward - backward) / (2.0 * H);
            assert!(
                (analytic[axis] - numeric).abs() < 1e-8,
                "axis {axis}: analytic {}, numeric {numeric}",
                analytic[axis]
            );
        }
    }
}
