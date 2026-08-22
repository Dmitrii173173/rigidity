//! Absolute orientation: the rigid motion that best carries one set of
//! points onto another.
//!
//! Given correspondences that are *known* rather than searched for — a
//! person clicking the same corner in two scans — the answer is a closed
//! form, not an iteration. Centre both sets, take the singular value
//! decomposition of their correlation, and the rotation falls out. Horn
//! (1987) and Kabsch (1976) reached it independently and it has not needed
//! improving since.
//!
//! This is what stands between a registration and a wrong local minimum.
//! ICP started from the identity on a pair thirty degrees apart converges
//! confidently to nonsense, and no amount of conditioning analysis will
//! say so — the spectrum describes the shape of the cost function around
//! wherever the solver stopped, not whether it stopped in the right place.
//! Three clicked pairs put it in the right basin, and then the
//! conditioning report means what it says.

use nalgebra::{Matrix3, Vector3};

use crate::lie::{Se3, So3};

/// The rigid motion carrying `from` onto `to`, in the least-squares sense.
///
/// Returns `None` when there is nothing to solve: fewer than three pairs,
/// mismatched lengths, or points so nearly collinear that the rotation
/// about their common line is not determined. That last case is not a
/// numerical accident — three points on a line genuinely do not fix a
/// rigid motion, and answering anyway would be inventing the missing
/// degree of freedom.
///
/// # Reflections
///
/// The decomposition can produce an improper rotation — a reflection —
/// when the points are coplanar and the noise happens to favour it.
/// Flipping the sign of the least significant singular direction gives
/// the best *proper* rotation instead. Without that correction the result
/// mirrors the cloud, which looks almost right and is entirely wrong.
pub fn absolute_orientation(from: &[Vector3<f64>], to: &[Vector3<f64>]) -> Option<Se3> {
    /// How much smaller the second singular value may be than the first
    /// before the points count as collinear.
    const DEGENERATE: f64 = 1e-8;

    if from.len() < 3 || from.len() != to.len() {
        return None;
    }

    let count = from.len() as f64;
    let centre = |points: &[Vector3<f64>]| points.iter().sum::<Vector3<f64>>() / count;
    let (from_centre, to_centre) = (centre(from), centre(to));

    let mut correlation = Matrix3::zeros();
    for (a, b) in from.iter().zip(to) {
        correlation += (b - to_centre) * (a - from_centre).transpose();
    }

    let svd = correlation.svd(true, true);
    let (u, v_t) = (svd.u?, svd.v_t?);
    let values = svd.singular_values;
    if values[0] <= 0.0 || values[1] <= values[0] * DEGENERATE {
        return None;
    }

    let mut rotation = u * v_t;
    if rotation.determinant() < 0.0 {
        // The best orthogonal matrix here is a reflection. The best
        // *rotation* is what you get by flipping the direction that
        // contributed least, which is the last column.
        let mut flip = Matrix3::identity();
        flip[(2, 2)] = -1.0;
        rotation = u * flip * v_t;
    }

    let rotation = So3::from_matrix_unchecked(rotation);
    let translation = to_centre - rotation.matrix() * from_centre;
    Some(Se3::from_parts(rotation, translation))
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::Vector6;

    use super::*;

    /// Points spread over three dimensions, so nothing is degenerate by
    /// accident.
    fn cloud() -> Vec<Vector3<f64>> {
        vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.7, -0.3, 0.2),
        ]
    }

    /// A known motion is recovered exactly from exact correspondences.
    #[test]
    fn a_known_motion_comes_back() {
        let truth = Se3::exp(&Vector6::new(0.4, -0.2, 0.1, 0.3, -0.5, 0.9));
        let from = cloud();
        let to: Vec<_> = from.iter().map(|p| truth.transform_point(p)).collect();

        let found = absolute_orientation(&from, &to).expect("three points are enough");
        assert_relative_eq!(found.matrix(), truth.matrix(), epsilon = 1e-12);
    }

    /// Three pairs are the fewest that fix a motion, and they suffice.
    #[test]
    fn three_pairs_are_enough() {
        let truth = Se3::exp(&Vector6::new(-1.0, 2.0, 0.5, 0.0, 0.0, 1.2));
        let from = cloud()[..3].to_vec();
        let to: Vec<_> = from.iter().map(|p| truth.transform_point(p)).collect();

        let found = absolute_orientation(&from, &to).expect("three points are enough");
        assert_relative_eq!(found.matrix(), truth.matrix(), epsilon = 1e-12);
    }

    /// Two pairs are not, and the answer is that rather than a guess.
    #[test]
    fn two_pairs_are_not_enough() {
        let from = cloud()[..2].to_vec();
        let to = from.clone();
        assert!(absolute_orientation(&from, &to).is_none());
    }

    /// Neither are three on a line: the rotation about it is free, and
    /// inventing a value for it would be worse than saying so.
    #[test]
    fn collinear_points_determine_nothing() {
        let from = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(2.0, 0.0, 0.0),
            Vector3::new(3.0, 0.0, 0.0),
        ];
        let to = from.clone();
        assert!(absolute_orientation(&from, &to).is_none());
    }

    /// Coplanar points still give a rotation and never a reflection.
    ///
    /// This is the case that produces one if the determinant is not
    /// checked: the mirrored answer fits the points exactly and is not the
    /// motion that happened.
    #[test]
    fn a_coplanar_set_does_not_come_back_mirrored() {
        let truth = Se3::exp(&Vector6::new(0.1, 0.2, -0.3, 0.0, 0.0, 2.0));
        let from = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(1.0, 1.0, 0.0),
        ];
        let to: Vec<_> = from.iter().map(|p| truth.transform_point(p)).collect();

        let found = absolute_orientation(&from, &to).expect("four coplanar points are enough");
        assert_relative_eq!(
            found.rotation().matrix().determinant(),
            1.0,
            epsilon = 1e-12
        );
        assert_relative_eq!(found.matrix(), truth.matrix(), epsilon = 1e-12);
    }

    /// Noise on the correspondences moves the answer a little and not a lot.
    #[test]
    fn noise_perturbs_the_answer_in_proportion() {
        let truth = Se3::exp(&Vector6::new(0.5, 0.0, 0.0, 0.0, 0.4, 0.0));
        let from = cloud();
        let mut to: Vec<_> = from.iter().map(|p| truth.transform_point(p)).collect();
        // A millimetre on one point of a metre-sized set.
        to[2] += Vector3::new(0.001, -0.001, 0.001);

        let found = absolute_orientation(&from, &to).expect("three points are enough");
        let error = (found * truth.inverse()).log().norm();
        assert!(error < 0.01, "a millimetre moved the answer by {error}");
    }
}
