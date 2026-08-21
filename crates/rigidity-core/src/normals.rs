//! Normal estimation by principal component analysis.

use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;

use crate::cloud::PointCloud;
use crate::neighbors::NeighborSearch;

/// Estimates the normal at every point from its `k` nearest neighbours.
///
/// The covariance of the neighbourhood is decomposed into eigenvectors;
/// the normal is the eigenvector of the smallest eigenvalue, that is, the
/// direction of least spread.
///
/// # Sign
///
/// Normals are not consistently oriented across points: consistent
/// orientation requires propagation over the neighbourhood graph and is
/// beside the point here. For a point-to-plane constraint the sign is
/// irrelevant — the residual enters squared, and flipping `n` in both the
/// Jacobian row and the residual leaves `JᵀWJ` and `JᵀWe` unchanged.
///
/// The sign is nevertheless fixed deterministically, by the component of
/// largest magnitude, so that results stay reproducible.
///
/// # Determinism
///
/// Points are independent, output order is preserved, and each
/// neighbourhood's covariance accumulates in order of increasing distance
/// as guaranteed by the [`NeighborSearch`] contract.
pub fn estimate_normals<S>(cloud: &PointCloud, search: &S, k: usize) -> Vec<Vector3<f64>>
where
    S: NeighborSearch + Sync,
{
    assert!(k >= 3, "normal estimation needs at least three neighbours");
    (0..cloud.len())
        .into_par_iter()
        .map(|index| {
            let query = cloud.point(index);
            let neighbours = search.knn(&query, k);
            if neighbours.len() < 3 {
                return Vector3::z();
            }

            let mut centroid = Vector3::zeros();
            for neighbour in &neighbours {
                centroid += cloud.point(neighbour.index as usize);
            }
            centroid /= neighbours.len() as f64;

            let mut covariance = Matrix3::zeros();
            for neighbour in &neighbours {
                let delta = cloud.point(neighbour.index as usize) - centroid;
                covariance += delta * delta.transpose();
            }

            let eigen = nalgebra::SymmetricEigen::new(covariance);
            let mut smallest = 0;
            for axis in 1..3 {
                if eigen.eigenvalues[axis] < eigen.eigenvalues[smallest] {
                    smallest = axis;
                }
            }
            let normal: Vector3<f64> = eigen.eigenvectors.column(smallest).into();

            // Deterministic sign.
            let mut dominant = 0;
            for axis in 1..3 {
                if normal[axis].abs() > normal[dominant].abs() {
                    dominant = axis;
                }
            }
            if normal[dominant] < 0.0 {
                -normal
            } else {
                normal
            }
        })
        .collect()
}
