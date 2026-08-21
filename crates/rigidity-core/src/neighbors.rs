//! Neighbour search: the contract, and a brute-force reference implementation.

use nalgebra::Vector3;

use crate::cloud::PointCloud;

/// A neighbour that was found.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neighbor {
    /// Index of the point in the source cloud.
    pub index: u32,
    /// Squared distance to the query. Squared rather than plain: the
    /// square root is never needed for ranking neighbours, and taking it
    /// costs cycles and adds rounding.
    pub distance_squared: f64,
}

/// The order neighbours are returned in: by increasing distance, ties
/// broken by increasing index.
///
/// Breaking ties by index is mandatory. Without it two equally correct
/// implementations return different orders for equidistant points, and
/// further up the stack that turns into a different order of summation.
pub fn compare_neighbors(a: &Neighbor, b: &Neighbor) -> std::cmp::Ordering {
    a.distance_squared
        .total_cmp(&b.distance_squared)
        .then(a.index.cmp(&b.index))
}

/// Nearest-neighbour search.
///
/// Implementations must honour the order defined by [`compare_neighbors`].
pub trait NeighborSearch {
    /// The `k` nearest points to `query`, written into `out`, whose
    /// previous contents are discarded.
    ///
    /// The buffer is supplied by the caller so that the ICP hot loop does
    /// not allocate once per point.
    fn knn_into(&self, query: &Vector3<f64>, k: usize, out: &mut Vec<Neighbor>);

    /// Every point inside the ball of radius `radius`, written into `out`.
    fn radius_into(&self, query: &Vector3<f64>, radius: f64, out: &mut Vec<Neighbor>);

    /// Allocating convenience wrapper over [`knn_into`](Self::knn_into).
    fn knn(&self, query: &Vector3<f64>, k: usize) -> Vec<Neighbor> {
        let mut out = Vec::with_capacity(k);
        self.knn_into(query, k, &mut out);
        out
    }

    /// Allocating convenience wrapper over [`radius_into`](Self::radius_into).
    fn radius(&self, query: &Vector3<f64>, radius: f64) -> Vec<Neighbor> {
        let mut out = Vec::new();
        self.radius_into(query, radius, &mut out);
        out
    }
}

/// Brute force.
///
/// It exists as the reference the kd-tree is checked against. There is no
/// reason to optimise it; it never sits in the hot path.
pub struct BruteForce<'a> {
    cloud: &'a PointCloud,
}

impl<'a> BruteForce<'a> {
    /// Wraps a cloud.
    pub fn new(cloud: &'a PointCloud) -> Self {
        Self { cloud }
    }

    fn all_distances(&self, query: &Vector3<f64>, out: &mut Vec<Neighbor>) {
        out.clear();
        out.reserve(self.cloud.len());
        for i in 0..self.cloud.len() {
            out.push(Neighbor {
                index: i as u32,
                distance_squared: (self.cloud.point(i) - query).norm_squared(),
            });
        }
    }
}

impl NeighborSearch for BruteForce<'_> {
    fn knn_into(&self, query: &Vector3<f64>, k: usize, out: &mut Vec<Neighbor>) {
        self.all_distances(query, out);
        out.sort_unstable_by(compare_neighbors);
        out.truncate(k);
    }

    fn radius_into(&self, query: &Vector3<f64>, radius: f64, out: &mut Vec<Neighbor>) {
        let limit = radius * radius;
        self.all_distances(query, out);
        out.retain(|n| n.distance_squared <= limit);
        out.sort_unstable_by(compare_neighbors);
    }
}
