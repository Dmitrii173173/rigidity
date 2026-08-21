//! The kd-tree must give exactly the same answer as brute force.
//!
//! Brute force is the reference: it is trivially correct. Any divergence
//! in indices means a bug in the tree, not a "different but equally
//! valid" result.

use nalgebra::Vector3;
use rigidity_core::{BruteForce, NeighborSearch, PointCloud};
use rigidity_spatial::KdTree;

/// A linear congruential generator: reproducibility matters more than
/// distribution quality here, and an extra dependency on `rand` is not
/// worth it.
struct Lcg(u64);

impl Lcg {
    fn next_unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn next_point(&mut self, half_extent: f64) -> Vector3<f64> {
        Vector3::new(
            (self.next_unit() * 2.0 - 1.0) * half_extent,
            (self.next_unit() * 2.0 - 1.0) * half_extent,
            (self.next_unit() * 2.0 - 1.0) * half_extent,
        )
    }
}

fn random_cloud(count: usize, seed: u64) -> PointCloud {
    let mut rng = Lcg(seed);
    let mut cloud = PointCloud::with_capacity(count);
    for _ in 0..count {
        cloud.push(rng.next_point(50.0));
    }
    cloud
}

#[test]
fn knn_matches_brute_force_on_10k_points() {
    let cloud = random_cloud(10_000, 0x5EED);
    let tree = KdTree::build(&cloud).unwrap();
    let brute = BruteForce::new(&cloud);
    let mut queries = Lcg(0xC0FFEE);

    for _ in 0..200 {
        let query = queries.next_point(55.0);
        for k in [1usize, 8, 32] {
            let from_tree = tree.knn(&query, k);
            let from_brute = brute.knn(&query, k);
            assert_eq!(from_tree.len(), from_brute.len(), "k = {k}");
            for (a, b) in from_tree.iter().zip(from_brute.iter()) {
                assert_eq!(a.index, b.index, "k = {k}: indices diverged");
                assert_eq!(
                    a.distance_squared, b.distance_squared,
                    "k = {k}: distances diverged"
                );
            }
        }
    }
}

#[test]
fn radius_matches_brute_force() {
    let cloud = random_cloud(5_000, 0xBEEF);
    let tree = KdTree::build(&cloud).unwrap();
    let brute = BruteForce::new(&cloud);
    let mut queries = Lcg(0xFACE);

    let mut total = 0usize;
    for _ in 0..100 {
        let query = queries.next_point(50.0);
        for radius in [1.0, 5.0, 15.0] {
            let from_tree = tree.radius(&query, radius);
            let from_brute = brute.radius(&query, radius);
            assert_eq!(
                from_tree.len(),
                from_brute.len(),
                "radius = {radius}: different neighbour counts"
            );
            for (a, b) in from_tree.iter().zip(from_brute.iter()) {
                assert_eq!(a.index, b.index);
                assert_eq!(a.distance_squared, b.distance_squared);
            }
            total += from_tree.len();
        }
    }
    // The test is meaningless if the queries find nothing.
    assert!(total > 1_000, "the sample is too sparse: found {total}");
}

#[test]
fn degenerate_queries() {
    let cloud = random_cloud(64, 1);
    let tree = KdTree::build(&cloud).unwrap();

    assert!(tree.knn(&Vector3::zeros(), 0).is_empty());
    // k larger than the point count: return everything, without panicking.
    assert_eq!(tree.knn(&Vector3::zeros(), 1000).len(), 64);
    assert!(tree.radius(&Vector3::new(1e6, 1e6, 1e6), 1.0).is_empty());

    let empty = KdTree::build(&PointCloud::new()).unwrap();
    assert!(empty.is_empty());
    assert!(empty.knn(&Vector3::zeros(), 5).is_empty());
}
