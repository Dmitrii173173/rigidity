//! Spatial indices.
//!
//! This crate is separate from the core because the core's direct
//! dependencies are fixed and checked in CI: `kiddo` does not fit there,
//! and file I/O is no place for a data structure.
//!
//! The core defines the [`NeighborSearch`] contract and a brute-force
//! reference implementation; the working one lives here. Drawing the
//! boundary at a trait rather than a type leaves room for a grid-based or
//! GPU search later without touching ICP.

use kiddo::{ImmutableKdTree, SquaredEuclidean};
use nalgebra::Vector3;
use rigidity_core::{Neighbor, NeighborSearch, PointCloud, neighbors::compare_neighbors};
use std::num::NonZeroUsize;

/// An error while building the index.
#[derive(Debug, thiserror::Error)]
pub enum SpatialError {
    /// Kiddo failed to build the tree.
    #[error("kd-tree construction failed: {0}")]
    Construction(String),
}

/// A kd-tree built over a cloud.
///
/// The tree is built in **absolute** coordinates, the same ones
/// [`PointCloud::point`] returns. That makes distances bit-for-bit
/// comparable with the brute-force reference; otherwise the agreement test
/// would need a tolerance and would stop catching indexing mistakes.
pub struct KdTree {
    tree: ImmutableKdTree<f64, 3>,
    len: usize,
}

impl KdTree {
    /// Builds a tree over a cloud.
    pub fn build(cloud: &PointCloud) -> Result<Self, SpatialError> {
        Self::build_observed(cloud, |_, _| {})
    }

    /// The same, reporting progress after each internal phase.
    ///
    /// `progress` receives `(completed, total)` in phases: the
    /// coordinates are extracted, then the tree is built. Two steps is all
    /// the resolution there is — `kiddo` builds the tree in one opaque
    /// call, and on a large cloud that call is the larger half of the
    /// wait. Pretending otherwise would mean a progress bar that lies.
    pub fn build_observed<F>(cloud: &PointCloud, mut progress: F) -> Result<Self, SpatialError>
    where
        F: FnMut(usize, usize),
    {
        /// Coordinates, then construction.
        const PHASES: usize = 2;

        let entries: Vec<[f64; 3]> = (0..cloud.len())
            .map(|i| {
                let p = cloud.point(i);
                [p.x, p.y, p.z]
            })
            .collect();
        progress(1, PHASES);

        let tree = ImmutableKdTree::new_from_slice(&entries)
            .map_err(|e| SpatialError::Construction(e.to_string()))?;
        progress(2, PHASES);

        Ok(Self {
            tree,
            len: cloud.len(),
        })
    }

    /// Number of indexed points.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl NeighborSearch for KdTree {
    fn knn_into(&self, query: &Vector3<f64>, k: usize, out: &mut Vec<Neighbor>) {
        out.clear();
        let Some(count) = NonZeroUsize::new(k.min(self.len)) else {
            return;
        };
        let point = [query.x, query.y, query.z];

        // Fast path: ICP asks for exactly one neighbour, while `nearest_n`
        // returns a `Vec` and therefore allocates per query. On a million
        // points that is a million allocator calls per iteration.
        // `nearest_one` returns a single value instead.
        if count.get() == 1 {
            let found = self
                .tree
                .query(&point)
                .nearest_one::<SquaredEuclidean<f64>>()
                .execute();
            out.push(Neighbor {
                index: found.item,
                distance_squared: found.distance,
            });
            return;
        }

        let found = self
            .tree
            .query(&point)
            .nearest_n::<SquaredEuclidean<f64>>(count)
            .execute();
        for item in found {
            out.push(Neighbor {
                index: item.item,
                distance_squared: item.distance,
            });
        }
        // kiddo makes no promise about breaking ties by index, so impose
        // the contract here.
        out.sort_unstable_by(compare_neighbors);
    }

    fn radius_into(&self, query: &Vector3<f64>, radius: f64, out: &mut Vec<Neighbor>) {
        out.clear();
        let point = [query.x, query.y, query.z];
        // The threshold is in the metric's own units: for SquaredEuclidean
        // that is a squared distance, not a distance.
        let found = self
            .tree
            .query(&point)
            .within::<SquaredEuclidean<f64>>(radius * radius)
            .execute();
        for item in found {
            out.push(Neighbor {
                index: item.item,
                distance_squared: item.distance,
            });
        }
        out.sort_unstable_by(compare_neighbors);
    }
}
