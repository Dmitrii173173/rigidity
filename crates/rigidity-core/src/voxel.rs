//! Voxel downsampling with a bit-for-bit reproducible result.

use nalgebra::Vector3;
use rayon::prelude::*;

use crate::cloud::{CloudError, PointCloud};

/// A voxel key. `i64` rather than `i32`: with a fine voxel and
/// georeferenced coordinates 32 bits overflow, and an overflow here
/// silently merges distant points into a single cell.
type VoxelKey = [i64; 3];

/// Downsamples a cloud onto a grid of cells with edge `voxel_size`,
/// replacing the points of each occupied cell by their centroid.
///
/// # Determinism
///
/// The result is bit-for-bit identical at any thread count, both in values
/// and in point order. This is a requirement, not a side effect: further up
/// the stack these points assemble the Jacobian, and a floating order of
/// summation would make the smallest singular value float — that is, the
/// degeneracy detector itself.
///
/// Three decisions achieve it:
/// 1. points are sorted by the key `(cell, original index)`, which is a
///    total order, so the sorted sequence is unique regardless of the
///    sorting algorithm;
/// 2. within a cell, accumulation runs sequentially in increasing original
///    index, so the order of summation is fixed;
/// 3. parallelism happens only between cells, which are independent.
///
/// A hash table instead of the sort would be faster, but neither its
/// traversal order nor its order of merging partial sums is defined.
///
/// # Attributes
///
/// Attributes are not carried over: averaging depends on what the column
/// means — colour averages, a class label does not. The returned cloud has
/// coordinates only.
pub fn voxel_downsample(cloud: &PointCloud, voxel_size: f64) -> Result<PointCloud, CloudError> {
    voxel_downsample_observed(cloud, voxel_size, |_, _| {})
}

/// The same, reporting progress after each internal phase.
///
/// `progress` receives `(completed, total)` in phases, not in points:
/// the work is a sequence of whole-cloud passes — keys, sort, cell
/// boundaries, centroids — and none of them can report a fraction of
/// itself without giving up the parallelism that makes it fast.
///
/// Nothing guarantees the callback ever runs: a cloud that is empty or
/// rejected returns before the first phase. Completion is signalled by the
/// function returning, not by a final call.
pub fn voxel_downsample_observed<F>(
    cloud: &PointCloud,
    voxel_size: f64,
    mut progress: F,
) -> Result<PointCloud, CloudError>
where
    F: FnMut(usize, usize),
{
    /// Keys, sort, cell boundaries, centroids.
    const PHASES: usize = 4;

    if !(voxel_size.is_finite() && voxel_size > 0.0) {
        return Err(CloudError::InvalidVoxelSize(voxel_size));
    }
    cloud.check_finite()?;

    let count = cloud.len();
    if count == 0 {
        return Ok(PointCloud::with_origin(cloud.origin()));
    }

    let inverse_size = 1.0 / voxel_size;
    let keys: Vec<VoxelKey> = (0..count)
        .into_par_iter()
        .map(|i| {
            let p = cloud.local(i);
            [
                (p.x * inverse_size).floor() as i64,
                (p.y * inverse_size).floor() as i64,
                (p.z * inverse_size).floor() as i64,
            ]
        })
        .collect();
    progress(1, PHASES);

    // The sort key includes the original index, so it is strict and the
    // ordered sequence is unique.
    let mut order: Vec<u32> = (0..count as u32).collect();
    order.par_sort_unstable_by_key(|&i| (keys[i as usize], i));
    progress(2, PHASES);

    // Boundaries of the runs of equal keys.
    let mut segments: Vec<(usize, usize)> = Vec::new();
    let mut start = 0usize;
    for i in 1..count {
        if keys[order[i] as usize] != keys[order[start] as usize] {
            segments.push((start, i));
            start = i;
        }
    }
    segments.push((start, count));
    progress(3, PHASES);

    // Across cells in parallel; within a cell strictly in order.
    let centroids: Vec<Vector3<f64>> = segments
        .par_iter()
        .map(|&(begin, end)| {
            let mut sum = Vector3::zeros();
            for &index in &order[begin..end] {
                sum += cloud.local(index as usize);
            }
            sum / (end - begin) as f64
        })
        .collect();

    let mut result = PointCloud::with_capacity(centroids.len());
    result.rebase(cloud.origin());
    for centroid in centroids {
        result.push(cloud.origin() + centroid);
    }
    progress(4, PHASES);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid_cloud(side: usize) -> PointCloud {
        let mut cloud = PointCloud::with_capacity(side * side * side);
        for i in 0..side {
            for j in 0..side {
                for k in 0..side {
                    cloud.push(Vector3::new(i as f64 * 0.1, j as f64 * 0.1, k as f64 * 0.1));
                }
            }
        }
        cloud
    }

    #[test]
    fn rejects_bad_voxel_size() {
        let cloud = grid_cloud(2);
        assert!(voxel_downsample(&cloud, 0.0).is_err());
        assert!(voxel_downsample(&cloud, -1.0).is_err());
        assert!(voxel_downsample(&cloud, f64::NAN).is_err());
    }

    #[test]
    fn empty_cloud_stays_empty() {
        let cloud = PointCloud::new();
        assert!(voxel_downsample(&cloud, 1.0).unwrap().is_empty());
    }

    /// A cell larger than the whole cloud collapses it to a single point,
    /// the centroid of the original coordinates.
    #[test]
    fn single_voxel_gives_centroid() {
        let cloud = grid_cloud(4);
        let reduced = voxel_downsample(&cloud, 100.0).unwrap();
        assert_eq!(reduced.len(), 1);
        let expected = Vector3::new(0.15, 0.15, 0.15);
        // The tolerance is set by f32 storage, not by the algorithm: 0.1
        // is not exactly representable, and the mean of four coordinates
        // drifts from 0.15 by 4.1e-9. Nothing here can go below that floor
        // by construction — accumulating in f64 removes the summation
        // error, not the storage error.
        let error = (reduced.point(0) - expected).norm();
        assert!(error < 1e-6, "centroid deviation {error:.3e}");
        assert!(
            error > 1e-10,
            "the f32 floor disappeared — check whether storage became f64"
        );
    }

    /// The point count never grows, and every original point lies within
    /// a cell diagonal of some output point.
    #[test]
    fn output_covers_input() {
        let cloud = grid_cloud(6);
        let size = 0.25;
        let reduced = voxel_downsample(&cloud, size).unwrap();
        assert!(reduced.len() < cloud.len());
        let radius = size * 3f64.sqrt();
        for i in 0..cloud.len() {
            let p = cloud.point(i);
            let nearest = (0..reduced.len())
                .map(|j| (reduced.point(j) - p).norm())
                .fold(f64::INFINITY, f64::min);
            assert!(nearest <= radius, "point {i}: nearest at {nearest}");
        }
    }
}
