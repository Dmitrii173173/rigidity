//! What the measurement examples share, and nothing else.
//!
//! This crate exists for one reason: examples of two different crates
//! cannot share a line of code without one of them owning it, and the
//! alternatives were all worse. Four examples — M9's pairs, the survey, the
//! spectrum sweep and the basin detector — crop scans by the same rule, and
//! they have to crop by the *same* rule or three separate measurements stop
//! being comparable. Four copies obliged to stay identical is a promise
//! nobody can keep; a shared file included by path is not carried into a
//! published crate's tarball, so the published examples would not build;
//! and putting it in a library crate would mean a published API grown for
//! the convenience of this repository's own experiments.
//!
//! So: a crate, `publish = false`, holding only what the examples need.
//! Nothing in any `src/` may depend on it.

pub mod probabilistic;
pub mod thresholded;

use rigidity_core::PointCloud;

/// Keeps the points of a scan that lie inside an angular sector.
///
/// `half_width` is measured in radians from `centre`, about the sensor's
/// own vertical axis — a point's azimuth is `atan2(y, x)` in the cloud's
/// **absolute** frame, where the sensor stands at the coordinate origin —
/// so the result is the scan as a sensor with that field of view would
/// have recorded it. A ±40° crop of a corridor scan is a forward-looking
/// lidar on a robot; the same scan uncropped is a tripod.
///
/// The frame is the whole of it, and reading `local()` here was wrong.
/// [`PointCloud::local`] is relative to the cloud's origin, and
/// `read_csv` puts that origin at the centre of the bounding box so that
/// global coordinates survive `f32` storage. Azimuth taken there is
/// measured from an arbitrary point in the room rather than from the
/// scanner: on the ETH ASL `apartment` a ±40° crop kept 1490 points of
/// 370277 instead of 50368, on `stairs` it emptied some scans outright,
/// and in neither case was what it kept a sector of anything.
///
/// The difference is not cosmetic, and it is why this function exists. On
/// the ETH ASL corridor the uncropped scans are not degenerate at all —
/// condition numbers of 2.3 to 2.8, every pair converging — because a 360°
/// view sees both walls, the floor, the ceiling and the ends. Every
/// degenerate case this project has measured on real data was made here.
///
/// Points keep their original order and the result carries the same origin.
pub fn crop_sector(cloud: &PointCloud, centre: f64, half_width: f64) -> PointCloud {
    let mut result = PointCloud::with_origin(cloud.origin());
    for index in 0..cloud.len() {
        let point = cloud.point(index);
        let azimuth = point.y.atan2(point.x);
        let mut delta = azimuth - centre;
        while delta > std::f64::consts::PI {
            delta -= std::f64::consts::TAU;
        }
        while delta < -std::f64::consts::PI {
            delta += std::f64::consts::TAU;
        }
        if delta.abs() <= half_width {
            result.push(cloud.point(index));
        }
    }
    result
}

/// What share of a moving cloud finds a correspondence in a fixed one, at a
/// given pose.
///
/// The measure of whether two scans see the same place. It is deliberately
/// not symmetric — a small scan inside a large one overlaps it entirely
/// while the reverse is barely true — so a caller comparing two stations
/// should take both and decide which it means.
pub fn overlap(
    moving: &rigidity_pipeline::Prepared,
    fixed: &rigidity_pipeline::Prepared,
    pose: &rigidity_core::lie::Se3,
    params: &rigidity_pipeline::RegisterParams,
) -> f64 {
    use rigidity_core::neighbors::NeighborSearch;

    let limit = params.max_distance * params.max_distance;
    let mut found = Vec::with_capacity(1);
    let mut matched = 0usize;
    for index in 0..moving.cloud.len() {
        let point = pose.transform_point(&moving.cloud.point(index));
        fixed.tree.knn_into(&point, 1, &mut found);
        if found
            .first()
            .is_some_and(|nearest| nearest.distance_squared <= limit)
        {
            matched += 1;
        }
    }
    matched as f64 / moving.cloud.len().max(1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use rigidity_core::nalgebra::Vector3;

    /// A ring cropped to a quarter turn keeps a quarter of it — and keeps
    /// the right quarter, including across the seam at ±π, which is the
    /// case a plain difference of angles gets wrong.
    #[test]
    fn a_sector_keeps_the_points_it_names() {
        let mut cloud = PointCloud::new();
        const COUNT: usize = 360;
        for step in 0..COUNT {
            let angle = step as f64 * std::f64::consts::TAU / COUNT as f64;
            cloud.push(Vector3::new(angle.cos(), angle.sin(), 0.0));
        }

        let quarter = crop_sector(&cloud, 0.0, std::f64::consts::FRAC_PI_4);
        assert_eq!(quarter.len(), 91, "±45° of 360 points, both ends included");
        for index in 0..quarter.len() {
            let point = quarter.point(index);
            let azimuth = point.y.atan2(point.x);
            assert!(
                azimuth.abs() <= std::f64::consts::FRAC_PI_4 + 1e-12,
                "kept a point at {azimuth:.3} rad"
            );
        }

        let seam = crop_sector(&cloud, std::f64::consts::PI, 0.1);
        assert!(seam.len() >= 11, "the seam kept only {}", seam.len());
        for index in 0..seam.len() {
            let point = seam.point(index);
            let off = (point.y.atan2(point.x).abs() - std::f64::consts::PI).abs();
            assert!(off <= 0.1 + 1e-12, "kept a point {off:.3} rad off the seam");
        }
    }

    /// The same ring, stored against an origin somewhere else, is the same
    /// ring — a sector is a property of where the sensor stood, not of the
    /// number a reader subtracted to make the coordinates fit `f32`.
    ///
    /// The test above cannot see the difference: it builds its cloud with
    /// `PointCloud::new()`, whose origin is zero, and there `local` and
    /// `point` are the same vector. `read_csv` puts the origin at the
    /// centre of the bounding box, and every real scan this project crops
    /// arrives that way.
    #[test]
    fn a_sector_does_not_depend_on_where_the_cloud_stores_its_origin() {
        const COUNT: usize = 360;
        let ring = |origin: Vector3<f64>| {
            let mut cloud = PointCloud::with_origin(origin);
            for step in 0..COUNT {
                let angle = step as f64 * std::f64::consts::TAU / COUNT as f64;
                cloud.push(Vector3::new(angle.cos(), angle.sin(), 0.0));
            }
            crop_sector(&cloud, 0.0, std::f64::consts::FRAC_PI_4)
        };

        let centred = ring(Vector3::zeros());
        let offset = ring(Vector3::new(37.0, -11.0, 4.0));
        assert_eq!(
            centred.len(),
            offset.len(),
            "the same sector kept {} points about the origin and {} about (37, -11, 4)",
            centred.len(),
            offset.len()
        );
        for index in 0..offset.len() {
            let point = offset.point(index);
            let azimuth = point.y.atan2(point.x);
            assert!(
                azimuth.abs() <= std::f64::consts::FRAC_PI_4 + 1e-6,
                "kept a point at {azimuth:.3} rad"
            );
        }
    }
}
