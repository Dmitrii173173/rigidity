//! Reading and writing E57, the surveying interchange format.
//!
//! Through the `e57` crate, which does both directions — which is why this
//! module could be written and tested at all. `rigidity`'s own plan
//! deferred E57 "for want of test data", and a format that can be written
//! supplies its own: a cloud goes out and comes back, and the round trip
//! is the test.
//!
//! An E57 file holds *several* clouds, each with its own pose — a survey
//! is a set of scans, not one cloud. Reading concatenates them, applying
//! each scan's transform, because everything upstream of here works on one
//! cloud at a time. Stage three of the viewer's plan is where the scans
//! stay apart.

use std::path::Path;

use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use rigidity_core::PointCloud;

use crate::IoError;

/// Reads a cloud from E57, concatenating every scan it holds.
pub fn read_e57(path: &Path) -> Result<PointCloud, IoError> {
    let mut file = e57::E57Reader::from_file(path).map_err(|e| IoError::E57(e.to_string()))?;
    let clouds = file.pointclouds();

    let mut points: Vec<Vector3<f64>> = Vec::new();
    for cloud in &clouds {
        // Each scan states where it was taken from; its points are in its
        // own frame until that is applied.
        let (rotation, translation) = cloud.transform.as_ref().map_or_else(
            || (UnitQuaternion::identity(), Vector3::zeros()),
            |transform| {
                let quaternion = Quaternion::new(
                    transform.rotation.w,
                    transform.rotation.x,
                    transform.rotation.y,
                    transform.rotation.z,
                );
                (
                    UnitQuaternion::from_quaternion(quaternion),
                    Vector3::new(
                        transform.translation.x,
                        transform.translation.y,
                        transform.translation.z,
                    ),
                )
            },
        );

        let reader = file
            .pointcloud_simple(cloud)
            .map_err(|e| IoError::E57(e.to_string()))?;
        for point in reader {
            let point = point.map_err(|e| IoError::E57(e.to_string()))?;
            if let e57::CartesianCoordinate::Valid { x, y, z } = point.cartesian {
                points.push(rotation * Vector3::new(x, y, z) + translation);
            }
        }
    }

    if points.is_empty() {
        return Ok(PointCloud::new());
    }
    let mut min = Vector3::repeat(f64::INFINITY);
    let mut max = Vector3::repeat(f64::NEG_INFINITY);
    for point in &points {
        min = min.inf(point);
        max = max.sup(point);
    }
    let mut cloud = PointCloud::with_origin((min + max) * 0.5);
    for point in points {
        cloud.push(point);
    }
    Ok(cloud)
}

/// Writes a cloud as a single-scan E57 file.
///
/// The coordinates go out as `f64` in the file's own frame, with no scan
/// transform: what came in as absolute coordinates goes out as absolute
/// coordinates, and a reader that ignores transforms still gets the right
/// answer.
pub fn write_e57(cloud: &PointCloud, path: &Path) -> Result<(), IoError> {
    let mut file =
        e57::E57Writer::from_file(path, "rigidity").map_err(|e| IoError::E57(e.to_string()))?;
    let prototype = vec![
        e57::Record::CARTESIAN_X_F64,
        e57::Record::CARTESIAN_Y_F64,
        e57::Record::CARTESIAN_Z_F64,
    ];
    let mut writer = file
        .add_pointcloud("rigidity-scan", prototype)
        .map_err(|e| IoError::E57(e.to_string()))?;
    for index in 0..cloud.len() {
        let point = cloud.point(index);
        writer
            .add_point(vec![
                e57::RecordValue::Double(point.x),
                e57::RecordValue::Double(point.y),
                e57::RecordValue::Double(point.z),
            ])
            .map_err(|e| IoError::E57(e.to_string()))?;
    }
    writer.finalize().map_err(|e| IoError::E57(e.to_string()))?;
    file.finalize().map_err(|e| IoError::E57(e.to_string()))?;
    Ok(())
}
