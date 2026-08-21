//! Reading LAS and LAZ through the `las` crate.

use std::path::Path;

use nalgebra::Vector3;
use rigidity_core::{Attribute, AttributeData, PointCloud};

use crate::IoError;

/// Reads a cloud from LAS/LAZ.
///
/// # Origin
///
/// LAS is almost always georeferenced: coordinates are UTM or a similar
/// projection with values around a million metres. The origin is placed at
/// the centre of the bounding box, since otherwise `f32` storage would
/// destroy millimetres before the first computation.
///
/// # Attributes
///
/// Intensity is always carried over; colour is carried when the point
/// format has it.
pub fn read_las(path: &Path) -> Result<PointCloud, IoError> {
    let mut reader = las::Reader::from_path(path).map_err(|e| IoError::Las(e.to_string()))?;
    let data = reader.read_all().map_err(|e| IoError::Las(e.to_string()))?;

    // First pass: the bounds, to choose the origin.
    let mut min = Vector3::repeat(f64::INFINITY);
    let mut max = Vector3::repeat(f64::NEG_INFINITY);
    let mut count = 0usize;
    for point in data.points() {
        let point = point.map_err(|e| IoError::Las(e.to_string()))?;
        let p = Vector3::new(point.x, point.y, point.z);
        min = min.inf(&p);
        max = max.sup(&p);
        count += 1;
    }
    if count == 0 {
        return Ok(PointCloud::new());
    }
    let origin = (min + max) * 0.5;

    // Second pass: the points themselves and their attributes.
    let mut cloud = PointCloud::with_origin(origin);
    let mut intensity = Vec::with_capacity(count);
    let mut colors: Option<(Vec<u16>, Vec<u16>, Vec<u16>)> = None;
    for point in data.points() {
        let point = point.map_err(|e| IoError::Las(e.to_string()))?;
        cloud.push(Vector3::new(point.x, point.y, point.z));
        intensity.push(point.intensity);
        if let Some(color) = point.color {
            let channels = colors.get_or_insert_with(|| {
                (
                    Vec::with_capacity(count),
                    Vec::with_capacity(count),
                    Vec::with_capacity(count),
                )
            });
            channels.0.push(color.red);
            channels.1.push(color.green);
            channels.2.push(color.blue);
        }
    }

    cloud.push_attribute(Attribute {
        name: "intensity".into(),
        data: AttributeData::U16(intensity),
    })?;
    if let Some((red, green, blue)) = colors {
        for (name, channel) in [("red", red), ("green", green), ("blue", blue)] {
            cloud.push_attribute(Attribute {
                name: name.into(),
                data: AttributeData::U16(channel),
            })?;
        }
    }
    Ok(cloud)
}
