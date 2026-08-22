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

/// Writes a cloud as LAS, or as LAZ when the path says so.
///
/// The scale is a millimetre. LAS stores coordinates as scaled integers
/// about an offset, and the scale is the quantum: at a tenth of a
/// millimetre a survey-sized extent overflows the thirty-two bits the
/// format gives each axis, and at a centimetre the file is coarser than
/// the instrument that made it. A millimetre reaches ±2000 km from the
/// offset, which is more than any projected coordinate system needs.
pub fn write_las(cloud: &PointCloud, path: &Path) -> Result<(), IoError> {
    /// Metres per stored unit.
    const SCALE: f64 = 0.001;

    let origin = cloud.origin();
    let mut header = las::Builder::from((1, 4));
    header.transforms = las::Vector {
        x: las::Transform {
            scale: SCALE,
            offset: origin.x,
        },
        y: las::Transform {
            scale: SCALE,
            offset: origin.y,
        },
        z: las::Transform {
            scale: SCALE,
            offset: origin.z,
        },
    };
    // Compression is chosen by the extension, the same way the reader
    // chooses it.
    header.point_format.is_compressed = path
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("laz"));
    let header = header
        .into_header()
        .map_err(|e| IoError::Las(e.to_string()))?;

    let mut writer =
        las::Writer::from_path(path, header).map_err(|e| IoError::Las(e.to_string()))?;
    let intensity = cloud.attribute("intensity");
    for index in 0..cloud.len() {
        let point = cloud.point(index);
        writer
            .write_point(las::Point {
                x: point.x,
                y: point.y,
                z: point.z,
                intensity: match intensity.map(|attribute| &attribute.data) {
                    Some(AttributeData::U16(values)) => values.get(index).copied().unwrap_or(0),
                    _ => 0,
                },
                ..Default::default()
            })
            .map_err(|e| IoError::Las(e.to_string()))?;
    }
    writer.close().map_err(|e| IoError::Las(e.to_string()))?;
    Ok(())
}
