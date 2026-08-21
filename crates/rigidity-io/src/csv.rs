//! Reading point clouds from CSV with a header row.
//!
//! The format of the ETH ASL datasets: the first line holds column names,
//! the rest are comma-separated numbers. Coordinate names differ between
//! sets (`x`/`y`/`z`, sometimes prefixed), so columns are located by name
//! rather than by position.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use nalgebra::Vector3;
use rigidity_core::PointCloud;

use crate::IoError;

/// Finds the column whose name matches one of the candidates.
fn find_column(header: &[String], names: &[&str]) -> Option<usize> {
    header.iter().position(|column| {
        let trimmed = column.trim().trim_matches('"').to_ascii_lowercase();
        names.iter().any(|candidate| trimmed == *candidate)
    })
}

/// Reads a cloud from CSV.
///
/// The origin is placed at the centre of the bounding box: dataset files
/// sometimes hold global coordinates with large values, and `f32` storage
/// cannot take those.
pub fn read_csv(path: &Path) -> Result<PointCloud, IoError> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 {
        return Err(IoError::BadHeader("the file is empty".into()));
    }
    let header: Vec<String> = line.trim().split(',').map(|s| s.to_string()).collect();

    let x = find_column(&header, &["x"]).ok_or_else(|| IoError::BadHeader("no x column".into()))?;
    let y = find_column(&header, &["y"]).ok_or_else(|| IoError::BadHeader("no y column".into()))?;
    let z = find_column(&header, &["z"]).ok_or_else(|| IoError::BadHeader("no z column".into()))?;
    let width = header.len();

    let mut points: Vec<Vector3<f64>> = Vec::new();
    let mut minimum = Vector3::repeat(f64::INFINITY);
    let mut maximum = Vector3::repeat(f64::NEG_INFINITY);

    line.clear();
    while reader.read_line(&mut line)? != 0 {
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            let fields: Vec<&str> = trimmed.split(',').collect();
            if fields.len() >= width {
                let parse = |index: usize| -> Result<f64, IoError> {
                    fields[index]
                        .trim()
                        .parse()
                        .map_err(|_| IoError::BadNumber(fields[index].to_string()))
                };
                let point = Vector3::new(parse(x)?, parse(y)?, parse(z)?);
                if point.iter().all(|value| value.is_finite()) {
                    minimum = minimum.inf(&point);
                    maximum = maximum.sup(&point);
                    points.push(point);
                }
            }
        }
        line.clear();
    }

    if points.is_empty() {
        return Ok(PointCloud::new());
    }
    let origin = (minimum + maximum) * 0.5;
    let mut cloud = PointCloud::with_origin(origin);
    for point in &points {
        cloud.push(*point);
    }
    Ok(cloud)
}

/// Reads a `pose_scanner_leica.csv` pose file.
///
/// Each line holds a homogeneous 4×4 matrix row by row, plus bookkeeping
/// columns. Matrices are returned in file order.
pub fn read_poses(path: &Path) -> Result<Vec<nalgebra::Matrix4<f64>>, IoError> {
    let text = std::fs::read_to_string(path)?;
    let mut lines = text.lines();
    let header: Vec<String> = lines
        .next()
        .ok_or_else(|| IoError::BadHeader("the pose file is empty".into()))?
        .trim()
        .split(',')
        .map(|s| s.trim().trim_matches('"').to_ascii_lowercase())
        .collect();

    // Columns named T00…T33 or m00…m33.
    let mut indices = [[usize::MAX; 4]; 4];
    for (row, slots) in indices.iter_mut().enumerate() {
        for (column, slot) in slots.iter_mut().enumerate() {
            let names = [
                format!("t{row}{column}"),
                format!("m{row}{column}"),
                format!("t_{row}{column}"),
            ];
            let found = header
                .iter()
                .position(|name| names.iter().any(|candidate| name == candidate));
            match found {
                Some(index) => *slot = index,
                None => {
                    return Err(IoError::BadHeader(format!(
                        "no matrix column T{row}{column}"
                    )));
                }
            }
        }
    }

    let mut poses = Vec::new();
    for line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let fields: Vec<&str> = trimmed.split(',').collect();
        let mut matrix = nalgebra::Matrix4::zeros();
        for (row, slots) in indices.iter().enumerate() {
            for (column, index) in slots.iter().enumerate() {
                matrix[(row, column)] = fields
                    .get(*index)
                    .ok_or_else(|| IoError::BadHeader("row shorter than the header".into()))?
                    .trim()
                    .parse()
                    .map_err(|_| IoError::BadNumber(fields[*index].to_string()))?;
            }
        }
        poses.push(matrix);
    }
    Ok(poses)
}
