//! Reading and writing PCD, the Point Cloud Library's own format.
//!
//! Written here rather than taken from a crate for the same reason PLY is:
//! the format is a text header and a block of numbers, the whole of it fits
//! on one page, and a dependency for that costs more than the code does.
//! It is also the format most likely to arrive from a robot, since
//! everything built on PCL writes it.
//!
//! Supported: `ascii` and `binary` data, `F`/`U`/`I` fields of one, two,
//! four or eight bytes, and any field order. Not supported: `binary_compressed`,
//! which is an LZF stream nobody writes by hand and which no file this
//! application has been shown has used.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

use nalgebra::Vector3;
use rigidity_core::PointCloud;

use crate::IoError;

/// One column of the header.
#[derive(Clone)]
struct Field {
    name: String,
    kind: char,
    size: usize,
    count: usize,
}

/// What the header said.
struct Header {
    fields: Vec<Field>,
    points: usize,
    binary: bool,
}

/// Reads a cloud from PCD.
///
/// The origin is placed at the centre of the bounding box, as for every
/// other georeferenced format: `f32` storage cannot hold absolute
/// coordinates in the hundreds of thousands and millimetres at the same
/// time.
pub fn read_pcd(path: &Path) -> Result<PointCloud, IoError> {
    let mut reader = BufReader::new(File::open(path)?);
    let header = read_header(&mut reader)?;
    let (x, y, z) = coordinates(&header)?;

    let stride: usize = header
        .fields
        .iter()
        .map(|field| field.size * field.count)
        .sum();
    let mut rows: Vec<[f64; 3]> = Vec::with_capacity(header.points);

    if header.binary {
        let mut row = vec![0u8; stride];
        for _ in 0..header.points {
            reader.read_exact(&mut row)?;
            rows.push([
                value(&row, &header.fields, x)?,
                value(&row, &header.fields, y)?,
                value(&row, &header.fields, z)?,
            ]);
        }
    } else {
        for line in reader.lines() {
            let line = line?;
            let numbers: Vec<&str> = line.split_whitespace().collect();
            if numbers.is_empty() {
                continue;
            }
            let pick = |at: usize| -> Result<f64, IoError> {
                numbers
                    .get(at)
                    .ok_or_else(|| IoError::BadPcd("a row is shorter than the header".into()))?
                    .parse()
                    .map_err(|_| IoError::BadNumber(numbers[at].to_owned()))
            };
            rows.push([pick(x)?, pick(y)?, pick(z)?]);
            if rows.len() == header.points {
                break;
            }
        }
    }

    if rows.is_empty() {
        return Ok(PointCloud::new());
    }
    let mut min = Vector3::repeat(f64::INFINITY);
    let mut max = Vector3::repeat(f64::NEG_INFINITY);
    for row in &rows {
        let point = Vector3::new(row[0], row[1], row[2]);
        min = min.inf(&point);
        max = max.sup(&point);
    }

    let mut cloud = PointCloud::with_origin((min + max) * 0.5);
    for row in rows {
        cloud.push(Vector3::new(row[0], row[1], row[2]));
    }
    Ok(cloud)
}

/// Writes a cloud as binary PCD.
///
/// Binary rather than ascii: a million points in text is thirty megabytes
/// of decimal digits and a lossy round trip unless every one is printed to
/// seventeen significant figures.
///
/// # Why the coordinates are `f64`
///
/// PCL's own point types are `f32`, and writing `f32` here would be the
/// compatible choice. It is also wrong for the data this application
/// exists to handle: at a UTM northing of four million metres the `f32`
/// step is a quarter of a metre, and a survey written that way comes back
/// having moved further than the thing it was measuring. The storage keeps
/// `f32` *offsets from an `f64` origin* precisely to avoid that, and a
/// writer that flattens the two throws the invariant away at the last
/// step. The format allows `SIZE 8`; tools that only read `f32` will say
/// so, which is better than silently rounding.
pub fn write_pcd(cloud: &PointCloud, path: &Path) -> Result<(), IoError> {
    let mut out = BufWriter::new(File::create(path)?);
    writeln!(out, "# .PCD v0.7 - Point Cloud Data file format")?;
    writeln!(out, "VERSION 0.7")?;
    writeln!(out, "FIELDS x y z")?;
    writeln!(out, "SIZE 8 8 8")?;
    writeln!(out, "TYPE F F F")?;
    writeln!(out, "COUNT 1 1 1")?;
    writeln!(out, "WIDTH {}", cloud.len())?;
    writeln!(out, "HEIGHT 1")?;
    writeln!(out, "VIEWPOINT 0 0 0 1 0 0 0")?;
    writeln!(out, "POINTS {}", cloud.len())?;
    writeln!(out, "DATA binary")?;

    for index in 0..cloud.len() {
        let point = cloud.point(index);
        for value in [point.x, point.y, point.z] {
            out.write_all(&value.to_le_bytes())?;
        }
    }
    out.flush()?;
    Ok(())
}

fn read_header<R: BufRead>(reader: &mut R) -> Result<Header, IoError> {
    let mut fields: Vec<String> = Vec::new();
    let mut sizes: Vec<usize> = Vec::new();
    let mut kinds: Vec<char> = Vec::new();
    let mut counts: Vec<usize> = Vec::new();
    let mut points = None;
    let mut width = None;
    let mut height = None;

    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            return Err(IoError::BadPcd("the header never ended".into()));
        }
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (key, rest) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
        let words: Vec<&str> = rest.split_whitespace().collect();
        let numbers = |words: &[&str]| -> Result<Vec<usize>, IoError> {
            words
                .iter()
                .map(|word| {
                    word.parse()
                        .map_err(|_| IoError::BadNumber((*word).to_owned()))
                })
                .collect()
        };
        match key.to_ascii_uppercase().as_str() {
            "FIELDS" => fields = words.iter().map(|word| (*word).to_owned()).collect(),
            "SIZE" => sizes = numbers(&words)?,
            "TYPE" => {
                kinds = words
                    .iter()
                    .filter_map(|word| word.chars().next())
                    .collect()
            }
            "COUNT" => counts = numbers(&words)?,
            "WIDTH" => width = numbers(&words)?.first().copied(),
            "HEIGHT" => height = numbers(&words)?.first().copied(),
            "POINTS" => points = numbers(&words)?.first().copied(),
            "DATA" => {
                let binary = match words.first().map(|word| word.to_ascii_lowercase()) {
                    Some(word) if word == "binary" => true,
                    Some(word) if word == "ascii" => false,
                    other => {
                        return Err(IoError::UnsupportedPcd(
                            other.unwrap_or_else(|| "nothing".into()),
                        ));
                    }
                };
                // `POINTS` is optional in the specification; `WIDTH ×
                // HEIGHT` is not, and an organised cloud states its shape
                // there.
                let points = points
                    .or_else(|| Some(width? * height?))
                    .ok_or_else(|| IoError::BadPcd("the header gives no point count".into()))?;
                if fields.len() != sizes.len() || fields.len() != kinds.len() {
                    return Err(IoError::BadPcd(
                        "FIELDS, SIZE and TYPE disagree about how many columns there are".into(),
                    ));
                }
                let fields = fields
                    .iter()
                    .enumerate()
                    .map(|(index, name)| Field {
                        name: name.clone(),
                        kind: kinds[index],
                        size: sizes[index],
                        count: counts.get(index).copied().unwrap_or(1),
                    })
                    .collect();
                return Ok(Header {
                    fields,
                    points,
                    binary,
                });
            }
            _ => {}
        }
    }
}

/// Where x, y and z are, by name rather than by position: PCL writes them
/// first by convention and not by rule.
fn coordinates(header: &Header) -> Result<(usize, usize, usize), IoError> {
    let find = |wanted: &str| {
        header
            .fields
            .iter()
            .position(|field| field.name.eq_ignore_ascii_case(wanted))
    };
    match (find("x"), find("y"), find("z")) {
        (Some(x), Some(y), Some(z)) => Ok((x, y, z)),
        _ => Err(IoError::BadPcd("no x, y and z fields".into())),
    }
}

/// One field of one binary row, as `f64`.
fn value(row: &[u8], fields: &[Field], wanted: usize) -> Result<f64, IoError> {
    let at: usize = fields[..wanted]
        .iter()
        .map(|field| field.size * field.count)
        .sum();
    let field = &fields[wanted];
    let bytes = row
        .get(at..at + field.size)
        .ok_or_else(|| IoError::BadPcd("a row is shorter than the header".into()))?;
    let signed = |bytes: &[u8]| -> i64 {
        let mut wide = [0u8; 8];
        wide[..bytes.len()].copy_from_slice(bytes);
        let raw = u64::from_le_bytes(wide);
        // Sign-extend from the field's own width.
        let shift = 64 - bytes.len() * 8;
        ((raw << shift) as i64) >> shift
    };
    Ok(match (field.kind.to_ascii_uppercase(), field.size) {
        ('F', 4) => f64::from(f32::from_le_bytes(bytes.try_into().unwrap())),
        ('F', 8) => f64::from_le_bytes(bytes.try_into().unwrap()),
        ('U', _) => {
            let mut wide = [0u8; 8];
            wide[..bytes.len()].copy_from_slice(bytes);
            u64::from_le_bytes(wide) as f64
        }
        ('I', _) => signed(bytes) as f64,
        (kind, size) => {
            return Err(IoError::UnsupportedPcd(format!("{kind}{}", size * 8)));
        }
    })
}
