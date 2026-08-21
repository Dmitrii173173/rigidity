//! Reading and writing PLY.
//!
//! `ascii` and `binary_little_endian` are supported. `binary_big_endian`
//! is rejected explicitly: it is practically never seen, and silently
//! misinterpreting bytes is worse than refusing.
//!
//! Only the first element is read, and it must be called `vertex`. Faces
//! are of no use here, and partial support for complex files creates an
//! illusion of compatibility.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

use nalgebra::Vector3;
use rigidity_core::{Attribute, AttributeData, PointCloud};

use crate::IoError;

/// The comment that carries the cloud's origin.
///
/// The PLY standard has no place for an offset. Other programs ignore
/// comments, so the file stays readable while our own round trip stays
/// exact. For clouds with a zero origin — that is, for all synthetic data
/// — the comment is not written at all.
const ORIGIN_COMMENT: &str = "rigidity_origin";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    Ascii,
    BinaryLittleEndian,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScalarType {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl ScalarType {
    fn parse(name: &str) -> Result<Self, IoError> {
        Ok(match name {
            "char" | "int8" => Self::I8,
            "uchar" | "uint8" => Self::U8,
            "short" | "int16" => Self::I16,
            "ushort" | "uint16" => Self::U16,
            "int" | "int32" => Self::I32,
            "uint" | "uint32" => Self::U32,
            "float" | "float32" => Self::F32,
            "double" | "float64" => Self::F64,
            other => return Err(IoError::UnknownPropertyType(other.to_string())),
        })
    }

    fn size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    /// The type name used when writing: the canonical one, not a synonym.
    fn ply_name(self) -> &'static str {
        match self {
            Self::I8 => "char",
            Self::U8 => "uchar",
            Self::I16 => "short",
            Self::U16 => "ushort",
            Self::I32 => "int",
            Self::U32 => "uint",
            Self::F32 => "float",
            Self::F64 => "double",
        }
    }

    /// An empty column of the matching type. `i8` and `i16` widen to
    /// `i32`: separate storage variants are not worth it for rare types.
    fn empty_column(self, capacity: usize) -> AttributeData {
        match self {
            Self::U8 => AttributeData::U8(Vec::with_capacity(capacity)),
            Self::U16 => AttributeData::U16(Vec::with_capacity(capacity)),
            Self::U32 => AttributeData::U32(Vec::with_capacity(capacity)),
            Self::I8 | Self::I16 | Self::I32 => AttributeData::I32(Vec::with_capacity(capacity)),
            Self::F32 => AttributeData::F32(Vec::with_capacity(capacity)),
            Self::F64 => AttributeData::F64(Vec::with_capacity(capacity)),
        }
    }
}

/// A value as read, before conversion to the column type.
#[derive(Debug, Clone, Copy)]
enum Scalar {
    Signed(i64),
    Unsigned(u64),
    Float(f64),
}

impl Scalar {
    fn as_f64(self) -> f64 {
        match self {
            Self::Signed(v) => v as f64,
            Self::Unsigned(v) => v as f64,
            Self::Float(v) => v,
        }
    }

    fn as_i64(self) -> i64 {
        match self {
            Self::Signed(v) => v,
            Self::Unsigned(v) => v as i64,
            Self::Float(v) => v as i64,
        }
    }
}

fn read_binary(ty: ScalarType, buffer: &[u8], offset: usize) -> Scalar {
    match ty {
        ScalarType::I8 => Scalar::Signed(i64::from(buffer[offset] as i8)),
        ScalarType::U8 => Scalar::Unsigned(u64::from(buffer[offset])),
        ScalarType::I16 => Scalar::Signed(i64::from(i16::from_le_bytes(
            buffer[offset..offset + 2].try_into().unwrap(),
        ))),
        ScalarType::U16 => Scalar::Unsigned(u64::from(u16::from_le_bytes(
            buffer[offset..offset + 2].try_into().unwrap(),
        ))),
        ScalarType::I32 => Scalar::Signed(i64::from(i32::from_le_bytes(
            buffer[offset..offset + 4].try_into().unwrap(),
        ))),
        ScalarType::U32 => Scalar::Unsigned(u64::from(u32::from_le_bytes(
            buffer[offset..offset + 4].try_into().unwrap(),
        ))),
        ScalarType::F32 => Scalar::Float(f64::from(f32::from_le_bytes(
            buffer[offset..offset + 4].try_into().unwrap(),
        ))),
        ScalarType::F64 => Scalar::Float(f64::from_le_bytes(
            buffer[offset..offset + 8].try_into().unwrap(),
        )),
    }
}

fn parse_ascii(ty: ScalarType, token: &str) -> Result<Scalar, IoError> {
    let bad = || IoError::BadNumber(token.to_string());
    Ok(match ty {
        ScalarType::F32 | ScalarType::F64 => Scalar::Float(token.parse().map_err(|_| bad())?),
        ScalarType::U8 | ScalarType::U16 | ScalarType::U32 => {
            Scalar::Unsigned(token.parse().map_err(|_| bad())?)
        }
        ScalarType::I8 | ScalarType::I16 | ScalarType::I32 => {
            Scalar::Signed(token.parse().map_err(|_| bad())?)
        }
    })
}

fn push_scalar(column: &mut AttributeData, value: Scalar) {
    match column {
        AttributeData::U8(v) => v.push(value.as_i64() as u8),
        AttributeData::U16(v) => v.push(value.as_i64() as u16),
        AttributeData::U32(v) => v.push(value.as_i64() as u32),
        AttributeData::I32(v) => v.push(value.as_i64() as i32),
        AttributeData::F32(v) => v.push(value.as_f64() as f32),
        AttributeData::F64(v) => v.push(value.as_f64()),
    }
}

/// Where a property's value goes.
enum Target {
    X,
    Y,
    Z,
    Column(usize),
}

struct Property {
    ty: ScalarType,
    target: Target,
}

struct Header {
    format: Format,
    count: usize,
    properties: Vec<Property>,
    column_names: Vec<String>,
    column_types: Vec<ScalarType>,
    origin: Vector3<f64>,
}

fn parse_header(reader: &mut impl BufRead) -> Result<Header, IoError> {
    let mut line = String::new();
    reader.read_line(&mut line)?;
    if line.trim_end() != "ply" {
        return Err(IoError::NotPly);
    }

    let mut format = None;
    let mut count = None;
    let mut properties = Vec::new();
    let mut column_names = Vec::new();
    let mut column_types = Vec::new();
    let mut origin = Vector3::zeros();
    let mut inside_vertex = false;

    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Err(IoError::BadHeader("no end_header line".into()));
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        let mut tokens = trimmed.split_whitespace();
        let Some(keyword) = tokens.next() else {
            continue;
        };

        match keyword {
            "end_header" => break,
            "comment" => {
                let rest: Vec<&str> = tokens.collect();
                if rest.len() == 4 && rest[0] == ORIGIN_COMMENT {
                    for (axis, text) in rest[1..].iter().enumerate() {
                        origin[axis] = text
                            .parse()
                            .map_err(|_| IoError::BadNumber((*text).to_string()))?;
                    }
                }
            }
            "format" => {
                let name = tokens.next().unwrap_or("");
                format = Some(match name {
                    "ascii" => Format::Ascii,
                    "binary_little_endian" => Format::BinaryLittleEndian,
                    other => return Err(IoError::UnsupportedFormat(other.to_string())),
                });
            }
            "element" => {
                let name = tokens.next().unwrap_or("");
                if count.is_some() {
                    // A second element: vertices are already described,
                    // so stop here.
                    inside_vertex = false;
                    continue;
                }
                if name != "vertex" {
                    return Err(IoError::VertexNotFirst(name.to_string()));
                }
                let text = tokens.next().unwrap_or("");
                count = Some(
                    text.parse()
                        .map_err(|_| IoError::BadNumber(text.to_string()))?,
                );
                inside_vertex = true;
            }
            "property" if inside_vertex => {
                let first = tokens.next().unwrap_or("");
                if first == "list" {
                    return Err(IoError::ListInVertex);
                }
                let ty = ScalarType::parse(first)?;
                let name = tokens.next().unwrap_or("").to_string();
                let target = match name.as_str() {
                    "x" => Target::X,
                    "y" => Target::Y,
                    "z" => Target::Z,
                    _ => {
                        column_names.push(name);
                        column_types.push(ty);
                        Target::Column(column_names.len() - 1)
                    }
                };
                properties.push(Property { ty, target });
            }
            _ => {}
        }
    }

    let format = format.ok_or_else(|| IoError::BadHeader("no format line".into()))?;
    let count = count.ok_or_else(|| IoError::BadHeader("no vertex element".into()))?;

    let has_all_coordinates = properties
        .iter()
        .filter(|p| matches!(p.target, Target::X | Target::Y | Target::Z))
        .count()
        == 3;
    if !has_all_coordinates {
        return Err(IoError::MissingCoordinates);
    }

    Ok(Header {
        format,
        count,
        properties,
        column_names,
        column_types,
        origin,
    })
}

/// Reads a cloud from a PLY file.
pub fn read_ply(path: &Path) -> Result<PointCloud, IoError> {
    let mut reader = BufReader::new(File::open(path)?);
    let header = parse_header(&mut reader)?;

    let mut x = Vec::with_capacity(header.count);
    let mut y = Vec::with_capacity(header.count);
    let mut z = Vec::with_capacity(header.count);
    let mut columns: Vec<AttributeData> = header
        .column_types
        .iter()
        .map(|ty| ty.empty_column(header.count))
        .collect();

    match header.format {
        Format::BinaryLittleEndian => {
            let record_size: usize = header.properties.iter().map(|p| p.ty.size()).sum();
            let expected = record_size * header.count;
            let mut body = Vec::with_capacity(expected);
            reader.read_to_end(&mut body)?;
            if body.len() < expected {
                return Err(IoError::Truncated {
                    expected,
                    actual: body.len(),
                });
            }
            for record in 0..header.count {
                let mut offset = record * record_size;
                for property in &header.properties {
                    let value = read_binary(property.ty, &body, offset);
                    match property.target {
                        Target::X => x.push(value.as_f64() as f32),
                        Target::Y => y.push(value.as_f64() as f32),
                        Target::Z => z.push(value.as_f64() as f32),
                        Target::Column(index) => push_scalar(&mut columns[index], value),
                    }
                    offset += property.ty.size();
                }
            }
        }
        Format::Ascii => {
            let mut line = String::new();
            for _ in 0..header.count {
                line.clear();
                if reader.read_line(&mut line)? == 0 {
                    return Err(IoError::Truncated {
                        expected: header.count,
                        actual: x.len(),
                    });
                }
                let mut tokens = line.split_whitespace();
                for property in &header.properties {
                    let token = tokens
                        .next()
                        .ok_or_else(|| IoError::BadHeader("row shorter than the header".into()))?;
                    let value = parse_ascii(property.ty, token)?;
                    match property.target {
                        Target::X => x.push(value.as_f64() as f32),
                        Target::Y => y.push(value.as_f64() as f32),
                        Target::Z => z.push(value.as_f64() as f32),
                        Target::Column(index) => push_scalar(&mut columns[index], value),
                    }
                }
            }
        }
    }

    let mut cloud = PointCloud::from_columns(header.origin, x, y, z)?;
    for (name, data) in header.column_names.into_iter().zip(columns) {
        cloud.push_attribute(Attribute { name, data })?;
    }
    Ok(cloud)
}

fn attribute_scalar_type(data: &AttributeData) -> ScalarType {
    match data {
        AttributeData::F32(_) => ScalarType::F32,
        AttributeData::F64(_) => ScalarType::F64,
        AttributeData::U8(_) => ScalarType::U8,
        AttributeData::U16(_) => ScalarType::U16,
        AttributeData::U32(_) => ScalarType::U32,
        AttributeData::I32(_) => ScalarType::I32,
    }
}

fn write_attribute_value(
    writer: &mut impl Write,
    data: &AttributeData,
    index: usize,
) -> Result<(), IoError> {
    match data {
        AttributeData::F32(v) => writer.write_all(&v[index].to_le_bytes())?,
        AttributeData::F64(v) => writer.write_all(&v[index].to_le_bytes())?,
        AttributeData::U8(v) => writer.write_all(&v[index].to_le_bytes())?,
        AttributeData::U16(v) => writer.write_all(&v[index].to_le_bytes())?,
        AttributeData::U32(v) => writer.write_all(&v[index].to_le_bytes())?,
        AttributeData::I32(v) => writer.write_all(&v[index].to_le_bytes())?,
    }
    Ok(())
}

/// Writes a cloud to PLY (`binary_little_endian`).
///
/// Coordinates are written as stored, that is, relative to the origin;
/// the origin itself goes into a comment. A round trip through
/// [`read_ply`] returns bit-for-bit the same values.
pub fn write_ply(cloud: &PointCloud, path: &Path) -> Result<(), IoError> {
    let mut writer = BufWriter::new(File::create(path)?);

    writeln!(writer, "ply")?;
    writeln!(writer, "format binary_little_endian 1.0")?;
    writeln!(writer, "comment written by rigidity")?;
    let origin = cloud.origin();
    if origin != Vector3::zeros() {
        writeln!(
            writer,
            "comment {ORIGIN_COMMENT} {:.17} {:.17} {:.17}",
            origin.x, origin.y, origin.z
        )?;
    }
    writeln!(writer, "element vertex {}", cloud.len())?;
    for axis in ["x", "y", "z"] {
        writeln!(writer, "property float {axis}")?;
    }
    for attribute in cloud.attributes() {
        writeln!(
            writer,
            "property {} {}",
            attribute_scalar_type(&attribute.data).ply_name(),
            attribute.name
        )?;
    }
    writeln!(writer, "end_header")?;

    let (xs, ys, zs) = cloud.columns();
    for i in 0..cloud.len() {
        writer.write_all(&xs[i].to_le_bytes())?;
        writer.write_all(&ys[i].to_le_bytes())?;
        writer.write_all(&zs[i].to_le_bytes())?;
        for attribute in cloud.attributes() {
            write_attribute_value(&mut writer, &attribute.data, i)?;
        }
    }
    writer.flush()?;
    Ok(())
}
