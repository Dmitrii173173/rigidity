//! Point-cloud input and output.
//!
//! The heavy format dependencies are isolated here; the core knows nothing
//! about files.
//!
//! PLY is parsed by our own reader — `ply-rs` has not been updated since
//! 2020, and the format is simple enough that a six-year-old dependency
//! costs more than the code does. LAS is read through the `las` crate,
//! which handles non-trivial headers, coordinate scaling and LAZ
//! compression.

pub mod csv;
pub mod e57_format;
pub mod las_format;
pub mod pcd;
pub mod ply;

pub use csv::{read_csv, read_poses};
pub use e57_format::{read_e57, write_e57};
pub use las_format::{read_las, write_las};
pub use pcd::{read_pcd, write_pcd};
pub use ply::{read_ply, write_ply};

use std::path::Path;

use rigidity_core::PointCloud;

/// Reads a cloud, choosing the format by the file's extension.
///
/// The extension is all there is to go on and all anyone uses. A reader
/// that sniffed the contents would be right more often and would also
/// silently open a file the person did not mean to open.
pub fn read(path: &Path) -> Result<PointCloud, IoError> {
    match extension(path).as_str() {
        "ply" => read_ply(path),
        "las" | "laz" => read_las(path),
        "e57" => read_e57(path),
        "pcd" => read_pcd(path),
        "csv" | "txt" => read_csv(path),
        other => Err(IoError::UnknownFormat(other.to_owned())),
    }
}

/// Writes a cloud, choosing the format by the file's extension.
pub fn write(cloud: &PointCloud, path: &Path) -> Result<(), IoError> {
    match extension(path).as_str() {
        "ply" => write_ply(cloud, path),
        "las" | "laz" => write_las(cloud, path),
        "e57" => write_e57(cloud, path),
        "pcd" => write_pcd(cloud, path),
        other => Err(IoError::UnknownFormat(other.to_owned())),
    }
}

/// Every extension `read` understands, for a file dialog to offer.
pub const READABLE: &[&str] = &["ply", "las", "laz", "e57", "pcd", "csv", "txt"];

/// Every extension `write` understands.
pub const WRITABLE: &[&str] = &["ply", "las", "laz", "e57", "pcd"];

fn extension(path: &Path) -> String {
    path.extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
}

/// Read and write errors.
#[derive(Debug, thiserror::Error)]
pub enum IoError {
    /// A filesystem error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// The file does not begin with the `ply` signature.
    #[error("not a PLY file: the first line is not \"ply\"")]
    NotPly,
    /// Unsupported PLY format.
    #[error("PLY format \"{0}\" is not supported: need ascii or binary_little_endian")]
    UnsupportedFormat(String),
    /// The header is malformed.
    #[error("malformed PLY header: {0}")]
    BadHeader(String),
    /// Unknown property type.
    #[error("unknown PLY property type: \"{0}\"")]
    UnknownPropertyType(String),
    /// The `vertex` element carries no coordinates.
    #[error("the vertex element is missing the x, y, z properties")]
    MissingCoordinates,
    /// A list property inside `vertex`.
    #[error("list properties inside the vertex element are not supported")]
    ListInVertex,
    /// The first element is not `vertex`.
    #[error("the first PLY element must be vertex, found \"{0}\"")]
    VertexNotFirst(String),
    /// The data end earlier than the header promised.
    #[error("truncated data: need {expected} bytes, {actual} available")]
    Truncated {
        /// How many bytes the header requires.
        expected: usize,
        /// How many bytes are present.
        actual: usize,
    },
    /// A number could not be parsed.
    #[error("could not parse the number \"{0}\"")]
    BadNumber(String),
    /// An error raised by the `las` crate.
    #[error("LAS error: {0}")]
    Las(String),
    /// An error raised by the `e57` crate.
    #[error("E57 error: {0}")]
    E57(String),
    /// The PCD header is malformed.
    #[error("malformed PCD header: {0}")]
    BadPcd(String),
    /// A PCD field or layout this reader does not handle.
    #[error("unsupported PCD data: \"{0}\"")]
    UnsupportedPcd(String),
    /// The extension names no format this crate knows.
    ///
    /// The readable ones are listed in [`READABLE`], the writable ones in
    /// [`WRITABLE`]; the message names the extension rather than reciting
    /// the list, because a file dialog offers the list already.
    #[error("unknown format: \"{0}\"")]
    UnknownFormat(String),
    /// An error while building the cloud.
    #[error(transparent)]
    Cloud(#[from] rigidity_core::CloudError),
}
