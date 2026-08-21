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
pub mod las_format;
pub mod ply;

pub use csv::{read_csv, read_poses};
pub use las_format::read_las;
pub use ply::{read_ply, write_ply};

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
    /// An error while building the cloud.
    #[error(transparent)]
    Cloud(#[from] rigidity_core::CloudError),
}
