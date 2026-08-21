//! Structure-of-arrays storage for point clouds.

use nalgebra::Vector3;

/// Errors from building or modifying a cloud.
#[derive(Debug, thiserror::Error)]
pub enum CloudError {
    /// The coordinate arrays have different lengths.
    #[error("coordinate arrays differ in length: x={x}, y={y}, z={z}")]
    MismatchedCoordinates {
        /// Length of the x array.
        x: usize,
        /// Length of the y array.
        y: usize,
        /// Length of the z array.
        z: usize,
    },
    /// An attribute column does not match the point count.
    #[error("attribute \"{name}\" has length {actual}, expected {expected}")]
    MismatchedAttribute {
        /// Attribute name.
        name: String,
        /// Actual length.
        actual: usize,
        /// Expected length.
        expected: usize,
    },
    /// An attribute of that name already exists.
    #[error("attribute \"{0}\" already exists")]
    DuplicateAttribute(String),
    /// Invalid voxel size.
    #[error("voxel size must be positive and finite, got {0}")]
    InvalidVoxelSize(f64),
    /// A coordinate is not a finite number.
    #[error("point {index} has a non-finite coordinate")]
    NonFinite {
        /// Index of the point.
        index: usize,
    },
}

/// A column of attribute values.
///
/// The column type is preserved as read: turning a `u8` colour into `f32`
/// is a decision for application code, not for storage.
#[derive(Debug, Clone, PartialEq)]
pub enum AttributeData {
    /// 32-bit floating point.
    F32(Vec<f32>),
    /// 64-bit floating point.
    F64(Vec<f64>),
    /// Unsigned 8-bit.
    U8(Vec<u8>),
    /// Unsigned 16-bit.
    U16(Vec<u16>),
    /// Unsigned 32-bit.
    U32(Vec<u32>),
    /// Signed 32-bit.
    I32(Vec<i32>),
}

impl AttributeData {
    /// Number of elements in the column.
    pub fn len(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::F64(v) => v.len(),
            Self::U8(v) => v.len(),
            Self::U16(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::I32(v) => v.len(),
        }
    }

    /// Whether the column is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A named attribute column.
#[derive(Debug, Clone, PartialEq)]
pub struct Attribute {
    /// The name as written in the file.
    pub name: String,
    /// The values.
    pub data: AttributeData,
}

/// A point cloud.
///
/// # Layout
///
/// Coordinates live in three separate arrays rather than in an array of
/// structs. At hundreds of millions of points that matters: filtering by a
/// mask does not drag unused attributes along, and the coordinate arrays
/// map directly onto SIMD registers and GPU buffers.
///
/// # Precision and origin
///
/// Coordinates are stored as `f32` **relative to** [`origin`](Self::origin),
/// which is kept in `f64`. The reason is georeferenced data: at a
/// coordinate of 500 000 m the `f32` step is about 3 cm, so storing
/// absolute coordinates directly would destroy millimetre accuracy before
/// the first computation. Offset storage keeps both the halved memory and
/// the absolute accuracy.
///
/// [`point`](Self::point) returns the absolute coordinate in `f64`;
/// [`local`](Self::local) returns the offset one.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PointCloud {
    origin: Vector3<f64>,
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,
    attributes: Vec<Attribute>,
}

impl PointCloud {
    /// An empty cloud whose origin is at zero.
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty cloud with the given origin.
    pub fn with_origin(origin: Vector3<f64>) -> Self {
        Self {
            origin,
            ..Self::default()
        }
    }

    /// Moves the origin while preserving absolute point coordinates.
    ///
    /// The operation loses precision, since coordinates are recomputed
    /// through `f32`. It is worth calling once, right after reading a file.
    pub fn rebase(&mut self, new_origin: Vector3<f64>) {
        let shift = self.origin - new_origin;
        let (dx, dy, dz) = (shift.x as f32, shift.y as f32, shift.z as f32);
        for i in 0..self.len() {
            self.x[i] += dx;
            self.y[i] += dy;
            self.z[i] += dz;
        }
        self.origin = new_origin;
    }

    /// An empty cloud with capacity reserved.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            origin: Vector3::zeros(),
            x: Vec::with_capacity(capacity),
            y: Vec::with_capacity(capacity),
            z: Vec::with_capacity(capacity),
            attributes: Vec::new(),
        }
    }

    /// Builds a cloud from three coordinate arrays.
    ///
    /// Coordinates are interpreted as offsets from `origin`.
    pub fn from_columns(
        origin: Vector3<f64>,
        x: Vec<f32>,
        y: Vec<f32>,
        z: Vec<f32>,
    ) -> Result<Self, CloudError> {
        if x.len() != y.len() || y.len() != z.len() {
            return Err(CloudError::MismatchedCoordinates {
                x: x.len(),
                y: y.len(),
                z: z.len(),
            });
        }
        Ok(Self {
            origin,
            x,
            y,
            z,
            attributes: Vec::new(),
        })
    }

    /// Number of points.
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// Whether the cloud is empty.
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// The origin the stored coordinates are relative to.
    pub fn origin(&self) -> Vector3<f64> {
        self.origin
    }

    /// Absolute coordinate of a point.
    pub fn point(&self, index: usize) -> Vector3<f64> {
        self.origin + self.local(index)
    }

    /// Coordinate of a point relative to [`origin`](Self::origin).
    pub fn local(&self, index: usize) -> Vector3<f64> {
        Vector3::new(
            f64::from(self.x[index]),
            f64::from(self.y[index]),
            f64::from(self.z[index]),
        )
    }

    /// The coordinate columns.
    pub fn columns(&self) -> (&[f32], &[f32], &[f32]) {
        (&self.x, &self.y, &self.z)
    }

    /// Iterator over absolute coordinates.
    pub fn iter(&self) -> impl Iterator<Item = Vector3<f64>> + '_ {
        (0..self.len()).map(|i| self.point(i))
    }

    /// Appends a point given in absolute coordinates.
    ///
    /// # Panics
    ///
    /// If the cloud already carries attributes: appending a point without
    /// attribute values would break column consistency.
    pub fn push(&mut self, point: Vector3<f64>) {
        assert!(
            self.attributes.is_empty(),
            "pushing into a cloud with attributes would break column lengths"
        );
        let local = point - self.origin;
        self.x.push(local.x as f32);
        self.y.push(local.y as f32);
        self.z.push(local.z as f32);
    }

    /// All attributes.
    pub fn attributes(&self) -> &[Attribute] {
        &self.attributes
    }

    /// An attribute by name.
    pub fn attribute(&self, name: &str) -> Option<&Attribute> {
        self.attributes.iter().find(|a| a.name == name)
    }

    /// Appends an attribute column.
    pub fn push_attribute(&mut self, attribute: Attribute) -> Result<(), CloudError> {
        if attribute.data.len() != self.len() {
            return Err(CloudError::MismatchedAttribute {
                name: attribute.name,
                actual: attribute.data.len(),
                expected: self.len(),
            });
        }
        if self.attribute(&attribute.name).is_some() {
            return Err(CloudError::DuplicateAttribute(attribute.name));
        }
        self.attributes.push(attribute);
        Ok(())
    }

    /// Checks that every coordinate is finite.
    ///
    /// Called by algorithms for which a NaN means not "a bad point" but a
    /// silently corrupted result — voxelisation, for instance, where
    /// `NaN as i64` yields zero and the point lands in an arbitrary cell.
    pub fn check_finite(&self) -> Result<(), CloudError> {
        for i in 0..self.len() {
            if !(self.x[i].is_finite() && self.y[i].is_finite() && self.z[i].is_finite()) {
                return Err(CloudError::NonFinite { index: i });
            }
        }
        Ok(())
    }

    /// Axis-aligned bounding box in absolute coordinates.
    pub fn bounds(&self) -> Option<(Vector3<f64>, Vector3<f64>)> {
        if self.is_empty() {
            return None;
        }
        let mut min = self.point(0);
        let mut max = min;
        for i in 1..self.len() {
            let p = self.point(i);
            min = min.inf(&p);
            max = max.sup(&p);
        }
        Some((min, max))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Offset storage keeps millimetres at georeferenced coordinates while
    /// plain `f32` does not. That is the entire reason `origin` exists.
    #[test]
    fn origin_preserves_precision_at_utm_scale() {
        let absolute = Vector3::new(499_123.456_7, 5_432_198.765_4, 231.0);
        let mut cloud = PointCloud::with_origin(Vector3::new(499_000.0, 5_432_000.0, 0.0));
        cloud.push(absolute);

        let error = (cloud.point(0) - absolute).norm();
        assert!(error < 1e-4, "offset storage: error {error:.3e} m");

        // For comparison, the same without the offset.
        let naive = f64::from(absolute.x as f32) - absolute.x;
        assert!(
            naive.abs() > 1e-2,
            "plain f32 must lose centimetres here, lost {naive:.3e}"
        );
    }

    #[test]
    fn mismatched_columns_are_rejected() {
        let err =
            PointCloud::from_columns(Vector3::zeros(), vec![1.0, 2.0], vec![1.0], vec![1.0, 2.0]);
        assert!(matches!(err, Err(CloudError::MismatchedCoordinates { .. })));
    }

    #[test]
    fn attribute_length_is_checked() {
        let mut cloud =
            PointCloud::from_columns(Vector3::zeros(), vec![0.0; 3], vec![0.0; 3], vec![0.0; 3])
                .unwrap();
        let bad = Attribute {
            name: "intensity".into(),
            data: AttributeData::U16(vec![1, 2]),
        };
        assert!(cloud.push_attribute(bad).is_err());
    }
}
