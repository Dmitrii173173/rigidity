//! PLY round trip: what is written and what is read back agree bit for bit.

use std::fs;
use std::path::PathBuf;

use nalgebra::Vector3;
use rigidity_core::{Attribute, AttributeData, PointCloud};
use rigidity_io::{IoError, read_ply, write_ply};

fn temp_path(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("rigidity-io-tests");
    fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn sample_cloud(origin: Vector3<f64>) -> PointCloud {
    let mut cloud = PointCloud::with_origin(origin);
    for i in 0..97 {
        let t = i as f64 * 0.137;
        cloud.push(origin + Vector3::new(t.sin() * 10.0, t.cos() * 7.5, t * 0.31));
    }
    cloud
}

fn assert_columns_identical(a: &PointCloud, b: &PointCloud) {
    let (ax, ay, az) = a.columns();
    let (bx, by, bz) = b.columns();
    for (left, right) in [(ax, bx), (ay, by), (az, bz)] {
        let left: Vec<u32> = left.iter().map(|v| v.to_bits()).collect();
        let right: Vec<u32> = right.iter().map(|v| v.to_bits()).collect();
        assert_eq!(left, right, "coordinates differ bit for bit");
    }
}

#[test]
fn binary_roundtrip_preserves_coordinates_bit_for_bit() {
    let cloud = sample_cloud(Vector3::zeros());
    let path = temp_path("binary.ply");
    write_ply(&cloud, &path).unwrap();
    let restored = read_ply(&path).unwrap();

    assert_eq!(restored.len(), cloud.len());
    assert_eq!(restored.origin(), cloud.origin());
    assert_columns_identical(&cloud, &restored);
}

#[test]
fn roundtrip_preserves_origin_and_attributes() {
    let mut cloud = sample_cloud(Vector3::new(499_000.0, 5_432_000.0, 120.5));
    let count = cloud.len();
    cloud
        .push_attribute(Attribute {
            name: "intensity".into(),
            data: AttributeData::U16((0..count as u16).collect()),
        })
        .unwrap();
    cloud
        .push_attribute(Attribute {
            name: "curvature".into(),
            data: AttributeData::F32((0..count).map(|i| i as f32 * 0.25).collect()),
        })
        .unwrap();

    let path = temp_path("attributes.ply");
    write_ply(&cloud, &path).unwrap();
    let restored = read_ply(&path).unwrap();

    assert_eq!(restored.origin(), cloud.origin());
    assert_columns_identical(&cloud, &restored);
    assert_eq!(restored.attributes().len(), 2);
    assert_eq!(
        restored.attribute("intensity").unwrap().data,
        cloud.attribute("intensity").unwrap().data
    );
    assert_eq!(
        restored.attribute("curvature").unwrap().data,
        cloud.attribute("curvature").unwrap().data
    );

    // Absolute coordinates survived the georeferenced scale.
    for i in 0..cloud.len() {
        assert!((restored.point(i) - cloud.point(i)).norm() < 1e-4);
    }
}

#[test]
fn reads_ascii() {
    let path = temp_path("ascii.ply");
    fs::write(
        &path,
        "ply\n\
         format ascii 1.0\n\
         comment example\n\
         element vertex 3\n\
         property float x\n\
         property float y\n\
         property float z\n\
         property uchar intensity\n\
         end_header\n\
         1.5 2.5 3.5 10\n\
         -1.0 0.0 1.0 20\n\
         0.25 0.5 0.75 30\n",
    )
    .unwrap();

    let cloud = read_ply(&path).unwrap();
    assert_eq!(cloud.len(), 3);
    assert!((cloud.point(0) - Vector3::new(1.5, 2.5, 3.5)).norm() < 1e-7);
    assert!((cloud.point(2) - Vector3::new(0.25, 0.5, 0.75)).norm() < 1e-7);
    assert_eq!(
        cloud.attribute("intensity").unwrap().data,
        AttributeData::U8(vec![10, 20, 30])
    );
}

#[test]
fn rejects_malformed_files() {
    let path = temp_path("bad.ply");

    fs::write(&path, "not a ply file\n").unwrap();
    assert!(matches!(read_ply(&path), Err(IoError::NotPly)));

    fs::write(
        &path,
        "ply\nformat binary_big_endian 1.0\nelement vertex 1\n\
         property float x\nproperty float y\nproperty float z\nend_header\n",
    )
    .unwrap();
    assert!(matches!(
        read_ply(&path),
        Err(IoError::UnsupportedFormat(_))
    ));

    fs::write(
        &path,
        "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n0.0\n",
    )
    .unwrap();
    assert!(matches!(read_ply(&path), Err(IoError::MissingCoordinates)));

    // The header promises 4 points; there is data for one.
    fs::write(
        &path,
        "ply\nformat binary_little_endian 1.0\nelement vertex 4\n\
         property float x\nproperty float y\nproperty float z\nend_header\n",
    )
    .unwrap();
    fs::write(&path, {
        let mut bytes = fs::read(&path).unwrap();
        bytes.extend_from_slice(&[0u8; 12]);
        bytes
    })
    .unwrap();
    assert!(matches!(read_ply(&path), Err(IoError::Truncated { .. })));
}
