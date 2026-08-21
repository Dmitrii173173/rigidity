//! LAS round trip.
//!
//! Bit-for-bit agreement is neither possible nor expected here: LAS stores
//! coordinates as integers scaled by a header value, so the format
//! quantises data by its very nature. The test checks that we land inside
//! that quantisation rather than beside it.

use std::fs;
use std::path::PathBuf;

use las::{Header, Point, Writer};
use nalgebra::Vector3;
use rigidity_io::read_las;

fn temp_path(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("rigidity-io-tests");
    fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn reference_points() -> Vec<(Vector3<f64>, u16)> {
    (0..250)
        .map(|i| {
            let t = i as f64 * 0.211;
            (
                Vector3::new(t.sin() * 120.0, t.cos() * 95.0, 30.0 + t * 0.7),
                (i * 7 % 65_535) as u16,
            )
        })
        .collect()
}

#[test]
fn las_roundtrip_within_format_quantisation() {
    let path = temp_path("roundtrip.las");
    let reference = reference_points();

    let mut writer = Writer::from_path(&path, Header::default()).unwrap();
    for (position, intensity) in &reference {
        writer
            .write_point(Point {
                x: position.x,
                y: position.y,
                z: position.z,
                intensity: *intensity,
                ..Default::default()
            })
            .unwrap();
    }
    writer.close().unwrap();

    let cloud = read_las(&path).unwrap();
    assert_eq!(cloud.len(), reference.len());

    // The origin sits at the centre of the bounds, not at zero.
    let (min, max) = cloud.bounds().unwrap();
    let centre = (min + max) * 0.5;
    assert!(
        (cloud.origin() - centre).norm() < 1.0,
        "origin {} is far from the centre of the bounds {}",
        cloud.origin(),
        centre
    );

    // Tolerance: the LAS quantisation step plus the f32 floor at these
    // coordinates.
    for (i, (expected, _)) in reference.iter().enumerate() {
        let error = (cloud.point(i) - expected).norm();
        assert!(error < 1e-2, "point {i}: deviation {error:.3e} m");
    }

    let intensity = cloud.attribute("intensity").unwrap();
    match &intensity.data {
        rigidity_core::AttributeData::U16(values) => {
            let expected: Vec<u16> = reference.iter().map(|(_, v)| *v).collect();
            assert_eq!(*values, expected);
        }
        other => panic!("intensity was read as {other:?}"),
    }
}
