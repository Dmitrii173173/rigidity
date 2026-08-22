//! Every format this crate writes must read back what it wrote.
//!
//! Millimetre fidelity, and for LAS a second requirement: a georeferenced
//! cloud has to keep its absolute coordinates. That is what the
//! `f32`-offset-from-an-`f64`-origin storage exists for, so a regression
//! here is a regression in the core's central invariant rather than a
//! detail of a file format.

use nalgebra::Vector3;
use rigidity_core::PointCloud;

/// A millimetre. Every format here is written from `f32` storage or to a
/// millimetre grid, and neither is worth pretending otherwise about.
const TOLERANCE: f64 = 1e-3;

/// A scene at the origin, spread over a few metres.
fn nearby() -> PointCloud {
    let mut cloud = PointCloud::new();
    for index in 0..2_000 {
        let angle = index as f64 * 0.017;
        cloud.push(Vector3::new(
            angle.cos() * 3.0,
            angle.sin() * 3.0,
            index as f64 * 0.001,
        ));
    }
    cloud
}

/// The same scene, where a survey would actually put it: half a million
/// metres east and four million north, which is an ordinary UTM
/// coordinate and the case `f32` alone cannot hold.
fn georeferenced() -> PointCloud {
    let far = Vector3::new(512_345.678, 4_123_456.789, 231.5);
    let mut cloud = PointCloud::with_origin(far);
    for index in 0..2_000 {
        let angle = index as f64 * 0.017;
        cloud.push(far + Vector3::new(angle.cos() * 3.0, angle.sin() * 3.0, index as f64 * 0.001));
    }
    cloud
}

fn temporary(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("rigidity-io-round-trip-{name}"))
}

fn worst_error(before: &PointCloud, after: &PointCloud) -> f64 {
    assert_eq!(before.len(), after.len(), "the point count changed");
    (0..before.len())
        .map(|index| (before.point(index) - after.point(index)).norm())
        .fold(0.0, f64::max)
}

#[test]
fn every_format_reads_back_what_it_wrote() {
    for extension in rigidity_io::WRITABLE {
        let cloud = nearby();
        let path = temporary(&format!("nearby.{extension}"));
        rigidity_io::write(&cloud, &path).unwrap_or_else(|e| panic!("{extension}: {e}"));
        let back = rigidity_io::read(&path).unwrap_or_else(|e| panic!("{extension}: {e}"));

        let error = worst_error(&cloud, &back);
        assert!(
            error < TOLERANCE,
            "{extension}: a point moved {error} m in the round trip"
        );
    }
}

/// The case the storage exists for: coordinates in the millions, kept to
/// the millimetre.
#[test]
fn a_georeferenced_cloud_keeps_its_coordinates() {
    for extension in rigidity_io::WRITABLE {
        let cloud = georeferenced();
        let path = temporary(&format!("far.{extension}"));
        rigidity_io::write(&cloud, &path).unwrap_or_else(|e| panic!("{extension}: {e}"));
        let back = rigidity_io::read(&path).unwrap_or_else(|e| panic!("{extension}: {e}"));

        let error = worst_error(&cloud, &back);
        assert!(
            error < TOLERANCE,
            "{extension}: half a million metres from zero, a point moved {error} m"
        );
        // And it came back where it was, not near the origin: a format
        // that quietly re-centres is worse than one that fails.
        let apart = (cloud.point(0) - back.point(0)).norm();
        assert!(
            apart < TOLERANCE,
            "{extension}: the cloud came back {apart} m away"
        );
    }
}

/// An unknown extension is refused rather than guessed at.
#[test]
fn an_unknown_extension_is_an_error() {
    let path = temporary("cloud.xyzzy");
    assert!(rigidity_io::write(&nearby(), &path).is_err());
    assert!(rigidity_io::read(&path).is_err());
}
