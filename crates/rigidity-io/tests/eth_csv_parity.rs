//! The dispatcher's `.csv` entry changed hands. It has to read the same file.
//!
//! `read` used to send `.csv` to `read_csv` and now sends it to `read_text`.
//! `read_csv` was written for the ETH ASL datasets and is still exported, so
//! the two readers have to agree on exactly the shape of file it was written
//! for — otherwise the change is a silent regression on the only real data
//! this project has been measured against.
//!
//! The fixture is written here rather than read from the datasets: those
//! are not in the repository, and a test that skips when its input is
//! missing is a test that passes for the wrong reason.

use nalgebra::Vector3;
use rigidity_core::PointCloud;

/// A file in the ETH Hauptgebäude layout: a named header, seven columns,
/// coordinates in the middle three, and a metre-scale scene offset far
/// enough from zero to matter.
fn eth_shaped(points: usize) -> String {
    let mut text = String::from("Time_in_sec,x,y,z,Intensities,2DscanId,PointId\n");
    for index in 0..points {
        let angle = index as f64 * 0.017;
        let (x, y, z) = (
            512_345.678_9 + angle.cos() * 3.0,
            4_123_456.789_1 + angle.sin() * 3.0,
            231.5 + index as f64 * 0.001,
        );
        text.push_str(&format!(
            "1314117928.{index:08},{x:.8},{y:.8},{z:.8},-1.00000000,{index},{index}\n"
        ));
    }
    text
}

fn temporary(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("rigidity-io-eth-parity-{name}"))
}

#[test]
fn the_text_reader_agrees_with_the_csv_reader_on_an_eth_shaped_file() {
    let path = temporary("scan.csv");
    std::fs::write(&path, eth_shaped(2_000)).expect("a temporary file");

    let old = rigidity_io::read_csv(&path).expect("the csv reader");
    let new = rigidity_io::read_text(&path).expect("the text reader");
    std::fs::remove_file(&path).ok();

    assert_eq!(old.len(), 2_000, "the fixture lost points on the way in");
    assert_eq!(old.len(), new.len(), "the point count changed");
    assert_eq!(old.origin(), new.origin(), "the origin changed");
    let worst = (0..old.len())
        .map(|index| (old.point(index) - new.point(index)).norm())
        .fold(0.0f64, f64::max);
    assert_eq!(worst, 0.0, "coordinates differ by up to {worst} m");
}

/// And the same file, written back out and read again, is still that file.
///
/// `.csv` became writable with `.txt`; this is the georeferenced half of
/// that claim at the coordinates the fixture uses, where the gap between
/// adjacent `f32`s is a quarter of a metre.
#[test]
fn a_georeferenced_scan_survives_being_written_as_csv() {
    let source = temporary("source.csv");
    std::fs::write(&source, eth_shaped(2_000)).expect("a temporary file");
    let before = rigidity_io::read(&source).expect("read");
    std::fs::remove_file(&source).ok();

    let written = temporary("written.csv");
    rigidity_io::write(&before, &written).expect("write");
    let after = rigidity_io::read(&written).expect("read back");
    std::fs::remove_file(&written).ok();

    assert_eq!(before.len(), after.len());
    let worst = (0..before.len())
        .map(|index| (before.point(index) - after.point(index)).norm())
        .fold(0.0f64, f64::max);
    // The fixture is written with eight decimals, so the coordinates are on
    // a nanometre grid before either reader sees them; the round trip below
    // is the only place precision could be lost, and it loses none.
    assert!(worst < 1e-9, "the round trip lost {worst} m");
}

/// The writer is fed a cloud built in memory, not one read from a file.
#[test]
fn a_cloud_that_never_was_a_file_writes_as_text() {
    let mut cloud = PointCloud::with_origin(Vector3::new(512_345.0, 4_123_456.0, 231.0));
    for index in 0..500 {
        let angle = index as f64 * 0.05;
        cloud.push(Vector3::new(
            512_345.0 + angle.cos(),
            4_123_456.0 + angle.sin(),
            231.0 + index as f64 * 0.002,
        ));
    }
    let path = temporary("memory.txt");
    rigidity_io::write(&cloud, &path).expect("write");
    let back = rigidity_io::read(&path).expect("read back");
    std::fs::remove_file(&path).ok();

    assert_eq!(cloud.len(), back.len());
    let worst = (0..cloud.len())
        .map(|index| (cloud.point(index) - back.point(index)).norm())
        .fold(0.0f64, f64::max);
    assert!(worst < 1e-6, "the round trip lost {worst} m");
}
