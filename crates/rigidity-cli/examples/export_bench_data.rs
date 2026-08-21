//! Prepares the shared input for the implementation comparison.
//!
//! Every entrant receives the same two PLY files and the same ground-truth
//! pose. None has access to the generation parameters.

use nalgebra::Vector6;
use rigidity_core::PointCloud;
use rigidity_core::lie::Se3;
use rigidity_io::write_ply;
use rigidity_scenes::{Scene, SceneKind, SceneParams};
use std::path::Path;

fn main() {
    let total: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(1_000_000);

    let scene = Scene::generate(
        SceneKind::Corner,
        SceneParams {
            points_per_face: total / 3,
            noise_sigma: 5e-4,
            ..SceneParams::default()
        },
    );
    let truth = Se3::exp(&Vector6::new(0.021, -0.013, 0.017, 0.011, -0.007, 0.009));

    let inverse = truth.inverse();
    let mut source = PointCloud::with_capacity(scene.len());
    for index in 0..scene.len() {
        source.push(inverse.transform_point(&scene.points[index]));
    }

    let directory = Path::new("bench-external/data");
    write_ply(&scene.cloud, &directory.join("target.ply")).unwrap();
    write_ply(&source, &directory.join("source.ply")).unwrap();

    // The true transform, as the rows of a 4×4 matrix.
    let matrix = truth.matrix();
    let mut text = String::new();
    for row in 0..4 {
        for column in 0..4 {
            text.push_str(&format!("{:.17e} ", matrix[(row, column)]));
        }
        text.push('\n');
    }
    std::fs::write(directory.join("truth.txt"), text).unwrap();

    println!("points: {}", scene.len());
    println!(
        "true translation {:.1} mm, rotation {:.3}°",
        truth.translation().norm() * 1e3,
        truth.rotation().log().norm().to_degrees()
    );
    println!("written to {}", directory.display());
}
