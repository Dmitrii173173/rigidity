//! Comparison entrant: rigidity. Protocol: `bench-external/PROTOCOL.md`.

use std::path::Path;
use std::time::Instant;

use nalgebra::{Matrix4, Vector3};
use rigidity_core::icp::{IcpConfig, Kernel, register, surface};
use rigidity_core::lie::{Se3, So3};
use rigidity_core::normals::estimate_normals;
use rigidity_core::voxel::voxel_downsample;
use rigidity_io::read_ply;
use rigidity_spatial::KdTree;

const VOXEL: f64 = 0.02;
const NEIGHBOURS: usize = 16;
const MAX_DISTANCE: f64 = 0.5;

fn load_truth(path: &Path) -> Matrix4<f64> {
    let text = std::fs::read_to_string(path).unwrap();
    let values: Vec<f64> = text
        .split_whitespace()
        .map(|token| token.parse().unwrap())
        .collect();
    Matrix4::from_row_slice(&values[..16])
}

fn translation_error(found: &Se3, truth: &Matrix4<f64>) -> f64 {
    let error = truth * found.matrix().try_inverse().unwrap();
    Vector3::new(error[(0, 3)], error[(1, 3)], error[(2, 3)]).norm()
}

struct Prepared {
    source: rigidity_core::PointCloud,
    target: rigidity_core::PointCloud,
    tree: KdTree,
    target_normals: Vec<Vector3<f64>>,
}

fn prepare(directory: &Path) -> Prepared {
    let source = read_ply(&directory.join("source.ply")).unwrap();
    let target = read_ply(&directory.join("target.ply")).unwrap();
    let source = voxel_downsample(&source, VOXEL).unwrap();
    let target = voxel_downsample(&target, VOXEL).unwrap();
    let tree = KdTree::build(&target).unwrap();
    let target_normals = estimate_normals(&target, &tree, NEIGHBOURS);
    Prepared {
        source,
        target,
        tree,
        target_normals,
    }
}

fn solve(prepared: &Prepared, iterations: usize) -> Se3 {
    // Source normals are not estimated: angle rejection is off, and the
    // other entrants do not compute them for point-to-plane either.
    let source_normals = vec![Vector3::zeros(); prepared.source.len()];
    let config = IcpConfig {
        kernel: Kernel::Squared,
        max_correspondence_distance: MAX_DISTANCE,
        min_normal_cosine: 0.0,
        max_iterations: iterations,
        translation_tolerance: 0.0,
        rotation_tolerance: 0.0,
        ..IcpConfig::default()
    };
    register(
        &surface(&prepared.source, &source_normals),
        &surface(&prepared.target, &prepared.target_normals),
        &prepared.tree,
        Se3::from_parts(So3::identity(), Vector3::zeros()),
        &config,
    )
    .pose
}

/// The full pipeline with exactly `iterations` iterations: what gets
/// measured.
fn pipeline(directory: &Path, iterations: usize) -> (Se3, f64) {
    let start = Instant::now();
    let source = read_ply(&directory.join("source.ply")).unwrap();
    let target = read_ply(&directory.join("target.ply")).unwrap();

    let source = voxel_downsample(&source, VOXEL).unwrap();
    let target = voxel_downsample(&target, VOXEL).unwrap();

    let tree = KdTree::build(&target).unwrap();
    let target_normals = estimate_normals(&target, &tree, NEIGHBOURS);
    let source_normals = vec![Vector3::zeros(); source.len()];

    let config = IcpConfig {
        kernel: Kernel::Squared,
        max_correspondence_distance: MAX_DISTANCE,
        min_normal_cosine: 0.0,
        max_iterations: iterations,
        translation_tolerance: 0.0,
        rotation_tolerance: 0.0,
        ..IcpConfig::default()
    };
    let result = register(
        &surface(&source, &source_normals),
        &surface(&target, &target_normals),
        &tree,
        Se3::from_parts(So3::identity(), Vector3::zeros()),
        &config,
    );
    (result.pose, start.elapsed().as_secs_f64())
}

fn main() {
    let directory = Path::new("bench-external/data");
    let truth = load_truth(&directory.join("truth.txt"));
    let targets = [1e-3f64, 1e-4];

    // Calibration is not timed, so preprocessing happens once.
    let prepared = prepare(directory);
    println!("# points after downsampling: {}", prepared.source.len());

    // A stage-by-stage breakdown of the total time: without it the
    // comparison table cannot be interpreted, since it is unclear what it
    // measures.
    {
        let mark = Instant::now();
        let raw_source = read_ply(&directory.join("source.ply")).unwrap();
        let raw_target = read_ply(&directory.join("target.ply")).unwrap();
        let reading = mark.elapsed().as_secs_f64();

        let mark = Instant::now();
        let small_source = voxel_downsample(&raw_source, VOXEL).unwrap();
        let small_target = voxel_downsample(&raw_target, VOXEL).unwrap();
        let thinning = mark.elapsed().as_secs_f64();

        let mark = Instant::now();
        let tree = KdTree::build(&small_target).unwrap();
        let index = mark.elapsed().as_secs_f64();

        let mark = Instant::now();
        let _ = estimate_normals(&small_target, &tree, NEIGHBOURS);
        let normals = mark.elapsed().as_secs_f64();

        let mark = Instant::now();
        let _ = solve(&prepared, 2);
        let solving = mark.elapsed().as_secs_f64();

        println!(
            "# stages, s: read {reading:.4}, downsample {thinning:.4}, \
index {index:.4}, normals {normals:.4}, two ICP iterations {solving:.4}"
        );
        let _ = (small_source, raw_source, raw_target);
    }

    for target in targets {
        let mut best: Option<usize> = None;
        for iterations in 1..=40 {
            if translation_error(&solve(&prepared, iterations), &truth) < target {
                best = Some(iterations);
                break;
            }
        }
        let Some(iterations) = best else {
            println!("impl=rigidity target={target:e} k=- seconds=- error=-");
            continue;
        };
        // Phase 2: the measurement.
        let (pose, seconds) = pipeline(directory, iterations);
        println!(
            "impl=rigidity target={target:e} k={iterations} seconds={seconds:.4} error={:.3e}",
            translation_error(&pose, &truth)
        );
    }
}
