//! The iso-accuracy protocol: time to a required accuracy.
//!
//! # Why not "time to convergence"
//!
//! Every implementation has its own stopping criterion, its own
//! correspondence rejection and its own iteration cap. Comparing by those
//! measures a difference in settings, not in code: loosening the stopping
//! threshold is enough to "win".
//!
//! What is fixed here is the **result**, not the process: a required pose
//! accuracy is set, and the time to reach it is measured. Everything
//! counts, preprocessing included: downsampling, normal estimation and
//! index construction. Otherwise a configuration that shifts work into
//! preparation would look free.

use std::time::Instant;

use nalgebra::Vector6;
use rigidity_core::PointCloud;
use rigidity_core::icp::{IcpConfig, Kernel, register_observed, surface};
use rigidity_core::lie::Se3;
use rigidity_core::normals::estimate_normals;
use rigidity_core::voxel::voxel_downsample;
use rigidity_scenes::{Scene, SceneKind, SceneParams};
use rigidity_spatial::KdTree;

/// The required translation accuracy levels, metres.
const TARGETS: [f64; 3] = [1e-2, 1e-3, 1e-4];

fn translation_error(found: &Se3, truth: &Se3) -> f64 {
    (*truth * found.inverse()).log().fixed_rows::<3>(0).norm()
}

struct Outcome {
    label: String,
    points: usize,
    preparation: f64,
    /// Time to each accuracy level; `None` if the level was not reached.
    reached: [Option<f64>; TARGETS.len()],
    final_error: f64,
    iterations: usize,
}

fn run(label: &str, voxel: Option<f64>, scene: &Scene, truth: &Se3) -> Outcome {
    let start = Instant::now();

    let inverse = truth.inverse();
    let mut raw_source = PointCloud::with_capacity(scene.len());
    for index in 0..scene.len() {
        raw_source.push(inverse.transform_point(&scene.points[index]));
    }

    let (source, target) = match voxel {
        Some(size) => (
            voxel_downsample(&raw_source, size).unwrap(),
            voxel_downsample(&scene.cloud, size).unwrap(),
        ),
        None => (raw_source, scene.cloud.clone()),
    };

    let source_tree = KdTree::build(&source).unwrap();
    let target_tree = KdTree::build(&target).unwrap();
    let source_normals = estimate_normals(&source, &source_tree, 16);
    let target_normals = estimate_normals(&target, &target_tree, 16);
    let preparation = start.elapsed().as_secs_f64();

    let config = IcpConfig {
        kernel: Kernel::Squared,
        max_correspondence_distance: 0.5,
        max_iterations: 60,
        ..IcpConfig::default()
    };

    let mut reached: [Option<f64>; TARGETS.len()] = [None; TARGETS.len()];
    let mut final_error = f64::INFINITY;
    let mut iterations = 0usize;
    let solve_start = Instant::now();
    let result = register_observed(
        &surface(&source, &source_normals),
        &surface(&target, &target_normals),
        &target_tree,
        Se3::identity(),
        &config,
        |report| {
            let elapsed = preparation + solve_start.elapsed().as_secs_f64();
            let error = translation_error(&report.pose, truth);
            final_error = error;
            iterations = report.iteration;
            for (slot, target) in reached.iter_mut().zip(TARGETS.iter()) {
                if slot.is_none() && error < *target {
                    *slot = Some(elapsed);
                }
            }
        },
    );
    let _ = result;

    Outcome {
        label: label.to_string(),
        points: source.len(),
        preparation,
        reached,
        final_error,
        iterations,
    }
}

fn main() {
    let total: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(1_000_000);

    let scene = Scene::generate(
        SceneKind::Corner,
        SceneParams {
            points_per_face: total / 3,
            ..SceneParams::default()
        },
    );
    let truth = Se3::exp(&Vector6::new(0.021, -0.013, 0.017, 0.011, -0.007, 0.009));

    println!("iso-accuracy protocol");
    println!(
        "scene: trihedral corner, {} points, 2 m across",
        scene.len()
    );
    println!(
        "true translation: {:.1} mm, rotation {:.2}°",
        truth.translation().norm() * 1e3,
        truth.rotation().log().norm().to_degrees()
    );
    println!("total time measured: downsampling + normals + index + solve\n");

    let configurations: [(&str, Option<f64>); 4] = [
        ("full cloud", None),
        ("voxel 5 mm", Some(0.005)),
        ("voxel 20 mm", Some(0.02)),
        ("voxel 50 mm", Some(0.05)),
    ];

    println!(
        "{:<16} {:>9} {:>10} {:>10} {:>10} {:>10} {:>11} {:>5}",
        "configuration",
        "points",
        "prep, s",
        "to 10 mm",
        "to 1 mm",
        "to 0.1 mm",
        "final, m",
        "iter"
    );
    for (label, voxel) in configurations {
        let outcome = run(label, voxel, &scene, &truth);
        let show = |value: Option<f64>| match value {
            Some(seconds) => format!("{seconds:.3} s"),
            None => "—".to_string(),
        };
        println!(
            "{:<16} {:>9} {:>10.3} {:>10} {:>10} {:>10} {:>11.2e} {:>5}",
            outcome.label,
            outcome.points,
            outcome.preparation,
            show(outcome.reached[0]),
            show(outcome.reached[1]),
            show(outcome.reached[2]),
            outcome.final_error,
            outcome.iterations
        );
    }
}
