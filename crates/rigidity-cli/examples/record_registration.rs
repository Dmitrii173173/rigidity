//! Recording the course of a registration into Rerun.
//!
//! Builds only with the `viz` feature:
//!
//! ```text
//! cargo run -p rigidity-cli --features viz --release \
//!     --example record_registration -- corridor.rrd
//! ```
//!
//! It writes an `.rrd` file that the Rerun viewer opens. On the shared
//! "iteration" axis you can watch the solution approach the answer and see
//! what the spectrum does meanwhile: if a direction is unobservable, its
//! spread does not fall along with the others.

use std::ops::ControlFlow;
use std::path::PathBuf;

use nalgebra::{Vector3, Vector6};
use rigidity_core::icp::{IcpConfig, Kernel, register_observed, surface};
use rigidity_core::lie::Se3;
use rigidity_core::normals::estimate_normals;
use rigidity_core::observability::{Correspondence, ObservabilityCriteria, analyse};
use rigidity_core::{NeighborSearch, PointCloud};
use rigidity_scenes::{Scene, SceneKind, SceneParams};
use rigidity_spatial::KdTree;
use rigidity_viz::Recorder;

const MAX_DISTANCE: f64 = 0.5;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output: PathBuf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "corridor.rrd".to_string())
        .into();

    let scene = Scene::generate(
        SceneKind::Corridor,
        SceneParams {
            points_per_face: 8_000,
            noise_sigma: 2e-3,
            ..SceneParams::default()
        },
    );
    let truth = Se3::exp(&Vector6::new(0.03, 0.02, 0.01, 0.009, 0.005, 0.007));
    // The source gets **its own** noise: otherwise it would match the
    // target point for point, RMSE would come out zero, and the
    // demonstration would mean nothing.
    let inverse = truth.inverse();
    let mut rng = rigidity_scenes::rng::Rng::new(0xBEEF);
    let mut source = PointCloud::with_capacity(scene.len());
    for index in 0..scene.len() {
        let jitter = Vector3::new(rng.normal(2e-3), rng.normal(2e-3), rng.normal(2e-3));
        source.push(inverse.transform_point(&(scene.points[index] + jitter)));
    }

    let target_tree = KdTree::build(&scene.cloud)?;
    let source_tree = KdTree::build(&source)?;
    let target_normals = estimate_normals(&scene.cloud, &target_tree, 16);
    let source_normals = estimate_normals(&source, &source_tree, 16);

    let criteria = ObservabilityCriteria {
        noise_sigma: 2e-3,
        tolerance: 1e-3,
    };
    let recorder = Recorder::save("rigidity", &output)?;
    recorder.set_iteration(0);
    recorder.log_cloud("scene/target", &scene.cloud, [42, 120, 214])?;

    let kernel = Kernel::Huber(0.05);
    let config = IcpConfig {
        kernel,
        max_correspondence_distance: MAX_DISTANCE,
        min_normal_cosine: 0.0,
        max_iterations: 30,
        ..IcpConfig::default()
    };

    let observe = |report: &rigidity_core::icp::IterationReport| {
        recorder.set_iteration(report.iteration as i64);

        let mut moved = PointCloud::with_capacity(source.len());
        for index in 0..source.len() {
            moved.push(report.pose.transform_point(&source.point(index)));
        }
        let _ = recorder.log_cloud("scene/source", &moved, [235, 104, 52]);
        let _ = recorder.log_pose("scene/pose", &report.pose);

        let conditioning = analyse(source.len(), kernel, |index| {
            let point = report.pose.transform_point(&source.point(index));
            let mut found = Vec::with_capacity(1);
            target_tree.knn_into(&point, 1, &mut found);
            let nearest = found.first()?;
            if nearest.distance_squared > MAX_DISTANCE * MAX_DISTANCE {
                return None;
            }
            let matched = nearest.index as usize;
            Some(Correspondence {
                point,
                normal: target_normals[matched],
                residual: target_normals[matched].dot(&(point - scene.cloud.point(matched))),
            })
        });
        if let Some(analysis) = conditioning {
            let _ = recorder.log_conditioning("conditioning", &analysis.conditioning, &criteria);
        }
        ControlFlow::Continue(())
    };

    let result = register_observed(
        &surface(&source, &source_normals),
        &surface(&scene.cloud, &target_normals),
        &target_tree,
        Se3::identity(),
        &config,
        observe,
    );

    let error = (truth * result.pose.inverse()).log();
    println!(
        "iterations {}, RMSE {:.5} m, pose error {:.4} m",
        result.iterations,
        result.rmse,
        error.fixed_rows::<3>(0).norm()
    );
    println!("written: {}", output.display());
    println!("open with: rerun {}", output.display());
    Ok(())
}
