//! The residuals themselves, at a right minimum and at a wrong one.
//!
//! # Why this exists
//!
//! `median_absolute_residual` returns the median and nothing else, which is
//! the right shape for a detector: a rule with one number in it is a rule
//! somebody can apply. It is the wrong shape for showing why the rule
//! works. The claim underneath it is about the distribution — at the right
//! minimum what is left over is the sensor's own error, so half of it falls
//! inside the stated noise; at a wrong one it is the geometry disagreeing
//! with itself, and it does not — and a median cannot show a distribution.
//!
//! So this walks the same correspondences the detector walks, under the
//! same rejection rule, and writes every residual instead of their middle.
//!
//! ```text
//! cargo run --release -p rigidity-graph --example residuals -- <directory> <i> <j>
//! ```
//!
//! Three poses are reported for the pair `i → j`: what a cold start finds,
//! what the search finds, and the theodolite's. The first two are what a
//! system without truth could have; the third says which of them was right.
//! Output is CSV on stdout, one row per residual, so the histogram is drawn
//! wherever histograms are drawn.

use std::path::{Path, PathBuf};

use rigidity_core::lie::{Se3, So3};
use rigidity_core::nalgebra::{Matrix3, Matrix4, Vector3};
use rigidity_core::neighbors::NeighborSearch;
use rigidity_pipeline::{
    PrepareParams, Prepared, RegisterParams, SearchParams, prepare_cloud, register_globally,
    register_pair_observed,
};

/// Sensor noise along the normal, metres. The Hokuyo, as every other
/// measurement here takes it.
const NOISE: f64 = 0.03;

fn main() {
    let mut args = std::env::args().skip(1);
    let directory = PathBuf::from(args.next().unwrap_or_else(|| {
        eprintln!("usage: residuals <directory> <i> <j>");
        std::process::exit(2);
    }));
    let from: usize = args.next().and_then(|t| t.parse().ok()).unwrap_or(0);
    let to: usize = args.next().and_then(|t| t.parse().ok()).unwrap_or(1);

    let truth = read_truth(&directory.join("pose_scanner_leica.csv"), to + 1);
    let want = truth[from].inverse() * truth[to];
    let params = RegisterParams::default();
    let prepared: Vec<Prepared> = [from, to]
        .iter()
        .map(|index| read_scan(&directory, *index))
        .collect();
    let (moving, fixed) = (&prepared[1], &prepared[0]);

    let cold = register_pair_observed(moving, fixed, Se3::identity(), &params, |_| {
        std::ops::ControlFlow::Continue(())
    })
    .pose;
    let found = register_globally(moving, fixed, &params, &SearchParams::default())
        .map(|search| search.result.pose);

    eprintln!("noise {NOISE} m");
    report("cold", &cold, &want);
    if let Some(found) = &found {
        report("search", found, &want);
    }
    report("truth", &want, &want);

    println!("pose,residual");
    dump("cold", moving, fixed, &cold, &params);
    if let Some(found) = &found {
        dump("search", moving, fixed, found, &params);
    }
    dump("truth", moving, fixed, &want, &params);
}

fn report(name: &str, pose: &Se3, want: &Se3) {
    eprintln!(
        "{name:<7} {:.4} m from the theodolite",
        (pose.translation() - want.translation()).norm()
    );
}

/// Every residual at one pose, under the rejection rule the detector uses.
fn dump(name: &str, moving: &Prepared, fixed: &Prepared, pose: &Se3, params: &RegisterParams) {
    let limit = params.max_distance * params.max_distance;
    let mut found = Vec::with_capacity(1);
    for index in 0..moving.cloud.len() {
        let point = pose.transform_point(&moving.cloud.point(index));
        fixed.tree.knn_into(&point, 1, &mut found);
        let Some(nearest) = found.first() else {
            continue;
        };
        if nearest.distance_squared > limit {
            continue;
        }
        let matched = nearest.index as usize;
        let normal = fixed.normals[matched];
        let residual = normal.dot(&(point - fixed.cloud.point(matched))).abs();
        println!("{name},{residual:.6}");
    }
}

fn read_scan(directory: &Path, index: usize) -> Prepared {
    let path = directory.join(format!("Hokuyo_{index}.csv"));
    let raw = rigidity_io::read(&path).unwrap_or_else(|error| {
        eprintln!("{}: {error}", path.display());
        std::process::exit(1);
    });
    prepare_cloud(
        &raw,
        &PrepareParams {
            voxel: 0.05,
            neighbours: 16,
        },
    )
    .expect("the scan prepares")
}

fn read_truth(path: &Path, stations: usize) -> Vec<Se3> {
    let matrices = rigidity_io::read_poses(path).unwrap_or_else(|error| {
        eprintln!("{}: {error}", path.display());
        std::process::exit(1);
    });
    matrices[..stations].iter().map(from_matrix).collect()
}

fn from_matrix(matrix: &Matrix4<f64>) -> Se3 {
    let rotation: Matrix3<f64> = matrix.fixed_view::<3, 3>(0, 0).into();
    let translation: Vector3<f64> = matrix.fixed_view::<3, 1>(0, 3).into();
    Se3::from_parts(So3::from_matrix_unchecked(rotation), translation)
}
