//! The correspondences of one registration, in the form somebody else's
//! code asks for.
//!
//! # Why this exists
//!
//! Section VII of the paper says the comparison against published
//! degeneracy-aware methods is a comparison against our reimplementation of
//! their idea. `rigidity_harness::probabilistic` is that reimplementation,
//! of Hatleskog and Alexis (RA-L 2024), and a reimplementation can agree
//! with a paper and disagree with the code the authors published. The only
//! way to find out is to run theirs.
//!
//! Theirs is `ntnu-arl/drpm`: a header of three functions taking points,
//! normals, weights and per-point normal covariances. So this writes those
//! four things for one scan pair, at the pose our own registration found,
//! and prints what our reimplementation says about the same pair on the
//! side. Two programs, one input, two answers.
//!
//! Nothing is translated between the two: the C++ side builds its own
//! Hessian its own way, in its own `[δr; δt]` ordering, and the comparison
//! is of what each concludes rather than of intermediate matrices.
//!
//! ```text
//! cargo run --release -p rigidity-graph --example drpm_dump -- <directory> <i> <j> > pair.csv
//! ```

use std::path::{Path, PathBuf};

use rigidity_core::lie::Se3;
use rigidity_core::nalgebra::{Matrix3, SymmetricEigen, Vector3};
use rigidity_core::neighbors::NeighborSearch;
use rigidity_harness::probabilistic::{NoiseParams, probabilistic_information};
use rigidity_pipeline::{
    PrepareParams, Prepared, RegisterParams, prepare_cloud, register_pair_observed,
};

fn main() {
    let mut args = std::env::args().skip(1);
    let directory = PathBuf::from(args.next().unwrap_or_else(|| {
        eprintln!("usage: drpm_dump <directory> <i> <j>");
        std::process::exit(2);
    }));
    let from: usize = args.next().and_then(|t| t.parse().ok()).unwrap_or(0);
    let to: usize = args.next().and_then(|t| t.parse().ok()).unwrap_or(1);

    let params = RegisterParams::default();
    let noise = NoiseParams::default();
    let fixed = read_scan(&directory, from);
    let moving = read_scan(&directory, to);

    let pose = register_pair_observed(&moving, &fixed, Se3::identity(), &params, |_| {
        std::ops::ControlFlow::Continue(())
    })
    .pose;

    // The same neighbourhood covariance the reimplementation uses, so that
    // both sides are given the same normals *and* the same uncertainty
    // about them. Feeding theirs an isotropic covariance while ours has a
    // per-point one would compare the inputs, not the methods.
    let covariances = normal_covariances(&fixed, &noise);

    println!("# point_sigma {}", noise.point_sigma);
    println!("# signal_to_noise {}", noise.signal_to_noise);
    println!("px,py,pz,nx,ny,nz,w,c00,c01,c02,c11,c12,c22");

    let limit = params.max_distance * params.max_distance;
    let kernel = params.kernel();
    let mut found = Vec::with_capacity(1);
    let mut kept = 0usize;
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
        let residual = normal.dot(&(point - fixed.cloud.point(matched)));
        let weight = kernel.weight(residual);
        let c = covariances[matched];
        println!(
            "{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
            point.x,
            point.y,
            point.z,
            normal.x,
            normal.y,
            normal.z,
            weight,
            c[(0, 0)],
            c[(0, 1)],
            c[(0, 2)],
            c[(1, 1)],
            c[(1, 2)],
            c[(2, 2)]
        );
        kept += 1;
    }

    eprintln!("correspondences {kept}");
    match probabilistic_information(&moving, &fixed, &pose, &params, &noise) {
        Some(ours) => {
            // Sorted by eigenvalue so that the two sides can be lined up:
            // the orderings differ by a permutation of the state's blocks,
            // which leaves the spectrum alone.
            let mut order = [0usize, 1, 2, 3, 4, 5];
            order.sort_by(|a, b| ours.eigenvalues[*a].total_cmp(&ours.eigenvalues[*b]));
            eprint!("ours eigenvalues ");
            for k in order {
                eprint!("{:.6e} ", ours.eigenvalues[k]);
            }
            eprint!("\nours probabilities ");
            for k in order {
                eprint!("{:.6} ", ours.probability[k]);
            }
            eprintln!();
        }
        None => eprintln!("ours: no correspondences"),
    }
}

/// The neighbourhood covariance of each normal, as
/// `rigidity_harness::probabilistic` computes it.
fn normal_covariances(cloud: &Prepared, noise: &NoiseParams) -> Vec<Matrix3<f64>> {
    let count = noise.neighbours.max(3);
    let scale = noise.point_sigma * noise.point_sigma / count as f64;
    let mut out = Vec::with_capacity(cloud.cloud.len());
    let mut found = Vec::with_capacity(count);
    for index in 0..cloud.cloud.len() {
        let query = cloud.cloud.point(index);
        cloud.tree.knn_into(&query, count, &mut found);
        if found.len() < 3 {
            out.push(Matrix3::zeros());
            continue;
        }
        let n = found.len() as f64;
        let mut centroid = Vector3::zeros();
        for neighbour in &found {
            centroid += cloud.cloud.point(neighbour.index as usize);
        }
        centroid /= n;
        let mut covariance = Matrix3::zeros();
        for neighbour in &found {
            let delta = cloud.cloud.point(neighbour.index as usize) - centroid;
            covariance += delta * delta.transpose();
        }
        covariance /= n - 1.0;

        let eigen = SymmetricEigen::new(covariance);
        let mut order = [0usize, 1, 2];
        order.sort_by(|a, b| eigen.eigenvalues[*b].total_cmp(&eigen.eigenvalues[*a]));
        let (large, middle) = (eigen.eigenvalues[order[0]], eigen.eigenvalues[order[1]]);
        let axes = [
            eigen.eigenvectors.column(order[0]).into_owned(),
            eigen.eigenvectors.column(order[1]).into_owned(),
        ];
        // Crossed, and it matters: the normal turns most easily about the
        // patch's *long* axis, so the direction of the largest eigenvalue
        // carries the variance divided by the middle one, and the other way
        // round. Pairing them straight through is what this file did at
        // first, and it made the two implementations disagree by eleven per
        // cent in the trace of the noise — a difference that looked like a
        // difference between the two methods and was a difference between
        // this file and the library it is meant to mirror.
        let mut result = Matrix3::zeros();
        if middle > 0.0 {
            result += axes[0] * axes[0].transpose() * (scale / middle);
        }
        if large > 0.0 {
            result += axes[1] * axes[1].transpose() * (scale / large);
        }
        out.push(result);
    }
    out
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
