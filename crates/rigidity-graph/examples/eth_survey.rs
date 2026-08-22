//! A whole survey against theodolite ground truth.
//!
//! Everything stage three rests on has been checked against scenes this
//! project generated itself. That is the same position M7 left the core in,
//! and M9 is what the project did about it: run on data somebody else
//! measured, with a truth nobody here chose, and publish whichever number
//! comes out.
//!
//! This is that, one level up. The core's M9 compared *pairs* of real scans
//! against a theodolite. A survey is the pairs chained together, where each
//! registration's error is carried into the next, and the whole claim of
//! `rigidity-graph` is that weighting an edge by what its geometry actually
//! determined leaves the chain closer to the truth than weighting it by
//! `JᵀWJ`. Nothing has tested that on anything but scenes built to show it.
//!
//! ```text
//! cargo run --release -p rigidity-graph --example eth_survey -- \
//!     <directory of Hokuyo_*.csv and pose_scanner_leica.csv> [stations]
//! ```
//!
//! The data is the ETH ASL Challenging Datasets for Point Cloud
//! Registration, which this repository does not carry: `datasets/fetch.sh`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use rigidity_core::lie::{Se3, So3};
use rigidity_core::nalgebra::{Matrix3, Matrix4, Matrix6, Vector3};
use rigidity_core::observability::ObservabilityCriteria;
use rigidity_graph::{Edge, OptimiseParams, PoseGraph, weighted_information};
use rigidity_pipeline::{
    PrepareParams, RegisterParams, analyse_registration, prepare, register_pair_observed,
};

/// Sensor noise along the normal, metres.
///
/// The Hokuyo UTM-30LX, taken at the same three centimetres M9 used, so
/// that this run and that one are talking about the same instrument.
const NOISE: f64 = 0.03;

/// The correction M9 measured between predicted and actual spread on this
/// very dataset.
///
/// Applied to the noise and not to the tolerance, which is where the rest
/// of the project puts it: the prediction is optimistic, the requirement is
/// not negotiable. Without it every direction of every edge looks
/// determined and the weighting has nothing to say — which is itself worth
/// knowing, and is why the run prints both.
const CALIBRATION: f64 = 17.0;

/// How accurate the survey is asked to be, metres.
const TOLERANCE: f64 = 0.05;

fn main() {
    let mut args = std::env::args().skip(1);
    let directory = PathBuf::from(args.next().unwrap_or_else(|| {
        eprintln!("usage: eth_survey <directory> [stations]");
        std::process::exit(2);
    }));
    let stations: usize = args.next().and_then(|text| text.parse().ok()).unwrap_or(16);

    let truth = read_truth(&directory.join("pose_scanner_leica.csv"), stations);
    let prepared = read_scans(&directory, stations);

    // Consecutive pairs and then the skip-one pairs, which is what turns
    // the survey from a chain into something with redundancy. A chain has
    // none: it is a tree, its residual is zero at whatever answer it gives,
    // and no weighting of any kind can change that answer. The first run of
    // this example registered only consecutive pairs and the two weightings
    // agreed to the last digit, for that reason and no other.
    //
    // Each leg starts from the leg before it rather than from the identity.
    // A survey walks, so the motion between one pair is a decent guess at
    // the next, and it uses no truth. Starting from the identity put nine
    // of fifteen of these pairs in the wrong basin.
    println!("\nregistering {} consecutive pairs", stations - 1);
    let params = RegisterParams::default();
    let criteria = ObservabilityCriteria {
        noise_sigma: NOISE * CALIBRATION,
        tolerance: TOLERANCE,
    };
    let mut measurements = Vec::new();
    let mut naive = Vec::new();
    let mut weighted: Vec<rigidity_core::observability::Conditioning> = Vec::new();
    let mut determined = Vec::new();

    let mut carried = Se3::identity();
    for index in 0..stations - 1 {
        let started = Instant::now();
        let result = register_pair_observed(
            &prepared[index + 1],
            &prepared[index],
            carried,
            &params,
            |_| std::ops::ControlFlow::Continue(()),
        );
        carried = result.pose;
        let analysis = analyse_registration(
            &prepared[index + 1],
            &prepared[index],
            &result.pose,
            &params,
        )
        .expect("the registration should analyse");
        let high = analysis
            .conditioning
            .classify(&criteria)
            .iter()
            .filter(|state| **state == rigidity_core::observability::Observability::High)
            .count();

        // The truth for this leg, for the report only. Nothing below uses it.
        let expected = truth[index].inverse() * truth[index + 1];
        let error = (result.pose.translation() - expected.translation()).norm();
        println!(
            "  {index:2} → {:2}   rmse {:.4} m   {high} of 6 determined   \
             against theodolite {:.4} m   {:.2} s",
            index + 1,
            result.rmse,
            error,
            started.elapsed().as_secs_f64()
        );

        measurements.push(result.pose);
        naive.push(result.information / (NOISE * NOISE));
        weighted.push(analysis.conditioning);
        determined.push(high);
    }

    // The skip-one pairs. Their initial guess is the two legs composed,
    // which is again only what the survey already believes.
    println!(
        "\nregistering {} skip-one pairs",
        stations.saturating_sub(2)
    );
    let mut extra = Vec::new();
    for index in 0..stations.saturating_sub(2) {
        let started = Instant::now();
        let guess = measurements[index] * measurements[index + 1];
        let result = register_pair_observed(
            &prepared[index + 2],
            &prepared[index],
            guess,
            &params,
            |_| std::ops::ControlFlow::Continue(()),
        );
        let analysis = analyse_registration(
            &prepared[index + 2],
            &prepared[index],
            &result.pose,
            &params,
        )
        .expect("the registration should analyse");
        let high = analysis
            .conditioning
            .classify(&criteria)
            .iter()
            .filter(|state| **state == rigidity_core::observability::Observability::High)
            .count();
        let expected = truth[index].inverse() * truth[index + 2];
        println!(
            "  {index:2} → {:2}   rmse {:.4} m   {high} of 6 determined   \
             against theodolite {:.4} m   {:.2} s",
            index + 2,
            result.rmse,
            (result.pose.translation() - expected.translation()).norm(),
            started.elapsed().as_secs_f64()
        );
        extra.push((
            index,
            index + 2,
            result.pose,
            result.information / (NOISE * NOISE),
            analysis.conditioning,
            high,
        ));
    }

    // The survey as it would be built without any truth: chain the
    // measurements from the first station, which is held at its true pose
    // because something has to be the survey's origin.
    let mut chained = vec![truth[0]];
    for pose in &measurements {
        let last = *chained.last().expect("seeded");
        chained.push(last * *pose);
    }

    let solve = |weights: Vec<Matrix6<f64>>, label: &str| -> f64 {
        let mut graph = PoseGraph::new(chained.clone());
        let legs = measurements.len();
        for (index, weight) in weights.iter().take(legs).enumerate() {
            graph
                .push(Edge {
                    from: index,
                    to: index + 1,
                    measurement: measurements[index],
                    information: *weight,
                })
                .expect("the edge names real nodes");
        }
        for (offset, edge) in extra.iter().enumerate() {
            graph
                .push(Edge {
                    from: edge.0,
                    to: edge.1,
                    measurement: edge.2,
                    information: weights[legs + offset],
                })
                .expect("the edge names real nodes");
        }
        let report = graph
            .optimise(&OptimiseParams::default())
            .expect("the survey solves");
        let drift: Vec<f64> = graph
            .poses()
            .iter()
            .zip(&truth)
            .map(|(found, want)| (found.translation() - want.translation()).norm())
            .collect();
        let worst = drift.iter().fold(0.0f64, |best, value| best.max(*value));
        let mut sorted = drift.clone();
        sorted.sort_by(f64::total_cmp);
        println!(
            "  {label:>10}:  worst {worst:.4} m   median {:.4} m   \
             ({} iterations, cost {:.3e} to {:.3e})",
            sorted[sorted.len() / 2],
            report.iterations,
            report.cost[0],
            report.cost[1],
        );
        worst
    };

    let chain_only = chained
        .iter()
        .zip(&truth)
        .map(|(found, want)| (found.translation() - want.translation()).norm())
        .fold(0.0f64, f64::max);
    println!("\nagainst the theodolite, over {stations} stations");
    println!("  {:>10}:  worst {chain_only:.4} m", "unsolved");

    let every_naive: Vec<Matrix6<f64>> = naive
        .iter()
        .copied()
        .chain(extra.iter().map(|edge| edge.3))
        .collect();
    println!("\n  the same edges weighted by JᵀWJ, once:");
    let baseline = solve(every_naive, "JᵀWJ");

    // The tolerance sweep. `weighted_information` drops a direction when its
    // predicted spread exceeds what the survey asks for, so how demanding
    // the survey is decides whether there is anything to drop. Registration
    // is not repeated: the conditioning of an edge is a fact about the
    // geometry, and only the judgement passed on it moves.
    println!("\n  weighted by conditioning, as the survey's accuracy tightens:");
    for tolerance in [0.05, 0.02, 0.01, 0.005, 0.002, 0.001] {
        let criteria = ObservabilityCriteria {
            noise_sigma: NOISE * CALIBRATION,
            tolerance,
        };
        let lost: usize = weighted
            .iter()
            .chain(extra.iter().map(|edge| &edge.4))
            .map(|conditioning| {
                6 - conditioning
                    .classify(&criteria)
                    .iter()
                    .filter(|state| **state == rigidity_core::observability::Observability::High)
                    .count()
            })
            .sum();
        let weights: Vec<Matrix6<f64>> = weighted
            .iter()
            .chain(extra.iter().map(|edge| &edge.4))
            .map(|conditioning| weighted_information(conditioning, &criteria))
            .collect();
        let label = format!("{:.0} mm", tolerance * 1e3);
        println!(
            "  asking {label:>7}, {lost:3} of {} directions dropped",
            6 * weights.len()
        );
        let worst = solve(weights, "weighted");
        println!(
            "              {} than JᵀWJ by {:.1}%\n",
            if worst < baseline { "better" } else { "worse" },
            (worst - baseline).abs() / baseline * 100.0
        );
    }
}

/// The theodolite poses, as `Se3`.
fn read_truth(path: &Path, stations: usize) -> Vec<Se3> {
    let matrices = rigidity_io::read_poses(path).unwrap_or_else(|error| {
        eprintln!("{}: {error}", path.display());
        std::process::exit(1);
    });
    assert!(
        matrices.len() >= stations,
        "the pose file holds {} poses and {stations} were asked for",
        matrices.len()
    );
    matrices[..stations].iter().map(from_matrix).collect()
}

/// A 4×4 from the file, as a rigid motion.
///
/// Unchecked, deliberately: a theodolite's rotation block is orthogonal to
/// the precision it was written at and not to the last bit, and demanding
/// otherwise would reject the ground truth for being ground truth.
fn from_matrix(matrix: &Matrix4<f64>) -> Se3 {
    let rotation: Matrix3<f64> = matrix.fixed_view::<3, 3>(0, 0).into();
    let translation: Vector3<f64> = matrix.fixed_view::<3, 1>(0, 3).into();
    Se3::from_parts(So3::from_matrix_unchecked(rotation), translation)
}

fn read_scans(directory: &Path, stations: usize) -> Vec<rigidity_pipeline::Prepared> {
    let params = PrepareParams {
        voxel: 0.05,
        neighbours: 16,
    };
    println!("reading {stations} scans");
    (0..stations)
        .map(|index| {
            let path = directory.join(format!("Hokuyo_{index}.csv"));
            let started = Instant::now();
            let prepared = prepare(&path, &params).unwrap_or_else(|error| {
                eprintln!("{}: {error}", path.display());
                std::process::exit(1);
            });
            println!(
                "  {index:2}  {} points  {:.2} s",
                prepared.cloud.len(),
                started.elapsed().as_secs_f64()
            );
            prepared
        })
        .collect()
}
