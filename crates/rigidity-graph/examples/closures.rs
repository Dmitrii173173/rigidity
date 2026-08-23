//! Finding the pairs of stations that see the same place, and measuring
//! whether the finding is any good.
//!
//! # Why this is a milestone and not a feature
//!
//! A survey walked as a chain drifts, and the only thing that takes the
//! drift out is a measurement between two stations that are *not*
//! neighbours in the walk. S3 built the machinery to use such an edge and
//! left the finding of them to the operator, and the plan has carried the
//! debt ever since with the terms of its settlement written down: a closure
//! detector is judged by precision and recall on the same loop, not shown
//! in a screenshot.
//!
//! # What is measured
//!
//! Every pair of stations more than two apart in the walk is labelled from
//! the theodolite. The first version of this labelled a pair a closure when
//! the two scans, placed at their true poses, shared a third of a cloud —
//! and that turned out to measure the scene rather than anything else: in a
//! corridor a 360° scanner sees everything from everywhere, so all 153
//! pairs qualified, and on an open plain 248 of 253 did. A label that says
//! yes to everything grades nothing.
//!
//! So a pair is a closure here when it would give the survey a **correct
//! edge**: started from the true relative pose, so that finding the right
//! basin is not part of the question, the registration stays within
//! `USABLE` of the truth and still shares `OVERLAP` of a cloud. That is
//! what a closure is for — an edge that pulls the graph towards where the
//! stations really are — and a pair that cannot supply one is not a missed
//! closure, whatever it can see.
//!
//! The detector then gets the same pairs without the truth. It proposes by
//! the survey's own estimate — the chain, drift and all — and confirms by
//! registering and asking two things of the result: does a third of the
//! cloud match, and are the residuals the sensor's own (S7's rule). The two
//! questions are different and both are needed: overlap without agreement
//! is two scans of different places at the same distance, and agreement
//! without overlap is two walls that happen to be parallel.
//!
//! ```text
//! cargo run --release -p rigidity-graph --example closures -- <directory> [stations]
//! ```

use std::path::{Path, PathBuf};

use rigidity_core::lie::{Se3, So3};
use rigidity_core::nalgebra::{Matrix3, Matrix4, Matrix6, Vector3};
use rigidity_graph::{Edge, OptimiseParams, PoseGraph};
use rigidity_harness::overlap;
use rigidity_pipeline::{
    PrepareParams, Prepared, RegisterParams, median_absolute_residual, prepare_cloud,
    register_pair_observed,
};

/// Sensor noise along the normal, metres — S7's rule needs it, and it is
/// the Hokuyo's, as M9 took it.
const NOISE: f64 = 0.03;

/// How much of a cloud has to find the other before the two count as
/// seeing the same place.
const OVERLAP: f64 = 1.0 / 3.0;

/// How close to the truth a registration has to land before the pair counts
/// as able to supply an edge, metres. The same tenth of a metre S7 uses to
/// call a basin wrong, for the same reason: two orders above the
/// theodolite and an order above an honest registration.
const USABLE: f64 = 0.10;

/// How far apart two stations may be estimated to stand and still be worth
/// registering, metres. A survey whose legs are 0.6 m and whose scanner
/// reaches thirty could close over a long distance; what bounds this is
/// the cost of trying, not the physics.
const REACH: f64 = 6.0;

/// The same, overridable, because the right value is a property of the
/// instrument and the site rather than of this file: a Hokuyo reaching
/// thirty metres in a corridor overlaps everything with everything, and on
/// a traverse across a field it overlaps almost nothing.
fn reach() -> f64 {
    std::env::var("RIGIDITY_REACH")
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(REACH)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let directory = PathBuf::from(args.next().unwrap_or_else(|| {
        eprintln!("usage: closures <directory> [stations]");
        std::process::exit(2);
    }));
    let stations: usize = args.next().and_then(|text| text.parse().ok()).unwrap_or(20);

    let truth = read_truth(&directory.join("pose_scanner_leica.csv"), stations);
    let prepared = read_scans(&directory, stations);
    let params = RegisterParams::default();
    let quiet = |_: &_| std::ops::ControlFlow::Continue(());
    let reach = reach();
    println!("pairs are tried out to {reach} m of the survey's own estimate");

    // The survey as it is built: a chain, each leg from the one before.
    println!("\nwalking {} legs", stations - 1);
    let mut chained = vec![Se3::identity()];
    let mut carried = Se3::identity();
    for index in 0..stations - 1 {
        let result = register_pair_observed(
            &prepared[index + 1],
            &prepared[index],
            carried,
            &params,
            quiet,
        );
        carried = result.pose;
        let last = *chained.last().expect("seeded");
        chained.push(last * result.pose);
    }
    let drift: Vec<f64> = chained
        .iter()
        .zip(&truth)
        .map(|(found, want)| ((truth[0] * *found).translation() - want.translation()).norm())
        .collect();
    println!(
        "the chain ends {:.3} m from where the theodolite says, worst station {:.3} m",
        drift.last().copied().unwrap_or_default(),
        drift.iter().fold(0.0f64, |best, value| best.max(*value))
    );

    // The legs, kept so that later rounds can rebuild the graph.
    let mut legs: Vec<(Se3, Matrix6<f64>)> = Vec::new();
    {
        let mut carried = Se3::identity();
        for index in 0..stations - 1 {
            let result = register_pair_observed(
                &prepared[index + 1],
                &prepared[index],
                carried,
                &params,
                quiet,
            );
            carried = result.pose;
            legs.push((result.pose, result.information / (NOISE * NOISE)));
        }
    }

    // Round by round: propose from what the survey believes, confirm by
    // registering, then put the confirmed closures into the graph and solve.
    // A closure taken in one round improves the estimate the next round
    // proposes from, which is the only lever there is on the recall below.
    for round in 1..=ROUNDS {
        println!("\n── round {round} ──");
        let outcome = search(&prepared, &truth, &chained, &params, reach, stations);
        report(&outcome, stations);
        if round == ROUNDS || outcome.found.is_empty() {
            break;
        }

        let mut graph = PoseGraph::new(chained.clone());
        for (index, (measurement, information)) in legs.iter().enumerate() {
            graph
                .push(Edge {
                    from: index,
                    to: index + 1,
                    measurement: *measurement,
                    information: *information,
                })
                .expect("the edge names real nodes");
        }
        for closure in &outcome.found {
            graph
                .push(Edge {
                    from: closure.0,
                    to: closure.1,
                    measurement: closure.2,
                    information: closure.3,
                })
                .expect("the edge names real nodes");
        }
        graph.optimise(&OptimiseParams::default()).expect("solves");
        chained = graph.poses().to_vec();
        let moved: f64 = chained
            .iter()
            .zip(&truth)
            .map(|(found, want)| ((truth[0] * *found).translation() - want.translation()).norm())
            .fold(0.0, f64::max);
        println!(
            "  solved with {} closures; worst station now {moved:.3} m",
            outcome.found.len()
        );
    }
}

/// How many times to propose, confirm and solve.
const ROUNDS: usize = 3;

/// What one round found.
struct Outcome {
    truths: usize,
    proposed: usize,
    hit: usize,
    wrong: usize,
    missed: Vec<f64>,
    found: Vec<(usize, usize, Se3, Matrix6<f64>)>,
}

fn report(outcome: &Outcome, stations: usize) {
    let precision = if outcome.hit + outcome.wrong == 0 {
        f64::NAN
    } else {
        outcome.hit as f64 / (outcome.hit + outcome.wrong) as f64
    };
    let recall = if outcome.truths == 0 {
        f64::NAN
    } else {
        outcome.hit as f64 / outcome.truths as f64
    };
    println!(
        "  {} of {} pairs more than two apart could supply a correct edge; {} were near enough \
         to try",
        outcome.truths,
        (0..stations)
            .map(|from| stations.saturating_sub(from + 3))
            .sum::<usize>(),
        outcome.proposed
    );
    println!(
        "  found {}, missed {}, wrong {} — precision {precision:.3}, recall {recall:.3}",
        outcome.hit,
        outcome.missed.len(),
        outcome.wrong
    );
    if !outcome.missed.is_empty() {
        println!(
            "  the ones it did not find shared {:.2}…{:.2} of a cloud when placed correctly",
            outcome.missed.iter().copied().fold(f64::INFINITY, f64::min),
            outcome.missed.iter().copied().fold(0.0f64, f64::max)
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn search(
    prepared: &[Prepared],
    truth: &[Se3],
    chained: &[Se3],
    params: &RegisterParams,
    reach: f64,
    stations: usize,
) -> Outcome {
    let quiet = |_: &_| std::ops::ControlFlow::Continue(());
    let mut truths = 0usize;
    let mut proposed = 0usize;
    let mut hit = 0usize;
    let mut missed: Vec<f64> = Vec::new();
    let mut wrong = 0usize;
    let mut found: Vec<(usize, usize, Se3, Matrix6<f64>)> = Vec::new();
    for from in 0..stations {
        for to in (from + 3)..stations {
            // The label. The truth is used twice and only here: to start the
            // registration, so that the label is about the geometry rather
            // than about ICP's basin, and to judge where it landed.
            let exact = truth[from].inverse() * truth[to];
            let ideal =
                register_pair_observed(&prepared[to], &prepared[from], exact, params, quiet);
            let shared = overlap(&prepared[to], &prepared[from], &ideal.pose, params);
            let closes = shared >= OVERLAP
                && (ideal.pose.translation() - exact.translation()).norm() <= USABLE;
            if closes {
                truths += 1;
            }

            // The detector. It may only look at what the survey believes.
            let guess = chained[from].inverse() * chained[to];
            let reachable = guess.translation().norm() <= reach;
            let mut confirmed = false;
            if reachable {
                proposed += 1;
                let result =
                    register_pair_observed(&prepared[to], &prepared[from], guess, params, quiet);
                let registered = overlap(&prepared[to], &prepared[from], &result.pose, params);
                let median =
                    median_absolute_residual(&prepared[to], &prepared[from], &result.pose, params);
                confirmed = registered >= OVERLAP && median.is_some_and(|value| value <= NOISE);
                if confirmed {
                    found.push((from, to, result.pose, result.information / (NOISE * NOISE)));
                }
            }

            match (closes, confirmed) {
                (true, true) => hit += 1,
                (true, false) => missed.push(shared),
                (false, true) => wrong += 1,
                (false, false) => {}
            }
        }
    }

    Outcome {
        truths,
        proposed,
        hit,
        wrong,
        missed,
        found,
    }
}

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

fn from_matrix(matrix: &Matrix4<f64>) -> Se3 {
    let rotation: Matrix3<f64> = matrix.fixed_view::<3, 3>(0, 0).into();
    let translation: Vector3<f64> = matrix.fixed_view::<3, 1>(0, 3).into();
    Se3::from_parts(So3::from_matrix_unchecked(rotation), translation)
}

fn read_scans(directory: &Path, stations: usize) -> Vec<Prepared> {
    let params = PrepareParams {
        voxel: 0.05,
        neighbours: 16,
    };
    println!("reading {stations} scans");
    (0..stations)
        .map(|index| {
            let path = directory.join(format!("Hokuyo_{index}.csv"));
            let raw = rigidity_io::read(&path).unwrap_or_else(|error| {
                eprintln!("{}: {error}", path.display());
                std::process::exit(1);
            });
            prepare_cloud(&raw, &params).expect("the scan prepares")
        })
        .collect()
}
