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
//! registration's error is carried into the next, and the claim
//! `rigidity-graph` used to make is that weighting an edge by what its
//! geometry actually determined leaves the chain closer to the truth than
//! weighting it by `JᵀWJ`. Nothing had tested that on anything but scenes
//! built to show it.
//!
//! This run is what tested it, and the claim did not survive: see
//! `thresholded_information` at the bottom of this file, which is that
//! weighting, kept here because the numbers in `PLAN.md` are its numbers and
//! removed from the library because they are what they are.
//!
//! ```text
//! cargo run --release -p rigidity-graph --example eth_survey -- \
//!     <directory of Hokuyo_*.csv and pose_scanner_leica.csv> [stations]
//! ```
//!
//! The data is the ETH ASL Challenging Datasets for Point Cloud
//! Registration, which this repository does not carry: `datasets/fetch.sh`.
//!
//! # The field of view, and why it is a variable here
//!
//! S5 ran this on the scans as recorded and found the weighting never
//! beating `JᵀWJ`. M9 had already measured why that might be: a scanner on
//! a tripod sees 360°, and both these scenes come back fully observable —
//! the corridor at κ = 2.3…2.8, every pair converged. There was nothing
//! degenerate to drop, so a threshold could only throw good information
//! away.
//!
//! `RIGIDITY_SECTOR=<degrees>` keeps only the points within that half-width
//! of the sensor's own forward axis, which is how a lidar on a robot sees a
//! corridor. M9 used the same crop and the same variable, and at ±40° the
//! detector started ranking failures correctly. This is that experiment at
//! the scale of a survey.

use std::path::{Path, PathBuf};
use std::time::Instant;

use rigidity_core::lie::{Se3, So3};
use rigidity_core::nalgebra::{Matrix3, Matrix4, Matrix6, Vector3};
use rigidity_core::observability::ObservabilityCriteria;
use rigidity_graph::{Edge, OptimiseParams, PoseGraph, calibrated_information};
use rigidity_harness::probabilistic::{NoiseParams, probabilistic_information};
use rigidity_harness::thresholded::thresholded_information;
use rigidity_harness::{crop_sector, overlap};

use rigidity_core::nalgebra::Vector6;
use rigidity_core::observability::Conditioning;
use rigidity_pipeline::{
    PrepareParams, RegisterParams, analyse_registration, median_absolute_residual, prepare_cloud,
    register_pair_observed,
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
    let sector = sector_half_width();
    let prepared = read_scans(&directory, stations, sector);

    // S7's detector, turned on the survey's own edges. The claim it is here
    // to test is mine: that the sign of S6.1's effect flips between sectors
    // because the surveys are full of registrations that landed in the
    // wrong minimum, and that no weighting can be judged over those. The
    // rule is S7's exactly — the median absolute residual against the
    // sensor's own noise, uncalibrated, because that is the number the
    // detector was measured with.
    //
    // Both weightings lose the same edges, so the comparison stays a
    // comparison of weights. What it stops being is comparable with the
    // unfiltered run: a survey of fewer edges is a different survey, and
    // only the difference *within* a run means anything.
    let drop_basins = std::env::var("RIGIDITY_DROP_BASINS").is_ok();
    if drop_basins {
        println!(
            "\ndropping edges whose median residual is past {NOISE} m — S7's rule, on this survey"
        );
    }

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
    let params = RegisterParams::default();

    // A cropped view loses two things at once, and they must not be read as
    // one number: the geometry that determined some directions, and the
    // overlap ICP needs to find the right basin at all. Uncropped, the
    // survey walks — each leg starts from the leg before it — and that
    // chain is what breaks first: one leg in the wrong basin poisons the
    // guess for every leg after it.
    //
    // `RIGIDITY_SEED_FROM_FULL=1` hands each cropped registration the pose
    // the uncropped scans found for that same pair, which is the prior a
    // robot would have from its odometry, and leaves everything else to the
    // cropped data — correspondences, spectrum, weights, all of it. It is
    // an ablation and says so: if the basin is taken care of, does the
    // weighting help where directions really are lost?
    let seeds = (sector.is_some() && std::env::var("RIGIDITY_SEED_FROM_FULL").is_ok())
        .then(|| seeds_from_full(&directory, stations, &params));
    if seeds.is_some() {
        println!("\nseeding every pair with the pose the uncropped scans give it");
    }

    println!("\nregistering {} consecutive pairs", stations - 1);
    let criteria = ObservabilityCriteria {
        noise_sigma: NOISE * CALIBRATION,
        tolerance: TOLERANCE,
    };
    let mut measurements = Vec::new();
    let mut rmse = Vec::new();
    let mut leg_error = Vec::new();
    // Which legs are edges. A leg with no overlap still has to hold a place
    // in every array, so that a station's index never means two things, but
    // it carries no measurement into the graph.
    let mut is_edge: Vec<bool> = Vec::new();
    // Hatleskog and Alexis's weighting, computed edge by edge beside ours.
    // Their method is told the noise on a *point it is given*, and the
    // points it is given here are voxel centroids, not raw returns. A 5 cm
    // voxel over these scans holds three or four returns, so the centroid's
    // noise is the sensor's over the root of that, not the sensor's. Which
    // of the two to feed it is not a detail — it is the whole input — so it
    // is a variable and it gets swept.
    let point_sigma = std::env::var("RIGIDITY_POINT_SIGMA")
        .ok()
        .and_then(|text| text.parse::<f64>().ok())
        .unwrap_or(NOISE);
    println!("\nnoise on a point taken as {point_sigma} m");
    let noise = NoiseParams {
        point_sigma,
        neighbours: 16,
        signal_to_noise: 10.0,
    };
    let mut probable: Vec<Matrix6<f64>> = Vec::new();
    let mut probable_extra: Vec<Matrix6<f64>> = Vec::new();
    // The same weighting with the per-edge residual scaling taken out and
    // one sigma put in for every edge, so that the two halves of their
    // formula — the probabilities and the scaling — can be told apart.
    let mut probable_flat: Vec<Matrix6<f64>> = Vec::new();
    let mut probable_flat_extra: Vec<Matrix6<f64>> = Vec::new();
    // The median absolute residual of every edge: S11 measured that a bias
    // is proportional to it.
    let mut medians: Vec<f64> = Vec::new();
    let mut improbable_directions = 0usize;
    let mut last_conditioning: Option<rigidity_core::observability::Conditioning> = None;
    let mut naive = Vec::new();
    let mut weighted: Vec<rigidity_core::observability::Conditioning> = Vec::new();
    let mut determined = Vec::new();

    let mut carried = Se3::identity();
    for index in 0..stations - 1 {
        let started = Instant::now();
        let guess = match &seeds {
            Some((legs, _)) => legs[index],
            None => carried,
        };
        let result = register_pair_observed(
            &prepared[index + 1],
            &prepared[index],
            guess,
            &params,
            |_| std::ops::ControlFlow::Continue(()),
        );
        carried = result.pose;
        let Some(analysis) = analysed(
            &prepared[index + 1],
            &prepared[index],
            &result.pose,
            &params,
            index,
            index + 1,
        ) else {
            // The walk continues on the motion the survey already believes:
            // it is the initial guess for the next leg and the placeholder
            // that keeps this station on the map, and it is *not* pushed as
            // an edge, because nothing measured it.
            measurements.push(carried);
            medians.push(f64::NAN);
            probable.push(Matrix6::zeros());
            probable_flat.push(Matrix6::zeros());
            rmse.push(f64::NAN);
            leg_error.push(f64::NAN);
            is_edge.push(false);
            naive.push(Matrix6::zeros());
            weighted.push(
                last_conditioning
                    .clone()
                    .expect("the first leg of a survey cannot be the one with no overlap"),
            );
            determined.push(0);
            continue;
        };
        last_conditioning = Some(analysis.conditioning.clone());

        // Whether this leg is believable at all, which is a different
        // question from what its geometry determined.
        let median = median_absolute_residual(
            &prepared[index + 1],
            &prepared[index],
            &result.pose,
            &params,
        );
        medians.push(median.unwrap_or(f64::NAN));
        let suspect = drop_basins && median.is_some_and(|value| value > NOISE);
        if suspect {
            println!(
                "        median residual {:.4} m — wrong basin suspected, not an edge",
                median.unwrap_or(f64::NAN)
            );
        }
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

        let report = probabilistic_information(
            &prepared[index + 1],
            &prepared[index],
            &result.pose,
            &params,
            &noise,
        );
        improbable_directions += report
            .as_ref()
            .map_or(0, |r| r.probability.iter().filter(|p| **p < 0.5).count());
        probable.push(
            report
                .as_ref()
                .map_or_else(Matrix6::zeros, |r| r.information),
        );
        probable_flat.push(report.as_ref().map_or_else(Matrix6::zeros, |r| {
            r.information * r.residual_variance / (NOISE * NOISE)
        }));

        measurements.push(result.pose);
        rmse.push(result.rmse);
        leg_error.push(error);
        is_edge.push(!suspect);
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
    let mut skip_error = Vec::new();
    for index in 0..stations.saturating_sub(2) {
        let started = Instant::now();
        let guess = match &seeds {
            Some((_, skips)) => skips[index],
            None => measurements[index] * measurements[index + 1],
        };
        let result = register_pair_observed(
            &prepared[index + 2],
            &prepared[index],
            guess,
            &params,
            |_| std::ops::ControlFlow::Continue(()),
        );
        let Some(analysis) = analysed(
            &prepared[index + 2],
            &prepared[index],
            &result.pose,
            &params,
            index,
            index + 2,
        ) else {
            continue;
        };
        let high = analysis
            .conditioning
            .classify(&criteria)
            .iter()
            .filter(|state| **state == rigidity_core::observability::Observability::High)
            .count();
        let expected = truth[index].inverse() * truth[index + 2];
        let error = (result.pose.translation() - expected.translation()).norm();
        println!(
            "  {index:2} → {:2}   rmse {:.4} m   {high} of 6 determined   \
             against theodolite {:.4} m   {:.2} s",
            index + 2,
            result.rmse,
            error,
            started.elapsed().as_secs_f64()
        );
        skip_error.push(error);
        let median = median_absolute_residual(
            &prepared[index + 2],
            &prepared[index],
            &result.pose,
            &params,
        );
        if drop_basins && median.is_some_and(|value| value > NOISE) {
            println!(
                "        median residual {:.4} m — wrong basin suspected, not an edge",
                median.unwrap_or(f64::NAN)
            );
            continue;
        }
        rmse.push(result.rmse);
        medians.push(median.unwrap_or(f64::NAN));
        let report = probabilistic_information(
            &prepared[index + 2],
            &prepared[index],
            &result.pose,
            &params,
            &noise,
        );
        improbable_directions += report
            .as_ref()
            .map_or(0, |r| r.probability.iter().filter(|p| **p < 0.5).count());
        probable_extra.push(
            report
                .as_ref()
                .map_or_else(Matrix6::zeros, |r| r.information),
        );
        probable_flat_extra.push(report.as_ref().map_or_else(Matrix6::zeros, |r| {
            r.information * r.residual_variance / (NOISE * NOISE)
        }));
        extra.push((
            index,
            index + 2,
            result.pose,
            result.information / (NOISE * NOISE),
            analysis.conditioning,
            high,
        ));
    }

    // A registration that landed in another basin is a different failure
    // from an underestimated spread, and reading the two as one number is
    // how a survey gets called accurate to millimetres while a station sits
    // half a metre away. The threshold is arbitrary and therefore stated:
    // a tenth of a metre is two orders above the theodolite's own accuracy
    // and an order above the worst honest registration on these scenes.
    const BASIN: f64 = 0.10;
    let all_errors: Vec<f64> = leg_error.iter().chain(&skip_error).copied().collect();
    let lost = all_errors.iter().filter(|error| **error > BASIN).count();
    println!(
        "\n{lost} of {} registrations landed further than {BASIN:.2} m from \
         the theodolite; worst {:.4} m",
        all_errors.len(),
        all_errors
            .iter()
            .fold(0.0f64, |best, error| best.max(*error))
    );

    // What the edges make, said before anything is solved on them. A leg
    // with no overlap does not stop the run, and this is the line that says
    // what that cost: `adrift` counts the stations no chain of edges reaches
    // from the anchor, and a number there means the drift below is measured
    // over a survey in pieces.
    let missing = is_edge.iter().filter(|kept| !**kept).count();
    if missing > 0 {
        println!(
            "\n{missing} of {} legs are not edges — no overlap, or a basin the detector \
             does not believe",
            stations - 1
        );
    }
    println!(
        "{} of {} skip-one pairs are edges",
        extra.len(),
        stations.saturating_sub(2)
    );

    // The survey as it would be built without any truth: chain the
    // measurements from the first station, which is held at its true pose
    // because something has to be the survey's origin.
    let mut chained = vec![truth[0]];
    for pose in &measurements {
        let last = *chained.last().expect("seeded");
        chained.push(last * *pose);
    }

    // Every pair that sees the same place, not just the neighbours. The
    // chain and its skip-one pairs are what S2–S8 measured on, and that
    // graph is fragile: dropping eight edges of fifty-seven cut it into
    // pieces, because a skip-one bridges exactly one missing leg. A survey
    // whose stations mostly see each other has far more to say, and the
    // closure detector is what finds those pairs — proposed by the survey's
    // own estimate, confirmed by registering and by S7's rule.
    if let Some(reach) = std::env::var("RIGIDITY_DENSE")
        .ok()
        .and_then(|text| text.parse::<f64>().ok())
    {
        let before = extra.len();
        for from in 0..stations {
            for to in (from + 3)..stations {
                let guess = chained[from].inverse() * chained[to];
                if guess.translation().norm() > reach {
                    continue;
                }
                let result =
                    register_pair_observed(&prepared[to], &prepared[from], guess, &params, |_| {
                        std::ops::ControlFlow::Continue(())
                    });
                let shared = overlap(&prepared[to], &prepared[from], &result.pose, &params);
                let median = rigidity_pipeline::median_absolute_residual(
                    &prepared[to],
                    &prepared[from],
                    &result.pose,
                    &params,
                );
                if shared < 1.0 / 3.0 || !median.is_some_and(|value| value <= NOISE) {
                    continue;
                }
                let Ok(analysis) =
                    analyse_registration(&prepared[to], &prepared[from], &result.pose, &params)
                else {
                    continue;
                };
                let report = probabilistic_information(
                    &prepared[to],
                    &prepared[from],
                    &result.pose,
                    &params,
                    &noise,
                );
                rmse.push(result.rmse);
                medians.push(median.unwrap_or(f64::NAN));
                probable_extra.push(
                    report
                        .as_ref()
                        .map_or_else(Matrix6::zeros, |r| r.information),
                );
                probable_flat_extra.push(report.as_ref().map_or_else(Matrix6::zeros, |r| {
                    r.information * r.residual_variance / (NOISE * NOISE)
                }));
                let high = analysis
                    .conditioning
                    .classify(&criteria)
                    .iter()
                    .filter(|state| **state == rigidity_core::observability::Observability::High)
                    .count();
                extra.push((
                    from,
                    to,
                    result.pose,
                    result.information / (NOISE * NOISE),
                    analysis.conditioning,
                    high,
                ));
            }
        }
        println!(
            "\ndense: {} closures added out to {reach} m, on top of {before} skip-one pairs",
            extra.len() - before
        );
    }

    let solve = |weights: Vec<Matrix6<f64>>, label: &str| -> f64 {
        // Every variant below is brought to the same magnitude before it is
        // solved. Scaling all of a survey's edges by one number cannot move
        // the least-squares answer — but it can move what the Levenberg
        // damping does on the way there, and these variants differ in
        // magnitude by ten orders. Without this the comparison would be
        // partly a comparison of damping schedules.
        let scale: f64 =
            weights.iter().map(|matrix| matrix.trace()).sum::<f64>() / (6.0 * weights.len() as f64);
        let weights: Vec<Matrix6<f64>> = if scale.is_finite() && scale > 0.0 {
            weights.iter().map(|matrix| matrix / scale).collect()
        } else {
            weights
        };

        let mut graph = PoseGraph::new(chained.clone());
        // A leg that produced no correspondences holds its place in the
        // arrays and is skipped here, so the survey is built out of what
        // was measured rather than out of what was numbered.
        let legs = is_edge.iter().filter(|kept| **kept).count();
        for (index, weight) in weights
            .iter()
            .take(legs)
            .zip((0..measurements.len()).filter(|leg| is_edge[*leg]))
            .map(|(weight, leg)| (leg, weight))
        {
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

        // What the survey says about itself, from S4's own machinery. A
        // drift the survey announces as an undetermined station is a
        // different thing from the same drift reported confidently, and the
        // difference is the only one that would matter to somebody standing
        // in the building.
        let diagnosis = graph
            .diagnose(OptimiseParams::default().anchor)
            .expect("the survey diagnoses");
        let adrift = diagnosis
            .nodes
            .iter()
            .filter(|node| !node.determined)
            .count();
        let told = diagnosis
            .nodes
            .iter()
            .filter(|node| node.determined)
            .map(|node| node.position().0)
            .fold(0.0f64, f64::max);
        println!(
            "              it says: {adrift} of {} stations undetermined, \
             worst spread it admits {told:.4} m",
            diagnosis.nodes.len()
        );
        worst
    };

    let chain_only = chained
        .iter()
        .zip(&truth)
        .map(|(found, want)| (found.translation() - want.translation()).norm())
        .fold(0.0f64, f64::max);
    let mut skeleton = PoseGraph::new(chained.clone());
    for leg in (0..measurements.len()).filter(|leg| is_edge[*leg]) {
        skeleton
            .push(Edge {
                from: leg,
                to: leg + 1,
                measurement: measurements[leg],
                information: Matrix6::identity(),
            })
            .expect("the edge names real nodes");
    }
    for edge in &extra {
        skeleton
            .push(Edge {
                from: edge.0,
                to: edge.1,
                measurement: edge.2,
                information: Matrix6::identity(),
            })
            .expect("the edge names real nodes");
    }
    let shape = skeleton.shape(OptimiseParams::default().anchor);
    println!(
        "\nthe survey's shape: {} stations joined, {} closures, {} adrift",
        shape.joined, shape.closures, shape.adrift
    );

    println!("\nagainst the theodolite, over {stations} stations");
    println!("  {:>10}:  worst {chain_only:.4} m", "unsolved");

    let every_naive: Vec<Matrix6<f64>> = naive
        .iter()
        .enumerate()
        .filter(|(leg, _)| is_edge[*leg])
        .map(|(_, matrix)| *matrix)
        .chain(extra.iter().map(|edge| edge.3))
        .collect();
    println!("\n  the same edges weighted by JᵀWJ, once:");
    let baseline = solve(every_naive.clone(), "JᵀWJ");

    // The same matrices with the calibration in them, which changes nothing
    // about where the survey lands — a weighting scaled by one number is
    // the same weighting — and everything about what it claims to know. The
    // row exists so that the uncertainty the conditioning weighting reports
    // is not credited with what is really M9's factor of seventeen: that
    // weighting uses a calibrated sigma and this one, as every package
    // takes it, does not.
    println!("\n  the same again, with M9's calibration in the sigma:");
    let _ = solve(
        every_naive
            .iter()
            .map(|matrix| matrix / (CALIBRATION * CALIBRATION))
            .collect(),
        "JᵀWJ ×17",
    );

    // What the crate ships, once. It carries the calibration and leaves out
    // a direction the geometry is blind to, and on these scenes there is no
    // such direction, so this row is `JᵀWJ ×17` to the last digit. It is
    // printed anyway: a row that agrees is the evidence that it agrees.
    println!("\n  the crate's own calibrated_information:");
    let _ = solve(
        weighted
            .iter()
            .enumerate()
            .filter(|(leg, _)| is_edge[*leg])
            .map(|(_, conditioning)| conditioning)
            .chain(extra.iter().map(|edge| &edge.4))
            .map(|conditioning| calibrated_information(conditioning, NOISE * CALIBRATION))
            .collect(),
        "weighted",
    );

    // Somebody else's method, on our ground truth. Their probability
    // attenuates each direction of the Hessian by how far its own signal
    // stands above the noise the points and the normals put into it — no
    // threshold, and nothing tuned on this data: the sigma is the Hokuyo's
    // and the neighbour count is the one the normals were built with.
    println!(
        "\n  Hatleskog & Alexis: {improbable_directions} directions of {} came back below \
         p = 0.5",
        6 * (is_edge.iter().filter(|kept| **kept).count() + extra.len())
    );
    let worst = solve(
        probable
            .iter()
            .enumerate()
            .filter(|(leg, _)| is_edge[*leg])
            .map(|(_, matrix)| *matrix)
            .chain(probable_extra.iter().copied())
            .collect(),
        "probabilistic",
    );
    println!(
        "              {} than JᵀWJ by {:.1}%",
        if worst < baseline { "BETTER" } else { "worse" },
        (worst - baseline).abs() / baseline * 100.0
    );

    let worst = solve(
        probable_flat
            .iter()
            .enumerate()
            .filter(|(leg, _)| is_edge[*leg])
            .map(|(_, matrix)| *matrix)
            .chain(probable_flat_extra.iter().copied())
            .collect(),
        "p, one sigma",
    );
    println!(
        "              {} than JᵀWJ by {:.1}%",
        if worst < baseline { "BETTER" } else { "worse" },
        (worst - baseline).abs() / baseline * 100.0
    );

    // The family. `p = 2` is printed too, and it must reproduce the `JᵀWJ`
    // line above to the last digit — a row that does not is a bug in the
    // family, not a finding about it.
    println!("\n  Λ = Σ vvᵀ/spread^p, the family through JᵀWJ at p = 2:");
    for power in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0] {
        let weights: Vec<Matrix6<f64>> = weighted
            .iter()
            .enumerate()
            .filter(|(leg, _)| is_edge[*leg])
            .map(|(_, conditioning)| conditioning)
            .chain(extra.iter().map(|edge| &edge.4))
            .map(|conditioning| power_information(conditioning, NOISE * CALIBRATION, power))
            .collect();
        let label = format!("p = {power:.1}");
        let worst = solve(weights, &label);
        println!(
            "              {} than JᵀWJ by {:.1}%",
            if worst < baseline { "BETTER" } else { "worse" },
            (worst - baseline).abs() / baseline * 100.0
        );
    }

    // The two nulls, without which a floor that helps proves nothing. As
    // the floor grows every direction ends up with the same weight, and the
    // question becomes what is left doing the work: the spectrum, or merely
    // the fact that `JᵀWJ`'s anisotropy is gone.
    //
    // The crude null is one identity matrix for every edge — no geometry at
    // all, every edge and every direction believed alike. The sharper null
    // keeps everything the conditioning knows *except* the spectrum: the
    // centroid each direction is referred to and the size of the scene,
    // which is what `to_normalised` carries. If the floor cannot beat that
    // one, then what helps is the normalisation and not the spectrum, and
    // saying so is the difference between a finding and a press release.
    println!("\n  the nulls:");
    let _ = solve(
        (0..is_edge.iter().filter(|kept| **kept).count() + extra.len())
            .map(|_| Matrix6::identity())
            .collect(),
        "identity",
    );
    let _ = solve(
        weighted
            .iter()
            .enumerate()
            .filter(|(leg, _)| is_edge[*leg])
            .map(|(_, conditioning)| conditioning)
            .chain(extra.iter().map(|edge| &edge.4))
            .map(|conditioning| floored_information(conditioning, NOISE * CALIBRATION, 1e6))
            .collect(),
        "no spectrum",
    );

    // The floor. Same reading as the family above: `floor = 0` must
    // reproduce `JᵀWJ` exactly, and does.
    println!("\n  Λ = Σ vvᵀ/(spread² + floor²), a systematic floor under every direction:");
    for floor in [0.0, 0.003, 0.01, 0.03, 0.06, 0.1, 0.2, 0.3, 1.0] {
        let weights: Vec<Matrix6<f64>> = weighted
            .iter()
            .enumerate()
            .filter(|(leg, _)| is_edge[*leg])
            .map(|(_, conditioning)| conditioning)
            .chain(extra.iter().map(|edge| &edge.4))
            .map(|conditioning| floored_information(conditioning, NOISE * CALIBRATION, floor))
            .collect();
        let label = format!("{:.0} mm", floor * 1e3);
        let worst = solve(weights, &label);
        println!(
            "              {} than JᵀWJ by {:.1}%",
            if worst < baseline { "BETTER" } else { "worse" },
            (worst - baseline).abs() / baseline * 100.0
        );
    }

    // S11's model. The floor is neither swept nor read off this run. S11
    // measured that an edge's error is bias rather than scatter, that the
    // closed form describes the scatter correctly and is blind to the bias,
    // and that the bias is proportional to the median absolute residual —
    // by 2.24 on the plain and 2.24 in the apartment, two scenes with
    // nothing in common. That number comes from outside and is not touched
    // here.
    const BIAS_PER_RESIDUAL: f64 = 2.24;
    let edge_medians: Vec<f64> = medians
        .iter()
        .take(stations - 1)
        .enumerate()
        .filter(|(leg, _)| is_edge[*leg])
        .map(|(_, value)| *value)
        .chain(medians.iter().skip(stations - 1).copied())
        .collect();
    println!("\n  Λ = Σ vvᵀ/(spread² + (2.24·median)²), the bias S11 measured:");
    {
        let weights: Vec<Matrix6<f64>> = weighted
            .iter()
            .enumerate()
            .filter(|(leg, _)| is_edge[*leg])
            .map(|(_, conditioning)| conditioning)
            .chain(extra.iter().map(|edge| &edge.4))
            .zip(&edge_medians)
            .map(|(conditioning, median)| {
                let floor = if median.is_finite() {
                    BIAS_PER_RESIDUAL * median
                } else {
                    0.0
                };
                floored_information(conditioning, NOISE * CALIBRATION, floor)
            })
            .collect();
        let worst = solve(weights, "S11 model");
        println!(
            "              {} than JᵀWJ by {:.1}%",
            if worst < baseline { "BETTER" } else { "worse" },
            (worst - baseline).abs() / baseline * 100.0
        );
    }

    // The floor as a model rather than a number. The constant that worked
    // above is about a tenth of a metre on a survey whose legs are 0.6 m,
    // which is not a property of the world — it is a property of *these*
    // registrations, and the quantity that already measures how well two
    // surfaces actually agreed is the residual. Systematic disagreement is
    // exactly what a residual is made of and exactly what `σ/√N` throws
    // away, so `floor = c · rmse` has a mechanism behind it and one
    // dimensionless number in front.
    // The residual of every edge, in the order the edges are pushed. Built
    // once and explicitly: the same list assembled lazily inside a `zip`
    // read correctly only as long as no leg was dropped, which stopped
    // being true the moment a leg could be.
    let edge_rmse: Vec<f64> = rmse
        .iter()
        .take(stations - 1)
        .enumerate()
        .filter(|(leg, _)| is_edge[*leg])
        .map(|(_, residual)| *residual)
        .chain(rmse.iter().skip(stations - 1).copied())
        .collect();
    println!("\n  Λ = Σ vvᵀ/(spread² + (c·rmse)²), the floor tied to the residual:");
    for c in [0.25, 0.5, 1.0, 2.0, 4.0] {
        let weights: Vec<Matrix6<f64>> = weighted
            .iter()
            .enumerate()
            .filter(|(leg, _)| is_edge[*leg])
            .map(|(_, conditioning)| conditioning)
            .chain(extra.iter().map(|edge| &edge.4))
            .zip(&edge_rmse)
            .map(|(conditioning, residual)| {
                floored_information(conditioning, NOISE * CALIBRATION, c * residual)
            })
            .collect();
        let label = format!("c = {c:.2}");
        let worst = solve(weights, &label);
        println!(
            "              {} than JᵀWJ by {:.1}%",
            if worst < baseline { "BETTER" } else { "worse" },
            (worst - baseline).abs() / baseline * 100.0
        );
    }

    // The tolerance sweep. `thresholded_information` drops a direction when its
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
            .enumerate()
            .filter(|(leg, _)| is_edge[*leg])
            .map(|(_, conditioning)| conditioning)
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
            .enumerate()
            .filter(|(leg, _)| is_edge[*leg])
            .map(|(_, conditioning)| conditioning)
            .chain(extra.iter().map(|edge| &edge.4))
            .map(|conditioning| thresholded_information(conditioning, &criteria))
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

/// `Λ = Σ vᵢvᵢᵀ / (spreadᵢ² + floor²)`: the same information with a floor
/// under every direction.
///
/// In covariance terms this is `C + floor²·I` — no direction of an edge is
/// believed to better than `floor`, whatever its geometry says. That is not
/// a tuning knob looking for a home. M9 measured the prediction to be
/// optimistic by about seventeenfold and named the mechanism: the formula
/// counts `N` independent measurements where real laser error is correlated
/// over patches of surface, so of twenty-five thousand points some seventy
/// do the work. A correlated error that does not average away is exactly an
/// additive floor, and it bites hardest where the random part is smallest —
/// which is why it changes the *relative* weight of directions and a single
/// calibration constant cannot.
///
/// The power family below says the same thing less honestly: the data
/// preferred `p < 2`, that is, flatter than `JᵀWJ`, and a floor is what
/// flattening means when written as a model rather than an exponent. This
/// one also stays an inverse covariance, so what `diagnose` says about the
/// result still means something.
fn floored_information(conditioning: &Conditioning, noise_sigma: f64, floor: f64) -> Matrix6<f64> {
    let mut to_normalised = Matrix6::zeros();
    for axis in 0..6 {
        let mut basis = Vector6::zeros();
        basis[axis] = 1.0;
        to_normalised.set_column(axis, &conditioning.to_normalised(basis));
    }

    let spreads = conditioning.uncertainty(noise_sigma);
    let mut normalised = Matrix6::zeros();
    for (index, spread) in spreads.iter().enumerate() {
        // An infinite spread is still absent: a floor says "no better than
        // this", not "no worse than this".
        if !spread.is_finite() {
            continue;
        }
        let direction = conditioning.direction(index);
        normalised += direction * direction.transpose() / (spread * spread + floor * floor);
    }

    to_normalised.transpose() * normalised * to_normalised
}

/// `Λ = Σ vᵢvᵢᵀ / spreadᵢᵖ`, the one-parameter family that contains both
/// ends of the argument.
///
/// At `p = 2` this is `JᵀWJ/σ²` exactly — the weighting every package uses.
/// As `p` grows, a direction with twice the spread of another is punished
/// not four times but `2ᵖ`, and in the limit only the best direction of an
/// edge carries anything, which is what the threshold did discontinuously.
///
/// The point of the family is that it needs no gap. The threshold failed
/// because it had to decide where to cut a spectrum whose six values sit
/// inside one order of magnitude; this uses the whole ordering instead, and
/// an ordering that is only weakly right — the project measured rank
/// correlation +0.33 against the truth — can still help when nothing has to
/// be thrown away on the strength of it.
///
/// For `p ≠ 2` the result is no longer an inverse covariance and the
/// spreads `diagnose` reports from it mean nothing. It is a weighting, and
/// only the drift it produces is read below.
fn power_information(conditioning: &Conditioning, noise_sigma: f64, power: f64) -> Matrix6<f64> {
    let mut to_normalised = Matrix6::zeros();
    for axis in 0..6 {
        let mut basis = Vector6::zeros();
        basis[axis] = 1.0;
        to_normalised.set_column(axis, &conditioning.to_normalised(basis));
    }

    let spreads = conditioning.uncertainty(noise_sigma);
    let mut normalised = Matrix6::zeros();
    for (index, spread) in spreads.iter().enumerate() {
        if !(spread.is_finite() && *spread > 0.0) {
            continue;
        }
        let direction = conditioning.direction(index);
        normalised += direction * direction.transpose() / spread.powf(power);
    }

    to_normalised.transpose() * normalised * to_normalised
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

fn read_scans(
    directory: &Path,
    stations: usize,
    sector: Option<f64>,
) -> Vec<rigidity_pipeline::Prepared> {
    let params = PrepareParams {
        voxel: 0.05,
        neighbours: 16,
    };
    match sector {
        Some(half_width) => println!(
            "reading {stations} scans, cropped to ±{:.0}°",
            half_width.to_degrees()
        ),
        None => println!("reading {stations} scans"),
    }
    (0..stations)
        .map(|index| {
            let path = directory.join(format!("Hokuyo_{index}.csv"));
            let started = Instant::now();
            // `prepare` is this pair of calls with nothing between them, so
            // an uncropped run here is the same run S5 made, to the bit.
            let raw = rigidity_io::read(&path).unwrap_or_else(|error| {
                eprintln!("{}: {error}", path.display());
                std::process::exit(1);
            });
            let kept = match sector {
                Some(half_width) => crop_sector(&raw, 0.0, half_width),
                None => raw,
            };
            let prepared = prepare_cloud(&kept, &params).unwrap_or_else(|error| {
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

/// The conditioning of one pair, or `None` with the reason said out loud.
///
/// A narrow sector can leave two stations with no points close enough to
/// match at all, and `analyse_registration` says so by returning
/// `NoCorrespondences`. That used to arrive as a panic three hundred lines
/// from the cause. It is not fatal either: a survey missing one leg is a
/// survey in two pieces, which is a thing this crate can represent and
/// report — `Shape::adrift` counts exactly the stations that lost their way
/// to the anchor. So the pair is named, the edge is not built, and the run
/// goes on to say what the rest of the survey does.
fn analysed(
    moving: &rigidity_pipeline::Prepared,
    fixed: &rigidity_pipeline::Prepared,
    pose: &Se3,
    params: &RegisterParams,
    from: usize,
    to: usize,
) -> Option<rigidity_core::observability::Analysis> {
    match analyse_registration(moving, fixed, pose, params) {
        Ok(analysis) => Some(analysis),
        Err(error) => {
            println!("  {from:2} → {to:2}   {error} — no edge from this pair");
            None
        }
    }
}

/// The poses the uncropped scans give for the same pairs, in the same
/// order the run below needs them: the consecutive legs first, then the
/// skip-one pairs.
///
/// This reads all sixteen scans a second time. That is the honest cost of
/// an ablation whose whole point is that the two runs differ in nothing but
/// the field of view.
fn seeds_from_full(
    directory: &Path,
    stations: usize,
    params: &RegisterParams,
) -> (Vec<Se3>, Vec<Se3>) {
    let full = read_scans(directory, stations, None);
    let mut carried = Se3::identity();
    let mut legs = Vec::new();
    for index in 0..stations - 1 {
        let result =
            register_pair_observed(&full[index + 1], &full[index], carried, params, |_| {
                std::ops::ControlFlow::Continue(())
            });
        carried = result.pose;
        legs.push(result.pose);
    }
    let mut skips = Vec::new();
    for index in 0..stations.saturating_sub(2) {
        let guess = legs[index] * legs[index + 1];
        let result = register_pair_observed(&full[index + 2], &full[index], guess, params, |_| {
            std::ops::ControlFlow::Continue(())
        });
        skips.push(result.pose);
    }
    (legs, skips)
}

/// The half-width of the sector to keep, radians, from `RIGIDITY_SECTOR`.
///
/// The same variable and the same units as `real_data.rs`, so that a run of
/// one can be quoted next to a run of the other.
fn sector_half_width() -> Option<f64> {
    std::env::var("RIGIDITY_SECTOR")
        .ok()
        .and_then(|text| text.parse::<f64>().ok())
        .map(f64::to_radians)
}
