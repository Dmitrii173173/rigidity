//! Which registrations landed in the wrong basin, asked without the truth.
//!
//! # Why this exists
//!
//! Conditioning answers "is the pose determined by the geometry here". It
//! does not answer "is this the right place", and the project has measured
//! that it cannot: on the ETH plain the pairs that converged to a wrong
//! minimum had condition numbers of 2–4, exactly like the ones that did
//! not, and every one of them reported six directions of six determined.
//!
//! That failure is not a footnote. It sets the worst station of every
//! survey S6 measured, it is where the variance that stopped S6.1 from
//! having a domain comes from, and it is the one place where the viewer
//! shows a confident number next to a registration that is half a metre
//! out. Until something detects it, no survey-level result on real data is
//! limited by anything else.
//!
//! # What is measured
//!
//! Every edge of a survey — the consecutive legs and the skip-one pairs —
//! gets a label from the theodolite (`> LOST` metres of translation error
//! is the wrong basin) and five numbers that a system without truth could
//! have computed. Each number is then swept as a detector and scored by
//! precision and recall against the label. Nothing here is shown as a
//! screenshot; a detector that cannot be scored is not a detector.
//!
//! The five, and why each might work:
//!
//! - **`rmse`**, relative to the survey's own median. M9 found the raw
//!   value separates cleanly on the plateau — 4.5–6 cm against 11–19 cm —
//!   but a raw threshold in metres does not travel between scenes, and a
//!   ratio to the survey's median does.
//! - **overlap**: the share of moving points that found a correspondence at
//!   all. A pose in the wrong place usually has less of the other scan
//!   under it.
//! - **inlier share**: of the correspondences kept, the share whose
//!   residual is inside the sensor noise. At the right minimum the
//!   residuals are the sensor's; at a wrong one they are the geometry's,
//!   and the shape differs even where the RMSE does not.
//! - **restart spread**: register again from nine perturbed starts and take
//!   the largest disagreement. This asks the question directly rather than
//!   by proxy — a minimum that nine pushes all fall back into is a wide one.
//!   It costs nine extra registrations per edge and is the only candidate
//!   here that does.
//! - **cycle error**: the survey has triangles — `i→i+1`, `i+1→i+2` and the
//!   skip-one `i→i+2` — and composing the two legs should return the third.
//!   It uses redundancy nothing else here uses, and it needs no scan at all.
//!
//! ```text
//! cargo run --release -p rigidity-graph --example basin -- <directory> [stations]
//! RIGIDITY_SECTOR=60 …    # the cropped view, where wrong basins are common
//! RIGIDITY_SCAN_EXT=pcd …  # scans other than Hokuyo_<i>.csv
//! RIGIDITY_NOISE=0.02 …    # another sensor's noise; 0.03, the Hokuyo's, by default
//! ```

use std::path::{Path, PathBuf};

use rigidity_core::lie::{Se3, So3};
use rigidity_core::nalgebra::{Matrix3, Matrix4, Vector3, Vector6};
use rigidity_core::neighbors::NeighborSearch;
use rigidity_harness::crop_sector;
use rigidity_pipeline::{
    PrepareParams, Prepared, RegisterParams, median_absolute_residual, prepare_cloud,
    register_pair_observed,
};

/// Sensor noise along the normal, metres. The Hokuyo UTM-30LX, as M9 took it,
/// unless `RIGIDITY_NOISE` names another sensor's. The thresholds of Table 4
/// were fixed with the Hokuyo's value, and a second sensor is reported both
/// ways: with them unchanged, which is the test out of sample, and with its
/// own noise put in their place.
fn noise() -> f64 {
    static NOISE: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *NOISE.get_or_init(|| {
        std::env::var("RIGIDITY_NOISE")
            .ok()
            .and_then(|text| text.parse().ok())
            .unwrap_or(0.03)
    })
}

/// How far from the theodolite counts as the wrong basin, metres.
///
/// Two orders above the theodolite's own accuracy and an order above the
/// worst honest registration these scenes produce, so the label is about
/// the failure and not about the noise around success. The run also reports
/// what changes at 0.30 m, which is where M9 drew it.
const LOST: f64 = 0.10;

/// How far the restarts are pushed, metres and radians.
///
/// Large enough to leave the immediate neighbourhood of the answer — the
/// legs here are 0.6 m — and small enough that a wide basin still pulls
/// them back.
const PUSH: f64 = 0.20;
const TWIST: f64 = 0.05;

struct EdgeCase {
    from: usize,
    to: usize,
    pose: Se3,
    error: f64,
    rmse: f64,
    overlap: f64,
    inlier: f64,
    /// The median absolute point-to-plane residual at the pose found.
    ///
    /// The cheap rule, and the one the paper reports beside the restart: at
    /// the right minimum half the residuals fall inside the sensor's own
    /// error, at a wrong one they do not. It costs one pass where the
    /// restart costs nine registrations, and it was measured here only
    /// after the table that compares the two was written without it.
    median: f64,
    restart: f64,
    cycle: f64,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let directory = PathBuf::from(args.next().unwrap_or_else(|| {
        eprintln!("usage: basin <directory> [stations]");
        std::process::exit(2);
    }));
    let stations: usize = args.next().and_then(|text| text.parse().ok()).unwrap_or(16);
    let sector = std::env::var("RIGIDITY_SECTOR")
        .ok()
        .and_then(|text| text.parse::<f64>().ok())
        .map(f64::to_radians);

    let truth = read_truth(&directory.join("pose_scanner_leica.csv"), stations);
    let prepared = read_scans(&directory, stations, sector);
    let params = RegisterParams::default();

    // The survey as the survey builds it: each leg starts from the one
    // before it, no truth anywhere.
    let mut cases: Vec<EdgeCase> = Vec::new();
    let mut legs: Vec<Se3> = Vec::new();
    let mut carried = Se3::identity();
    for index in 0..stations - 1 {
        let case = measure(&prepared, index, index + 1, carried, &params, &truth);
        carried = case.pose;
        legs.push(case.pose);
        cases.push(case);
    }
    for index in 0..stations.saturating_sub(2) {
        let guess = legs[index] * legs[index + 1];
        let case = measure(&prepared, index, index + 2, guess, &params, &truth);
        cases.push(case);
    }

    // The triangles, and the disagreement each edge is implicated in. An
    // edge sits in at most two of them, and the worst is the one that would
    // make somebody look at it.
    let count = cases.len();
    for triangle in 0..stations.saturating_sub(2) {
        let first = triangle;
        let second = triangle + 1;
        let skip = stations - 1 + triangle;
        if second >= stations - 1 || skip >= count {
            continue;
        }
        let composed = cases[first].pose * cases[second].pose;
        let disagreement = (composed.inverse() * cases[skip].pose).log().norm();
        for member in [first, second, skip] {
            cases[member].cycle = cases[member].cycle.max(disagreement);
        }
    }

    let median_rmse = median(cases.iter().map(|case| case.rmse).collect());
    let lost: Vec<bool> = cases.iter().map(|case| case.error > LOST).collect();
    let far: Vec<bool> = cases.iter().map(|case| case.error > 0.30).collect();
    println!(
        "\n{} edges, {} in the wrong basin at {LOST:.2} m, {} at 0.30 m, median rmse {median_rmse:.4} m",
        count,
        lost.iter().filter(|flag| **flag).count(),
        far.iter().filter(|flag| **flag).count(),
    );

    println!(
        "\n  edge   error     rmse   rmse/med  overlap  inliers   median  restart   cycle   basin"
    );
    for (case, flag) in cases.iter().zip(&lost) {
        println!(
            "  {:2}→{:2}  {:7.4}  {:7.4}  {:7.2}  {:7.3}  {:7.3}  {:7.4}  {:7.4}  {:6.4}   {}",
            case.from,
            case.to,
            case.error,
            case.rmse,
            case.rmse / median_rmse,
            case.overlap,
            case.inlier,
            case.median,
            case.restart,
            case.cycle,
            if *flag { "LOST" } else { "" }
        );
    }

    // Each candidate swept as a detector. `higher_is_worse` says which side
    // of the threshold the suspicion lies on, and the sweep runs over the
    // observed values themselves so that no grid resolution is invented.
    let detectors: [(&str, Vec<f64>, bool); 7] = [
        (
            // What the viewer already does. It was never measured against
            // the truth, and the point of the list below is that it now is.
            "rmse (viewer)",
            cases.iter().map(|case| case.rmse).collect(),
            true,
        ),
        (
            "rmse / median",
            cases.iter().map(|case| case.rmse / median_rmse).collect(),
            true,
        ),
        (
            "overlap",
            cases.iter().map(|case| case.overlap).collect(),
            false,
        ),
        (
            "inlier share",
            cases.iter().map(|case| case.inlier).collect(),
            false,
        ),
        (
            "median residual",
            cases.iter().map(|case| case.median).collect(),
            true,
        ),
        (
            "restart spread",
            cases.iter().map(|case| case.restart).collect(),
            true,
        ),
        (
            "cycle error",
            cases.iter().map(|case| case.cycle).collect(),
            true,
        ),
    ];

    println!(
        "\n  detector          best F1   at threshold   precision  recall   |  recall at precision 1.0"
    );
    for (name, values, higher_is_worse) in &detectors {
        let report = score(values, &lost, *higher_is_worse);
        println!(
            "  {name:<16}  {:6.3}   {:12.4}   {:8.3}  {:6.3}   |  {:6.3}",
            report.f1, report.threshold, report.precision, report.recall, report.clean_recall
        );
    }

    // The best threshold *in this survey* says nothing about whether one
    // threshold serves two. What does is where the two populations lie: if
    // the good edges of one scene reach past the lost edges of another,
    // there is no fixed number to ship, however cleanly each survey
    // separates on its own.
    println!("\n  detector           good edges         lost edges        separated?");
    for (name, values, higher_is_worse) in &detectors {
        let good = range(values, &lost, false);
        let bad = range(values, &lost, true);
        let clear = match (good, bad) {
            (Some((glo, ghi)), Some((blo, bhi))) => {
                if *higher_is_worse {
                    if blo > ghi {
                        format!("yes, gap {:.4} … {:.4}", ghi, blo)
                    } else {
                        format!("no, they overlap ({:.4} … {:.4})", blo, ghi.min(bhi))
                    }
                } else if bhi < glo {
                    format!("yes, gap {:.4} … {:.4}", bhi, glo)
                } else {
                    format!("no, they overlap ({:.4} … {:.4})", glo.max(blo), bhi)
                }
            }
            _ => "one class is empty".to_string(),
        };
        let show = |bounds: Option<(f64, f64)>| match bounds {
            Some((lo, hi)) => format!("{lo:7.4} … {hi:7.4}"),
            None => "      none      ".to_string(),
        };
        println!("  {name:<16}  {}  {}  {clear}", show(good), show(bad));
    }

    // And the fixed operating points, chosen once on the plain at 360° and
    // then not touched. Every other run below is out of sample for them.
    println!("\n  at the fixed thresholds chosen on plain 360°:");
    println!("  detector          threshold   caught  missed  false alarms  clean");
    for ((name, values, higher_is_worse), fixed) in detectors.iter().zip(fixed()) {
        let (mut caught, mut missed, mut alarms, mut clean) = (0, 0, 0, 0);
        for (value, is_lost) in values.iter().zip(&lost) {
            // A value that is not finite means the registration fell over
            // rather than that it is fine, and the negation used to say so
            // by accident. It says so on purpose now.
            let flagged = if !value.is_finite() {
                true
            } else if *higher_is_worse {
                *value >= fixed
            } else {
                *value <= fixed
            };
            match (flagged, is_lost) {
                (true, true) => caught += 1,
                (false, true) => missed += 1,
                (true, false) => alarms += 1,
                (false, false) => clean += 1,
            }
        }
        println!("  {name:<16}  {fixed:9.4}   {caught:6}  {missed:6}  {alarms:12}  {clean:5}");
    }

    // The rule the viewer already uses, and the one measured here, taken
    // together. They are not the same measurement twice: an RMSE is a mean
    // and answers "how far apart are these surfaces on average", while the
    // median answers "are the residuals the sensor's own". A registration
    // can pass the first on a good average and fail the second because
    // most of its correspondences are wrong.
    let (mut caught, mut missed, mut alarms, mut clean) = (0, 0, 0, 0);
    let mut escaped: Vec<f64> = Vec::new();
    for (index, is_lost) in lost.iter().enumerate() {
        let flagged = cases[index].rmse >= fixed()[0] || cases[index].inlier <= fixed()[3];
        match (flagged, is_lost) {
            (true, true) => caught += 1,
            (false, true) => {
                missed += 1;
                escaped.push(cases[index].error);
            }
            (true, false) => alarms += 1,
            (false, false) => clean += 1,
        }
    }
    escaped.sort_by(f64::total_cmp);
    println!(
        "  {:<16}  {:>9}   {caught:6}  {missed:6}  {alarms:12}  {clean:5}",
        "rmse or median", "—"
    );
    println!(
        "\n  the wrong-basin edges it let through were {} m out",
        if escaped.is_empty() {
            "— none —".to_string()
        } else {
            escaped
                .iter()
                .map(|error| format!("{error:.3}"))
                .collect::<Vec<_>>()
                .join(", ")
        }
    );
}

/// The operating points, in the order the detectors are listed.
///
/// Read off the plain at 360° with sixteen stations, rounded into the
/// middle of the gap that survey shows, and then left alone. A threshold
/// tuned on the run it is reported on is not a measurement.
fn fixed() -> [f64; 7] {
    [3.0 * noise(), 1.50, 0.960, 0.500, noise(), 0.050, 0.050]
}

/// The interval one class occupies, or `None` if the class is empty.
fn range(values: &[f64], lost: &[bool], want_lost: bool) -> Option<(f64, f64)> {
    let mut low = f64::INFINITY;
    let mut high = f64::NEG_INFINITY;
    let mut seen = false;
    for (value, is_lost) in values.iter().zip(lost) {
        if *is_lost != want_lost || !value.is_finite() {
            continue;
        }
        seen = true;
        low = low.min(*value);
        high = high.max(*value);
    }
    seen.then_some((low, high))
}

struct Score {
    f1: f64,
    threshold: f64,
    precision: f64,
    recall: f64,
    /// The share of wrong-basin edges caught at a threshold that flags no
    /// good edge at all. This is the number that decides whether a detector
    /// can be acted on without a human: a false alarm on a survey costs a
    /// re-run, a missed basin costs the survey.
    clean_recall: f64,
}

fn score(values: &[f64], lost: &[bool], higher_is_worse: bool) -> Score {
    let positives = lost.iter().filter(|flag| **flag).count() as f64;
    let mut best = Score {
        f1: 0.0,
        threshold: f64::NAN,
        precision: 0.0,
        recall: 0.0,
        clean_recall: 0.0,
    };
    if positives == 0.0 {
        return best;
    }
    for threshold in values.iter().copied() {
        let flagged = |value: f64| {
            if higher_is_worse {
                value >= threshold
            } else {
                value <= threshold
            }
        };
        let (mut hit, mut miss) = (0.0, 0.0);
        for (value, is_lost) in values.iter().zip(lost) {
            if flagged(*value) {
                if *is_lost {
                    hit += 1.0;
                } else {
                    miss += 1.0;
                }
            }
        }
        if hit == 0.0 {
            continue;
        }
        let precision = hit / (hit + miss);
        let recall = hit / positives;
        let f1 = 2.0 * precision * recall / (precision + recall);
        if f1 > best.f1 {
            best = Score {
                f1,
                threshold,
                precision,
                recall,
                clean_recall: best.clean_recall,
            };
        }
        if miss == 0.0 && recall > best.clean_recall {
            best.clean_recall = recall;
        }
    }
    best
}

/// Registers one pair and measures everything about it that does not need
/// the truth, plus the one thing that does, for the label.
fn measure(
    prepared: &[Prepared],
    from: usize,
    to: usize,
    guess: Se3,
    params: &RegisterParams,
    truth: &[Se3],
) -> EdgeCase {
    let result = register_pair_observed(&prepared[to], &prepared[from], guess, params, |_| {
        std::ops::ControlFlow::Continue(())
    });

    // The restarts. Nine pushes: both signs along each translation axis and
    // one about each rotation axis, applied on the left in the target's
    // frame — the same side the ICP updates on, so a push of PUSH metres is
    // PUSH metres of the answer. Nine and not six; the count is the whole
    // cost of this detector and the reason to prefer the residual when it
    // cannot be paid, so it is worth stating correctly.
    let mut restart: f64 = 0.0;
    for axis in 0..6 {
        for sign in [1.0, -1.0] {
            if axis >= 3 && sign < 0.0 {
                continue;
            }
            let mut twist = Vector6::zeros();
            twist[axis] = sign * if axis < 3 { PUSH } else { TWIST };
            let pushed = Se3::exp(&twist) * result.pose;
            let again =
                register_pair_observed(&prepared[to], &prepared[from], pushed, params, |_| {
                    std::ops::ControlFlow::Continue(())
                });
            restart = restart.max((again.pose.translation() - result.pose.translation()).norm());
        }
    }

    let (overlap, inlier) = residual_shape(&prepared[to], &prepared[from], &result.pose, params);
    let median = median_absolute_residual(&prepared[to], &prepared[from], &result.pose, params)
        .unwrap_or(f64::INFINITY);
    let expected = truth[from].inverse() * truth[to];
    EdgeCase {
        from,
        to,
        pose: result.pose,
        error: (result.pose.translation() - expected.translation()).norm(),
        rmse: result.rmse,
        overlap,
        inlier,
        median,
        restart,
        cycle: 0.0,
    }
}

/// The share of moving points that found a correspondence, and the share of
/// those correspondences whose residual is inside the sensor noise.
fn residual_shape(
    moving: &Prepared,
    fixed: &Prepared,
    pose: &Se3,
    params: &RegisterParams,
) -> (f64, f64) {
    let limit = params.max_distance * params.max_distance;
    let mut found = 0usize;
    let mut inside = 0usize;
    let mut scratch = Vec::with_capacity(1);
    for index in 0..moving.cloud.len() {
        let point = pose.transform_point(&moving.cloud.point(index));
        fixed.tree.knn_into(&point, 1, &mut scratch);
        let Some(nearest) = scratch.first() else {
            continue;
        };
        if nearest.distance_squared > limit {
            continue;
        }
        found += 1;
        let target = fixed.cloud.point(nearest.index as usize);
        let normal = fixed.normals[nearest.index as usize];
        if normal.dot(&(point - target)).abs() < noise() {
            inside += 1;
        }
    }
    let total = moving.cloud.len().max(1) as f64;
    (found as f64 / total, inside as f64 / found.max(1) as f64)
}

fn median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
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

fn read_scans(directory: &Path, stations: usize, sector: Option<f64>) -> Vec<Prepared> {
    let params = PrepareParams {
        voxel: 0.05,
        neighbours: 16,
    };
    match sector {
        Some(half) => println!(
            "reading {stations} scans, cropped to ±{:.0}°",
            half.to_degrees()
        ),
        None => println!("reading {stations} scans"),
    }
    (0..stations)
        .map(|index| {
            // `RIGIDITY_SCAN_EXT` as in `global.rs`: the Oxford Spires keyframes
            // are binary PCD, laid out as `Hokuyo_<i>.pcd` by
            // `datasets/spires/prepare_spires.py`.
            let extension = std::env::var("RIGIDITY_SCAN_EXT").unwrap_or_else(|_| "csv".into());
            let path = directory.join(format!("Hokuyo_{index}.{extension}"));
            let raw = rigidity_io::read(&path).unwrap_or_else(|error| {
                eprintln!("{}: {error}", path.display());
                std::process::exit(1);
            });
            let kept = match sector {
                Some(half) => crop_sector(&raw, 0.0, half),
                None => raw,
            };
            prepare_cloud(&kept, &params).expect("the scan prepares")
        })
        .collect()
}
