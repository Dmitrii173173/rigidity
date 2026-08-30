//! Does the global search need the survey, and does it know when it fails?
//!
//! # Why this exists
//!
//! `rigidity_pipeline::register_globally` was measured on one survey, and
//! one survey is one survey. Every other claim this project makes about
//! real data stands on four of them or on sixty combinations; a headline
//! number resting on fifty-nine edges of a single staircase would be the
//! one place where the standard slipped.
//!
//! # What it asks
//!
//! Each edge of a survey — the legs `i→i+1` and the pairs through one — is
//! registered twice. Once the way a survey does it, starting from the leg
//! before, which is the best guess anything has without ground truth. Once
//! with no guess at all, by the search. The theodolite then says which of
//! the two landed in the right place.
//!
//! Three numbers come out, and the third is the one that matters:
//!
//! - how many edges the survey walk loses to a wrong basin;
//! - how many the search loses;
//! - **of those the search loses, how many it says nothing about.** A
//!   search that fails and reports a median residual past the sensor's
//!   noise has failed honestly, and the caller is told. A search that fails
//!   with a median inside the noise is the dangerous case: a confident
//!   wrong answer, which is the exact failure this whole project exists to
//!   refuse.
//!
//! ```text
//! cargo run --release -p rigidity-graph --example global -- <directory> [stations]
//! ```
//!
//! The directory is a scene's `csv_local`, and it needs
//! `pose_scanner_leica.csv` beside the scans — the datasets ship that file
//! in `csv_global` and sometimes in `leica`, and any of the three will do
//! as long as it sits next to the `Hokuyo_*.csv` the run reads.

use std::path::{Path, PathBuf};
use std::time::Instant;

use rigidity_core::lie::{Se3, So3};
use rigidity_core::nalgebra::{Matrix3, Matrix4, Vector3, Vector6};
use rigidity_pipeline::{
    PrepareParams, Prepared, RegisterParams, SearchParams, median_absolute_residual, prepare_cloud,
    register_globally, register_pair_observed,
};

/// Sensor noise along the normal, metres. The Hokuyo UTM-30LX, as M9 took
/// it and as `basin.rs` takes it.
const NOISE: f64 = 0.03;

/// How far from the theodolite counts as the wrong basin, metres. The same
/// line `basin.rs` draws, so the two runs count the same failures.
const LOST: f64 = 0.10;

/// One edge, registered both ways.
struct Edge {
    from: usize,
    to: usize,
    /// Translation error against the theodolite, starting from the leg
    /// before, metres.
    walked: f64,
    /// The same, starting from nothing.
    searched: f64,
    /// What the search said about its own answer, metres.
    median: f64,
    /// And what the same criterion says at the pose the theodolite gives.
    ///
    /// The decisive column when the search fails silently. If the wrong
    /// answer fits no worse than the right one, no rule built on residuals
    /// could have chosen between them, and the failure belongs to the scene
    /// rather than to the search.
    at_truth: f64,
    /// How far the answer moves when it is pushed, metres, or `NAN` when
    /// `RIGIDITY_RESTARTS` is unset.
    ///
    /// The column that says whether the silent failures are a limit of
    /// residual rules or a limit of the scene. A residual cannot see a
    /// wrong place that fits as well as the right one — that is measured.
    /// A restart asks a different question, whether the minimum is wide,
    /// and a repeating scene might answer it either way: the second bay of
    /// a corridor is a real minimum with real walls, so pushes should fall
    /// back into it and the restart should be as quiet as the residual. It
    /// should, and that is not a measurement until it is measured.
    restart: f64,
    seconds: f64,
    /// The walked pose, so the survey can carry it forward without paying
    /// for the same registration twice.
    pose: Se3,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let directory = PathBuf::from(args.next().unwrap_or_else(|| {
        eprintln!("usage: global <directory> [stations]");
        std::process::exit(2);
    }));
    let stations: usize = args.next().and_then(|text| text.parse().ok()).unwrap_or(16);

    let truth = read_truth(&directory.join("pose_scanner_leica.csv"), stations);
    let prepared = read_scans(&directory, stations);
    let params = RegisterParams::default();
    // The net is the thing this run is here to vary, so it comes from the
    // environment rather than from a recompile.
    fn read<T: std::str::FromStr>(name: &str, fallback: T) -> T {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(fallback)
    }
    let search = SearchParams {
        yaws: read("RIGIDITY_YAWS", 12),
        radius: read("RIGIDITY_RADIUS", 0.5),
        grid: read("RIGIDITY_GRID", 3),
        ..SearchParams::default()
    };
    println!(
        "net: {} yaws × {}² offsets over ±{:.2} m = {} starts",
        search.yaws,
        search.grid,
        search.radius,
        search.yaws * search.grid * search.grid
    );

    // The survey as a survey builds it: each leg starts from the one
    // before, and nothing anywhere uses the truth.
    let mut edges: Vec<Edge> = Vec::new();
    let mut legs: Vec<Se3> = Vec::new();
    let mut carried = Se3::identity();
    for index in 0..stations - 1 {
        let edge = measure(
            &prepared,
            index,
            index + 1,
            carried,
            &params,
            &search,
            &truth,
        );
        carried = edge.pose;
        legs.push(carried);
        edges.push(edge);
    }
    for index in 0..stations.saturating_sub(2) {
        let guess = legs[index] * legs[index + 1];
        let edge = measure(&prepared, index, index + 2, guess, &params, &search, &truth);
        edges.push(edge);
    }

    println!(
        "\n  {:>3} {:>3}   {:>9} {:>9} {:>9} {:>9} {:>9}  {:>5}",
        "i", "j", "walked", "searched", "median", "at truth", "restart", "s"
    );
    for edge in &edges {
        let mark = match (edge.walked > LOST, edge.searched > LOST) {
            (true, true) => "still lost",
            (true, false) => "recovered",
            (false, true) => "LOST BY THE SEARCH",
            (false, false) => "",
        };
        println!(
            "  {:>3} {:>3}   {:9.4} {:9.4} {:9.4} {:9.4} {:9.4}  {:5.1}  {mark}",
            edge.from,
            edge.to,
            edge.walked,
            edge.searched,
            edge.median,
            edge.at_truth,
            edge.restart,
            edge.seconds
        );
    }

    let walked_lost = edges.iter().filter(|edge| edge.walked > LOST).count();
    let searched_lost: Vec<&Edge> = edges.iter().filter(|edge| edge.searched > LOST).collect();
    // The dangerous half: wrong, and saying nothing about it.
    let silent = searched_lost
        .iter()
        .filter(|edge| edge.median <= NOISE)
        .count();
    // And the false alarms: right, but reported as suspect.
    let cried_wolf = edges
        .iter()
        .filter(|edge| edge.searched <= LOST && edge.median > NOISE)
        .count();
    let total: f64 = edges.iter().map(|edge| edge.seconds).sum();

    println!("\n  {} edges over {stations} stations", edges.len());
    println!("  in a wrong basin at {LOST:.2} m:");
    println!("    walking the survey  {walked_lost}");
    println!("    searching, no guess {}", searched_lost.len());
    println!(
        "  of the {} the search lost, {} said so (median past {NOISE:.2} m) \
         and {silent} did not",
        searched_lost.len(),
        searched_lost.len() - silent
    );
    println!(
        "  false alarms: {cried_wolf} of {} sound edges",
        edges.len() - searched_lost.len()
    );
    // Of the silent failures, the ones no residual rule could have caught:
    // the wrong place fits at least as well as the right one, so the scene
    // is ambiguous and the criterion is not at fault.
    let unresolvable = searched_lost
        .iter()
        .filter(|edge| edge.median <= NOISE && edge.median <= edge.at_truth)
        .count();
    println!(
        "  of the {silent} silent, {unresolvable} fit at least as well as the truth does \
         — the scene repeats, and no residual could tell"
    );
    println!(
        "  {:.1} s of searching, {:.1} s an edge",
        total,
        total / edges.len().max(1) as f64
    );
}

/// One edge, registered from the guess and again from nothing.
#[allow(clippy::too_many_arguments)]
fn measure(
    prepared: &[Prepared],
    from: usize,
    to: usize,
    guess: Se3,
    params: &RegisterParams,
    search: &SearchParams,
    truth: &[Se3],
) -> Edge {
    let want = truth[from].inverse() * truth[to];
    let error = |pose: &Se3| (pose.translation() - want.translation()).norm();

    let walked = register_pair_observed(&prepared[to], &prepared[from], guess, params, |_| {
        std::ops::ControlFlow::Continue(())
    });

    let began = Instant::now();
    let found = register_globally(&prepared[to], &prepared[from], params, search);
    let seconds = began.elapsed().as_secs_f64();

    // No candidate at all is not an error to hide: it is the answer, and it
    // is reported as a failure the search declared rather than one it
    // stumbled into.
    let (searched, median) = match &found {
        Some(found) => (error(&found.result.pose), found.median_residual),
        None => (f64::INFINITY, f64::INFINITY),
    };
    let at_truth = median_absolute_residual(&prepared[to], &prepared[from], &want, params)
        .unwrap_or(f64::INFINITY);

    // The same nine pushes `basin.rs` applies, from the pose the search
    // found rather than the pose the walk found, so the number is
    // comparable with the one Table IV scores.
    let restart = match (std::env::var("RIGIDITY_RESTARTS").is_ok(), &found) {
        (true, Some(found)) => spread(prepared, from, to, &found.result.pose, params),
        _ => f64::NAN,
    };

    Edge {
        from,
        to,
        walked: error(&walked.pose),
        searched,
        median,
        at_truth,
        restart,
        seconds,
        pose: walked.pose,
    }
}

/// How far an answer moves when it is pushed away and re-registered.
///
/// `basin.rs` constants, deliberately: PUSH of 0.20 m leaves the immediate
/// neighbourhood of a 0.6 m leg while a wide basin still pulls it back, and
/// the pushes go on the left, the side ICP updates on, so a push of 0.20 m
/// is 0.20 m of the answer.
fn spread(
    prepared: &[Prepared],
    from: usize,
    to: usize,
    pose: &Se3,
    params: &RegisterParams,
) -> f64 {
    const PUSH: f64 = 0.20;
    const TWIST: f64 = 0.05;
    let mut worst: f64 = 0.0;
    for axis in 0..6 {
        for sign in [1.0, -1.0] {
            if axis >= 3 && sign < 0.0 {
                continue;
            }
            let mut twist = Vector6::zeros();
            twist[axis] = sign * if axis < 3 { PUSH } else { TWIST };
            let again = register_pair_observed(
                &prepared[to],
                &prepared[from],
                Se3::exp(&twist) * *pose,
                params,
                |_| std::ops::ControlFlow::Continue(()),
            );
            worst = worst.max((again.pose.translation() - pose.translation()).norm());
        }
    }
    worst
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
