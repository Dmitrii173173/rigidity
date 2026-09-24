//! Is there a second place that fits as well as the right one, and does a
//! restart see it?
//!
//! # Why this exists
//!
//! `global.rs` found the limit by searching from nothing: in the ETH ASL
//! corridor five edges landed one bay along, fitted at least as well as the
//! theodolite's pose, and nine pushes did not move them. That is one
//! corridor. The obvious second scene is a terrestrial scan of a façade,
//! whose windows repeat, but the search cannot be carried across unchanged:
//! it lays its net in yaw and in the two horizontal offsets and leaves the
//! height to the registration, which is right for a robot on a floor and
//! wrong for tripods whose heights differ by metres. Widening the net to
//! reach them is a change of method and hours of compute.
//!
//! The question the corridor raised does not need a blind search, though.
//! It is whether, near the right answer, there is a wrong one that fits as
//! well, and whether that wrong minimum is as wide as the right one. This
//! asks it directly: each edge is registered from a ring of starts around
//! the true pose — every `360 / directions` degrees, every `step` metres out
//! to `range` — and every place the registration settles more than `LOST`
//! from the truth is a wrong minimum the scene offers. A wrong minimum whose
//! median residual is no worse than at the truth is one no residual rule
//! could reject; for each of those the nine pushes of `basin.rs` are applied
//! and the spread is read against the threshold of Table 4.
//!
//! The ring uses the truth to place its starts, and says so: it is a probe of
//! what the scene offers, not a registration pipeline, and nothing it finds
//! is a rate of anything. It is run on the corridor first, where the answer
//! is known, and then on the façade.
//!
//! ```text
//! cargo run --release -p rigidity-graph --example repeat -- <directory> [stations]
//! RIGIDITY_SCAN_EXT=ply …          # the ETH TLS scans, laid out by datasets/tls/prepare_tls.py
//! RIGIDITY_PROBE_EDGES=11-13,12-14 # only these edges; default the legs and the pairs through one
//! RIGIDITY_PROBE_RANGE=6 RIGIDITY_PROBE_STEP=0.5 RIGIDITY_PROBE_DIRECTIONS=8
//! RIGIDITY_PROBE_WORLD_AXIS=120   # lay the first ray along this bearing of the first
//!                                 # station's frame, in degrees, turned into each edge's own
//! ```
//!
//! The axis exists for scenes whose repetition runs one way. A corridor's runs along
//! the robot's heading, which a ray of the default ring already follows; a façade's
//! runs along the wall, at whatever heading the tripod happened to face, and eight
//! rays forty-five degrees apart leave more than a metre between them three metres
//! out — wider than the basin the façade's own edges show. So there the rays go along
//! the wall and back, closer together.

use std::path::{Path, PathBuf};

use rigidity_core::lie::{Se3, So3};
use rigidity_core::nalgebra::{Matrix3, Matrix4, Vector3, Vector6};
use rigidity_pipeline::{
    PrepareParams, Prepared, RegisterParams, median_absolute_residual, prepare_cloud,
    register_pair_observed,
};

/// How far from the reference counts as the wrong basin, metres. The line
/// `basin.rs` and `global.rs` draw.
const LOST: f64 = 0.10;

/// The restart threshold of Table 4, metres.
const THRESHOLD: f64 = 0.05;

/// Two wrong answers closer than this are the same minimum, metres.
const SAME: f64 = 0.10;

/// A wrong place whose median is within this factor of the reference's is a
/// near tie, and its spread is read too. The corridor needs it: three of its
/// five unresolvable edges fit the copy within half a millimetre of the
/// truth, and which side of equal a ring start lands on is a hair.
const NEAR: f64 = 1.15;

fn main() {
    let mut args = std::env::args().skip(1);
    let directory = PathBuf::from(args.next().unwrap_or_else(|| {
        eprintln!("usage: repeat <directory> [stations]");
        std::process::exit(2);
    }));
    let stations: usize = args.next().and_then(|text| text.parse().ok()).unwrap_or(16);
    fn read<T: std::str::FromStr>(name: &str, fallback: T) -> T {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(fallback)
    }
    let range: f64 = read("RIGIDITY_PROBE_RANGE", 6.0);
    let step: f64 = read("RIGIDITY_PROBE_STEP", 0.5);
    let directions: usize = read("RIGIDITY_PROBE_DIRECTIONS", 8);
    let axis: Option<f64> = std::env::var("RIGIDITY_PROBE_WORLD_AXIS")
        .ok()
        .and_then(|value| value.parse().ok());

    let edges: Vec<(usize, usize)> = match std::env::var("RIGIDITY_PROBE_EDGES") {
        Ok(list) => list
            .split(',')
            .filter_map(|pair| {
                let (a, b) = pair.split_once('-')?;
                Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
            })
            .collect(),
        Err(_) => (0..stations - 1)
            .map(|index| (index, index + 1))
            .chain((0..stations.saturating_sub(2)).map(|index| (index, index + 2)))
            .collect(),
    };
    let needed = edges.iter().map(|&(a, b)| a.max(b) + 1).max().unwrap_or(0);
    let truth = read_truth(&directory.join("pose_scanner_leica.csv"), needed);
    let prepared = read_scans(&directory, needed);
    let params = RegisterParams::default();

    let ring = |first: f64| -> Vec<Vector3<f64>> {
        let mut rings = Vec::new();
        for k in 0..directions {
            let angle = first + std::f64::consts::TAU * k as f64 / directions as f64;
            let mut distance = step;
            while distance <= range + 1e-9 {
                rings.push(Vector3::new(angle.cos(), angle.sin(), 0.0) * distance);
                distance += step;
            }
        }
        rings
    };
    let per_edge = ring(0.0).len();
    println!(
        "ring: {directions} directions × {} distances to {range:.1} m = {per_edge} starts an edge{}",
        per_edge / directions.max(1),
        match axis {
            Some(bearing) =>
                format!(", the first along {bearing:.1}° of the first station's frame"),
            None => String::new(),
        }
    );

    let mut with_twin = 0;
    let mut twin_flagged = 0;
    let mut twin_quiet = 0;
    let mut near_flagged = 0;
    let mut near_quiet = 0;
    for &(from, to) in &edges {
        let want = truth[from].inverse() * truth[to];
        // The rays are laid in the frame of `from`, so a bearing given in the first
        // station's frame is turned by that station's heading.
        let first = axis.map_or(0.0, |bearing| {
            let rotation = truth[from].rotation().matrix();
            bearing.to_radians() - rotation[(1, 0)].atan2(rotation[(0, 0)])
        });
        let rings = ring(first);
        let at_truth = median_absolute_residual(&prepared[to], &prepared[from], &want, &params)
            .unwrap_or(f64::INFINITY);

        // Every start is independent, so they run side by side, as many at a
        // time as the machine has cores; the result does not depend on the
        // order they finish in.
        let width = std::thread::available_parallelism().map_or(8, |n| n.get());
        let mut settled: Vec<(Se3, f64, f64)> = Vec::with_capacity(rings.len());
        for chunk in rings.chunks(width) {
            settled.extend(std::thread::scope(|scope| {
                let handles: Vec<_> = chunk
                    .iter()
                    .map(|offset| {
                        let (prepared, params) = (&prepared, &params);
                        scope.spawn(move || {
                            let start = Se3::from_parts(So3::identity(), *offset) * want;
                            let found = register_pair_observed(
                                &prepared[to],
                                &prepared[from],
                                start,
                                params,
                                |_| std::ops::ControlFlow::Continue(()),
                            );
                            let error = (found.pose.translation() - want.translation()).norm();
                            let median = median_absolute_residual(
                                &prepared[to],
                                &prepared[from],
                                &found.pose,
                                params,
                            )
                            .unwrap_or(f64::INFINITY);
                            (found.pose, error, median)
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|handle| handle.join().expect("a start"))
                    .collect::<Vec<_>>()
            }));
        }

        let returned = settled
            .iter()
            .filter(|(_, error, _)| *error <= LOST)
            .count();
        // The distinct wrong places, in the order the ring found them.
        let mut minima: Vec<(Se3, f64, f64, usize)> = Vec::new();
        for (pose, error, median) in settled.iter().filter(|(_, error, _)| *error > LOST) {
            match minima
                .iter_mut()
                .find(|(other, ..)| (pose.translation() - other.translation()).norm() < SAME)
            {
                Some(existing) => existing.3 += 1,
                None => minima.push((*pose, *error, *median, 1)),
            }
        }

        println!(
            "\n  edge {from}-{to}: median at the reference {:.4} m; {returned} of {} starts \
             return to it, {} wrong places",
            at_truth,
            settled.len(),
            minima.len()
        );
        let mut twin_here = false;
        for (pose, error, median, count) in &minima {
            let as_good = *median <= at_truth;
            let near = !as_good && *median <= at_truth * NEAR;
            let spread = if as_good || near {
                spread(&prepared, from, to, pose, &params)
            } else {
                f64::NAN
            };
            let verdict = match (as_good, near, spread >= THRESHOLD) {
                (true, _, true) => "FITS AS WELL — restart flags it",
                (true, _, false) => "FITS AS WELL — restart quiet",
                (false, true, true) => "near tie — restart flags it",
                (false, true, false) => "near tie — restart quiet",
                (false, false, _) => "fits worse",
            };
            if near {
                if spread >= THRESHOLD {
                    near_flagged += 1;
                } else {
                    near_quiet += 1;
                }
            }
            if as_good {
                twin_here = true;
                if spread >= THRESHOLD {
                    twin_flagged += 1;
                } else {
                    twin_quiet += 1;
                }
            }
            println!(
                "    {error:7.3} m out   median {median:.4}   from {count:>3} starts   \
                 spread {spread:8.4}   {verdict}"
            );
        }
        if twin_here {
            with_twin += 1;
        }
    }

    println!("\n  {} edges probed", edges.len());
    println!("  with a wrong place that fits at least as well as the reference: {with_twin}");
    println!(
        "  such places: {} — the restart flags {twin_flagged} and is quiet on {twin_quiet} \
         (threshold {THRESHOLD:.2} m)",
        twin_flagged + twin_quiet
    );
    println!(
        "  near ties (median within {:.0} % of the reference's): {} — the restart flags \
         {near_flagged} and is quiet on {near_quiet}",
        (NEAR - 1.0) * 100.0,
        near_flagged + near_quiet
    );
}

/// The nine pushes of `basin.rs` and `global.rs`, from `pose`.
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
            let extension = std::env::var("RIGIDITY_SCAN_EXT").unwrap_or_else(|_| "csv".into());
            let path = directory.join(format!("Hokuyo_{index}.{extension}"));
            let raw = rigidity_io::read(&path).unwrap_or_else(|error| {
                eprintln!("{}: {error}", path.display());
                std::process::exit(1);
            });
            prepare_cloud(&raw, &params).expect("the scan prepares")
        })
        .collect()
}
