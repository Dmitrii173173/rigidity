//! The gate: a loop with one leg that could not see along itself.
//!
//! A survey walks a closed circuit and registers each scan against the last.
//! One leg of it runs down a corridor, where the registration has nothing to
//! measure along the corridor's axis and returns whatever it started with.
//! Every pose-graph package takes that edge's `JᵀWJ` at face value, and
//! `JᵀWJ` is not zero along the axis — it is merely small, and a small
//! weight on an arbitrary number still drags the survey towards it.
//!
//! Both halves of the comparison are built from the *same* correspondences.
//! The only difference is what is made of them: `JᵀWJ` as it stands, or the
//! spectrum with the directions the geometry did not determine removed. If
//! the test rebuilt one of the two from different geometry it would be
//! measuring the geometry.

use rigidity_core::icp::{Kernel, point_to_plane_row};
use rigidity_core::lie::{Se3, So3};
use rigidity_core::nalgebra::{Matrix6, Vector3, Vector6};
use rigidity_core::observability::{
    Analysis, Conditioning, Correspondence, ObservabilityCriteria, analyse,
};
use rigidity_graph::{Edge, OptimiseParams, PoseGraph, weighted_information};

/// How finely the end faces are sampled in the closed room.
const CLOSED: usize = 60;
/// And in the corridor: a few points at the far end, not none.
const CORRIDOR_CAP: usize = 1;

/// Measurement noise the scenes are analysed under, metres.
const NOISE: f64 = 2e-3;
/// The accuracy asked of the survey, metres.
///
/// A millimetre, which is what the ETH theodolite ground truth the rest of
/// this project is measured against actually delivers. It is also what puts
/// the corridor's worst direction outside and its other five well inside:
/// the spreads are 2.4e-5 … 1.4e-4 and then 1.4e-3, so the line falls in
/// the gap rather than through the middle of a cluster.
const TOLERANCE: f64 = 1e-3;

fn criteria() -> ObservabilityCriteria {
    ObservabilityCriteria {
        noise_sigma: NOISE,
        tolerance: TOLERANCE,
    }
}

/// Points and normals on the faces of a box, in the scan's own frame.
///
/// `cap_steps` is how finely the two faces across the corridor's axis are
/// sampled, and it is the whole difference between the two edges here. At
/// `STEPS` the room is closed and every direction is determined. At a
/// handful it is a corridor with something at the far end — a door frame, a
/// wall at the edge of the range — which is what a real corridor scan
/// actually contains.
///
/// At zero it would be a *perfectly* degenerate corridor, and that is the
/// case this test deliberately does not use. Every normal would then be
/// exactly perpendicular to the axis, `JᵀWJ` would be exactly singular
/// along it, and the naive weighting would ignore the direction for free —
/// there would be nothing for conditioning to be better at. The interesting
/// case, and the real one, is the matrix that is invertible and wrong: a
/// few points make it so, and the pose along the axis is still arbitrary.
fn room(cap_steps: usize) -> Vec<Correspondence> {
    const HALF_LENGTH: f64 = 10.0;
    const HALF_WIDTH: f64 = 2.0;
    const STEPS: usize = 60;

    let mut out = Vec::new();
    let mut push = |point: Vector3<f64>, normal: Vector3<f64>| {
        out.push(Correspondence {
            point,
            normal,
            residual: 0.0,
        });
    };

    for a in 0..STEPS {
        let along = -HALF_LENGTH + 2.0 * HALF_LENGTH * (a as f64 + 0.5) / STEPS as f64;
        for b in 0..STEPS {
            let across = -HALF_WIDTH + 2.0 * HALF_WIDTH * (b as f64 + 0.5) / STEPS as f64;
            let up = 0.5 + 2.0 * (b as f64 + 0.5) / STEPS as f64;
            // Two walls facing each other, and a floor between them.
            push(
                Vector3::new(along, HALF_WIDTH, up),
                Vector3::new(0.0, -1.0, 0.0),
            );
            push(
                Vector3::new(along, -HALF_WIDTH, up),
                Vector3::new(0.0, 1.0, 0.0),
            );
            push(
                Vector3::new(along, across, 0.0),
                Vector3::new(0.0, 0.0, 1.0),
            );
        }
    }
    if cap_steps > 0 {
        for a in 0..cap_steps {
            let across = -HALF_WIDTH + 2.0 * HALF_WIDTH * (a as f64 + 0.5) / cap_steps as f64;
            for b in 0..cap_steps {
                let up = 0.5 + 2.0 * (b as f64 + 0.5) / cap_steps as f64;
                push(
                    Vector3::new(HALF_LENGTH, across, up),
                    Vector3::new(-1.0, 0.0, 0.0),
                );
                push(
                    Vector3::new(-HALF_LENGTH, across, up),
                    Vector3::new(1.0, 0.0, 0.0),
                );
            }
        }
    }
    out
}

fn conditioning_of(points: &[Correspondence]) -> Analysis {
    analyse(points.len(), Kernel::Squared, |index| Some(points[index]))
        .expect("the correspondences should analyse")
}

/// `JᵀWJ / σ²`, which is what a pose-graph package is handed and believes.
///
/// Divided by the noise variance so that it is an inverse covariance rather
/// than a sum of squares, and therefore comparable in magnitude with what
/// `weighted_information` produces. Without that the comparison would be
/// between two matrices scaled differently, and the winner would be
/// whichever happened to be larger.
fn naive_information(points: &[Correspondence]) -> Matrix6<f64> {
    let mut hessian = Matrix6::zeros();
    for item in points {
        let row = point_to_plane_row(&item.point, &item.normal);
        hessian += row * row.transpose();
    }
    hessian / (NOISE * NOISE)
}

/// Eight stations around a rectangle, each turned to face the way it walks.
fn truth() -> Vec<Se3> {
    let corners = [
        (0.0, 0.0),
        (20.0, 0.0),
        (40.0, 0.0),
        (40.0, 20.0),
        (40.0, 40.0),
        (20.0, 40.0),
        (0.0, 40.0),
        (0.0, 20.0),
    ];
    corners
        .iter()
        .enumerate()
        .map(|(index, (x, y))| {
            let yaw = index as f64 * std::f64::consts::FRAC_PI_4 * 0.5;
            Se3::from_parts(
                So3::exp(&Vector3::new(0.0, 0.0, yaw)),
                Vector3::new(*x, *y, 0.0),
            )
        })
        .collect()
}

/// How far the estimate is from the truth, worst node, metres.
///
/// Translation only, and reported as the worst rather than the mean: a
/// survey is judged by the station that ended up furthest from where it is,
/// not by the average station.
fn drift(estimate: &[Se3], truth: &[Se3]) -> f64 {
    estimate
        .iter()
        .zip(truth)
        .map(|(a, b)| (a.translation() - b.translation()).norm())
        .fold(0.0, f64::max)
}

/// Builds the loop. `weighted` chooses which information every edge carries.
///
/// The corridor is leg 2→3, and its measurement is displaced along the
/// corridor's own axis by `slip` — which is what an unobservable direction
/// gives you: not noise around the right answer, an arbitrary answer.
fn loop_graph(weighted: bool, slip: f64, conditioning: &(Conditioning, Conditioning)) -> PoseGraph {
    let truth = truth();
    let nodes = truth.len();
    const CORRIDOR: usize = 2;

    let capped_naive = naive_information(&room(CLOSED));
    let corridor_naive = naive_information(&room(CORRIDOR_CAP));
    let capped_weighted = weighted_information(&conditioning.0, &criteria());
    let corridor_weighted = weighted_information(&conditioning.1, &criteria());

    // Every node starts at the truth, so that what the optimisation has to
    // undo is the corridor's slip and nothing else. A test that also
    // perturbed the starting poses would be measuring convergence.
    let mut graph = PoseGraph::new(truth.clone());
    for from in 0..nodes {
        let to = (from + 1) % nodes;
        let exact = truth[from].inverse() * truth[to];
        let corridor = from == CORRIDOR;
        // The slip is a left perturbation in the from-node's frame, along
        // `x`, which is the axis `room(CORRIDOR_CAP)` cannot see.
        let measurement = if corridor {
            Se3::exp(&Vector6::new(slip, 0.0, 0.0, 0.0, 0.0, 0.0)) * exact
        } else {
            exact
        };
        let information = match (corridor, weighted) {
            (true, true) => corridor_weighted,
            (true, false) => corridor_naive,
            (false, true) => capped_weighted,
            (false, false) => capped_naive,
        };
        graph
            .push(Edge {
                from,
                to,
                measurement,
                information,
            })
            .expect("the edge names real nodes");
    }
    graph
}

/// The corridor really is degenerate, and the capped room really is not.
///
/// Asserted before the comparison rather than assumed by it: if the
/// geometry stopped being degenerate the gate below would still pass, for
/// the uninteresting reason that there was nothing to weight away.
#[test]
fn the_corridor_loses_exactly_one_direction() {
    let corridor = conditioning_of(&room(CORRIDOR_CAP)).conditioning;
    let capped = conditioning_of(&room(CLOSED)).conditioning;

    let lost = |c: &Conditioning| {
        c.uncertainty(NOISE)
            .iter()
            .filter(|spread| **spread > TOLERANCE)
            .count()
    };
    assert_eq!(lost(&capped), 0, "the closed room should determine all six");
    assert_eq!(lost(&corridor), 1, "the corridor should lose exactly one");

    // And the direction it loses is translation along the corridor.
    let worst = corridor
        .uncertainty(NOISE)
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index)
        .expect("six directions");
    let direction = corridor.direction_in_world(worst).normalize();
    let along_x = direction[0].abs();
    assert!(
        along_x > 0.99,
        "the lost direction is {direction:?}, and it should be translation along x"
    );
}

/// The gate. Weighting by conditioning leaves the survey measurably closer
/// to where it actually is.
#[test]
fn the_weighted_loop_drifts_less_than_the_naive_one() {
    const SLIP: f64 = 0.40;

    let capped = conditioning_of(&room(CLOSED)).conditioning;
    let corridor = conditioning_of(&room(CORRIDOR_CAP)).conditioning;
    let pair = (capped, corridor);

    let truth = truth();
    let params = OptimiseParams::default();

    let mut naive = loop_graph(false, SLIP, &pair);
    let naive_report = naive.optimise(&params).expect("the naive graph solves");
    let naive_drift = drift(naive.poses(), &truth);

    let mut weighted = loop_graph(true, SLIP, &pair);
    let weighted_report = weighted
        .optimise(&params)
        .expect("the weighted graph solves");
    let weighted_drift = drift(weighted.poses(), &truth);

    println!(
        "slip {SLIP} m\n  naive:    drift {naive_drift:.6} m  \
         ({} iterations, cost {:.3e} → {:.3e})\n  weighted: drift {weighted_drift:.6} m  \
         ({} iterations, cost {:.3e} → {:.3e})",
        naive_report.iterations,
        naive_report.cost[0],
        naive_report.cost[1],
        weighted_report.iterations,
        weighted_report.cost[0],
        weighted_report.cost[1],
    );

    assert!(
        weighted_drift < naive_drift,
        "weighting did not help: {weighted_drift} m against {naive_drift} m"
    );
    // "Measurably less" made a number: the naive survey should carry a
    // sizeable part of the slip and the weighted one should not.
    assert!(
        weighted_drift < 0.1 * naive_drift,
        "weighting helped, but only by {:.1}×",
        naive_drift / weighted_drift
    );
}

/// The same graph twice gives the same numbers, bit for bit.
///
/// The other half of the gate. The crate is single-threaded, so this cannot
/// fail by a race — what it guards against is an iteration order that
/// depends on a hash, which is how determinism is usually lost in code that
/// has no threads at all.
#[test]
fn the_answer_is_reproducible_bit_for_bit() {
    let capped = conditioning_of(&room(CLOSED)).conditioning;
    let corridor = conditioning_of(&room(CORRIDOR_CAP)).conditioning;
    let pair = (capped, corridor);
    let params = OptimiseParams::default();

    let bits = |graph: &PoseGraph| -> Vec<u64> {
        graph
            .poses()
            .iter()
            .flat_map(|pose| {
                let m = pose.matrix();
                (0..16).map(move |index| m[index].to_bits())
            })
            .collect()
    };

    let mut first = loop_graph(true, 0.4, &pair);
    first.optimise(&params).expect("solves");
    for _ in 0..3 {
        let mut again = loop_graph(true, 0.4, &pair);
        again.optimise(&params).expect("solves");
        assert_eq!(bits(&first), bits(&again), "two runs disagreed");
    }
}
