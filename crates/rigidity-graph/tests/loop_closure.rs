//! The gate for S3: one extra edge takes the drift out of a survey.
//!
//! A survey walked as a chain — every scan registered against the last one —
//! has no redundancy anywhere. Each measurement is believed exactly, because
//! nothing contradicts it, and every small error is carried forward to the
//! end. That is not a failure of the optimiser: a chain is a tree, its
//! residual is zero at the answer it gives, and the answer is wrong.
//!
//! Closing the loop is one more measurement, between two scans that are next
//! to each other in the room and far apart in the chain. It is the first
//! thing in the survey that can disagree with anything, and that is what
//! makes the accumulated error visible and removable.

use rigidity_core::lie::{Se3, So3};
use rigidity_core::nalgebra::{Matrix6, Vector3, Vector6};
use rigidity_graph::{Edge, OptimiseParams, PoseGraph};

/// Stations around a closed circuit, each turned to face the way it walks.
fn truth(stations: usize) -> Vec<Se3> {
    let radius = 20.0;
    (0..stations)
        .map(|index| {
            let angle = index as f64 / stations as f64 * std::f64::consts::TAU;
            Se3::from_parts(
                So3::exp(&Vector3::new(0.0, 0.0, angle)),
                Vector3::new(radius * angle.cos(), radius * angle.sin(), 0.0),
            )
        })
        .collect()
}

/// A registration that is slightly wrong, always in the same direction.
///
/// Systematic rather than random, and that is the case worth testing: a
/// scanner that is a little out of calibration, or a voxel that is a little
/// too coarse, biases every pair the same way. Random errors partly cancel
/// along a chain; a bias does not, which is why surveys are walked in loops.
fn bias() -> Se3 {
    Se3::exp(&Vector6::new(0.004, -0.002, 0.0, 0.0, 0.0, 0.0015))
}

/// Even weight in every direction: this test is about redundancy, not about
/// conditioning. `degenerate_leg.rs` is where the weighting is on trial.
fn information() -> Matrix6<f64> {
    Matrix6::identity() * 1e6
}

/// Builds the survey. `closed` adds the one edge between the last station
/// and the first.
fn survey(stations: usize, closed: bool) -> PoseGraph {
    let truth = truth(stations);
    // Every node starts at the truth. What the solve has to cope with is the
    // measurements disagreeing with each other, not a bad initial guess.
    let mut graph = PoseGraph::new(truth.clone());
    let legs = if closed { stations } else { stations - 1 };
    for from in 0..legs {
        let to = (from + 1) % stations;
        let exact = truth[from].inverse() * truth[to];
        // The closing edge is the honest one: those two scans are next to
        // each other and their registration is as good as any other single
        // one. What makes it valuable is not that it is better, it is that
        // it is the first measurement anything can be checked against.
        let measurement = if closed && from == stations - 1 {
            exact
        } else {
            bias() * exact
        };
        graph
            .push(Edge {
                from,
                to,
                measurement,
                information: information(),
            })
            .expect("the edge names real nodes");
    }
    graph
}

/// How far the worst station ends up from where it is, metres.
fn drift(graph: &PoseGraph, truth: &[Se3]) -> f64 {
    graph
        .poses()
        .iter()
        .zip(truth)
        .map(|(a, b)| (a.translation() - b.translation()).norm())
        .fold(0.0, f64::max)
}

/// The gate. Closing the loop removes most of the drift.
#[test]
fn closing_the_loop_takes_the_drift_out() {
    const STATIONS: usize = 12;
    let truth = truth(STATIONS);
    let params = OptimiseParams::default();

    let mut open = survey(STATIONS, false);
    let open_report = open.optimise(&params).expect("a chain solves");
    let open_drift = drift(&open, &truth);

    let mut closed = survey(STATIONS, true);
    let closed_report = closed.optimise(&params).expect("a loop solves");
    let closed_drift = drift(&closed, &truth);

    println!(
        "{STATIONS} stations\n  open chain: drift {open_drift:.4} m  \
         (cost {:.3e} → {:.3e})\n  closed:     drift {closed_drift:.4} m  \
         (cost {:.3e} → {:.3e})",
        open_report.cost[0], open_report.cost[1], closed_report.cost[0], closed_report.cost[1],
    );

    // A chain has nothing to argue with, so its residual is zero at an
    // answer that is wrong. Asserted, because it is the reason the closure
    // is worth anything and it would be easy to believe the opposite.
    assert!(
        open_report.cost[1] < 1e-12,
        "a chain should reach zero cost, and reached {:e}",
        open_report.cost[1]
    );
    assert!(
        closed_report.cost[1] > open_report.cost[1],
        "a closed loop cannot reach zero: its measurements disagree"
    );

    assert!(
        closed_drift < 0.35 * open_drift,
        "closing the loop cut the drift only from {open_drift} m to {closed_drift} m"
    );
}

/// And the closure is what did it, not the extra weight.
///
/// The obvious alternative explanation for the test above is that a
/// twelve-edge graph simply has more constraint than an eleven-edge one.
/// Adding a *second* edge between two stations that are already joined adds
/// exactly as much weight and no redundancy across the loop, and it changes
/// nothing — which is what says the shape of the graph is what mattered.
#[test]
fn a_duplicate_edge_is_not_a_closure() {
    const STATIONS: usize = 12;
    let truth = truth(STATIONS);
    let params = OptimiseParams::default();

    let mut open = survey(STATIONS, false);
    open.optimise(&params).expect("a chain solves");
    let open_drift = drift(&open, &truth);

    let mut doubled = survey(STATIONS, false);
    let exact = truth[0].inverse() * truth[1];
    doubled
        .push(Edge {
            from: 0,
            to: 1,
            measurement: bias() * exact,
            information: information(),
        })
        .expect("the edge names real nodes");
    doubled.optimise(&params).expect("it solves");
    let doubled_drift = drift(&doubled, &truth);

    let change = (doubled_drift - open_drift).abs() / open_drift;
    assert!(
        change < 1e-6,
        "a duplicate edge changed the drift by {change} relative, so the \
         previous test may be measuring weight rather than closure"
    );
}

/// Where the closure is put decides who carries the remaining error.
///
/// Not an assertion about which is better — both are correct answers to
/// different questions — but a check that the anchor does what it says.
/// The anchored station does not move at all, and the error is spread over
/// the others.
#[test]
fn the_anchor_stays_exactly_where_it_was() {
    const STATIONS: usize = 12;
    let truth = truth(STATIONS);
    for anchor in [0, 5, STATIONS - 1] {
        let mut graph = survey(STATIONS, true);
        graph
            .optimise(&OptimiseParams {
                anchor,
                ..OptimiseParams::default()
            })
            .expect("it solves");
        let moved = (graph.poses()[anchor].matrix() - truth[anchor].matrix())
            .abs()
            .max();
        assert_eq!(moved, 0.0, "anchor {anchor} moved by {moved}");
    }
}

/// What `shape` says about the three graphs a survey can be.
#[test]
fn the_shape_says_whether_anything_is_checked() {
    const STATIONS: usize = 12;

    let chain = survey(STATIONS, false).shape(0);
    assert_eq!(chain.joined, STATIONS);
    assert_eq!(chain.closures, 0, "a chain closes nothing");
    assert_eq!(chain.adrift, 0);

    let closed = survey(STATIONS, true).shape(0);
    assert_eq!(closed.closures, 1, "one loop, one closure");
    assert_eq!(closed.adrift, 0);

    // A second closure across the middle, which is what a survey that walked
    // a figure of eight would have.
    let mut across = survey(STATIONS, true);
    let truth = truth(STATIONS);
    across
        .push(Edge {
            from: 0,
            to: STATIONS / 2,
            measurement: truth[0].inverse() * truth[STATIONS / 2],
            information: information(),
        })
        .expect("the edge names real nodes");
    assert_eq!(across.shape(0).closures, 2);
}

/// A survey in two pieces says so before the solve says it cannot.
#[test]
fn a_disconnected_survey_is_visible_in_advance() {
    let truth = truth(4);
    let mut graph = PoseGraph::new(truth.clone());
    // Two pairs, joined to each other and to nothing else.
    for (from, to) in [(0, 1), (2, 3)] {
        graph
            .push(Edge {
                from,
                to,
                measurement: truth[from].inverse() * truth[to],
                information: information(),
            })
            .expect("the edge names real nodes");
    }

    let shape = graph.shape(0);
    assert_eq!(shape.joined, 4);
    assert_eq!(shape.closures, 0);
    assert_eq!(shape.adrift, 2, "the far pair is not joined to the anchor");

    // And the anchor being outside every edge leaves the whole thing adrift.
    let mut orphaned = PoseGraph::new(truth.clone());
    orphaned
        .push(Edge {
            from: 1,
            to: 2,
            measurement: truth[1].inverse() * truth[2],
            information: information(),
        })
        .expect("the edge names real nodes");
    assert_eq!(orphaned.shape(0).adrift, 2);
}

/// A node no edge touches is not adrift — it is simply not in the survey.
#[test]
fn a_scan_with_no_edges_is_not_counted_against_the_survey() {
    let truth = truth(5);
    let mut graph = PoseGraph::new(truth.clone());
    for from in 0..2 {
        graph
            .push(Edge {
                from,
                to: from + 1,
                measurement: truth[from].inverse() * truth[from + 1],
                information: information(),
            })
            .expect("the edge names real nodes");
    }
    let shape = graph.shape(0);
    assert_eq!(shape.joined, 3, "two of the five are in no edge");
    assert_eq!(shape.adrift, 0);
    assert_eq!(shape.closures, 0);
}
