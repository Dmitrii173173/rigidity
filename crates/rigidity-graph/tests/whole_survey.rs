//! The gate for S4: the weak leg is findable without being pointed at.
//!
//! A survey of a dozen stations is a dozen registrations, each of which came
//! back with a pose and a residual and looked fine. The question this view
//! answers is which of them to go back and do again, and the answer is not
//! the one with the largest residual — an edge that disagrees by half a
//! metre and costs nothing is an edge whose weight along that direction was
//! removed on purpose, and the survey settled by another path.
//!
//! It is the *pair* that identifies the weak leg: a large disagreement that
//! costs nothing. That is the shape of "this registration could not see
//! along here", and no single number says it.

use rigidity_core::lie::{Se3, So3};
use rigidity_core::nalgebra::{Matrix6, Vector3, Vector6};
use rigidity_graph::{Edge, OptimiseParams, PoseGraph};

/// Which leg of the survey is the corridor.
const WEAK: usize = 3;
/// How many stations.
const STATIONS: usize = 8;

/// Stations around a rectangle, all facing the same way.
///
/// No turning between them, which is what a tripod survey looks like and
/// what makes every station's body frame the world frame. Without that, a
/// direction reported for one node would have to be carried into another's
/// frame before the two could be compared, and this test would be about
/// that arithmetic instead of about the view.
fn truth() -> Vec<Se3> {
    let corners = [
        (0.0, 0.0),
        (15.0, 0.0),
        (30.0, 0.0),
        (45.0, 0.0),
        (45.0, 25.0),
        (30.0, 25.0),
        (15.0, 25.0),
        (0.0, 25.0),
    ];
    corners
        .iter()
        .map(|(x, y)| Se3::from_parts(So3::identity(), Vector3::new(*x, *y, 0.0)))
        .collect()
}

/// A well-conditioned edge: everything determined, evenly.
fn solid() -> Matrix6<f64> {
    Matrix6::identity() * 1e6
}

/// A corridor running along `x`: every direction but that one.
///
/// Zero and not merely small, which is what `calibrated_information` produces
/// for a direction whose predicted spread exceeds the tolerance. The whole
/// question of this view is what such an edge looks like from above.
fn corridor() -> Matrix6<f64> {
    let mut out = solid();
    out[(0, 0)] = 0.0;
    out
}

/// The survey, walked as a closed loop with a systematic bias on every leg.
fn survey() -> PoseGraph {
    let truth = truth();
    let bias = Se3::exp(&Vector6::new(0.01, 0.004, 0.0, 0.0, 0.0, 0.0));
    let mut graph = PoseGraph::new(truth.clone());
    for from in 0..STATIONS {
        let to = (from + 1) % STATIONS;
        graph
            .push(Edge {
                from,
                to,
                measurement: bias * (truth[from].inverse() * truth[to]),
                information: if from == WEAK { corridor() } else { solid() },
            })
            .expect("the edge names real nodes");
    }
    graph
}

/// The gate. One row stands out, and it is the right one.
#[test]
fn the_weak_leg_is_the_one_the_view_puts_first() {
    let mut graph = survey();
    graph.optimise(&OptimiseParams::default()).expect("solves");
    let diagnosis = graph.diagnose(0).expect("diagnoses");

    for report in &diagnosis.edges {
        println!(
            "edge {} to {}:  apart {:.4} m   cost {:.3e}   stiffness {:.3e}",
            graph.edges()[report.edge].from,
            graph.edges()[report.edge].to,
            report.translation,
            report.cost,
            report.cost / report.translation.powi(2),
        );
    }

    // The column a person reads first, and it is enough on its own.
    let mut by_gap = diagnosis.edges.clone();
    by_gap.sort_by(|a, b| b.translation.total_cmp(&a.translation));
    assert_eq!(
        by_gap[0].edge, WEAK,
        "the widest gap is on edge {}, not the corridor",
        by_gap[0].edge
    );
    assert!(
        by_gap[0].translation > 100.0 * by_gap[1].translation,
        "the corridor's gap is {:.4} m and the next is {:.4} m — not a margin \
         anybody would notice",
        by_gap[0].translation,
        by_gap[1].translation
    );

    // And the second column says *why*, which is the half a residual on its
    // own never says. The corridor is not the cheapest edge — every edge
    // here costs about the same — but it is the only one paying that little
    // for a gap that size. Cost over gap squared is how hard an edge pulls
    // per metre of disagreement, and the corridor's is four orders below
    // everybody else's because its weight along that direction was removed.
    let stiffness = |report: &rigidity_graph::EdgeReport| {
        if report.translation > 0.0 {
            report.cost / report.translation.powi(2)
        } else {
            f64::INFINITY
        }
    };
    let mut by_stiffness = diagnosis.edges.clone();
    by_stiffness.sort_by(|a, b| stiffness(a).total_cmp(&stiffness(b)));
    assert_eq!(
        by_stiffness[0].edge, WEAK,
        "the corridor is not the slackest edge, so the pair does not identify it"
    );
    assert!(
        stiffness(&by_stiffness[0]) < 1e-4 * stiffness(&by_stiffness[1]),
        "the corridor pulls at {:.3e} per square metre and the next slackest \
         at {:.3e} — not a distinction",
        stiffness(&by_stiffness[0]),
        stiffness(&by_stiffness[1]),
    );
}

/// What the weak leg costs, which depends entirely on whether it is
/// load-bearing.
///
/// The first guess at this test was that the stations would be least sure of
/// themselves along the corridor. They are not, and it was worth finding
/// out why. The largest spread in a planar survey is out of plane, put there
/// by a milliradian of rotational uncertainty acting through a thirty-metre
/// lever arm, and the corridor has nothing to do with it.
///
/// In a *closed* loop the corridor costs almost nothing at all — one per
/// cent of the variance along its own axis — because seven other edges
/// still say where those stations are. That is not a disappointing result;
/// it is S3's whole argument arriving from the other direction. The place a
/// weak leg is expensive is a chain, where it is the only thing holding one
/// half of the survey to the other, and there it costs everything: nothing
/// beyond it is determined along that axis at all.
///
/// A view that showed only one of the two would be misleading in whichever
/// case it left out.
#[test]
fn a_weak_leg_costs_nothing_in_a_loop_and_everything_in_a_chain() {
    let build = |closed: bool| {
        let truth = truth();
        let bias = Se3::exp(&Vector6::new(0.01, 0.004, 0.0, 0.0, 0.0, 0.0));
        let mut graph = PoseGraph::new(truth.clone());
        let legs = if closed { STATIONS } else { STATIONS - 1 };
        for from in 0..legs {
            let to = (from + 1) % STATIONS;
            graph
                .push(Edge {
                    from,
                    to,
                    measurement: bias * (truth[from].inverse() * truth[to]),
                    information: if from == WEAK { corridor() } else { solid() },
                })
                .expect("the edge names real nodes");
        }
        graph.optimise(&OptimiseParams::default()).expect("solves");
        graph.diagnose(0).expect("diagnoses")
    };

    let spread_along_x =
        |report: &rigidity_graph::NodeReport| report.position_covariance()[(0, 0)].sqrt();

    let chain = build(false);
    for report in &chain.nodes {
        println!(
            "chain, station {}:  along x ±{:.5} m   worst ±{:.5} m",
            report.node,
            spread_along_x(report),
            report.position().0
        );
    }
    // Before the corridor: held by the chain. After it: held by nothing.
    for report in &chain.nodes {
        let beyond = report.node > WEAK;
        assert_eq!(
            spread_along_x(report).is_infinite(),
            beyond,
            "station {} is {}the corridor and its spread along x is {}",
            report.node,
            if beyond { "beyond " } else { "before " },
            spread_along_x(report)
        );
    }

    let loop_ = build(true);
    for report in &loop_.nodes {
        assert!(
            spread_along_x(report).is_finite(),
            "closing the loop should determine station {} along x",
            report.node
        );
    }
    println!(
        "loop, worst station along x: ±{:.5} m",
        loop_
            .nodes
            .iter()
            .map(spread_along_x)
            .fold(0.0f64, f64::max)
    );
}

/// A station nothing joins to the anchor is reported as unconstrained, not
/// as a number.
///
/// The alternative is a plausible finite spread produced by damping, which
/// is the one answer this view must never give: it would say a station is
/// known to a centimetre when in truth nothing in the survey knows where it
/// is at all.
#[test]
fn an_unconstrained_station_says_so() {
    let truth = truth();
    let mut graph = PoseGraph::new(truth.clone());
    // 0-1-2 joined; 3 onwards left alone.
    for from in 0..2 {
        graph
            .push(Edge {
                from,
                to: from + 1,
                measurement: truth[from].inverse() * truth[from + 1],
                information: solid(),
            })
            .expect("the edge names real nodes");
    }
    let diagnosis = graph.diagnose(0).expect("diagnoses");
    let spread = |node: usize| {
        diagnosis
            .nodes
            .iter()
            .find(|report| report.node == node)
            .expect("every free node has a row")
            .position()
            .0
    };
    assert!(spread(1).is_finite(), "station 1 is joined to the anchor");
    assert!(spread(2).is_finite(), "station 2 is joined through 1");
    for node in 3..STATIONS {
        assert!(
            spread(node).is_infinite(),
            "station {node} is joined to nothing, and its spread came back \
             as {} m",
            spread(node)
        );
    }
}
