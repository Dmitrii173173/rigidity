//! A reduced version of the Monte Carlo experiment.
//!
//! The full thousand-trial run lives in the `monte_carlo` example and
//! produces the repository's figure. This is the fast version, pinning
//! down the conclusions so they cannot quietly fall apart.

use rigidity_cli::monte_carlo::{TrialConfig, run};
use rigidity_core::observability::Observability;
use rigidity_scenes::SceneKind;

fn config() -> TrialConfig {
    TrialConfig {
        trials: 200,
        points_per_face: 250,
        ..TrialConfig::default()
    }
}

/// Where the detector promises reliability, the predicted spread matches
/// the empirical one.
///
/// The headline result. The tolerance comes from statistics: an estimate
/// of a standard deviation from `n` samples carries its own error of about
/// `1/√(2n)`, that is 5 % at two hundred trials. The bounds 0.75…1.3 are
/// roughly four such errors.
#[test]
fn prediction_matches_reality_where_it_claims_confidence() {
    let config = config();
    let mut checked = 0usize;
    for kind in SceneKind::ALL {
        let outcome = run(kind, &config);
        for direction in &outcome.directions {
            if direction.observability == Observability::Low {
                continue;
            }
            let ratio = direction.ratio();
            assert!(
                (0.75..1.3).contains(&ratio),
                "{} direction {}: predicted {:.3e}, got {:.3e}, ratio \
                 {ratio:.3}",
                kind.name(),
                direction.index,
                direction.predicted,
                direction.empirical
            );
            checked += 1;
        }
    }
    assert!(checked >= 25, "only {checked} directions were checked");
}

/// The detector issues no false confidence.
///
/// The most important property in practice. Calling a good degree of
/// freedom bad is harmless: the system merely asks for more data. The
/// opposite mistake means driving confidently in the wrong direction.
#[test]
fn nothing_marked_reliable_is_actually_broken() {
    let config = config();
    for kind in SceneKind::ALL {
        let outcome = run(kind, &config);
        for direction in &outcome.directions {
            if direction.observability != Observability::High {
                continue;
            }
            assert!(
                direction.empirical < 3.0 * config.tolerance,
                "{} direction {}: marked HIGH, yet the spread is {:.3e} m \
                 against a tolerance of {:.3e} m",
                kind.name(),
                direction.index,
                direction.empirical,
                config.tolerance
            );
            assert!(
                direction.bias.abs() < direction.empirical,
                "{} direction {}: the bias {:.3e} exceeds the spread {:.3e}",
                kind.name(),
                direction.index,
                direction.bias,
                direction.empirical
            );
        }
    }
}

/// Along unobservable directions registration does not move the pose.
///
/// The error there stays equal to the initial perturbation: ICP receives
/// no information about them and honestly does nothing with them. That is
/// exactly the quantity ordinary ICP says nothing about.
#[test]
fn unobservable_directions_keep_their_initial_error() {
    let config = config();
    // The component of a random perturbation along one axis out of six.
    let expected = config.initial_translation * config.scale / 3f64.sqrt();

    for kind in [SceneKind::Plane, SceneKind::TwoPlanes, SceneKind::Corridor] {
        let outcome = run(kind, &config);
        let untouched: Vec<_> = outcome
            .directions
            .iter()
            .filter(|d| !d.predicted.is_finite())
            .collect();
        assert!(
            !untouched.is_empty(),
            "{}: no directions with an infinite prediction were found",
            kind.name()
        );
        for direction in untouched {
            assert!(
                direction.empirical > 0.2 * expected,
                "{} direction {}: a spread of {:.3e} m is small — so ICP did \
                 move the pose along a supposedly unobservable direction",
                kind.name(),
                direction.index,
                direction.empirical
            );
        }
    }
}

/// A fully observable scene loses no degree of freedom, and its accuracy
/// improves with the point count as `1/√N`.
///
/// The first half is about geometry: a trihedral corner has no
/// unobservable direction at any point count. The second is about
/// statistics: whether a direction reads HIGH or MEDIUM depends not only
/// on shape but on how much data there is, and that is the content of the
/// classification rather than a defect of it. The spread falls as the
/// square root of the number of measurements, and the report must show
/// that.
#[test]
fn a_fully_observable_scene_tightens_as_data_accumulates() {
    let sparse = run(
        SceneKind::Corner,
        &TrialConfig {
            points_per_face: 250,
            ..config()
        },
    );
    let dense = run(
        SceneKind::Corner,
        &TrialConfig {
            points_per_face: 1_500,
            ..config()
        },
    );

    for outcome in [&sparse, &dense] {
        for direction in &outcome.directions {
            assert_ne!(
                direction.observability,
                Observability::Low,
                "the trihedral corner lost direction {}",
                direction.index
            );
        }
        assert!(
            outcome.condition_number.is_finite(),
            "the condition number {} is not finite",
            outcome.condition_number
        );
    }

    // At six times the data, all six directions are reliable.
    for direction in &dense.directions {
        assert_eq!(
            direction.observability,
            Observability::High,
            "dense scene: direction {} is marked {}",
            direction.index,
            direction.observability.label()
        );
    }

    // The spread fell as the square root of the point-count ratio.
    let expected = (1_500f64 / 250.0).sqrt();
    for index in 0..6 {
        let ratio = sparse.directions[index].empirical / dense.directions[index].empirical;
        assert!(
            (ratio / expected - 1.0).abs() < 0.25,
            "direction {index}: the spread fell by {ratio:.2}×, while the \
             1/√N law expects {expected:.2}"
        );
    }

    assert!(
        dense.converged * 10 >= dense.trials * 9,
        "only {} of {} converged",
        dense.converged,
        dense.trials
    );
}
