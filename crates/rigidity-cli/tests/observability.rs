//! The degeneracy detector against the analytical answers of the scenes.

use nalgebra::Vector6;
use rigidity_core::icp::Kernel;
use rigidity_core::linalg::singular_values;
use rigidity_core::observability::{Correspondence, Observability, ObservabilityCriteria, analyse};
use rigidity_scenes::{Scene, SceneKind, SceneParams};

/// Welding-cell requirements: 1 mm of sensor noise, 0.1 mm of required
/// accuracy.
fn criteria(scale: f64) -> ObservabilityCriteria {
    ObservabilityCriteria {
        noise_sigma: 1e-3 * scale,
        tolerance: 1e-4 * scale,
    }
}

fn scene_of(kind: SceneKind, scale: f64) -> Scene {
    Scene::generate(
        kind,
        SceneParams {
            points_per_face: 1_500,
            scale,
            ..SceneParams::default()
        },
    )
}

/// Correspondences are built from the **cloud**, not from exact
/// coordinates: the whole path is checked, `f32` quantisation included.
fn analyse_scene(scene: &Scene) -> rigidity_core::observability::Analysis {
    analyse(scene.inlier_count, Kernel::Squared, |index| {
        Some(Correspondence {
            point: scene.cloud.point(index),
            normal: scene.normals[index],
            residual: 0.0,
        })
    })
    .expect("the scene is non-empty")
}

fn orthonormalise(vectors: &[Vector6<f64>]) -> Vec<Vector6<f64>> {
    let mut basis: Vec<Vector6<f64>> = Vec::new();
    for vector in vectors {
        let mut residue = *vector;
        for existing in &basis {
            residue -= existing * existing.dot(&residue);
        }
        let norm = residue.norm();
        if norm > 1e-12 {
            basis.push(residue / norm);
        }
    }
    basis
}

/// The largest principal angle between two subspaces, in degrees.
///
/// Bases cannot be compared element by element: one subspace has
/// infinitely many of them. The largest principal angle is the one
/// quantity that says "this is the same subspace" regardless of which
/// basis was chosen.
fn largest_principal_angle(left: &[Vector6<f64>], right: &[Vector6<f64>]) -> f64 {
    if left.len() != right.len() {
        return 180.0;
    }
    let (left, right) = (orthonormalise(left), orthonormalise(right));
    if left.len() != right.len() {
        return 180.0;
    }
    if left.is_empty() {
        return 0.0;
    }

    let rank = left.len();
    let mut product = [[0.0f64; 6]; 6];
    for (i, a) in left.iter().enumerate() {
        for (j, b) in right.iter().enumerate() {
            product[i][j] = a.dot(b);
        }
    }
    // Padding with an identity block: its singular values are one and do
    // not affect the minimum.
    for (offset, row) in product.iter_mut().enumerate().skip(rank) {
        row[offset] = 1.0;
    }
    singular_values(&product)[5]
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
}

/// The detected null space matches the analytical one on every scene.
///
/// The headline criterion. Both the dimension and the subspace itself are
/// checked.
#[test]
fn detected_nullspace_matches_the_analytic_answer() {
    for kind in SceneKind::ALL {
        let scene = scene_of(kind, 1.0);
        let analysis = analyse_scene(&scene);
        let found = analysis
            .conditioning
            .unobservable_directions(&criteria(1.0));

        assert_eq!(
            found.len(),
            kind.nullspace_dimension(),
            "{}: found {} unobservable directions instead of {}",
            kind.name(),
            found.len(),
            kind.nullspace_dimension()
        );

        let angle = largest_principal_angle(&found, &scene.nullspace());
        assert!(
            angle < 1.0,
            "{}: the subspaces differ by {angle:.3}°",
            kind.name()
        );
    }
}

/// Changing the units does not change the verdict.
///
/// The same scene in metres and in millimetres. The classification of the
/// degrees of freedom must agree on every scene. The condition number is
/// compared only where geometry determines it: on a degenerate scene
/// `σ_min` sits at the `f32` storage floor, that is, it is rounding noise,
/// and its exact value legitimately depends on scale.
#[test]
fn changing_units_does_not_change_the_verdict() {
    for kind in SceneKind::ALL {
        let metres = analyse_scene(&scene_of(kind, 1.0));
        let millimetres = analyse_scene(&scene_of(kind, 1_000.0));

        assert_eq!(
            metres.conditioning.classify(&criteria(1.0)),
            millimetres.conditioning.classify(&criteria(1_000.0)),
            "{}: the classification of the degrees of freedom changed with the units",
            kind.name()
        );

        let a = metres.conditioning.condition_number();
        let b = millimetres.conditioning.condition_number();

        if kind.nullspace_dimension() == 0 {
            let relative = (a - b).abs() / a.max(b);
            assert!(
                relative < 1e-6,
                "{}: condition number {a:.4e} against {b:.4e}",
                kind.name()
            );
        } else {
            // A degenerate scene: conditioning runs into storage. Planar
            // scenes give exactly zero — what is unobservable there are
            // translations, and the relevant normal components are exact.
            // Curved ones give the measured f32 floor of about 3·10⁻⁸,
            // that is, κ ≈ 10⁷.
            for value in [a, b] {
                assert!(
                    value > 1e6,
                    "{}: κ = {value:.3e} — the scene stopped being degenerate",
                    kind.name()
                );
            }
        }
    }
}

/// Without the normalisation the verdict does depend on the units.
///
/// A test with teeth: it shows that normalising by the radius of gyration
/// is not decoration. An unnormalised Jacobian has columns in different
/// units, and going from metres to millimetres grows the rotational block
/// a thousandfold together with the condition number.
#[test]
fn without_normalisation_units_change_the_answer() {
    let raw_condition = |scale: f64| {
        let scene = scene_of(SceneKind::Corner, scale);
        let mut hessian = [[0.0f64; 6]; 6];
        for index in 0..scene.inlier_count {
            let row = rigidity_core::icp::point_to_plane_row(
                &scene.cloud.point(index),
                &scene.normals[index],
            );
            for i in 0..6 {
                for j in 0..6 {
                    hessian[i][j] += row[i] * row[j];
                }
            }
        }
        let values = singular_values(&hessian);
        values[0] / values[5]
    };

    let metres = raw_condition(1.0);
    let millimetres = raw_condition(1_000.0);
    let ratio = (millimetres / metres).max(metres / millimetres);
    assert!(
        ratio > 100.0,
        "the unnormalised condition number changed by only {ratio:.1}× — \
         if it became stable on its own, the normalisation is no longer needed"
    );

    // The normalised one, meanwhile, did not change.
    let normalised = |scale: f64| {
        analyse_scene(&scene_of(SceneKind::Corner, scale))
            .conditioning
            .condition_number()
    };
    let stable = (normalised(1_000.0) / normalised(1.0) - 1.0).abs();
    assert!(stable < 1e-6, "the normalised one changed by {stable:.3e}");
}

/// The report flags degenerate scenes and leaves complete ones alone.
#[test]
fn the_report_names_the_lost_degrees_of_freedom() {
    let corner = analyse_scene(&scene_of(SceneKind::Corner, 1.0));
    let text = corner.describe(&criteria(1.0));
    assert!(
        !text.contains("LOW"),
        "the trihedral corner was flagged as degenerate:\n{text}"
    );
    assert!(text.contains("HIGH"));

    let corridor = analyse_scene(&scene_of(SceneKind::Corridor, 1.0));
    let text = corridor.describe(&criteria(1.0));
    assert!(
        text.contains("LOW"),
        "the corridor was not flagged:\n{text}"
    );
    assert!(text.contains("warning"));
}

/// Richer geometry loses fewer degrees of freedom.
///
/// The condition numbers themselves cannot be compared here: for a plane
/// and for two planes `σ_min` is **exactly** zero and both give infinity.
/// That is not a defect but the correct answer; what distinguishes them is
/// the dimension of the lost subspace.
#[test]
fn richer_geometry_loses_fewer_degrees_of_freedom() {
    let lost = |kind| {
        analyse_scene(&scene_of(kind, 1.0))
            .conditioning
            .classify(&criteria(1.0))
            .iter()
            .filter(|state| **state == Observability::Low)
            .count()
    };

    assert_eq!(lost(SceneKind::Corner), 0);
    assert_eq!(lost(SceneKind::TwoPlanes), 1);
    assert_eq!(lost(SceneKind::Plane), 3);

    // A fully observable scene is well conditioned, not merely finite.
    let corner = analyse_scene(&scene_of(SceneKind::Corner, 1.0))
        .conditioning
        .condition_number();
    assert!(
        corner.is_finite() && corner < 1e3,
        "the trihedral corner gave κ = {corner:.3e}"
    );
}

/// The sandwich estimate diverges from the naive one under outliers.
#[test]
fn sandwich_differs_from_the_naive_estimate_under_outliers() {
    let scene = scene_of(SceneKind::Corner, 1.0);
    let bias = 0.05;
    let kernel = Kernel::Huber(0.005);

    let analysis = analyse(scene.inlier_count, kernel, |index| {
        Some(Correspondence {
            point: scene.cloud.point(index),
            normal: scene.normals[index],
            // Every fifth point is a badly mismatched correspondence.
            residual: if index % 5 == 0 { bias } else { 0.0 },
        })
    })
    .unwrap();

    let sandwich = analysis.sandwich.expect("the corner is not degenerate");
    let naive = analysis.naive.expect("the corner is not degenerate");

    let sandwich_scale = sandwich.diagonal().iter().sum::<f64>().sqrt();
    let naive_scale = naive.diagonal().iter().sum::<f64>().sqrt();
    let ratio = naive_scale / sandwich_scale;

    assert!(
        ratio > 2.0,
        "the naive estimate {naive_scale:.3e} and the sandwich \
         {sandwich_scale:.3e} differ by only {ratio:.2}× — under outliers the \
         difference must be visible, or the sandwich is pointless"
    );
}

/// Directions and magnitudes are reproducible at any thread count.
#[test]
fn analysis_is_bit_identical_across_thread_counts() {
    let scene = scene_of(SceneKind::Corridor, 1.0);
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| analyse_scene(&scene).conditioning.singular_values())
    };

    let reference: Vec<u64> = run(1).iter().map(|v| v.to_bits()).collect();
    for threads in [2usize, 4, 8] {
        let actual: Vec<u64> = run(threads).iter().map(|v| v.to_bits()).collect();
        assert_eq!(
            actual, reference,
            "{threads} threads: the spectrum differs bit for bit"
        );
    }
}

/// The corridor report: the very output this project exists for.
#[test]
fn corridor_report_is_readable() {
    let scene = scene_of(SceneKind::Corridor, 1.0);
    let analysis = analyse_scene(&scene);
    let text = analysis.describe(&criteria(1.0));
    println!("\n{text}");

    let states = analysis.conditioning.classify(&criteria(1.0));
    assert_eq!(
        states.iter().filter(|s| **s == Observability::Low).count(),
        1
    );
}
