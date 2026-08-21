//! Checking the analytical null spaces.
//!
//! Every claim in the documentation of [`SceneKind::nullspace`] is checked
//! numerically. A hand derivation without such a check is a comment, not a
//! standard: getting the sign of a cross product wrong is easy, and the
//! mistake would migrate quietly into the conditioning analysis, where
//! nothing is left to give it away.

use nalgebra::{Matrix6, Vector3, Vector6};
use rigidity_core::icp::point_to_plane_row;
use rigidity_core::lie::Se3;
use rigidity_scenes::{Scene, SceneKind, SceneParams};

/// The largest `row · v` in absolute value over the first `count` points,
/// divided by the scale of the scene.
///
/// The normalisation is needed because the row contains the moment
/// `p × n`, which grows with the size of the scene: an absolute tolerance
/// would mean different things for a corridor and for a plane.
fn violation(points: &[Vector3<f64>], normals: &[Vector3<f64>], v: &Vector6<f64>) -> f64 {
    let mut extent: f64 = 0.0;
    let mut worst: f64 = 0.0;
    for (point, normal) in points.iter().zip(normals) {
        extent = extent.max(point.norm());
        worst = worst.max(point_to_plane_row(point, normal).dot(v).abs());
    }
    worst / (1.0 + extent)
}

/// The violation on exact coordinates: what the mathematics claims.
fn exact_violation(scene: &Scene, v: &Vector6<f64>, count: usize) -> f64 {
    violation(&scene.points[..count], &scene.normals[..count], v)
}

/// The violation on the cloud: what processing sees after `f32`
/// quantisation.
fn stored_violation(scene: &Scene, v: &Vector6<f64>, count: usize) -> f64 {
    let points: Vec<Vector3<f64>> = (0..count).map(|i| scene.cloud.point(i)).collect();
    violation(&points, &scene.normals[..count], v)
}

/// `JᵀJ` over the inliers.
///
/// The matrix is assembled explicitly, which is acceptable here: what is
/// checked is a known answer with generous margin, not the detection of a
/// small eigenvalue. TSQR exists for the latter.
fn information_matrix(scene: &Scene) -> Matrix6<f64> {
    let mut h = Matrix6::zeros();
    for i in 0..scene.inlier_count {
        let row = point_to_plane_row(&scene.points[i], &scene.normals[i]);
        h += row * row.transpose();
    }
    h
}

fn sorted_eigenvalues(h: &Matrix6<f64>) -> Vec<f64> {
    let mut values: Vec<f64> = nalgebra::SymmetricEigen::new(*h)
        .eigenvalues
        .iter()
        .copied()
        .collect();
    values.sort_by(f64::total_cmp);
    values
}

fn noiseless(kind: SceneKind) -> Scene {
    Scene::generate(
        kind,
        SceneParams {
            points_per_face: 1_500,
            ..SceneParams::default()
        },
    )
}

/// The null-space dimensions, written out explicitly.
///
/// The table duplicates `SceneKind::nullspace`, deliberately: the test
/// must break if someone changes the implementation without changing the
/// claim.
#[test]
fn nullspace_dimensions_are_as_documented() {
    let expected = [
        (SceneKind::Plane, 3),
        (SceneKind::Cylinder, 2),
        (SceneKind::Sphere, 3),
        (SceneKind::TwoPlanes, 1),
        (SceneKind::Corner, 0),
        (SceneKind::TeeJoint, 1),
        (SceneKind::Corridor, 1),
    ];
    assert_eq!(expected.len(), SceneKind::ALL.len());
    for (kind, dimension) in expected {
        assert_eq!(
            kind.nullspace_dimension(),
            dimension,
            "{}: expected dimension {dimension}",
            kind.name()
        );
    }
}

#[test]
fn nullspace_basis_is_orthonormal() {
    for kind in SceneKind::ALL {
        let basis = kind.nullspace();
        for (i, a) in basis.iter().enumerate() {
            assert!(
                (a.norm() - 1.0).abs() < 1e-15,
                "{}: vector {i} is not a unit vector",
                kind.name()
            );
            for b in basis.iter().skip(i + 1) {
                assert!(
                    a.dot(b).abs() < 1e-15,
                    "{}: the basis is not orthogonal",
                    kind.name()
                );
            }
        }
    }
}

/// The declared vectors annihilate the Jacobian row at every point.
///
/// That is the definition of an unobservable direction: moving along it
/// changes no residual, so the optimisation learns nothing about it.
#[test]
fn declared_nullspace_annihilates_every_row() {
    for kind in SceneKind::ALL {
        let scene = noiseless(kind);
        for (index, v) in scene.nullspace().iter().enumerate() {
            let violation = exact_violation(&scene, v, scene.inlier_count);
            assert!(
                violation < 1e-14,
                "{}: vector {index} leaves a residual of {violation:.3e}",
                kind.name()
            );
        }
    }
}

/// The spectrum of `JᵀJ` confirms the dimension: exactly `k` zero
/// eigenvalues, with the `k+1`-th firmly away from zero.
///
/// The second half matters as much as the first. It catches the case where
/// a scene is degenerate **more** than claimed — if, say, the parameters
/// made the corridor so narrow that lateral translation stopped being
/// observed too.
#[test]
fn spectrum_confirms_nullspace_dimension() {
    for kind in SceneKind::ALL {
        let scene = noiseless(kind);
        let eigenvalues = sorted_eigenvalues(&information_matrix(&scene));
        let largest = *eigenvalues.last().unwrap();
        let dimension = kind.nullspace_dimension();

        for (i, value) in eigenvalues.iter().take(dimension).enumerate() {
            assert!(
                value / largest < 1e-12,
                "{}: eigenvalue {i} is {value:.3e} against a maximum of \
                 {largest:.3e} — the null space is smaller than claimed",
                kind.name()
            );
        }
        if dimension < 6 {
            let first_observable = eigenvalues[dimension];
            assert!(
                first_observable / largest > 1e-3,
                "{}: the smallest observable eigenvalue {first_observable:.3e} is \
                 too small against a maximum of {largest:.3e} — the scene is more \
                 degenerate than claimed",
                kind.name()
            );
        }
    }
}

/// Moving the scene maps the null space to `Adj(T)·v`.
///
/// The test ties the Lie-group code to the scene geometry: it breaks on an
/// error in either. It also pins down that the canonical answers are tied
/// to the coordinate origin rather than being absolute.
#[test]
fn nullspace_transforms_by_adjoint() {
    let motion = Se3::exp(&Vector6::new(1.7, -0.4, 2.3, 0.6, -0.35, 0.9));
    let adjoint = motion.adjoint();

    for kind in SceneKind::ALL {
        let scene = noiseless(kind);
        let rotation = *motion.rotation().matrix();

        for (index, v) in scene.nullspace().iter().enumerate() {
            let moved = adjoint * v;
            let mut extent: f64 = 0.0;
            let mut worst: f64 = 0.0;
            for i in 0..scene.inlier_count {
                let point = motion.transform_point(&scene.points[i]);
                let normal = rotation * scene.normals[i];
                extent = extent.max(point.norm());
                worst = worst.max(point_to_plane_row(&point, &normal).dot(&moved).abs());
            }
            let violation = worst / (1.0 + extent);
            assert!(
                violation < 1e-13,
                "{}: the transported vector {index} leaves a residual of {violation:.3e}",
                kind.name()
            );
        }
    }
}

/// Noise degrades degeneracy smoothly rather than abruptly.
///
/// For planar scenes the null space stays exact even under noise: it
/// consists of translations, and the normals are analytical. On the
/// cylinder and the sphere a term of order σ appears, from `ε × n`.
#[test]
fn noise_degrades_nullspace_proportionally() {
    let sigma = 1e-3;
    for kind in SceneKind::ALL {
        let scene = Scene::generate(
            kind,
            SceneParams {
                points_per_face: 1_500,
                noise_sigma: sigma,
                ..SceneParams::default()
            },
        );
        for (index, v) in scene.nullspace().iter().enumerate() {
            let violation = exact_violation(&scene, v, scene.inlier_count);
            assert!(
                violation < 12.0 * sigma,
                "{}: vector {index} at σ = {sigma:.0e} gives {violation:.3e} — \
                 the degradation is not proportional to the noise",
                kind.name()
            );
        }
    }
}

/// Outliers destroy the null space, and precisely because they bring a
/// wrong normal with them.
///
/// A point displaced along its normal does not change the Jacobian row at
/// all: `(p + δn) × n = p × n`. Degeneracy is a property of the Jacobian,
/// not of the residuals, so modelling an outlier as a displacement along
/// the normal would be pointless.
#[test]
fn outliers_with_wrong_normals_destroy_the_nullspace() {
    for kind in SceneKind::ALL {
        if kind.nullspace_dimension() == 0 {
            continue;
        }
        let scene = Scene::generate(
            kind,
            SceneParams {
                points_per_face: 1_500,
                outlier_ratio: 0.1,
                ..SceneParams::default()
            },
        );
        assert!(scene.len() > scene.inlier_count, "no outliers were added");

        let v = &scene.nullspace()[0];
        let clean = exact_violation(&scene, v, scene.inlier_count);
        let polluted = exact_violation(&scene, v, scene.len());
        assert!(clean < 1e-14, "{}: the inliers are corrupted", kind.name());
        assert!(
            polluted > 1e-2,
            "{}: the outliers did not break the null space ({polluted:.3e})",
            kind.name()
        );
    }
}

/// One seed, one scene, bit for bit.
#[test]
fn generation_is_reproducible() {
    for kind in SceneKind::ALL {
        let params = SceneParams {
            points_per_face: 300,
            noise_sigma: 1e-3,
            outlier_ratio: 0.05,
            ..SceneParams::default()
        };
        let a = Scene::generate(kind, params);
        let b = Scene::generate(kind, params);
        let (ax, ay, az) = a.cloud.columns();
        let (bx, by, bz) = b.cloud.columns();
        for (left, right) in [(ax, bx), (ay, by), (az, bz)] {
            let left: Vec<u32> = left.iter().map(|v| v.to_bits()).collect();
            let right: Vec<u32> = right.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                left,
                right,
                "{}: a repeat produced a different cloud",
                kind.name()
            );
        }

        let other = Scene::generate(
            kind,
            SceneParams {
                seed: params.seed ^ 0xFFFF,
                ..params
            },
        );
        assert_ne!(
            other.cloud.point(0),
            a.cloud.point(0),
            "{}: a different seed produced the same scene",
            kind.name()
        );
    }
}

/// The normals are unit vectors and agree with the surface.
#[test]
fn normals_are_unit_and_consistent() {
    for kind in SceneKind::ALL {
        let scene = noiseless(kind);
        assert_eq!(scene.normals.len(), scene.len());
        for i in 0..scene.inlier_count {
            let normal = scene.normals[i];
            assert!(
                (normal.norm() - 1.0).abs() < 1e-15,
                "{}: normal {i} is not a unit vector",
                kind.name()
            );
        }
        // On a sphere the normal must equal the radial direction.
        if kind == SceneKind::Sphere {
            for i in 0..scene.inlier_count {
                let radial = scene.cloud.point(i).normalize();
                assert!((radial - scene.normals[i]).norm() < 1e-6);
            }
        }
    }
}

/// Overlap: the fraction of shared points matches what was asked for.
#[test]
fn overlap_split_matches_requested_fraction() {
    let scene = noiseless(SceneKind::Corner);
    for overlap in [0.3f64, 0.5, 0.7, 1.0] {
        let (source, target) = scene.split_with_overlap(overlap, 0xABCD);
        let total = scene.len() as f64;
        let shared = (source.len() + target.len()) as f64 - total;
        let fraction = shared / total;
        assert!(
            (fraction - overlap).abs() < 0.05,
            "overlap {overlap}: got {fraction:.3}"
        );
    }
}

/// The scale of the scene does not change the answer: a null space is a
/// property of shape.
#[test]
fn nullspace_is_scale_invariant() {
    for kind in SceneKind::ALL {
        for scale in [1e-3, 1.0, 1e3] {
            let scene = Scene::generate(
                kind,
                SceneParams {
                    points_per_face: 500,
                    scale,
                    ..SceneParams::default()
                },
            );
            for v in scene.nullspace() {
                let violation = exact_violation(&scene, &v, scene.inlier_count);
                assert!(
                    violation < 1e-14,
                    "{} at scale {scale:e}: residual {violation:.3e}",
                    kind.name()
                );
            }
        }
    }
}

/// The points really do lie on the claimed surface.
#[test]
fn points_lie_on_their_surface() {
    let scene = noiseless(SceneKind::Cylinder);
    let radius = scene.params.scale * 0.5;
    for i in 0..scene.inlier_count {
        let p = scene.points[i];
        let distance = Vector3::new(p.x, p.y, 0.0).norm();
        assert!(
            (distance - radius).abs() < 1e-15,
            "point {i} is not on the cylinder"
        );
    }

    let scene = noiseless(SceneKind::Sphere);
    for i in 0..scene.inlier_count {
        assert!((scene.points[i].norm() - scene.params.scale).abs() < 1e-15);
    }
}

/// How much `f32` storage on its own violates the null space.
///
/// This is not a defect of the generator but the measured accuracy ceiling
/// of the whole pipeline. The number matters downstream: the degeneracy
/// detector's threshold must sit **above** it, or the detector measures
/// the storage error instead.
///
/// Planar scenes show no violation at all: their null spaces consist of
/// translations, and `n_y` and `n_z` are exact and independent of the
/// coordinates. On the cylinder and the sphere the identity `p × n = 0`
/// rests on the coordinates themselves, and quantisation breaks it.
#[test]
fn f32_storage_sets_the_precision_floor() {
    let curved = [SceneKind::Cylinder, SceneKind::Sphere];
    for kind in SceneKind::ALL {
        let scene = noiseless(kind);
        let mut worst_exact: f64 = 0.0;
        let mut worst_stored: f64 = 0.0;
        for v in scene.nullspace() {
            worst_exact = worst_exact.max(exact_violation(&scene, &v, scene.inlier_count));
            worst_stored = worst_stored.max(stored_violation(&scene, &v, scene.inlier_count));
        }
        assert!(
            worst_exact < 1e-14,
            "{}: the exact arithmetic is already inexact ({worst_exact:.3e})",
            kind.name()
        );

        if curved.contains(&kind) {
            // Not every direction is affected: on the cylinder, sliding
            // along the axis gives `n_z = 0` regardless of coordinates,
            // while spinning about it rests on `p × n` and so quantises.
            assert!(
                (1e-9..1e-6).contains(&worst_stored),
                "{}: the f32 floor came out at {worst_stored:.3e}, about 3e-8 was \
                 expected — if it moved, the detector thresholds need recomputing",
                kind.name()
            );
        } else {
            assert!(
                worst_stored < 1e-14,
                "{}: a planar scene must not suffer from f32, got {worst_stored:.3e}",
                kind.name()
            );
        }
    }
}
