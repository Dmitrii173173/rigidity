//! End-to-end registration tests.
//!
//! They live here rather than in the core because they need three crates
//! at once — the core, the scenes and the spatial index — and only
//! `rigidity-cli` ties those together without creating a cycle.

use nalgebra::{Vector3, Vector6};
use rayon::ThreadPoolBuilder;
use rigidity_core::PointCloud;
use rigidity_core::icp::{IcpConfig, Kernel, register, surface};
use rigidity_core::lie::Se3;
use rigidity_core::normals::estimate_normals;
use rigidity_scenes::{Scene, SceneKind, SceneParams};
use rigidity_spatial::KdTree;

/// A scene in which all six degrees of freedom are observable.
fn corner() -> Scene {
    Scene::generate(
        SceneKind::Corner,
        SceneParams {
            points_per_face: 2_000,
            ..SceneParams::default()
        },
    )
}

/// The source is the same scene moved by the inverse transform.
///
/// The sought answer is then exactly `motion`, and registration must
/// recover it.
fn make_source(scene: &Scene, motion: &Se3) -> (PointCloud, Vec<Vector3<f64>>) {
    let inverse = motion.inverse();
    let rotation = *inverse.rotation().matrix();
    let mut cloud = PointCloud::with_capacity(scene.len());
    let mut normals = Vec::with_capacity(scene.len());
    for i in 0..scene.len() {
        cloud.push(inverse.transform_point(&scene.points[i]));
        normals.push(rotation * scene.normals[i]);
    }
    (cloud, normals)
}

/// How far the found pose is from the true one, in algebra coordinates.
fn pose_error(found: &Se3, truth: &Se3) -> (f64, f64) {
    let delta = (found.inverse() * *truth).log();
    (
        delta.fixed_rows::<3>(0).norm(),
        delta.fixed_rows::<3>(3).norm(),
    )
}

fn ground_truth() -> Se3 {
    Se3::exp(&Vector6::new(0.043, -0.021, 0.035, 0.031, -0.017, 0.026))
}

/// The full residual `e(ξ) = nᵀ(exp(ξ)·T·p − q)` agrees with its
/// analytical derivative.
///
/// Separate from the Jacobian-row check in the core: there the point sat
/// at the origin, here it comes with a real initial transform and a real
/// correspondence.
#[test]
fn full_residual_jacobian_matches_numeric() {
    let scene = corner();
    let pose = ground_truth();

    for index in [0usize, 137, 1_500, 4_321] {
        let source_point = scene.points[index];
        let target_point = scene.points[(index + 91) % scene.len()];
        let normal = scene.normals[(index + 91) % scene.len()];

        let transformed = pose.transform_point(&source_point);
        let analytic = rigidity_core::icp::point_to_plane_row(&transformed, &normal);

        const H: f64 = 1e-6;
        for axis in 0..6 {
            let mut delta = Vector6::zeros();
            delta[axis] = H;
            let forward = normal
                .dot(&((Se3::exp(&delta) * pose).transform_point(&source_point) - target_point));
            let backward = normal
                .dot(&((Se3::exp(&(-delta)) * pose).transform_point(&source_point) - target_point));
            let numeric = (forward - backward) / (2.0 * H);
            assert!(
                (analytic[axis] - numeric).abs() < 1e-7,
                "point {index}, axis {axis}: analytic {}, numeric {numeric}",
                analytic[axis]
            );
        }
    }
}

#[test]
fn converges_to_ground_truth_on_a_fully_observable_scene() {
    let scene = corner();
    let truth = ground_truth();
    let (source_cloud, source_normals) = make_source(&scene, &truth);

    let tree = KdTree::build(&scene.cloud).unwrap();
    let config = IcpConfig {
        kernel: Kernel::Squared,
        max_correspondence_distance: 0.5,
        ..IcpConfig::default()
    };

    let result = register(
        &surface(&source_cloud, &source_normals),
        &surface(&scene.cloud, &scene.normals),
        &tree,
        Se3::identity(),
        &config,
    );

    assert!(
        result.converged,
        "did not converge in {} iterations",
        result.iterations
    );
    let (translation, rotation) = pose_error(&result.pose, &truth);
    assert!(
        translation < 1e-4,
        "translation error {translation:.3e} m at a scene scale of 1 m"
    );
    assert!(rotation < 1e-4, "rotation error {rotation:.3e} rad");
    assert!(result.rmse < 1e-4, "RMSE {:.3e}", result.rmse);
    assert!(
        result.correspondences > scene.len() / 2,
        "only {} correspondences of {} survived",
        result.correspondences,
        scene.len()
    );
}

/// The result is bit-for-bit independent of the thread count.
///
/// The headline criterion for the solver. What is compared is not
/// "roughly equal" but the bits of the pose matrix: any divergence means
/// the order of summation in assembling `H` drifted somewhere.
#[test]
fn registration_is_bit_identical_across_thread_counts() {
    let scene = corner();
    let truth = ground_truth();
    let (source_cloud, source_normals) = make_source(&scene, &truth);
    let tree = KdTree::build(&scene.cloud).unwrap();
    let config = IcpConfig {
        kernel: Kernel::Huber(0.05),
        max_correspondence_distance: 0.5,
        ..IcpConfig::default()
    };

    let run = |threads: usize| {
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                register(
                    &surface(&source_cloud, &source_normals),
                    &surface(&scene.cloud, &scene.normals),
                    &tree,
                    Se3::identity(),
                    &config,
                )
            })
    };

    let reference = run(1);
    let expected: Vec<u64> = reference
        .pose
        .matrix()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    let expected_information: Vec<u64> =
        reference.information.iter().map(|v| v.to_bits()).collect();

    for threads in [2usize, 4, 8, 16] {
        let result = run(threads);
        let actual: Vec<u64> = result.pose.matrix().iter().map(|v| v.to_bits()).collect();
        assert_eq!(
            actual, expected,
            "{threads} threads: the pose differs bit for bit"
        );
        let information: Vec<u64> = result.information.iter().map(|v| v.to_bits()).collect();
        assert_eq!(
            information, expected_information,
            "{threads} threads: the matrix H differs bit for bit"
        );
        assert_eq!(result.iterations, reference.iterations);
        assert_eq!(result.correspondences, reference.correspondences);
    }
}

/// A source in which a third of the points are systematically displaced
/// along their normals.
///
/// A reflective surface looks like this, for instance. The displacement is
/// one-sided: a symmetric one would partly cancel itself and check
/// nothing.
fn biased_source(scene: &Scene, truth: &Se3, bias: f64) -> (PointCloud, Vec<Vector3<f64>>) {
    let (clean, normals) = make_source(scene, truth);
    let mut source = PointCloud::with_capacity(clean.len());
    for (index, normal) in normals.iter().enumerate() {
        let point = clean.point(index);
        source.push(if index % 3 == 0 {
            point + normal * bias
        } else {
            point
        });
    }
    (source, normals)
}

/// Robust kernels withstand a systematic outlier; the squared loss does
/// not.
///
/// The start is taken near the truth. That is not a concession: what is
/// checked here is the **robustness of the estimate**, while the size of
/// the basin of attraction is a separate property with its own test
/// below. Mixing them in one test yields a result whose meaning is
/// unclear.
#[test]
fn robust_kernels_resist_a_systematic_outlier_bias() {
    let scene = corner();
    let truth = ground_truth();
    let (source, normals) = biased_source(&scene, &truth, 0.15);
    let tree = KdTree::build(&scene.cloud).unwrap();

    let warm_start = Se3::exp(&Vector6::new(0.006, -0.004, 0.005, 0.003, -0.002, 0.004)) * truth;
    let run = |kernel: Kernel| {
        let result = register(
            &surface(&source, &normals),
            &surface(&scene.cloud, &scene.normals),
            &tree,
            warm_start,
            &IcpConfig {
                kernel,
                max_correspondence_distance: 0.5,
                ..IcpConfig::default()
            },
        );
        pose_error(&result.pose, &truth).0
    };

    let squared = run(Kernel::Squared);
    assert!(
        squared > 0.02,
        "the squared loss gave {squared:.4} — the displacement is too gentle \
         and the test checks nothing"
    );

    for kernel in [
        Kernel::Huber(0.02),
        Kernel::Cauchy(0.02),
        Kernel::Tukey(0.05),
        Kernel::GemanMcClure(0.02),
    ] {
        let robust = run(kernel);
        assert!(
            robust < 0.02,
            "{kernel:?}: error {robust:.4} m — robustness did not work"
        );
        assert!(
            squared > robust * 3.0,
            "{kernel:?}: squared {squared:.4} against robust {robust:.4}"
        );
    }
}

/// Basin of attraction: a convex kernel recovers from a cold start, a
/// redescending one does not.
///
/// Tukey zeroes the weight past its threshold. If the initial guess is
/// such that the residuals of the **inliers** already exceed that
/// threshold, the kernel cannot tell them from outliers and converges to
/// the wrong minimum — here exactly onto the displaced population, with an
/// error equal to the displacement.
///
/// The cure is not tuning the threshold but warming up with a convex
/// kernel, the graduated non-convexity trick. The test pins down all three
/// outcomes: the property is real, and it is worth knowing about before
/// relying on the estimator.
#[test]
fn redescending_kernels_need_a_warm_start() {
    let scene = corner();
    let truth = ground_truth();
    let bias = 0.15;
    let (source, normals) = biased_source(&scene, &truth, bias);
    let tree = KdTree::build(&scene.cloud).unwrap();

    let run = |kernel: Kernel, initial: Se3| {
        register(
            &surface(&source, &normals),
            &surface(&scene.cloud, &scene.normals),
            &tree,
            initial,
            &IcpConfig {
                kernel,
                max_correspondence_distance: 0.5,
                ..IcpConfig::default()
            },
        )
    };

    // Huber is convex: it creates no local minima.
    let huber = run(Kernel::Huber(0.02), Se3::identity());
    let (huber_error, _) = pose_error(&huber.pose, &truth);
    assert!(
        huber_error < 0.02,
        "Huber from a cold start gave {huber_error:.4} m"
    );

    // Tukey, from the same point, drifts to the outliers.
    let cold_tukey = run(Kernel::Tukey(0.05), Se3::identity());
    let (cold_error, _) = pose_error(&cold_tukey.pose, &truth);
    assert!(
        cold_error > bias * 0.5,
        "Tukey from a cold start gave {cold_error:.4} m — if it now copes, \
         the property has changed and the comment above is stale"
    );

    // The same kernel, warmed up by Huber's solution, finds the right
    // answer.
    let warm_tukey = run(Kernel::Tukey(0.05), huber.pose);
    let (warm_error, _) = pose_error(&warm_tukey.pose, &truth);
    assert!(
        warm_error < 0.02,
        "Tukey after the warm start gave {warm_error:.4} m"
    );
    assert!(
        warm_error < cold_error * 0.2,
        "the warm start did not help: was {cold_error:.4}, became {warm_error:.4}"
    );
}

/// Normals estimated by PCA agree with the analytical ones.
#[test]
fn pca_normals_match_analytic_normals() {
    for (kind, tolerance) in [
        (SceneKind::Plane, 1e-6),
        (SceneKind::Cylinder, 2e-2),
        (SceneKind::Sphere, 2e-2),
    ] {
        let scene = Scene::generate(
            kind,
            SceneParams {
                points_per_face: 4_000,
                ..SceneParams::default()
            },
        );
        let tree = KdTree::build(&scene.cloud).unwrap();
        let estimated = estimate_normals(&scene.cloud, &tree, 16);

        let mut worst: f64 = 0.0;
        for (estimate, analytic) in estimated
            .iter()
            .zip(&scene.normals)
            .take(scene.inlier_count)
        {
            // The estimate's sign is arbitrary, so compare the absolute
            // cosine.
            worst = worst.max(1.0 - estimate.dot(analytic).abs());
        }
        assert!(
            worst < tolerance,
            "{}: worst normal deviation 1 − |cos| = {worst:.3e}",
            kind.name()
        );
    }
}

/// Registration works on estimated normals too, not only on analytical
/// ones.
#[test]
fn converges_with_estimated_normals() {
    let scene = corner();
    let truth = ground_truth();
    let (source_cloud, _) = make_source(&scene, &truth);

    let target_tree = KdTree::build(&scene.cloud).unwrap();
    let source_tree = KdTree::build(&source_cloud).unwrap();
    let target_normals = estimate_normals(&scene.cloud, &target_tree, 16);
    let source_normals = estimate_normals(&source_cloud, &source_tree, 16);

    let result = register(
        &surface(&source_cloud, &source_normals),
        &surface(&scene.cloud, &target_normals),
        &target_tree,
        Se3::identity(),
        &IcpConfig {
            kernel: Kernel::Huber(0.02),
            max_correspondence_distance: 0.5,
            ..IcpConfig::default()
        },
    );

    let (translation, rotation) = pose_error(&result.pose, &truth);
    assert!(translation < 5e-3, "translation error {translation:.3e} m");
    assert!(rotation < 5e-3, "rotation error {rotation:.3e} rad");
}
