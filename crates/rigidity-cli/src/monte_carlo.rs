//! Checking the predicted spread against the empirical one.
//!
//! # The question
//!
//! The spectrum of the normalised Jacobian yields a prediction: the spread
//! of the pose along direction `i` is `σ_noise / σ'ᵢ`. Is that true?
//!
//! The answer comes from an experiment rather than an argument: the same
//! scene is registered a thousand times with independent noise and a
//! random initial guess, and the spread of the converged poses is compared
//! with the prediction.
//!
//! # Why this gates everything else
//!
//! Until the prediction has been checked against reality, the conditioning
//! report is a set of numbers with the right units and an unknown relation
//! to the world. Measuring how fast such a report is produced would be
//! premature.

use nalgebra::{Vector3, Vector6};
use rayon::prelude::*;
use rigidity_core::PointCloud;
use rigidity_core::icp::{IcpConfig, Kernel, register, surface};
use rigidity_core::lie::Se3;
use rigidity_core::normals::estimate_normals;
use rigidity_core::observability::{
    Conditioning, Correspondence, Observability, ObservabilityCriteria, analyse,
};
use rigidity_scenes::rng::Rng;
use rigidity_scenes::{Scene, SceneKind, SceneParams};
use rigidity_spatial::KdTree;

/// Experiment parameters.
#[derive(Debug, Clone, Copy)]
pub struct TrialConfig {
    /// How many independent trials.
    pub trials: usize,
    /// Points per face of the scene.
    pub points_per_face: usize,
    /// Characteristic size of the scene, metres.
    pub scale: f64,
    /// Standard deviation of the noise on source positions, metres.
    pub noise_sigma: f64,
    /// Required pose accuracy, metres, used for the classification.
    pub tolerance: f64,
    /// Norm of the translational part of the initial perturbation, metres.
    pub initial_translation: f64,
    /// Norm of the rotational part of the initial perturbation, radians.
    pub initial_rotation: f64,
    /// Estimate normals from neighbours instead of using analytical ones.
    pub estimated_normals: bool,
    /// Seed.
    pub seed: u64,
}

impl Default for TrialConfig {
    fn default() -> Self {
        Self {
            trials: 1_000,
            points_per_face: 800,
            scale: 1.0,
            noise_sigma: 1e-3,
            tolerance: 1e-4,
            initial_translation: 0.02,
            initial_rotation: 0.01,
            estimated_normals: false,
            seed: 0x7E57_5EED,
        }
    }
}

/// The outcome for a single direction of the spectrum.
#[derive(Debug, Clone, Copy)]
pub struct DirectionOutcome {
    /// Direction index, ordered by decreasing singular value.
    pub index: usize,
    /// The predicted spread `σ_noise / σ'ᵢ`, metres.
    pub predicted: f64,
    /// The empirical spread of the converged poses, metres.
    pub empirical: f64,
    /// The systematic bias, metres.
    pub bias: f64,
    /// How the report classified this direction.
    pub observability: Observability,
}

impl DirectionOutcome {
    /// By what factor the empirical spread exceeds the predicted one.
    ///
    /// Greater than one means the prediction understates the spread, that
    /// is, the system is overconfident. That is what the CELLO-3D
    /// experience leads one to expect.
    pub fn ratio(&self) -> f64 {
        self.empirical / self.predicted
    }
}

/// The outcome for one scene.
#[derive(Debug, Clone)]
pub struct SceneOutcome {
    /// Which scene.
    pub kind: SceneKind,
    /// Whether normals were estimated or taken analytically.
    pub estimated_normals: bool,
    /// How many trials converged by the step criterion.
    pub converged: usize,
    /// Total trials.
    pub trials: usize,
    /// The scene's condition number.
    pub condition_number: f64,
    /// The six directions.
    pub directions: Vec<DirectionOutcome>,
}

/// A random unit direction.
fn random_direction(rng: &mut Rng) -> Vector3<f64> {
    loop {
        let candidate = Vector3::new(rng.normal(1.0), rng.normal(1.0), rng.normal(1.0));
        if candidate.norm() > 1e-9 {
            return candidate.normalize();
        }
    }
}

fn build_normals(
    cloud: &PointCloud,
    analytic: &[Vector3<f64>],
    estimate: bool,
) -> Vec<Vector3<f64>> {
    if estimate {
        let tree = KdTree::build(cloud).expect("the tree builds");
        estimate_normals(cloud, &tree, 16)
    } else {
        analytic.to_vec()
    }
}

/// Runs the experiment on a single scene.
pub fn run(kind: SceneKind, config: &TrialConfig) -> SceneOutcome {
    let scene = Scene::generate(
        kind,
        SceneParams {
            points_per_face: config.points_per_face,
            scale: config.scale,
            ..SceneParams::default()
        },
    );

    let target_normals = build_normals(&scene.cloud, &scene.normals, config.estimated_normals);
    let tree = KdTree::build(&scene.cloud).expect("the tree builds");

    // The prediction is built from the unperturbed target: this is what a
    // user would get by looking at the map before scanning.
    let prediction = analyse(scene.len(), Kernel::Squared, |index| {
        Some(Correspondence {
            point: scene.cloud.point(index),
            normal: target_normals[index],
            residual: 0.0,
        })
    })
    .expect("the scene is non-empty");
    let conditioning: &Conditioning = &prediction.conditioning;

    let criteria = ObservabilityCriteria {
        noise_sigma: config.noise_sigma,
        tolerance: config.tolerance,
    };
    let states = conditioning.classify(&criteria);
    let predicted = conditioning.uncertainty(config.noise_sigma);

    let truth = Se3::exp(&Vector6::new(
        0.031 * config.scale,
        -0.017 * config.scale,
        0.024 * config.scale,
        0.021,
        -0.013,
        0.018,
    ));
    let icp = IcpConfig {
        kernel: Kernel::Squared,
        max_correspondence_distance: 0.5 * config.scale,
        max_iterations: 60,
        ..IcpConfig::default()
    };

    let outcomes: Vec<Option<([f64; 6], bool)>> = (0..config.trials)
        .into_par_iter()
        .map(|trial| {
            let mut rng = Rng::new(config.seed ^ (trial as u64).wrapping_mul(0x9E37_79B9));

            // The source: the same surface with independent noise, seen
            // from a different position.
            let inverse = truth.inverse();
            let rotation = *inverse.rotation().matrix();
            let mut source = PointCloud::with_capacity(scene.len());
            for index in 0..scene.len() {
                let jitter = Vector3::new(
                    rng.normal(config.noise_sigma),
                    rng.normal(config.noise_sigma),
                    rng.normal(config.noise_sigma),
                );
                source.push(inverse.transform_point(&(scene.points[index] + jitter)));
            }
            let source_normals = if config.estimated_normals {
                let source_tree = KdTree::build(&source).ok()?;
                estimate_normals(&source, &source_tree, 16)
            } else {
                scene.normals.iter().map(|n| rotation * n).collect()
            };

            let offset = random_direction(&mut rng) * config.initial_translation * config.scale;
            let turn = random_direction(&mut rng) * config.initial_rotation;
            let start = Se3::exp(&Vector6::new(
                offset.x, offset.y, offset.z, turn.x, turn.y, turn.z,
            )) * truth;

            let result = register(
                &surface(&source, &source_normals),
                &surface(&scene.cloud, &target_normals),
                &tree,
                start,
                &icp,
            );

            // The error as a left perturbation: `T_true = exp(δ)·T_found`.
            let error = (truth * result.pose.inverse()).log();
            let mut components = [0.0f64; 6];
            for (index, slot) in components.iter_mut().enumerate() {
                *slot = conditioning.component(index, error);
            }
            if components.iter().any(|v| !v.is_finite()) {
                return None;
            }
            Some((components, result.converged))
        })
        .collect();

    let samples: Vec<[f64; 6]> = outcomes.iter().flatten().map(|(c, _)| *c).collect();
    let converged = outcomes.iter().flatten().filter(|(_, ok)| *ok).count();
    let count = samples.len().max(1) as f64;

    let directions = (0..6)
        .map(|index| {
            let mean: f64 = samples.iter().map(|c| c[index]).sum::<f64>() / count;
            let variance: f64 = samples
                .iter()
                .map(|c| (c[index] - mean) * (c[index] - mean))
                .sum::<f64>()
                / count;
            DirectionOutcome {
                index,
                predicted: predicted[index],
                empirical: variance.sqrt(),
                bias: mean,
                observability: states[index],
            }
        })
        .collect();

    SceneOutcome {
        kind,
        estimated_normals: config.estimated_normals,
        converged,
        trials: samples.len(),
        condition_number: conditioning.condition_number(),
        directions,
    }
}

/// Runs every scene.
pub fn run_all(config: &TrialConfig) -> Vec<SceneOutcome> {
    SceneKind::ALL
        .iter()
        .map(|kind| run(*kind, config))
        .collect()
}
