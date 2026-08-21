//! The point-to-plane ICP solver: Levenberg–Marquardt over IRLS.

use nalgebra::{Matrix6, Vector3, Vector6};
use rayon::prelude::*;

use crate::cloud::PointCloud;
use crate::icp::kernel::Kernel;
use crate::icp::residual::point_to_plane_row;
use crate::lie::Se3;
use crate::neighbors::{Neighbor, NeighborSearch};

/// Block size of the deterministic reduction.
///
/// Block boundaries follow from this constant and the point count alone —
/// not from the thread count, and not from how the scheduler happened to
/// divide the work. Within a block, summation runs in increasing index
/// order; blocks are added in order. The reduction tree is fixed, so `H`
/// is bit-for-bit the same at any thread count.
///
/// `rayon::reduce` does not give this: its combination order depends on
/// how the worker threads split the range.
const REDUCTION_CHUNK: usize = 4_096;

/// A surface: a cloud together with its normals.
#[derive(Debug, Clone, Copy)]
pub struct Surface<'a> {
    /// The points.
    pub cloud: &'a PointCloud,
    /// The normal at each point; its length equals the point count.
    pub normals: &'a [Vector3<f64>],
}

/// Registration settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IcpConfig {
    /// Cap on the number of iterations.
    pub max_iterations: usize,
    /// Correspondences farther apart than this are discarded.
    pub max_correspondence_distance: f64,
    /// Minimum `|cos|` of the angle between normals.
    ///
    /// The absolute value, not the cosine itself: the sign a PCA estimate
    /// assigns to a normal is arbitrary and carries no meaning.
    pub min_normal_cosine: f64,
    /// The loss function.
    pub kernel: Kernel,
    /// Convergence threshold on the translational part of the step, metres.
    pub translation_tolerance: f64,
    /// Convergence threshold on the rotational part of the step, radians.
    pub rotation_tolerance: f64,
    /// Initial Levenberg–Marquardt damping.
    pub initial_damping: f64,
}

impl Default for IcpConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            max_correspondence_distance: 1.0,
            min_normal_cosine: 0.8,
            kernel: Kernel::Huber(0.1),
            translation_tolerance: 1e-8,
            rotation_tolerance: 1e-8,
            initial_damping: 1e-4,
        }
    }
}

/// Solver state after an accepted iteration.
#[derive(Debug, Clone, Copy)]
pub struct IterationReport {
    /// Iteration number, counting from one.
    pub iteration: usize,
    /// The pose after the iteration.
    pub pose: Se3,
    /// Root-mean-square residual.
    pub rmse: f64,
    /// How many correspondences survived rejection.
    pub correspondences: usize,
}

/// The result of a registration.
#[derive(Debug, Clone)]
pub struct IcpResult {
    /// The transform found: it carries the source into the target frame.
    pub pose: Se3,
    /// How many iterations ran.
    pub iterations: usize,
    /// Whether the step fell below the thresholds.
    pub converged: bool,
    /// Root-mean-square residual over the accepted correspondences.
    pub rmse: f64,
    /// How many correspondences survived rejection.
    pub correspondences: usize,
    /// The matrix `H = JᵀWJ` at the final pose.
    ///
    /// Keeping it in the result is a deliberate choice rather than a
    /// convenience: this matrix is what the whole project is about. It is
    /// what the TSQR path is compared against, and what the conditioning
    /// analysis is derived from.
    pub information: Matrix6<f64>,
}

/// A partial sum of the normal equations.
#[derive(Debug, Clone, Copy)]
struct SystemBlock {
    hessian: Matrix6<f64>,
    gradient: Vector6<f64>,
    cost: f64,
    squared_residual: f64,
    count: usize,
}

impl SystemBlock {
    fn zero() -> Self {
        Self {
            hessian: Matrix6::zeros(),
            gradient: Vector6::zeros(),
            cost: 0.0,
            squared_residual: 0.0,
            count: 0,
        }
    }

    fn absorb(&mut self, other: &Self) {
        self.hessian += other.hessian;
        self.gradient += other.gradient;
        self.cost += other.cost;
        self.squared_residual += other.squared_residual;
        self.count += other.count;
    }

    /// Mean robust cost per correspondence.
    ///
    /// The mean is what gets compared, not the sum: a sum drops merely by
    /// losing correspondences, so a step that threw away half the points
    /// would look like an improvement.
    fn mean_cost(&self) -> f64 {
        if self.count == 0 {
            f64::INFINITY
        } else {
            self.cost / self.count as f64
        }
    }
}

fn assemble<S>(
    source: &Surface<'_>,
    target: &Surface<'_>,
    search: &S,
    pose: &Se3,
    config: &IcpConfig,
) -> SystemBlock
where
    S: NeighborSearch + Sync,
{
    let count = source.cloud.len();
    if count == 0 {
        return SystemBlock::zero();
    }
    let rotation = *pose.rotation().matrix();
    let max_distance_squared =
        config.max_correspondence_distance * config.max_correspondence_distance;
    let chunks = count.div_ceil(REDUCTION_CHUNK);

    let blocks: Vec<SystemBlock> = (0..chunks)
        .into_par_iter()
        .map(|chunk| {
            let begin = chunk * REDUCTION_CHUNK;
            let end = ((chunk + 1) * REDUCTION_CHUNK).min(count);
            let mut block = SystemBlock::zero();
            let mut found: Vec<Neighbor> = Vec::with_capacity(1);

            for index in begin..end {
                let transformed = pose.transform_point(&source.cloud.point(index));
                search.knn_into(&transformed, 1, &mut found);
                let Some(nearest) = found.first() else {
                    continue;
                };
                if nearest.distance_squared > max_distance_squared {
                    continue;
                }

                let matched = nearest.index as usize;
                let target_normal = target.normals[matched];
                let source_normal = rotation * source.normals[index];
                if source_normal.dot(&target_normal).abs() < config.min_normal_cosine {
                    continue;
                }

                let residual = target_normal.dot(&(transformed - target.cloud.point(matched)));
                let weight = config.kernel.weight(residual);
                let row = point_to_plane_row(&transformed, &target_normal);

                block.hessian += (row * row.transpose()) * weight;
                block.gradient += row * (weight * residual);
                block.cost += config.kernel.loss(residual);
                block.squared_residual += residual * residual;
                block.count += 1;
            }
            block
        })
        .collect();

    let mut total = SystemBlock::zero();
    for block in &blocks {
        total.absorb(block);
    }
    total
}

/// Solves `(H + λ·D)·Δξ = −b`.
///
/// Marquardt damping, proportional to the diagonal rather than to the
/// identity. The reason is units: the translational block of `H` has
/// dimension 1/m² while the rotational block is dimensionless, and a
/// `λ·I` term would mix them.
///
/// The diagonal is floored at a fraction of its own maximum. Without that
/// floor a degenerate scene is not regularised at all: for the plane
/// `z = 0` the diagonal entry for `ρx` is identically zero, and `λ·diag`
/// never touches it. Degenerate scenes are the main case here, not the
/// exception.
fn solve_step(
    hessian: &Matrix6<f64>,
    gradient: &Vector6<f64>,
    damping: f64,
) -> Option<Vector6<f64>> {
    const DIAGONAL_FLOOR: f64 = 1e-6;
    let diagonal = hessian.diagonal();
    let largest = diagonal.max();
    // An explicit test rather than `!(largest > 0.0)`: a NaN must lead to
    // refusal too, not slip through the comparison.
    if !largest.is_finite() || largest <= 0.0 {
        return None;
    }
    let floor = largest * DIAGONAL_FLOOR;

    let mut damped = *hessian;
    for axis in 0..6 {
        damped[(axis, axis)] += damping * diagonal[axis].max(floor);
    }
    nalgebra::Cholesky::new(damped).map(|factorisation| factorisation.solve(&(-gradient)))
}

/// Registers `source` against `target`.
///
/// `search` must have been built over the `target` cloud. The returned
/// pose carries source points into the target frame: `q ≈ T · p`.
///
/// The update is left-multiplied, `T ← exp(Δξ)·T`, so the Jacobian row is
/// `[nᵀ | (p' × n)ᵀ]` with `p'` the already-transformed source point. The
/// convention runs through the project: the null spaces of the synthetic
/// scenes and the conditioning analysis assume the same one.
pub fn register<S>(
    source: &Surface<'_>,
    target: &Surface<'_>,
    search: &S,
    initial: Se3,
    config: &IcpConfig,
) -> IcpResult
where
    S: NeighborSearch + Sync,
{
    register_observed(source, target, search, initial, config, |_| {})
}

/// The same, with an observer called after every accepted iteration.
///
/// The iso-accuracy protocol needs it: comparing "time to a given
/// accuracy" rather than "time to one's own stopping criterion" requires
/// seeing the intermediate poses. Comparing stopping criteria measures a
/// difference in settings, not in code.
pub fn register_observed<S, F>(
    source: &Surface<'_>,
    target: &Surface<'_>,
    search: &S,
    initial: Se3,
    config: &IcpConfig,
    mut observer: F,
) -> IcpResult
where
    S: NeighborSearch + Sync,
    F: FnMut(&IterationReport),
{
    assert_eq!(
        source.cloud.len(),
        source.normals.len(),
        "the source has a different number of normals than points"
    );
    assert_eq!(
        target.cloud.len(),
        target.normals.len(),
        "the target has a different number of normals than points"
    );

    let mut pose = initial;
    let mut damping = config.initial_damping;
    let mut current = assemble(source, target, search, &pose, config);
    let mut iterations = 0;
    let mut converged = false;

    while iterations < config.max_iterations {
        iterations += 1;

        let Some(step) = solve_step(&current.hessian, &current.gradient, damping) else {
            damping *= 10.0;
            if damping > 1e12 {
                break;
            }
            continue;
        };

        let candidate_pose = Se3::exp(&step) * pose;
        let candidate = assemble(source, target, search, &candidate_pose, config);

        if candidate.mean_cost() <= current.mean_cost() {
            pose = candidate_pose;
            current = candidate;
            damping = (damping * 0.1).max(1e-12);
            observer(&IterationReport {
                iteration: iterations,
                pose,
                rmse: if current.count == 0 {
                    f64::INFINITY
                } else {
                    (current.squared_residual / current.count as f64).sqrt()
                },
                correspondences: current.count,
            });

            let translation_step = step.fixed_rows::<3>(0).norm();
            let rotation_step = step.fixed_rows::<3>(3).norm();
            if translation_step < config.translation_tolerance
                && rotation_step < config.rotation_tolerance
            {
                converged = true;
                break;
            }
        } else {
            damping *= 10.0;
            if damping > 1e12 {
                break;
            }
        }
    }

    let rmse = if current.count == 0 {
        f64::INFINITY
    } else {
        (current.squared_residual / current.count as f64).sqrt()
    };

    IcpResult {
        pose,
        iterations,
        converged,
        rmse,
        correspondences: current.count,
        information: current.hessian,
    }
}

/// Builds a [`Surface`] from an existing cloud and its normals.
///
/// It exists for readability at call sites: a literal
/// `Surface { cloud, normals }` gets lost in a chain of arguments.
pub fn surface<'a>(cloud: &'a PointCloud, normals: &'a [Vector3<f64>]) -> Surface<'a> {
    Surface { cloud, normals }
}
