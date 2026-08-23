//! Somebody else's answer to the question this project failed at, so that it
//! can be measured here rather than argued about.
//!
//! # Whose, and why it is here
//!
//! Weighting a pose-graph edge by which directions its geometry determined
//! is not this project's idea. Zhang and Singh (2016) thresholded the
//! eigenvalues of the point-to-plane Hessian and restricted the update to
//! the well-conditioned subspace; Hinduja, Bartlett and Kaess (IROS 2019)
//! carried that into a pose graph as a factor that constrains only the
//! non-degenerate directions. That is exactly what this project's
//! `calibrated_information` did until S6 measured it losing every time.
//!
//! Hatleskog and Alexis (RA-L 2024, *Probabilistic Degeneracy Detection for
//! Point-to-Plane Error Minimization*) say why such thresholds do not
//! travel: the eigenvalues of the Hessian depend on the measurement count,
//! the weights, the scale of the scene and the sensor's noise, so a
//! threshold tuned on one setup means nothing on another. Their remedy is
//! to stop thresholding the eigenvalue and start comparing it with its own
//! uncertainty: model the noise that enters the Hessian — from the points
//! *and* from the estimated normals — and ask, per direction, the
//! probability that the signal exceeds that noise by a factor `s`. The
//! probability then attenuates that direction's contribution smoothly.
//!
//! This is a reimplementation of that, in this project's coordinates, for
//! one purpose: this repository has millimetre theodolite ground truth on
//! whole surveys and a rig that isolates the edge weight, and their own
//! paper has ground truth for one of its four experiments. Measuring their
//! method against that truth is worth more than another idea of ours.
//!
//! # The perturbation matrix, derived here rather than transcribed
//!
//! Their state is `[δr; δt]`; this project's is `ξ = [ρ; φ]`, translation
//! first, so the blocks are transposed into that order rather than copied.
//! With `v = w·[n; p×n]` a correspondence's contribution to `H = Σ v vᵀ`,
//! a point error `ε` and a normal error `n̂ ≈ n + [n]×η`:
//!
//! ```text
//! v̂ = w·[ n + [n]×η ; (p+ε)×(n+[n]×η) ]
//!    ≈ v + w·[ 0        [n]×      ]·[ε; η]
//!            [ −[n]×   [p]×[n]×   ]
//! ```
//!
//! — because `ε×n = −[n]×ε` and `p×([n]×η) = [p]×[n]×η`. The second-order
//! term is dropped, as it is in the paper, since it does not enter the
//! covariance to first order.

use rigidity_core::icp::point_to_plane_row;
use rigidity_core::lie::Se3;
use rigidity_core::nalgebra::{Matrix3, Matrix6, SymmetricEigen, Vector3, Vector6};
use rigidity_core::neighbors::NeighborSearch;
use rigidity_pipeline::{Prepared, RegisterParams};

/// What the noise model needs to be told.
#[derive(Debug, Clone, Copy)]
pub struct NoiseParams {
    /// Standard deviation of the noise on a point, metres, isotropic.
    ///
    /// From the instrument's datasheet, which is the paper's whole point:
    /// nothing here is tuned on the data it is run over.
    pub point_sigma: f64,
    /// How many neighbours the normals were estimated from. The covariance
    /// of a normal falls as that count rises, and the method needs to know
    /// which count it was.
    pub neighbours: usize,
    /// How far the signal must exceed the noise before a direction counts
    /// as determined. The paper uses ten, targeting a relative error of at
    /// most a tenth.
    pub signal_to_noise: f64,
}

impl Default for NoiseParams {
    fn default() -> Self {
        Self {
            point_sigma: 0.03,
            neighbours: 16,
            signal_to_noise: 10.0,
        }
    }
}

/// What the method says about one registration.
#[derive(Debug, Clone, Copy)]
pub struct Probabilistic {
    /// The edge's information matrix, `ξ = [ρ; φ]`, about the origin —
    /// the same coordinates `IcpResult::information` is in.
    pub information: Matrix6<f64>,
    /// The probability that each direction is determined rather than noise,
    /// in the order of the directions below.
    pub probability: [f64; 6],
    /// The directions themselves, as columns: the eigenvectors of `H`.
    pub directions: Matrix6<f64>,
    /// The eigenvalues of `H`, in the same order.
    pub eigenvalues: [f64; 6],
    /// The weighted mean square point-to-plane residual this edge was
    /// measured at, which is the `σ_r²` the information above is divided
    /// by. Reported separately so that a caller can ask what that division
    /// is doing: it is a *per-edge* scaling, and in a survey a per-edge
    /// scaling is a statement about which edges to believe.
    pub residual_variance: f64,
}

/// The information an edge's geometry justifies, by Hatleskog and Alexis.
///
/// Returns `None` when nothing matched, which is its own answer.
pub fn probabilistic_information(
    moving: &Prepared,
    fixed: &Prepared,
    pose: &Se3,
    params: &RegisterParams,
    noise: &NoiseParams,
) -> Option<Probabilistic> {
    let normal_covariance = normal_covariances(fixed, noise);
    let point_covariance = Matrix3::from_diagonal_element(noise.point_sigma * noise.point_sigma);

    // One pass to collect what both passes need. Storing the pair costs
    // about three hundred bytes a correspondence and saves recomputing a
    // 6×6 congruence for every one of the six directions.
    let limit = params.max_distance * params.max_distance;
    let kernel = params.kernel();
    let mut rows: Vec<(Vector6<f64>, Matrix6<f64>)> = Vec::new();
    let mut hessian = Matrix6::zeros();
    let mut noise_sum = Matrix6::zeros();
    let mut residual_square = 0.0;
    let mut found = Vec::with_capacity(1);
    for index in 0..moving.cloud.len() {
        let point = pose.transform_point(&moving.cloud.point(index));
        fixed.tree.knn_into(&point, 1, &mut found);
        let Some(nearest) = found.first() else {
            continue;
        };
        if nearest.distance_squared > limit {
            continue;
        }
        let matched = nearest.index as usize;
        let normal = fixed.normals[matched];
        let residual = normal.dot(&(point - fixed.cloud.point(matched)));
        let weight = kernel.weight(residual);
        if weight <= 0.0 {
            continue;
        }

        let row = point_to_plane_row(&point, &normal);
        let contribution = row * weight;
        hessian += contribution * contribution.transpose();
        residual_square += weight * residual * residual;

        let cross_normal = skew(&normal);
        let mut perturbation = Matrix6::zeros();
        perturbation
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&cross_normal);
        perturbation
            .fixed_view_mut::<3, 3>(3, 0)
            .copy_from(&(-cross_normal));
        perturbation
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&(skew(&point) * cross_normal));

        let mut source = Matrix6::zeros();
        source
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&point_covariance);
        source
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&normal_covariance[matched]);
        let covariance = perturbation * source * perturbation.transpose() * (weight * weight);

        noise_sum += covariance;
        rows.push((contribution, covariance));
    }
    if rows.is_empty() {
        return None;
    }

    let eigen = SymmetricEigen::new(hessian);
    let mut information = Matrix6::zeros();
    let mut probability = [0.0; 6];
    let mut eigenvalues = [0.0; 6];
    let variance = residual_square / rows.len() as f64;
    for axis in 0..6 {
        let direction: Vector6<f64> = eigen.eigenvectors.column(axis).into();
        let value = eigen.eigenvalues[axis];
        eigenvalues[axis] = value;

        // The noise in this direction: its mean, and the variance of the
        // quadratic form that carries it.
        let mean = (direction.transpose() * noise_sum * direction)[(0, 0)];
        let mut spread = 0.0;
        for (row, covariance) in &rows {
            let quadratic = (direction.transpose() * covariance * direction)[(0, 0)];
            let projection = direction.dot(row);
            spread += 2.0 * quadratic * quadratic + 4.0 * quadratic * projection * projection;
        }
        let spread = spread.max(0.0).sqrt();

        // P(signal ≥ s·noise), with the noise normal about its mean.
        let threshold = value / (noise.signal_to_noise + 1.0);
        let p = if spread > 0.0 {
            standard_normal_cdf((threshold - mean) / spread)
        } else if threshold >= mean {
            1.0
        } else {
            0.0
        };
        probability[axis] = p;
        information += direction * direction.transpose() * (p * value);
    }
    if variance > 0.0 {
        information /= variance;
    }

    Some(Probabilistic {
        information,
        probability,
        directions: eigen.eigenvectors,
        eigenvalues,
        residual_variance: variance,
    })
}

/// The covariance of every normal of a cloud, from the neighbourhood it was
/// fitted to.
///
/// Following the paper: a plane fitted to `N` points of covariance `Ĉ` with
/// eigenvalues `λ₁ ≥ λ₂ ≥ λ₃` leaves the normal free to rotate about its
/// own long axis with variance `(σ²/N)/λ₂`, about the second with
/// `(σ²/N)/λ₁`, and not at all about itself. A patch that is long and thin
/// pins the normal against its long axis and barely at all across it, which
/// is what those two ratios say.
fn normal_covariances(cloud: &Prepared, noise: &NoiseParams) -> Vec<Matrix3<f64>> {
    let count = noise.neighbours.max(3);
    let scale = noise.point_sigma * noise.point_sigma / count as f64;
    let mut out = Vec::with_capacity(cloud.cloud.len());
    let mut found = Vec::with_capacity(count);
    for index in 0..cloud.cloud.len() {
        let query = cloud.cloud.point(index);
        cloud.tree.knn_into(&query, count, &mut found);
        if found.len() < 3 {
            out.push(Matrix3::zeros());
            continue;
        }
        let n = found.len() as f64;
        let mut centroid = Vector3::zeros();
        for neighbour in &found {
            centroid += cloud.cloud.point(neighbour.index as usize);
        }
        centroid /= n;
        let mut covariance = Matrix3::zeros();
        for neighbour in &found {
            let delta = cloud.cloud.point(neighbour.index as usize) - centroid;
            covariance += delta * delta.transpose();
        }
        covariance /= n - 1.0;

        let eigen = SymmetricEigen::new(covariance);
        let mut order = [0usize, 1, 2];
        order.sort_by(|a, b| eigen.eigenvalues[*b].total_cmp(&eigen.eigenvalues[*a]));
        let (large, middle) = (eigen.eigenvalues[order[0]], eigen.eigenvalues[order[1]]);
        let axes = [
            eigen.eigenvectors.column(order[0]).into_owned(),
            eigen.eigenvectors.column(order[1]).into_owned(),
        ];
        let mut result = Matrix3::zeros();
        if middle > 0.0 {
            result += axes[0] * axes[0].transpose() * (scale / middle);
        }
        if large > 0.0 {
            result += axes[1] * axes[1].transpose() * (scale / large);
        }
        out.push(result);
    }
    out
}

fn skew(vector: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(
        0.0, -vector.z, vector.y, vector.z, 0.0, -vector.x, -vector.y, vector.x, 0.0,
    )
}

/// `Φ(z)`, through the Abramowitz–Stegun rational approximation of `erf`.
///
/// Good to about 1.5·10⁻⁷, which is four orders finer than anything the
/// probability above is used for.
fn standard_normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    const A: [f64; 5] = [
        0.254829592,
        -0.284496736,
        1.421413741,
        -1.453152027,
        1.061405429,
    ];
    const P: f64 = 0.327_591_1;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + P * x);
    let poly = A.iter().rev().fold(0.0, |accumulated, coefficient| {
        (accumulated + coefficient) * t
    });
    sign * (1.0 - poly * (-x * x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rigidity_core::PointCloud;
    use rigidity_pipeline::{PrepareParams, prepare_cloud};

    /// `Φ` against values that can be looked up.
    #[test]
    fn the_normal_tail_is_where_it_should_be() {
        for (z, want) in [
            (0.0, 0.5),
            (1.0, 0.841_344_7),
            (-1.0, 0.158_655_3),
            (1.959_964, 0.975),
            (3.0, 0.998_650_1),
        ] {
            let got = standard_normal_cdf(z);
            assert!((got - want).abs() < 1e-6, "Φ({z}) = {got}, expected {want}");
        }
    }

    /// A corridor with a little at its far end: the axis along it is the
    /// one direction the geometry does not determine, and the method has to
    /// say so *without being told what a corridor is*.
    ///
    /// The scene is a box, not two walls. The first version of this test
    /// used two facing walls and failed, correctly: two parallel planes
    /// constrain one translation and two rotations and leave *three*
    /// directions free, so the least probable direction came back a mixture
    /// of translation along the corridor and rotation across it — an
    /// arbitrary vector inside a three-dimensional null space. A floor and
    /// a ceiling take two of those away, and then exactly one is left.
    #[test]
    fn the_corridor_axis_is_the_improbable_one() {
        const HALF_LENGTH: f64 = 10.0;
        const HALF_WIDTH: f64 = 2.0;
        const HALF_HEIGHT: f64 = 1.5;

        let mut cloud = PointCloud::new();
        let mut push = |point: Vector3<f64>| cloud.push(point);
        for step in 0..400 {
            let along = -HALF_LENGTH + step as f64 * 0.05;
            let jitter = ((step * 37) % 11) as f64 * 0.002;
            for lateral in 0..12 {
                let across = -HALF_WIDTH + lateral as f64 * (2.0 * HALF_WIDTH / 11.0) + jitter;
                let up = -HALF_HEIGHT + lateral as f64 * (2.0 * HALF_HEIGHT / 11.0) + jitter;
                push(Vector3::new(along, -HALF_WIDTH, up));
                push(Vector3::new(along, HALF_WIDTH, up));
                push(Vector3::new(along, across, -HALF_HEIGHT));
                push(Vector3::new(along, across, HALF_HEIGHT));
            }
        }
        // A small patch on the far end: enough for a normal to exist there,
        // which is what makes the along-axis Hessian invertible and wrong
        // rather than singular and honest.
        for a in 0..4 {
            for b in 0..4 {
                push(Vector3::new(
                    HALF_LENGTH,
                    -0.3 + a as f64 * 0.2,
                    -0.3 + b as f64 * 0.2,
                ));
            }
        }

        let prepared = prepare_cloud(
            &cloud,
            &PrepareParams {
                voxel: 0.0,
                neighbours: 16,
            },
        )
        .expect("the corridor prepares");

        let report = probabilistic_information(
            &prepared,
            &prepared,
            &Se3::identity(),
            &RegisterParams::default(),
            &NoiseParams {
                point_sigma: 0.01,
                neighbours: 16,
                signal_to_noise: 10.0,
            },
        )
        .expect("the corridor matches itself");

        let worst = (0..6)
            .min_by(|a, b| report.probability[*a].total_cmp(&report.probability[*b]))
            .expect("six directions");
        let direction: Vector6<f64> = report.directions.column(worst).into();
        assert!(
            direction[0].abs() > 0.9,
            "the least probable direction is {direction:?}, and it should be translation along x; \
             probabilities {:?}",
            report.probability
        );
        assert!(
            report.probability[worst] < 0.5,
            "the corridor's own axis came back at p = {}",
            report.probability[worst]
        );

        // And the rest of the scene has to survive: a method that called
        // everything degenerate would pass the assertion above.
        let confident = report.probability.iter().filter(|p| **p > 0.9).count();
        assert!(
            confident >= 4,
            "only {confident} directions of six came back determined: {:?}",
            report.probability
        );
    }
}
