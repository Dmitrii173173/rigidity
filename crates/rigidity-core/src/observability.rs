//! Conditioning of the registration problem and observability of the six
//! degrees of freedom.
//!
//! # What is computed here, and what is not
//!
//! What is computed is **conditioning**: how firmly the geometry of the
//! scene pins down each of the six motions. What is *not* computed is a
//! calibrated pose uncertainty — Censi's closed form understates real
//! spread by orders of magnitude. That is why the field is called
//! `conditioning` and not `covariance`.
//!
//! # The normalisation without which the answer is meaningless
//!
//! The Jacobian's columns carry different units: `∂e/∂ρ = n` is
//! dimensionless while `∂e/∂φ = p × n` is measured in metres. Singular
//! values of such a matrix are not comparable with one another, and
//! singular vectors that mix `ρ` and `φ` depend on the choice of units — a
//! verdict of "rotation about Z is degenerate" can flip when metres become
//! millimetres.
//!
//! The cure is a change of variable: `ξ' = [ρ; r_g·φ]`, where `r_g` is the
//! radius of gyration of the points about the centre of rotation. All six
//! coordinates are then in metres: one unit along a rotational axis means
//! one metre of displacement for a point at the characteristic distance.
//! The Jacobian in the new variables is `J` with its last three columns
//! divided by `r_g`, which is `S⁻¹HS⁻¹` without ever forming `H`.
//!
//! # Centre of rotation
//!
//! The report is built about the **centroid of the correspondences**,
//! not about the coordinate origin. Otherwise the same scene, referred to
//! a distant origin, would get a different condition number: `r_g` would
//! measure how far away the scene is rather than how large it is. The
//! decomposition into degrees of freedom genuinely depends on the choice
//! of centre, so the centre is part of the report.

use nalgebra::{Matrix6, Vector3, Vector6};
use rayon::prelude::*;

use crate::icp::{Kernel, point_to_plane_row};
use crate::lie::{Se3, So3};
use crate::linalg::{decompose, reduce};

/// Block size of the deterministic reduction used for the sums.
const CHUNK: usize = 4_096;

/// A single point-to-plane correspondence.
#[derive(Debug, Clone, Copy)]
pub struct Correspondence {
    /// The source point **after** the current pose has been applied.
    pub point: Vector3<f64>,
    /// The surface normal of the target.
    pub normal: Vector3<f64>,
    /// The residual `nᵀ(p − q)`.
    pub residual: f64,
}

/// How reliably a degree of freedom is determined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Observability {
    /// The spread fits within the required accuracy.
    High,
    /// The spread exceeds the required accuracy, but by less than tenfold.
    Medium,
    /// The spread exceeds the required accuracy by more than an order of
    /// magnitude.
    Low,
}

impl Observability {
    /// A short label for the report.
    pub fn label(self) -> &'static str {
        match self {
            Self::High => "HIGH",
            Self::Medium => "MEDIUM",
            Self::Low => "LOW",
        }
    }
}

/// What counts as reliable.
///
/// A degeneracy threshold without a required accuracy is meaningless:
/// 5 mm of spread is excellent for a mobile robot and catastrophic for a
/// welding cell.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObservabilityCriteria {
    /// Standard deviation of measurement noise along the normal, metres.
    pub noise_sigma: f64,
    /// Required pose accuracy, metres.
    pub tolerance: f64,
}

/// Conditioning of the problem.
#[derive(Debug, Clone)]
pub struct Conditioning {
    centre: Vector3<f64>,
    radius_of_gyration: f64,
    used: usize,
    values: [f64; 6],
    vectors: [[f64; 6]; 6],
}

impl Conditioning {
    /// The centre of rotation the report refers to.
    pub fn centre(&self) -> Vector3<f64> {
        self.centre
    }

    /// The radius of gyration: the characteristic length of the
    /// normalisation.
    pub fn radius_of_gyration(&self) -> f64 {
        self.radius_of_gyration
    }

    /// How many correspondences took part.
    pub fn used(&self) -> usize {
        self.used
    }

    /// The normalised singular values, in decreasing order.
    pub fn singular_values(&self) -> [f64; 6] {
        self.values
    }

    /// The condition number `σ_max / σ_min`.
    ///
    /// Invariant to the scale of the scene and to the choice of units,
    /// which is exactly what the normalisation is for.
    pub fn condition_number(&self) -> f64 {
        crate::linalg::condition_number(&self.values)
    }

    /// Direction number `index` in the normalised coordinates `ξ'`.
    pub fn direction(&self, index: usize) -> Vector6<f64> {
        Vector6::from_iterator((0..6).map(|axis| self.vectors[axis][index]))
    }

    /// The same direction in the original coordinates `ξ = [ρ; φ]`,
    /// referred to the coordinate origin.
    ///
    /// Undoes the normalisation (`φ` is divided by `r_g`) and moves the
    /// centre of rotation from the centroid back to the origin through the
    /// adjoint of that translation.
    pub fn direction_in_world(&self, index: usize) -> Vector6<f64> {
        let normalised = self.direction(index);
        let unscaled = Vector6::new(
            normalised[0],
            normalised[1],
            normalised[2],
            normalised[3] / self.radius_of_gyration,
            normalised[4] / self.radius_of_gyration,
            normalised[5] / self.radius_of_gyration,
        );
        let shift = Se3::from_parts(So3::identity(), self.centre);
        shift.adjoint() * unscaled
    }

    /// Converts a pose increment from world coordinates `ξ = [ρ; φ]` into
    /// the normalised `ξ'` the spectrum is expressed in.
    ///
    /// The inverse of [`direction_in_world`](Self::direction_in_world):
    /// first move the centre of rotation to the centroid, then scale the
    /// rotational part by the radius of gyration. Needed to compare a
    /// predicted spread against an empirical one — they must live in the
    /// same coordinates.
    pub fn to_normalised(&self, world: Vector6<f64>) -> Vector6<f64> {
        let shift = Se3::from_parts(So3::identity(), -self.centre);
        let centred = shift.adjoint() * world;
        Vector6::new(
            centred[0],
            centred[1],
            centred[2],
            centred[3] * self.radius_of_gyration,
            centred[4] * self.radius_of_gyration,
            centred[5] * self.radius_of_gyration,
        )
    }

    /// Projection of a pose increment onto direction `index`, metres.
    pub fn component(&self, index: usize, world: Vector6<f64>) -> f64 {
        self.direction(index).dot(&self.to_normalised(world))
    }

    /// Standard deviation along each direction, metres.
    ///
    /// `σ_noise / σ'ᵢ`. All six values share units: one unit of a
    /// normalised coordinate is one metre of displacement, for rotations
    /// too, at distance `r_g` from the centre.
    ///
    /// The point count is accounted for automatically: `σ'ᵢ` grows as the
    /// square root of the sum of weights, so the spread falls as `1/√N`.
    pub fn uncertainty(&self, noise_sigma: f64) -> [f64; 6] {
        let mut result = [f64::INFINITY; 6];
        for (slot, value) in result.iter_mut().zip(self.values.iter()) {
            *slot = if *value > 0.0 {
                noise_sigma / value
            } else {
                f64::INFINITY
            };
        }
        result
    }

    /// Classification of the six degrees of freedom.
    pub fn classify(&self, criteria: &ObservabilityCriteria) -> [Observability; 6] {
        /// By what factor the spread must exceed the tolerance before a
        /// degree of freedom counts as lost rather than merely weak.
        const MARGINAL_FACTOR: f64 = 10.0;

        let mut result = [Observability::Low; 6];
        for (slot, spread) in result
            .iter_mut()
            .zip(self.uncertainty(criteria.noise_sigma).iter())
        {
            *slot = if *spread < criteria.tolerance {
                Observability::High
            } else if *spread < MARGINAL_FACTOR * criteria.tolerance {
                Observability::Medium
            } else {
                Observability::Low
            };
        }
        result
    }

    /// Directions judged unobservable, in `ξ` coordinates relative to the
    /// origin.
    pub fn unobservable_directions(&self, criteria: &ObservabilityCriteria) -> Vec<Vector6<f64>> {
        self.classify(criteria)
            .iter()
            .enumerate()
            .filter(|(_, state)| **state == Observability::Low)
            .map(|(index, _)| self.direction_in_world(index))
            .collect()
    }
}

/// The full result of the analysis.
#[derive(Debug, Clone)]
pub struct Analysis {
    /// The conditioning.
    pub conditioning: Conditioning,
    /// The sandwich covariance estimate, in normalised coordinates.
    ///
    /// `None` when the weighted matrix is singular and cannot be inverted
    /// — which is precisely the case this project exists for.
    pub sandwich: Option<Matrix6<f64>>,
    /// The naive estimate `σ̂²·H⁻¹`, for comparison.
    pub naive: Option<Matrix6<f64>>,
}

fn triangle_to_matrix(triangle: &[[f64; 6]; 6]) -> Matrix6<f64> {
    let mut matrix = Matrix6::zeros();
    for (i, row) in triangle.iter().enumerate() {
        for (j, value) in row.iter().enumerate() {
            matrix[(i, j)] = *value;
        }
    }
    matrix
}

/// A deterministic sum over fixed-size blocks.
fn chunked_sum<T, F, C>(count: usize, zero: T, add: F, combine: C) -> T
where
    T: Copy + Send + Sync,
    F: Fn(usize, T) -> T + Sync,
    C: Fn(T, T) -> T,
{
    if count == 0 {
        return zero;
    }
    let chunks = count.div_ceil(CHUNK);
    let partials: Vec<T> = (0..chunks)
        .into_par_iter()
        .map(|chunk| {
            let begin = chunk * CHUNK;
            let end = ((chunk + 1) * CHUNK).min(count);
            let mut accumulator = zero;
            for index in begin..end {
                accumulator = add(index, accumulator);
            }
            accumulator
        })
        .collect();
    partials
        .iter()
        .fold(zero, |total, part| combine(total, *part))
}

/// Analyses a set of correspondences.
///
/// Returns `None` when no correspondence survived, in which case there is
/// nothing to say.
pub fn analyse<F>(count: usize, kernel: Kernel, correspondence: F) -> Option<Analysis>
where
    F: Fn(usize) -> Option<Correspondence> + Sync,
{
    let weight_of = |index: usize| {
        correspondence(index).map(|item| {
            let weight = kernel.weight(item.residual);
            (item, weight)
        })
    };

    // Centroid of the correspondences.
    let (weight_sum, moment, used) = chunked_sum(
        count,
        (0.0f64, Vector3::zeros(), 0usize),
        |index, (weight_sum, moment, used)| match weight_of(index) {
            Some((item, weight)) => (weight_sum + weight, moment + item.point * weight, used + 1),
            None => (weight_sum, moment, used),
        },
        |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
    );
    if used == 0 || weight_sum <= 0.0 {
        return None;
    }
    let centre = moment / weight_sum;

    // Radius of gyration in a separate pass, not through
    // `Σw|p|² − |c|²`. That expression subtracts nearly equal numbers when
    // the scene sits far from the origin — precisely the case the centroid
    // was introduced for.
    let spread = chunked_sum(
        count,
        0.0f64,
        |index, total| match weight_of(index) {
            Some((item, weight)) => total + weight * (item.point - centre).norm_squared(),
            None => total,
        },
        |a, b| a + b,
    );
    let radius_of_gyration = (spread / weight_sum).sqrt();
    if !radius_of_gyration.is_finite() || radius_of_gyration <= 0.0 {
        return None;
    }

    // A row of the normalised Jacobian: its last three columns divided by
    // the characteristic length.
    let normalised_row = |item: &Correspondence| {
        let row = point_to_plane_row(&(item.point - centre), &item.normal);
        [
            row[0],
            row[1],
            row[2],
            row[3] / radius_of_gyration,
            row[4] / radius_of_gyration,
            row[5] / radius_of_gyration,
        ]
    };

    let weighted = reduce::<6, _>(count, |index| {
        weight_of(index).map(|(item, weight)| {
            let scale = weight.sqrt();
            let row = normalised_row(&item);
            [
                row[0] * scale,
                row[1] * scale,
                row[2] * scale,
                row[3] * scale,
                row[4] * scale,
                row[5] * scale,
            ]
        })
    });

    let decomposition = decompose(weighted.triangle());
    let conditioning = Conditioning {
        centre,
        radius_of_gyration,
        used,
        values: decomposition.values,
        vectors: decomposition.vectors,
    };

    // The sandwich: `H⁻¹·M·H⁻¹` with `M = Σ ψ(e)²·JJᵀ`.
    //
    // The bread uses the IRLS weights rather than `ψ'`. This is the
    // standard approximation, and it has a reason: for redescending
    // kernels `ψ'` changes sign, the exact matrix stops being positive
    // semi-definite, and it cannot be decomposed the same way through
    // TSQR.
    let meat_triangle = reduce::<6, _>(count, |index| {
        weight_of(index).map(|(item, weight)| {
            let scale = (weight * item.residual).abs();
            let row = normalised_row(&item);
            [
                row[0] * scale,
                row[1] * scale,
                row[2] * scale,
                row[3] * scale,
                row[4] * scale,
                row[5] * scale,
            ]
        })
    });

    let bread = triangle_to_matrix(weighted.triangle());
    let hessian = bread.transpose() * bread;
    let meat_upper = triangle_to_matrix(meat_triangle.triangle());
    let meat = meat_upper.transpose() * meat_upper;

    let (sandwich, naive) = match hessian.try_inverse() {
        Some(inverse) => {
            let residual_energy = chunked_sum(
                count,
                0.0f64,
                |index, total| match weight_of(index) {
                    Some((item, weight)) => total + weight * item.residual * item.residual,
                    None => total,
                },
                |a, b| a + b,
            );
            let degrees = (used as f64 - 6.0).max(1.0);
            let variance = residual_energy / degrees;
            (Some(inverse * meat * inverse), Some(inverse * variance))
        }
        None => (None, None),
    };

    Some(Analysis {
        conditioning,
        sandwich,
        naive,
    })
}

impl Analysis {
    /// A human-readable report.
    pub fn describe(&self, criteria: &ObservabilityCriteria) -> String {
        let conditioning = &self.conditioning;
        let states = conditioning.classify(criteria);
        let spread = conditioning.uncertainty(criteria.noise_sigma);
        let names = ["σ₁", "σ₂", "σ₃", "σ₄", "σ₅", "σ₆"];

        let mut text = String::new();
        text.push_str(&format!(
            "correspondences:   {}\n\
             centre of rotation: [{:.3}, {:.3}, {:.3}] m\n\
             radius of gyration: {:.3} m\n\
             condition number:   {:.4e}\n\n",
            conditioning.used(),
            conditioning.centre().x,
            conditioning.centre().y,
            conditioning.centre().z,
            conditioning.radius_of_gyration(),
            conditioning.condition_number(),
        ));
        for index in 0..6 {
            let direction = conditioning.direction_in_world(index);
            text.push_str(&format!(
                "{}  spread {:>10.3e} m  {:<6}  ρ=[{:+.2} {:+.2} {:+.2}] φ=[{:+.2} {:+.2} {:+.2}]\n",
                names[index],
                spread[index],
                states[index].label(),
                direction[0],
                direction[1],
                direction[2],
                direction[3],
                direction[4],
                direction[5],
            ));
        }
        if states.contains(&Observability::Low) {
            text.push_str(
                "\nwarning: some degrees of freedom are not determined by the geometry\n",
            );
        }
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane_analysis() -> Analysis {
        analyse(400, Kernel::Squared, |index| {
            let x = (index % 20) as f64 * 0.1 - 1.0;
            let y = (index / 20) as f64 * 0.1 - 1.0;
            Some(Correspondence {
                point: Vector3::new(x, y, 0.0),
                normal: Vector3::z(),
                residual: 0.0,
            })
        })
        .unwrap()
    }

    /// The coordinate conversions are mutual inverses.
    #[test]
    fn normalised_and_world_coordinates_round_trip() {
        let analysis = plane_analysis();
        let conditioning = &analysis.conditioning;
        for index in 0..6 {
            let world = conditioning.direction_in_world(index);
            let back = conditioning.to_normalised(world);
            let expected = conditioning.direction(index);
            assert!(
                (back - expected).norm() < 1e-12,
                "direction {index}: mismatch {:.3e}",
                (back - expected).norm()
            );
        }
    }

    /// A plane has exactly three unobservable directions, with zero `σ`.
    #[test]
    fn plane_loses_three_degrees_of_freedom() {
        let values = plane_analysis().conditioning.singular_values();
        assert!(values[2] > 1.0, "third value {}", values[2]);
        for value in values.iter().skip(3) {
            assert!(*value < 1e-12, "residual value {value:.3e}");
        }
    }
}
