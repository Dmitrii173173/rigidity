//! Singular values by one-sided Jacobi.
//!
//! # Why not the QR algorithm
//!
//! Golub–Reinsch delivers singular values with **absolute** accuracy of
//! order `ε·σ_max`. For the largest ones that is excellent; for the
//! smallest it is useless: at `σ_min/σ_max = 10⁻¹²` the relative error is
//! `10⁻¹⁶/10⁻¹² = 10⁻⁴`.
//!
//! One-sided Jacobi (Demmel, Veselić) delivers **relative** accuracy for
//! all singular values, bounded by `ε·κ(A_c)` where `A_c` is the matrix
//! with normalised columns. The distinction is the whole point: the
//! smallest singular value is precisely what is being measured.
//!
//! The method orthogonalises the columns by rotations; when it finishes,
//! the singular values are the column norms.

/// Singular values together with the right singular vectors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Decomposition<const K: usize> {
    /// Singular values in decreasing order.
    pub values: [f64; K],
    /// Right singular vectors: column `j` belongs to `values[j]`.
    ///
    /// Element `vectors[i][j]` is coordinate `i` of vector `j`.
    pub vectors: [[f64; K]; K],
}

/// Singular values of a square matrix, in decreasing order.
///
/// The matrix is given row by row: `matrix[i][j]` is the element in row
/// `i`, column `j`.
///
/// Applicable both to `R` from TSQR and to an explicitly assembled `H`. In
/// the latter case it returns the eigenvalues of `H`, since `H` is
/// symmetric positive semi-definite. Using one and the same routine on
/// both paths is deliberate: then any difference in the results is
/// explained by squaring the condition number alone, not by the choice of
/// solver.
pub fn singular_values<const K: usize>(matrix: &[[f64; K]; K]) -> [f64; K] {
    decompose(matrix).values
}

/// The full decomposition: magnitudes and directions.
///
/// The method orthogonalises columns by rotations. The same rotations
/// accumulate into an identity matrix, which yields `V`, since
/// `A·V = U·Σ`. The directions are what let the answer be "this particular
/// motion is unobservable" rather than "the problem is ill-conditioned".
pub fn decompose<const K: usize>(matrix: &[[f64; K]; K]) -> Decomposition<K> {
    /// Threshold: a rotation is skipped when the columns are already
    /// nearly orthogonal.
    const TOLERANCE: f64 = 1e-17;
    /// Cap on the number of sweeps. At this size the method converges in
    /// a handful of them; the cap is a safety net, not an operating mode.
    const MAX_SWEEPS: usize = 40;

    let mut work = *matrix;
    let mut rotations = [[0.0f64; K]; K];
    for (index, row) in rotations.iter_mut().enumerate() {
        row[index] = 1.0;
    }

    for _ in 0..MAX_SWEEPS {
        let mut rotated = false;

        for p in 0..K {
            for q in (p + 1)..K {
                let mut alpha = 0.0;
                let mut beta = 0.0;
                let mut gamma = 0.0;
                for row in work.iter() {
                    alpha += row[p] * row[p];
                    beta += row[q] * row[q];
                    gamma += row[p] * row[q];
                }

                if gamma == 0.0 || alpha == 0.0 || beta == 0.0 {
                    continue;
                }
                // The criterion is relative: columns of different scale
                // must not be compared in absolute terms.
                if gamma.abs() <= TOLERANCE * (alpha * beta).sqrt() {
                    continue;
                }

                let zeta = (beta - alpha) / (2.0 * gamma);
                let tangent = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let cosine = 1.0 / (1.0 + tangent * tangent).sqrt();
                let sine = cosine * tangent;

                for row in work.iter_mut().chain(rotations.iter_mut()) {
                    let left = row[p];
                    let right = row[q];
                    row[p] = cosine * left - sine * right;
                    row[q] = sine * left + cosine * right;
                }
                rotated = true;
            }
        }

        if !rotated {
            break;
        }
    }

    let mut norms = [0.0f64; K];
    for (column, value) in norms.iter_mut().enumerate() {
        let mut sum = 0.0;
        for row in work.iter() {
            sum += row[column] * row[column];
        }
        *value = sum.sqrt();
    }

    // Sort in decreasing order, carrying the matching columns of `V`.
    let mut order: [usize; K] = [0; K];
    for (index, slot) in order.iter_mut().enumerate() {
        *slot = index;
    }
    order.sort_by(|a, b| norms[*b].total_cmp(&norms[*a]));

    let mut values = [0.0f64; K];
    let mut vectors = [[0.0f64; K]; K];
    for (target, source) in order.iter().enumerate() {
        values[target] = norms[*source];
        for axis in 0..K {
            vectors[axis][target] = rotations[axis][*source];
        }
    }
    Decomposition { values, vectors }
}

/// The condition number `σ_max / σ_min`.
///
/// Infinity when the smallest singular value is zero.
pub fn condition_number<const K: usize>(values: &[f64; K]) -> f64 {
    let largest = values[0];
    let smallest = values[K - 1];
    if smallest == 0.0 {
        f64::INFINITY
    } else {
        largest / smallest
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A diagonal matrix: the singular values are the absolute values of
    /// the diagonal.
    #[test]
    fn diagonal_matrix_is_trivial() {
        let mut matrix = [[0.0f64; 4]; 4];
        for (i, value) in [3.0, -1.0, 0.25, 8.0].into_iter().enumerate() {
            matrix[i][i] = value;
        }
        let values = singular_values(&matrix);
        assert!((values[0] - 8.0).abs() < 1e-15);
        assert!((values[1] - 3.0).abs() < 1e-15);
        assert!((values[2] - 1.0).abs() < 1e-15);
        assert!((values[3] - 0.25).abs() < 1e-15);
    }

    /// An orthogonal matrix: every singular value equals one.
    #[test]
    fn rotation_has_unit_spectrum() {
        let angle = 0.7f64;
        let matrix = [
            [angle.cos(), -angle.sin(), 0.0],
            [angle.sin(), angle.cos(), 0.0],
            [0.0, 0.0, 1.0],
        ];
        for value in singular_values(&matrix) {
            assert!((value - 1.0).abs() < 1e-15);
        }
    }

    /// A rank-deficient matrix yields a zero singular value.
    #[test]
    fn rank_deficient_matrix_yields_zero() {
        let matrix = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [-1.0, -2.0, -3.0]];
        let values = singular_values(&matrix);
        assert!(values[0] > 1.0);
        assert!(values[1] < 1e-15, "second value {}", values[1]);
        assert!(values[2] < 1e-15);
        assert_eq!(condition_number(&values), f64::INFINITY);
    }

    /// The method keeps relative accuracy on a diagonal with an enormous
    /// spread — something the QR algorithm does not promise.
    #[test]
    fn tiny_diagonal_entries_keep_relative_accuracy() {
        let scales = [1.0, 1e-4, 1e-8, 1e-12, 1e-16, 1e-20];
        let mut matrix = [[0.0f64; 6]; 6];
        for (i, scale) in scales.iter().enumerate() {
            matrix[i][i] = *scale;
        }
        let values = singular_values(&matrix);
        for (found, expected) in values.iter().zip(scales.iter()) {
            let relative = (found - expected).abs() / expected;
            assert!(
                relative < 1e-15,
                "expected {expected:e}, got {found:e}, error {relative:.3e}"
            );
        }
    }
}

#[cfg(test)]
mod vector_tests {
    use super::*;

    /// The directions form an orthonormal basis.
    #[test]
    fn right_vectors_are_orthonormal() {
        let matrix = [
            [3.0, 1.0, -2.0, 0.5],
            [0.0, 2.5, 1.0, -1.0],
            [1.0, 0.0, 4.0, 2.0],
            [-0.5, 1.5, 0.0, 3.0],
        ];
        let result = decompose(&matrix);
        for i in 0..4 {
            for j in 0..4 {
                let dot: f64 = (0..4)
                    .map(|axis| result.vectors[axis][i] * result.vectors[axis][j])
                    .sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1e-13, "V columns {i},{j}: {dot}");
            }
        }
    }

    /// `A·vⱼ` has norm `σⱼ`.
    #[test]
    fn vectors_match_their_values() {
        let matrix = [[2.0, 0.0, 1.0], [0.0, 3.0, 0.0], [1.0, 0.0, 2.0]];
        let result = decompose(&matrix);
        for j in 0..3 {
            let mut image = [0.0f64; 3];
            for (i, slot) in image.iter_mut().enumerate() {
                *slot = (0..3).map(|k| matrix[i][k] * result.vectors[k][j]).sum();
            }
            let norm = image.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!(
                (norm - result.values[j]).abs() < 1e-13,
                "‖A·v{j}‖ = {norm}, but σ{j} = {}",
                result.values[j]
            );
        }
    }

    /// The direction of a zero singular value lies in the kernel.
    #[test]
    fn null_direction_is_annihilated() {
        let matrix = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [-1.0, -2.0, -3.0]];
        let result = decompose(&matrix);
        let null = [
            result.vectors[0][2],
            result.vectors[1][2],
            result.vectors[2][2],
        ];
        for row in &matrix {
            let value: f64 = (0..3).map(|k| row[k] * null[k]).sum();
            assert!(value.abs() < 1e-13, "row gives {value}");
        }
    }
}
