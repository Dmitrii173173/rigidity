//! Tall-skinny QR by Givens rotations.
//!
//! # Why
//!
//! Moving to the normal equations squares the condition number:
//! `κ(JᵀJ) = κ(J)²`. This project is entirely about the small singular
//! values, and forming `H` destroys those first. At `κ(J) = 10⁷` — the
//! ceiling imposed by storing points as `f32` — we get `κ(H) = 10¹⁴`,
//! already comparable to the resolution of `f64`.
//!
//! `J = QR` with orthogonal `Q` gives `σᵢ(J) = σᵢ(R)` exactly, and `R` is
//! only 6×6 (or 7×7 with an appended residual column). The singular values
//! then come from `R`, and all available accuracy survives.
//!
//! # Why Givens rather than Householder
//!
//! Rows arrive one at a time, and a rotation folds a row into the triangle
//! in place: no buffer for all `N` rows, constant memory, strictly
//! streaming traversal. Backward stability is the same for both methods.
//!
//! The same "absorb a row" primitive also merges two triangles in the
//! reduction tree — feed one's rows to the other.

use rayon::prelude::*;

/// How many rows one block handles.
///
/// Block boundaries depend only on this constant and the row count.
/// Neither the thread count nor the scheduler's split affects the result.
const CHUNK: usize = 4_096;

/// The upper-triangular factor `R` of the decomposition `J = QR`.
///
/// `Q` is neither stored nor computed: the spectrum does not need it, and
/// for the least-squares solve it is enough to append a residual column to
/// `J` — then `Qᵀe` falls out of the same rotations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tsqr<const K: usize> {
    triangle: [[f64; K]; K],
}

impl<const K: usize> Default for Tsqr<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const K: usize> Tsqr<K> {
    /// An empty accumulator.
    pub fn new() -> Self {
        Self {
            triangle: [[0.0; K]; K],
        }
    }

    /// The upper-triangular matrix `R`, row by row.
    pub fn triangle(&self) -> &[[f64; K]; K] {
        &self.triangle
    }

    /// Folds a row into the triangle using Givens rotations.
    ///
    /// For each column a rotation is built that zeroes the next element of
    /// the row. The radius comes from [`f64::hypot`], which neither
    /// overflows on large values nor loses accuracy on small ones, unlike
    /// `(a*a + b*b).sqrt()`.
    pub fn absorb_row(&mut self, row: &[f64; K]) {
        let mut row = *row;
        for column in 0..K {
            let lower = row[column];
            if lower == 0.0 {
                continue;
            }
            let upper = self.triangle[column][column];
            let radius = upper.hypot(lower);
            if radius == 0.0 {
                continue;
            }
            let cosine = upper / radius;
            let sine = lower / radius;

            self.triangle[column][column] = radius;
            row[column] = 0.0;
            let tail = column + 1;
            for (above, below) in self.triangle[column][tail..]
                .iter_mut()
                .zip(row[tail..].iter_mut())
            {
                let upper_value = *above;
                let lower_value = *below;
                *above = cosine * upper_value + sine * lower_value;
                *below = cosine * lower_value - sine * upper_value;
            }
        }
    }

    /// Merges another triangle into this one.
    pub fn merge(&mut self, other: &Self) {
        for row in &other.triangle {
            self.absorb_row(row);
        }
    }
}

/// Assembles `R` from the rows returned by `row`.
///
/// `row(i)` returns `None` for rows to skip — rejected correspondences,
/// for instance.
///
/// # Determinism
///
/// The reduction tree is fully specified: blocks are cut by a fixed
/// internal constant, rows within a block are folded in increasing index
/// order, and blocks merge through a balanced binary tree in a fixed
/// order. The result is bit-for-bit identical at any thread count.
///
/// A balanced tree rather than a sequential fold: error accumulation grows
/// as `log P` for the former and as `P` for the latter.
pub fn reduce<const K: usize, F>(count: usize, row: F) -> Tsqr<K>
where
    F: Fn(usize) -> Option<[f64; K]> + Sync,
{
    if count == 0 {
        return Tsqr::new();
    }

    let chunks = count.div_ceil(CHUNK);
    let mut level: Vec<Tsqr<K>> = (0..chunks)
        .into_par_iter()
        .map(|chunk| {
            let begin = chunk * CHUNK;
            let end = ((chunk + 1) * CHUNK).min(count);
            let mut accumulator = Tsqr::new();
            for index in begin..end {
                if let Some(values) = row(index) {
                    accumulator.absorb_row(&values);
                }
            }
            accumulator
        })
        .collect();

    while level.len() > 1 {
        level = level
            .par_chunks(2)
            .map(|pair| {
                let mut left = pair[0];
                if let Some(right) = pair.get(1) {
                    left.merge(right);
                }
                left
            })
            .collect();
    }
    level[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Frobenius norm is invariant under orthogonal transformations,
    /// so `‖R‖_F` must equal `‖J‖_F`.
    ///
    /// This checks the orthogonality of `Q` without ever computing it: had
    /// the rotations been built wrongly, the norm would not survive.
    #[test]
    fn frobenius_norm_is_preserved() {
        let mut state = 12_345u64;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
        };

        let rows: Vec<[f64; 6]> = (0..5_000)
            .map(|_| [next(), next(), next(), next(), next(), next()])
            .collect();
        let source_norm: f64 = rows
            .iter()
            .flat_map(|row| row.iter())
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();

        let result = reduce::<6, _>(rows.len(), |i| Some(rows[i]));
        let triangle_norm: f64 = result
            .triangle()
            .iter()
            .flat_map(|row| row.iter())
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();

        let relative = (triangle_norm - source_norm).abs() / source_norm;
        assert!(relative < 1e-14, "‖R‖_F differs by {relative:.3e}");
    }

    /// `R` is upper triangular.
    #[test]
    fn result_is_upper_triangular() {
        let rows: Vec<[f64; 6]> = (0..100)
            .map(|i| {
                let t = i as f64;
                [t, t * 0.5, -t, 1.0, t * t * 0.01, 3.0]
            })
            .collect();
        let result = reduce::<6, _>(rows.len(), |i| Some(rows[i]));
        for (i, row) in result.triangle().iter().enumerate() {
            for value in row.iter().take(i) {
                assert_eq!(*value, 0.0, "an element below the diagonal is non-zero");
            }
        }
    }

    /// Skipping rows really skips them.
    #[test]
    fn skipped_rows_do_not_contribute() {
        let rows: Vec<[f64; 6]> = (0..1_000)
            .map(|i| [i as f64, 1.0, 2.0, 3.0, 4.0, 5.0])
            .collect();
        let all = reduce::<6, _>(rows.len(), |i| Some(rows[i]));
        let even = reduce::<6, _>(rows.len(), |i| (i % 2 == 0).then_some(rows[i]));
        assert_ne!(all, even);

        let only_even: Vec<[f64; 6]> = rows.iter().step_by(2).copied().collect();
        let direct = reduce::<6, _>(only_even.len(), |i| Some(only_even[i]));
        assert_eq!(even, direct);
    }
}
