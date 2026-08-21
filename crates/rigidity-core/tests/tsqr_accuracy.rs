//! Accuracy of the smallest singular value: `J` directly against `JᵀJ`.
//!
//! What is checked are **error growth laws**, not tuned thresholds.
//! Perturbing the matrix entries by a relative `ε` fixes `σ_min` to within
//! an absolute `ε·σ_max`, that is a relative `ε·κ`. That is the limit for
//! any algorithm working with `J`.
//!
//! Moving to `JᵀJ` squares the condition number and the limit becomes
//! `ε·κ²`. The difference shows not in a constant but in a slope: on
//! logarithmic axes one law gives slope one, the other slope two.

use nalgebra::{Matrix6, Vector6};
use rigidity_core::linalg::{reduce, singular_values};

const EPSILON: f64 = f64::EPSILON;

struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(f64::MIN_POSITIVE);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * self.unit()).cos()
    }
}

/// Builds `J = Q·Σ·Vᵀ` with prescribed singular values.
///
/// `Q` comes from Gram–Schmidt run twice: one pass leaves a departure from
/// orthogonality of order `ε·κ` of the original columns, two passes bring
/// it down to `ε`. `V` is a random orthogonal 6×6 matrix; without it the
/// singular vectors would coincide with the coordinate axes, `JᵀJ` would
/// come out nearly diagonal, and the cancellation this whole test is about
/// would never arise.
fn build(rows: usize, sigma: &[f64; 6], seed: u64) -> Vec<[f64; 6]> {
    let mut rng = Rng(seed);
    let mut columns: Vec<Vec<f64>> = (0..6)
        .map(|_| (0..rows).map(|_| rng.normal()).collect())
        .collect();

    for _ in 0..2 {
        for j in 0..6 {
            let (left, right) = columns.split_at_mut(j);
            for column in left.iter().take(j) {
                let dot: f64 = right[0].iter().zip(column).map(|(a, b)| a * b).sum();
                for (value, base) in right[0].iter_mut().zip(column) {
                    *value -= dot * base;
                }
            }
            let norm: f64 = right[0].iter().map(|v| v * v).sum::<f64>().sqrt();
            for value in right[0].iter_mut() {
                *value /= norm;
            }
        }
    }

    let mut raw = Matrix6::zeros();
    for i in 0..6 {
        for j in 0..6 {
            raw[(i, j)] = rng.normal();
        }
    }
    let rotation = raw.qr().q();

    (0..rows)
        .map(|i| {
            let scaled = Vector6::from_iterator((0..6).map(|j| columns[j][i] * sigma[j]));
            let row = rotation * scaled;
            [row[0], row[1], row[2], row[3], row[4], row[5]]
        })
        .collect()
}

fn spectrum_with_given_condition(power: i32) -> ([f64; 6], f64) {
    let smallest = 2f64.powi(-power);
    ([1.0, 0.5, 0.25, 0.125, 0.0625, smallest], 1.0 / smallest)
}

/// The error of both paths on one and the same matrix.
fn errors(power: i32) -> (f64, f64, f64) {
    let (sigma, kappa) = spectrum_with_given_condition(power);
    let rows = build(20_000, &sigma, 0xC0FFEE + power as u64);
    let smallest = sigma[5];

    let triangle = reduce::<6, _>(rows.len(), |i| Some(rows[i]));
    let from_tsqr = singular_values(triangle.triangle())[5];

    let mut hessian = [[0.0f64; 6]; 6];
    for row in &rows {
        for i in 0..6 {
            for j in 0..6 {
                hessian[i][j] += row[i] * row[j];
            }
        }
    }
    let from_normal = singular_values(&hessian)[5].sqrt();

    (
        kappa,
        (from_tsqr - smallest).abs() / smallest,
        (from_normal - smallest).abs() / smallest,
    )
}

/// The error of the `J` path stays under `ε·κ` with margin.
#[test]
fn tsqr_error_follows_epsilon_times_kappa() {
    for power in [12i32, 16, 20, 24, 28, 32, 36, 40, 44] {
        let (kappa, tsqr, _) = errors(power);
        let bound = 100.0 * EPSILON * kappa;
        assert!(
            tsqr < bound,
            "κ = {kappa:.2e}: TSQR error {tsqr:.3e} exceeded the bound {bound:.3e}"
        );
    }
}

/// The error of the `JᵀJ` path grows as `ε·κ²`.
///
/// Checked from both sides. The upper bound confirms the law; the lower
/// one confirms that the degradation really is quadratic rather than
/// linear — without it the test would also pass for a correct algorithm.
#[test]
fn normal_equations_error_follows_epsilon_times_kappa_squared() {
    for power in [12i32, 16, 20, 24] {
        let (kappa, _, normal) = errors(power);
        let scale = EPSILON * kappa * kappa;
        assert!(
            normal < 1_000.0 * scale,
            "κ = {kappa:.2e}: error {normal:.3e} above the prediction {scale:.3e}"
        );
        assert!(
            normal > 0.01 * scale,
            "κ = {kappa:.2e}: error {normal:.3e} below the prediction {scale:.3e} — \
             the degradation is not quadratic, the law has changed"
        );
    }
}

/// At the project's operating point only the `J` path works.
///
/// The ceiling from storing points as `f32` is `κ(J) ≈ 10⁷`. That is where
/// the question is settled, not in the asymptotics.
#[test]
fn only_the_direct_path_works_at_the_f32_operating_point() {
    // 2²⁴ ≈ 1.7·10⁷, the measured ceiling.
    let (kappa, tsqr, normal) = errors(24);
    assert!(
        (1e7..1e8).contains(&kappa),
        "the operating point moved: κ = {kappa:.2e}"
    );
    assert!(
        tsqr < 1e-8,
        "TSQR gives {tsqr:.3e} at the operating point — not enough for the detector"
    );
    assert!(
        normal > 1e-3,
        "the normal equations gave {normal:.3e}: if they are accurate enough \
         now, the entire motivation for TSQR needs revisiting"
    );
    assert!(
        normal / tsqr > 1e5,
        "the gap between the paths shrank to {:.1e}",
        normal / tsqr
    );
}

/// The `JᵀJ` path loses `σ_min` entirely before the direct one does.
#[test]
fn normal_equations_break_down_two_orders_earlier() {
    // κ ≈ 2.7·10⁸: the normal equations' error exceeds the value itself.
    let (_, tsqr, normal) = errors(28);
    assert!(
        normal > 1.0,
        "the normal equations are still holding: {normal:.3e}"
    );
    assert!(tsqr < 1e-6, "TSQR has already broken: {tsqr:.3e}");

    // κ ≈ 1.1·10¹²: the direct path still gives five significant digits.
    let (_, tsqr_high, _) = errors(40);
    assert!(tsqr_high < 1e-4, "TSQR at κ ≈ 10¹² gives {tsqr_high:.3e}");
}

/// `R` is bit-for-bit the same at any thread count.
///
/// The reduction tree is fixed by the block split and the balanced merge,
/// so the property follows from the construction. The test pins it down:
/// replacing the fold with `rayon::reduce` would look like a harmless
/// simplification and would quietly make `σ_min` float.
#[test]
fn tsqr_is_bit_identical_across_thread_counts() {
    let (sigma, _) = spectrum_with_given_condition(24);
    let rows = build(50_000, &sigma, 0xD0D0);

    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| reduce::<6, _>(rows.len(), |i| Some(rows[i])))
    };

    let reference = run(1);
    let expected: Vec<u64> = reference
        .triangle()
        .iter()
        .flat_map(|row| row.iter())
        .map(|v| v.to_bits())
        .collect();

    for threads in [2usize, 3, 4, 8, 16] {
        let result = run(threads);
        let actual: Vec<u64> = result
            .triangle()
            .iter()
            .flat_map(|row| row.iter())
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(actual, expected, "{threads} threads: R differs bit for bit");
    }
}
