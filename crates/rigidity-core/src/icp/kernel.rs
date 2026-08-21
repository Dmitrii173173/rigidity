//! Robust loss functions.
//!
//! A squared loss assumes Gaussian errors. Real clouds do not supply them:
//! wrong correspondences near boundaries, points from another surface,
//! reflections. One such point at a ten-sigma residual contributes as much
//! to the squared sum as a hundred well-behaved ones, and drags the
//! solution with it.
//!
//! A robust kernel bounds that contribution. What gets minimised is
//! `Σ ρ(e)` rather than `Σ e²`, solved by IRLS — iteratively reweighted
//! least squares with weight `w(e) = ψ(e)/e`, where `ψ = dρ/de`.

/// A loss function.
///
/// Each kernel's parameter is a scale in units of the residual, that is,
/// in metres. It sets the boundary of a "normal" error; three standard
/// deviations of the sensor noise is a reasonable starting point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Kernel {
    /// Squared: `ρ(e) = e²/2`. Not robust; kept for comparison.
    Squared,
    /// Huber: quadratic near zero, linear outside.
    ///
    /// An outlier's contribution still grows, but only linearly. Convex,
    /// so it creates no local minima.
    Huber(f64),
    /// Cauchy: the contribution grows logarithmically.
    ///
    /// Suppresses harder than Huber, at the cost of being non-convex.
    Cauchy(f64),
    /// Tukey: an outlier's contribution is exactly zero past the
    /// threshold.
    ///
    /// The harshest of these: a point beyond the threshold has no
    /// influence at all. It demands a decent initial guess.
    Tukey(f64),
    /// Geman–McClure: smooth saturation without a hard threshold.
    GemanMcClure(f64),
}

impl Kernel {
    /// The IRLS weight `w(e) = ψ(e)/e`.
    ///
    /// At zero every kernel gives one: near the origin all of them agree
    /// with the squared loss.
    pub fn weight(self, residual: f64) -> f64 {
        let e = residual.abs();
        match self {
            Self::Squared => 1.0,
            Self::Huber(delta) => {
                if e <= delta {
                    1.0
                } else {
                    delta / e
                }
            }
            Self::Cauchy(c) => {
                let t = e / c;
                1.0 / (1.0 + t * t)
            }
            Self::Tukey(c) => {
                if e <= c {
                    let t = e / c;
                    let s = 1.0 - t * t;
                    s * s
                } else {
                    0.0
                }
            }
            Self::GemanMcClure(c) => {
                let denominator = c * c + e * e;
                (c * c * c * c) / (denominator * denominator)
            }
        }
    }

    /// The value of `ρ(e)`.
    ///
    /// Needed by the LM step-acceptance test: what must be compared is the
    /// robust cost rather than the sum of squares, or a step that is right
    /// by the kernel's own measure gets rejected.
    pub fn loss(self, residual: f64) -> f64 {
        let e = residual.abs();
        match self {
            Self::Squared => 0.5 * e * e,
            Self::Huber(delta) => {
                if e <= delta {
                    0.5 * e * e
                } else {
                    delta * (e - 0.5 * delta)
                }
            }
            Self::Cauchy(c) => {
                let t = e / c;
                // ln_1p rather than ln(1 + x): at a small residual
                // `1 + t²` loses significant digits. At e = 10⁻⁶ and c = 1
                // the direct form has relative error 9·10⁻⁵ — four digits
                // out of sixteen.
                0.5 * c * c * (t * t).ln_1p()
            }
            Self::Tukey(c) => {
                let limit = c * c / 6.0;
                if e <= c {
                    let u = (e / c) * (e / c);
                    // `1 − (1 − u)³` expanded into `u·(3 − 3u + u²)`. The
                    // direct form subtracts nearly equal numbers and loses
                    // as many digits as the naive Cauchy does.
                    limit * u * (3.0 - 3.0 * u + u * u)
                } else {
                    limit
                }
            }
            Self::GemanMcClure(c) => {
                let e2 = e * e;
                0.5 * c * c * e2 / (c * c + e2)
            }
        }
    }

    /// The scale parameter, where one exists.
    pub fn scale(self) -> Option<f64> {
        match self {
            Self::Squared => None,
            Self::Huber(c) | Self::Cauchy(c) | Self::Tukey(c) | Self::GemanMcClure(c) => Some(c),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KERNELS: [Kernel; 5] = [
        Kernel::Squared,
        Kernel::Huber(1.0),
        Kernel::Cauchy(1.0),
        Kernel::Tukey(1.0),
        Kernel::GemanMcClure(1.0),
    ];

    /// Near zero every kernel agrees with the squared loss: `w → 1`,
    /// `ρ → e²/2`.
    #[test]
    fn all_kernels_agree_near_zero() {
        for kernel in KERNELS {
            assert!((kernel.weight(0.0) - 1.0).abs() < 1e-12, "{kernel:?}");
            let e = 1e-6;
            assert!((kernel.weight(e) - 1.0).abs() < 1e-10, "{kernel:?}");
            assert!(
                (kernel.loss(e) - 0.5 * e * e).abs() < 1e-20,
                "{kernel:?}: ρ = {}",
                kernel.loss(e)
            );
        }
    }

    /// The weight must equal `ψ(e)/e`, with `ψ` the numerical derivative
    /// of `ρ`.
    ///
    /// A weight that disagrees with its loss is a classic bug: IRLS still
    /// converges, just not to the minimum of the `ρ` you declared.
    #[test]
    fn weight_is_the_derivative_of_loss_over_residual() {
        const H: f64 = 1e-6;
        for kernel in KERNELS {
            for e in [0.1, 0.5, 0.9, 1.5, 3.0, 10.0] {
                let psi = (kernel.loss(e + H) - kernel.loss(e - H)) / (2.0 * H);
                let expected = psi / e;
                let actual = kernel.weight(e);
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "{kernel:?} at e = {e}: weight {actual}, but ψ/e = {expected}"
                );
            }
        }
    }

    /// Robust kernels bound an outlier's influence; the squared loss does
    /// not.
    #[test]
    fn robust_kernels_bound_outlier_influence() {
        let far = 1e4;
        assert!(Kernel::Squared.loss(far) > 1e7);
        assert!(Kernel::Huber(1.0).loss(far) < 1e5);
        assert!(Kernel::Cauchy(1.0).loss(far) < 20.0);
        assert_eq!(Kernel::Tukey(1.0).loss(far), 1.0 / 6.0);
        assert!(Kernel::GemanMcClure(1.0).loss(far) < 0.51);

        assert_eq!(Kernel::Tukey(1.0).weight(1.5), 0.0);
        assert!(Kernel::Huber(1.0).weight(far) < 1e-3);
    }

    /// The weight is non-increasing: farther is never more important.
    #[test]
    fn weight_is_non_increasing() {
        for kernel in KERNELS {
            let mut previous = f64::INFINITY;
            for step in 0..200 {
                let weight = kernel.weight(step as f64 * 0.05);
                assert!(weight <= previous + 1e-12, "{kernel:?} at step {step}");
                previous = weight;
            }
        }
    }
}
