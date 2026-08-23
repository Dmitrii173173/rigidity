//! The weighting this project shipped until S6, kept so that what it did
//! can still be run.
//!
//! Deleting it would have left two numbers in the README with nothing
//! behind them: it wins by a hundred and twenty fold on the scene built to
//! show it, and loses on every real survey it was tried on. Both halves of
//! that sentence are the point, and a claim whose evidence has been deleted
//! is a claim on trust. So the code stays here — in a crate that is never
//! published, next to the sector crop and the probabilistic method, where
//! measurement apparatus belongs — and out of the library, which should not
//! ship a weighting its own author has measured losing.

use rigidity_core::nalgebra::{Matrix6, Vector6};
use rigidity_core::observability::{Conditioning, Observability, ObservabilityCriteria};

/// The weighting S6 rejected.
///
/// This is what `calibrated_information` did until S6: every direction whose
/// predicted spread exceeded the survey's required accuracy was dropped
/// from the sum outright. The numbers in `PLAN.md` and in the README are
/// this function's, on both sides of the story: `eth_survey` measures what
/// it loses by on real surveys, and `degenerate_leg` keeps the hundred and
/// twenty fold it wins by on the scene that was built for it.
pub fn thresholded_information(
    conditioning: &Conditioning,
    criteria: &ObservabilityCriteria,
) -> Matrix6<f64> {
    let mut to_normalised = Matrix6::zeros();
    for axis in 0..6 {
        let mut basis = Vector6::zeros();
        basis[axis] = 1.0;
        to_normalised.set_column(axis, &conditioning.to_normalised(basis));
    }

    let spreads = conditioning.uncertainty(criteria.noise_sigma);
    let observable = conditioning.classify(criteria);
    let mut normalised = Matrix6::zeros();
    for index in 0..6 {
        if observable[index] != Observability::High {
            continue;
        }
        let spread = spreads[index];
        if !(spread.is_finite() && spread > 0.0) {
            continue;
        }
        let direction = conditioning.direction(index);
        normalised += direction * direction.transpose() / (spread * spread);
    }

    to_normalised.transpose() * normalised * to_normalised
}
