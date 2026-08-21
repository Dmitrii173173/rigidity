//! A deterministic pseudo-random generator.
//!
//! Home-grown rather than `rand`: the scenes are a measuring standard, and
//! reproducibility matters more here than distribution quality. An
//! external generator may change algorithm in a minor release, and every
//! number recorded in the tests would stop matching.
//!
//! The algorithm is splitmix64: ten lines, good bit quality, period 2⁶⁴.

/// A generator with explicit state.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Creates a generator from a seed.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform number in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        // 53 bits of mantissa: no more fits into an f64.
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A uniform number in `[-1, 1)`.
    pub fn symmetric(&mut self) -> f64 {
        self.unit() * 2.0 - 1.0
    }

    /// A normal number with standard deviation `sigma`.
    ///
    /// The Box–Muller transform. The second value of the pair is thrown
    /// away: keeping it would mean state that depends on call history, and
    /// the order of calls would start to affect the result.
    pub fn normal(&mut self, sigma: f64) -> f64 {
        let u1 = self.unit().max(f64::MIN_POSITIVE);
        let u2 = self.unit();
        sigma * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}
