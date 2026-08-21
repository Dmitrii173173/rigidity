//! Numerical linear algebra.
//!
//! Two building blocks carry the conditioning analysis: a decomposition of
//! a tall-skinny matrix that never forms `JᵀJ`, and singular values
//! computed to relative rather than absolute accuracy.

mod jacobi;
mod tsqr;

pub use jacobi::{Decomposition, condition_number, decompose, singular_values};
pub use tsqr::{Tsqr, reduce};
