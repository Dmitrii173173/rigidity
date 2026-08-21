//! The cost of the two assembly paths: `R` through TSQR against an
//! explicit `JᵀJ`.
//!
//! The accuracy of one of them is measured elsewhere
//! (`tests/tsqr_accuracy.rs`); this is the price. The question is not
//! idle: TSQR performs six Givens rotations per row against a single
//! symmetric outer product, so losing on speed is expected — and worth
//! knowing in numbers rather than assuming.

use criterion::{Criterion, criterion_group, criterion_main};
use rigidity_core::linalg::reduce;
use std::hint::black_box;

fn rows(count: usize) -> Vec<[f64; 6]> {
    let mut state = 0x5EEDu64;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    };
    (0..count)
        .map(|_| [next(), next(), next(), next(), next(), next()])
        .collect()
}

/// Explicit assembly of the normal equations, which is what ordinary ICP
/// does.
fn normal_equations(rows: &[[f64; 6]]) -> [[f64; 6]; 6] {
    let mut hessian = [[0.0f64; 6]; 6];
    for row in rows {
        for i in 0..6 {
            for j in 0..6 {
                hessian[i][j] += row[i] * row[j];
            }
        }
    }
    hessian
}

fn assembly(criterion: &mut Criterion) {
    for count in [10_000usize, 100_000, 1_000_000] {
        let data = rows(count);
        let mut group = criterion.benchmark_group(format!("assembly/{count}"));
        group.bench_function("tsqr", |bencher| {
            bencher.iter(|| black_box(reduce::<6, _>(data.len(), |i| Some(data[i]))))
        });
        group.bench_function("normal equations", |bencher| {
            bencher.iter(|| black_box(normal_equations(&data)))
        });
        group.finish();
    }
}

criterion_group!(benches, assembly);
criterion_main!(benches);
