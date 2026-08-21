//! Downsampling must produce a bit-for-bit identical result at any thread
//! count, both in values and in point order.
//!
//! This is not about convenience while debugging. These points go on to
//! assemble the Jacobian, and a floating order of summation would make the
//! smallest singular value float — that is, the degeneracy detector
//! itself.

use nalgebra::Vector3;
use rayon::ThreadPoolBuilder;
use rigidity_core::{PointCloud, voxel::voxel_downsample};

struct Lcg(u64);

impl Lcg {
    fn next_unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

/// A cloud of overlapping clusters: many points land in each cell, so the
/// order of summation inside a cell genuinely affects the result. Uniform
/// noise would weaken the test.
fn clustered_cloud(count: usize, seed: u64) -> PointCloud {
    let mut rng = Lcg(seed);
    let mut cloud = PointCloud::with_capacity(count);
    for i in 0..count {
        let center = Vector3::new(
            ((i % 17) as f64) * 0.7,
            ((i / 17 % 17) as f64) * 0.7,
            ((i / 289 % 17) as f64) * 0.7,
        );
        let jitter = Vector3::new(
            rng.next_unit() - 0.5,
            rng.next_unit() - 0.5,
            rng.next_unit() - 0.5,
        );
        cloud.push(center + jitter * 0.9);
    }
    cloud
}

/// The bit pattern of a column. Ordinary `==` on `f32` treats `0.0` and
/// `-0.0` as equal, whereas what is needed here is identity of bytes.
fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn columns_bits(cloud: &PointCloud) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let (x, y, z) = cloud.columns();
    (bits(x), bits(y), bits(z))
}

#[test]
fn downsample_is_bit_identical_across_thread_counts() {
    let cloud = clustered_cloud(60_000, 0xD00D);
    let voxel = 0.5;

    let reference = ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| voxel_downsample(&cloud, voxel).unwrap());
    let expected = columns_bits(&reference);
    assert!(
        reference.len() > 1_000,
        "downsampling collapsed the cloud to {} points — the test lost its point",
        reference.len()
    );

    for threads in [2usize, 3, 4, 8, 16] {
        let result = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| voxel_downsample(&cloud, voxel).unwrap());

        assert_eq!(
            result.len(),
            reference.len(),
            "{threads} threads: a different point count"
        );
        assert_eq!(
            columns_bits(&result),
            expected,
            "{threads} threads: the result differs bit for bit"
        );
    }
}

/// Repeating a run in the same pool must agree too — a separate case that
/// catches dependence on scheduler state.
#[test]
fn downsample_is_stable_across_repeated_runs() {
    let cloud = clustered_cloud(20_000, 0x1234);
    let first = voxel_downsample(&cloud, 0.4).unwrap();
    for _ in 0..5 {
        let again = voxel_downsample(&cloud, 0.4).unwrap();
        assert_eq!(columns_bits(&again), columns_bits(&first));
    }
}
