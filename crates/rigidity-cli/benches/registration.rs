//! Registration cost, measured separately by stage.
//!
//! Merging correspondence search and system assembly into a single number
//! is pointless: the first is decided by someone else's kd-tree, the
//! second by our own mathematics. Optimising without separating them means
//! tuning blind.

use criterion::{Criterion, criterion_group, criterion_main};
use nalgebra::{Vector3, Vector6};
use rigidity_core::icp::{IcpConfig, Kernel, point_to_plane_row, register, surface};
use rigidity_core::lie::Se3;
use rigidity_core::linalg::reduce;
use rigidity_core::{Neighbor, NeighborSearch, PointCloud};
use rigidity_scenes::{Scene, SceneKind, SceneParams};
use rigidity_spatial::KdTree;
use std::hint::black_box;

const SIZES: [usize; 3] = [10_000, 100_000, 1_000_000];

fn scene_with(points: usize) -> Scene {
    Scene::generate(
        SceneKind::Corner,
        SceneParams {
            points_per_face: points / 3,
            ..SceneParams::default()
        },
    )
}

fn offset_pose() -> Se3 {
    Se3::exp(&Vector6::new(0.021, -0.013, 0.017, 0.011, -0.007, 0.009))
}

/// Correspondence search only: transform the point, query the tree,
/// reject by distance and by normal angle.
fn correspondence_pass(
    cloud: &PointCloud,
    normals: &[Vector3<f64>],
    tree: &KdTree,
    pose: &Se3,
) -> usize {
    let mut found: Vec<Neighbor> = Vec::with_capacity(1);
    let mut accepted = 0usize;
    let rotation = *pose.rotation().matrix();
    for index in 0..cloud.len() {
        let transformed = pose.transform_point(&cloud.point(index));
        tree.knn_into(&transformed, 1, &mut found);
        let Some(nearest) = found.first() else {
            continue;
        };
        if nearest.distance_squared > 0.25 {
            continue;
        }
        let matched = nearest.index as usize;
        if (rotation * normals[index]).dot(&normals[matched]).abs() < 0.8 {
            continue;
        }
        accepted += 1;
    }
    accepted
}

/// Spreads 21 bits across every third position.
fn spread(value: u32) -> u64 {
    let mut x = u64::from(value) & 0x1F_FFFF;
    x = (x | x << 32) & 0x001F_0000_0000_FFFF;
    x = (x | x << 16) & 0x001F_0000_FF00_00FF;
    x = (x | x << 8) & 0x100F_00F0_0F00_F00F;
    x = (x | x << 4) & 0x10C3_0C30_C30C_30C3;
    x = (x | x << 2) & 0x1249_2492_4924_9249;
    x
}

/// Traversal order along a Morton curve.
fn locality_order(cloud: &PointCloud) -> Vec<u32> {
    let (min, max) = cloud.bounds().unwrap();
    let extent = (max - min).max().max(1e-12);
    let scale = (((1u32 << 21) - 1) as f64) / extent;
    let mut keys: Vec<(u64, u32)> = (0..cloud.len())
        .map(|index| {
            let p = cloud.point(index) - min;
            let code = spread((p.x * scale) as u32)
                | spread((p.y * scale) as u32) << 1
                | spread((p.z * scale) as u32) << 2;
            (code, index as u32)
        })
        .collect();
    keys.sort_unstable();
    keys.into_iter().map(|(_, index)| index).collect()
}

fn correspondence_pass_ordered(
    cloud: &PointCloud,
    normals: &[Vector3<f64>],
    tree: &KdTree,
    pose: &Se3,
    order: &[u32],
) -> usize {
    let mut found: Vec<Neighbor> = Vec::with_capacity(1);
    let mut accepted = 0usize;
    let rotation = *pose.rotation().matrix();
    for &index in order {
        let index = index as usize;
        let transformed = pose.transform_point(&cloud.point(index));
        tree.knn_into(&transformed, 1, &mut found);
        let Some(nearest) = found.first() else {
            continue;
        };
        if nearest.distance_squared > 0.25 {
            continue;
        }
        let matched = nearest.index as usize;
        if (rotation * normals[index]).dot(&normals[matched]).abs() < 0.8 {
            continue;
        }
        accepted += 1;
    }
    accepted
}

fn correspondences(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("correspondence search");
    group.sample_size(20);
    for size in SIZES {
        let scene = scene_with(size);
        let tree = KdTree::build(&scene.cloud).unwrap();
        let pose = offset_pose();
        let order = locality_order(&scene.cloud);
        let converged = Se3::identity();
        let reduced = rigidity_core::voxel::voxel_downsample(&scene.cloud, 0.02).unwrap();
        let reduced_tree = KdTree::build(&reduced).unwrap();
        let reduced_normals = vec![Vector3::z(); reduced.len()];

        group.bench_function(format!("{size}/converged pose"), |bencher| {
            bencher.iter(|| {
                black_box(correspondence_pass(
                    &scene.cloud,
                    &scene.normals,
                    &tree,
                    &converged,
                ))
            })
        });
        group.bench_function(
            format!("{size}/downsampled to {}", reduced.len()),
            |bencher| {
                bencher.iter(|| {
                    black_box(correspondence_pass(
                        &reduced,
                        &reduced_normals,
                        &reduced_tree,
                        &pose,
                    ))
                })
            },
        );
        group.bench_function(format!("{size}/generation order"), |bencher| {
            bencher.iter(|| {
                black_box(correspondence_pass(
                    &scene.cloud,
                    &scene.normals,
                    &tree,
                    &pose,
                ))
            })
        });
        group.bench_function(format!("{size}/Morton order"), |bencher| {
            bencher.iter(|| {
                black_box(correspondence_pass_ordered(
                    &scene.cloud,
                    &scene.normals,
                    &tree,
                    &pose,
                    &order,
                ))
            })
        });
    }
    group.finish();
}

/// System assembly from ready correspondences: the two paths, `JᵀJ` and
/// TSQR.
fn assembly(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("system assembly");
    group.sample_size(20);
    for size in SIZES {
        let scene = scene_with(size);
        let rows: Vec<[f64; 6]> = (0..scene.len())
            .map(|index| {
                let row = point_to_plane_row(&scene.cloud.point(index), &scene.normals[index]);
                [row[0], row[1], row[2], row[3], row[4], row[5]]
            })
            .collect();

        group.bench_function(format!("{size}/normal equations"), |bencher| {
            bencher.iter(|| {
                let mut hessian = [[0.0f64; 6]; 6];
                for row in &rows {
                    for i in 0..6 {
                        for j in 0..6 {
                            hessian[i][j] += row[i] * row[j];
                        }
                    }
                }
                black_box(hessian)
            })
        });
        group.bench_function(format!("{size}/tsqr"), |bencher| {
            bencher.iter(|| black_box(reduce::<6, _>(rows.len(), |i| Some(rows[i]))))
        });
    }
    group.finish();
}

/// Memory layout: three separate arrays against an array of triples.
///
/// The answer is not obvious. Separate arrays mean three read streams but
/// let unused attributes stay untouched; an array of triples gives one
/// stream but drags everything along. Measure, do not argue.
fn memory_layout(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("layout");
    group.sample_size(30);
    for size in [100_000usize, 1_000_000] {
        let scene = scene_with(size);
        let pose = offset_pose();
        let interleaved: Vec<[f32; 3]> = (0..scene.len())
            .map(|i| {
                let p = scene.cloud.local(i);
                [p.x as f32, p.y as f32, p.z as f32]
            })
            .collect();

        group.bench_function(format!("{size}/separate arrays"), |bencher| {
            bencher.iter(|| {
                let mut total = Vector3::zeros();
                for index in 0..scene.cloud.len() {
                    total += pose.transform_point(&scene.cloud.point(index));
                }
                black_box(total)
            })
        });
        group.bench_function(format!("{size}/array of triples"), |bencher| {
            bencher.iter(|| {
                let mut total = Vector3::zeros();
                for point in &interleaved {
                    let position = Vector3::new(
                        f64::from(point[0]),
                        f64::from(point[1]),
                        f64::from(point[2]),
                    );
                    total += pose.transform_point(&position);
                }
                black_box(total)
            })
        });
    }
    group.finish();
}

/// A full registration to convergence.
fn full_registration(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("full registration");
    group.sample_size(10);
    for size in [10_000usize, 100_000] {
        let scene = scene_with(size);
        let truth = offset_pose();
        let inverse = truth.inverse();
        let rotation = *inverse.rotation().matrix();
        let mut source = PointCloud::with_capacity(scene.len());
        let mut source_normals = Vec::with_capacity(scene.len());
        for index in 0..scene.len() {
            source.push(inverse.transform_point(&scene.points[index]));
            source_normals.push(rotation * scene.normals[index]);
        }
        let tree = KdTree::build(&scene.cloud).unwrap();
        let config = IcpConfig {
            kernel: Kernel::Squared,
            max_correspondence_distance: 0.5,
            ..IcpConfig::default()
        };

        group.bench_function(format!("{size}"), |bencher| {
            bencher.iter(|| {
                black_box(register(
                    &surface(&source, &source_normals),
                    &surface(&scene.cloud, &scene.normals),
                    &tree,
                    Se3::identity(),
                    &config,
                ))
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    correspondences,
    assembly,
    memory_layout,
    full_registration
);
criterion_main!(benches);
