//! Checking the prediction on the real ETH ASL data.
//!
//! # The question
//!
//! On synthetic scenes the predicted spread matched the empirical one with
//! a median of 0.993. Those scenes satisfy the model's premises by
//! construction: exact surfaces, correct correspondences, isotropic
//! Gaussian noise. None of that holds here.
//!
//! # Method
//!
//! The dataset provides a sequence of scans with poses measured by a
//! theodolite to millimetre accuracy. For each adjacent pair: register,
//! compute the conditioning at the solution found, and compare the
//! **actual** error against the theodolite with the predicted spread along
//! each degree of freedom.
//!
//! Two claims are under test:
//! 1. where the detector promises reliability, the error fits the
//!    prediction;
//! 2. where it warns, the error really is large.
//!
//! The second matters more: false confidence is worse than a false
//! alarm.

use std::path::{Path, PathBuf};

use nalgebra::{Matrix4, Vector3, Vector6};
use rigidity_core::PointCloud;
use rigidity_core::icp::{IcpConfig, Kernel, register, surface};
use rigidity_core::lie::Se3;
use rigidity_core::normals::estimate_normals;
use rigidity_core::observability::{Correspondence, Observability, ObservabilityCriteria, analyse};
use rigidity_core::voxel::voxel_downsample;
use rigidity_harness::crop_sector;
use rigidity_io::{read_csv, read_poses};
use rigidity_spatial::KdTree;

const VOXEL: f64 = 0.05;
const NEIGHBOURS: usize = 16;
const MAX_DISTANCE: f64 = 1.0;
/// The sensor's stated accuracy: Hokuyo UTM-30LX, about 3 cm.
const NOISE_SIGMA: f64 = 0.03;
/// Required pose accuracy, used for the classification.
const TOLERANCE: f64 = 0.05;

fn scan_files(directory: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(directory)
        .unwrap_or_else(|e| panic!("cannot open {}: {e}", directory.display()))
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            // Scans only: GPS and gravity-vector files sit alongside.
            name.starts_with("Hokuyo") && name.ends_with(".csv")
        })
        .collect();
    // Order by the number in the name, not lexicographically: otherwise
    // Hokuyo_10 would come before Hokuyo_2.
    files.sort_by_key(|path| {
        let name = path.file_stem().and_then(|n| n.to_str()).unwrap_or("");
        name.rsplit('_')
            .next()
            .and_then(|digits| digits.parse::<usize>().ok())
            .unwrap_or(usize::MAX)
    });
    files
}

fn prepare(path: &Path) -> (PointCloud, Vec<Vector3<f64>>, KdTree) {
    let raw = read_csv(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let sector: Option<f64> = std::env::var("RIGIDITY_SECTOR")
        .ok()
        .and_then(|v| v.parse().ok());
    let raw = match sector {
        Some(degrees) => crop_sector(&raw, 0.0, degrees.to_radians()),
        None => raw,
    };
    let cloud = voxel_downsample(&raw, VOXEL).unwrap();
    let tree = KdTree::build(&cloud).unwrap();
    let normals = estimate_normals(&cloud, &tree, NEIGHBOURS);
    (cloud, normals, tree)
}

fn main() {
    let directory = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "datasets/hauptgebaude/local_frame".to_string());
    let directory = Path::new(&directory);
    let pairs: usize = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);

    let poses =
        read_poses(&directory.join("pose_scanner_leica.csv")).expect("the pose file is readable");
    let files = scan_files(directory);
    println!("scans: {}, poses: {}", files.len(), poses.len());
    assert!(files.len() >= 2, "at least two scans are needed");

    // The empirical correction measured here: the formula understates the
    // spread by roughly 15–19×, because it treats measurements as
    // independent. The correction is applied through σ of the noise, which
    // enters the prediction linearly.
    let calibration: f64 = std::env::var("RIGIDITY_CALIBRATION")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);
    let effective_sigma = NOISE_SIGMA * calibration;
    let criteria = ObservabilityCriteria {
        noise_sigma: effective_sigma,
        tolerance: TOLERANCE,
    };
    if calibration != 1.0 {
        println!("empirical correction: ×{calibration:.0}");
    }
    let config = IcpConfig {
        kernel: Kernel::Huber(0.1),
        max_correspondence_distance: MAX_DISTANCE,
        min_normal_cosine: 0.0,
        max_iterations: 60,
        ..IcpConfig::default()
    };

    println!(
        "\n{:>4} {:>7} {:>9} {:>8} {:>9} {:>4}  per direction: predicted → actual",
        "pair", "points", "κ", "RMSE, m", "|error|", "LOW"
    );

    let count = pairs.min(files.len() - 1);
    let mut agree = 0usize;
    let mut total = 0usize;
    let mut warned_and_wrong = 0usize;
    let mut confident_and_wrong = 0usize;
    let mut diverged = 0usize;
    let mut ratios: Vec<f64> = Vec::new();
    // Converged pairs kept separately. A convergence failure and an
    // understated uncertainty are two different failures and must not be
    // mixed.
    let mut converged_ratios: Vec<f64> = Vec::new();
    // (predicted, actual) pairs within each scene, to test the
    // comparative ability: even with the absolute magnitude off, the
    // ordering of directions may survive.
    let mut per_pair_rank: Vec<f64> = Vec::new();
    /// The RMSE threshold separating converged pairs from failed ones.
    const CONVERGED_RMSE: f64 = 0.08;

    for index in 0..count {
        let (source, source_normals, _) = prepare(&files[index + 1]);
        let (target, target_normals, tree) = prepare(&files[index]);

        // The true relative transform, from the theodolite.
        let truth =
            Se3::from_matrix_unchecked(poses[index].try_inverse().unwrap() * poses[index + 1]);

        let result = register(
            &surface(&source, &source_normals),
            &surface(&target, &target_normals),
            &tree,
            Se3::identity(),
            &config,
        );

        let error = (truth * result.pose.inverse()).log();

        let analysis = analyse(source.len(), config.kernel, |i| {
            let point = result.pose.transform_point(&source.point(i));
            let mut found = Vec::with_capacity(1);
            use rigidity_core::NeighborSearch;
            tree.knn_into(&point, 1, &mut found);
            let nearest = found.first()?;
            if nearest.distance_squared > MAX_DISTANCE * MAX_DISTANCE {
                return None;
            }
            let matched = nearest.index as usize;
            Some(Correspondence {
                point,
                normal: target_normals[matched],
                residual: target_normals[matched].dot(&(point - target.point(matched))),
            })
        })
        .expect("there are correspondences");

        let conditioning = &analysis.conditioning;
        let states = conditioning.classify(&criteria);
        let predicted = conditioning.uncertainty(effective_sigma);
        let low = states.iter().filter(|s| **s == Observability::Low).count();

        let mut cells = Vec::new();
        for axis in 0..6 {
            let actual = conditioning.component(axis, error).abs();
            let bound = predicted[axis].max(TOLERANCE);
            total += 1;
            if actual <= bound {
                agree += 1;
            }
            if states[axis] == Observability::Low && actual > TOLERANCE {
                warned_and_wrong += 1;
            }
            if states[axis] == Observability::High && actual > 3.0 * TOLERANCE {
                confident_and_wrong += 1;
            }
            if predicted[axis].is_finite() && predicted[axis] > 0.0 {
                ratios.push(actual / predicted[axis]);
                if result.rmse < CONVERGED_RMSE {
                    converged_ratios.push(actual / predicted[axis]);
                }
            }
            cells.push(format!(
                "{:.0e}→{:.0e}{}",
                predicted[axis],
                actual,
                match states[axis] {
                    Observability::High => "",
                    Observability::Medium => "~",
                    Observability::Low => "!",
                }
            ));
        }

        // Spearman rank correlation within the pair: six directions,
        // prediction against fact.
        {
            let mut items: Vec<(f64, f64)> = (0..6)
                .filter(|axis| predicted[*axis].is_finite())
                .map(|axis| (predicted[axis], conditioning.component(axis, error).abs()))
                .collect();
            if items.len() >= 4 {
                let rank = |values: &mut Vec<(usize, f64)>| {
                    values.sort_by(|a, b| a.1.total_cmp(&b.1));
                    let mut ranks = vec![0.0; values.len()];
                    for (position, (index, _)) in values.iter().enumerate() {
                        ranks[*index] = position as f64;
                    }
                    ranks
                };
                let mut left: Vec<(usize, f64)> =
                    items.iter().enumerate().map(|(i, v)| (i, v.0)).collect();
                let mut right: Vec<(usize, f64)> =
                    items.iter().enumerate().map(|(i, v)| (i, v.1)).collect();
                let (a, b) = (rank(&mut left), rank(&mut right));
                let n = a.len() as f64;
                let sum: f64 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
                per_pair_rank.push(1.0 - 6.0 * sum / (n * (n * n - 1.0)));
            }
            items.clear();
        }

        let magnitude = error.fixed_rows::<3>(0).norm();
        println!(
            "{:>4} {:>7} {:>9.2e} {:>8.4} {:>9.3e} {:>4}  {}",
            index,
            source.len(),
            conditioning.condition_number(),
            result.rmse,
            magnitude,
            low,
            cells.join(" ")
        );
        if magnitude > 0.3 {
            diverged += 1;
        }
    }

    println!(
        "\ndirections in total {total}, within the prediction {agree} ({:.0} %)",
        100.0 * agree as f64 / total as f64
    );
    println!("warned and was right: {warned_and_wrong}");
    println!("was confident and wrong: {confident_and_wrong}");
    println!("pairs with error above 0.3 m (convergence failure): {diverged} of {count}");
    let report = |name: &str, values: &mut Vec<f64>| {
        values.sort_by(f64::total_cmp);
        if values.is_empty() {
            return;
        }
        let quantile = |q: f64| values[((values.len() - 1) as f64 * q) as usize];
        println!(
            "{name}: directions {}, median {:.0}×, quartiles {:.0}×…{:.0}×, \
maximum {:.0}×",
            values.len(),
            quantile(0.5),
            quantile(0.25),
            quantile(0.75),
            values[values.len() - 1]
        );
        // By what factor the effective number of measurements is
        // overstated: the error falls as 1/√N, so understating by k means
        // there are k² fewer independent measurements than claimed.
        let median = quantile(0.5);
        println!(
            "  effective measurements fewer than claimed by a factor of {:.0}",
            median * median
        );
    };
    per_pair_rank.sort_by(f64::total_cmp);
    if !per_pair_rank.is_empty() {
        let mean: f64 = per_pair_rank.iter().sum::<f64>() / per_pair_rank.len() as f64;
        println!(
            "\nrank correlation of prediction with fact within a pair: \
mean {mean:+.2}, median {:+.2} (0 means the prediction does not \
discriminate)",
            per_pair_rank[per_pair_rank.len() / 2]
        );
    }
    println!("\nratio of actual error to predicted:");
    report("  all pairs", &mut ratios);
    report("  converged only", &mut converged_ratios);
    let _: Vector6<f64> = Vector6::zeros();
    let _: Matrix4<f64> = Matrix4::identity();
}
