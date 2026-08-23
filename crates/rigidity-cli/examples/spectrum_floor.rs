//! Where the smallest singular value comes from: the scene, or our normals.
//!
//! # The question
//!
//! S5 found that on real scans the predicted spreads form a continuum,
//! where the synthetic gates had a gap of four orders. A threshold works in
//! a gap and cuts a cloud in half in a continuum, so which of the two the
//! data has decides whether thresholding can work at all.
//!
//! There is a mechanism that would close a gap without the scene having
//! anything to do with it. Normals come from a plane fitted to `k`
//! neighbours of a cloud that has already been voxelised, and a normal off
//! by an angle `ε` puts a component of size `ε` into a direction the
//! geometry does not constrain. An exactly unobservable direction then
//! reads as `σ_min/σ_max ≈ ε` instead of zero — a floor that belongs to the
//! estimator, not to the building.
//!
//! # Method
//!
//! Register each pair once at the reference settings, then hold that pose
//! and recompute the conditioning at every `(voxel, k)`. The spectrum is a
//! property of the geometry at a pose, so holding the pose is what makes
//! the settings the only thing that varies. Beside it, `ε` itself, measured
//! as the median disagreement between the normal at `k` and the normal at
//! `2k` — the same points, twice the averaging.
//!
//! If `σ_min/σ_max` follows `ε` down as `k` grows, the floor is ours and
//! better normals lift it. If it sits still while `ε` moves by a factor of
//! three, the floor belongs to the scene and no estimator will move it.
//!
//! ```text
//! cargo run --release -p rigidity-cli --example spectrum_floor -- <dir> [pairs]
//! RIGIDITY_SECTOR=60 …   # the cropped view, where there is a floor to find
//! ```

use std::path::PathBuf;

use nalgebra::{Matrix3, SymmetricEigen, Vector3};
use rigidity_core::PointCloud;
use rigidity_core::lie::Se3;
use rigidity_core::neighbors::NeighborSearch;
use rigidity_harness::crop_sector;
use rigidity_pipeline::{
    PrepareParams, Prepared, RegisterParams, analyse_registration, prepare_cloud,
    register_pair_observed,
};

const REFERENCE_VOXEL: f64 = 0.05;
const REFERENCE_K: usize = 16;
const VOXELS: [f64; 3] = [0.02, 0.05, 0.10];
const NEIGHBOURS: [usize; 4] = [8, 16, 32, 64];
/// The ball the reference normal is fitted over, metres. Ten voxels at the
/// reference setting: large enough that its own error is negligible, small
/// enough that a wall is still flat across it.
const REFERENCE_RADIUS: f64 = 0.5;

fn main() {
    let mut args = std::env::args().skip(1);
    let directory = PathBuf::from(args.next().unwrap_or_else(|| {
        eprintln!("usage: spectrum_floor <directory> [pairs]");
        std::process::exit(2);
    }));
    let pairs: usize = args.next().and_then(|text| text.parse().ok()).unwrap_or(6);
    let sector = std::env::var("RIGIDITY_SECTOR")
        .ok()
        .and_then(|text| text.parse::<f64>().ok())
        .map(f64::to_radians);
    match sector {
        Some(half) => println!("{pairs} pairs, cropped to ±{:.0}°", half.to_degrees()),
        None => println!("{pairs} pairs, the whole 360°"),
    }

    let params = RegisterParams::default();
    let scans: Vec<PointCloud> = (0..=pairs)
        .map(|index| {
            let path = directory.join(format!("Hokuyo_{index}.csv"));
            let raw = rigidity_io::read(&path).unwrap_or_else(|error| {
                eprintln!("{}: {error}", path.display());
                std::process::exit(1);
            });
            match sector {
                Some(half) => crop_sector(&raw, 0.0, half),
                None => raw,
            }
        })
        .collect();

    // The poses, found once at the reference settings and then held. Each
    // leg starts from the one before it, as the survey does.
    let reference: Vec<Prepared> = scans
        .iter()
        .map(|cloud| at(cloud, REFERENCE_VOXEL, REFERENCE_K))
        .collect();
    let mut poses = Vec::new();
    let mut carried = Se3::identity();
    for index in 0..pairs {
        let result = register_pair_observed(
            &reference[index + 1],
            &reference[index],
            carried,
            &params,
            |_| std::ops::ControlFlow::Continue(()),
        );
        carried = result.pose;
        poses.push(result.pose);
    }

    println!("\n  median over {pairs} pairs, at the pose the reference settings found\n");
    println!("  voxel     k     σ_min/σ_max         ε        planar");
    for voxel in VOXELS {
        for k in NEIGHBOURS {
            let prepared: Vec<Prepared> = scans.iter().map(|cloud| at(cloud, voxel, k)).collect();
            let mut ratios = Vec::new();
            let mut errors = Vec::new();
            let mut flat_share = 0.0;
            for index in 0..pairs {
                let analysis = analyse_registration(
                    &prepared[index + 1],
                    &prepared[index],
                    &poses[index],
                    &params,
                );
                if let Ok(analysis) = analysis {
                    ratios.push(1.0 / analysis.conditioning.condition_number());
                }
                let (error, flat) = normal_error(&prepared[index], REFERENCE_RADIUS);
                errors.push(error);
                flat_share += flat;
            }
            let ratio = median(&mut ratios);
            let error = median(&mut errors);
            let flat = flat_share / pairs.max(1) as f64;
            // A scene with no flat half-metre in it — a narrow sector of a
            // corridor is one — has no ε to report, and a number would be
            // worse than the dash. The share beside it says why.
            let epsilon = if error.is_finite() {
                format!("{:6.2}°", error.to_degrees())
            } else {
                "     —".to_string()
            };
            println!(
                "  {voxel:.2} m  {k:3}     {ratio:.3e}  (log10 {:5.2})   {epsilon}   {:4.0}% of samples on a plane",
                ratio.log10(),
                flat * 100.0
            );
        }
        println!();
    }
}

/// The same cloud, prepared at these settings.
fn at(cloud: &PointCloud, voxel: f64, neighbours: usize) -> Prepared {
    prepare_cloud(cloud, &PrepareParams { voxel, neighbours }).expect("the cloud prepares")
}

/// ε: how far the `k`-neighbourhood normal is from the surface's own,
/// measured the way the plan specifies rather than the way that was easy.
///
/// The first version of this compared the normal at `k` with the normal at
/// `2k` and reported the median disagreement. That number is not an error.
/// Two nested neighbourhoods share whatever the surface's curvature and the
/// sampling do to both, and the difference cancels it, so the estimate is a
/// lower bound that saturates — it stopped falling at 5.4° however large
/// `k` grew. It was also taken over every point, including the vegetation
/// on the plain and the corners in the corridor, where the normal genuinely
/// differs from its neighbours' and calling that an error is calling the
/// scene a mistake. Contaminated in both directions at once, which is to
/// say it estimated nothing in particular.
///
/// What is measured here instead: a reference normal fitted over a ball of
/// `RADIUS`, and only where that ball really is a plane — its off-plane
/// RMS within the sensor's own noise. The reference is good to about a
/// fifth of a degree at these densities (some three hundred points over
/// half a metre), which is two orders under what is being measured, so it
/// can be read as a truth without saying so too loudly.
///
/// Returns the median angle and the share of sampled points that stood on a
/// plane at all — the second number decides how much the first is about.
fn normal_error(prepared: &Prepared, radius: f64) -> (f64, f64) {
    /// How flat a ball has to be before its normal counts as one number.
    /// The sensor's own noise: a patch that departs from its plane by less
    /// than the instrument's error is a plane as far as this instrument can
    /// tell.
    const FLATNESS: f64 = 0.03;
    /// Below this a ball has too few points to fit a plane worth trusting.
    const ENOUGH: usize = 32;

    let mut angles: Vec<f64> = Vec::new();
    let mut planar = 0usize;
    let mut sampled = 0usize;
    let mut found = Vec::new();
    for index in (0..prepared.cloud.len()).step_by(8) {
        sampled += 1;
        let query = prepared.cloud.point(index);
        prepared.tree.radius_into(&query, radius, &mut found);
        if found.len() < ENOUGH {
            continue;
        }
        let count = found.len() as f64;
        let mut centroid = Vector3::zeros();
        for neighbour in &found {
            centroid += prepared.cloud.point(neighbour.index as usize);
        }
        centroid /= count;
        let mut covariance = Matrix3::zeros();
        for neighbour in &found {
            let delta = prepared.cloud.point(neighbour.index as usize) - centroid;
            covariance += delta * delta.transpose();
        }
        let eigen = SymmetricEigen::new(covariance / count);
        let mut smallest = 0;
        for axis in 1..3 {
            if eigen.eigenvalues[axis] < eigen.eigenvalues[smallest] {
                smallest = axis;
            }
        }
        // The smallest eigenvalue of the covariance is the mean square
        // departure from the plane, so its root is that departure in metres.
        if eigen.eigenvalues[smallest].max(0.0).sqrt() > FLATNESS {
            continue;
        }
        planar += 1;
        let reference: Vector3<f64> = eigen.eigenvectors.column(smallest).into();
        let normal = prepared.normals[index];
        angles.push(normal.dot(&reference).abs().clamp(-1.0, 1.0).acos());
    }
    (
        median(&mut angles),
        if sampled == 0 {
            0.0
        } else {
            planar as f64 / sampled as f64
        },
    )
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}
