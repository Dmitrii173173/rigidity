//! Is an ICP edge's error a noise or a bias, and can the bias be predicted?
//!
//! # The question every failure of this project points at
//!
//! Four attempts at a better edge weight have failed here, and they share a
//! premise: that a registration's error is *noise around the right answer*,
//! so that a covariance describes it and a weight can exploit it. Three
//! measurements say otherwise. The predicted spread is optimistic by
//! seventeenfold and the factor is the same on an open plain and in a
//! closed corridor, which is what a systematic error looks like and not
//! what a random one does. Adding a hundred and fifty more edges to a
//! survey made it worse, when noise would have averaged down. And the
//! prediction ranks an edge's own six directions barely better than chance.
//!
//! Brossard, Bonnabel and Barrau (2020) say it outright — "assuming white
//! sensor noise leads to overoptimism as ICP is biased" — and add the other
//! half: the uncertainty of ICP is only meaningful *relative to the
//! uncertainty of its initialisation*. Both halves are measurable here, on
//! ground truth accurate to a millimetre, and neither has been measured.
//!
//! # What is measured
//!
//! Every pair is registered many times from initialisations drawn around
//! the truth, as an odometry would supply them. Each run's error against
//! the theodolite is a vector in the tangent; over the runs it has a mean
//! and a scatter.
//!
//! * The **mean** is the bias: where this pair lands whatever it started
//!   from. No amount of averaging removes it, and no covariance describes
//!   it.
//! * The **scatter** is what a covariance is entitled to describe.
//! * Their ratio says which of the two an edge weight is arguing about.
//!
//! Then: is the bias predictable from anything a system without truth can
//! see — the residual, the overlap, the conditioning, the distance? A bias
//! that is predictable is a correction. One that is not is a limit.
//!
//! ```text
//! cargo run --release -p rigidity-graph --example bias -- <directory> [pairs]
//! RIGIDITY_INIT_SIGMA=0.10 …   # how far the initialisations are thrown
//! RIGIDITY_ALL_PAIRS=1 …       # every combination of stations, not the legs
//! ```

use std::path::{Path, PathBuf};

use rigidity_core::lie::{Se3, So3};
use rigidity_core::nalgebra::{Matrix3, Matrix4, Vector3, Vector6};
use rigidity_core::observability::ObservabilityCriteria;
use rigidity_harness::overlap;
use rigidity_pipeline::{
    PrepareParams, Prepared, RegisterParams, analyse_registration, median_absolute_residual,
    prepare_cloud, register_pair_observed,
};
use rigidity_scenes::rng::Rng;

/// Sensor noise along the normal, metres — the Hokuyo, as M9 took it.
const NOISE: f64 = 0.03;
/// The accuracy a survey of this kind asks for, metres.
const TOLERANCE: f64 = 0.05;
/// How many initialisations each pair is registered from.
const DRAWS: usize = 24;
/// How far they are thrown: metres of translation and radians of rotation.
/// A tenth of a metre is what one leg of a walked survey drifts by.
const INIT_TRANSLATION: f64 = 0.10;
const INIT_ROTATION: f64 = 0.02;

fn main() {
    let mut args = std::env::args().skip(1);
    let directory = PathBuf::from(args.next().unwrap_or_else(|| {
        eprintln!("usage: bias <directory> [pairs]");
        std::process::exit(2);
    }));
    let pairs: usize = args.next().and_then(|text| text.parse().ok()).unwrap_or(12);
    let throw = std::env::var("RIGIDITY_INIT_SIGMA")
        .ok()
        .and_then(|text| text.parse::<f64>().ok())
        .unwrap_or(INIT_TRANSLATION);

    // Which pairs to register. A walked survey has legs: station i to
    // station i + 1, and stations further apart share too little of a room
    // to be registered at all. A set of terrestrial scans is not walked —
    // the ETH TLS stations stand around one office, and the benchmark ships
    // a reference for every one of the ten combinations of its five, which
    // is what its own `pairs.txt` names as the pairs to register. Taking
    // only the four consecutive ones there discards six pairs whose truth
    // is already on disk, and six pairs is the difference between an
    // anecdote and a sample.
    let legs: Vec<(usize, usize)> = if std::env::var("RIGIDITY_ALL_PAIRS").is_ok() {
        (0..=pairs)
            .flat_map(|from| (from + 1..=pairs).map(move |to| (from, to)))
            .collect()
    } else {
        (0..pairs).map(|index| (index, index + 1)).collect()
    };

    let truth = read_truth(&directory.join("pose_scanner_leica.csv"), pairs + 1);
    let prepared = read_scans(&directory, pairs + 1);
    let params = RegisterParams::default();
    let criteria = ObservabilityCriteria {
        noise_sigma: NOISE,
        tolerance: TOLERANCE,
    };
    let quiet = |_: &_| std::ops::ControlFlow::Continue(());

    println!(
        "\n{DRAWS} initialisations per pair, thrown {throw} m and \
         {INIT_ROTATION} rad about the truth"
    );
    println!(
        "\n pair    bias    scatter   ratio   predicted   rmse   overlap  median    κ     step"
    );
    let mut rows: Vec<[f64; 11]> = Vec::new();
    for (place, &(from, to)) in legs.iter().enumerate() {
        let exact = truth[from].inverse() * truth[to];
        // Seeded by position, which in the walked case is the leg index, so
        // the draws are the ones the earlier tables were measured from.
        let mut rng = Rng::new(0x5EED + place as u64);

        let mut errors: Vec<Vector6<f64>> = Vec::new();
        for _ in 0..DRAWS {
            let mut twist = Vector6::zeros();
            for axis in 0..3 {
                twist[axis] = rng.normal(throw);
                twist[axis + 3] = rng.normal(INIT_ROTATION);
            }
            let start = Se3::exp(&twist) * exact;
            let result = register_pair_observed(
                &prepared[to],
                &prepared[from],
                start,
                &params,
                quiet,
            );
            errors.push((exact.inverse() * result.pose).log());
        }

        // The mean is the bias; the spread about it is what a covariance
        // may claim to describe.
        let mean: Vector6<f64> =
            errors.iter().fold(Vector6::zeros(), |sum, e| sum + e) / DRAWS as f64;

        // What the project would have predicted, and what a system without
        // truth can see about this pair.
        let settled = register_pair_observed(
            &prepared[to],
            &prepared[from],
            exact,
            &params,
            quiet,
        );
        let analysis = analyse_registration(
            &prepared[to],
            &prepared[from],
            &settled.pose,
            &params,
        )
        .expect("the pair analyses");

        // Bias and scatter are taken in the coordinates the spread lives in
        // — metres of point displacement at the radius of gyration — so
        // that a coefficient between them means something. In world
        // coordinates the norm of a six-vector adds metres to radians, and
        // the number would depend on which units somebody chose.
        let scatter: f64 = (errors
            .iter()
            .map(|e| analysis.conditioning.to_normalised(e - mean).norm_squared())
            .sum::<f64>()
            / DRAWS as f64)
            .sqrt();
        let bias = analysis.conditioning.to_normalised(mean).norm();

        // Where the bias points. A magnitude cannot be corrected for — a
        // covariance that is too large only moves trust to another edge,
        // which is biased too. A direction can: subtract it. So the
        // question is whether the bias lies along anything the geometry
        // already names.
        let unit = if bias > 0.0 {
            analysis.conditioning.to_normalised(mean) / bias
        } else {
            Vector6::zeros()
        };
        let spreads = analysis.conditioning.uncertainty(criteria.noise_sigma);
        let weakest = (0..6)
            .max_by(|a, b| spreads[*a].total_cmp(&spreads[*b]))
            .expect("six");
        let strongest = (0..6)
            .min_by(|a, b| spreads[*a].total_cmp(&spreads[*b]))
            .expect("six");

        // `RIGIDITY_DUMP_RUNS` writes the individual draws instead of only
        // the two numbers they reduce to. A mean and a standard deviation
        // cannot show that twenty-four starts landed on top of each other
        // in the wrong place, and that is the whole claim.
        //
        // The plane is the two least-determined directions, and the
        // coordinates are theirs: the prediction is diagonal in this basis,
        // so an ellipse drawn from `spreads` is axis-aligned and needs no
        // covariance anyone has to trust. `component` and `uncertainty`
        // both return metres at the radius of gyration, so the cluster and
        // the ellipse are in the same units.
        if std::env::var("RIGIDITY_DUMP_RUNS").is_ok() {
            let mut order: Vec<usize> = (0..6).collect();
            order.sort_by(|a, b| spreads[*b].total_cmp(&spreads[*a]));
            let (first, second) = (order[0], order[1]);
            println!(
                "PLANE,{place},{first},{second},{:.9},{:.9}",
                spreads[first], spreads[second]
            );
            for error in &errors {
                println!(
                    "DRAW,{place},{:.9},{:.9}",
                    analysis.conditioning.component(first, *error),
                    analysis.conditioning.component(second, *error)
                );
            }
        }
        let along_weak = unit.dot(&analysis.conditioning.direction(weakest)).abs();
        let along_strong = unit.dot(&analysis.conditioning.direction(strongest)).abs();
        let travel = {
            let mut world = Vector6::zeros();
            let step = exact.translation();
            if step.norm() > 0.0 {
                world
                    .fixed_rows_mut::<3>(0)
                    .copy_from(&(step / step.norm()));
            }
            let mapped = analysis.conditioning.to_normalised(world);
            if mapped.norm() > 0.0 {
                unit.dot(&(mapped / mapped.norm())).abs()
            } else {
                f64::NAN
            }
        };

        let predicted = analysis
            .conditioning
            .uncertainty(criteria.noise_sigma)
            .iter()
            .fold(0.0f64, |worst, spread| worst.max(*spread));
        let shared = overlap(
            &prepared[to],
            &prepared[from],
            &settled.pose,
            &params,
        );
        let median = median_absolute_residual(
            &prepared[to],
            &prepared[from],
            &settled.pose,
            &params,
        )
        .unwrap_or(f64::NAN);
        let condition = analysis.conditioning.condition_number();
        let step = exact.translation().norm();

        println!(
            " {:>5}   {bias:.4}   {scatter:.4}   {:5.1}   {predicted:.4}    {:.3}  {shared:.3}   {median:.4}  {condition:5.2}  {step:.3}",
            format!("{from}-{to}"),
            bias / scatter.max(1e-9),
            settled.rmse
        );
        rows.push([
            bias,
            scatter,
            predicted,
            settled.rmse,
            shared,
            median,
            condition,
            step,
            along_weak,
            along_strong,
            travel,
        ]);
    }

    let median_of = |column: usize| {
        let mut values: Vec<f64> = rows.iter().map(|row| row[column]).collect();
        values.sort_by(f64::total_cmp);
        values[values.len() / 2]
    };
    println!(
        "\nmedian bias {:.4} m, median scatter {:.4} m, ratio {:.1}",
        median_of(0),
        median_of(1),
        median_of(0) / median_of(1).max(1e-9)
    );
    println!(
        "median predicted spread {:.4} m — {:.1}× under the bias it is supposed to cover",
        median_of(2),
        median_of(0) / median_of(2).max(1e-9)
    );

    // Where it points, which decides whether it can be corrected at all.
    // The six directions are columns of a singular-vector matrix, so they
    // are orthonormal and the null is the plain one. One direction drawn at
    // random in six dimensions has a median |cos| of 0.309 against a fixed
    // axis — but that is not the figure to read the lines below against.
    // What is printed is `values[n / 2]`, the upper of the two middle order
    // statistics, which sits above the median it estimates: its null is
    // 0.337 over twelve pairs and 0.343 over ten, with a standard deviation
    // near 0.10 in both. Two million draws give those. Anything within a
    // deviation of them is no alignment, and the pairs of one sequence are
    // not independent besides — consecutive pairs share a scan.
    println!("\nthe bias direction, |cos| against something the geometry names:");
    for (offset, name) in [
        (8usize, "the weakest direction"),
        (9, "the best determined one"),
        (10, "the way the survey walks"),
    ] {
        let mut values: Vec<f64> = rows
            .iter()
            .map(|row| row[offset])
            .filter(|value| value.is_finite())
            .collect();
        values.sort_by(f64::total_cmp);
        println!(
            "  {name:<24} {:.3} median, {:.3}…{:.3} quartiles",
            values[values.len() / 2],
            values[values.len() / 4],
            values[3 * values.len() / 4]
        );
    }

    // The coefficient a model would need: how many median residuals a bias
    // is. A median rather than a least-squares fit, because a line through
    // twelve points, two of which may be in another basin, is a line
    // through those two points.
    let mut ratios: Vec<f64> = rows
        .iter()
        .filter(|row| row[5] > 0.0)
        .map(|row| row[0] / row[5])
        .collect();
    ratios.sort_by(f64::total_cmp);
    println!(
        "\nbias / median residual: {:.2} median, {:.2}…{:.2} quartiles",
        ratios[ratios.len() / 2],
        ratios[ratios.len() / 4],
        ratios[3 * ratios.len() / 4]
    );

    // Is the bias predictable from anything visible? Spearman, because the
    // question is whether the ordering carries information, not whether the
    // relation is a line.
    let names = ["rmse", "overlap", "median residual", "κ", "step"];
    println!("\nrank correlation of the bias with what a system can see:");
    for (offset, name) in names.iter().enumerate() {
        let column = offset + 3;
        let rho = spearman(
            &rows.iter().map(|row| row[0]).collect::<Vec<_>>(),
            &rows.iter().map(|row| row[column]).collect::<Vec<_>>(),
        );
        println!("  {name:<16} ρ = {rho:+.3}");
    }
}

/// Spearman's rank correlation, ties broken by position.
fn spearman(a: &[f64], b: &[f64]) -> f64 {
    let rank = |values: &[f64]| -> Vec<f64> {
        let mut order: Vec<usize> = (0..values.len()).collect();
        order.sort_by(|x, y| values[*x].total_cmp(&values[*y]));
        let mut ranks = vec![0.0; values.len()];
        for (place, index) in order.iter().enumerate() {
            ranks[*index] = place as f64;
        }
        ranks
    };
    let (x, y) = (rank(a), rank(b));
    let n = x.len() as f64;
    let mean = (n - 1.0) / 2.0;
    let (mut top, mut left, mut right) = (0.0, 0.0, 0.0);
    for index in 0..x.len() {
        let (dx, dy) = (x[index] - mean, y[index] - mean);
        top += dx * dy;
        left += dx * dx;
        right += dy * dy;
    }
    top / (left * right).sqrt().max(1e-12)
}

fn read_truth(path: &Path, count: usize) -> Vec<Se3> {
    let matrices = rigidity_io::read_poses(path).unwrap_or_else(|error| {
        eprintln!("{}: {error}", path.display());
        std::process::exit(1);
    });
    matrices[..count].iter().map(from_matrix).collect()
}

fn from_matrix(matrix: &Matrix4<f64>) -> Se3 {
    let rotation: Matrix3<f64> = matrix.fixed_view::<3, 3>(0, 0).into();
    let translation: Vector3<f64> = matrix.fixed_view::<3, 1>(0, 3).into();
    Se3::from_parts(So3::from_matrix_unchecked(rotation), translation)
}

fn read_scans(directory: &Path, count: usize) -> Vec<Prepared> {
    let params = PrepareParams {
        voxel: 0.05,
        neighbours: 16,
    };
    println!("reading {count} scans");
    (0..count)
        .map(|index| {
            // `RIGIDITY_SCAN_EXT` because a second sensor writes a second
            // format: the ETH ASL scans are CSV and the ETH TLS scans are
            // binary PLY, and `rigidity_io::read` already dispatches on the
            // extension. The name stays `Hokuyo_<i>` for both, which is a
            // small lie about the instrument and a large saving in code.
            let extension = std::env::var("RIGIDITY_SCAN_EXT").unwrap_or_else(|_| "csv".into());
            let path = directory.join(format!("Hokuyo_{index}.{extension}"));
            let raw = rigidity_io::read(&path).unwrap_or_else(|error| {
                eprintln!("{}: {error}", path.display());
                std::process::exit(1);
            });
            prepare_cloud(&raw, &params).expect("the scan prepares")
        })
        .collect()
}
