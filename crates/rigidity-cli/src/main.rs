//! The `rigidity` command-line interface.
//!
//! Three actions: build a synthetic scene, assess the conditioning of a
//! single cloud, and register two clouds with an observability report.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use nalgebra::Vector3;
use rigidity_core::icp::{IcpConfig, Kernel, register, surface};
use rigidity_core::lie::Se3;
use rigidity_core::normals::estimate_normals;
use rigidity_core::observability::{Analysis, Correspondence, ObservabilityCriteria, analyse};
use rigidity_core::voxel::voxel_downsample;
use rigidity_core::{NeighborSearch, PointCloud};
use rigidity_io::{read_ply, write_ply};
use rigidity_scenes::{Scene, SceneKind, SceneParams};
use rigidity_spatial::KdTree;

/// Conditioning of point-cloud registration.
///
/// Returns not only a pose, but also which degrees of freedom the geometry
/// actually determined.
#[derive(Parser)]
#[command(name = "rigidity", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Build a synthetic scene with a known null space.
    Scene {
        /// Which surface.
        #[arg(short, long, value_enum, default_value_t = Kind::Corridor)]
        kind: Kind,
        /// Points per face.
        #[arg(short, long, default_value_t = 20_000)]
        points: usize,
        /// Characteristic size, metres.
        #[arg(short, long, default_value_t = 1.0)]
        scale: f64,
        /// Positional noise, metres.
        #[arg(long, default_value_t = 0.0)]
        noise: f64,
        /// Translate the scene by "x,y,z" metres before writing.
        ///
        /// This is how a registration pair is produced: write one scene
        /// unshifted and another shifted, and the right answer is known.
        #[arg(long, value_delimiter = ',')]
        shift: Option<Vec<f64>>,
        /// Rotate by "roll,pitch,yaw" degrees before writing.
        #[arg(long, value_delimiter = ',')]
        turn: Option<Vec<f64>>,
        /// Where to write the PLY.
        #[arg(short, long)]
        out: PathBuf,
    },

    /// Assess the conditioning of a single cloud.
    ///
    /// It answers the question: if anything were registered against this
    /// surface, which degrees of freedom would end up determined?
    Analyse {
        /// The PLY file.
        cloud: PathBuf,
        #[command(flatten)]
        common: Common,
    },

    /// Register two clouds and assess how much the answer is worth.
    Register {
        /// What is moved.
        source: PathBuf,
        /// What it is moved onto.
        target: PathBuf,
        #[command(flatten)]
        common: Common,
        /// Maximum correspondence distance, metres.
        #[arg(long, default_value_t = 0.5)]
        max_distance: f64,
        /// Threshold of the robust Huber kernel, metres.
        #[arg(long, default_value_t = 0.1)]
        huber: f64,
        /// Where to write the transformed source.
        #[arg(long)]
        out: Option<PathBuf>,
    },
}

#[derive(clap::Args)]
struct Common {
    /// Edge of the downsampling voxel, metres. Zero disables it.
    #[arg(long, default_value_t = 0.05)]
    voxel: f64,
    /// Neighbours used for normal estimation.
    #[arg(long, default_value_t = 16)]
    neighbours: usize,
    /// Standard deviation of the sensor noise, metres.
    #[arg(long, default_value_t = 0.01)]
    noise: f64,
    /// Required pose accuracy, metres.
    #[arg(long, default_value_t = 0.001)]
    tolerance: f64,
    /// Empirical correction applied to the predicted spread.
    ///
    /// On real data the formula understates the spread by roughly a factor
    /// of 17: it treats measurements as independent while they are
    /// correlated. See the README for the measurement.
    #[arg(long, default_value_t = 1.0)]
    calibration: f64,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Kind {
    Plane,
    Cylinder,
    Sphere,
    TwoPlanes,
    Corner,
    TeeJoint,
    Corridor,
}

impl From<Kind> for SceneKind {
    fn from(kind: Kind) -> Self {
        match kind {
            Kind::Plane => SceneKind::Plane,
            Kind::Cylinder => SceneKind::Cylinder,
            Kind::Sphere => SceneKind::Sphere,
            Kind::TwoPlanes => SceneKind::TwoPlanes,
            Kind::Corner => SceneKind::Corner,
            Kind::TeeJoint => SceneKind::TeeJoint,
            Kind::Corridor => SceneKind::Corridor,
        }
    }
}

struct Prepared {
    cloud: PointCloud,
    normals: Vec<Vector3<f64>>,
    tree: KdTree,
}

fn prepare(path: &Path, common: &Common) -> Result<Prepared, String> {
    let raw = read_ply(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let cloud = if common.voxel > 0.0 {
        voxel_downsample(&raw, common.voxel).map_err(|e| e.to_string())?
    } else {
        raw
    };
    if cloud.is_empty() {
        return Err(format!(
            "{}: no points left after downsampling",
            path.display()
        ));
    }
    let tree = KdTree::build(&cloud).map_err(|e| e.to_string())?;
    let normals = estimate_normals(&cloud, &tree, common.neighbours);
    Ok(Prepared {
        cloud,
        normals,
        tree,
    })
}

fn criteria(common: &Common) -> ObservabilityCriteria {
    ObservabilityCriteria {
        noise_sigma: common.noise * common.calibration,
        tolerance: common.tolerance,
    }
}

fn print_report(analysis: &Analysis, common: &Common) {
    if common.calibration != 1.0 {
        println!("empirical correction: ×{:.0}\n", common.calibration);
    }
    print!("{}", analysis.describe(&criteria(common)));
}

fn run() -> Result<(), String> {
    match Cli::parse().command {
        Command::Scene {
            kind,
            points,
            scale,
            noise,
            shift,
            turn,
            out,
        } => {
            let scene = Scene::generate(
                kind.into(),
                SceneParams {
                    points_per_face: points,
                    scale,
                    noise_sigma: noise,
                    ..SceneParams::default()
                },
            );
            let offset = shift.unwrap_or_else(|| vec![0.0; 3]);
            let angles = turn.unwrap_or_else(|| vec![0.0; 3]);
            let motion = Se3::exp(&nalgebra::Vector6::new(
                offset[0],
                offset[1],
                offset[2],
                angles[0].to_radians(),
                angles[1].to_radians(),
                angles[2].to_radians(),
            ));
            let written = if motion == Se3::identity() {
                scene.cloud.clone()
            } else {
                let mut moved = PointCloud::with_capacity(scene.len());
                for index in 0..scene.len() {
                    moved.push(motion.transform_point(&scene.cloud.point(index)));
                }
                println!(
                    "scene translated by [{:+.4}, {:+.4}, {:+.4}] m and rotated by \
[{:+.3}, {:+.3}, {:+.3}]°",
                    offset[0], offset[1], offset[2], angles[0], angles[1], angles[2]
                );
                moved
            };
            write_ply(&written, &out).map_err(|e| e.to_string())?;
            println!("scene \"{}\": {} points", scene.kind.name(), scene.len());
            println!(
                "unobservable degrees of freedom by construction: {}",
                scene.nullspace_dimension()
            );
            println!("written: {}", out.display());
        }

        Command::Analyse { cloud, common } => {
            let prepared = prepare(&cloud, &common)?;
            println!("points after downsampling: {}\n", prepared.cloud.len());
            let analysis = analyse(prepared.cloud.len(), Kernel::Squared, |index| {
                Some(Correspondence {
                    point: prepared.cloud.point(index),
                    normal: prepared.normals[index],
                    residual: 0.0,
                })
            })
            .ok_or("the cloud is empty")?;
            print_report(&analysis, &common);
        }

        Command::Register {
            source,
            target,
            common,
            max_distance,
            huber,
            out,
        } => {
            let moving = prepare(&source, &common)?;
            let fixed = prepare(&target, &common)?;
            println!(
                "source {} points, target {} points\n",
                moving.cloud.len(),
                fixed.cloud.len()
            );

            let kernel = Kernel::Huber(huber);
            let config = IcpConfig {
                kernel,
                max_correspondence_distance: max_distance,
                min_normal_cosine: 0.0,
                ..IcpConfig::default()
            };
            let result = register(
                &surface(&moving.cloud, &moving.normals),
                &surface(&fixed.cloud, &fixed.normals),
                &fixed.tree,
                Se3::identity(),
                &config,
            );

            let translation = result.pose.translation();
            let rotation = result.pose.rotation().log();
            println!(
                "translation: x = {:+8.4} m  y = {:+8.4} m  z = {:+8.4} m",
                translation.x, translation.y, translation.z
            );
            println!(
                "rotation:    roll {:+7.3}°  pitch {:+7.3}°  yaw {:+7.3}°",
                rotation.x.to_degrees(),
                rotation.y.to_degrees(),
                rotation.z.to_degrees()
            );
            println!(
                "RMSE: {:.5} m   correspondences: {}   iterations: {}   converged: {}\n",
                result.rmse, result.correspondences, result.iterations, result.converged
            );

            let analysis = analyse(moving.cloud.len(), kernel, |index| {
                let point = result.pose.transform_point(&moving.cloud.point(index));
                let mut found = Vec::with_capacity(1);
                fixed.tree.knn_into(&point, 1, &mut found);
                let nearest = found.first()?;
                if nearest.distance_squared > max_distance * max_distance {
                    return None;
                }
                let matched = nearest.index as usize;
                Some(Correspondence {
                    point,
                    normal: fixed.normals[matched],
                    residual: fixed.normals[matched].dot(&(point - fixed.cloud.point(matched))),
                })
            })
            .ok_or("no correspondences left")?;
            print_report(&analysis, &common);

            if let Some(path) = out {
                let mut moved = PointCloud::with_capacity(moving.cloud.len());
                for index in 0..moving.cloud.len() {
                    moved.push(result.pose.transform_point(&moving.cloud.point(index)));
                }
                write_ply(&moved, &path).map_err(|e| e.to_string())?;
                println!("\ntransformed source written: {}", path.display());
            }
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}
