//! From files on disk to a conditioning report.
//!
//! The core provides the pieces — downsampling, normals, an index, ICP,
//! conditioning — and every front end has to assemble them in the same
//! order, with the same defaults, or its numbers stop being comparable
//! with anyone else's. While that assembly lived inside the binary the
//! second front end had to copy it, and two copies drift. It lives here
//! instead, and the command line and the viewer call the same code.
//!
//! Nothing here knows about argument parsing or about drawing.
//!
//! ```no_run
//! use rigidity_pipeline::{PrepareParams, RegisterParams, prepare, register_pair};
//!
//! let params = PrepareParams::default();
//! let moving = prepare("source.ply".as_ref(), &params)?;
//! let fixed = prepare("target.ply".as_ref(), &params)?;
//! let result = register_pair(&moving, &fixed, &RegisterParams::default());
//! println!("{:.5} m", result.rmse);
//! # Ok::<(), rigidity_pipeline::PipelineError>(())
//! ```

use std::ops::ControlFlow;
use std::path::{Path, PathBuf};

use rigidity_core::icp::{
    IcpConfig, IcpResult, IterationReport, Kernel, Surface, register_observed, surface,
};
use rigidity_core::lie::Se3;
use rigidity_core::nalgebra::Vector3;
use rigidity_core::normals::estimate_normals_observed;
use rigidity_core::observability::{Analysis, Correspondence, ObservabilityCriteria, analyse};
use rigidity_core::voxel::voxel_downsample_observed;
use rigidity_core::{NeighborSearch, PointCloud};
use rigidity_spatial::KdTree;

/// Anything that can go wrong between a file and a report.
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    /// The file could not be read.
    #[error("{}: {source}", path.display())]
    Read {
        /// The file that was being read.
        path: PathBuf,
        /// What the reader said.
        source: rigidity_io::IoError,
    },
    /// Preparation of a named cloud failed.
    ///
    /// It exists so that a message keeps the file it belongs to: the
    /// inner error is the same one [`prepare_cloud`] returns for a cloud
    /// held in memory, which has no name to report.
    #[error("{}: {source}", path.display())]
    Prepare {
        /// The file the cloud came from.
        path: PathBuf,
        /// What went wrong once it was read.
        source: Box<PipelineError>,
    },
    /// The voxel grid was coarse enough to leave nothing behind.
    #[error("no points left after downsampling")]
    EmptyAfterDownsampling,
    /// There is nothing to analyse.
    #[error("the cloud is empty")]
    EmptyCloud,
    /// Registration ended with no pair close enough to say anything about.
    #[error("no correspondences left")]
    NoCorrespondences,
    /// The cloud itself is malformed.
    #[error(transparent)]
    Cloud(#[from] rigidity_core::CloudError),
    /// The index could not be built.
    #[error(transparent)]
    Spatial(#[from] rigidity_spatial::SpatialError),
}

/// Which part of the work a [`Progress`] report is about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Reading the file.
    Reading,
    /// Voxel downsampling.
    Downsampling,
    /// Building the spatial index.
    Indexing,
    /// Estimating normals.
    Normals,
}

impl Stage {
    /// A label fit to be shown to a person.
    pub fn label(self) -> &'static str {
        match self {
            Self::Reading => "reading",
            Self::Downsampling => "downsampling",
            Self::Indexing => "indexing",
            Self::Normals => "normals",
        }
    }
}

/// How far along one stage is.
///
/// Every stage reports `done == 0` as it begins and `done == total` as it
/// ends, so a caller can show the name of the stage before any of its work
/// has happened. The unit of `total` belongs to the stage and is not
/// comparable between them: points for the normals, internal phases for
/// the rest — see the functions in the core for why.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Progress {
    /// What is running.
    pub stage: Stage,
    /// How much of it is finished.
    pub done: usize,
    /// How much there is.
    pub total: usize,
}

/// Settings shared by everything that turns a raw cloud into a surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrepareParams {
    /// Edge of the downsampling voxel, metres. Zero disables it.
    pub voxel: f64,
    /// Neighbours used for normal estimation.
    pub neighbours: usize,
}

impl Default for PrepareParams {
    fn default() -> Self {
        Self {
            voxel: 0.05,
            neighbours: 16,
        }
    }
}

/// A cloud ready to be registered: points, normals and an index over them.
///
/// Deliberately not `Debug`: printing one would dump a million
/// coordinates and the whole index behind them. It is a handle to a
/// working set, not a value to inspect.
pub struct Prepared {
    /// The points, after downsampling.
    pub cloud: PointCloud,
    /// One normal per point.
    pub normals: Vec<Vector3<f64>>,
    /// An index over [`cloud`](Self::cloud).
    pub tree: KdTree,
}

impl Prepared {
    /// The pair the ICP takes.
    pub fn surface(&self) -> Surface<'_> {
        surface(&self.cloud, &self.normals)
    }

    /// How many points survived downsampling.
    pub fn len(&self) -> usize {
        self.cloud.len()
    }

    /// Whether anything survived.
    pub fn is_empty(&self) -> bool {
        self.cloud.is_empty()
    }
}

/// Reads a cloud and prepares it.
///
/// Whatever format the extension names: both front ends get every reader
/// the io crate has, and neither has a list of its own to fall behind.
pub fn prepare(path: &Path, params: &PrepareParams) -> Result<Prepared, PipelineError> {
    prepare_observed(path, params, |_| {})
}

/// The same, reporting progress.
pub fn prepare_observed<F>(
    path: &Path,
    params: &PrepareParams,
    mut observer: F,
) -> Result<Prepared, PipelineError>
where
    F: FnMut(Progress),
{
    // The reader has no progress of its own to report, so the stage is
    // announced and then confirmed. Naming it still matters: on a large
    // file this is where the wait is, and silence there reads as a hang.
    observer(Progress {
        stage: Stage::Reading,
        done: 0,
        total: 1,
    });
    let raw = rigidity_io::read(path).map_err(|source| PipelineError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    observer(Progress {
        stage: Stage::Reading,
        done: 1,
        total: 1,
    });

    prepare_cloud_observed(&raw, params, observer).map_err(|source| PipelineError::Prepare {
        path: path.to_path_buf(),
        source: Box::new(source),
    })
}

/// Prepares a cloud that is already in memory.
///
/// The viewer needs it twice over: scenes are generated rather than read,
/// and changing the voxel size must not re-read a file that has not
/// changed.
pub fn prepare_cloud(
    cloud: &PointCloud,
    params: &PrepareParams,
) -> Result<Prepared, PipelineError> {
    prepare_cloud_observed(cloud, params, |_| {})
}

/// The same, reporting progress.
///
/// The cloud is borrowed rather than consumed: the caller usually holds it
/// behind a handle it cannot give up — a viewer is still drawing it — and
/// the common path never needs to own it anyway, since downsampling reads
/// the input and writes a new cloud. Only a disabled voxel grid copies, and
/// that is a copy the old signature merely moved somewhere else.
pub fn prepare_cloud_observed<F>(
    cloud: &PointCloud,
    params: &PrepareParams,
    mut observer: F,
) -> Result<Prepared, PipelineError>
where
    F: FnMut(Progress),
{
    let cloud = if params.voxel > 0.0 {
        voxel_downsample_observed(cloud, params.voxel, |done, total| {
            observer(Progress {
                stage: Stage::Downsampling,
                done,
                total,
            })
        })?
    } else {
        cloud.clone()
    };
    if cloud.is_empty() {
        return Err(PipelineError::EmptyAfterDownsampling);
    }

    let tree = KdTree::build_observed(&cloud, |done, total| {
        observer(Progress {
            stage: Stage::Indexing,
            done,
            total,
        })
    })?;

    observer(Progress {
        stage: Stage::Normals,
        done: 0,
        total: cloud.len(),
    });
    let normals = estimate_normals_observed(&cloud, &tree, params.neighbours, |done, total| {
        observer(Progress {
            stage: Stage::Normals,
            done,
            total,
        })
    });

    Ok(Prepared {
        cloud,
        normals,
        tree,
    })
}

/// Settings of the registration itself.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegisterParams {
    /// Correspondences farther apart than this are discarded, metres.
    pub max_distance: f64,
    /// Threshold of the robust Huber kernel, metres.
    pub huber: f64,
    /// Cap on the number of iterations.
    pub max_iterations: usize,
    /// Minimum `|cos|` between normals for a pair to be kept.
    ///
    /// Zero — accept any — is the default here rather than the core's
    /// 0.8: the pairs this rejects are exactly the ones that constrain the
    /// directions the report is about, and dropping them flatters the
    /// spectrum.
    pub min_normal_cosine: f64,
}

impl Default for RegisterParams {
    fn default() -> Self {
        Self {
            max_distance: 0.5,
            huber: 0.1,
            max_iterations: IcpConfig::default().max_iterations,
            min_normal_cosine: 0.0,
        }
    }
}

impl RegisterParams {
    /// The loss function these settings imply.
    pub fn kernel(&self) -> Kernel {
        Kernel::Huber(self.huber)
    }

    /// The core's configuration these settings imply.
    pub fn icp_config(&self) -> IcpConfig {
        IcpConfig {
            kernel: self.kernel(),
            max_correspondence_distance: self.max_distance,
            min_normal_cosine: self.min_normal_cosine,
            max_iterations: self.max_iterations,
            ..IcpConfig::default()
        }
    }
}

/// Registers `moving` onto `fixed`, starting from the identity.
pub fn register_pair(moving: &Prepared, fixed: &Prepared, params: &RegisterParams) -> IcpResult {
    register_pair_observed(moving, fixed, Se3::identity(), params, |_| {
        ControlFlow::Continue(())
    })
}

/// The same, from a given initial pose and with an observer.
///
/// Returning [`ControlFlow::Break`] from the observer stops the run; see
/// [`rigidity_core::icp::register_observed`] for what that costs in
/// latency and what the result then means.
pub fn register_pair_observed<F>(
    moving: &Prepared,
    fixed: &Prepared,
    initial: Se3,
    params: &RegisterParams,
    observer: F,
) -> IcpResult
where
    F: FnMut(&IterationReport) -> ControlFlow<()>,
{
    register_observed(
        &moving.surface(),
        &fixed.surface(),
        &fixed.tree,
        initial,
        &params.icp_config(),
        observer,
    )
}

/// The conditioning of a single surface.
///
/// This is the question asked before a scan rather than after one: if
/// anything were registered against this cloud, which degrees of freedom
/// would it determine? Every point is its own correspondence at zero
/// residual, so the answer describes the geometry alone.
pub fn analyse_cloud(prepared: &Prepared) -> Result<Analysis, PipelineError> {
    analyse(prepared.cloud.len(), Kernel::Squared, |index| {
        Some(Correspondence {
            point: prepared.cloud.point(index),
            normal: prepared.normals[index],
            residual: 0.0,
        })
    })
    .ok_or(PipelineError::EmptyCloud)
}

/// The conditioning of a registration, at a given pose.
///
/// The pose is a parameter rather than an [`IcpResult`] on purpose: the
/// viewer asks this question while scrubbing a recorded run, and the
/// answer at iteration twelve is as legitimate as the answer at the end.
pub fn analyse_registration(
    moving: &Prepared,
    fixed: &Prepared,
    pose: &Se3,
    params: &RegisterParams,
) -> Result<Analysis, PipelineError> {
    let limit = params.max_distance * params.max_distance;
    analyse(moving.cloud.len(), params.kernel(), |index| {
        let point = pose.transform_point(&moving.cloud.point(index));
        let mut found = Vec::with_capacity(1);
        fixed.tree.knn_into(&point, 1, &mut found);
        let nearest = found.first()?;
        if nearest.distance_squared > limit {
            return None;
        }
        let matched = nearest.index as usize;
        Some(Correspondence {
            point,
            normal: fixed.normals[matched],
            residual: fixed.normals[matched].dot(&(point - fixed.cloud.point(matched))),
        })
    })
    .ok_or(PipelineError::NoCorrespondences)
}

/// The median absolute point-to-plane residual at a pose, metres.
///
/// # What it is for
///
/// Conditioning answers "is the pose determined by the geometry here". It
/// does not answer "is this the right place", and the project has measured
/// that it cannot: registrations that converged to a wrong minimum came
/// back with condition numbers of two to four, exactly like the ones that
/// did not, and reported six directions of six determined. That failure
/// sets the worst station of a survey and is the one thing a confident
/// report can be most wrong about.
///
/// This is what does answer it. At the right minimum the residuals that
/// remain are the sensor's own, so half of them fall inside `σ`; at a wrong
/// one they are the geometry's disagreement, and they do not. **Suspect the
/// registration when this exceeds the sensor's noise** — that is the whole
/// rule, and it has no free parameter beyond the `σ` the pipeline is
/// already told.
///
/// # What it was measured on
///
/// Four surveys of the ETH ASL Challenging Datasets against theodolite
/// ground truth — both scenes, at 360° and cropped to ±40° and ±90°, 29 to
/// 57 edges each. Taking "wrong basin" to mean more than 0.10 m of
/// translation error against the theodolite:
///
/// | | |
/// |---|---|
/// | wrong-basin edges caught | 62 of 64 |
/// | let through | 2, out by 0.14 m and 2.15 m |
/// | false alarms | 6 of 164 sound edges |
/// | false alarms on the two surveys where nothing failed | 0 of 57 |
///
/// The threshold was fixed on one survey and every other number above is
/// out of sample. The one comparable alternative — an edge whose RMSE is
/// more than one and a half times the survey's median — never raises a
/// false alarm but misses more, and collapses outright when over half a
/// survey is bad, because then the median is itself a failure. This one
/// needs no survey around it and works on a single pair.
///
/// # Cost, and where this will live eventually
///
/// One extra pass over the moving cloud, with one nearest-neighbour query
/// per point — about the price of a single ICP iteration. The ICP already
/// computes every one of these residuals and throws them away; when there
/// is next a reason to break `IcpResult`, this belongs in it, and this
/// function becomes the way to ask the same question at a pose the
/// registration did not stop at.
///
/// Returns `None` when nothing matched, which is its own answer.
pub fn median_absolute_residual(
    moving: &Prepared,
    fixed: &Prepared,
    pose: &Se3,
    params: &RegisterParams,
) -> Option<f64> {
    // The same rejection rule as `analyse_registration`, deliberately: two
    // functions that walk the same correspondences must agree about which
    // ones there are, or a report and the warning beside it will one day
    // describe different registrations.
    let limit = params.max_distance * params.max_distance;
    let mut residuals: Vec<f64> = Vec::new();
    let mut found = Vec::with_capacity(1);
    for index in 0..moving.cloud.len() {
        let point = pose.transform_point(&moving.cloud.point(index));
        fixed.tree.knn_into(&point, 1, &mut found);
        let Some(nearest) = found.first() else {
            continue;
        };
        if nearest.distance_squared > limit {
            continue;
        }
        let matched = nearest.index as usize;
        let normal = fixed.normals[matched];
        residuals.push(normal.dot(&(point - fixed.cloud.point(matched))).abs());
    }
    if residuals.is_empty() {
        return None;
    }
    // `select_nth_unstable_by` rather than a sort: the median is all that is
    // wanted and the clouds here run to millions of points.
    let middle = residuals.len() / 2;
    let (_, median, _) = residuals.select_nth_unstable_by(middle, f64::total_cmp);
    Some(*median)
}

/// What the numbers in a report are measured against.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReportParams {
    /// Standard deviation of the sensor noise, metres.
    pub noise: f64,
    /// Required pose accuracy, metres.
    pub tolerance: f64,
    /// Empirical correction applied to the predicted spread.
    ///
    /// On real data the formula understates the spread by roughly a factor
    /// of 17: it treats measurements as independent while they are
    /// correlated. See the README for the measurement.
    pub calibration: f64,
}

impl Default for ReportParams {
    fn default() -> Self {
        Self {
            noise: 0.01,
            tolerance: 0.001,
            calibration: 1.0,
        }
    }
}

impl ReportParams {
    /// The criteria the classification uses.
    ///
    /// The correction multiplies the noise, which is the only place it
    /// may be applied: it is a statement about how many measurements are
    /// really independent, not about how accurate the application needs
    /// to be.
    pub fn criteria(&self) -> ObservabilityCriteria {
        ObservabilityCriteria {
            noise_sigma: self.noise * self.calibration,
            tolerance: self.tolerance,
        }
    }
}

/// Applies a pose to every point of a cloud.
///
/// Attributes are not carried over, for the same reason downsampling
/// drops them: what a column means decides whether it survives a
/// transform, and the cloud does not know.
pub fn transform_cloud(cloud: &PointCloud, pose: &Se3) -> PointCloud {
    let mut moved = PointCloud::with_capacity(cloud.len());
    for index in 0..cloud.len() {
        moved.push(pose.transform_point(&cloud.point(index)));
    }
    moved
}

#[cfg(test)]
mod tests {
    use super::*;
    use rigidity_core::PointCloud;
    use rigidity_core::lie::So3;
    use rigidity_core::nalgebra::Vector6;

    /// Two perpendicular walls. Enough geometry that no direction is free,
    /// so a displaced pose is wrong in every sense and not merely
    /// unobservable.
    ///
    /// The sampling is deliberately not a lattice. The first version of
    /// this test used one at five centimetres, displaced the cloud by ten,
    /// and measured a median residual of exactly zero — the grid had slid
    /// into itself, every point landing on where another point had been.
    /// A regular grid has translations that are invisible to any residual,
    /// and a test built on one is measuring the sampling.
    fn corner() -> PointCloud {
        let mut cloud = PointCloud::new();
        for i in 0..60 {
            for j in 0..60 {
                let jitter = ((i * 37 + j * 17) % 13) as f64 * 0.003;
                let a = i as f64 * 0.05 + jitter;
                let b = j as f64 * 0.05 - jitter;
                cloud.push(Vector3::new(a, 0.0, b));
                cloud.push(Vector3::new(0.0, b, a));
            }
        }
        cloud
    }

    /// The property the detector rests on: at the pose that is right the
    /// median residual is far under the sensor's noise, and at a pose in
    /// the wrong place it is far over it. Both sides are asserted, because
    /// a statistic that is always small or always large would pass a test
    /// that only checked one.
    #[test]
    fn a_displaced_pose_shows_in_the_median_residual() {
        const NOISE: f64 = 0.01;
        let params = PrepareParams {
            voxel: 0.0,
            neighbours: 12,
        };
        let prepared = prepare_cloud(&corner(), &params).expect("the corner prepares");

        let right = median_absolute_residual(
            &prepared,
            &prepared,
            &Se3::identity(),
            &RegisterParams::default(),
        )
        .expect("the cloud matches itself");
        assert!(
            right < 0.1 * NOISE,
            "at the true pose the median residual is {right} m"
        );

        // Seven centimetres along the diagonal, so that both walls move off
        // themselves rather than one sliding along itself: a real
        // displacement, well under the correspondence cutoff, and nothing
        // about the geometry that excuses it.
        let displaced = Se3::exp(&Vector6::new(0.07, 0.07, 0.0, 0.0, 0.0, 0.0));
        let wrong =
            median_absolute_residual(&prepared, &prepared, &displaced, &RegisterParams::default())
                .expect("the displaced cloud still matches");
        assert!(
            wrong > NOISE,
            "displaced by 0.1 m the median residual is only {wrong} m"
        );

        // And a rotation, which moves the far end of the wall much further
        // than the near one — the case an RMSE dominated by the near end
        // can miss.
        let turned = Se3::from_parts(So3::exp(&Vector3::new(0.0, 0.0, 0.02)), Vector3::zeros());
        let turned =
            median_absolute_residual(&prepared, &prepared, &turned, &RegisterParams::default())
                .expect("the turned cloud still matches");
        assert!(
            turned > NOISE,
            "turned by 0.02 rad the median residual is only {turned} m"
        );
    }
}
