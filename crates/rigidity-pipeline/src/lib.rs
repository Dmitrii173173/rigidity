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
use rigidity_core::{Attribute, AttributeData, NeighborSearch, PointCloud};
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
    /// Trying the starting poses a global search laid out.
    ///
    /// `done` counts screened starts rather than points, so this stage's
    /// total is small and its steps are large.
    Searching,
}

impl Stage {
    /// A label fit to be shown to a person.
    pub fn label(self) -> &'static str {
        match self {
            Self::Reading => "reading",
            Self::Downsampling => "downsampling",
            Self::Indexing => "indexing",
            Self::Normals => "normals",
            Self::Searching => "searching for a start",
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

/// How wide a net [`register_globally`] casts.
///
/// The defaults are the configuration measured on the ETH ASL `stairs`
/// survey, where they recovered every one of the ten edges a cold start
/// had lost. They are a starting net, not a law: a survey whose stations
/// are metres apart wants a larger `radius`, and one taken with a tilted
/// sensor wants more than yaw.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchParams {
    /// How many rotations about the vertical to try, spread over a full
    /// turn.
    ///
    /// Yaw alone, because a levelled scanner leaves roll and pitch to the
    /// registration and only the heading genuinely unknown. This is the
    /// search's main assumption and the first thing to question when it
    /// fails: on a hand-held or a drone it does not hold.
    pub yaws: usize,
    /// Half-width of the grid of horizontal offsets, metres.
    ///
    /// Three positions per axis — `-radius`, zero, `+radius` — so the net
    /// is `yaws × 9` starts wide.
    pub radius: f64,
    /// How the clouds are prepared for the screening pass.
    ///
    /// Deliberately coarser than the caller's own preparation. Screening
    /// does not have to finish any start, only to rank them, and a cloud
    /// at four times the voxel is sixteen times cheaper to walk.
    pub screen: PrepareParams,
    /// Iteration cap for the screening pass.
    pub screen_iterations: usize,
    /// How many of the screened starts are then refined in full.
    pub refine: usize,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            yaws: 12,
            radius: 0.5,
            screen: PrepareParams {
                voxel: 0.20,
                neighbours: 12,
            },
            screen_iterations: 12,
            refine: 5,
        }
    }
}

/// What [`register_globally`] found, and what it is worth.
#[derive(Debug, Clone)]
pub struct Search {
    /// The registration, refined at the caller's own resolution.
    pub result: IcpResult,
    /// Its median absolute residual, metres.
    ///
    /// **This is the number that decides whether to believe the answer**,
    /// and the caller must look at it. A search always returns its best
    /// candidate; whether the best of a bad set is worth anything is
    /// settled by comparing this with the sensor's noise, exactly as
    /// [`median_absolute_residual`] describes.
    pub median_residual: f64,
    /// How many starts were tried.
    pub starts: usize,
}

/// Registers without being told where to start.
///
/// # The problem this solves
///
/// ICP needs a starting pose inside the right basin, and a survey normally
/// has one: the previous leg. The first leg of a survey has none, and any
/// pair opened on their own has none either. Started from the identity,
/// ICP converges — reports `converged`, a small residual and a healthy
/// spectrum — into whatever minimum happens to be nearest, which on the
/// ETH ASL `stairs` survey was the wrong one for ten of fifty-nine edges,
/// by as much as 0.46 m.
///
/// # How
///
/// Nothing is learned and nothing is described. `yaws × 9` starts are laid
/// out — a turn about the vertical crossed with a coarse grid of
/// horizontal offsets — every one is run to a short ICP on a deliberately
/// coarse pair of clouds, and they are ranked. Only [`SearchParams::refine`]
/// of them are then registered properly, at the caller's own resolution.
/// The coarse screen is the whole reason this is interactive: it turned 38
/// seconds into 2 on the measurement below, before any threads.
///
/// # What ranks them, and why it has to be this
///
/// The median absolute residual, not the RMSE. Picking the candidate whose
/// surfaces agree most tightly is precisely the mistake this project spent
/// its stage-three measurements documenting: a wrong minimum agrees
/// tightly too, and the conditioning report is blind to the difference. The
/// median residual is the one criterion here calibrated against theodolite
/// truth, and a global search is only as honest as its selection rule.
///
/// # What it was measured on
///
/// The ETH ASL `stairs` survey, thirty-one stations, against theodolite
/// ground truth. Ten of the fifty-nine edges were in a wrong basin from a
/// cold start, out by 0.10 to 0.46 m. All ten come back within 12 mm, with
/// a median residual inside the sensor's noise, in one to three seconds an
/// edge before any parallelism. One survey is one survey.
///
/// # What it does not do
///
/// It does not promise to find the right minimum, and it must not be read
/// as if it did. It returns the best candidate it saw, and
/// [`Search::median_residual`] is how the caller finds out whether that
/// candidate is worth anything: past the sensor's noise means the search
/// failed and said so, which is a different thing from the search failing
/// quietly.
///
/// Returns `None` when the clouds never matched at all, at any start.
pub fn register_globally(
    moving: &Prepared,
    fixed: &Prepared,
    params: &RegisterParams,
    search: &SearchParams,
) -> Option<Search> {
    register_globally_observed(moving, fixed, params, search, |_, _| {
        ControlFlow::Continue(())
    })
}

/// The same, reporting progress and able to stop.
///
/// The observer is called from several threads at once, once per screened
/// start, with the number finished and the number there are. Returning
/// [`ControlFlow::Break`] abandons the search, which then returns `None`.
pub fn register_globally_observed<F>(
    moving: &Prepared,
    fixed: &Prepared,
    params: &RegisterParams,
    search: &SearchParams,
    observer: F,
) -> Option<Search>
where
    F: Fn(usize, usize) -> ControlFlow<()> + Sync,
{
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    let starts = laid_out_starts(search);
    if starts.is_empty() {
        return None;
    }

    // Coarse copies, made once and shared by every start.
    let coarse_moving = prepare_cloud(&moving.cloud, &search.screen).ok()?;
    let coarse_fixed = prepare_cloud(&fixed.cloud, &search.screen).ok()?;
    let screen = RegisterParams {
        max_iterations: search.screen_iterations,
        ..*params
    };

    let done = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    // `collect` into a vector preserves the order of `starts`, so the
    // ranking below does not depend on which thread finished first. The
    // ICP itself is already independent of the thread count.
    let screened: Vec<Option<(f64, Se3)>> = starts
        .par_iter()
        .map(|start| {
            if stop.load(Ordering::Relaxed) {
                return None;
            }
            let found =
                register_pair_observed(&coarse_moving, &coarse_fixed, *start, &screen, |_| {
                    ControlFlow::Continue(())
                });
            let median =
                median_absolute_residual(&coarse_moving, &coarse_fixed, &found.pose, &screen);
            let finished = done.fetch_add(1, Ordering::Relaxed) + 1;
            if observer(finished, starts.len()).is_break() {
                stop.store(true, Ordering::Relaxed);
            }
            median.map(|median| (median, found.pose))
        })
        .collect();
    if stop.load(Ordering::Relaxed) {
        return None;
    }

    let mut ranked: Vec<(f64, Se3)> = screened.into_iter().flatten().collect();
    ranked.sort_by(|left, right| left.0.total_cmp(&right.0));

    // Refine the survivors at the caller's own resolution, and let the same
    // criterion choose between them again: a start that screened best on a
    // twenty-centimetre voxel is not always the one that finishes best.
    let mut best: Option<Search> = None;
    for (_, start) in ranked.iter().take(search.refine) {
        let result =
            register_pair_observed(moving, fixed, *start, params, |_| ControlFlow::Continue(()));
        let Some(median_residual) = median_absolute_residual(moving, fixed, &result.pose, params)
        else {
            continue;
        };
        if best
            .as_ref()
            .is_none_or(|found| median_residual < found.median_residual)
        {
            best = Some(Search {
                result,
                median_residual,
                starts: starts.len(),
            });
        }
    }
    best
}

/// The net: a turn about the vertical crossed with a grid of offsets.
fn laid_out_starts(search: &SearchParams) -> Vec<Se3> {
    use rigidity_core::lie::So3;

    let mut starts = Vec::with_capacity(search.yaws * 9);
    for step in 0..search.yaws {
        let yaw = step as f64 * std::f64::consts::TAU / search.yaws.max(1) as f64;
        let turn = So3::exp(&Vector3::new(0.0, 0.0, yaw));
        for x in [-search.radius, 0.0, search.radius] {
            for y in [-search.radius, 0.0, search.radius] {
                starts.push(Se3::from_parts(turn, Vector3::new(x, y, 0.0)));
            }
        }
    }
    starts
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

/// Several placed clouds written as one.
///
/// Each cloud is carried into a common frame by the pose beside it, and
/// the result is the union — what a viewer showing them all is drawing,
/// as a single file. The pairwise case is the useful one: registration
/// answers *where* the second scan goes, and this is what turns that
/// answer into something another program can open.
///
/// # The frame, and why it is the first cloud's
///
/// The result takes the origin of the first cloud given. Coordinates are
/// held as `f32` offsets from a cloud's origin, so an origin thrown away —
/// zero, say — costs precision in proportion to how far the survey sits
/// from it: at four million metres, a `f32` step is a quarter of a metre.
/// Keeping the first cloud's origin means a merge is exact where the
/// clouds are, which is the only place it has to be.
///
/// # Attributes
///
/// A column survives only if **every** cloud has one of that name and the
/// same type; the rest are dropped. Intensity from one scan and nothing
/// from the next is not a column, and inventing values to fill the gap
/// would put numbers in a file that no instrument measured.
pub fn merge(placed: &[(&PointCloud, Se3)]) -> PointCloud {
    let Some((first, _)) = placed.first() else {
        return PointCloud::new();
    };
    let total: usize = placed.iter().map(|(cloud, _)| cloud.len()).sum();
    // Room for all of it up front: a survey merge is tens of millions of
    // points, and growing into that a doubling at a time copies most of it
    // several times over. `rebase` on an empty cloud only sets the origin.
    let mut merged = PointCloud::with_capacity(total);
    merged.rebase(first.origin());
    for (cloud, pose) in placed {
        for index in 0..cloud.len() {
            merged.push(pose.transform_point(&cloud.point(index)));
        }
    }

    // The columns every cloud has, in the first cloud's order. Order rather
    // than a set, so that merging the same clouds twice writes the same
    // file.
    for column in first.attributes() {
        let mut gathered = match &column.data {
            AttributeData::F32(_) => AttributeData::F32(Vec::with_capacity(total)),
            AttributeData::F64(_) => AttributeData::F64(Vec::with_capacity(total)),
            AttributeData::U8(_) => AttributeData::U8(Vec::with_capacity(total)),
            AttributeData::U16(_) => AttributeData::U16(Vec::with_capacity(total)),
            AttributeData::U32(_) => AttributeData::U32(Vec::with_capacity(total)),
            AttributeData::I32(_) => AttributeData::I32(Vec::with_capacity(total)),
        };
        let complete = placed.iter().all(|(cloud, _)| {
            cloud
                .attribute(&column.name)
                .is_some_and(|found| extend(&mut gathered, &found.data))
        });
        if !complete || gathered.len() != total {
            continue;
        }
        // The only way this fails is a length disagreement, which the line
        // above has just ruled out.
        let _ = merged.push_attribute(Attribute {
            name: column.name.clone(),
            data: gathered,
        });
    }
    merged
}

/// Appends one column to another, if the two are the same kind.
///
/// Returns whether it was: a mismatch means the name is shared and the
/// meaning is not, and a column of intensities appended to a column of
/// return numbers would be worse than no column at all.
fn extend(into: &mut AttributeData, from: &AttributeData) -> bool {
    match (into, from) {
        (AttributeData::F32(into), AttributeData::F32(from)) => into.extend_from_slice(from),
        (AttributeData::F64(into), AttributeData::F64(from)) => into.extend_from_slice(from),
        (AttributeData::U8(into), AttributeData::U8(from)) => into.extend_from_slice(from),
        (AttributeData::U16(into), AttributeData::U16(from)) => into.extend_from_slice(from),
        (AttributeData::U32(into), AttributeData::U32(from)) => into.extend_from_slice(from),
        (AttributeData::I32(into), AttributeData::I32(from)) => into.extend_from_slice(from),
        _ => return false,
    }
    true
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

    /// A merge places every cloud, keeps the columns they share, and drops
    /// the ones they do not.
    ///
    /// Three things are asserted because three things can go wrong quietly.
    /// The pose must be *applied* — a merge that concatenates raw
    /// coordinates produces a file that opens, looks like two scans side by
    /// side, and is wrong. A column present in both must survive with its
    /// values in the same order as the points. And a column present in only
    /// one must vanish rather than be padded, since a padded column is a
    /// measurement nobody made.
    #[test]
    fn a_merge_places_the_clouds_and_keeps_only_shared_columns() {
        let mut first = PointCloud::new();
        first.push(Vector3::new(1.0, 0.0, 0.0));
        first.push(Vector3::new(2.0, 0.0, 0.0));
        first
            .push_attribute(Attribute {
                name: "intensity".to_owned(),
                data: AttributeData::U16(vec![10, 20]),
            })
            .expect("two values for two points");
        first
            .push_attribute(Attribute {
                name: "returns".to_owned(),
                data: AttributeData::U8(vec![1, 1]),
            })
            .expect("two values for two points");

        let mut second = PointCloud::new();
        second.push(Vector3::new(0.0, 0.0, 0.0));
        second
            .push_attribute(Attribute {
                name: "intensity".to_owned(),
                data: AttributeData::U16(vec![30]),
            })
            .expect("one value for one point");

        let shift = Se3::from_parts(
            rigidity_core::lie::So3::identity(),
            Vector3::new(0.0, 5.0, 0.0),
        );
        let merged = merge(&[(&first, Se3::identity()), (&second, shift)]);

        assert_eq!(merged.len(), 3);
        assert!(
            (merged.point(2) - Vector3::new(0.0, 5.0, 0.0)).norm() < 1e-6,
            "the second cloud was written where it lay rather than where it was placed: {:?}",
            merged.point(2)
        );
        match merged.attribute("intensity").map(|column| &column.data) {
            Some(AttributeData::U16(values)) => assert_eq!(values, &[10, 20, 30]),
            other => panic!("intensity did not survive the merge: {other:?}"),
        }
        assert!(
            merged.attribute("returns").is_none(),
            "a column only one cloud had was carried into the merge"
        );
    }

    /// A staircase, and one wall closing off one end of it.
    ///
    /// A staircase is the shape this search exists for: shifted by one
    /// step it lies almost on top of itself, so ICP started anywhere near
    /// has a wrong minimum to fall into. The wall is what makes the right
    /// answer unique — an unbounded flight of steps has no right answer,
    /// and a test on one would be measuring an ambiguity rather than a
    /// search.
    ///
    /// Sampled with a jitter, for the reason recorded above: a regular
    /// lattice slides into itself and every residual on it measures the
    /// sampling.
    fn staircase() -> PointCloud {
        const STEPS: usize = 12;
        const TREAD: f64 = 0.30;
        const RISE: f64 = 0.17;
        let mut cloud = PointCloud::new();
        for step in 0..STEPS {
            let x0 = step as f64 * TREAD;
            let z0 = step as f64 * RISE;
            for i in 0..24 {
                for j in 0..24 {
                    let jitter = ((i * 31 + j * 13 + step * 7) % 11) as f64 * 0.002;
                    let across = j as f64 * 0.05 + jitter;
                    // The tread, and the riser that climbs to the next one.
                    cloud.push(Vector3::new(x0 + i as f64 * TREAD / 24.0, across, z0));
                    cloud.push(Vector3::new(
                        x0 + TREAD,
                        across,
                        z0 + i as f64 * RISE / 24.0,
                    ));
                }
            }
        }
        // The wall across the top, which no shift along the flight maps
        // onto itself.
        for i in 0..40 {
            for j in 0..40 {
                let jitter = ((i * 17 + j * 29) % 7) as f64 * 0.003;
                cloud.push(Vector3::new(
                    STEPS as f64 * TREAD,
                    j as f64 * 0.03 + jitter,
                    STEPS as f64 * RISE + i as f64 * 0.05,
                ));
            }
        }
        cloud
    }

    /// The search finds the basin that a cold start misses.
    ///
    /// Both halves are asserted, and the first is not a formality: if ICP
    /// from the identity happened to land correctly, the second assertion
    /// would pass while testing nothing at all.
    #[test]
    fn the_search_finds_a_basin_a_cold_start_misses() {
        const NOISE: f64 = 0.01;
        let params = PrepareParams {
            voxel: 0.03,
            neighbours: 12,
        };
        let fixed = prepare_cloud(&staircase(), &params).expect("the staircase prepares");
        // A heading nobody told the registration about: fifty degrees, and
        // a step and a half up the flight with it. Fifty rather than a
        // multiple of thirty on purpose — the search's net is twelve yaws,
        // so the nearest start is ten degrees away and the ICP has to close
        // the rest. A truth sitting exactly on a start would test nothing.
        let truth = Se3::from_parts(
            rigidity_core::lie::So3::exp(&Vector3::new(0.0, 0.0, 50f64.to_radians())),
            Vector3::new(0.45, 0.0, 0.255),
        );
        let moved = transform_cloud(&staircase(), &truth.inverse());
        let moving = prepare_cloud(&moved, &params).expect("the moved staircase prepares");

        let register = RegisterParams::default();
        let error = |pose: &Se3| (pose.translation() - truth.translation()).norm();

        let cold = register_pair_observed(&moving, &fixed, Se3::identity(), &register, |_| {
            ControlFlow::Continue(())
        });
        let cold_median = median_absolute_residual(&moving, &fixed, &cold.pose, &register)
            .expect("the cold start matched something");
        assert!(
            error(&cold.pose) > 0.10,
            "the cold start already landed correctly, at {} m — this scene no longer has \
             the failure the search is for",
            error(&cold.pose)
        );
        assert!(
            cold_median > NOISE,
            "the cold start is wrong and its median residual does not say so: {cold_median} m"
        );

        let search = SearchParams {
            radius: 0.4,
            screen: PrepareParams {
                voxel: 0.12,
                neighbours: 10,
            },
            ..SearchParams::default()
        };
        let found = register_globally(&moving, &fixed, &register, &search).expect("a candidate");
        assert!(
            error(&found.result.pose) < 0.02,
            "the search landed {} m from the truth",
            error(&found.result.pose)
        );
        assert!(
            found.median_residual <= NOISE,
            "the search's own verdict on itself is {} m, past the noise",
            found.median_residual
        );
        assert_eq!(found.starts, search.yaws * 9);
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
