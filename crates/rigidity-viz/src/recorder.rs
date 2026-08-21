//! The Rerun log sink.

use std::path::Path;

use rigidity_core::PointCloud;
use rigidity_core::lie::Se3;
use rigidity_core::observability::{Conditioning, Observability, ObservabilityCriteria};

/// A visualisation error.
#[derive(Debug, thiserror::Error)]
pub enum VizError {
    /// The recording stream could not be opened.
    #[error("Rerun: {0}")]
    Stream(String),
}

/// The log sink.
///
/// Everything written through it lands on a shared "iteration" timeline:
/// the clouds, the pose, the six spreads and the condition number. What
/// you see is not a final picture but how the solution approached the
/// answer, and what the spectrum did along the way.
pub struct Recorder {
    stream: rerun::RecordingStream,
}

impl Recorder {
    /// Spawns the viewer and streams into it directly.
    ///
    /// The application name must be `'static`: Rerun keeps it inside its
    /// own identifier for the lifetime of the stream.
    pub fn spawn(application: &'static str) -> Result<Self, VizError> {
        rerun::RecordingStreamBuilder::new(application)
            .spawn()
            .map(|stream| Self { stream })
            .map_err(|error| VizError::Stream(error.to_string()))
    }

    /// Writes to an `.rrd` file that the viewer can open later.
    pub fn save(application: &'static str, path: &Path) -> Result<Self, VizError> {
        rerun::RecordingStreamBuilder::new(application)
            .save(path)
            .map(|stream| Self { stream })
            .map_err(|error| VizError::Stream(error.to_string()))
    }

    /// Moves the recording to the given iteration.
    pub fn set_iteration(&self, iteration: i64) {
        self.stream.set_time_sequence("iteration", iteration);
    }

    /// Logs a point cloud.
    ///
    /// Downsampling first is not required but is worth doing: the viewer
    /// slows noticeably on clouds of several million points, as its own
    /// documentation admits.
    pub fn log_cloud(
        &self,
        entity: &str,
        cloud: &PointCloud,
        colour: [u8; 3],
    ) -> Result<(), VizError> {
        let positions: Vec<[f32; 3]> = (0..cloud.len())
            .map(|index| {
                let point = cloud.point(index);
                [point.x as f32, point.y as f32, point.z as f32]
            })
            .collect();
        self.stream
            .log(
                entity.to_owned(),
                &rerun::Points3D::new(positions)
                    .with_colors([rerun::Color::from_rgb(colour[0], colour[1], colour[2])])
                    .with_radii([0.01_f32]),
            )
            .map_err(|error| VizError::Stream(error.to_string()))
    }

    /// Logs a pose as a coordinate transform.
    pub fn log_pose(&self, entity: &str, pose: &Se3) -> Result<(), VizError> {
        let translation = pose.translation();
        let matrix = pose.rotation().matrix();
        let columns = [
            [
                matrix[(0, 0)] as f32,
                matrix[(1, 0)] as f32,
                matrix[(2, 0)] as f32,
            ],
            [
                matrix[(0, 1)] as f32,
                matrix[(1, 1)] as f32,
                matrix[(2, 1)] as f32,
            ],
            [
                matrix[(0, 2)] as f32,
                matrix[(1, 2)] as f32,
                matrix[(2, 2)] as f32,
            ],
        ];
        self.stream
            .log(
                entity.to_owned(),
                &rerun::Transform3D::from_translation([
                    translation.x as f32,
                    translation.y as f32,
                    translation.z as f32,
                ])
                .with_mat3x3(columns),
            )
            .map_err(|error| VizError::Stream(error.to_string()))
    }

    /// Logs the conditioning report: six spreads, the condition number
    /// and an arrow along the worst direction.
    ///
    /// The arrow is the point of this. A number in the log says "something
    /// is off"; the arrow shows exactly which way the cloud can drift
    /// without the solver noticing.
    pub fn log_conditioning(
        &self,
        entity: &str,
        conditioning: &Conditioning,
        criteria: &ObservabilityCriteria,
    ) -> Result<(), VizError> {
        let spread = conditioning.uncertainty(criteria.noise_sigma);
        for (index, value) in spread.iter().enumerate() {
            // Infinity is useless in a plot; substitute a visible ceiling.
            let plotted = if value.is_finite() {
                *value
            } else {
                criteria.tolerance * 1e3
            };
            self.stream
                .log(
                    format!("{entity}/spread/{index}"),
                    &rerun::Scalars::single(plotted),
                )
                .map_err(|error| VizError::Stream(error.to_string()))?;
        }
        self.stream
            .log(
                format!("{entity}/condition_number"),
                &rerun::Scalars::single(conditioning.condition_number()),
            )
            .map_err(|error| VizError::Stream(error.to_string()))?;

        let states = conditioning.classify(criteria);
        if let Some(worst) = states
            .iter()
            .position(|state| *state == Observability::Low)
            .or_else(|| states.iter().position(|s| *s == Observability::Medium))
        {
            let direction = conditioning.direction_in_world(worst);
            let centre = conditioning.centre();
            let length = conditioning.radius_of_gyration();
            self.stream
                .log(
                    format!("{entity}/unobservable"),
                    &rerun::Arrows3D::from_vectors([[
                        (direction[0] * length) as f32,
                        (direction[1] * length) as f32,
                        (direction[2] * length) as f32,
                    ]])
                    .with_origins([[centre.x as f32, centre.y as f32, centre.z as f32]])
                    .with_colors([rerun::Color::from_rgb(235, 104, 52)]),
                )
                .map_err(|error| VizError::Stream(error.to_string()))?;
        }
        Ok(())
    }
}
