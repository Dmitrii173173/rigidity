//! Synthetic scenes with analytically known null spaces.
//!
//! These are the project's measuring standard. The degeneracy detector is
//! not judged by eye but against an answer derived by hand: for every
//! scene it is known in advance how many degrees of freedom the geometry
//! fails to determine, and which ones.
//!
//! # Convention
//!
//! Null spaces are written in `ξ = [ρ; φ]` coordinates and refer to
//! rotation about the **coordinate origin**. Scenes are built in a
//! canonical pose where the answer takes a simple form. Moving a scene by
//! a transform `T` maps the null space to `Adj(T)·v` — checked by its own
//! test, which incidentally confirms the adjoint implementation.
//!
//! # What "known null space" means
//!
//! A row of the point-to-plane Jacobian is `[nᵀ | (p × n)ᵀ]`. A vector `v`
//! is unobservable when `[nᵀ | (p × n)ᵀ]·v = 0` for **every** point of the
//! scene. That condition is solved by hand below for each surface.

pub mod rng;
mod surfaces;

use nalgebra::{Vector3, Vector6};
use rigidity_core::PointCloud;

pub use surfaces::SceneKind;

/// Generation parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneParams {
    /// Points per face (or over the whole surface, for the cylinder and
    /// the sphere).
    pub points_per_face: usize,
    /// Characteristic size of the scene, metres.
    pub scale: f64,
    /// Standard deviation of isotropic Gaussian noise on positions.
    ///
    /// Isotropic rather than along the normal: the latter is a model that
    /// happens to suit point-to-plane, and is less honest about a real
    /// sensor.
    pub noise_sigma: f64,
    /// Fraction of outliers among the points.
    pub outlier_ratio: f64,
    /// How far an outlier strays from the surface, in units of `scale`.
    pub outlier_extent: f64,
    /// Generator seed.
    pub seed: u64,
}

impl Default for SceneParams {
    fn default() -> Self {
        Self {
            points_per_face: 2_000,
            scale: 1.0,
            noise_sigma: 0.0,
            outlier_ratio: 0.0,
            outlier_extent: 0.25,
            seed: 0x1234_5678_9ABC_DEF0,
        }
    }
}

/// A generated scene.
#[derive(Debug, Clone)]
pub struct Scene {
    /// Which surface.
    pub kind: SceneKind,
    /// The parameters it was built with.
    pub params: SceneParams,
    /// Exact point coordinates in `f64`.
    ///
    /// The scene is a mathematical object; the cloud is its in-memory
    /// representation. Keeping them apart is mandatory, because `f32`
    /// storage destroys exact identities on curved surfaces. On a cylinder
    /// `(p × n)_z` is identically zero, yet after quantisation it becomes
    /// about 3·10⁻⁸ of the scene scale. Analytical claims are checked
    /// against this field; all processing runs on [`cloud`](Self::cloud).
    pub points: Vec<Vector3<f64>>,
    /// The points in working form: `f32` relative to an origin.
    pub cloud: PointCloud,
    /// The exact analytical normal at each point.
    ///
    /// Analytical, not estimated from neighbours: normal-estimation error
    /// is a separate source of inaccuracy and must not be mixed into
    /// geometric degeneracy.
    pub normals: Vec<Vector3<f64>>,
    /// The first `inlier_count` points lie on the surface; the rest are
    /// outliers.
    ///
    /// The null-space guarantee applies to inliers only: an outlier is not
    /// on the surface, and the declared vector does not annihilate its
    /// Jacobian row.
    pub inlier_count: usize,
}

impl Scene {
    /// Builds the scene.
    pub fn generate(kind: SceneKind, params: SceneParams) -> Self {
        surfaces::generate(kind, params)
    }

    /// The analytical basis of the null space.
    pub fn nullspace(&self) -> Vec<Vector6<f64>> {
        self.kind.nullspace()
    }

    /// Dimension of the null space.
    pub fn nullspace_dimension(&self) -> usize {
        self.kind.nullspace_dimension()
    }

    /// Number of points.
    pub fn len(&self) -> usize {
        self.cloud.len()
    }

    /// Whether the scene is empty.
    pub fn is_empty(&self) -> bool {
        self.cloud.is_empty()
    }

    /// Splits the scene into two partially overlapping clouds.
    ///
    /// A fraction `overlap` of the points goes into both clouds; the rest
    /// is divided evenly. Needed for source/target pairs in registration:
    /// full overlap is an unrealistic and far too easy case.
    pub fn split_with_overlap(&self, overlap: f64, seed: u64) -> (PointCloud, PointCloud) {
        let overlap = overlap.clamp(0.0, 1.0);
        let exclusive = (1.0 - overlap) * 0.5;
        let mut rng = rng::Rng::new(seed);
        let mut source = PointCloud::with_origin(self.cloud.origin());
        let mut target = PointCloud::with_origin(self.cloud.origin());
        for i in 0..self.cloud.len() {
            let point = self.cloud.point(i);
            let draw = rng.unit();
            if draw < overlap {
                source.push(point);
                target.push(point);
            } else if draw < overlap + exclusive {
                source.push(point);
            } else {
                target.push(point);
            }
        }
        (source, target)
    }
}
