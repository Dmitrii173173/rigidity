//! Scene geometry and the derivation of the null spaces.

use nalgebra::{Vector3, Vector6};
use rigidity_core::PointCloud;

use crate::rng::Rng;
use crate::{Scene, SceneParams};

/// Basis vector `index` in the ordering `ξ = [ρx, ρy, ρz, φx, φy, φz]`.
fn basis(index: usize) -> Vector6<f64> {
    let mut v = Vector6::zeros();
    v[index] = 1.0;
    v
}

/// The set of scenes whose answer is known.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SceneKind {
    /// A single plane.
    Plane,
    /// The lateral surface of a cylinder.
    Cylinder,
    /// A sphere.
    Sphere,
    /// Two non-parallel planes.
    TwoPlanes,
    /// A trihedral corner.
    Corner,
    /// A T-shaped weld joint.
    TeeJoint,
    /// A long corridor: a floor and two walls.
    Corridor,
}

impl SceneKind {
    /// Every scene.
    pub const ALL: [SceneKind; 7] = [
        Self::Plane,
        Self::Cylinder,
        Self::Sphere,
        Self::TwoPlanes,
        Self::Corner,
        Self::TeeJoint,
        Self::Corridor,
    ];

    /// A name for messages.
    pub fn name(self) -> &'static str {
        match self {
            Self::Plane => "plane",
            Self::Cylinder => "cylinder",
            Self::Sphere => "sphere",
            Self::TwoPlanes => "two planes",
            Self::Corner => "trihedral corner",
            Self::TeeJoint => "tee joint",
            Self::Corridor => "corridor",
        }
    }

    /// The analytical basis of the null space in the canonical pose.
    ///
    /// The derivation for each scene, from the Jacobian row
    /// `[nᵀ | (p × n)ᵀ]`:
    ///
    /// **Plane** `z = 0`, `n = (0,0,1)`, points `(a, b, 0)`. Here
    /// `p × n = (b, −a, 0)`, and requiring `ρz + b·φx − a·φy = 0` for all
    /// `a, b` gives `φx = φy = ρz = 0`. Free: `ρx`, `ρy`, `φz` — the
    /// in-plane translations and rotation about the normal, three degrees
    /// of freedom.
    ///
    /// **Cylinder** with axis `z`, points `(r cosθ, r sinθ, h)`,
    /// `n = (cosθ, sinθ, 0)`. Here `p × n = (−h sinθ, h cosθ, 0)`, and
    /// requiring `cosθ·(ρx + h·φy) + sinθ·(ρy − h·φx) = 0` for all `θ` and
    /// at least two distinct `h` gives `ρx = ρy = φx = φy = 0`. Free: `ρz`
    /// and `φz` — sliding along the axis and spinning about it.
    ///
    /// **Sphere** centred at the origin: `p = r·n`, hence
    /// `p × n = r·(n × n) = 0`, so the rotational block of every row
    /// vanishes identically. The normals cover all directions, so `ρ = 0`.
    /// Free: all three rotations. Note that the answer depends on the
    /// sphere's centre coinciding with the centre of rotation; a displaced
    /// sphere has a different basis.
    ///
    /// **Two planes** `z = 0` and `x = 0`. The first gives
    /// `φx = φy = ρz = 0`, the second `φy = φz = ρx = 0`. Together only
    /// `ρy` is free: translation along the line of intersection.
    ///
    /// **Trihedral corner** adds the plane `y = 0` with the condition
    /// `ρy = φx = φz = 0`. Nothing is left — full observability.
    ///
    /// **Tee joint** is the same pair of orthogonal planes as above, with
    /// realistic extents. Translation along the seam is free.
    ///
    /// **Corridor**: the floor `z = 0` gives `ρz = φx = φy = 0`, and the
    /// walls `x = ±s` give `ρx = φy = φz = 0`. Translation along the
    /// corridor is free.
    pub fn nullspace(self) -> Vec<Vector6<f64>> {
        match self {
            Self::Plane => vec![basis(0), basis(1), basis(5)],
            Self::Cylinder => vec![basis(2), basis(5)],
            Self::Sphere => vec![basis(3), basis(4), basis(5)],
            Self::TwoPlanes | Self::TeeJoint | Self::Corridor => vec![basis(1)],
            Self::Corner => Vec::new(),
        }
    }

    /// Dimension of the null space.
    pub fn nullspace_dimension(self) -> usize {
        self.nullspace().len()
    }
}

/// A rectangular patch of a plane.
struct Patch {
    center: Vector3<f64>,
    u: Vector3<f64>,
    v: Vector3<f64>,
    half_u: f64,
    half_v: f64,
    normal: Vector3<f64>,
}

fn axes() -> (Vector3<f64>, Vector3<f64>, Vector3<f64>) {
    (Vector3::x(), Vector3::y(), Vector3::z())
}

fn patches(kind: SceneKind, s: f64) -> Vec<Patch> {
    let (x, y, z) = axes();
    let patch = |center: Vector3<f64>,
                 u: Vector3<f64>,
                 v: Vector3<f64>,
                 half_u: f64,
                 half_v: f64,
                 normal: Vector3<f64>| Patch {
        center,
        u,
        v,
        half_u,
        half_v,
        normal,
    };

    match kind {
        SceneKind::Plane => vec![patch(Vector3::zeros(), x, y, s, s, z)],
        SceneKind::TwoPlanes => vec![
            patch(Vector3::zeros(), x, y, s, s, z),
            patch(Vector3::new(0.0, 0.0, s), y, z, s, s, x),
        ],
        SceneKind::Corner => vec![
            patch(Vector3::new(s, s, 0.0), x, y, s, s, z),
            patch(Vector3::new(0.0, s, s), y, z, s, s, x),
            patch(Vector3::new(s, 0.0, s), x, z, s, s, y),
        ],
        SceneKind::TeeJoint => vec![
            // Flange and web: the weld seam runs along y.
            patch(Vector3::zeros(), x, y, s, 4.0 * s, z),
            patch(Vector3::new(0.0, 0.0, s), y, z, 4.0 * s, s, x),
        ],
        SceneKind::Corridor => vec![
            patch(Vector3::zeros(), x, y, s, 10.0 * s, z),
            patch(Vector3::new(-s, 0.0, s), y, z, 10.0 * s, s, x),
            patch(Vector3::new(s, 0.0, s), y, z, 10.0 * s, s, -x),
        ],
        SceneKind::Cylinder | SceneKind::Sphere => Vec::new(),
    }
}

fn sample_patch(
    patch: &Patch,
    count: usize,
    rng: &mut Rng,
    points: &mut Vec<Vector3<f64>>,
    normals: &mut Vec<Vector3<f64>>,
) {
    for _ in 0..count {
        let position = patch.center
            + patch.u * (rng.symmetric() * patch.half_u)
            + patch.v * (rng.symmetric() * patch.half_v);
        points.push(position);
        normals.push(patch.normal);
    }
}

fn sample_cylinder(
    scale: f64,
    count: usize,
    rng: &mut Rng,
    points: &mut Vec<Vector3<f64>>,
    normals: &mut Vec<Vector3<f64>>,
) {
    let radius = scale * 0.5;
    for _ in 0..count {
        let angle = std::f64::consts::TAU * rng.unit();
        let height = rng.symmetric() * scale;
        let normal = Vector3::new(angle.cos(), angle.sin(), 0.0);
        points.push(Vector3::new(radius * normal.x, radius * normal.y, height));
        normals.push(normal);
    }
}

fn sample_sphere(
    scale: f64,
    count: usize,
    rng: &mut Rng,
    points: &mut Vec<Vector3<f64>>,
    normals: &mut Vec<Vector3<f64>>,
) {
    for _ in 0..count {
        // Three normal deviates give a uniform direction once normalised,
        // unlike uniform angles, which crowd points near the poles.
        let mut direction = Vector3::new(rng.normal(1.0), rng.normal(1.0), rng.normal(1.0));
        while direction.norm() < 1e-12 {
            direction = Vector3::new(rng.normal(1.0), rng.normal(1.0), rng.normal(1.0));
        }
        let normal = direction.normalize();
        points.push(normal * scale);
        normals.push(normal);
    }
}

pub(crate) fn generate(kind: SceneKind, params: SceneParams) -> Scene {
    let mut rng = Rng::new(params.seed);
    let mut points: Vec<Vector3<f64>> = Vec::new();
    let mut normals: Vec<Vector3<f64>> = Vec::new();

    match kind {
        SceneKind::Cylinder => sample_cylinder(
            params.scale,
            params.points_per_face,
            &mut rng,
            &mut points,
            &mut normals,
        ),
        SceneKind::Sphere => sample_sphere(
            params.scale,
            params.points_per_face,
            &mut rng,
            &mut points,
            &mut normals,
        ),
        _ => {
            for patch in patches(kind, params.scale) {
                sample_patch(
                    &patch,
                    params.points_per_face,
                    &mut rng,
                    &mut points,
                    &mut normals,
                );
            }
        }
    }

    if params.noise_sigma > 0.0 {
        for point in &mut points {
            *point += Vector3::new(
                rng.normal(params.noise_sigma),
                rng.normal(params.noise_sigma),
                rng.normal(params.noise_sigma),
            );
        }
    }

    let inlier_count = points.len();

    // An outlier is not a point displaced along its normal but a **wrong
    // correspondence**: a point matched to the wrong piece of surface. The
    // distinction is essential. Displacing along the normal changes the
    // residual but not the Jacobian row: `(p + δn) × n = p × n`.
    // Degeneracy is a property of the Jacobian alone, so an outlier must
    // bring a wrong normal with it as well.
    let outlier_count = (inlier_count as f64 * params.outlier_ratio).round() as usize;
    if outlier_count > 0 {
        let extent = params.outlier_extent * params.scale;
        let (mut min, mut max) = (points[0], points[0]);
        for point in &points {
            min = min.inf(point);
            max = max.sup(point);
        }
        for _ in 0..outlier_count {
            let position = Vector3::new(
                min.x - extent + rng.unit() * (max.x - min.x + 2.0 * extent),
                min.y - extent + rng.unit() * (max.y - min.y + 2.0 * extent),
                min.z - extent + rng.unit() * (max.z - min.z + 2.0 * extent),
            );
            let mut direction = Vector3::new(rng.normal(1.0), rng.normal(1.0), rng.normal(1.0));
            while direction.norm() < 1e-12 {
                direction = Vector3::new(rng.normal(1.0), rng.normal(1.0), rng.normal(1.0));
            }
            points.push(position);
            normals.push(direction.normalize());
        }
    }

    let mut cloud = PointCloud::with_capacity(points.len());
    for point in &points {
        cloud.push(*point);
    }

    Scene {
        kind,
        params,
        points,
        cloud,
        normals,
        inlier_count,
    }
}
