//! Pose-graph optimisation weighted by each edge's own conditioning.
//!
//! Every pairwise registration already produces `JᵀWJ` at its solution, and
//! every pose-graph package in this space takes that matrix at face value.
//! The rest of this project exists to say why that is wrong: on a corridor,
//! the matrix is confident about a direction the geometry never determined,
//! and the residual agrees with it. An edge built from such a registration
//! should carry no weight along the corridor — not a small weight, not one
//! deflated by a fudge factor, *none* — and the conditioning analysis is
//! what can say which direction that is.
//!
//! That is the whole content of this crate. Everything else here is the
//! ordinary machinery a pose graph needs in order for the weighting to have
//! somewhere to act: nodes, edges, Gauss–Newton on `SE(3)`, an anchor.
//!
//! # Frames, and the one thing that will go wrong if they are misread
//!
//! A node's pose is *world-from-scan*: it carries that scan's own
//! coordinates into the survey. An edge from `i` to `j` measures
//! `Z ≈ T_i⁻¹·T_j` — scan `j`'s coordinates expressed in scan `i`'s — which
//! is exactly what [`rigidity_core::icp`] returns when `j` is the source and
//! `i` is the target.
//!
//! The edge's information matrix lives in the tangent at `Z`, under a
//! **left** perturbation, in scan `i`'s frame. That is not a choice made
//! here: the ICP updates its pose as `T ← exp(Δξ)·T` and builds its
//! Jacobian rows from points in the target's frame, so `IcpResult::
//! information` is already in those coordinates and
//! [`weighted_information`] produces its replacement in the same ones. An
//! information matrix in the wrong frame does not fail loudly — it
//! converges to a slightly wrong answer, which is the failure mode this
//! paragraph is here to prevent.
//!
//! Nodes are perturbed on the **right**, `T ← T·exp(δ)`, because that keeps
//! each increment in the body frame of its own scan, where the
//! measurements were taken.
//!
//! # Determinism
//!
//! Single-threaded, and the dense solve is a fixed sequence of operations
//! on a fixed matrix, so the result does not depend on a thread count that
//! does not exist. A few hundred poses is a 1200×1200 Cholesky, which is
//! milliseconds; sparse storage waits until a survey asks for it.

use rigidity_core::lie::{Se3, inverse_right_jacobian_se3};
use rigidity_core::nalgebra::{DMatrix, DVector, Matrix6, Vector6};
use rigidity_core::observability::{Conditioning, Observability, ObservabilityCriteria};

/// One measured relative pose, and how much of it to believe.
#[derive(Debug, Clone, Copy)]
pub struct Edge {
    /// The node the measurement is expressed in.
    pub from: usize,
    /// The node it measures.
    pub to: usize,
    /// `Z`: scan `to`'s coordinates in scan `from`'s frame.
    pub measurement: Se3,
    /// The inverse covariance of `Z`, in scan `from`'s frame.
    ///
    /// Either [`weighted_information`], which is the point of this crate,
    /// or `IcpResult::information` for the naive comparison the gate makes.
    pub information: Matrix6<f64>,
}

/// What can be wrong with a graph.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum GraphError {
    /// An edge names a node that is not there.
    #[error("edge {edge} names node {node}, and the graph has {nodes}")]
    NoSuchNode {
        /// Which edge.
        edge: usize,
        /// The index it named.
        node: usize,
        /// How many there are.
        nodes: usize,
    },
    /// An edge joins a node to itself.
    ///
    /// Not merely useless: its two Jacobian blocks land in the same place
    /// and cancel, so it contributes a row of zeros and makes the system
    /// harder to solve while measuring nothing.
    #[error("edge {edge} joins node {node} to itself")]
    SelfLoop {
        /// Which edge.
        edge: usize,
        /// The node on both ends.
        node: usize,
    },
    /// The anchor is not a node.
    #[error("the anchor is node {anchor}, and the graph has {nodes}")]
    NoSuchAnchor {
        /// The index given.
        anchor: usize,
        /// How many there are.
        nodes: usize,
    },
    /// The normal equations could not be factorised at any damping.
    ///
    /// Reached only when the system is degenerate in a way damping cannot
    /// repair, which in practice means a graph whose edges leave part of it
    /// unconnected to the anchor.
    #[error("the normal equations are singular at damping {damping:e}")]
    Singular {
        /// The largest damping that was tried.
        damping: f64,
    },
}

/// How hard to try.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OptimiseParams {
    /// Which node is held fixed.
    ///
    /// A pose graph determines its nodes only up to a common rigid motion:
    /// six directions of the system are free no matter how many edges there
    /// are. Holding one node still is how that gauge freedom is removed,
    /// and the survey's coordinates then mean "relative to this scan".
    pub anchor: usize,
    /// Stop after this many accepted steps.
    pub max_iterations: usize,
    /// Stop when a step moves every node less than this, in the units of
    /// the algebra — metres, and radians at one metre.
    pub step_tolerance: f64,
    /// Where the Levenberg damping starts.
    ///
    /// Damping is not a luxury here. Zeroing the unobservable directions of
    /// an edge is the entire point of the crate, and it can leave the whole
    /// system rank-deficient — a survey where nothing at all constrains one
    /// direction is a survey this crate should still return an answer for,
    /// rather than a factorisation error.
    pub initial_damping: f64,
}

impl Default for OptimiseParams {
    fn default() -> Self {
        Self {
            anchor: 0,
            max_iterations: 100,
            step_tolerance: 1e-10,
            initial_damping: 1e-9,
        }
    }
}

/// What an optimisation did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Report {
    /// How many steps were accepted.
    pub iterations: usize,
    /// Whether it stopped because the step became small rather than
    /// because it ran out of iterations.
    pub converged: bool,
    /// The cost before and after.
    pub cost: [f64; 2],
}

/// What the edges add up to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    /// How many nodes at least one edge touches.
    pub joined: usize,
    /// How many independent loops the edges form.
    ///
    /// Zero is the number that matters. A survey with no closure is a tree:
    /// its residual is zero at whatever answer it gives, because no two
    /// measurements are ever compared, and every error made along the way is
    /// still in the answer.
    pub closures: usize,
    /// How many joined nodes no chain of edges connects to the anchor.
    ///
    /// Anything above zero makes the normal equations singular, and the
    /// survey has more than one piece.
    pub adrift: usize,
}

/// Nodes, edges, and the optimisation over them.
#[derive(Debug, Clone, Default)]
pub struct PoseGraph {
    poses: Vec<Se3>,
    edges: Vec<Edge>,
}

impl PoseGraph {
    /// A graph with these nodes and no edges.
    pub fn new(poses: Vec<Se3>) -> Self {
        Self {
            poses,
            edges: Vec::new(),
        }
    }

    /// The nodes, in order.
    pub fn poses(&self) -> &[Se3] {
        &self.poses
    }

    /// The edges, in the order they were added.
    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    /// Adds an edge, refusing one that names a node that is not there.
    pub fn push(&mut self, edge: Edge) -> Result<(), GraphError> {
        let nodes = self.poses.len();
        let index = self.edges.len();
        for node in [edge.from, edge.to] {
            if node >= nodes {
                return Err(GraphError::NoSuchNode {
                    edge: index,
                    node,
                    nodes,
                });
            }
        }
        if edge.from == edge.to {
            return Err(GraphError::SelfLoop {
                edge: index,
                node: edge.from,
            });
        }
        self.edges.push(edge);
        Ok(())
    }

    /// What shape the edges make, which decides what a solve can do.
    ///
    /// Reported rather than discovered by failing: a survey with a node
    /// nothing joins to the anchor is one the normal equations cannot
    /// factorise, and finding that out from `GraphError::Singular` after the
    /// solve is a worse way to learn it than being told before.
    pub fn shape(&self, anchor: usize) -> Shape {
        let nodes = self.poses.len();
        let mut neighbours: Vec<Vec<usize>> = vec![Vec::new(); nodes];
        for edge in &self.edges {
            neighbours[edge.from].push(edge.to);
            neighbours[edge.to].push(edge.from);
        }
        let touched: Vec<bool> = neighbours.iter().map(|list| !list.is_empty()).collect();

        // Components over the nodes an edge touches, by breadth-first walk.
        // Sorted work list rather than a hash set, so the traversal order —
        // and therefore nothing at all — depends on a hasher.
        let mut seen = vec![false; nodes];
        let mut components = 0;
        for start in 0..nodes {
            if !touched[start] || seen[start] {
                continue;
            }
            components += 1;
            let mut queue = vec![start];
            seen[start] = true;
            while let Some(node) = queue.pop() {
                for next in &neighbours[node] {
                    if !seen[*next] {
                        seen[*next] = true;
                        queue.push(*next);
                    }
                }
            }
        }

        // Which of them the anchor can be reached from. The anchor itself
        // may be untouched — a survey of edges that all avoid it — and then
        // nothing is anchored and everything is adrift.
        let mut reachable = vec![false; nodes];
        if anchor < nodes {
            reachable[anchor] = true;
            let mut queue = vec![anchor];
            while let Some(node) = queue.pop() {
                for next in &neighbours[node] {
                    if !reachable[*next] {
                        reachable[*next] = true;
                        queue.push(*next);
                    }
                }
            }
        }

        let joined = touched.iter().filter(|t| **t).count();
        Shape {
            joined,
            // The cyclomatic number: how many edges could be removed before
            // the graph stops being connected the way it is. Zero means
            // every measurement is believed exactly because nothing
            // contradicts it — which is what a survey walked as a chain is,
            // and why it drifts.
            closures: (self.edges.len() + components).saturating_sub(joined),
            adrift: (0..nodes)
                .filter(|node| touched[*node] && !reachable[*node])
                .count(),
        }
    }

    /// The disagreement on one edge: `log(T_i⁻¹·T_j·Z⁻¹)`.
    ///
    /// Zero when the two nodes sit exactly as the measurement says. The
    /// ordering — the measurement inverted on the *right* — is what puts
    /// the residual in the same frame and the same left-perturbation
    /// convention as the information matrix beside it.
    pub fn residual(&self, edge: &Edge) -> Vector6<f64> {
        self.error(edge).log()
    }

    fn error(&self, edge: &Edge) -> Se3 {
        self.poses[edge.from].inverse() * self.poses[edge.to] * edge.measurement.inverse()
    }

    /// `Σ rᵀΛr` over the edges.
    pub fn cost(&self) -> f64 {
        self.edges
            .iter()
            .map(|edge| {
                let r = self.residual(edge);
                (r.transpose() * edge.information * r)[(0, 0)]
            })
            .sum()
    }

    /// Levenberg-damped Gauss–Newton until the steps stop mattering.
    pub fn optimise(&mut self, params: &OptimiseParams) -> Result<Report, GraphError> {
        let nodes = self.poses.len();
        if params.anchor >= nodes {
            return Err(GraphError::NoSuchAnchor {
                anchor: params.anchor,
                nodes,
            });
        }

        let before = self.cost();
        let mut damping = params.initial_damping;
        let mut iterations = 0;
        let mut converged = false;

        while iterations < params.max_iterations {
            let (hessian, gradient) = self.normal_equations(params.anchor);
            let mut step = None;
            // Ten increases of the damping, each by a factor of ten: from
            // 1e-9 that reaches 1e1, which is far past the point where the
            // system is dominated by the damping and the step is a tiny
            // gradient descent. Failing beyond that is a graph problem, not
            // a conditioning one.
            for _ in 0..12 {
                let mut damped = hessian.clone();
                for index in 0..damped.nrows() {
                    damped[(index, index)] += damping;
                }
                if let Some(cholesky) = damped.cholesky() {
                    step = Some(cholesky.solve(&(-&gradient)));
                    break;
                }
                damping *= 10.0;
            }
            let Some(step) = step else {
                return Err(GraphError::Singular { damping });
            };

            let previous = self.poses.clone();
            self.apply(&step, params.anchor);
            let after = self.cost();
            if after.is_finite() && after < self.cost_of(&previous) {
                iterations += 1;
                damping = (damping * 0.1).max(f64::MIN_POSITIVE);
                if step.amax() < params.step_tolerance {
                    converged = true;
                    break;
                }
            } else {
                // Rejected: the linearisation was not good enough at this
                // damping, so put the poses back and lean harder on the
                // gradient.
                self.poses = previous;
                damping *= 10.0;
                if damping > 1e12 {
                    converged = true;
                    break;
                }
            }
        }

        Ok(Report {
            iterations,
            converged,
            cost: [before, self.cost()],
        })
    }

    fn cost_of(&self, poses: &[Se3]) -> f64 {
        let mut probe = self.clone();
        probe.poses = poses.to_vec();
        probe.cost()
    }

    /// `H = ΣJᵀΛJ` and `b = ΣJᵀΛr`, over the nodes that are free to move.
    fn normal_equations(&self, anchor: usize) -> (DMatrix<f64>, DVector<f64>) {
        let free = self.poses.len() - 1;
        let mut hessian = DMatrix::zeros(6 * free, 6 * free);
        let mut gradient = DVector::zeros(6 * free);
        // The anchor has no block, so every node after it shifts down one.
        let slot = |node: usize| -> Option<usize> {
            match node.cmp(&anchor) {
                std::cmp::Ordering::Less => Some(node),
                std::cmp::Ordering::Equal => None,
                std::cmp::Ordering::Greater => Some(node - 1),
            }
        };

        for edge in &self.edges {
            let error = self.error(edge);
            let residual = error.log();
            // The right Jacobian's inverse is what turns a perturbation of
            // the group element into a perturbation of its logarithm.
            // Approximating it by the identity — which plenty of
            // implementations do — is exact only where the residual is
            // already zero, which is the one place the answer does not
            // matter.
            let lift = inverse_right_jacobian_se3(&residual);
            let jacobian_from = -lift * error.inverse().adjoint();
            let jacobian_to = lift * edge.measurement.adjoint();

            let blocks = [(edge.from, jacobian_from), (edge.to, jacobian_to)];
            for (node, jacobian) in blocks {
                let Some(row) = slot(node) else { continue };
                let weighted = jacobian.transpose() * edge.information;
                let contribution = weighted * residual;
                for axis in 0..6 {
                    gradient[6 * row + axis] += contribution[axis];
                }
                for (other, other_jacobian) in blocks {
                    let Some(column) = slot(other) else { continue };
                    let block = weighted * other_jacobian;
                    for r in 0..6 {
                        for c in 0..6 {
                            hessian[(6 * row + r, 6 * column + c)] += block[(r, c)];
                        }
                    }
                }
            }
        }
        (hessian, gradient)
    }

    fn apply(&mut self, step: &DVector<f64>, anchor: usize) {
        let mut row = 0;
        for (index, pose) in self.poses.iter_mut().enumerate() {
            if index == anchor {
                continue;
            }
            let delta = Vector6::new(
                step[6 * row],
                step[6 * row + 1],
                step[6 * row + 2],
                step[6 * row + 3],
                step[6 * row + 4],
                step[6 * row + 5],
            );
            *pose = *pose * Se3::exp(&delta);
            row += 1;
        }
    }
}

/// The information an edge's own conditioning justifies.
///
/// This is the crate's reason to exist. `IcpResult::information` is
/// `JᵀWJ`, which is a statement about how tightly the surfaces agreed; it
/// says nothing about whether agreeing tightly *meant* anything, and on a
/// corridor it does not. This builds the matrix again from the spectrum:
///
/// ```text
/// Λ = Σ  vᵢ vᵢᵀ / spreadᵢ²      over the directions the geometry determined
/// ```
///
/// and the directions it did not determine are simply absent from the sum.
/// Not down-weighted — absent. A direction whose predicted spread exceeds
/// the tolerance is one where the registration's answer is arbitrary, and
/// an arbitrary number given a small weight still pulls a survey towards
/// itself; given no weight it is what it is, which is no information.
///
/// Which directions those are comes from
/// [`Conditioning::classify`](rigidity_core::observability::Conditioning::classify)
/// rather than from comparing the spread here, so that the graph and the
/// report the user reads can never disagree about the same edge.
///
/// The calibration is the caller's business. On real data the predicted
/// spread is optimistic — the project measured about seventeenfold — but
/// the correction belongs to `criteria.noise_sigma`, where the rest of the
/// project already applies it, and not to a constant buried in here.
pub fn weighted_information(
    conditioning: &Conditioning,
    criteria: &ObservabilityCriteria,
) -> Matrix6<f64> {
    // The spectrum lives in normalised coordinates and an edge lives in
    // world ones. `to_normalised` is the linear map between them; it is
    // applied to the six basis vectors rather than rebuilt from the centre
    // and the radius of gyration, so this cannot drift away from the
    // transform the report itself uses.
    let mut to_normalised = Matrix6::zeros();
    for axis in 0..6 {
        let mut basis = Vector6::zeros();
        basis[axis] = 1.0;
        to_normalised.set_column(axis, &conditioning.to_normalised(basis));
    }

    let spreads = conditioning.uncertainty(criteria.noise_sigma);
    let observable = conditioning.classify(criteria);
    let mut normalised = Matrix6::zeros();
    for index in 0..6 {
        if observable[index] != Observability::High {
            continue;
        }
        let spread = spreads[index];
        if !(spread.is_finite() && spread > 0.0) {
            continue;
        }
        let direction = conditioning.direction(index);
        normalised += direction * direction.transpose() / (spread * spread);
    }

    to_normalised.transpose() * normalised * to_normalised
}
