// Entrant: DRPM, by Hatleskog and Alexis. Their code, not our reading of it.
//
// `src/degeneracy.h` of github.com/ntnu-arl/drpm is included unmodified and
// called exactly as `src/example.cpp` calls it: build the Hessian in their
// convention, take its eigenvectors, estimate the noise, turn it into a
// per-direction probability. The only thing this file adds is reading the
// correspondences from a CSV instead of from their bundled `data.h`, so
// that the correspondences can be ours.
//
// Their state is [dr; dt] — rotation first — and ours is [rho; phi], so the
// two Hessians differ by a permutation of blocks. The permutation leaves the
// spectrum alone, which is why both sides can be printed sorted by
// eigenvalue and compared row for row without either being rewritten.
//
//   cl /std:c++14 /EHsc /O2 /I <eigen> /I <boost-parent> /I <drpm/src> \
//      drpm_bridge.cpp
//   drpm_bridge pair.csv

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "degeneracy.h"

namespace {

struct Input {
  degeneracy::VectorVector3<double> points;
  degeneracy::VectorVector3<double> normals;
  std::vector<double> weights;
  degeneracy::VectorMatrix3<double> covariances;
  double point_sigma = 0.03;
  double snr = 10.0;
};

Input Read(const std::string& path) {
  Input in;
  std::ifstream file(path);
  if (!file) {
    std::cerr << "cannot open " << path << "\n";
    std::exit(1);
  }
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty()) continue;
    if (line[0] == '#') {
      std::istringstream head(line.substr(1));
      std::string name;
      double value = 0.0;
      head >> name >> value;
      if (name == "point_sigma") in.point_sigma = value;
      if (name == "signal_to_noise") in.snr = value;
      continue;
    }
    if (line[0] == 'p') continue;  // the column header
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream row(line);
    double px, py, pz, nx, ny, nz, w, c00, c01, c02, c11, c12, c22;
    if (!(row >> px >> py >> pz >> nx >> ny >> nz >> w >> c00 >> c01 >> c02 >> c11 >> c12 >> c22)) {
      continue;
    }
    in.points.emplace_back(px, py, pz);
    in.normals.emplace_back(nx, ny, nz);
    in.weights.push_back(w);
    Eigen::Matrix3d c;
    c << c00, c01, c02, c01, c11, c12, c02, c12, c22;
    in.covariances.push_back(c);
  }
  return in;
}

// Verbatim from their example.cpp, so that the matrix their functions are
// given is the matrix their example would have given them.
Eigen::Matrix<double, 6, 6> ComputeHessian(const degeneracy::VectorVector3<double>& points,
                                           const degeneracy::VectorVector3<double>& normals,
                                           const std::vector<double>& weights) {
  const size_t nPoints = points.size();
  Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero(6, 6);
  for (size_t i = 0; i < nPoints; i++) {
    const Eigen::Vector3d point = points[i];
    const Eigen::Vector3d normal = normals[i];
    const Eigen::Vector3d pxn = point.cross(normal);
    const double w = std::sqrt(weights[i]);
    Eigen::Matrix<double, 6, 1> v;
    v.head(3) = w * pxn;
    v.tail(3) = w * normal;
    H += v * v.transpose();
  }
  return H;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: drpm_bridge <pair.csv>\n";
    return 2;
  }
  const Input in = Read(argv[1]);
  std::cerr << "correspondences " << in.points.size() << "\n";
  if (in.points.empty()) return 1;

  const auto H = ComputeHessian(in.points, in.normals, in.weights);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eigensolver(H);
  const auto eigenvectors = eigensolver.eigenvectors();
  const auto eigenvalues = eigensolver.eigenvalues();

  Eigen::Matrix<double, 6, 6> noise_mean;
  Eigen::Matrix<double, 6, 1> noise_variance;
  std::tie(noise_mean, noise_variance) = degeneracy::ComputeNoiseEstimate<double, double>(
      in.points, in.normals, in.weights, in.covariances, eigenvectors, in.point_sigma);
  const Eigen::Matrix<double, 6, 1> probabilities =
      degeneracy::ComputeSignalToNoiseProbabilities<double>(H, noise_mean, noise_variance,
                                                            eigenvectors, in.snr);

  // Eigen's SelfAdjointEigenSolver already returns them ascending, but the
  // ordering is asserted rather than assumed: the whole comparison rests on
  // the two sides being lined up by eigenvalue.
  std::vector<size_t> order(6);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](size_t a, size_t b) { return eigenvalues[a] < eigenvalues[b]; });

  if (std::getenv("RIGIDITY_PROB_DEBUG")) {
    std::fprintf(stderr, "  theirs noise trace %.6e\n", noise_mean.trace());
    std::fprintf(stderr, "THEIRS_NOISE\n");
    for (int r = 0; r < 6; r++) {
      for (int c = 0; c < 6; c++) std::fprintf(stderr, "%.9e ", noise_mean(r, c));
      std::fprintf(stderr, "\n");
    }
    for (size_t k = 0; k < 6; k++) {
      const Eigen::Matrix<double, 6, 1> u = eigenvectors.col(k);
      const double measurement = (u.transpose() * H * u).value();
      const double expected = (u.transpose() * noise_mean * u).value();
      std::fprintf(stderr, "  dir %zu: eigen %.6e mean %.6e spread %.6e thr %.6e\n", k,
                   measurement, expected, std::sqrt(noise_variance[k]),
                   measurement / (1.0 + in.snr));
    }
  }
  std::printf("theirs eigenvalues ");
  for (size_t k : order) std::printf("%.6e ", eigenvalues[k]);
  std::printf("\ntheirs probabilities ");
  for (size_t k : order) std::printf("%.6f ", probabilities[k]);
  std::printf("\n");
  return 0;
}
