// Участник сравнения: PCL. Протокол — PROTOCOL.md.

#include <chrono>
#include <cstdio>
#include <fstream>
#include <vector>

#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/registration/icp.h>

namespace {

constexpr float kVoxel = 0.02F;
constexpr int kNeighbours = 16;
constexpr double kMaxDistance = 0.5;
const char* kDirectory = "bench-external/data";

using Cloud = pcl::PointCloud<pcl::PointXYZ>;
using CloudNormal = pcl::PointCloud<pcl::PointNormal>;

Eigen::Matrix4d LoadTruth() {
  std::ifstream input(std::string(kDirectory) + "/truth.txt");
  Eigen::Matrix4d truth;
  for (int row = 0; row < 4; ++row)
    for (int column = 0; column < 4; ++column) input >> truth(row, column);
  return truth;
}

double TranslationError(const Eigen::Matrix4d& found,
                        const Eigen::Matrix4d& truth) {
  const Eigen::Matrix4d error = truth * found.inverse();
  return error.block<3, 1>(0, 3).norm();
}

Cloud::Ptr Load(const std::string& name) {
  Cloud::Ptr cloud(new Cloud);
  pcl::io::loadPLYFile(std::string(kDirectory) + "/" + name, *cloud);
  return cloud;
}

Cloud::Ptr Downsample(const Cloud::Ptr& cloud) {
  Cloud::Ptr reduced(new Cloud);
  pcl::VoxelGrid<pcl::PointXYZ> filter;
  filter.setInputCloud(cloud);
  filter.setLeafSize(kVoxel, kVoxel, kVoxel);
  filter.filter(*reduced);
  return reduced;
}

// Нормали считаются только у цели: связь «точка — плоскость» больше
// ничего не требует, и у остальных участников так же.
CloudNormal::Ptr WithNormals(const Cloud::Ptr& cloud, bool estimate) {
  CloudNormal::Ptr result(new CloudNormal);
  pcl::copyPointCloud(*cloud, *result);
  if (!estimate) return result;

  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> estimator;
  estimator.setInputCloud(cloud);
  estimator.setSearchMethod(
      pcl::search::KdTree<pcl::PointXYZ>::Ptr(new pcl::search::KdTree<pcl::PointXYZ>));
  estimator.setKSearch(kNeighbours);
  estimator.compute(*normals);
  for (std::size_t i = 0; i < result->size(); ++i) {
    (*result)[i].normal_x = (*normals)[i].normal_x;
    (*result)[i].normal_y = (*normals)[i].normal_y;
    (*result)[i].normal_z = (*normals)[i].normal_z;
  }
  return result;
}

Eigen::Matrix4d Solve(const CloudNormal::Ptr& source,
                      const CloudNormal::Ptr& target, int iterations) {
  pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal> icp;
  icp.setInputSource(source);
  icp.setInputTarget(target);
  icp.setMaxCorrespondenceDistance(kMaxDistance);
  icp.setMaximumIterations(iterations);
  icp.setTransformationEpsilon(0.0);
  icp.setEuclideanFitnessEpsilon(0.0);
  CloudNormal aligned;
  icp.align(aligned);
  return icp.getFinalTransformation().cast<double>();
}

}  // namespace

int main() {
  const Eigen::Matrix4d truth = LoadTruth();

  // Калибровка: препроцессинг один раз, он не измеряется.
  CloudNormal::Ptr source = WithNormals(Downsample(Load("source.ply")), false);
  CloudNormal::Ptr target = WithNormals(Downsample(Load("target.ply")), true);
  std::printf("# точек после прореживания: %zu\n", source->size());
  std::fflush(stdout);

  const double targets[] = {1e-3, 1e-4};
  for (double goal : targets) {
    int found_at = -1;
    for (int iterations = 1; iterations <= 40; ++iterations) {
      if (TranslationError(Solve(source, target, iterations), truth) < goal) {
        found_at = iterations;
        break;
      }
    }
    if (found_at < 0) {
      std::printf("impl=pcl target=%e k=- seconds=- error=-\n", goal);
      continue;
    }

    // Замер: полный конвейер ровно с найденным числом итераций.
    const auto start = std::chrono::steady_clock::now();
    CloudNormal::Ptr measured_source =
        WithNormals(Downsample(Load("source.ply")), false);
    CloudNormal::Ptr measured_target =
        WithNormals(Downsample(Load("target.ply")), true);
    const Eigen::Matrix4d pose = Solve(measured_source, measured_target, found_at);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    std::printf("impl=pcl target=%e k=%d seconds=%.4f error=%.3e\n", goal,
                found_at, seconds, TranslationError(pose, truth));
    std::fflush(stdout);
  }
  return 0;
}
