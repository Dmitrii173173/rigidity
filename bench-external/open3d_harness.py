"""Участник сравнения: Open3D. Протокол — PROTOCOL.md."""

import sys
import time

import numpy as np
import open3d as o3d

VOXEL = 0.02
NEIGHBOURS = 16
MAX_DISTANCE = 0.5
DIRECTORY = "bench-external/data"


def load_truth():
    values = np.loadtxt(f"{DIRECTORY}/truth.txt")
    return values.reshape(4, 4)


def translation_error(found, truth):
    error = truth @ np.linalg.inv(found)
    return float(np.linalg.norm(error[:3, 3]))


def prepare():
    source = o3d.io.read_point_cloud(f"{DIRECTORY}/source.ply")
    target = o3d.io.read_point_cloud(f"{DIRECTORY}/target.ply")
    source = source.voxel_down_sample(VOXEL)
    target = target.voxel_down_sample(VOXEL)
    target.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=NEIGHBOURS)
    )
    return source, target


def solve(source, target, iterations):
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        MAX_DISTANCE,
        np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=0.0,
            relative_rmse=0.0,
            max_iteration=iterations,
        ),
    )
    return np.asarray(result.transformation)


def pipeline(iterations):
    """Полный конвейер: именно он и замеряется."""
    start = time.perf_counter()
    source = o3d.io.read_point_cloud(f"{DIRECTORY}/source.ply")
    target = o3d.io.read_point_cloud(f"{DIRECTORY}/target.ply")

    source = source.voxel_down_sample(VOXEL)
    target = target.voxel_down_sample(VOXEL)
    target.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=NEIGHBOURS)
    )

    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        MAX_DISTANCE,
        np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=0.0,
            relative_rmse=0.0,
            max_iteration=iterations,
        ),
    )
    return np.asarray(result.transformation), time.perf_counter() - start


def main():
    truth = load_truth()
    # Калибровка не измеряется, поэтому препроцессинг делается один раз.
    source, target_cloud = prepare()
    print(f"# точек после прореживания: {len(source.points)}", flush=True)
    for target in (1e-3, 1e-4):
        found_at = None
        for iterations in range(1, 41):
            pose = solve(source, target_cloud, iterations)
            if translation_error(pose, truth) < target:
                found_at = iterations
                break
        if found_at is None:
            print(f"impl=open3d target={target:e} k=- seconds=- error=-")
            continue
        pose, seconds = pipeline(found_at)
        print(
            f"impl=open3d target={target:e} k={found_at} "
            f"seconds={seconds:.4f} error={translation_error(pose, truth):.3e}"
        )


if __name__ == "__main__":
    sys.exit(main())
