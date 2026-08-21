#!/bin/sh
# Сборка PCL-харнесса. Eigen 5 из Homebrew не находится сам:
# PCLConfig ищет Eigen3Config.cmake, а формула кладёт его в свой префикс.
set -e
cd "$(dirname "$0")/.."
cmake -S bench-external -B bench-external/build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="/opt/homebrew;/opt/homebrew/opt/eigen/share/eigen3/cmake"
cmake --build bench-external/build --parallel
