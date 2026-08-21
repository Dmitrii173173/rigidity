cmake_minimum_required(VERSION 3.20)
project(pcl_harness CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(PCL 1.15 REQUIRED COMPONENTS common io filters features registration)

add_executable(pcl_harness pcl_harness.cpp)
target_include_directories(pcl_harness PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(pcl_harness PRIVATE ${PCL_LIBRARIES})
target_compile_definitions(pcl_harness PRIVATE ${PCL_DEFINITIONS})
