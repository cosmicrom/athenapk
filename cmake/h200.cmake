get_filename_component(PIXI_PYTHON "${CMAKE_CURRENT_LIST_DIR}/../.pixi/envs/default/bin/python" ABSOLUTE)
get_filename_component(KOKKOS_NVCC_WRAPPER "${CMAKE_CURRENT_LIST_DIR}/../external/Kokkos/bin/nvcc_wrapper" ABSOLUTE)

set(Python3_EXECUTABLE "${PIXI_PYTHON}" CACHE FILEPATH "Python interpreter from Pixi environment")
set(CMAKE_CXX_COMPILER "${KOKKOS_NVCC_WRAPPER}" CACHE FILEPATH "Use Kokkos nvcc_wrapper for CUDA builds")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Default release build")
set(Kokkos_ARCH_ZEN4 ON CACHE BOOL "Target AMD Zen4 CPUs")
set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Enable CUDA")
set(Kokkos_ARCH_HOPPER90 ON CACHE BOOL "Target NVIDIA H200 GPUs on amd24 nodes")
