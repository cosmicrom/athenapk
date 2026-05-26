get_filename_component(PIXI_PYTHON "${CMAKE_CURRENT_LIST_DIR}/../.pixi/envs/default/bin/python" ABSOLUTE)
set(Python3_EXECUTABLE "${PIXI_PYTHON}" CACHE FILEPATH "Python interpreter from Pixi environment")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Default release build")
set(Kokkos_ARCH_ZEN4 ON CACHE BOOL "Target AMD Zen4 CPUs")
