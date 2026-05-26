get_filename_component(PIXI_PYTHON "${CMAKE_CURRENT_LIST_DIR}/../.pixi/envs/default/bin/python" ABSOLUTE)
set(Python3_EXECUTABLE "${PIXI_PYTHON}" CACHE FILEPATH "Python interpreter from Pixi environment")
set(Kokkos_ARCH_NATIVE ON CACHE BOOL "Use native CPU architecture")
