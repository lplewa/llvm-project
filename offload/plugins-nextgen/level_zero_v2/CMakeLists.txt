# Create the library and add the default arguments.
add_target_library(omptarget.rtl.level_zero_v2 LEVEL_ZERO_V2)

target_sources(omptarget.rtl.level_zero_v2 PRIVATE
  src/rtl.cpp
  src/L0V2DynWrapper.cpp
)

target_include_directories(omptarget.rtl.level_zero_v2 PRIVATE
  ${LIBOMPTARGET_INCLUDE_DIR}
  ${LIBOMPTARGET_DEP_LEVEL_ZERO_INCLUDE_DIR}
  ${LIBOMPTARGET_LLVM_INCLUDE_DIRS}
  ${LIBOMPTARGET_OMP_HEADER_DIR}
)

option(LIBOMPTARGET_FORCE_LEVELZERO_V2_TESTS "Build Level Zero V2 libomptarget tests" OFF)
if (LIBOMPTARGET_FOUND_INTELGPU_GPU OR LIBOMPTARGET_FORCE_LEVELZERO_V2_TESTS)
  set(LIBOMPTARGET_SYSTEM_TARGETS
      "${LIBOMPTARGET_SYSTEM_TARGETS} spirv64-intel" PARENT_SCOPE)
  list(APPEND LIBOMPTARGET_TESTED_PLUGINS "omptarget.rtl.level_zero_v2")
  set(LIBOMPTARGET_TESTED_PLUGINS "${LIBOMPTARGET_TESTED_PLUGINS}" PARENT_SCOPE)
else()
  message(STATUS "Not generating Level Zero V2 tests, no supported devices detected."
                   " Use 'LIBOMPTARGET_FORCE_LEVELZERO_V2_TESTS' to override.")
endif()
