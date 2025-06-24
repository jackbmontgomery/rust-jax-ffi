cmake_minimum_required(VERSION 3.16)
project(rust_jax_ffi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# ---------- XLA ----------

include(FetchContent)

FetchContent_Declare(
  xla
  GIT_REPOSITORY https://github.com/openxla/xla.git
  GIT_TAG main            # Consider pinning to a commit hash for stability
  GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(xla)

message(STATUS "XLA headers path: ${xla_SOURCE_DIR}")

# ---------- Setup nanobind ----------

set(Python_EXECUTABLE "${CMAKE_SOURCE_DIR}/.venv/bin/python")
set(ENV{VIRTUAL_ENV} "${CMAKE_SOURCE_DIR}/.venv")
set(ENV{PATH} "${CMAKE_SOURCE_DIR}/.venv/bin:$ENV{PATH}")

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

# Find Python first, using our specified executable
find_package(Python 3.12 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

# Verify we're using the right Python
message(STATUS "Using Python: ${Python_EXECUTABLE}")
message(STATUS "Python version: ${Python_VERSION}")

# Get nanobind cmake directory from the venv Python
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
  RESULT_VARIABLE nanobind_RESULT
)

if(NOT nanobind_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to get nanobind cmake directory. Make sure nanobind is installed in your venv.")
endif()

message(STATUS "nanobind root: ${nanobind_ROOT}")

# Find nanobind
find_package(nanobind CONFIG REQUIRED PATHS ${nanobind_ROOT})

# Set build type
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# ---------- Rust/Cargo ----------

set(CARGO_MANIFEST ${CMAKE_SOURCE_DIR}/Cargo.toml)
set(CARGO_TARGET_DIR ${CMAKE_SOURCE_DIR}/target)

set(RUST_SOURCE_FILE ${CMAKE_SOURCE_DIR}/src/lib.rs)
set(RUST_BRIDGE_CPP ${CARGO_TARGET_DIR}/cxxbridge/rust_jax_ffi/src/lib.rs.cc)
set(RUST_LIB ${CARGO_TARGET_DIR}/debug/${CMAKE_STATIC_LIBRARY_PREFIX}rust_jax_ffi${CMAKE_STATIC_LIBRARY_SUFFIX})

# Add a custom command that builds the rust crate and generates C++ bridge code
add_custom_command(
        OUTPUT ${RUST_BRIDGE_CPP} ${RUST_LIB}
        COMMAND cargo build --manifest-path ${CARGO_MANIFEST}
        DEPENDS ${RUST_SOURCE_FILE}
        USES_TERMINAL
        COMMENT "Running cargo..."
)

# ---------- Create python package and stub ----------

# Create the nanobind Python module
nanobind_add_module(rust_jax_ffi src/ffi.cc ${RUST_BRIDGE_CPP})

# Create the .pyi stub file for the module
nanobind_add_stub(
  rust_jax_ffi_stub
  MODULE rust_jax_ffi
  OUTPUT rust_jax_ffi.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:rust_jax_ffi>
  DEPENDS rust_jax_ffi
)

target_include_directories(
        rust_jax_ffi
        PRIVATE
        include/
        ffi
        ${CARGO_TARGET_DIR}/cxxbridge/rust_jax_ffi/src/
        ${xla_SOURCE_DIR}
)

# Link the Rust library
target_link_libraries(rust_jax_ffi PRIVATE ${RUST_LIB})

# Set additional clean files
set_target_properties(
        rust_jax_ffi
        PROPERTIES ADDITIONAL_CLEAN_FILES ${CARGO_TARGET_DIR}
)
