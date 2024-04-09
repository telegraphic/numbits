cmake_minimum_required(VERSION 3.15...3.27)

# Scikit-build-core sets these values for you, or you can just hard-code the
# name and version.
project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG -march=native -ffast-math")

# Find the module development requirements (requires FindPython from 3.17 or
# scikit-build-core's built-in backport)
set(PYBIND11_NEWPYTHON ON)
find_package(OpenMP)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(numbits src/numbits.cpp)

target_link_libraries(numbits PRIVATE pybind11::pybind11)

# Check if OpenMP was found
if(OpenMP_CXX_FOUND)
    target_link_libraries(numbits PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(numbits PRIVATE USE_OPENMP)
endif()

# This is passing in the version as a define just as an example
target_compile_definitions(numbits PRIVATE VERSION_INFO=${PROJECT_VERSION})

# The install directory is the output (wheel) directory
install(TARGETS numbits DESTINATION .)