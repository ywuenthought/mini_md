cmake_minimum_required(VERSION 3.18)
project(mini_md LANGUAGES CXX)

# Use OpenMP by default.
option(USE_OPENMP "Enable OpenMP parallelization" ON)

# Find dependencies.
find_package(pybind11 CONFIG REQUIRED)

# Add target and set its properties.
add_library(mini_md_core MODULE src/mini_md/_core.cpp)
set_target_properties(
    mini_md_core PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

# Run pybind11 macros.
pybind11_extension(mini_md_core)
pybind11_strip(mini_md_core)

# Link used libraries.
target_link_libraries(mini_md_core PRIVATE pybind11::module)

if(USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(mini_md_core PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(mini_md_core PRIVATE MINI_MD_USE_OPENMP=1)
    endif()
endif()

# Set installation instructions.
install(TARGETS mini_md_core LIBRARY DESTINATION mini_md)
