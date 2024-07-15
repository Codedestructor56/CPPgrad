cmake_minimum_required(VERSION 3.10)
project(CPPgrad)

# Find Python and pybind11 with required components
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(${Python3_INCLUDE_DIRS})
include_directories(include)
# Create the shared library
add_library(CPPgrad SHARED
    src/autograd.cpp
    src/mlp.cpp
    src/bindings.cpp
)

# Link against Python and pybind11
target_link_libraries(CPPgrad PRIVATE ${Python3_LIBRARIES} pybind11::module)

# Specify the module name
set_target_properties(CPPgrad PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)
