cmake_minimum_required(VERSION 3.10)
project(CPPgrad)

set(CMAKE_CXX_STANDARD 17)

# Include directories
include_directories(include)

# Find all .cpp files in src directory
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS src/*.cpp)

# Add executable target
add_executable(CPPgrad ${SRC_FILES})
