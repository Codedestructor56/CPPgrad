# CMAKE generated file: DO NOT EDIT!
# Generated by CMake Version 3.29
cmake_policy(SET CMP0009 NEW)

# SRC_FILES at CMakeLists.txt:15 (file)
file(GLOB_RECURSE NEW_GLOB LIST_DIRECTORIES false "/home/richardM/CPPgrad/src/*.cpp")
set(OLD_GLOB
  "/home/richardM/CPPgrad/src/autograd.cpp"
  "/home/richardM/CPPgrad/src/bindings.cpp"
  "/home/richardM/CPPgrad/src/main.cpp"
  "/home/richardM/CPPgrad/src/mlp.cpp"
  )
if(NOT "${NEW_GLOB}" STREQUAL "${OLD_GLOB}")
  message("-- GLOB mismatch!")
  file(TOUCH_NOCREATE "/home/richardM/CPPgrad/CMakeFiles/cmake.verify_globs")
endif()
