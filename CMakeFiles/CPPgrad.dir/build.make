# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.29

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/richardM/CPPgrad

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/richardM/CPPgrad

# Include any dependencies generated for this target.
include CMakeFiles/CPPgrad.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/CPPgrad.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/CPPgrad.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/CPPgrad.dir/flags.make

CMakeFiles/CPPgrad.dir/src/autograd.cpp.o: CMakeFiles/CPPgrad.dir/flags.make
CMakeFiles/CPPgrad.dir/src/autograd.cpp.o: src/autograd.cpp
CMakeFiles/CPPgrad.dir/src/autograd.cpp.o: CMakeFiles/CPPgrad.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/richardM/CPPgrad/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/CPPgrad.dir/src/autograd.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/CPPgrad.dir/src/autograd.cpp.o -MF CMakeFiles/CPPgrad.dir/src/autograd.cpp.o.d -o CMakeFiles/CPPgrad.dir/src/autograd.cpp.o -c /home/richardM/CPPgrad/src/autograd.cpp

CMakeFiles/CPPgrad.dir/src/autograd.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/CPPgrad.dir/src/autograd.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/richardM/CPPgrad/src/autograd.cpp > CMakeFiles/CPPgrad.dir/src/autograd.cpp.i

CMakeFiles/CPPgrad.dir/src/autograd.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/CPPgrad.dir/src/autograd.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/richardM/CPPgrad/src/autograd.cpp -o CMakeFiles/CPPgrad.dir/src/autograd.cpp.s

CMakeFiles/CPPgrad.dir/src/main.cpp.o: CMakeFiles/CPPgrad.dir/flags.make
CMakeFiles/CPPgrad.dir/src/main.cpp.o: src/main.cpp
CMakeFiles/CPPgrad.dir/src/main.cpp.o: CMakeFiles/CPPgrad.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/richardM/CPPgrad/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/CPPgrad.dir/src/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/CPPgrad.dir/src/main.cpp.o -MF CMakeFiles/CPPgrad.dir/src/main.cpp.o.d -o CMakeFiles/CPPgrad.dir/src/main.cpp.o -c /home/richardM/CPPgrad/src/main.cpp

CMakeFiles/CPPgrad.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/CPPgrad.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/richardM/CPPgrad/src/main.cpp > CMakeFiles/CPPgrad.dir/src/main.cpp.i

CMakeFiles/CPPgrad.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/CPPgrad.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/richardM/CPPgrad/src/main.cpp -o CMakeFiles/CPPgrad.dir/src/main.cpp.s

# Object files for target CPPgrad
CPPgrad_OBJECTS = \
"CMakeFiles/CPPgrad.dir/src/autograd.cpp.o" \
"CMakeFiles/CPPgrad.dir/src/main.cpp.o"

# External object files for target CPPgrad
CPPgrad_EXTERNAL_OBJECTS =

CPPgrad: CMakeFiles/CPPgrad.dir/src/autograd.cpp.o
CPPgrad: CMakeFiles/CPPgrad.dir/src/main.cpp.o
CPPgrad: CMakeFiles/CPPgrad.dir/build.make
CPPgrad: CMakeFiles/CPPgrad.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/richardM/CPPgrad/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable CPPgrad"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/CPPgrad.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/CPPgrad.dir/build: CPPgrad
.PHONY : CMakeFiles/CPPgrad.dir/build

CMakeFiles/CPPgrad.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/CPPgrad.dir/cmake_clean.cmake
.PHONY : CMakeFiles/CPPgrad.dir/clean

CMakeFiles/CPPgrad.dir/depend:
	cd /home/richardM/CPPgrad && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/richardM/CPPgrad /home/richardM/CPPgrad /home/richardM/CPPgrad /home/richardM/CPPgrad /home/richardM/CPPgrad/CMakeFiles/CPPgrad.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/CPPgrad.dir/depend

