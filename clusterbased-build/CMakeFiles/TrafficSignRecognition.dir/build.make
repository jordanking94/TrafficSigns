# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CMake.app/Contents/bin/cmake

# The command to remove a file.
RM = /Applications/CMake.app/Contents/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/Jordan/Desktop/Thesis/October/clusterbased

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/Jordan/Desktop/Thesis/October/clusterbased-build

# Include any dependencies generated for this target.
include CMakeFiles/TrafficSignRecognition.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/TrafficSignRecognition.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/TrafficSignRecognition.dir/flags.make

CMakeFiles/TrafficSignRecognition.dir/tsr.cpp.o: CMakeFiles/TrafficSignRecognition.dir/flags.make
CMakeFiles/TrafficSignRecognition.dir/tsr.cpp.o: /Users/Jordan/Desktop/Thesis/October/clusterbased/tsr.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/Jordan/Desktop/Thesis/October/clusterbased-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/TrafficSignRecognition.dir/tsr.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/TrafficSignRecognition.dir/tsr.cpp.o -c /Users/Jordan/Desktop/Thesis/October/clusterbased/tsr.cpp

CMakeFiles/TrafficSignRecognition.dir/tsr.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TrafficSignRecognition.dir/tsr.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/Jordan/Desktop/Thesis/October/clusterbased/tsr.cpp > CMakeFiles/TrafficSignRecognition.dir/tsr.cpp.i

CMakeFiles/TrafficSignRecognition.dir/tsr.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TrafficSignRecognition.dir/tsr.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/Jordan/Desktop/Thesis/October/clusterbased/tsr.cpp -o CMakeFiles/TrafficSignRecognition.dir/tsr.cpp.s

# Object files for target TrafficSignRecognition
TrafficSignRecognition_OBJECTS = \
"CMakeFiles/TrafficSignRecognition.dir/tsr.cpp.o"

# External object files for target TrafficSignRecognition
TrafficSignRecognition_EXTERNAL_OBJECTS =

TrafficSignRecognition: CMakeFiles/TrafficSignRecognition.dir/tsr.cpp.o
TrafficSignRecognition: CMakeFiles/TrafficSignRecognition.dir/build.make
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_gapi.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_stitching.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_aruco.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_bgsegm.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_bioinspired.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_ccalib.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_dnn_objdetect.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_dpm.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_face.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_freetype.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_fuzzy.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_hfs.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_img_hash.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_line_descriptor.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_quality.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_reg.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_rgbd.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_saliency.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_sfm.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_stereo.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_structured_light.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_superres.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_surface_matching.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_tracking.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_videostab.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_xfeatures2d.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_xobjdetect.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_xphoto.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_highgui.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_shape.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_datasets.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_plot.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_text.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_dnn.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_ml.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_phase_unwrapping.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_optflow.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_ximgproc.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_video.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_videoio.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_imgcodecs.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_objdetect.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_calib3d.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_features2d.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_flann.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_photo.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_imgproc.4.1.2.dylib
TrafficSignRecognition: /Users/Jordan/opencv/build/lib/libopencv_core.4.1.2.dylib
TrafficSignRecognition: CMakeFiles/TrafficSignRecognition.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/Jordan/Desktop/Thesis/October/clusterbased-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable TrafficSignRecognition"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/TrafficSignRecognition.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/TrafficSignRecognition.dir/build: TrafficSignRecognition

.PHONY : CMakeFiles/TrafficSignRecognition.dir/build

CMakeFiles/TrafficSignRecognition.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/TrafficSignRecognition.dir/cmake_clean.cmake
.PHONY : CMakeFiles/TrafficSignRecognition.dir/clean

CMakeFiles/TrafficSignRecognition.dir/depend:
	cd /Users/Jordan/Desktop/Thesis/October/clusterbased-build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/Jordan/Desktop/Thesis/October/clusterbased /Users/Jordan/Desktop/Thesis/October/clusterbased /Users/Jordan/Desktop/Thesis/October/clusterbased-build /Users/Jordan/Desktop/Thesis/October/clusterbased-build /Users/Jordan/Desktop/Thesis/October/clusterbased-build/CMakeFiles/TrafficSignRecognition.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/TrafficSignRecognition.dir/depend

