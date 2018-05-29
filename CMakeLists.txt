CMAKE_MINIMUM_REQUIRED(VERSION 3.0 FATAL_ERROR)

PROJECT(optical_calibration_tool)

FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

ADD_EXECUTABLE(optical_calibration
	optical_calibration.cpp)
TARGET_LINK_LIBRARIES(optical_calibration ${OpenCV_LIBS})
