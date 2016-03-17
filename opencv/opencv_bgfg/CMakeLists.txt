cmake_minimum_required(VERSION 2.8)
project(gpu-test)
find_package( OpenCV REQUIRED )

find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")


message(STATUS ${OpenCV_INCLUDE_DIRS})


add_executable( main main.cpp )
target_link_libraries( main ${OpenCV_LIBS} )