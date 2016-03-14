cmake_minimum_required(VERSION 2.8)
project(gpu-test)
find_package( OpenCV REQUIRED )

SET(OpenCV_LIB_DIR "${OpenCV_INSTALL_PATH}/libs/${ARMEABI_NDK_NAME}")
message("HELLO")
message(STATUS ${OpenCV_INCLUDE_DIRS})
message(STATUS ${OpenCV_LIBRARIES})

message("HELLO")
message(STATUS ${OpenCV_LIBS})

add_executable( main main.cpp )
target_link_libraries( main ${OpenCV_LIBS} )