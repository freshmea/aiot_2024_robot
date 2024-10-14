cmake_minimum_required(VERSION 3.0)
project(helloOpenCV)

find_package(OpenCV REQUIRED)

add_executable(main helloOpenCV.cpp)
target_link_libraries(main ${OpenCV_LIBS})
target_link_directories(main PUBLIC ${OpenCV_LIB_DIR})

add_executable(mainclass mainclass.cpp)
target_link_libraries(mainclass ${OpenCV_LIBS})
target_link_directories(mainclass PUBLIC ${OpenCV_LIB_DIR})