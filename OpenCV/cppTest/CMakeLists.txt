cmake_minimum_required(VERSION 3.0)
project(helloOpenCV)

find_package(OpenCV REQUIRED)
file(GLOB SOURCES "*.cpp")

foreach(SOURCE ${SOURCES})
    get_filename_component(EXE_NAME ${SOURCE} NAME_WE)
    add_executable(${EXE_NAME} ${SOURCE})
    target_link_libraries(${EXE_NAME} ${OpenCV_LIBS})
    target_link_directories(${EXE_NAME} PUBLIC ${OpenCV_LIB_DIR})
endforeach()
