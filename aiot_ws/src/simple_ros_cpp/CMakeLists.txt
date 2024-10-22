cmake_minimum_required(VERSION 3.8)
project(simple_ros_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(turtlesim REQUIRED)
find_package(std_srvs REQUIRED)
find_package(user_interface REQUIRED)

# include directories
include_directories(include)

add_executable(hello_pub src/hello_pub.cpp)
ament_target_dependencies(hello_pub rclcpp)

add_executable(hello_pub_class src/hello_pub_class.cpp src/hello_pub_class_method.cpp)
ament_target_dependencies(hello_pub_class rclcpp std_msgs)

add_executable(hello_pub_lambda src/hello_pub_lambda.cpp)
ament_target_dependencies(hello_pub_lambda rclcpp std_msgs)

add_executable(hello_sub src/hello_sub.cpp)
ament_target_dependencies(hello_sub rclcpp std_msgs)

add_executable(hello_sub_opencv src/hello_sub_opencv.cpp)
target_link_libraries(hello_sub_opencv ${OpenCV_LIBS})
target_link_directories(hello_sub_opencv PUBLIC ${OpenCV_LIB_DIR})
ament_target_dependencies(hello_sub_opencv rclcpp std_msgs)

add_executable(move_turtle src/move_turtle.cpp)
ament_target_dependencies(move_turtle rclcpp geometry_msgs turtlesim)

add_executable(service_server src/service_server.cpp)
ament_target_dependencies(service_server rclcpp std_srvs)

add_executable(service_client src/service_client.cpp)
ament_target_dependencies(service_client rclcpp std_srvs)

add_executable(service_client_int src/service_client_int.cpp)
ament_target_dependencies(service_client_int rclcpp user_interface)

install(TARGETS
  hello_pub
  hello_pub_lambda
  hello_pub_class
  hello_sub
  hello_sub_opencv
  move_turtle
  service_server
  service_client
  service_client_int
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
