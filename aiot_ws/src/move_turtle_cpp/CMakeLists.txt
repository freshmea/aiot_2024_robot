cmake_minimum_required(VERSION 3.8)
project(move_turtle_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(lifecycle_msgs REQUIRED)

# add executable
add_executable(canny src/canny.cpp)
ament_target_dependencies(canny rclcpp sensor_msgs OpenCV cv_bridge)

add_executable(servo_sub src/servo_sub.cpp)
ament_target_dependencies(servo_sub rclcpp std_msgs lifecycle_msgs)
target_link_libraries(servo_sub wiringPi)
# target_link_directories(servo_sub PUBLIC /usr/include)

# install
install(TARGETS
  canny
  servo_sub
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
