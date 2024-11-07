cmake_minimum_required(VERSION 3.8)
project(simple_ros_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc videoio)
find_package(geometry_msgs REQUIRED)
find_package(turtlesim REQUIRED)
find_package(std_srvs REQUIRED)
find_package(user_interface REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rcutils REQUIRED)
find_package(rclcpp_lifecycle  REQUIRED)
find_package(lifecycle_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)


# library
add_library(talker_component SHARED src/talker_component.cpp)
target_compile_definitions(talker_component PRIVATE "COMPOSITION_BUILDING_DLL")
ament_target_dependencies(talker_component rclcpp std_msgs rclcpp_components)
rclcpp_components_register_nodes(talker_component "composition::Talker")

add_library(listener_component SHARED src/listener_component.cpp)
target_compile_definitions(listener_component PRIVATE "COMPOSITION_BUILDING_DLL")
ament_target_dependencies(listener_component rclcpp std_msgs rclcpp_components)
rclcpp_components_register_nodes(listener_component "composition::Listener")

install(TARGETS
  talker_component
  listener_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

# include directories
include_directories(include)

# executable
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

add_executable(simple_parameter src/simple_parameter.cpp)
ament_target_dependencies(simple_parameter rclcpp rcl_interfaces)

add_executable(action_server src/action_server.cpp)
ament_target_dependencies(action_server rclcpp user_interface rclcpp_action)

add_executable(action_client src/action_client.cpp)
ament_target_dependencies(action_client rclcpp user_interface rclcpp_action)

add_executable(logging_example src/logging_example.cpp)
ament_target_dependencies(logging_example rclcpp std_msgs)

add_executable(two_node_pipline src/two_node_pipline.cpp)
ament_target_dependencies(two_node_pipline rclcpp std_msgs)

add_executable(cyclic_pipline src/cyclic_pipline.cpp)
ament_target_dependencies(cyclic_pipline rclcpp std_msgs)

add_executable(image_pipeline_all_in_one
  src/image_pipeline_all_in_one.cpp)
target_link_libraries(image_pipeline_all_in_one
  rclcpp::rclcpp
  ${builtin_interfaces_TARGETS}
  ${sensor_msgs_TARGETS}
  opencv_core
  opencv_highgui)

add_executable(manual_composition src/manual_composition.cpp)
target_link_libraries(manual_composition
  talker_component
  listener_component)
ament_target_dependencies(manual_composition rclcpp)

add_executable(lifecycle_talker src/lifecycle_talker.cpp)
ament_target_dependencies(lifecycle_talker
  lifecycle_msgs
  rclcpp_lifecycle
  std_msgs
)

add_executable(lifecycle_listener src/lifecycle_listener.cpp)
ament_target_dependencies(lifecycle_listener
  lifecycle_msgs
  rclcpp_lifecycle
  std_msgs
)

add_executable(lifecycle_service_client src/lifecycle_service_client.cpp)
ament_target_dependencies(lifecycle_service_client
  lifecycle_msgs
  rclcpp_lifecycle
  std_msgs
)

add_executable(lifecycle_service_client_thread src/lifecycle_service_client_thread.cpp)
ament_target_dependencies(lifecycle_service_client_thread
  lifecycle_msgs
  rclcpp_lifecycle
  std_msgs
)

add_executable(static_tf src/static_tf.cpp)
ament_target_dependencies(static_tf
  rclcpp
  tf2
  tf2_ros
  geometry_msgs
)

add_executable(dynamic_tf src/dynamic_tf.cpp)
ament_target_dependencies(dynamic_tf
  rclcpp
  tf2
  tf2_ros
  geometry_msgs
  turtlesim
)

add_executable(dynamic_tf2 src/dynamic_tf2.cpp)
ament_target_dependencies(dynamic_tf2
  rclcpp
  tf2
  tf2_ros
  geometry_msgs
  turtlesim
)

add_executable(follow_turtle_tf src/follow_turtle_tf.cpp)
ament_target_dependencies(follow_turtle_tf
  rclcpp
  tf2
  tf2_ros
  geometry_msgs
  turtlesim
)

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
  simple_parameter
  action_server
  action_client
  logging_example
  two_node_pipline
  cyclic_pipline
  image_pipeline_all_in_one
  manual_composition
  lifecycle_talker
  lifecycle_listener
  lifecycle_service_client
  lifecycle_service_client_thread
  static_tf
  dynamic_tf
  dynamic_tf2
  follow_turtle_tf
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
