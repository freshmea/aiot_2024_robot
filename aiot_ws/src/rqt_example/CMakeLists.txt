cmake_minimum_required(VERSION 3.8)
project(rqt_example)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# install
ament_python_install_package(${PROJECT_NAME} PACKAGE_DIR src/${PROJECT_NAME})

install(FILES
  plugin.xml
  DESTINATION share/${PROJECT_NAME})

install(DIRECTORY
  resource
  # launch
  DESTINATION share/${PROJECT_NAME})

install(PROGRAMS
  scripts/rqt_example
  DESTINATION lib/${PROJECT_NAME})


if(BUILD_TESTING)

  find_package(ament_lint_auto REQUIRED)
   set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
