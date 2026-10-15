cmake_minimum_required(VERSION 3.18)
project(ehr_message)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
    "msg/LocoVelocity.msg"
    "service/LocomotionModeSwitch.srv"
    "service/MotionCmd.srv"
    DEPENDENCIES std_msgs)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
