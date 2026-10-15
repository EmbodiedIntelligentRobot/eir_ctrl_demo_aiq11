cmake_minimum_required(VERSION 3.18)
project(eir_ctrl_demo_aiq11 LANGUAGES CXX)
option(EIR_BUILD_CTRL "Build the AIQ11 RL application" OFF)
if(NOT EIR_BUILD_CTRL)
    return()
endif()
if(NOT TARGET eir::loco_runtime)
    find_package(eir_loco_runtime CONFIG REQUIRED)
endif()
set(EIR_SDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../sdk" CACHE PATH "EIR SDK")
option(EIR_CTRL_WITH_TORCH "Compile CPU LibTorch backend" ON)
option(EIR_CTRL_WITH_ONNX "Compile CPU ONNX Runtime backend" ON)
if(NOT EIR_CTRL_WITH_TORCH AND NOT EIR_CTRL_WITH_ONNX)
    message(FATAL_ERROR "Enable at least one inference backend")
endif()
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)
add_library(eir_ctrl_core src/inference.cpp src/config.cpp src/hardware.cpp src/control.cpp src/policy.cpp src/policies/amp.cpp src/policies/mimic.cpp src/policies/perception.cpp src/mapping.cpp src/remote.cpp src/runner.cpp src/recording.cpp src/external_depth.cpp src/inference_split.cpp src/motion.cpp src/hand.cpp)
find_library(EIR_KINEMATICS_LIBRARY eir_closed_kinematics PATHS "${EIR_SDK_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
target_link_libraries(eir_ctrl_core PUBLIC eir::loco_runtime yaml-cpp PRIVATE Threads::Threads Eigen3::Eigen OpenSSL::Crypto "${EIR_KINEMATICS_LIBRARY}")
target_compile_features(eir_ctrl_core PUBLIC cxx_std_17)
target_include_directories(eir_ctrl_core PUBLIC include)
target_compile_options(eir_ctrl_core PRIVATE -Wall -Wextra -Wpedantic)
execute_process(COMMAND git rev-parse --short HEAD WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE EIR_CTRL_REVISION OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
execute_process(COMMAND git diff --quiet WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" RESULT_VARIABLE EIR_CTRL_DIRTY)
if(EIR_CTRL_DIRTY)
    string(APPEND EIR_CTRL_REVISION "-worktree")
endif()
if(NOT EIR_CTRL_REVISION)
    set(EIR_CTRL_REVISION unknown)
endif()
target_compile_definitions(eir_ctrl_core PRIVATE
    EIR_CTRL_REVISION="${EIR_CTRL_REVISION}"
    EIR_CTRL_WITH_TORCH=$<BOOL:${EIR_CTRL_WITH_TORCH}>
    EIR_CTRL_WITH_ONNX=$<BOOL:${EIR_CTRL_WITH_ONNX}>)
if(EIR_CTRL_WITH_TORCH)
    find_package(Torch REQUIRED)
    target_sources(eir_ctrl_core PRIVATE src/inference_torch.cpp)
    target_link_libraries(eir_ctrl_core PRIVATE "${TORCH_LIBRARIES}")
endif()
if(EIR_CTRL_WITH_ONNX)
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h HINTS "${EIR_SDK_ROOT}/include/onnxruntime" NO_DEFAULT_PATH REQUIRED)
    find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS "${EIR_SDK_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
    target_sources(eir_ctrl_core PRIVATE src/inference_onnx.cpp)
    target_include_directories(eir_ctrl_core PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
    target_link_libraries(eir_ctrl_core PRIVATE "${ONNXRUNTIME_LIBRARY}")
endif()
include(CTest)
add_executable(eir_ctrl_demo_aiq11-resource-check src/resource_check.cpp)
target_link_libraries(eir_ctrl_demo_aiq11-resource-check PRIVATE eir_ctrl_core)
target_include_directories(eir_ctrl_demo_aiq11-resource-check PRIVATE "${EIR_SDK_ROOT}/include")
set_target_properties(eir_ctrl_demo_aiq11-resource-check PROPERTIES
    INSTALL_RPATH "$ORIGIN/../lib;$ORIGIN/../sdk/lib"
    INSTALL_RPATH_USE_LINK_PATH FALSE)
install(TARGETS eir_ctrl_demo_aiq11-resource-check RUNTIME DESTINATION bin COMPONENT resource-check)
if(BUILD_TESTING)
    find_package(GTest REQUIRED)
    add_executable(ctrl_test tests/ctrl_test.cpp tests/mimic_test.cpp tests/config_test.cpp tests/sport_test.cpp tests/special_update_test.cpp tests/perception_test.cpp tests/external_depth_test.cpp tests/split_inference_test.cpp tests/motion_test.cpp tests/hand_test.cpp tests/recording_test.cpp)
    target_link_libraries(ctrl_test PRIVATE eir_ctrl_core GTest::gtest_main)
    target_compile_definitions(ctrl_test PRIVATE MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/policies/amp/pro/walk"
        PRO_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/config/pro.yaml"
        SPORT_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/config/sport.yaml"
        SPORT_SPECIAL_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/config/sport_special_mimic.yaml"
        PRO_SPECIAL_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/config/pro_special_mimic.yaml"
        EIR_CTRL_WITH_TORCH=$<BOOL:${EIR_CTRL_WITH_TORCH}>
        EIR_CTRL_WITH_ONNX=$<BOOL:${EIR_CTRL_WITH_ONNX}>)
    add_test(NAME ctrl_test COMMAND ctrl_test)
    add_executable(ctrl_hardware_test tests/hardware_test.cpp)
    target_link_libraries(ctrl_hardware_test PRIVATE eir_ctrl_core GTest::gtest_main Threads::Threads)
    target_compile_definitions(ctrl_hardware_test PRIVATE
        EIR_CTRL_WITH_ONNX=$<BOOL:${EIR_CTRL_WITH_ONNX}>
        EIR_CTRL_WITH_TORCH=$<BOOL:${EIR_CTRL_WITH_TORCH}>
        PRO_HARDWARE_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/config/pro_hardware.yaml"
        SPORT_HARDWARE_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/config/sport_hardware.yaml"
        PRO_SDK_CONFIG="${EIR_SDK_ROOT}/share/eir_base_sdk/config/humanoid.yaml")
    add_test(NAME ctrl_hardware_test COMMAND ctrl_hardware_test)
    set_tests_properties(ctrl_hardware_test PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE)
    add_executable(ctrl_recording_simulation tests/recording_simulation.cpp)
    target_link_libraries(ctrl_recording_simulation PRIVATE eir_ctrl_core Threads::Threads)
    if(TARGET EIR::mujoco_sdk)
        # Explicit acceptance executable; long dynamics runs are opt-in.
        add_executable(ctrl_sport_simulation_test tests/sport_simulation.cpp)
        target_link_libraries(ctrl_sport_simulation_test PRIVATE eir_ctrl_core EIR::mujoco_sdk Threads::Threads)
    endif()
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ehr_message REQUIRED)
find_package(prime_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
option(EIR_CTRL_WITH_DEPTH_ROS "Subscribe to an external ROS DepthStack camera" OFF)
add_executable(eir_ctrl_demo_aiq11 src/main.cpp)
set_target_properties(eir_ctrl_demo_aiq11 PROPERTIES
    INSTALL_RPATH "$ORIGIN/../lib;$ORIGIN/../sdk/lib;/opt/ros/humble/lib"
    INSTALL_RPATH_USE_LINK_PATH FALSE)
install(TARGETS eir_ctrl_demo_aiq11 RUNTIME DESTINATION bin)
target_include_directories(eir_ctrl_demo_aiq11 PRIVATE "${EIR_SDK_ROOT}/include")
target_compile_definitions(eir_ctrl_demo_aiq11 PRIVATE
    EIR_SERVICE_HARDWARE=$<AND:$<BOOL:${EIR_WITH_HARDWARE}>,$<STREQUAL:${EIR_HARDWARE_FAMILY},aiq11>>)
add_library(eir_ctrl_ros src/remote_node.cpp)
target_link_libraries(eir_ctrl_ros PUBLIC eir_ctrl_core)
ament_target_dependencies(eir_ctrl_ros PUBLIC rclcpp ehr_message prime_msgs std_msgs std_srvs sensor_msgs geometry_msgs)
target_compile_definitions(eir_ctrl_demo_aiq11 PRIVATE
    EIR_CTRL_DEFAULT_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/config/pro_perception_9500.yaml")
set(EIR_REMOTE_GUI_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../eir_remote_gui" CACHE PATH
    "Optional standalone remote GUI source")
if(EXISTS "${EIR_REMOTE_GUI_ROOT}/remote_gui.py")
    configure_file("${EIR_REMOTE_GUI_ROOT}/remote_gui.py"
        "${CMAKE_CURRENT_BINARY_DIR}/remote_gui/remote_gui.py" COPYONLY)
    configure_file("${EIR_REMOTE_GUI_ROOT}/xbox.py"
        "${CMAKE_CURRENT_BINARY_DIR}/remote_gui/xbox.py" COPYONLY)
    configure_file("${EIR_REMOTE_GUI_ROOT}/depth_viewer.py"
        "${CMAKE_CURRENT_BINARY_DIR}/remote_gui/depth_viewer.py" COPYONLY)
    configure_file("${EIR_REMOTE_GUI_ROOT}/config/remote_gui.yaml"
        "${CMAKE_CURRENT_BINARY_DIR}/remote_gui/config/remote_gui.yaml" COPYONLY)
endif()
target_compile_definitions(eir_ctrl_ros PRIVATE EIR_CTRL_WITH_DEPTH_ROS=$<BOOL:${EIR_CTRL_WITH_DEPTH_ROS}>)
if(EIR_CTRL_WITH_DEPTH_ROS)
    find_package(depth_transport_msgs REQUIRED)
    ament_target_dependencies(eir_ctrl_ros PUBLIC depth_transport_msgs)
endif()
target_link_libraries(eir_ctrl_demo_aiq11 PRIVATE eir_ctrl_ros Threads::Threads)
ament_target_dependencies(eir_ctrl_demo_aiq11 PUBLIC rclcpp ehr_message prime_msgs std_msgs sensor_msgs)
if(BUILD_TESTING)
    add_executable(ctrl_diagnostics_test tests/diagnostics_test.cpp)
    target_link_libraries(ctrl_diagnostics_test PRIVATE eir_ctrl_ros GTest::gtest)
    target_compile_definitions(ctrl_diagnostics_test PRIVATE
        SPORT_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/config/sport.yaml"
        PRO_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/config/pro.yaml")
    add_test(NAME ctrl_diagnostics_test COMMAND ctrl_diagnostics_test)
    set_tests_properties(ctrl_diagnostics_test PROPERTIES TIMEOUT 100 RUN_SERIAL TRUE
        ENVIRONMENT "ROS_LOCALHOST_ONLY=1;ROS_DOMAIN_ID=185;AIQ11_PRO_MODEL=${CMAKE_CURRENT_SOURCE_DIR}/../../src/eir_loco_runtime/tests/data/one_joint.xml")
    # Opt-in complete-cycle capture using the same ROS node and Runner as the app.
    add_executable(ctrl_debug_timing tests/debug_timing.cpp)
    target_link_libraries(ctrl_debug_timing PRIVATE eir_ctrl_ros)
endif()
set(EIR_CTRL_SIM_TEST_MODEL "" CACHE FILEPATH "Optional full Pro MJCF for ROS integration tests")
option(EIR_CTRL_TEST_SPECIAL_MIMIC "Run experimental PRO special-motion acceptance (not yet passing)" OFF)
if(BUILD_TESTING)
    add_test(NAME ctrl_startup_failure_test COMMAND /usr/bin/python3
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_failure_test.py
        --executable $<TARGET_FILE:eir_ctrl_demo_aiq11>)
    set_tests_properties(ctrl_startup_failure_test PROPERTIES TIMEOUT 20)
    set_tests_properties(ctrl_test PROPERTIES ENVIRONMENT
        "AIQ11_PRO_MODEL=${CMAKE_CURRENT_SOURCE_DIR}/../../src/eir_loco_runtime/tests/data/one_joint.xml")
    if(EIR_CTRL_SIM_TEST_MODEL)
        if(EIR_GUI_TESTS)
            add_test(NAME ctrl_gui_startup COMMAND /usr/bin/python3
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/integration_gui_startup.py
                --executable $<TARGET_FILE:eir_ctrl_demo_aiq11>
                --config ${CMAKE_CURRENT_SOURCE_DIR}/config/pro.yaml
                --model ${EIR_CTRL_SIM_TEST_MODEL})
            set_tests_properties(ctrl_gui_startup PROPERTIES TIMEOUT 90 RUN_SERIAL TRUE
                SKIP_RETURN_CODE 77)
        endif()
        add_executable(ctrl_pro_simulation_test tests/pro_simulation_test.cpp)
        add_executable(ctrl_special_simulation_test tests/special_mimic_simulation.cpp)
        target_link_libraries(ctrl_special_simulation_test PRIVATE eir_ctrl_core Threads::Threads)
        target_compile_definitions(ctrl_special_simulation_test PRIVATE
            PRO_TEST_MODEL="${EIR_CTRL_SIM_TEST_MODEL}")
        target_link_libraries(ctrl_pro_simulation_test PRIVATE eir_ctrl_core EIR::mujoco_sdk GTest::gtest_main Threads::Threads)
        target_compile_definitions(ctrl_pro_simulation_test PRIVATE
            PRO_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/config/pro.yaml"
            PRO_PERCEPTION_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/config/pro_perception.yaml"
            PRO_TEST_MODEL="${EIR_CTRL_SIM_TEST_MODEL}")
        add_test(NAME ctrl_pro_simulation_test COMMAND ctrl_pro_simulation_test)
        set_tests_properties(ctrl_pro_simulation_test PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE
            ENVIRONMENT "AIQ11_PRO_MODEL=${EIR_CTRL_SIM_TEST_MODEL}")
        foreach(backend IN ITEMS onnx libtorch)
            if((backend STREQUAL "onnx" AND EIR_CTRL_WITH_ONNX) OR
               (backend STREQUAL "libtorch" AND EIR_CTRL_WITH_TORCH))
                if(EIR_CTRL_TEST_SPECIAL_MIMIC)
                  foreach(pair IN ITEMS "35;37" "36;38" "42;43")
                    list(GET pair 0 down)
                    add_test(NAME ctrl_special_${backend}_${down}
                        COMMAND ctrl_special_simulation_test ${CMAKE_CURRENT_SOURCE_DIR}/config/pro_special_mimic.yaml ${backend} ${pair})
                    set_tests_properties(ctrl_special_${backend}_${down} PROPERTIES TIMEOUT 100
                        RUN_SERIAL TRUE ENVIRONMENT "AIQ11_PRO_MODEL=${EIR_CTRL_SIM_TEST_MODEL}")
                  endforeach()
                endif()
                add_test(NAME ctrl_ros_split_${backend} COMMAND /usr/bin/python3
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/integration_ros.py
                    --executable $<TARGET_FILE:eir_ctrl_demo_aiq11>
                    --config ${CMAKE_CURRENT_SOURCE_DIR}/config/pro_perception.yaml
                    --model ${EIR_CTRL_SIM_TEST_MODEL} --backend ${backend}
                    --perception-execution split --release-after-ready --mode 41 --walk-seconds 10
                    --switch-modes 28 30 41 --switch-seconds 3)
                set_tests_properties(ctrl_ros_split_${backend} PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE
                    ENVIRONMENT "ROS_LOCALHOST_ONLY=1;ROS_DOMAIN_ID=184")
                if(EIR_CTRL_WITH_DEPTH_ROS)
                    foreach(fault IN ITEMS dropout invalid)
                        add_test(NAME ctrl_ros_camera_${backend}_${fault} COMMAND /usr/bin/python3
                            ${CMAKE_CURRENT_SOURCE_DIR}/tests/integration_ros.py
                            --executable $<TARGET_FILE:eir_ctrl_demo_aiq11>
                            --config ${CMAKE_CURRENT_SOURCE_DIR}/config/pro_perception_camera.yaml
                            --model ${EIR_CTRL_SIM_TEST_MODEL} --backend ${backend}
                            --mode 41 --walk-seconds 2 --external-depth-fault ${fault})
                        set_tests_properties(ctrl_ros_camera_${backend}_${fault} PROPERTIES
                            TIMEOUT 45 RUN_SERIAL TRUE ENVIRONMENT "ROS_LOCALHOST_ONLY=1;ROS_DOMAIN_ID=186")
                        add_test(NAME ctrl_ros_camera_split_${backend}_${fault} COMMAND /usr/bin/python3
                            ${CMAKE_CURRENT_SOURCE_DIR}/tests/integration_ros.py
                            --executable $<TARGET_FILE:eir_ctrl_demo_aiq11>
                            --config ${CMAKE_CURRENT_SOURCE_DIR}/config/pro_perception_camera.yaml
                            --model ${EIR_CTRL_SIM_TEST_MODEL} --backend ${backend}
                            --perception-execution split --mode 41 --walk-seconds 2 --external-depth-fault ${fault})
                        set_tests_properties(ctrl_ros_camera_split_${backend}_${fault} PROPERTIES
                            TIMEOUT 45 RUN_SERIAL TRUE ENVIRONMENT "ROS_LOCALHOST_ONLY=1;ROS_DOMAIN_ID=186")
                    endforeach()
                endif()
                add_test(NAME ctrl_ros_requests_${backend} COMMAND /usr/bin/python3
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/integration_ros.py
                    --executable $<TARGET_FILE:eir_ctrl_demo_aiq11>
                    --config ${CMAKE_CURRENT_SOURCE_DIR}/config/pro.yaml
                    --model ${EIR_CTRL_SIM_TEST_MODEL} --backend ${backend}
                    --burst-requests 64 --check-velocity-expiry --walk-seconds 2 --velocity-seconds 0)
                set_tests_properties(ctrl_ros_requests_${backend} PROPERTIES TIMEOUT 45 RUN_SERIAL TRUE
                    ENVIRONMENT "ROS_LOCALHOST_ONLY=1;ROS_DOMAIN_ID=190")
                add_test(NAME ctrl_ros_${backend} COMMAND /usr/bin/python3
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/integration_ros.py
                    --executable $<TARGET_FILE:eir_ctrl_demo_aiq11>
                    --config ${CMAKE_CURRENT_SOURCE_DIR}/config/pro.yaml
                    --model ${EIR_CTRL_SIM_TEST_MODEL} --backend ${backend}
                    --idle-seconds 15 --expect-limit-warning)
                set_tests_properties(ctrl_ros_${backend} PROPERTIES TIMEOUT 45 RUN_SERIAL TRUE
                    ENVIRONMENT "ROS_LOCALHOST_ONLY=1;ROS_DOMAIN_ID=184")
                add_test(NAME ctrl_ros_motion_${backend} COMMAND /usr/bin/python3
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/integration_ros.py
                    --executable $<TARGET_FILE:eir_ctrl_demo_aiq11>
                    --config ${CMAKE_CURRENT_SOURCE_DIR}/config/pro_motion.yaml
                    --model ${EIR_CTRL_SIM_TEST_MODEL} --backend ${backend}
                    --release-after-ready --walk-seconds 1 --velocity-seconds 0 --upper-body-actions)
                set_tests_properties(ctrl_ros_motion_${backend} PROPERTIES TIMEOUT 100 RUN_SERIAL TRUE
                    ENVIRONMENT "ROS_LOCALHOST_ONLY=1;ROS_DOMAIN_ID=188")
                add_test(NAME ctrl_ros_release_${backend} COMMAND /usr/bin/python3
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/integration_ros.py
                    --executable $<TARGET_FILE:eir_ctrl_demo_aiq11>
                    --config ${CMAKE_CURRENT_SOURCE_DIR}/config/pro.yaml
                    --model ${EIR_CTRL_SIM_TEST_MODEL} --backend ${backend}
                    --release-after-ready --ready-hold-seconds 2 --walk-seconds 5)
                set_tests_properties(ctrl_ros_release_${backend} PROPERTIES TIMEOUT 45 RUN_SERIAL TRUE
                    ENVIRONMENT "ROS_LOCALHOST_ONLY=1;ROS_DOMAIN_ID=184")
                add_test(NAME ctrl_ros_perception_${backend} COMMAND /usr/bin/python3
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/integration_ros.py
                    --executable $<TARGET_FILE:eir_ctrl_demo_aiq11>
                    --config ${CMAKE_CURRENT_SOURCE_DIR}/config/pro_perception.yaml
                    --model ${EIR_CTRL_SIM_TEST_MODEL} --backend ${backend}
                    --release-after-ready --mode 41 --walk-seconds 10
                    --switch-modes 28 30 41 --switch-seconds 3)
                set_tests_properties(ctrl_ros_perception_${backend} PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE
                    ENVIRONMENT "ROS_LOCALHOST_ONLY=1;ROS_DOMAIN_ID=184")
                add_test(NAME ctrl_ros_perception_9500_${backend} COMMAND /usr/bin/python3
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/integration_ros.py
                    --executable $<TARGET_FILE:eir_ctrl_demo_aiq11>
                    --config ${CMAKE_CURRENT_SOURCE_DIR}/config/pro_perception_9500.yaml
                    --model ${EIR_CTRL_SIM_TEST_MODEL} --backend ${backend}
                    --mode 41 --walk-seconds 5
                    --switch-modes 28 30 31 32 33 34 41 --switch-seconds 1)
                set_tests_properties(ctrl_ros_perception_9500_${backend} PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE
                    ENVIRONMENT "ROS_LOCALHOST_ONLY=1;ROS_DOMAIN_ID=184")
                add_test(NAME ctrl_ros_perception_9500_motion_${backend} COMMAND /usr/bin/python3
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/integration_ros.py
                    --executable $<TARGET_FILE:eir_ctrl_demo_aiq11>
                    --config ${CMAKE_CURRENT_SOURCE_DIR}/config/pro_perception_9500.yaml
                    --model ${EIR_CTRL_SIM_TEST_MODEL} --backend ${backend}
                    --release-after-ready --walk-seconds 1 --velocity-seconds 0 --upper-body-actions
                    --switch-modes 41 28 --switch-seconds 1)
                set_tests_properties(ctrl_ros_perception_9500_motion_${backend} PROPERTIES TIMEOUT 100 RUN_SERIAL TRUE
                    ENVIRONMENT "ROS_LOCALHOST_ONLY=1;ROS_DOMAIN_ID=188")
                add_test(NAME ctrl_ros_switch_${backend} COMMAND /usr/bin/python3
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/integration_ros.py
                    --executable $<TARGET_FILE:eir_ctrl_demo_aiq11>
                    --config ${CMAKE_CURRENT_SOURCE_DIR}/config/pro.yaml
                    --model ${EIR_CTRL_SIM_TEST_MODEL} --backend ${backend}
                    --release-after-ready --walk-seconds 3
                    --switch-modes 30 31 32 33 34 28 --switch-seconds 4)
                set_tests_properties(ctrl_ros_switch_${backend} PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE
                    ENVIRONMENT "ROS_LOCALHOST_ONLY=1;ROS_DOMAIN_ID=184")
            endif()
        endforeach()
    endif()
endif()
