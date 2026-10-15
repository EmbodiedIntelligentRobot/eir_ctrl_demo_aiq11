#pragma once
#include <eir_ctrl/control.hpp>
#include <eir_ctrl/external_depth.hpp>
#include <eir_ctrl/motion.hpp>
#include <eir_ctrl/hand.hpp>
#include <eir_loco_runtime/runtime.hpp>
#include <map>
#include <yaml-cpp/yaml.h>
namespace eir::ctrl {
struct PairMapping {
    std::string kind, resources;
    std::array<std::size_t, 2> indices;
};
struct Config {
    std::string robot, remote_source, velocity_topic, mode_service, raw_topic;
    loco::Config runtime;
    bool release_after_ready{false};
    bool remote_gui{true};
    // w,x,y,z: rotates IMU-frame vectors into the base frame.
    std::array<double, 4> base_from_imu{1, 0, 0, 0};
    ControlConfig control;
    MotionConfig motion;
    HandConfig hand;
    std::map<Mode, PolicyConfig> policies;
    // PRO A-E only. Auto resolves once before workers start; ambiguous hands select gripper.
    std::string mimic_variant{"fist"}, mimic_hand_type;
    bool mimic_fallback{false};
    double mimic_stable_seconds{0.1}, mimic_detection_seconds{1.0};
    std::map<Mode, PolicyConfig> gripper_policies;
    std::vector<PairMapping> pairs;
    std::vector<std::size_t> torque_indices;
    Values motor_kd{}, torque_limit{};
    double control_hz{}, io_hz{}, inference_hz{}, ros_hz{}, command_timeout{}, velocity_timeout{},
        request_timeout{};
    int control_cpu{}, control_priority{}, inference_cpu{};
    double limit_warning_period{2.0};
    std::string depth_image_topic;
    ExternalDepthConfig external_depth;
    YAML::Node sbus, debug;
};
Config loadConfig(const std::string &path, const std::string &mimic_override = {});
// Explicit hardware diagnostic: zero all body gains and inhibit motion/hand commands.
void enableJointObservation(Config &config);
// The hardware runtime transports motor coordinates. Application Mapping supplies
// anatomical joint feedback using the kinematics SDK and waist encoders.
loco::Config runtimeConfig(const Config &config);
} // namespace eir::ctrl
