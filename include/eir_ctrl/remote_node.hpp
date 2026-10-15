#pragma once
#include <eir_ctrl/runner.hpp>
#include <rclcpp/rclcpp.hpp>

namespace eir::ctrl {
// Run on a single ROS executor; Runner owns the control and IO threads.
std::shared_ptr<rclcpp::Node> makeRemoteNode(const Config&, Runner&);
} // namespace eir::ctrl
