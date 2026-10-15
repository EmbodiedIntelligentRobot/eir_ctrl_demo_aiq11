#include <eir_ctrl/remote_node.hpp>
#include <ehr_message/srv/locomotion_mode_switch.hpp>
#include <prime_msgs/msg/debug_joint_array.hpp>
#include <prime_msgs/msg/debug_motor_array.hpp>
#include <prime_msgs/msg/debug_runtime.hpp>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <thread>

using namespace eir::ctrl;
using namespace std::chrono_literals;
using Service = ehr_message::srv::LocomotionModeSwitch;
int main(int argc, char** argv) {
    try {
        if (argc != 8) throw std::invalid_argument(
            "Usage: ctrl_debug_timing config.yaml on|off mode warmup_seconds seconds output.csv backend");
        auto config = loadConfig(argv[1]);
        if (config.runtime.backend != eir::loco::BackendType::Simulation)
            throw std::invalid_argument("Acceptance capture is simulation-only");
        const std::string toggle = argv[2], backend = argv[7];
        if (toggle != "on" && toggle != "off") throw std::invalid_argument("Expected on|off");
        if (backend != "onnx" && backend != "libtorch") throw std::invalid_argument("Invalid backend");
        const int mode = std::stoi(argv[3]);
        if (mode != 1 && mode != 2 && mode != 28 && mode != 41)
            throw std::invalid_argument("Use STILL, READY, AMP or perception for sustained capture");
        const double warmup = std::stod(argv[4]), seconds = std::stod(argv[5]);
        if (!std::isfinite(warmup) || !std::isfinite(seconds) || warmup < 0 || seconds <= 0)
            throw std::invalid_argument("Invalid duration");
        config.debug["enabled"] = toggle == "on";
        config.runtime.headless = true;
        config.runtime.suspend = true;
        config.release_after_ready = false;
        config.remote_gui = false;
        config.remote_source = "external";
        for (const auto& entry : config.policies)
            if (entry.second.inference.backend != backend)
                throw std::invalid_argument("Backend must match the supplied configuration");
        std::ofstream csv(argv[6]);
        csv.exceptions(std::ios::badbit | std::ios::failbit);
        csv << "start_ns,interval_ns,execution_ns,skipped_ticks,mode,fault\n";
        rclcpp::init(argc, argv);
        Runner runner(config, {}, true);
        auto remote = makeRemoteNode(config, runner);
        auto observer = std::make_shared<rclcpp::Node>("debug_timing_observer");
        rclcpp::executors::SingleThreadedExecutor executor;
        executor.add_node(remote); executor.add_node(observer);
        auto client = observer->create_client<Service>(config.mode_service);
        std::array<uint64_t,4> counts{};
        std::size_t checked_motors = 0;
        double max_torque_error = 0;
        auto runtime_sub = observer->create_subscription<prime_msgs::msg::DebugRuntime>(
            config.debug["runtime_topic"].as<std::string>(), rclcpp::SensorDataQoS(),
            [&](prime_msgs::msg::DebugRuntime::ConstSharedPtr m) {
                ++counts[0];
                if (m->ctrl_loop_duration != m->prime_ctl_duration || m->ctrl_name == "FAULT")
                    throw std::runtime_error("Invalid runtime diagnostic or FAULT");
            });
        auto joint_sub = observer->create_subscription<prime_msgs::msg::DebugJointArray>(
            config.debug["joint_topic"].as<std::string>(), rclcpp::SensorDataQoS(),
            [&](prime_msgs::msg::DebugJointArray::ConstSharedPtr m) {
                ++counts[1];
                if (m->joint_num != config.control.count) throw std::runtime_error("Joint count differs");
                for (const auto* values : {&m->q_act,&m->q_des,&m->v_act,&m->v_des,&m->tau_act,&m->tau_des,&m->tau_des_fake}) {
                    if (values->size() != config.control.count) throw std::runtime_error("Joint array truncated");
                    for (auto x : *values) if (!std::isfinite(x)) throw std::runtime_error("Nonfinite joint debug");
                }
            });
        auto motor_sub = observer->create_subscription<prime_msgs::msg::DebugMotorArray>(
            config.debug["motor_topic"].as<std::string>(), rclcpp::SensorDataQoS(),
            [&](prime_msgs::msg::DebugMotorArray::ConstSharedPtr m) {
                ++counts[2];
                if (m->motor_num != config.control.count) throw std::runtime_error("Motor count differs");
                for (const auto* values : {&m->q_act,&m->q_des,&m->v_act,&m->v_des,&m->tau_act,&m->tau_des,&m->tau_ff,&m->kp,&m->kd,&m->current,&m->temperature}) {
                    if (values->size() != config.control.count) throw std::runtime_error("Motor array truncated");
                    for (auto x : *values) if (!std::isfinite(x)) throw std::runtime_error("Nonfinite motor debug");
                }
                if (m->status_code.size() != config.control.count || m->error_code.size() != config.control.count)
                    throw std::runtime_error("Motor status array truncated");
                for (std::size_t i = 0; i < config.control.count; ++i) {
                    const auto estimate = double(m->kp[i])*(double(m->q_des[i])-m->q_act[i]) +
                                          double(m->kd[i])*(double(m->v_des[i])-m->v_act[i]) + m->tau_ff[i];
                    const auto error = std::abs(m->tau_des[i] - estimate);
                    max_torque_error = std::max(max_torque_error, error);
                    if (error > 5e-5 * std::max(1.,std::abs(estimate))) throw std::runtime_error("Torque estimate mismatch");
                    ++checked_motors;
                }
            });
        auto hand_sub = observer->create_subscription<prime_msgs::msg::DebugJointArray>(
            config.debug["hand_topic"].as<std::string>(), rclcpp::SensorDataQoS(),
            [&](prime_msgs::msg::DebugJointArray::ConstSharedPtr m) {
                ++counts[3];
                if (m->joint_num != 12 || m->q_act.size() != 12 || !m->v_act.empty())
                    throw std::runtime_error("Invalid hand diagnostic");
            });
        runner.start();
        uint64_t capture_begin = 0, capture_end = 0, captured = 0;
        auto drain = [&] {
            ControlTiming sample;
            while (runner.popTiming(sample)) {
                if (sample.start_ns >= capture_begin && sample.start_ns < capture_end) {
                    csv << sample.start_ns << ',' << sample.interval_ns << ',' << sample.execution_ns << ','
                        << sample.skipped_ticks << ',' << unsigned(sample.mode) << ',' << unsigned(sample.fault) << '\n';
                    ++captured;
                    if (sample.fault != Fault::None || unsigned(sample.mode) != unsigned(mode))
                        throw std::runtime_error("Wrong mode or fault during capture");
                }
            }
        };
        auto step = [&] {
            executor.spin_some(); drain();
            if (runner.status().fault != Fault::None) throw std::runtime_error(runner.error());
            std::this_thread::sleep_for(1ms);
        };
        auto call = [&](int cmd) {
            auto request = std::make_shared<Service::Request>(); request->cmd = cmd;
            auto future = client->async_send_request(request);
            const auto end = monotonicNow() + 3000000000;
            while (future.wait_for(0s) != std::future_status::ready) {
                step();
                if (monotonicNow() > end) throw std::runtime_error("Mode request timed out");
            }
            const auto result = future.get();
            if (!result->result) throw std::runtime_error(result->fb_msg);
        };
        const auto discovery_end = monotonicNow() + 5000000000;
        while (!client->service_is_ready()) {
            step(); if (monotonicNow() > discovery_end) throw std::runtime_error("Service unavailable");
        }
        if (mode != 1) {
            call(2);
            const auto ready_end = monotonicNow() + 15000000000;
            while (!runner.status().ready) {
                step(); if (monotonicNow() > ready_end) throw std::runtime_error("READY timeout");
            }
            if (mode != 2) call(mode);
        }
        const auto warmup_end = monotonicNow() + uint64_t(warmup*1e9);
        while (monotonicNow() < warmup_end) step();
        counts.fill(0); checked_motors = 0; max_torque_error = 0;
        capture_begin = monotonicNow(); capture_end = capture_begin + uint64_t(seconds*1e9);
        while (monotonicNow() < capture_end) step();
        runner.stop(); drain(); csv.close();
        if (runner.timingDrops()) throw std::runtime_error("Incomplete capture: timing samples dropped");
        for (auto count : counts) if ((toggle == "on") != (count > 0))
            throw std::runtime_error("Debug toggle does not match received topics");
        std::cout << "T10_RESULT {\"samples\":" << captured << ",\"drops\":" << runner.timingDrops()
                  << ",\"overrun_events_including_warmup\":" << runner.overruns()
                  << ",\"debug_counts\":[" << counts[0] << ',' << counts[1] << ',' << counts[2] << ',' << counts[3]
                  << "],\"checked_motor_values\":" << checked_motors
                  << ",\"max_torque_error_Nm\":" << std::setprecision(10) << max_torque_error << "}\n";
        rclcpp::shutdown();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "T10 capture failed: " << e.what() << '\n';
        if (rclcpp::ok()) rclcpp::shutdown();
        return 1;
    }
}
