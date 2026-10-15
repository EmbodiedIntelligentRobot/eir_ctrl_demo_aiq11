#include <eir_ctrl/remote_node.hpp>
#include <iostream>
#include <eir_service/health.hpp>
#include <memory>
#include "gui_process.hpp"

int main(int argc, char **argv) {
    bool ros_initialized = false;
    try {
        auto args = rclcpp::remove_ros_arguments(argc, argv);
        const bool managed_check = args.size() > 1 && args[1] == "--manager-check";
        const bool managed_run = args.size() > 1 && args[1] == "--manager-run";
        const bool managed = managed_check || managed_run;
        if (managed && args.size() != 3)
            throw std::invalid_argument("Managed usage: --manager-check|--manager-run CONFIG");
        const bool observe = args.size() > 1 && args[1] == "--observe-joints";
        if (observe) args.erase(args.begin() + 1);
        const bool check = args.size() > 1 && args[1] == "--check-config";
        const bool help = args.size() == 2 && (args[1] == "--help" || args[1] == "-h");
        if (!managed && (help || args.size() > (check ? 3U : 2U) || (check && args.size() != 3) ||
            (observe && args.size() == 1))) {
            std::cout << "Usage: eir_ctrl_demo_aiq11 [config.yaml] [--ros-args ...]\n"
                         "       eir_ctrl_demo_aiq11 --check-config config.yaml\n"
                         "       eir_ctrl_demo_aiq11 --observe-joints [--check-config] hardware.yaml\n"
                         "Default: PRO perception 9500 simulation\n"
                         "Hardware: explicit config/{pro,sport}_hardware.yaml and an AIQ hardware build\n";
            return help ? 0 : 2;
        }
        const std::string config_path = managed ? args[2] : check ? args[2] : (args.size() == 2 ? args[1] : EIR_CTRL_DEFAULT_CONFIG);
        auto config = eir::ctrl::loadConfig(config_path);
        if (managed) {
#if EIR_SERVICE_HARDWARE
            if (config.runtime.backend != eir::loco::BackendType::Hardware)
                throw std::invalid_argument("Managed service requires AIQ11 hardware");
            config.remote_gui = false;
            config.runtime.headless = true;
            config.release_after_ready = false;
#else
            throw std::invalid_argument("Managed service requires a hardware build");
#endif
        }
        if (observe) {
            eir::ctrl::enableJointObservation(config);
            std::cout << "Joint observation: all body Kp=Kd=torque=0; motion modes disabled\n";
        }
        if (!managed_check && config.runtime.backend == eir::loco::BackendType::Hardware) {
            std::cout << "MIMIC variant: " << config.mimic_variant
                      << (config.mimic_variant == "auto" ? " (pending hardware feedback; both candidates checked)" : "") << '\n';
            std::cout << "Three-finger hand control: " << (config.hand.enabled ? "enabled" : "disabled") << '\n';
        }
        if (check || managed_check) {
            eir::ctrl::Mapping mapping(config);
            if (managed_check) {
                eir::service::checked("aiq11-" + config.robot,
                    std::filesystem::canonical(config.runtime.hardware_config).string());
                return 0;
            }
            std::cout << "Configuration and kinematics resources valid; no backend started\n";
            if (config.runtime.backend == eir::loco::BackendType::Hardware) {
                std::cout << "AIQ11 " << config.robot << " SDK config: " << config.runtime.hardware_config << '\n';
                for (std::size_t i = 0; i < config.runtime.motor_names.size(); ++i)
                    std::cout << config.runtime.motor_names[i] << " -> SDK logical index "
                              << config.runtime.motor_slots[i] << ", startup/STILL Kd="
                              << (config.runtime.motor_damping_kd.empty() ? config.runtime.damping_kd
                                                                        : config.runtime.motor_damping_kd[i]) << '\n';
            }
            return 0;
        }
        rclcpp::init(argc, argv);
        ros_initialized = true;
        eir::ctrl::Runner runner(config);
        runner.start();
        config = runner.effectiveConfig();
        if (config.robot == "pro") {
            std::cout << "PRO MIMIC A-E variant: " << config.mimic_variant << '\n';
            if (config.mimic_fallback)
                std::cerr << "MIMIC hand feedback inconclusive; using gripper policies for this run\n";
            else if (config.mimic_variant == "fist" && !config.mimic_hand_type.empty())
                std::cerr << "Fist selected by offline/default feedback rule; this does not prove physical absence\n";
        }
        auto node = eir::ctrl::makeRemoteNode(config, runner);
        std::unique_ptr<eir::service::Health> health;
        rclcpp::TimerBase::SharedPtr health_timer;
        if (managed_run) {
            health = std::make_unique<eir::service::Health>("aiq11-" + config.robot);
            health_timer = node->create_wall_timer(std::chrono::milliseconds(100), [&] {
                const auto status = runner.status();
                health->publish(unsigned(status.mode), unsigned(status.fault),
                                status.feedback_sequence, status.feedback_stamp);
            });
        }
        {
            eir::ctrl::GuiProcess gui(config, config_path);
            rclcpp::spin(node);
            runner.stop(); // Stop outputs before waiting for GUI children to exit.
        }
        rclcpp::shutdown();
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        if (ros_initialized && rclcpp::ok())
            rclcpp::shutdown();
        return 1;
    }
}
