#include <algorithm>
#include <cmath>
#include <cstring>
#include <ehr_message/msg/loco_velocity.hpp>
#include <ehr_message/srv/locomotion_mode_switch.hpp>
#include <ehr_message/srv/motion_cmd.hpp>
#include <eir_ctrl/remote.hpp>
#include <eir_ctrl/runner.hpp>
#include <geometry_msgs/msg/quaternion_stamped.hpp>
#include <iostream>
#include <prime_msgs/msg/debug_joint_array.hpp>
#include <prime_msgs/msg/debug_motor_array.hpp>
#include <prime_msgs/msg/debug_runtime.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/u_int16_multi_array.hpp>
#include <std_srvs/srv/trigger.hpp>
#if EIR_CTRL_WITH_DEPTH_ROS
#include <depth_transport_msgs/msg/depth_stack.hpp>
#endif
#include <unordered_map>
#include <eir_ctrl/remote_node.hpp>

namespace eir::ctrl {
using ModeService = ehr_message::srv::LocomotionModeSwitch;
using MotionService = ehr_message::srv::MotionCmd;
using VelocityMessage = ehr_message::msg::LocoVelocity;
namespace {
const char *modeName(Mode mode) {
    switch (mode) {
    case Mode::Still:
        return "STILL";
    case Mode::Ready:
        return "READY";
    case Mode::MimicA:
        return "MIMIC_A";
    case Mode::MimicB:
        return "MIMIC_B";
    case Mode::MimicC:
        return "MIMIC_C";
    case Mode::MimicD:
        return "MIMIC_D";
    case Mode::MimicE:
        return "MIMIC_E";
    case Mode::LiedownSupine: return "LIEDOWN_SUPINE";
    case Mode::LiedownProne: return "LIEDOWN_PRONE";
    case Mode::GetupSupine: return "GETUP_SUPINE";
    case Mode::GetupProne: return "GETUP_PRONE";
    case Mode::StandToSquat: return "STAND_TO_SQUAT";
    case Mode::SquatToStand: return "SQUAT_TO_STAND";
    case Mode::Walk:
        return "AMP_HUMANOID_WALK";
    case Mode::Perception:
        return "PERCEPTION_WALK";
    default:
        return "FAULT";
    }
}
} // namespace
class RemoteNode final : public rclcpp::Node {
    Config config_;
    Runner &runner_;
    Sbus sbus_;
    VelocityPriority velocity_priority_;
    rclcpp::Subscription<VelocityMessage>::SharedPtr velocity_sub_;
    rclcpp::Publisher<VelocityMessage>::SharedPtr velocity_pub_;
    rclcpp::Publisher<std_msgs::msg::UInt16MultiArray>::SharedPtr raw_pub_;
    rclcpp::Service<ModeService>::SharedPtr service_;
    rclcpp::Service<MotionService>::SharedPtr motion_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr recorder_service_;
    rclcpp::Publisher<prime_msgs::msg::DebugRuntime>::SharedPtr runtime_pub_;
    rclcpp::Publisher<prime_msgs::msg::DebugJointArray>::SharedPtr joint_pub_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_states_pub_;
    rclcpp::Publisher<geometry_msgs::msg::QuaternionStamped>::SharedPtr orientation_pub_;
    rclcpp::Publisher<prime_msgs::msg::DebugJointArray>::SharedPtr hand_pub_;
    rclcpp::Publisher<prime_msgs::msg::DebugMotorArray>::SharedPtr motor_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub_;
#if EIR_CTRL_WITH_DEPTH_ROS
    rclcpp::Subscription<depth_transport_msgs::msg::DepthStack>::SharedPtr depth_sub_;
#endif
    uint64_t depth_published_at_{};
    rclcpp::TimerBase::SharedPtr timer_;
    struct Pending {
        std::shared_ptr<rmw_request_id_t> header;
        uint64_t deadline;
        Mode requested;
    };
    std::unordered_map<uint64_t, Pending> pending_;
    struct MotionPending {
        std::shared_ptr<rmw_request_id_t> header;
        uint64_t deadline;
    };
    std::unordered_map<uint64_t, MotionPending> motion_pending_;
    uint64_t next_id_{1}, debug_at_{0};
    Mode previous_{Mode::Fault};
    Status status_;
    uint8_t previous_motion_cmd_{}, previous_motion_phase_{};
    uint64_t limit_warning_at_{}, limit_seen_at_{}, torque_warning_at_{}, mechanism_warning_at_{}, motion_warning_at_{};
    bool suspension_seen_{}, previous_suspended_{};
    bool limits_warned_{};
    std::string limitWarning() const {
        std::string message;
        for (std::size_t i = 0; i < config_.control.count; ++i) {
            const auto bit = uint64_t{1} << i;
            const bool measured = status_.position_limit_mask & bit;
            const bool target = status_.target_limit_mask & bit;
            if (!measured && !target)
                continue;
            if (!message.empty())
                message += "; ";
            message += config_.runtime.joint_names[i] + " index=" + std::to_string(i);
            if (measured)
                message += " measured=" + std::to_string(status_.limit_position[i]);
            if (target)
                message += " target=" + std::to_string(status_.limit_target[i]);
            message += " range=[" + std::to_string(config_.control.lower[i]) + "," +
                       std::to_string(config_.control.upper[i]) + "] rad";
        }
        return message;
    }
    bool queue(Mode mode, uint64_t id, uint64_t deadline,
               RequestOrigin origin = RequestOrigin::ModeService) {
        const bool accepted = mode == Mode::Still ? runner_.still_requests.push({id, deadline, mode, origin})
                                                 : runner_.requests.push({id, deadline, mode, origin});
        if (!accepted && origin == RequestOrigin::Sbus)
            runner_.recordRejectedMode(unsigned(mode), 12, origin);
        return accepted;
    }
    void respond(const std::shared_ptr<rmw_request_id_t> &header, bool success, Mode mode,
                 const std::string &message) {
        auto response = std::make_shared<ModeService::Response>();
        response->result = success;
        // Existing wire protocol has no FAULT mode. Fault actuation is latched damping.
        response->mode = static_cast<uint8_t>(mode == Mode::Fault ? Mode::Still : mode);
        response->fb_msg = mode == Mode::Fault ? "FAULT (latched): " + message : message;
        service_->send_response(*header, *response);
    }
    void respondMotion(const std::shared_ptr<rmw_request_id_t> &header, uint32_t error) {
        MotionService::Response response;
        response.result = error == MotionService::Response::ERR_NONE;
        response.err_code = error;
        motion_service_->send_response(*header, response);
    }
    void velocity(const VelocityMessage &msg) {
        const auto now = monotonicNow();
        const std::array<double, 3> raw{msg.linear_vx, msg.linear_vy, msg.angular_wz};
        Velocity v;
        double keep = msg.keep_time;
        bool valid = std::isfinite(keep) && keep >= 0;
        for (auto x : raw)
            valid = valid && std::isfinite(x);
        if (valid && keep > 0) {
            for (int i = 0; i < 3; ++i)
                v.value[i] = std::clamp(raw[i], config_.policies.at(Mode::Walk).velocity_min[i],
                                        config_.policies.at(Mode::Walk).velocity_max[i]);
            v.deadline = now +
                         static_cast<uint64_t>(std::min(keep, config_.velocity_timeout) * 1e9);
        }
        if (!velocity_priority_.accept(msg.priority, v.deadline, now))
            return;
        runner_.velocities.push(
            v); // Invalid input publishes zero, never extends the previous lease.
    }
    void tick() {
        if (depth_pub_ && depth_pub_->get_subscription_count()) {
            const auto depth = runner_.depthDebug();
            const auto at = monotonicNow();
            if (depth.valid && depth.stamp > depth_published_at_ && at >= depth.stamp &&
                at - depth.stamp <= config_.control.feedback_timeout_ns) {
                sensor_msgs::msg::Image image;
                image.header.stamp = this->now() - rclcpp::Duration::from_nanoseconds(at - depth.stamp);
                image.header.frame_id = config_.runtime.depth.site;
                image.height = loco::DepthStack::height;
                image.width = loco::DepthStack::width;
                image.encoding = "32FC1";
                const uint16_t endian = 1;
                image.is_bigendian = *reinterpret_cast<const uint8_t*>(&endian) == 0;
                image.step = image.width * sizeof(float);
                image.data.resize(image.height * image.step);
                // Show the newest frame that the policy consumes, converted back to metres.
                constexpr auto size = loco::DepthStack::height * loco::DepthStack::width;
                for (std::size_t i = 0; i < size; ++i) {
                    const float metres = depth.data[depth.data.size() - size + i] *
                                         config_.runtime.depth.max_distance;
                    std::memcpy(image.data.data() + i * sizeof(float), &metres, sizeof(float));
                }
                depth_pub_->publish(image);
                depth_published_at_ = depth.stamp;
            }
        }
        status_ = runner_.status();
        if (status_.mode != previous_) {
            RCLCPP_INFO(get_logger(), "mode=%s fault=%u detail=%s", modeName(status_.mode),
                        unsigned(status_.fault), runner_.error().c_str());
            previous_ = status_.mode;
        }
        const auto now = monotonicNow();
        if (status_.upper_motion_cmd != previous_motion_cmd_ ||
            status_.upper_motion_phase != previous_motion_phase_) {
            RCLCPP_INFO(get_logger(), "Upper-body motion cmd=%u phase=%u seconds=%.3f",
                unsigned(status_.upper_motion_cmd), unsigned(status_.upper_motion_phase),
                status_.upper_motion_seconds);
            previous_motion_cmd_ = status_.upper_motion_cmd;
            previous_motion_phase_ = status_.upper_motion_phase;
        }
        MotionCompletion motion_completion;
        while (runner_.motion_completions.pop(motion_completion)) {
            const auto it = motion_pending_.find(motion_completion.id);
            if (it != motion_pending_.end()) {
                respondMotion(it->second.header, motion_completion.error);
                motion_pending_.erase(it);
            }
        }
        for (auto it = motion_pending_.begin(); it != motion_pending_.end();) {
            if (now > it->second.deadline) {
                respondMotion(it->second.header, MotionService::Response::ERR_REQUEST_REJECTED);
                it = motion_pending_.erase(it);
            } else ++it;
        }
        if (status_.position_limit_mask || status_.target_limit_mask) {
            limit_seen_at_ = now;
            if (!limit_warning_at_ ||
                (now - limit_warning_at_) * 1e-9 >= config_.limit_warning_period) {
                RCLCPP_WARN(get_logger(), "Joint position limit warning (continuing): %s",
                            limitWarning().c_str());
                limit_warning_at_ = now;
                limits_warned_ = true;
            }
        } else if (limits_warned_ && status_.mode != Mode::Fault &&
                   (now - limit_seen_at_) * 1e-9 >= config_.limit_warning_period) {
            RCLCPP_INFO(get_logger(), "Joint position limit warning cleared");
            limits_warned_ = false;
        }
        const bool suspended = runner_.suspended();
        if (config_.runtime.backend == loco::BackendType::Simulation &&
            (!suspension_seen_ || suspended != previous_suspended_)) {
            RCLCPP_INFO(get_logger(), "Simulation suspended=%s", suspended ? "true" : "false");
            previous_suspended_ = suspended;
            suspension_seen_ = true;
        }
        if (!torque_warning_at_ ||
            (now - torque_warning_at_) * 1e-9 >= config_.limit_warning_period) {
            const auto warning = runner_.takeTorqueWarning();
            if (warning.mask) {
                std::string message;
                for (std::size_t i = 0; i < config_.control.count; ++i)
                    if (warning.mask & (uint64_t{1} << i)) {
                        if (!message.empty())
                            message += "; ";
                        message += config_.runtime.motor_names[i] + " index=" + std::to_string(i) +
                                   " requested=" + std::to_string(warning.requested[i]) +
                                   " clamped=" +
                                   std::to_string(std::clamp(warning.requested[i],
                                                             -config_.torque_limit[i],
                                                             config_.torque_limit[i])) +
                                   " Nm";
                    }
                RCLCPP_WARN(get_logger(), "Motor feedforward torque saturated (continuing): %s",
                            message.c_str());
                torque_warning_at_ = now;
            }
        }
        if (!mechanism_warning_at_ ||
            (now - mechanism_warning_at_) * 1e-9 >= config_.limit_warning_period) {
            const auto warning = runner_.takeMechanismWarning();
            if (warning.mask) {
                for (std::size_t side = 0; side < warning.detail.size(); ++side)
                    if (warning.mask & (uint8_t{1} << side))
                        RCLCPP_WARN(get_logger(),
                                    "Mechanism mapping fallback (holding last valid motor command): %s",
                                    warning.detail[side].c_str());
                mechanism_warning_at_ = now;
            }
        }
        if (status_.upper_motion_recovery_delayed &&
            (!motion_warning_at_ || (now - motion_warning_at_) * 1e-9 >= config_.limit_warning_period)) {
            RCLCPP_WARN(get_logger(), "Upper-body motion recovery delayed (continuing): "
                        "holding upper-body target while waiting for fresh AMP inference");
            motion_warning_at_ = now;
        }
        if (!status_.upper_motion_recovery_delayed) motion_warning_at_ = 0;
        Completion completion;
        for (int i = 0; i < 31 && runner_.completions.pop(completion); ++i) {
            const auto it = pending_.find(completion.id);
            if (it != pending_.end()) {
                respond(it->second.header, completion.success, completion.mode,
                        completion.success ? "Mode applied"
                                           : completion.rejection == EntryRejection::SourceMode
                                                 ? "Entry requires the corresponding lie-down/squat mode"
                                           : completion.rejection == EntryRejection::MotionIncomplete
                                                 ? "Current reference motion must reach its final frame first"
                                           : completion.rejection == EntryRejection::Posture
                                                 ? "Initial posture/rotation speed does not match the motion; current mode retained"
                                           : (it->second.requested == Mode::Ready && isPolicy(completion.mode)
                                                  ? "READY is only allowed from STILL; current policy remains active"
                                                  : "Transition rejected, expired or preempted"));
                pending_.erase(it);
            }
        }
        for (auto it = pending_.begin(); it != pending_.end();) {
            if (now > it->second.deadline) {
                respond(it->second.header, false, status_.mode,
                        "Response deadline elapsed; query actual mode before retrying");
                it = pending_.erase(it);
            } else
                ++it;
        }
        const bool debug_due =
            runtime_pub_ &&
            now - debug_at_ >= uint64_t(1e9 / config_.debug["rate_hz"].as<double>());
        if (config_.remote_source == "sbus" || debug_due) {
            const auto snapshot = runner_.debug();
            if (config_.remote_source == "sbus") {
                auto output = sbus_.update(snapshot.feedback, now,
                                           static_cast<uint64_t>(config_.velocity_timeout * 1e9));
                const bool sbus_emit = velocity_priority_.updateSbus(output.valid);
                if (sbus_emit)
                    runner_.velocities.push(output.velocity);
                if (output.has_request)
                    queue(output.request, 0, now + uint64_t(config_.request_timeout * 1e9), RequestOrigin::Sbus);
                VelocityMessage message;
                message.priority = VelocityMessage::PRIORITY_REMOTE;
                message.linear_vx = output.velocity.value[0];
                message.linear_vy = output.velocity.value[1];
                message.angular_wz = output.velocity.value[2];
                message.keep_time = output.valid ? config_.velocity_timeout : 0;
                if (sbus_emit)
                    velocity_pub_->publish(message);
                std_msgs::msg::UInt16MultiArray raw;
                raw.data.assign(snapshot.feedback.remote.begin(), snapshot.feedback.remote.end());
                raw_pub_->publish(raw);
            }
            if (debug_due) {
                publishDebug(snapshot);
                debug_at_ = now;
            }
        }
    }
    void publishDebug(const DebugSnapshot &s) {
        const auto stamp = this->now();
        const auto now = monotonicNow();
        const auto n = config_.control.count;
        const bool valid = s.feedback.valid && s.feedback.stamp && now >= s.feedback.stamp &&
            now - s.feedback.stamp <= config_.control.feedback_timeout_ns &&
            s.raw.joints.size() == n && s.raw.motors.size() == n &&
            s.motor_command.position.size() == n;
        prime_msgs::msg::DebugRuntime runtime;
        runtime.header.stamp = stamp;
        runtime.ctrl_name = modeName(status_.mode);
        runtime.ctrl_loop_duration = runner_.controlUs() * 1e-6;
        runtime.prime_ctl_duration = runtime.ctrl_loop_duration;
        runtime.infer_time_us = runner_.inferenceUs();
        runtime.feedback_sequence = s.feedback.sequence;
        runtime.feedback_stamp_ns = s.feedback.stamp;
        runtime.feedback_age_ns = s.feedback.stamp && now >= s.feedback.stamp ?
            static_cast<int64_t>(now - s.feedback.stamp) : -1;
        runtime.feedback_valid = valid;
        runtime.command_stamp_ns = s.command.stamp;
        runtime.controller_fault = static_cast<uint8_t>(status_.fault);
        runtime.fault_joint = status_.fault_joint;
        runtime_pub_->publish(runtime);
        // Do not re-label a stale/absent snapshot as current joint/motor telemetry.
        if (!valid) return;
        sensor_msgs::msg::JointState joint_states;
        // Stamp both body telemetry messages at the local feedback receipt time.
        joint_states.header.stamp = stamp - rclcpp::Duration::from_nanoseconds(now - s.feedback.stamp);
        joint_states.name = config_.runtime.joint_names;
        for (std::size_t i = 0; i < n; ++i) {
            joint_states.position.push_back(s.feedback.q[i]);
            joint_states.velocity.push_back(s.feedback.dq[i]);
            joint_states.effort.push_back(s.raw.joints[i].torque);
        }
        joint_states_pub_->publish(joint_states);
        geometry_msgs::msg::QuaternionStamped orientation;
        orientation.header.stamp = joint_states.header.stamp;
        orientation.header.frame_id = "imu_world";
        // feedback() already normalizes SDK wxyz and applies base_from_imu.
        const auto &q = s.feedback.orientation;
        orientation.quaternion.w = q[0];
        orientation.quaternion.x = q[1];
        orientation.quaternion.y = q[2];
        orientation.quaternion.z = q[3];
        orientation_pub_->publish(orientation);
        prime_msgs::msg::DebugJointArray joint;
        prime_msgs::msg::DebugMotorArray motor;
        joint.header.stamp = stamp;
        motor.header.stamp = stamp;
        joint.joint_num = n;
        motor.motor_num = n;
        for (std::size_t i = 0; i < n; ++i) {
            joint.q_act.push_back(s.feedback.q[i]);
            joint.v_act.push_back(s.feedback.dq[i]);
            joint.tau_act.push_back(s.raw.joints[i].torque);
            joint.q_des.push_back(s.command.q[i]);
            joint.v_des.push_back(s.command.dq[i]);
            joint.tau_des.push_back(0);
            const bool policy_gain = s.command.policy_target && !(s.command.gain_mask & (uint64_t{1} << i));
            const auto gains = policy_gain ? policyGains(config_.policies.at(s.command.mode), s.command, i)
                                          : std::array<double, 2>{s.command.kp[i], s.command.kd[i]};
            const double kp = gains[0], kd = gains[1];
            joint.tau_des_fake.push_back(kp * (s.command.q[i] - s.feedback.q[i]) -
                                         kd * (s.feedback.dq[i] - s.command.dq[i]));
            const auto &m = s.raw.motors[i];
            const auto &c = s.motor_command;
            motor.q_act.push_back(m.position);
            motor.v_act.push_back(m.velocity);
            motor.tau_act.push_back(m.torque);
            motor.current.push_back(m.current);
            motor.q_des.push_back(c.position[i]);
            motor.v_des.push_back(c.velocity[i]);
            motor.tau_ff.push_back(c.torque[i]);
            motor.kp.push_back(c.kp[i]);
            motor.kd.push_back(c.kd[i]);
            motor.tau_des.push_back(c.kp[i] * (c.position[i] - m.position) +
                                    c.kd[i] * (c.velocity[i] - m.velocity) + c.torque[i]);
            motor.temperature.push_back(m.temperature);
            motor.status_code.push_back(std::min<uint32_t>(m.status_code, 255));
            motor.error_code.push_back(std::min<uint32_t>(m.error_code, 255));
        }
        joint_pub_->publish(joint);
        motor_pub_->publish(motor);
        if (hand_pub_ && s.feedback.hand_valid && s.raw.sdk.hand_states.tau.size() == 12) {
            prime_msgs::msg::DebugJointArray hand;
            hand.header.stamp = stamp;
            hand.joint_num = 12;
            hand.q_act.assign(s.feedback.hand.begin(), s.feedback.hand.end());
            hand.q_des.assign(s.command.hand.begin(), s.command.hand.end());
            hand.tau_act.assign(s.raw.sdk.hand_states.tau.begin(), s.raw.sdk.hand_states.tau.end());
            // Hand closure is dimensionless; do not mislabel SDK joint rad/s as closure/s.
            hand_pub_->publish(hand);
        }
    }

  public:
    RemoteNode(const Config &c, Runner &runner)
        : Node("eir_ctrl_demo_aiq11"), config_(c), runner_(runner), sbus_(c.sbus) {
        if (!c.external_depth.topic.empty()) {
#if EIR_CTRL_WITH_DEPTH_ROS
            if (get_parameter("use_sim_time").as_bool())
                throw std::invalid_argument("External camera requires system time; use_sim_time must be false");
            depth_sub_ = create_subscription<depth_transport_msgs::msg::DepthStack>(
                c.external_depth.topic, rclcpp::QoS(1).best_effort().durability_volatile(),
                [this](depth_transport_msgs::msg::DepthStack::ConstSharedPtr msg) {
                    auto error = runner_.receiveDepth(msg->frame_count, msg->height, msg->width,
                        msg->data, msg->preprocessing_version, msg->header.frame_id,
                        msg->header.stamp.nanosec < 1000000000U ?
                            int64_t(msg->header.stamp.sec) * 1000000000LL + msg->header.stamp.nanosec : 0,
                        msg->source_sequence, this->now().nanoseconds());
                    if (!error.empty()) RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "%s", error.c_str());
                });
#else
            throw std::invalid_argument("External camera requires EIR_CTRL_WITH_DEPTH_ROS=ON");
#endif
        }
        velocity_sub_ = create_subscription<VelocityMessage>(
            c.velocity_topic, rclcpp::QoS(1).best_effort(),
            [this](VelocityMessage::ConstSharedPtr msg, const rclcpp::MessageInfo &info) {
                // The SBUS publisher is diagnostic; do not loop its zeros back
                // into the external fallback input after SBUS becomes invalid.
                if (velocity_pub_ && std::memcmp(velocity_pub_->get_gid().data,
                        info.get_rmw_message_info().publisher_gid.data, RMW_GID_STORAGE_SIZE) == 0)
                    return;
                velocity(*msg);
            });
        if (c.remote_source == "sbus") {
            velocity_pub_ =
                create_publisher<VelocityMessage>(c.velocity_topic, rclcpp::QoS(1).best_effort());
            raw_pub_ = create_publisher<std_msgs::msg::UInt16MultiArray>(c.raw_topic,
                                                                         rclcpp::SensorDataQoS());
        }
        recorder_service_ = create_service<std_srvs::srv::Trigger>(
            "flight_recorder/save",
            [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                   std::shared_ptr<std_srvs::srv::Trigger::Response> reply) {
                reply->success = runner_.saveFlightRecord();
                reply->message =
                    reply->success
                        ? "Save request queued; not yet durable"
                        : "Recorder unavailable, disabled, exempted or request already pending";
            });
        // Deferred response keeps the single ROS executor available for velocity and STILL.
        service_ = create_service<ModeService>(
            c.mode_service, [this](std::shared_ptr<rmw_request_id_t> header,
                                   std::shared_ptr<ModeService::Request> req) {
                status_ = runner_.status();
                if (req->cmd == 0) {
                    respond(
                        header, true, status_.mode,
                        std::string(modeName(status_.mode)) +
                            (status_.ready ? " ready=true" : " ready=false") +
                            (runner_.suspended() ? " suspended=true" : " suspended=false") +
                            (config_.runtime.passive_observation ? " joint_observation=true" : "") +
                            (config_.remote_source == "sbus"
                                 ? (velocity_priority_.sbusActive() ? " sbus_input=valid"
                                                                    : " sbus_input=unavailable")
                                 : "") +
                            ((config_.runtime.depth.enabled ||
                              !config_.external_depth.topic.empty())
                                 ? " depth_age_ms=" + std::to_string(runner_.depthAgeMs())
                                 : "") +
                            (isPolicy(status_.mode)
                                 ? " input_vx=" + std::to_string(runner_.policyVelocity()[0])
                                 : "") +
                            (status_.mode == Mode::Perception
                                 ? std::string(" execution=") +
                                       (config_.policies.at(Mode::Perception).split_perception
                                            ? "split"
                                            : "combined") +
                                       " infer_us=" + std::to_string(runner_.inferenceUs()) +
                                       " encoder_us=" + std::to_string(runner_.encoderUs()) +
                                       " actor_us=" + std::to_string(runner_.actorUs()) +
                                       " feature_age_ms=" + std::to_string(runner_.featureAgeMs()) +
                                       " feature_age_at_action_ms=" +
                                       std::to_string(runner_.featureAgeAtActionMs()) +
                                       " policy_calls=" + std::to_string(runner_.inferenceCalls()) +
                                       " encoder_calls=" + std::to_string(runner_.encoderCalls()) +
                                       " actor_calls=" + std::to_string(runner_.actorCalls())
                                 : "") +
                            " recorder_healthy=" +
                            std::to_string(runner_.flightStatistics().healthy) +
                            " recorder_drops=" +
                            std::to_string(runner_.flightStatistics().dropped) +
                            " motion_frame=" + std::to_string(status_.motion_frame) + "/" +
                            std::to_string(status_.motion_frames) +
                            " upper_motion_cmd=" + std::to_string(status_.upper_motion_cmd) +
                            " upper_motion_phase=" + std::to_string(status_.upper_motion_phase) +
                            " upper_motion_seconds=" +
                            std::to_string(status_.upper_motion_seconds) +
                            " upper_motion_recovery_delayed=" +
                            (status_.upper_motion_recovery_delayed ? "true" : "false") +
                            " fault=" + std::to_string(unsigned(status_.fault)) + " joint=" +
                            std::to_string(status_.fault_joint) + " " + runner_.error() +
                            ((status_.position_limit_mask || status_.target_limit_mask)
                                 ? " warning: " + limitWarning()
                                 : ""));
                    return;
                }
                if (req->cmd != 1 && req->cmd != 2 &&
                    !config_.policies.count(static_cast<Mode>(req->cmd))) {
                    runner_.recordRejectedMode(uint32_t(req->cmd),11);
                    respond(header, false, status_.mode, "Unsupported mode");
                    return;
                }
                if (pending_.size() >= 14 || (pending_.size() >= 7 && req->cmd != 1)) {
                    runner_.recordRejectedMode(uint32_t(req->cmd),12);
                    respond(header, false, status_.mode, "Mode request queue busy");
                    return;
                }
                const auto id = next_id_++,
                           deadline = monotonicNow() + uint64_t(config_.request_timeout * 1e9);
                if (!queue(static_cast<Mode>(req->cmd), id, deadline)) {
                    runner_.recordRejectedMode(uint32_t(req->cmd),12);
                    respond(header, false, status_.mode, "Mode request queue full");
                    return;
                }
                pending_.emplace(id, Pending{header, deadline, static_cast<Mode>(req->cmd)});
            }, rclcpp::ServicesQoS().keep_last(64).get_rmw_qos_profile());
        if (!c.motion.motions.empty()) {
            motion_service_ = create_service<MotionService>(c.motion.service,
                [this](std::shared_ptr<rmw_request_id_t> header,
                       std::shared_ptr<MotionService::Request> req) {
                    // session_id remains a caller label; controller generations prevent stale targets.
                    const bool stop = req->cmd == MotionService::Request::CMD_STOP;
                    if (req->cmd < 11 && !stop) {
                        respondMotion(header, MotionService::Response::ERR_UNKNOWN_CMD);
                        return;
                    }
                    if (req->session_id.size() > 128 || motion_pending_.size() >= 14 ||
                        (!stop && motion_pending_.size() >= 7)) {
                        respondMotion(header, MotionService::Response::ERR_REQUEST_REJECTED);
                        return;
                    }
                    const auto id = next_id_++, deadline = monotonicNow() + uint64_t(config_.request_timeout * 1e9);
                    const MotionRequest request{id, deadline, req->cmd};
                    if (!(stop ? runner_.motion_stops.push(request) : runner_.motion_requests.push(request))) {
                        respondMotion(header, MotionService::Response::ERR_REQUEST_REJECTED);
                        return;
                    }
                    motion_pending_.emplace(id, MotionPending{header, deadline});
                }, rclcpp::ServicesQoS().keep_last(64).get_rmw_qos_profile());
        }
        if (c.debug["enabled"].as<bool>()) {
            joint_states_pub_ = create_publisher<sensor_msgs::msg::JointState>(
                "joint_states", rclcpp::SensorDataQoS());
            orientation_pub_ = create_publisher<geometry_msgs::msg::QuaternionStamped>(
                "base_orientation", rclcpp::SensorDataQoS());
            runtime_pub_ = create_publisher<prime_msgs::msg::DebugRuntime>(
                c.debug["runtime_topic"].as<std::string>(), rclcpp::QoS(1).best_effort());
            joint_pub_ = create_publisher<prime_msgs::msg::DebugJointArray>(
                c.debug["joint_topic"].as<std::string>(), rclcpp::QoS(1).best_effort());
            motor_pub_ = create_publisher<prime_msgs::msg::DebugMotorArray>(
                c.debug["motor_topic"].as<std::string>(), rclcpp::QoS(1).best_effort());
            if (c.hand.enabled && c.debug["hand_topic"])
                hand_pub_ = create_publisher<prime_msgs::msg::DebugJointArray>(
                    c.debug["hand_topic"].as<std::string>(), rclcpp::QoS(1).best_effort());
        }
        if (!c.depth_image_topic.empty())
            depth_pub_ = create_publisher<sensor_msgs::msg::Image>(c.depth_image_topic, rclcpp::SensorDataQoS());
        timer_ = create_wall_timer(
            std::chrono::nanoseconds(static_cast<int64_t>(1e9 / c.ros_hz)), [this] {
                const auto recorder = runner_.flightStatistics();
                if (recorder.enabled && (recorder.write_failures || recorder.quota_rejections))
                    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                                         "Flight recorder degraded: write_failures=%lu "
                                         "quota_rejections=%lu drops=%lu",
                                         recorder.write_failures, recorder.quota_rejections,
                                         recorder.dropped);
                tick();
            });
    }
};
std::shared_ptr<rclcpp::Node> makeRemoteNode(const Config& config, Runner& runner) {
    return std::make_shared<RemoteNode>(config, runner);
}
} // namespace eir::ctrl
