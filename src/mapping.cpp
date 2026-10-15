#include <Eigen/Geometry>
#include <algorithm>
#include <eir_closed_kinematics/ankle_interface.h>
#include <eir_closed_kinematics/pr_gimbal_interface.h>
#include <eir_ctrl/mapping.hpp>
#include <filesystem>
#include <stdexcept>
namespace eir::ctrl {
std::array<double, 2> policyGains(const PolicyConfig &p, const Command &cmd,
                                 std::size_t i) noexcept {
    double kp = p.kp[i], kd = p.kd[i];
    const auto &m = p.mimic;
    if (m.post_gain_seconds > 0 && cmd.motion_finished_stamp &&
        cmd.stamp >= cmd.motion_finished_stamp) {
        // Hold the last frame for one policy period, then ramp at the IO rate.
        const double progress = std::clamp(((cmd.stamp - cmd.motion_finished_stamp) * 1e-9 -
                                           p.step_seconds) / m.post_gain_seconds, 0.0, 1.0);
        if (!(m.post_kp_keep_mask & (uint64_t{1} << i)))
            kp *= 1 + (m.post_kp_ratio - 1) * progress;
        kd *= 1 + (m.post_kd_ratio - 1) * progress;
    }
    return {kp, kd};
}
struct Mapping::Impl {
    struct Pair {
        PairMapping config;
        std::unique_ptr<AnkleInterface> ankle;
        std::unique_ptr<PrGimbalInterface> gimbal;
        struct Motor {
            double position{}, velocity{}, torque{}, kp{}, kd{};
        };
        std::array<Motor, 2> last;
        bool valid{}, fallback{};
        uint32_t dk_bad_samples{}, ik_bad_samples{}, warning_samples{};
    };
    Config config;
    std::vector<Pair> pairs;
    uint64_t ready_session{};
    uint64_t damping_session{};
    Values ready_start{};
    explicit Impl(const Config &c) : config(c) {
        for (const auto &mapping : c.pairs) {
            Pair p;
            p.config = mapping;
            std::vector<std::string> files;
            for (const auto *name :
                 {"ang_search_param.txt", "fit_pitch_ploy33.txt", "fit_roll_ploy33.txt",
                  "pitch.txt", "roll.txt", "pitch_roll_rge.txt"}) {
                auto path = std::filesystem::path(mapping.resources) / name;
                if (!std::filesystem::is_regular_file(path))
                    throw std::runtime_error("Missing kinematics resource: " + path.string());
                files.push_back(path.string());
            }
            if (mapping.kind == "left_ankle" || mapping.kind == "right_ankle")
                p.ankle = std::make_unique<AnkleInterface>(ANKLE_MOTOR_POSITION_VELOCITY, files[0],
                                                           files[1], files[2], files[3], files[4],
                                                           files[5]);
            else
                p.gimbal = std::make_unique<PrGimbalInterface>(
                    mapping.kind == "waist" ? PR_GIMBAL_TYPE_WAIST : PR_GIMBAL_TYPE_NECK,
                    PR_GIMBAL_MOTOR_POSITION_VELOCITY, files[0], files[1], files[2], files[3],
                    files[4], files[5]);
            pairs.push_back(std::move(p));
        }
    }
};
Mapping::Mapping(const Config &c) : impl_(std::make_unique<Impl>(c)) {}
Mapping::~Mapping() = default;
void Mapping::configureMimic(const std::map<Mode, PolicyConfig> &policies) {
    for (int wire = 30; wire <= 34; ++wire) {
        const auto mode = static_cast<Mode>(wire);
        impl_->config.policies.erase(mode);
        if (const auto found = policies.find(mode); found != policies.end())
            impl_->config.policies.emplace(mode, found->second);
    }
}
void Mapping::updateFeedback(loco::State &state) {
    const auto &c = impl_->config;
    if (c.runtime.backend != loco::BackendType::Hardware) return;
    if (state.motors.size() != c.control.count || state.joints.size() != c.control.count)
        throw std::runtime_error("Incomplete hardware motor feedback");
    for (std::size_t i = 0; i < c.control.count; ++i) {
        const auto &m = state.motors[i];
        if (!std::isfinite(m.position) || !std::isfinite(m.velocity) || !std::isfinite(m.torque))
            throw std::runtime_error("Nonfinite hardware motor feedback");
        state.joints[i] = m; // Direct drives, including the existing 1:1 wrist contract.
    }
    const auto &encoders = state.sdk.encoder_states;
    // Keep encoder transport checks, but reconstruct anatomical waist feedback
    // from motors below, matching the validated PRO controller's motor mode.
    if (encoders.q.size() != 2 || encoders.dq.size() != 2 || encoders.status.size() != 2 ||
        c.runtime.encoder_names.size() != 2)
        throw std::runtime_error("Hardware waist feedback requires two encoders with status");
    for (std::size_t i = 0; i < 2; ++i) {
        if (!std::isfinite(encoders.q[i]) || !std::isfinite(encoders.dq[i]) ||
            (encoders.status[i] != 0 && encoders.status[i] != 1))
            throw std::runtime_error("Invalid hardware waist encoder feedback");
    }
    for (auto &p : impl_->pairs) {
        const auto a = p.config.indices[0], b = p.config.indices[1];
        const Eigen::Vector2d motor(state.motors[a].position, state.motors[b].position);
        const Eigen::Vector2d velocity(state.motors[a].velocity, state.motors[b].velocity);
        Eigen::Vector2d q, dq;
        bool valid = false;
        if (p.ankle) {
            const bool left = p.config.kind == "left_ankle";
            q = left ? p.ankle->left_ankle_DK(motor) : p.ankle->right_ankle_DK(motor);
            dq = left ? p.ankle->left_ankle_DK_vel2vel(motor, velocity)
                      : p.ankle->right_ankle_DK_vel2vel(motor, velocity);
            const auto d = p.ankle->get_flight_recorder_diagnostics(left);
            valid = d.dk_snapshot_valid && (d.dk_flags & ANKLE_DIAG_VALID) &&
                    !(d.dk_flags & (ANKLE_DIAG_FALLBACK | ANKLE_DIAG_CLAMPED |
                                   ANKLE_DIAG_NUMERIC_FAILURE | ANKLE_DIAG_OUT_OF_DOMAIN));
        } else {
            q = p.gimbal->pr_gimbal_DK(motor);
            dq = p.gimbal->pr_gimbal_DK_vel2vel(motor, velocity);
            const auto d = p.gimbal->get_flight_recorder_diagnostics();
            valid = d.snapshot_valid && (d.dk_flags & PR_DIAG_VALID) &&
                    !(d.dk_flags & (PR_DIAG_FALLBACK | PR_DIAG_CLAMPED |
                                   PR_DIAG_NUMERIC_FAILURE | PR_DIAG_OUT_OF_DOMAIN));
        }
        if (!valid || !q.allFinite() || !dq.allFinite())
            throw std::runtime_error("Invalid hardware joint reconstruction: " + p.config.kind);
        // No independent joint torque sensor. Preserve actual motor torque separately.
        state.joints[a] = {q[0], dq[0], 0};
        state.joints[b] = {q[1], dq[1], 0};
    }
}
loco::Command Mapping::command(const Command &cmd, const Feedback &f, TorqueWarning *warning,
                               MechanismWarning *mechanism_warning) {
    if (mechanism_warning)
        *mechanism_warning = {};
    if (warning)
        *warning = {};
    const auto &c = impl_->config;
    const auto *policy = cmd.policy_target ? &c.policies.at(cmd.mode) : nullptr;
    if (impl_->damping_session != cmd.session) {
        for (auto &pair : impl_->pairs)
            if (pair.ankle) pair.ankle->reset_ankle_motor_kd_mapping();
        impl_->damping_session = cmd.session;
    }
    loco::Command out(c.control.count);
    for (std::size_t i = 0; i < c.control.count; ++i) {
        if (!std::isfinite(cmd.q[i]) || !std::isfinite(cmd.dq[i]) || !std::isfinite(f.q[i]) ||
            !std::isfinite(f.dq[i]) || !std::isfinite(f.motor[i]))
            throw std::runtime_error("Nonfinite mapping input: index=" + std::to_string(i));
        out.position[i] = cmd.mode == Mode::Ready ? c.control.ready[i] : cmd.q[i];
        out.velocity[i] = cmd.dq[i];
        out.kp[i] = cmd.kp[i];
        out.kd[i] = cmd.kd[i];
        if (policy != nullptr && !(cmd.gain_mask & (uint64_t{1} << i))) {
            const auto gains = policyGains(*policy, cmd, i);
            out.kp[i] = gains[0];
            out.kd[i] = gains[1];
        }
    }
    if (policy != nullptr)
        for (auto i : c.torque_indices) {
            const bool motor_only = policy->motor_only_damping_mask & (uint64_t{1} << i);
            const double kd = out.kd[i];
            out.torque[i] = out.kp[i] * (cmd.q[i] - f.q[i]) - (motor_only ? 0 : kd) * f.dq[i];
            out.position[i] = f.q[i];
            out.kp[i] = 0;
            out.kd[i] = motor_only ? kd : policy->motor_kd[i];
        }
    for (auto &p : impl_->pairs) {
        const auto a = p.config.indices[0], b = p.config.indices[1];
        Eigen::Vector2d q(out.position[a], out.position[b]), actual(f.q[a], f.q[b]),
            tau(out.torque[a], out.torque[b]), motor;
        p.fallback = false;
        try {
            if (p.ankle) {
                const bool left = p.config.kind == "left_ankle";
                auto check = [&](const char *stage, const Eigen::Vector2d &input) {
                    const auto d = p.ankle->get_flight_recorder_diagnostics(left);
                    if (!(d.ik_flags & ANKLE_DIAG_VALID) ||
                        (d.ik_flags &
                         (ANKLE_DIAG_CLAMPED | ANKLE_DIAG_NUMERIC_FAILURE | ANKLE_DIAG_OUT_OF_DOMAIN)))
                        throw std::runtime_error(
                            "Ankle mapping invalid or outside domain: " + p.config.kind +
                            " stage=" + stage + " indices=" + std::to_string(a) + "," +
                            std::to_string(b) + " q=[" + std::to_string(input[0]) + "," +
                            std::to_string(input[1]) + "] rad flags=" + std::to_string(d.ik_flags));
                };
                motor = left ? p.ankle->left_ankle_IK(q) : p.ankle->right_ankle_IK(q);
                check("position", q);
                // Position control only needs target IK. READY interpolates in motor
                // coordinates from measured motors, even when passive joints have sagged.
                // Validate measured geometry when it is actually used for force mapping.
                if (tau[0] != 0 || tau[1] != 0) {
                    if (left)
                        p.ankle->left_ankle_IK(actual);
                    else
                        p.ankle->right_ankle_IK(actual);
                    check("measured", actual);
                    tau = left ? p.ankle->left_ankle_IK_force2tau(actual, tau)
                               : p.ankle->right_ankle_IK_force2tau(actual, tau);
                    check("force", actual);
                }
                if (!motor.allFinite() || !tau.allFinite())
                    throw std::runtime_error("Nonfinite ankle solver output: " + p.config.kind);
                if (policy && policy->ankle_damping.enabled) {
                    const auto &d = policy->ankle_damping;
                    AnkleMotorKdRequest request;
                    request.strategy = d.soft_qp ? AnkleMotorKdStrategy::SOFT_QP : AnkleMotorKdStrategy::ZERO_CROSS;
                    request.pitch_kd_target = out.kd[a]; request.roll_kd_target = out.kd[b];
                    request.pitch_weight = d.pitch_weight; request.roll_weight = d.roll_weight;
                    request.cross_weight = d.cross_weight; request.fallback_motor_kd = d.fallback_motor_kd;
                    request.dt = 1.0 / c.io_hz;
                    const auto result = left ? p.ankle->left_ankle_joint_kd_to_motor_kd(actual, request)
                                             : p.ankle->right_ankle_joint_kd_to_motor_kd(actual, request);
                    if (!result.valid || !result.motor_kd.allFinite() || (result.motor_kd.array() < 0).any())
                        throw std::runtime_error("Ankle damping mapping failed: " + p.config.kind);
                    out.kd[a] = result.motor_kd[0]; out.kd[b] = result.motor_kd[1];
                }
            } else {
                auto check = [&](const char *stage, const Eigen::Vector2d &input) {
                    const auto d = p.gimbal->get_flight_recorder_diagnostics();
                    if (!(d.ik_flags & PR_DIAG_VALID) ||
                        (d.ik_flags &
                         (PR_DIAG_CLAMPED | PR_DIAG_NUMERIC_FAILURE | PR_DIAG_OUT_OF_DOMAIN)))
                        throw std::runtime_error(
                            "Gimbal mapping invalid or outside domain: " + p.config.kind +
                            " stage=" + stage + " indices=" + std::to_string(a) + "," +
                            std::to_string(b) + " q=[" + std::to_string(input[0]) + "," +
                            std::to_string(input[1]) + "] rad flags=" + std::to_string(d.ik_flags));
                };
                motor = p.gimbal->pr_gimbal_IK(q);
                check("position", q);
                if (out.velocity[a] != 0 || out.velocity[b] != 0) {
                    const auto velocity =
                        p.gimbal->pr_gimbal_IK_vel2vel(q, {out.velocity[a], out.velocity[b]});
                    check("velocity", q);
                    if (!velocity.allFinite())
                        throw std::runtime_error("Nonfinite gimbal velocity");
                    out.velocity[a] = velocity[0];
                    out.velocity[b] = velocity[1];
                }
                if (tau[0] != 0 || tau[1] != 0) {
                    p.gimbal->pr_gimbal_IK(actual);
                    check("measured", actual);
                    tau = p.gimbal->pr_gimbal_IK_force2tau(actual, tau);
                    check("force", actual);
                }
            }
            if (!motor.allFinite() || !tau.allFinite())
                throw std::runtime_error("Nonfinite mechanism output");
        } catch (const std::runtime_error &e) {
            if (!p.valid)
                throw std::runtime_error(std::string(e.what()) +
                                         "; no valid mechanism command available");
            p.fallback = true;
            if (mechanism_warning) {
                const auto index = p.config.kind == "left_ankle" ? 0 :
                                   p.config.kind == "right_ankle" ? 1 :
                                   p.config.kind == "waist" ? 2 : 3;
                mechanism_warning->mask |= uint8_t{1} << index;
                mechanism_warning->detail[index] = e.what();
            }
            continue; // Restore the final cached command after interpolation/saturation.
        }
        out.position[a] = motor[0];
        out.position[b] = motor[1];
        out.torque[a] = tau[0];
        out.torque[b] = tau[1];
    }
    if (cmd.mode == Mode::Ready) {
        if (impl_->ready_session != cmd.session) {
            impl_->ready_start = f.motor;
            impl_->ready_session = cmd.session;
        }
        for (std::size_t i = 0; i < c.control.count; ++i)
            out.position[i] =
                impl_->ready_start[i] + cmd.ready_blend * (out.position[i] - impl_->ready_start[i]);
    }
    for (const auto &p : impl_->pairs)
        if (p.fallback)
            for (std::size_t j = 0; j < 2; ++j) {
                const auto i = p.config.indices[j];
                const auto &last = p.last[j];
                out.position[i] = last.position;
                out.velocity[i] = last.velocity;
                out.torque[i] = last.torque;
                out.kp[i] = last.kp;
                out.kd[i] = last.kd;
            }
    for (std::size_t i = 0; i < c.control.count; ++i) {
        if (!std::isfinite(out.position[i]) || !std::isfinite(out.torque[i]))
            throw std::runtime_error("Nonfinite motor command: " + c.runtime.motor_names[i] +
                                     " index=" + std::to_string(i));
        if (std::abs(out.torque[i]) > c.torque_limit[i]) {
            if (warning) {
                warning->mask |= uint64_t{1} << i;
                warning->requested[i] = out.torque[i];
            }
            out.torque[i] = std::clamp(out.torque[i], -c.torque_limit[i], c.torque_limit[i]);
        }
    }
    // Cache only complete, validated, limited commands. Fallback never refreshes the cache.
    for (auto &p : impl_->pairs)
        if (!p.fallback) {
            for (std::size_t j = 0; j < 2; ++j) {
                const auto i = p.config.indices[j];
                p.last[j] = {out.position[i], out.velocity[i], out.torque[i], out.kp[i], out.kd[i]};
            }
            p.valid = true;
        }
    return out;
}
std::array<MechanismEvidence, 4> Mapping::evidence() noexcept {
    std::array<MechanismEvidence, 4> out{};
    for (std::size_t i = 0; i < impl_->pairs.size(); ++i) {
        auto &p = impl_->pairs[i];
        auto &r = out[i];
        r.kind = p.config.kind == "left_ankle" ? 1 : p.config.kind == "right_ankle" ? 2 :
                 p.config.kind == "waist" ? 3 : 4;
        auto fill = [&](const auto &s) {
            r.sequence = s.sequence; r.dk_flags = s.dk_flags; r.ik_flags = s.ik_flags;
            for (int j = 0; j < 2; ++j) {
                r.values[j] = s.dk_input[j]; r.values[2+j] = s.dk_raw[j];
                r.values[4+j] = s.dk_output[j]; r.values[6+j] = s.ik_input[j];
                r.values[8+j] = s.ik_output[j];
            }
        };
        if (p.ankle) {
            const auto s = p.ankle->get_flight_recorder_diagnostics(p.config.kind == "left_ankle");
            fill(s);
            r.validity = unsigned(s.snapshot_valid) | (unsigned(s.dk_snapshot_valid) << 1) |
                         (unsigned(s.ik_snapshot_valid) << 2);
        } else {
            const auto s = p.gimbal->get_flight_recorder_diagnostics(); fill(s);
            r.validity = unsigned(s.snapshot_valid) | 8;
            if(s.snapshot_valid && (s.dk_flags&PR_DIAG_AVAILABLE)) r.validity|=2;
            if(s.snapshot_valid && (s.ik_flags&PR_DIAG_AVAILABLE)) r.validity|=4;
            if(s.snapshot_valid && std::isfinite(s.jacobian_determinant) &&
               std::isfinite(s.jacobian_condition) && s.jacobian_condition>=1) r.validity|=32;
            r.values[10] = s.jacobian_determinant; r.values[11] = s.jacobian_condition;
        }
        if (p.fallback) r.validity |= 16;
        auto count=[](uint32_t &n,bool failed) {n=failed ? (n==UINT32_MAX?n:n+1) : 0;};
        if(r.validity&1) {
            const uint32_t failure=p.ankle ? ANKLE_DIAG_NUMERIC_FAILURE|ANKLE_DIAG_UNAVAILABLE :
                                            PR_DIAG_NUMERIC_FAILURE|PR_DIAG_UNAVAILABLE;
            const uint32_t warning=p.ankle ? ANKLE_DIAG_FALLBACK|ANKLE_DIAG_OUT_OF_DOMAIN|ANKLE_DIAG_CLAMPED :
                                            PR_DIAG_FALLBACK|PR_DIAG_OUT_OF_DOMAIN|PR_DIAG_CLAMPED;
            if(r.validity&2) count(p.dk_bad_samples,r.dk_flags&failure);
            if(r.validity&4) count(p.ik_bad_samples,r.ik_flags&failure);
            count(p.warning_samples,(r.dk_flags|r.ik_flags)&warning);
        }
        r.dk_bad_samples=p.dk_bad_samples;r.ik_bad_samples=p.ik_bad_samples;r.warning_samples=p.warning_samples;
    }
    return out;
}
Feedback feedback(const loco::State &state, uint64_t stamp,
                  const std::array<double, 4> &base_from_imu) {
    Feedback f;
    f.sequence = state.sequence;
    f.stamp = stamp;
    if (state.joints.size() > capacity || state.motors.size() != state.joints.size() ||
        state.sdk.imu_state.empty())
        return f;
    for (std::size_t i = 0; i < state.joints.size(); ++i) {
        f.q[i] = state.joints[i].position;
        f.dq[i] = state.joints[i].velocity;
        f.motor[i] = state.motors[i].position;
        if (!std::isfinite(f.q[i]) || !std::isfinite(f.dq[i]) || !std::isfinite(f.motor[i]))
            return f;
    }
    const auto &imu = state.sdk.imu_state[0];
    Eigen::Quaterniond q(imu.quat[0], imu.quat[1], imu.quat[2], imu.quat[3]);
    if (!q.coeffs().allFinite() || q.norm() < 1e-6)
        return f;
    q.normalize();
    const Eigen::Quaterniond mount(base_from_imu[0], base_from_imu[1], base_from_imu[2],
                                   base_from_imu[3]);
    const auto world_from_base = (q * mount.conjugate()).normalized();
    f.orientation = {world_from_base.w(), world_from_base.x(), world_from_base.y(),
                     world_from_base.z()};
    const Eigen::Vector3d g = mount * (q.conjugate() * Eigen::Vector3d(0, 0, -1));
    const Eigen::Vector3d omega = mount * Eigen::Vector3d(imu.gyro[0], imu.gyro[1], imu.gyro[2]);
    for (int i = 0; i < 3; ++i) {
        if (!std::isfinite(omega[i]) || !std::isfinite(g[i]))
            return f;
        f.omega[i] = omega[i];
        f.gravity[i] = g[i];
    }
    if (state.sdk.remote_state.sbus_ch.size() >= f.remote.size()) {
        std::copy_n(state.sdk.remote_state.sbus_ch.begin(), f.remote.size(), f.remote.begin());
        f.remote_valid =
            std::any_of(f.remote.begin(), f.remote.end(), [](auto x) { return x != 0; });
    }
    f.valid = true;
    const auto& hand = state.sdk.hand_states;
    f.hand_valid = hand.q.size() == 12 && hand.code.size() == 2 &&
                   hand.code[0] == 1 && hand.code[1] == 1;
    if (f.hand_valid)
        for (std::size_t i = 0; i < 12; ++i) {
            if (!std::isfinite(hand.q[i]) || hand.q[i] < 0 || hand.q[i] > 100) {
                f.hand_valid = false; break;
            }
            f.hand[i] = hand.q[i] / 100;
        }
    return f;
}
} // namespace eir::ctrl
