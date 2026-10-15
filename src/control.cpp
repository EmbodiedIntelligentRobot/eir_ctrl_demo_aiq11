#include <algorithm>
#include <chrono>
#include <cmath>
#include <eir_ctrl/control.hpp>
namespace eir::ctrl {
uint64_t monotonicNow() noexcept {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
uint64_t receiveControlInputs(Channel<Feedback> &feedback, Channel<Target> &targets,
                              Channel<Velocity> &velocities, Feedback &f, Target &t, Velocity &v,
                              uint64_t (*clock)() noexcept) noexcept {
    feedback.latest(f);
    targets.latest(t);
    velocities.latest(v);
    return clock();
}
std::string faultDescription(const Status &s, const ControlConfig &config) {
    switch (s.fault) {
    case Fault::None:
        return {};
    case Fault::Feedback: {
        std::string reason;
        switch (s.feedback_fault) {
        case FeedbackFault::Invalid:
            reason = "Invalid feedback (joint/motor values or base IMU)";
            break;
        case FeedbackFault::Future:
            reason = "Feedback timestamp is in the future";
            break;
        case FeedbackFault::Stale:
            reason = "Feedback timeout";
            break;
        default:
            return "Feedback unavailable or invalid";
        }
        const double age_ms = s.fault_time >= s.feedback_stamp
                                  ? double(s.fault_time - s.feedback_stamp) / 1e6
                                  : -double(s.feedback_stamp - s.fault_time) / 1e6;
        return reason + "; sequence=" + std::to_string(s.feedback_sequence) +
               " age_ms=" + std::to_string(age_ms) +
               " limit_ms=" + std::to_string(config.feedback_timeout_ns / 1e6);
    }
    case Fault::Inference:
        return "Inference failed or policy target expired";
    case Fault::ReadyTimeout:
        return "READY did not settle before its deadline";
    case Fault::JointLimit:
        return "Nonfinite joint position, velocity or target; index=" +
               std::to_string(s.fault_joint);
    case Fault::Runtime:
        return "Runtime/SDK operation failed";
    case Fault::Scheduling:
        return "Thread affinity or scheduling setup failed";
    case Fault::Hand:
        return "Hand feedback unavailable/offline or invalid hand target";
    }
    return "Unknown fault";
}
void Control::fault(Fault reason) noexcept {
    if (status_.mode != Mode::Fault) {
        status_.fault_mode = status_.mode;
        status_.fault = reason;
        status_.ready = false;
        status_.mode = Mode::Fault;
        ++status_.session;
    }
}
Completion Control::request(Request r, const Feedback &f, uint64_t now) noexcept {
    bool ok = false;
    if (config_.still_only && r.mode != Mode::Still)
        return {r.id, false, status_.mode};
    if (r.deadline >= now && status_.mode != Mode::Fault) {
        if (r.mode == Mode::Still) {
            status_.mode = Mode::Still;
            status_.ready = false;
            ++status_.session;
            ok = true;
        } else if (r.mode == Mode::Ready &&
                   status_.mode == Mode::Still && f.valid &&
                   now >= f.stamp && now - f.stamp <= config_.feedback_timeout_ns) {
            start_ = f.q;
            entered_ = now;
            settled_ = 0;
            awaiting_ready_since_ = now;
            status_.ready = false;
            status_.mode = Mode::Ready;
            ++status_.session;
            ok = true;
        } else if (isPolicy(r.mode) && (config_.policy_mask & (uint64_t{1} << unsigned(r.mode))) &&
                   ((status_.mode == Mode::Ready && status_.ready) || isPolicy(status_.mode))) {
            const auto &entry = config_.entries[unsigned(r.mode)];
            if (r.mode == Mode::SquatToStand && status_.mode != Mode::StandToSquat)
                return {r.id, false, status_.mode, EntryRejection::SourceMode};
            if (entry.from_modes && !(entry.from_modes & (uint64_t{1} << unsigned(status_.mode))))
                return {r.id, false, status_.mode, EntryRejection::SourceMode};
            if (entry.require_motion_complete &&
                (!status_.motion_frames || status_.motion_frame + 1 != status_.motion_frames))
                return {r.id, false, status_.mode, EntryRejection::MotionIncomplete};
            std::array<double, 4> yaw{1, 0, 0, 0};
            if (entry.heading.projected && !alignMimicHeading(entry.heading, f, yaw))
                return {r.id, false, status_.mode, EntryRejection::Posture};
            status_.entry_yaw = yaw;
            status_.entry_yaw_valid = entry.heading.projected;
            start_ = f.q;
            blend_policy_ = isPolicy(status_.mode);
            status_.mode = r.mode;
            entered_ = now;
            first_target_ = 0;
            received_target_ = false;
            ++status_.session;
            ok = true;
        }
    }
    if (ok)
        status_.motion_frame = status_.motion_frames = 0;
    return {r.id, ok, status_.mode};
}
Completion Control::complete(Completion result) const noexcept {
    result.success = result.success && status_.mode != Mode::Fault;
    result.mode = status_.mode;
    return result;
}
Command Control::step(const Feedback &f, const Target &target, uint64_t now) noexcept {
    Command cmd;
    cmd.stamp = now;
    status_.position_limit_mask = status_.target_limit_mask = 0;
    // Report current feedback in healthy modes; retain the first fault snapshot.
    if (status_.mode != Mode::Fault) {
        status_.feedback_stamp = f.stamp;
        status_.feedback_sequence = f.sequence;
    }
    if (status_.mode != Mode::Fault &&
        (!f.valid || now < f.stamp || now - f.stamp > config_.feedback_timeout_ns)) {
        status_.feedback_fault =
            !f.valid ? FeedbackFault::Invalid
                     : (now < f.stamp ? FeedbackFault::Future : FeedbackFault::Stale);
        status_.fault_time = now;
        fault(Fault::Feedback);
    }
    if (status_.mode != Mode::Fault) {
        for (std::size_t i = 0; i < config_.count; ++i)
            if (!std::isfinite(f.q[i]) || !std::isfinite(f.dq[i])) {
                status_.fault_joint = i;
                fault(Fault::JointLimit);
                break; // Keep the first invalid joint, not the last one in the loop.
            } else if (f.q[i] < config_.lower[i] || f.q[i] > config_.upper[i]) {
                status_.position_limit_mask |= uint64_t{1} << i;
                status_.limit_position[i] = f.q[i];
            }
    }
    cmd.mode = status_.mode;
    cmd.session = status_.session;
    if (status_.mode == Mode::Ready) {
        const double elapsed = (now - entered_) * 1e-9;
        const double x = std::clamp(elapsed / config_.ready_seconds, 0.0, 1.0);
        const double blend = x * x * x * (10 + x * (-15 + 6 * x));
        cmd.ready_blend = blend;
        bool reached = x >= 1;
        for (std::size_t i = 0; i < config_.count; ++i) {
            cmd.q[i] = start_[i] + blend * (config_.ready[i] - start_[i]);
            cmd.kp[i] = config_.ready_kp[i];
            cmd.kd[i] = config_.ready_kd[i];
            reached = reached && std::abs(f.q[i] - config_.ready[i]) <= config_.ready_error &&
                      std::abs(f.dq[i]) <= config_.ready_velocity;
        }
        if (reached) {
            if (!settled_)
                settled_ = now;
            status_.ready = (now - settled_) * 1e-9 >= config_.settle_seconds;
        } else {
            settled_ = 0;
            status_.ready = false;
        }
        if (status_.ready) {
            awaiting_ready_since_ = 0;
        } else {
            // A new deviation gets one fixed recovery window, including settling.
            // Jitter cannot extend it; only complete settling starts a new window.
            if (!awaiting_ready_since_) awaiting_ready_since_ = now;
            if ((now - awaiting_ready_since_) * 1e-9 > config_.ready_timeout)
                fault(Fault::ReadyTimeout);
        }
    } else if (isPolicy(status_.mode)) {
        const bool fresh = target.valid && target.session == status_.session &&
                           target.source_stamp >= entered_ && now >= target.source_stamp &&
                           now - target.source_stamp <= config_.inference_timeout_ns;
        if (!fresh) {
            if (received_target_ || now - entered_ > config_.first_inference_timeout_ns)
                fault(Fault::Inference);
            cmd.q = start_;
            cmd.kp = config_.ready_kp;
            cmd.kd = config_.ready_kd;
        } else {
            if (!received_target_) {
                first_target_ = now;
                motion_finished_ = 0;
            }
            received_target_ = true;
            cmd.policy_target = true;
            cmd.evidence.policy = target.evidence;
            cmd.evidence.policy_q = target.q;
            if (!motion_finished_ && target.motion_frames &&
                target.motion_frame + 1 == target.motion_frames)
                motion_finished_ = now;
            cmd.motion_finished_stamp = motion_finished_;
            // Gains are assigned by the non-RT mapping stage.
            const double x =
                blend_policy_ && config_.policy_transition_seconds > 0
                    ? std::clamp((now - first_target_) * 1e-9 / config_.policy_transition_seconds,
                                 0.0, 1.0)
                    : 1.0;
            const double blend = x * x * x * (10 + x * (-15 + 6 * x));
            for (std::size_t i = 0; i < config_.count; ++i) {
                // Do not let interpolation hide an invalid incoming target.
                cmd.q[i] = x >= 1 ? target.q[i] : start_[i] + blend * (target.q[i] - start_[i]);
                cmd.dq[i] = blend * target.dq[i];
            }
            status_.motion_frame = target.motion_frame;
            status_.motion_frames = target.motion_frames;
        }
    }
    validateCommand(cmd);
    // Walk requires settled READY; preserve completion if IO skips the final READY frame.
    cmd.ready_complete = status_.ready && (status_.mode == Mode::Ready || isPolicy(status_.mode));
    return cmd;
}
void Control::validateCommand(Command &cmd) noexcept {
    status_.target_limit_mask = 0;
    if (status_.mode == Mode::Ready || isPolicy(status_.mode)) {
        for (std::size_t i = 0; i < config_.count; ++i) {
            if (!std::isfinite(cmd.q[i]) || !std::isfinite(cmd.dq[i])) {
                status_.fault_joint = i;
                fault(Fault::JointLimit);
                break;
            } else if (cmd.q[i] < config_.lower[i] || cmd.q[i] > config_.upper[i]) {
                status_.target_limit_mask |= uint64_t{1} << i;
                status_.limit_target[i] = cmd.q[i];
            }
            // Finite target limits are diagnostic. Torque-mode PD uses the
            // original target; motor torque bounds are applied after mapping.
        }
    }
    cmd.mode = status_.mode;
    cmd.session = status_.session;
    if (status_.mode == Mode::Fault) cmd.ready_complete = false;
}
} // namespace eir::ctrl
