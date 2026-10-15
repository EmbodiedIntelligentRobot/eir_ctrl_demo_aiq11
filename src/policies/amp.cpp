#include <algorithm>
#include <cmath>
#include <eir_ctrl/control.hpp>
#include <stdexcept>
namespace eir::ctrl {
namespace {
// Same phase progression and command limits as the reference SPORT walk.
// No WBC/COM estimator: choose the first foot from lateral/yaw intent, otherwise
// the configured deterministic default. Only the inference worker calls this.
class WalkPhase {
    std::array<double, 3> velocity_{};
    double phase_{};
    bool previous_zero_{true}, stopping_{}, left_{true};
    int supports_{};
  public:
    const std::array<double, 3>& velocity() const { return velocity_; }
    std::array<double, 2> step(const std::array<double, 3>& target,
                               const AmpGaitConfig& c, double dt) {
        for (std::size_t i = 0; i < 3; ++i) {
            const bool accelerating = std::abs(target[i]) > std::abs(velocity_[i]) &&
                                      target[i] * velocity_[i] >= 0;
            const double rate = i < 2 ? (accelerating ? c.linear_accel : c.linear_decel)
                                      : (accelerating ? c.angular_accel : c.angular_decel);
            velocity_[i] += std::clamp(target[i] - velocity_[i], -rate * dt, rate * dt);
        }
        const double speed = std::hypot(velocity_[0], velocity_[1]);
        const double ratio = std::clamp((speed + c.yaw_weight * std::abs(velocity_[2])) /
                                       c.velocity_reference, 0.0, 1.0);
        const double swing = c.swing_max - (c.swing_max - c.swing_min) * ratio;
        const double support = c.double_support_max -
                               (c.double_support_max - c.double_support_min) * ratio;
        const double phase_max = 2 + 2 * support / swing;
        const double cycle = std::max(2 * (swing + support), 2 * c.swing_min);
        const bool target_zero = std::hypot(target[0], target[1]) + std::abs(target[2]) < c.command_threshold;
        const bool current_zero = speed + std::abs(velocity_[2]) < c.command_threshold;
        if (!previous_zero_ && target_zero && !stopping_) {
            stopping_ = true;
            supports_ = c.extra_supports;
        }
        if (previous_zero_ && !stopping_ && !target_zero) {
            left_ = std::abs(target[1]) > c.direction_threshold ? target[1] > 0 :
                    std::abs(target[2]) > c.direction_threshold ? target[2] > 0 : c.default_left;
            phase_ = 0;
            supports_ = 0;
        }
        if (stopping_ && !target_zero) { stopping_ = false; supports_ = 0; }
        previous_zero_ = target_zero;
        const bool stopped = current_zero && !stopping_;
        if (!current_zero || stopping_ || phase_ > 0.01)
            phase_ = std::fmod(phase_ + dt / cycle * phase_max, phase_max);
        const auto swing_ratio = [](double phase) {
            return phase < 1 ? (1 - std::cos(2 * M_PI * phase)) / 2 : 0.0;
        };
        const double first = swing_ratio(phase_);
        const double second = swing_ratio(std::fmod(phase_ - (1 + support / swing) + phase_max, phase_max));
        std::array<double, 2> out{left_ ? first : second, left_ ? second : first};
        if (stopping_ && first < c.stop_threshold && second < c.stop_threshold) {
            if (supports_ > 0) --supports_;
            else { stopping_ = false; phase_ = 0; out = {}; }
        }
        if (stopped) { phase_ = 0; out = {}; }
        return out;
    }
};
class AmpWalk final : public Policy {
    PolicyConfig c_;
    std::unique_ptr<Inference> inference_;
    std::vector<float> history_, last_;
    bool initialized_{};
    WalkPhase phase_;
    std::vector<double> filtered_velocity_;
    double filter_ratio_{};

  public:
    AmpWalk(PolicyConfig c, std::unique_ptr<Inference> inference)
        : c_(std::move(c)), inference_(std::move(inference)), history_(c_.inference.input_size),
          last_(c_.indices.size()), filtered_velocity_(c_.indices.size()),
          filter_ratio_(c_.amp_gait.velocity_filter_hz > 0 ?
              std::exp(-2 * M_PI * c_.amp_gait.velocity_filter_hz * c_.step_seconds) : 0) {}
    void reset() override {
        initialized_ = false;
        phase_ = {};
        std::fill(filtered_velocity_.begin(), filtered_velocity_.end(), 0);
        std::fill(last_.begin(), last_.end(), 0);
    }
    Target step(const Observation &input) override {
        const auto &f = input.feedback;
        std::vector<float> frame;
        frame.reserve(11 + 3 * c_.indices.size());
        for (auto x : f.omega)
            frame.push_back(x);
        for (auto x : f.gravity)
            frame.push_back(x);
        std::array<double, 3> command{};
        for (std::size_t i = 0; i < 3; ++i) {
            if (!std::isfinite(input.velocity.value[i])) throw std::runtime_error("Invalid AMP velocity");
            command[i] = std::clamp(input.velocity.value[i], c_.velocity_min[i], c_.velocity_max[i]);
        }
        std::array<double, 2> swing{};
        if (c_.amp_gait.enabled) {
            // The first history consists of a repeated stationary frame.
            if (initialized_) swing = phase_.step(command, c_.amp_gait, c_.step_seconds);
            command = phase_.velocity();
        }
        for (auto x : command) frame.push_back(x);
        for (std::size_t i = 0; i < c_.indices.size(); ++i) {
            const auto j = c_.indices[i];
            frame.push_back((input.motion_mask & (uint64_t{1} << j)
                                 ? input.motion_observation_position[j] : f.q[j]) - c_.defaults[i]);
        }
        for (std::size_t j = 0; j < c_.indices.size(); ++j) {
            const auto i = c_.indices[j];
            double v = (input.motion_mask & (uint64_t{1} << i)) ||
                       std::find(c_.velocity_mask.begin(), c_.velocity_mask.end(), i) != c_.velocity_mask.end()
                       ? 0 : f.dq[i];
            if (!std::isfinite(v)) throw std::runtime_error("Invalid AMP joint velocity");
            filtered_velocity_[j] = filter_ratio_ * filtered_velocity_[j] + (1 - filter_ratio_) * v;
            frame.push_back(filtered_velocity_[j]);
        }
        for (std::size_t i = 0; i < last_.size(); ++i)
            frame.push_back(input.motion_mask & (uint64_t{1} << c_.indices[i]) ? 0 : last_[i]);
        if (c_.amp_gait.enabled)
            for (auto x : swing) frame.push_back(x);
        const std::array<std::size_t, 7> widths{3, 3, 3, c_.indices.size(), c_.indices.size(),
                                              c_.indices.size(), c_.amp_gait.enabled ? 2U : 0U};
        std::size_t offset = 0;
        for (std::size_t term = 0; term < widths.size(); ++term) {
            const float clip = c_.amp_gait.enabled ? c_.amp_gait.term_clips[term] : c_.observation_clip;
            for (std::size_t j = 0; j < widths[term]; ++j) {
                auto& x = frame[offset++];
                if (!std::isfinite(x)) throw std::runtime_error("Invalid AMP observation");
                x = std::clamp(x, -clip, clip);
            }
        }
        if (!initialized_) {
            for (std::size_t k = 0; k < c_.history; ++k)
                std::copy(frame.begin(), frame.end(), history_.begin() + k * frame.size());
            initialized_ = true;
        } else {
            std::move(history_.begin() + frame.size(), history_.end(), history_.begin());
            std::copy(frame.begin(), frame.end(), history_.end() - frame.size());
        }
        auto output = inference_->run(history_);
        Target result;
        result.source_stamp = f.stamp;
        result.session = input.session;
        result.q = c_.hold;
        result.evidence.count = output.size();
        result.evidence.valid = result.evidence.history_ready = true;
        result.evidence.velocity = command;
        result.evidence.phase = swing;
        result.evidence.gait = c_.amp_gait.enabled;
        for (std::size_t i = 0; i < output.size(); ++i) {
            result.evidence.raw_action[i] = output[i];
            last_[i] = std::clamp(output[i], -float(c_.action_clip), float(c_.action_clip));
            result.evidence.applied_action[i] = last_[i];
            result.q[c_.indices[i]] = c_.defaults[i] + c_.scales[i] * last_[i];
        }
        result.stamp = monotonicNow();
        result.valid = true;
        return result;
    }
};
} // namespace
std::unique_ptr<Policy> makeAmp(PolicyConfig c, std::unique_ptr<Inference> inference) {
    return std::make_unique<AmpWalk>(std::move(c), std::move(inference));
}
} // namespace eir::ctrl
