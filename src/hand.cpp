#include <eir_ctrl/hand.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace eir::ctrl {
HandConfig loadHandConfig(const YAML::Node& n) {
    auto require = [](bool ok) { if (!ok) throw std::invalid_argument("Invalid three-finger hand configuration"); };
    require(n.IsMap());
    const std::array<std::string, 9> keys{"enabled", "durations", "open_position", "main_position",
        "closed_position", "max_velocity", "dt_cap", "kp", "kd"};
    for (const auto& entry : n)
        require(std::find(keys.begin(), keys.end(), entry.first.as<std::string>()) != keys.end());
    HandConfig c;
    c.enabled = n["enabled"].as<bool>();
    require(n["durations"].IsSequence() && n["durations"].size() == 3);
    for (std::size_t i = 0; i < 3; ++i) {
        c.durations[i] = n["durations"][i].as<double>();
        require(std::isfinite(c.durations[i]) && c.durations[i] >= .01 && c.durations[i] <= 60);
        const auto p = n[keys[i + 2]];
        require(p.IsSequence() && p.size() == 12);
        for (std::size_t j = 0; j < 12; ++j) {
            c.positions[i][j] = p[j].as<double>();
            require(std::isfinite(c.positions[i][j]) && c.positions[i][j] >= 0 && c.positions[i][j] <= 1);
        }
        require(c.positions[i][5] == 0 && c.positions[i][11] == 0);
    }
    c.max_velocity = n["max_velocity"].as<double>();
    c.dt_cap = n["dt_cap"].as<double>();
    c.kp = n["kp"].as<double>(); c.kd = n["kd"].as<double>();
    require(std::isfinite(c.max_velocity) && c.max_velocity > 0 && c.max_velocity <= 10);
    require(std::isfinite(c.dt_cap) && c.dt_cap > 0 && c.dt_cap <= .02);
    require(std::isfinite(c.kp) && c.kp > 0 && c.kp <= 100 &&
            std::isfinite(c.kd) && c.kd > 0 && c.kd <= 10);
    return c;
}

bool HandControl::step(Command& cmd, const Feedback& feedback, uint64_t now) noexcept {
    cmd.hand_enabled = false;
    if (!config_.enabled) return true;
    const bool enabled = cmd.mode == Mode::Ready || isPolicy(cmd.mode);
    if (!enabled) {
        active_ = motion_ = false;
        last_at_ = now;
        cmd.hand = last_;
        return true;
    }
    if (!feedback.hand_valid) return false;
    if (!active_) { last_ = feedback.hand; last_[5] = last_[11] = 0; restart(); active_ = true; }
    const auto dt = last_at_ && now >= last_at_ ? std::min((now - last_at_) * 1e-9, config_.dt_cap) : 0;
    last_at_ = now;
    const bool motion = cmd.hand_mask != 0;
    if (motion_ && !motion) restart(); // STOP/end/mode exit: restart from the last sent target.
    motion_ = motion;
    HandValues desired = last_;
    if (motion) {
        for (std::size_t i = 0; i < 12; ++i)
            if (cmd.hand_mask & (uint16_t{1} << i)) desired[i] = cmd.hand[i];
    } else if (phase_ < 3) {
        elapsed_ = std::min(config_.durations[phase_], elapsed_ + dt);
        const double alpha = elapsed_ / config_.durations[phase_];
        for (std::size_t i = 0; i < 12; ++i)
            desired[i] = start_[i] + alpha * (config_.positions[phase_][i] - start_[i]);
        // A rate-limited phase must reach its target before advancing.
    } else desired = config_.positions.back();
    for (std::size_t i = 0; i < 12; ++i) {
        if (!std::isfinite(desired[i]) || desired[i] < 0 || desired[i] > 1) return false;
        last_[i] += std::clamp(desired[i] - last_[i], -config_.max_velocity * dt, config_.max_velocity * dt);
    }
    last_[5] = last_[11] = 0;
    if (!motion && phase_ < 3 && elapsed_ >= config_.durations[phase_] &&
        std::equal(last_.begin(), last_.end(), config_.positions[phase_].begin(),
                   [](double a, double b) { return std::abs(a - b) < 1e-9; })) {
        ++phase_; elapsed_ = 0; start_ = last_;
    }
    cmd.hand = last_;
    cmd.hand_enabled = true;
    return true;
}
} // namespace eir::ctrl
