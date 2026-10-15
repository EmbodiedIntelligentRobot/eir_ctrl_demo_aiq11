#include <algorithm>
#include <cmath>
#include <eir_ctrl/remote.hpp>
#include <stdexcept>
namespace eir::ctrl {
Sbus::Sbus(const YAML::Node &c) {
    min_ = c["minimum"].as<double>();
    center_ = c["center"].as<double>();
    max_ = c["maximum"].as<double>();
    deadband_ = c["deadband"].as<double>();
    if (!std::isfinite(min_) || !std::isfinite(center_) || !std::isfinite(max_) ||
        !std::isfinite(deadband_) ||
        !(min_ >= 0 && min_ < center_ && center_ < max_ && max_ <= 65535 && deadband_ >= 0 &&
          deadband_ < 1))
        throw std::invalid_argument("Invalid SBUS range");
    if (c["axes"].size() != 3 || c["axis_signs"].size() != 3 || c["axis_scales"].size() != 3)
        throw std::invalid_argument("SBUS needs three velocity axes");
    for (int i = 0; i < 3; ++i) {
        axes_[i] = c["axes"][i].as<std::size_t>();
        signs_[i] = c["axis_signs"][i].as<double>();
        scales_[i] = c["axis_scales"][i].as<double>();
        if (axes_[i] >= 12 || std::abs(signs_[i]) != 1 || !std::isfinite(scales_[i]) ||
            scales_[i] <= 0)
            throw std::invalid_argument("Invalid SBUS axis");
    }
    for (auto sc : c["shortcuts"]) {
        const auto mode = sc["mode"].as<int>();
        if (mode != 1 && mode != 2 &&
            !(mode >= 0 && mode <= 255 && isPolicy(static_cast<Mode>(mode))))
            throw std::invalid_argument("Unsupported SBUS mode");
        Shortcut s;
        s.mode = static_cast<Mode>(mode);
        s.channels = sc["channels"].as<std::vector<std::size_t>>();
        s.values = sc["values"].as<std::vector<int>>();
        if (s.channels.empty() || s.channels.size() != s.values.size())
            throw std::invalid_argument("Invalid SBUS shortcut");
        for (std::size_t i = 0; i < s.channels.size(); ++i)
            if (s.channels[i] >= 12 || s.values[i] < min_ || s.values[i] > max_)
                throw std::invalid_argument("Invalid SBUS shortcut condition");
        shortcuts_.push_back(std::move(s));
    }
}
RemoteOutput Sbus::update(const Feedback &f, uint64_t now, uint64_t timeout) {
    RemoteOutput out;
    out.velocity.deadline = now + timeout;
    out.valid = f.valid && f.remote_valid && now >= f.stamp && now - f.stamp <= timeout;
    for (auto value : f.remote)
        out.valid = out.valid && value >= min_ && value <= max_;
    if (!out.valid) {
        for (auto &s : shortcuts_)
            s.active = false;
        return out;
    }
    for (int i = 0; i < 3; ++i) {
        double x = f.remote[axes_[i]] - center_;
        x /= x >= 0 ? max_ - center_ : center_ - min_;
        out.velocity.value[i] =
            std::abs(x) <= deadband_
                ? 0
                : signs_[i] * scales_[i] *
                      std::copysign((std::abs(x) - deadband_) / (1 - deadband_), x);
    }
    for (auto &s : shortcuts_) {
        bool active = true;
        for (std::size_t i = 0; i < s.channels.size(); ++i)
            active = active && f.remote[s.channels[i]] == s.values[i];
        if (active && (!s.active || s.mode == Mode::Still) &&
            (!out.has_request || s.mode == Mode::Still)) {
            out.request = s.mode;
            out.has_request = true;
        }
        s.active = active;
    }
    if (out.has_request)
        out.velocity.value = {};
    return out;
}
} // namespace eir::ctrl
