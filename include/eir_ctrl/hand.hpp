#pragma once
#include <eir_ctrl/control.hpp>
#include <yaml-cpp/yaml.h>

namespace eir::ctrl {
struct HandConfig {
    bool enabled{false};
    std::array<double, 3> durations{};
    std::array<HandValues, 3> positions{}; // open, main closed, secondary closed; normalized closure.
    double max_velocity{}, dt_cap{}, kp{}, kd{};
};
HandConfig loadHandConfig(const YAML::Node&);

// Control-thread only: fixed storage, no IO or allocation. SDK conversion lives in IO.
class HandControl {
    const HandConfig& config_;
    HandValues last_{}, start_{};
    uint64_t last_at_{};
    std::size_t phase_{3};
    double elapsed_{};
    bool active_{false}, motion_{false};
    void restart() noexcept { phase_ = 0; elapsed_ = 0; start_ = last_; }
  public:
    explicit HandControl(const HandConfig& config) : config_(config) {}
    bool step(Command&, const Feedback&, uint64_t now) noexcept;
};
} // namespace eir::ctrl
