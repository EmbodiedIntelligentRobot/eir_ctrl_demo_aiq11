#pragma once
#include <eir_ctrl/config.hpp>
namespace eir::ctrl {
// ROS receive-thread arbitration. Lower numbers have higher priority; rejected
// messages are discarded, so an expired source cannot revive a queued command.
class VelocityPriority {
    uint64_t remote_deadline_{};
    bool sbus_active_{};

  public:
    bool sbusActive() const noexcept { return sbus_active_; }
    // Emit SBUS commands while valid, and one zero on loss. Repeated invalid
    // snapshots must not overwrite the newly operated Xbox/GUI fallback.
    bool updateSbus(bool valid) noexcept {
        const bool emit = valid || sbus_active_;
        if (valid != sbus_active_)
            remote_deadline_ = 0;
        sbus_active_ = valid;
        return emit;
    }
    bool accept(uint8_t priority, uint64_t deadline, uint64_t now) noexcept {
        if (sbus_active_ || priority > 1 || (priority == 1 && now < remote_deadline_))
            return false;
        if (priority == 0)
            remote_deadline_ = deadline;
        return true;
    }
};
struct RemoteOutput {
    Velocity velocity;
    Mode request{Mode::Still};
    bool has_request{}, valid{};
};
class Sbus {
    struct Shortcut {
        Mode mode;
        std::vector<std::size_t> channels;
        std::vector<int> values;
        bool active{};
    };
    std::array<std::size_t, 3> axes_{};
    std::array<double, 3> signs_{}, scales_{};
    std::vector<Shortcut> shortcuts_;
    double min_{}, center_{}, max_{}, deadband_{};

  public:
    explicit Sbus(const YAML::Node &config);
    RemoteOutput update(const Feedback &feedback, uint64_t now, uint64_t timeout_ns);
};
} // namespace eir::ctrl
