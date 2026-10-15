#pragma once
#include <eir_ctrl/control.hpp>

namespace eir::ctrl {
enum class MotionPhase : uint8_t { Idle, Playing, Recovering, Returning };
struct MotionRequest { uint64_t id{}, deadline{}; uint8_t cmd{}; };
struct MotionCompletion { uint64_t id{}; uint32_t error{}; };
struct MotionFrame { double time{}; Values q{}; HandValues hand{}; };
struct OfflineMotion {
    uint8_t cmd{};
    std::string name;
    uint64_t mask{};
    uint16_t hand_mask{};
    std::vector<MotionFrame> frames;
};
struct MotionConfig {
    std::string service;
    double blend_seconds{1}, recovery_warning_seconds{0.5}, dt_cap{0.01};
    std::size_t recovery_inferences{2};
    Values kp{}, kd{}, max_velocity{}, observation_position{};
    std::vector<uint64_t> observation_groups;
    std::vector<OfflineMotion> motions;
};
MotionConfig loadMotionConfig(const std::string &file, const std::string &robot,
                             const std::vector<std::string> &joints, const ControlConfig &control);

// Immutable resources are loaded before threads start. All methods below are bounded,
// allocation-free and owned by the existing control thread.
class MotionPlayer {
    const MotionConfig &config_;
    const PolicyConfig &policy_;
    const OfflineMotion *motion_{};
    MotionPhase phase_{MotionPhase::Idle};
    uint64_t session_{}, generation_{}, mask_{}, last_at_{}, recovery_at_{}, inference_stamp_{};
    std::size_t fresh_inferences_{};
    double elapsed_{}, blend_{};
    Command last_{}, start_{};
    Values hold_{};
    void recover(uint64_t now) noexcept;

  public:
    MotionPlayer(const MotionConfig &config, const PolicyConfig &policy)
        : config_(config), policy_(policy) {}
    MotionCompletion request(MotionRequest, const Command &, uint64_t now) noexcept;
    // Exactly once per control cycle, AFTER mode and policy validation.
    void step(Command &, const Target &, uint64_t now) noexcept;
    bool recoveryDelayed(uint64_t now) const noexcept {
        return phase_ == MotionPhase::Recovering && now >= recovery_at_ &&
               (now - recovery_at_) * 1e-9 > config_.recovery_warning_seconds;
    }
    void cancel() noexcept;
    uint64_t observationMask() const noexcept;
    uint64_t generation() const noexcept { return generation_; }
    uint8_t cmd() const noexcept { return motion_ ? motion_->cmd : 0; }
    MotionPhase phase() const noexcept { return phase_; }
    double elapsed() const noexcept { return elapsed_; }
};
} // namespace eir::ctrl
