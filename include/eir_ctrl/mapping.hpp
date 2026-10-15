#pragma once
#include <eir_ctrl/config.hpp>
namespace eir::ctrl {
struct TorqueWarning {
    uint64_t mask{};
    Values requested{}; // Motor output-axis feedforward torque before saturation, Nm.
};
struct MechanismWarning {
    uint8_t mask{}; // bits: left ankle, right ankle, waist, neck
    std::array<std::string, 4> detail;
};
struct MechanismEvidence {
    uint32_t kind{}, validity{}, dk_flags{}, ik_flags{}, dk_bad_samples{}, ik_bad_samples{}, warning_samples{};
    uint64_t sequence{};
    std::array<double, 12> values{}; // DK input/raw/output, IK input/output, optional det/condition.
};
// Non-real-time worker only: existing kinematics SDK may allocate or log.
class Mapping {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    explicit Mapping(const Config &config);
    ~Mapping();
    // Startup only, before command production; keeps all kinematic mappings intact.
    void configureMimic(const std::map<Mode, PolicyConfig> &policies);
    // Hardware motor -> joint coordinates; called only by startup / ordinary IO.
    // Simulation already supplies anatomical joint feedback and is unchanged.
    void updateFeedback(loco::State &state);
    std::array<MechanismEvidence, 4> evidence() noexcept;
    loco::Command command(const Command &, const Feedback &, TorqueWarning *warning = nullptr,
                          MechanismWarning *mechanism_warning = nullptr);
};
Feedback feedback(const loco::State &state, uint64_t received_at,
                  const std::array<double, 4> &base_from_imu);
} // namespace eir::ctrl
