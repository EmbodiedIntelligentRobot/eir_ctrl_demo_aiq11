#pragma once
#include <eir_ctrl/mapping.hpp>
#include <string_view>
namespace eir::ctrl {
// Stable numeric application reasons; public runtime events and fault codes stay unchanged.
enum class FailureOrigin : uint32_t { Control=1, Io, Inference, ExternalDepth, Scheduling };
enum class FailureReason : uint32_t {
    None, Unknown, DepthContract, DepthExpired, DepthSkew, Encoder, Actor,
    PerceptionDeadline, Nonfinite, TargetExpired, FeedbackInvalid, FeedbackFuture,
    FeedbackExpired, ReadyDeadline, JointNonfinite, Runtime, Scheduling, Hand
};
struct ModeRecord {
    uint64_t id{}, stamp{}, session{};
    Mode before{}, requested{}, actual{};
    EntryRejection rejection{};
    bool success{};
    uint32_t reason{}; // 0 none, 1 source, 2 incomplete, 3 posture, 4 disabled, 5 expired,
                       // 6 latched, 7 unsettled, 8 preempted, 9 validation, 10 STILL-only, 11 unsupported, 12 busy.
    RequestOrigin origin{RequestOrigin::Direct};
};
Config recordingConfig(Config); // Called before Runtime construction, including caller overrides.
FailureReason failureReason(Fault, FeedbackFault, std::string_view = {}) noexcept;
// These builders execute only on ordinary IO/ROS/inference threads.
loco::flight::Packet controllerPacket(const Config &, const Command &, const Feedback &,
    const std::array<MechanismEvidence,4> &, const TorqueWarning &,
    loco::flight::Event = loco::flight::Event::None);
loco::flight::Packet failurePacket(Fault, FailureOrigin, FailureReason, Mode, uint64_t session,
                                   uint64_t source_stamp);
loco::flight::Packet modePacket(const ModeRecord &);
void captureControlEvidence(Command &, const Status &, const Feedback &, const Target &,
                            uint64_t motion_generation) noexcept;
} // namespace eir::ctrl
