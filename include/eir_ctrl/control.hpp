#pragma once
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <eir_ctrl/inference.hpp>
#include <eir_loco_runtime/depth.hpp>
#include <string>
#include <vector>
namespace eir::ctrl {
constexpr std::size_t capacity = 64; // Storage bound; actual counts are validated from profiles.
using Values = std::array<double, capacity>;
using HandValues = std::array<double, 12>; // 0=open, 1=closed; reserved slots 5/11 always zero.
enum class Mode : uint8_t {
    Still = 1,
    Ready = 2,
    Walk = 28,
    MimicA = 30,
    MimicB = 31,
    MimicC = 32,
    MimicD = 33,
    MimicE = 34,
    LiedownSupine = 35,
    LiedownProne = 36,
    GetupSupine = 37,
    GetupProne = 38,
    Perception = 41,
    StandToSquat = 42,
    SquatToStand = 43,
    Fault = 255
};
constexpr bool isMimic(Mode mode) noexcept {
    return (mode >= Mode::MimicA && mode <= Mode::GetupProne) ||
           mode == Mode::StandToSquat || mode == Mode::SquatToStand;
}
constexpr bool isPolicy(Mode mode) noexcept {
    return mode == Mode::Walk || mode == Mode::Perception || isMimic(mode);
}
enum class Fault : uint8_t {
    None,
    Feedback,
    Inference,
    ReadyTimeout,
    JointLimit, // Legacy fault code 4: nonfinite joint data; finite limits are warnings.
    // Codes 5, 6 and 9 retired; keep remaining diagnostic codes stable.
    Runtime = 7,
    Scheduling = 8,
    Hand = 10
};
enum class FeedbackFault : uint8_t { None, Invalid, Future, Stale };
struct Feedback {
    uint64_t sequence{}, stamp{};
    Values q{}, dq{}, motor{};
    std::array<double, 3> omega{}, gravity{};
    std::array<double, 4> orientation{1, 0, 0, 0}; // world <- base, wxyz
    std::array<uint16_t, 12> remote{};
    bool valid{}, remote_valid{};
    HandValues hand{};
    bool hand_valid{};
};
// Fixed evidence belongs to the result that produced it, never to shared policy state.
struct PolicyEvidence {
    Values raw_action{}, applied_action{};
    std::array<double, 3> velocity{};
    std::array<double, 2> phase{};
    InferenceStats timing{};
    uint64_t sequence{}, duration_us{}, depth_stamp{};
    uint32_t count{};
    bool valid{}, history_ready{}, holding{}, warmup{}, split{}, gait{};
};
struct ControlEvidence {
    PolicyEvidence policy;
    Values policy_q{};
    uint64_t feedback_sequence{}, feedback_stamp{}, target_sequence{}, target_stamp{},
        target_session{}, target_source_stamp{}, motion_frame{}, motion_frames{},
        fault_time{}, measured_limit_mask{}, target_limit_mask{}, motion_generation{},
        control_us{}, interval_ns{}, overruns{};
    double motion_seconds{};
    uint32_t fault_joint{capacity}, fault_origin{}, fault_reason{};
    Fault fault{Fault::None};
    FeedbackFault feedback_fault{FeedbackFault::None};
    Mode fault_mode{Mode::Still};
    uint8_t motion_cmd{}, motion_phase{};
    bool recovery_delayed{};
};
struct Target {
    uint64_t stamp{}, source_stamp{}, session{};
    uint64_t motion_generation{}, inference_sequence{};
    std::size_t motion_frame{}, motion_frames{};
    Values q{}, dq{};
    bool valid{};
    PolicyEvidence evidence;
};
enum class RequestOrigin : uint8_t { Direct, ModeService, Sbus };
struct Request {
    uint64_t id{}, deadline{};
    Mode mode{Mode::Still};
    RequestOrigin origin{RequestOrigin::Direct};
};
enum class EntryRejection : uint8_t { None, SourceMode, MotionIncomplete, Posture };
struct Completion {
    uint64_t id{};
    bool success{};
    Mode mode{Mode::Still};
    EntryRejection rejection{EntryRejection::None};
};
struct Velocity {
    std::array<double, 3> value{};
    uint64_t deadline{};
};
struct Command {
    uint64_t stamp{}, session{};
    uint64_t motion_finished_stamp{}; // First accepted final-frame command in this session.
    Mode mode{Mode::Still};
    double ready_blend{};
    bool policy_target{}, ready_complete{};
    Values q{}, dq{}, kp{}, kd{};
    uint64_t gain_mask{}; // Explicit per-joint gain overrides, applied by non-RT mapping.
    HandValues hand{};
    uint16_t hand_mask{}; // Offline motion ownership; independent of body joints.
    bool hand_enabled{};
    ControlEvidence evidence;
};
struct Observation {
    Feedback feedback;
    Velocity velocity;
    uint64_t session{};
    bool enabled{};
    Mode mode{Mode::Walk};
    const loco::DepthStack* depth{}; // Set/read only on inference worker; never sent across threads.
    uint64_t motion_mask{}, motion_generation{};
    Values motion_observation_position{};
    bool entry_yaw_valid{};
    std::array<double, 4> entry_yaw{1, 0, 0, 0};
};
struct Status {
    std::size_t motion_frame{}, motion_frames{};
    Mode mode{Mode::Still};
    Fault fault{Fault::None};
    Mode fault_mode{Mode::Still};
    bool ready{};
    uint64_t session{};
    std::size_t fault_joint{capacity};
    FeedbackFault feedback_fault{FeedbackFault::None};
    uint64_t fault_time{}, feedback_stamp{}, feedback_sequence{};
    uint64_t position_limit_mask{}, target_limit_mask{};
    Values limit_position{}, limit_target{};
    uint8_t upper_motion_cmd{}, upper_motion_phase{};
    double upper_motion_seconds{};
    bool upper_motion_recovery_delayed{};
    bool entry_yaw_valid{};
    std::array<double, 4> entry_yaw{1, 0, 0, 0};
};
// One producer and one consumer per instance. Fixed capacity, no retry loops or heap use.
template <class T, std::size_t N = 8> class Channel {
    static_assert(N > 1);
    static_assert(std::atomic<std::size_t>::is_always_lock_free);
    std::array<T, N> data_{};
    alignas(64) std::atomic<std::size_t> head_{0};
    alignas(64) std::atomic<std::size_t> tail_{0};

  public:
    bool push(const T &value) noexcept {
        const auto h = head_.load(std::memory_order_relaxed), next = (h + 1) % N;
        if (next == tail_.load(std::memory_order_acquire))
            return false;
        data_[h] = value;
        head_.store(next, std::memory_order_release);
        return true;
    }
    bool pop(T &value) noexcept {
        const auto t = tail_.load(std::memory_order_relaxed);
        if (t == head_.load(std::memory_order_acquire))
            return false;
        value = data_[t];
        tail_.store((t + 1) % N, std::memory_order_release);
        return true;
    }
    bool latest(T &value) noexcept {
        bool found = false;
        for (std::size_t i = 0; i < N - 1; ++i) {
            if (!pop(value))
                break;
            found = true;
        }
        return found;
    }
};
// Latest-only snapshots: producer never blocks and never retains old data when full.
// Triple buffering keeps the reader's buffer private while the writer replaces the middle.
// Exactly one producer and one consumer; requests still use ordered Channel queues.
template <class T> class Snapshot {
    static_assert(std::atomic<unsigned>::is_always_lock_free);
    std::array<T, 3> data_{};
    alignas(64) std::atomic<unsigned> middle_{1};
    unsigned back_{2}, front_{0};
    static constexpr unsigned dirty = 4;
  public:
    void push(const T& value) noexcept {
        data_[back_] = value;
        back_ = middle_.exchange(back_ | dirty, std::memory_order_acq_rel) & 3;
    }
    bool latest(T& value) noexcept {
        if (!(middle_.load(std::memory_order_acquire) & dirty)) return false;
        front_ = middle_.exchange(front_, std::memory_order_acq_rel) & 3;
        value = data_[front_];
        return true;
    }
};
// Recheck at inference entry: a lease can expire while its observation waits.
inline void expireVelocity(Velocity& velocity, uint64_t now) noexcept {
    if (velocity.deadline < now) velocity.value = {};
}
struct MimicHeadingConfig {
    bool projected{};
    std::array<double, 4> reference{1, 0, 0, 0};
    double min_horizontal_projection{}, max_angular_velocity{}, max_residual_angle{};
};
// Fixed-size geometry only; shared by request validation and policy initialization.
bool alignMimicHeading(const MimicHeadingConfig &, const Feedback &,
                       std::array<double, 4> &yaw) noexcept;
struct PolicyEntry {
    uint64_t from_modes{}; // Zero preserves normal READY / policy entry rules.
    bool require_motion_complete{};
    MimicHeadingConfig heading;
};
struct ControlConfig {
    bool still_only{false}; // Reject motion modes; configured damping remains unchanged.
    uint64_t policy_mask{uint64_t{1} << 28};
    std::array<PolicyEntry, 64> entries{};
    std::size_t count{};
    Values ready{}, ready_kp{}, ready_kd{}, lower{}, upper{};
    double policy_transition_seconds{0.15};
    double ready_seconds{}, settle_seconds{}, ready_timeout{}, ready_error{}, ready_velocity{};
    uint64_t feedback_timeout_ns{}, inference_timeout_ns{}, first_inference_timeout_ns{};
};
// Diagnostic formatting belongs to the non-real-time caller.
std::string faultDescription(const Status &, const ControlConfig &);
uint64_t monotonicNow() noexcept;
// Take the validation timestamp AFTER consuming all producer snapshots.
uint64_t receiveControlInputs(Channel<Feedback> &, Channel<Target> &, Channel<Velocity> &,
                              Feedback &, Target &, Velocity &,
                              uint64_t (*clock)() noexcept = monotonicNow) noexcept;
class Control {
    ControlConfig config_;
    Status status_;
    Values start_{};
    uint64_t entered_{}, settled_{}, awaiting_ready_since_{}, first_target_{};
    uint64_t motion_finished_{};
    bool received_target_{}, blend_policy_{};

  public:
    explicit Control(ControlConfig config) : config_(std::move(config)) {}
    Completion request(Request request, const Feedback &feedback, uint64_t now) noexcept;
    // Confirm only after the final command and hand checks in this cycle.
    Completion complete(Completion result) const noexcept;
    void fault(Fault reason) noexcept;
    Command step(const Feedback &, const Target &, uint64_t now) noexcept;
    void validateCommand(Command &) noexcept;
    Status status() const noexcept {
        return status_;
    }
};
struct MimicConfig {
    std::string motion;
    std::vector<std::size_t> tracked_indices; // TXT-only joints, after policy joints
    std::array<std::size_t, 3> torso{};       // policy slots: yaw, roll, pitch
    double first_frame_hold_seconds{};
    std::size_t warmup_steps{};
    bool loop{};
    MimicHeadingConfig heading;
    // Optional final-pose gain ramp; motor-side damping remains separately configured.
    double post_kp_ratio{1}, post_kd_ratio{1}, post_gain_seconds{};
    uint64_t post_kp_keep_mask{};
};
struct AnkleDampingConfig {
    bool enabled{}, soft_qp{};
    double pitch_weight{1}, roll_weight{1}, cross_weight{1}, fallback_motor_kd{3};
};
// Optional SPORT walk observation contract. State lives only on the inference worker.
struct AmpGaitConfig {
    bool enabled{};
    std::array<double, 7> term_clips{}; // omega, gravity, command, q, dq, previous action, phase
    double velocity_filter_hz{}, velocity_reference{}, yaw_weight{};
    double swing_min{}, swing_max{}, double_support_min{}, double_support_max{};
    double command_threshold{}, direction_threshold{}, stop_threshold{};
    double linear_accel{}, linear_decel{}, angular_accel{}, angular_decel{};
    int extra_supports{}; // Matches reference extra_cycles_on_stop (counts support samples).
    bool default_left{};
};
struct PolicyConfig {
    std::string robot, kind;
    std::string contract_path, mimic_variant;
    std::vector<std::size_t> indices, velocity_mask;
    std::vector<double> defaults, scales;
    std::size_t history{};
    double observation_clip{}, action_clip{};
    Values hold{}, kp{}, kd{}, motor_kd{};
    double step_seconds{};
    MimicConfig mimic;
    AnkleDampingConfig ankle_damping;
    // HYBRID_PRO applies ankle Kd once, on the motor side; keep configured gains intact.
    uint64_t motor_only_damping_mask{};
    AmpGaitConfig amp_gait;
    double angular_velocity_scale{1}, joint_velocity_scale{1};
    uint64_t depth_timeout_ns{}, execution_timeout_ns{};
    std::array<double, 3> velocity_min{}, velocity_max{};
    InferenceConfig inference, encoder, actor;
    bool split_perception{};
    double encoder_hz{50};
};
std::array<double, 2> policyGains(const PolicyConfig &, const Command &, std::size_t joint) noexcept;
// Owns policy-specific temporal state on the inference worker only.
class Policy {
  public:
    virtual ~Policy() = default;
    virtual void reset() = 0;
    virtual InferenceStats inferenceStats() const { return {}; }
    virtual Target step(const Observation &) = 0;
};
std::unique_ptr<Policy> makePolicy(PolicyConfig config, std::unique_ptr<Inference> inference = {});
} // namespace eir::ctrl
