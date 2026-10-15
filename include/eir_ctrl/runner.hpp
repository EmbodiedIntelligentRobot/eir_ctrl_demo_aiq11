#pragma once
#include <eir_ctrl/recording.hpp>
#include <mutex>
#include <thread>
namespace eir::ctrl {
struct DebugSnapshot {
    loco::State raw;
    Feedback feedback;
    Command command;
    loco::Command motor_command;
};
struct ControlTiming {
    uint64_t start_ns{}, execution_ns{}, interval_ns{}, skipped_ticks{};
    Mode mode{};
    Fault fault{};
};
// Sole owner of runtime lifecycle. ROS callbacks never call runtime directly.
class Runner {
    Config config_;
    loco::Runtime runtime_;
    Mapping mapping_;
    std::map<Mode, std::unique_ptr<Policy>> policies_;
    std::map<Mode, std::unique_ptr<Policy>> gripper_policies_;
    Control control_;
    MotionPlayer motion_;
    HandControl hand_;
    ExternalDepth external_depth_;
    std::atomic<bool> perception_active_{false};
    Channel<Feedback> feedback_;
    Channel<Command> command_;
    Snapshot<Observation> observation_;
    Channel<Target> target_;
    Channel<loco::DepthStack, 3> depth_; // IO -> inference; bypasses the control thread.
    Channel<Status> status_;
    Channel<ModeRecord,32> mode_records_; // Control -> IO, drained after joins on shutdown.
    std::atomic<uint64_t> recording_drops_{0};
    std::atomic<bool> running_{false};
    std::atomic<Fault> fault_{Fault::None};
    std::atomic<uint32_t> failure_context_{0}; // Latched origin:reason, separate from motion protection.
    std::atomic<uint64_t> control_us_{0}, infer_us_{0}, overruns_{0};
    // Optional acceptance capture. No allocation, formatting or IO on the producer.
    std::unique_ptr<Channel<ControlTiming, 8192>> timings_;
    std::atomic<uint64_t> timing_drops_{0};
    std::atomic<uint64_t> depth_stamp_{0};
    std::atomic<uint64_t> encoder_us_{0}, actor_us_{0}, feature_stamp_{0};
    std::atomic<uint64_t> encoder_calls_{0}, actor_calls_{0}, feature_age_at_action_ns_{0}, inference_calls_{0};
    std::thread control_thread_, inference_thread_, io_thread_;
    std::mutex debug_mutex_;
    std::mutex depth_debug_mutex_;
    loco::DepthStack depth_debug_;
    std::array<std::atomic<double>, 3> policy_velocity_{};
    DebugSnapshot debug_;
    TorqueWarning torque_warning_;
    MechanismWarning mechanism_warning_;
    std::atomic<bool> suspended_{false};
    std::mutex error_mutex_;
    std::string error_;
    void controlLoop();
    void inferenceLoop();
    void ioLoop();
    void fail(Fault reason, const std::string &detail,
              FailureOrigin origin=FailureOrigin::Io, Mode mode=Mode::Still,
              uint64_t session=0, uint64_t source_stamp=0);
    void drainModeRecords() noexcept;
    void selectMimic(loco::State &state);

  public:
    // ROS executor is the sole producer/consumer of these queues.
    Channel<Request> requests, still_requests;
    Channel<Completion, 32> completions;
    Channel<MotionRequest> motion_requests, motion_stops;
    Channel<MotionCompletion, 32> motion_completions;
    Channel<Velocity> velocities;
    explicit Runner(Config config, std::unique_ptr<loco::Backend> backend = {}, bool capture_timing = false);
    ~Runner();
    // Called only by the single ROS executor, never by the control thread.
    std::string receiveDepth(uint32_t frames, uint32_t height, uint32_t width,
        const std::vector<float>& data, const std::string& version, const std::string& frame_id,
        int64_t stamp, uint64_t sequence, int64_t ros_now);
    void start();
    void stop() noexcept;
    // Immutable after start; use for ROS diagnostics after startup selection.
    const Config &effectiveConfig() const noexcept { return config_; }
    void recordRejectedMode(uint32_t requested, uint32_t reason,
                            RequestOrigin origin = RequestOrigin::ModeService); // Sole ROS executor only.
    bool saveFlightRecord() noexcept { return runtime_.saveFlightRecord(); }
    loco::flight::Statistics flightStatistics() const noexcept {
        return runtime_.flightStatistics();
    }
    Status status();
    DebugSnapshot debug();
    loco::DepthStack depthDebug();
    std::array<double, 3> policyVelocity() const {
        return {policy_velocity_[0].load(), policy_velocity_[1].load(), policy_velocity_[2].load()};
    }
    TorqueWarning takeTorqueWarning();
    MechanismWarning takeMechanismWarning();
    bool suspended() const {
        return suspended_.load();
    }
    std::string error();
    uint64_t controlUs() const {
        return control_us_.load();
    }
    uint64_t inferenceUs() const {
        return infer_us_.load();
    }
    uint64_t overruns() const {
        return overruns_.load();
    }
    // One non-RT consumer; capture must be selected before starting workers.
    bool popTiming(ControlTiming& sample) { return timings_ && timings_->pop(sample); }
    uint64_t timingDrops() const { return timing_drops_.load(); }
    double featureAgeAtActionMs() const { return double(feature_age_at_action_ns_.load()) / 1e6; }
    uint64_t inferenceCalls() const { return inference_calls_.load(); }
    uint64_t encoderCalls() const { return encoder_calls_.load(); }
    uint64_t actorCalls() const { return actor_calls_.load(); }
    uint64_t encoderUs() const { return encoder_us_.load(); }
    uint64_t actorUs() const { return actor_us_.load(); }
    double featureAgeMs() const {
        const auto stamp = feature_stamp_.load(), now = monotonicNow();
        return stamp && now >= stamp ? double(now - stamp) / 1e6 : -1.0;
    }
    double depthAgeMs() const {
        const auto stamp = depth_stamp_.load();
        const auto now = monotonicNow();
        return stamp && now >= stamp ? double(now - stamp) / 1e6 : -1.0;
    }

  private:
    Status latest_status_{}; // ROS thread only.
};
} // namespace eir::ctrl
