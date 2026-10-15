#include <algorithm>
#include <chrono>
#include <cmath>
#include <eir_ctrl/runner.hpp>
#include <pthread.h>
#include <sched.h>
#include <sstream>
#include <stdexcept>
#include <system_error>
namespace eir::ctrl {
namespace {
bool configureThread(int cpu, int priority) noexcept {
    if (cpu >= 0) {
        if (cpu >= CPU_SETSIZE)
            return false;
        cpu_set_t cpus;
        CPU_ZERO(&cpus);
        CPU_SET(cpu, &cpus);
        if (pthread_setaffinity_np(pthread_self(), sizeof(cpus), &cpus))
            return false;
    }
    sched_param param{};
    param.sched_priority = priority;
    return pthread_setschedparam(pthread_self(), priority ? SCHED_FIFO : SCHED_OTHER, &param) == 0;
}
class Period {
    std::chrono::steady_clock::time_point next_{std::chrono::steady_clock::now()};
    std::chrono::nanoseconds period_;

  public:
    explicit Period(double hz) : period_(static_cast<int64_t>(1e9 / hz)) {}
    void reset(double hz) noexcept {
        period_ = std::chrono::nanoseconds(static_cast<int64_t>(1e9 / hz));
        next_ = std::chrono::steady_clock::now();
    }
    uint64_t wait() noexcept {
        next_ += period_;
        const auto now = std::chrono::steady_clock::now();
        const uint64_t missed = now > next_ ? (now - next_) / period_ + 1 : 0;
        if (missed)
            next_ += period_ * missed; // Skip elapsed ticks without a burst.
        std::this_thread::sleep_until(next_);
        return missed;
    }
};
std::string initialFeedbackError(const loco::State &state, uint64_t samples) {
    std::ostringstream message;
    message << "Timed out waiting for valid base IMU; samples=" << samples
            << ", imu_count=" << state.sdk.imu_state.size();
    if (!state.sdk.imu_state.empty()) {
        const auto &imu = state.sdk.imu_state[0];
        message << ", quat(wxyz)=[" << imu.quat[0] << ',' << imu.quat[1] << ','
                << imu.quat[2] << ',' << imu.quat[3] << "], gyro=[" << imu.gyro[0]
                << ',' << imu.gyro[1] << ',' << imu.gyro[2] << ']';
    }
    return message.str();
}
} // namespace
Runner::Runner(Config c, std::unique_ptr<loco::Backend> backend, bool capture_timing)
    : config_(recordingConfig(std::move(c))), runtime_(runtimeConfig(config_), std::move(backend)), mapping_(config_),
      control_(config_.control), motion_(config_.motion, config_.policies.at(Mode::Walk)),
      hand_(config_.hand),
      external_depth_(config_.external_depth) {
    if (capture_timing) timings_ = std::make_unique<Channel<ControlTiming, 8192>>();
    for (const auto &[mode, policy] : config_.policies)
        policies_.emplace(mode, makePolicy(policy));
    for (const auto &[mode, policy] : config_.gripper_policies)
        gripper_policies_.emplace(mode, makePolicy(policy));
}
Runner::~Runner() {
    stop();
}
void Runner::fail(Fault reason, const std::string &detail, FailureOrigin origin,
                  Mode mode, uint64_t session, uint64_t stamp) {
    std::lock_guard lock(error_mutex_);
    Fault expected = Fault::None;
    const bool first=fault_.compare_exchange_strong(expected, reason);
    const auto cause=failureReason(reason,FeedbackFault::None,detail);
    if(first) failure_context_.store((unsigned(origin)<<16)|unsigned(cause),std::memory_order_release);
    auto event = failurePacket(reason, origin,
        failureReason(reason,FeedbackFault::None,detail),mode,session,stamp);
    if (!first || !running_ || (reason==Fault::Runtime && runtime_.status().mode==loco::Mode::Estopped))
        event.event=loco::flight::Event::Warning;
    runtime_.recordApplication(event);
    if(first) error_=detail;
}
void Runner::drainModeRecords() noexcept {
    ModeRecord record;
    for(unsigned i=0;i<31 && mode_records_.pop(record);++i)
        runtime_.recordApplication(modePacket(record));
}
void Runner::recordRejectedMode(uint32_t requested,uint32_t reason,RequestOrigin origin) {
    if(!config_.runtime.flight_recorder.enabled) return;
    const auto s=status();
    auto p=modePacket({0,monotonicNow(),s.session,s.mode,static_cast<Mode>(requested),s.mode,
                       EntryRejection::None,false,reason,origin});
    // Preserve unknown wire values without narrowing them to the Mode enum.
    uint32_t raw[]={requested};p.integers(536,raw,1);runtime_.recordApplication(p);
}
std::string Runner::error() {
    std::lock_guard lock(error_mutex_);
    if (!error_.empty())
        return error_;
    auto s = status(); // Same single ROS consumer as status(). No formatting in controlLoop.
    if (s.fault == Fault::None)
        s.fault = fault_.load(std::memory_order_acquire);
    return faultDescription(s, config_.control);
}
std::string Runner::receiveDepth(uint32_t frames, uint32_t height, uint32_t width,
    const std::vector<float>& data, const std::string& version, const std::string& frame_id,
    int64_t stamp, uint64_t sequence, int64_t ros_now) {
    auto error = external_depth_.receive(frames, height, width, data, version, frame_id,
                                        stamp, sequence, ros_now, monotonicNow());
    depth_stamp_ = external_depth_.acquisitionStamp();
    if (!error.empty() && perception_active_) fail(Fault::Inference, error, FailureOrigin::ExternalDepth, Mode::Perception,0,depth_stamp_);
    return error;
}
void Runner::selectMimic(loco::State &state) {
    if (config_.mimic_variant != "auto") return;
    const auto begin = monotonicNow();
    const auto deadline = begin + uint64_t(config_.mimic_detection_seconds * 1e9);
    const auto stable_ns = uint64_t(config_.mimic_stable_seconds * 1e9);
    uint64_t sequence = 0, since = 0, last = 0;
    int candidate = 0, selected = 2;
    bool confirmed = false;
    while (monotonicNow() < deadline) {
        const auto now = monotonicNow();
        const auto status = runtime_.status();
        if (status.mode == loco::Mode::Estopped) throw std::runtime_error(status.message);
        if (!runtime_.read(state)) throw std::runtime_error("MIMIC detection feedback unavailable");
        if (last && now - last > config_.control.feedback_timeout_ns) {
            candidate = 0; since = 0;
        }
        if (state.sequence != sequence) {
            sequence = state.sequence;
            last = now;
            const auto &h = state.sdk.hand_states;
            int next = 0;
            if (h.code.size() == 2 && h.q.size() == 12 &&
                std::all_of(h.q.begin(), h.q.end(), [](double q) {
                    return std::isfinite(q) && q >= 0 && q <= 100;
                })) {
                if (h.code[0] == 1 && h.code[1] == 1) next = 2;
                else if (h.code[0] == 0 && h.code[1] == 0 &&
                         std::all_of(h.q.begin(), h.q.end(), [](double q) { return q == 0; })) next = 1;
            }
            if (!next || next != candidate) { candidate = next; since = now; }
            else if (now - since >= stable_ns) { selected = candidate; confirmed = true; break; }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    config_.mimic_variant = selected == 2 ? "gripper" : "fist";
    config_.mimic_fallback = !confirmed;
    for (int wire = 30; wire <= 34; ++wire) {
        const auto mode = static_cast<Mode>(wire);
        if (!config_.policies.count(mode)) continue;
        if (selected == 2) {
            config_.policies.at(mode) = config_.gripper_policies.at(mode);
            policies_.at(mode) = std::move(gripper_policies_.at(mode));
        }
    }
    config_.gripper_policies.clear(); gripper_policies_.clear();
    mapping_.configureMimic(config_.policies);
    // Detection takes time. Re-read current feedback before starting workers.
    if (!runtime_.read(state)) throw std::runtime_error("Startup feedback unavailable after MIMIC detection");
    mapping_.updateFeedback(state);
    if (!feedback(state, monotonicNow(), config_.base_from_imu).valid)
        throw std::runtime_error("Invalid base feedback after MIMIC detection");
}
void Runner::start() {
    if (running_)
        throw std::logic_error("Runner already started");
    try {
        if (!runtime_.start())
            throw std::runtime_error(runtime_.status().message);
        loco::State first;
        Feedback f;
        const auto deadline = std::chrono::steady_clock::now() + config_.runtime.startup_timeout;
        uint64_t sequence = 0, samples = 0;
        // Motor feedback can precede the first IMU packet. Keep runtime in damping
        // and its watchdog active; no control/inference worker starts before valid feedback.
        for (;;) {
            const auto status = runtime_.status();
            if (status.mode == loco::Mode::Estopped)
                throw std::runtime_error(status.message);
            if (samples && std::chrono::steady_clock::now() >= deadline)
                throw std::runtime_error(initialFeedbackError(first, samples));
            if (!runtime_.read(first))
                throw std::runtime_error("Initial runtime feedback unavailable");
            if (first.sequence != sequence) {
                sequence = first.sequence;
                ++samples;
                const auto received = monotonicNow();
                mapping_.updateFeedback(first); // Mapping/encoder failures are never retried.
                f = feedback(first, received, config_.base_from_imu);
                if (f.valid)
                    break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        selectMimic(first);
        f = feedback(first, monotonicNow(), config_.base_from_imu);
        feedback_.push(f);
        suspended_ = runtime_.status().suspended;
        running_ = true;
        io_thread_ = std::thread(&Runner::ioLoop, this);
        inference_thread_ = std::thread(&Runner::inferenceLoop, this);
        control_thread_ = std::thread(&Runner::controlLoop, this);
    } catch (...) {
        stop();
        throw;
    }
}
void Runner::stop() noexcept {
    running_ = false;
    runtime_.estop(false); // Inhibit hardware before waiting for any worker or ROS/GUI teardown.
    if (control_thread_.joinable())
        control_thread_.join();
    if (io_thread_.joinable())
        io_thread_.join();
    // Actuation is already inhibited above. Join the last diagnostic producer
    // before runtime.stop() finishes the recorder and its pending incidents.
    if (inference_thread_.joinable())
        inference_thread_.join();
    drainModeRecords();
    const auto drops = recording_drops_.load();
    if (drops && config_.runtime.flight_recorder.enabled) {
        loco::flight::Packet packet;
        packet.event = loco::flight::Event::Warning;
        uint32_t count[] = {uint32_t(drops), uint32_t(drops >> 32)};
        packet.integers(535, count, 2);
        runtime_.recordApplication(packet);
    }
    runtime_.stop();
}
Status Runner::status() {
    status_.latest(latest_status_);
    return latest_status_;
}
DebugSnapshot Runner::debug() {
    std::lock_guard lock(debug_mutex_);
    return debug_;
}
loco::DepthStack Runner::depthDebug() {
    std::lock_guard lock(depth_debug_mutex_);
    return depth_debug_;
}
TorqueWarning Runner::takeTorqueWarning() {
    std::lock_guard lock(debug_mutex_);
    const auto warning = torque_warning_;
    torque_warning_ = {};
    return warning;
}
MechanismWarning Runner::takeMechanismWarning() {
    std::lock_guard lock(debug_mutex_);
    auto warning = std::move(mechanism_warning_);
    mechanism_warning_ = {};
    return warning;
}
void Runner::controlLoop() {
    if (!configureThread(config_.control_cpu, config_.control_priority)) {
        fault_.store(Fault::Scheduling); // No string formatting/logging on this thread.
    }
    Feedback f;
    Target t;
    Velocity velocity;
    Period period(config_.control_hz);
    uint64_t previous_begin = 0;
    uint32_t control_fault_context=0;
    while (running_.load(std::memory_order_relaxed)) {
        const auto begin = monotonicNow();
        const auto now = receiveControlInputs(feedback_, target_, velocities, f, t, velocity);
        const auto external_fault = fault_.load(std::memory_order_acquire);
        if (external_fault != Fault::None)
            control_.fault(external_fault);
        // Validate feedback/current state before accepting a transition.
        control_.step(f, t, now);
        std::array<Completion, 8> replies{};
        std::array<ModeRecord,8> request_evidence{};
        std::size_t reply_count = 0;
        Request request;
        auto change = [&](Request r) {
            const auto before=control_.status().mode;
            const auto result=control_.request(r,f,now);
            request_evidence[reply_count]={r.id,now,control_.status().session,before,r.mode,
                                            result.mode,result.rejection,result.success};
            auto &record=request_evidence[reply_count];
            record.origin=r.origin;
            if(!result.success) record.reason=
                result.rejection!=EntryRejection::None?unsigned(result.rejection):
                control_.status().mode==Mode::Fault?6:
                r.deadline<now?5:config_.control.still_only?10:
                isPolicy(r.mode) && !config_.policies.count(r.mode)?4:
                before==Mode::Ready && !control_.status().ready?7:1;
            return result;
        };
        if (still_requests.pop(request)) {
            replies[reply_count++] = change(request);
            // Already queued non-stop requests must not undo the stop.
            for (std::size_t i = 0; i < 7 && requests.pop(request); ++i) {
                request_evidence[reply_count]={request.id,now,control_.status().session,
                    control_.status().mode,request.mode,control_.status().mode,EntryRejection::None,false,8,request.origin};
                replies[reply_count++] = {request.id, false, control_.status().mode};
            }
            velocity = {};
        } else if (requests.pop(request)) {
            replies[reply_count++] = change(request);
            velocity = {};
        }
        auto cmd = control_.step(f, t, now);
        std::array<MotionCompletion, 8> motion_replies{};
        std::size_t motion_reply_count = 0;
        MotionRequest motion_request;
        if (motion_stops.pop(motion_request)) {
            motion_replies[motion_reply_count++] = motion_.request(motion_request, cmd, now);
            for (std::size_t i = 0; i < 7 && motion_requests.pop(motion_request); ++i)
                motion_replies[motion_reply_count++] = {motion_request.id, 3};
        } else if (motion_requests.pop(motion_request)) {
            motion_replies[motion_reply_count++] = motion_.request(motion_request, cmd, now);
        }
        if (!config_.motion.motions.empty()) {
            motion_.step(cmd, t, now);
            // Validate the final overlaid command, preserving all original protections.
            control_.validateCommand(cmd);
        }
        if (!hand_.step(cmd, f, now)) {
            control_.fault(Fault::Hand);
            control_.validateCommand(cmd);
            cmd.hand_enabled = false;
        }
        if (control_.status().mode == Mode::Fault) motion_.cancel();
        // ROS admits at most 14 pending requests; the completion queue holds 31.
        // Publish after all validation, so READY with invalid hands cannot report success.
        for (std::size_t i=0;i<reply_count;++i) {
            const auto result=control_.complete(replies[i]);
            if(result.id) completions.push(result);
            auto r=request_evidence[i];r.actual=result.mode;
            if(r.success && !result.success) r.reason=9;
            r.success=result.success;
            if(config_.runtime.flight_recorder.enabled && !mode_records_.push(r)) ++recording_drops_;
        }
        for (std::size_t i = 0; i < motion_reply_count; ++i) {
            if (control_.status().mode == Mode::Fault) motion_replies[i].error = 3;
            motion_completions.push(motion_replies[i]);
        }
        auto s = control_.status();
        s.upper_motion_cmd = motion_.cmd();
        s.upper_motion_phase = static_cast<uint8_t>(motion_.phase());
        s.upper_motion_seconds = motion_.elapsed();
        s.upper_motion_recovery_delayed = motion_.recoveryDelayed(now);
        perception_active_ = s.mode == Mode::Perception;
        expireVelocity(velocity, now);
        Observation obs{f, velocity, s.session, isPolicy(s.mode), s.mode};
        obs.entry_yaw_valid = s.entry_yaw_valid;
        obs.entry_yaw = s.entry_yaw;
        obs.motion_mask = motion_.observationMask();
        obs.motion_generation = motion_.generation();
        obs.motion_observation_position = config_.motion.observation_position;
        if(config_.runtime.flight_recorder.enabled) {
            captureControlEvidence(cmd,s,f,t,motion_.generation());
            if(s.fault!=Fault::None && !control_fault_context) {
                control_fault_context=failure_context_.load(std::memory_order_acquire);
                if(!control_fault_context && external_fault==Fault::None)
                    control_fault_context=(unsigned(FailureOrigin::Control)<<16)|
                        unsigned(failureReason(s.fault,s.feedback_fault));
                if(!control_fault_context && s.fault==Fault::Scheduling)
                    control_fault_context=(unsigned(FailureOrigin::Scheduling)<<16)|unsigned(FailureReason::Scheduling);
            }
            cmd.evidence.fault_origin=control_fault_context>>16;
            cmd.evidence.fault_reason=control_fault_context&0xffff;
            cmd.evidence.control_us=(monotonicNow()-begin)/1000;
            cmd.evidence.interval_ns=previous_begin ? begin-previous_begin : 0;
            cmd.evidence.overruns=overruns_.load(std::memory_order_relaxed);
        }
        observation_.push(obs);
        command_.push(cmd);
        status_.push(s);
        if (s.mode == Mode::Fault) {
            Fault expected = Fault::None;
            fault_.compare_exchange_strong(expected, s.fault, std::memory_order_acq_rel);
        }
        const auto execution = monotonicNow() - begin;
        control_us_.store(execution / 1000, std::memory_order_relaxed);
        const auto skipped = period.wait();
        if (skipped)
            overruns_.fetch_add(1, std::memory_order_relaxed);
        if (timings_ && !timings_->push({begin, execution, previous_begin ? begin - previous_begin : 0,
                                      skipped, s.mode, s.fault}))
            timing_drops_.fetch_add(1, std::memory_order_relaxed);
        previous_begin = begin;
    }
}
void Runner::inferenceLoop() {
    if (!configureThread(config_.inference_cpu, 0)) {
        fail(Fault::Scheduling, "Inference affinity setup failed",FailureOrigin::Scheduling);
        return;
    }
    Period period(config_.inference_hz);
    Observation obs;
    loco::DepthStack depth;
    uint64_t session = 0, sequence = 0;
    Policy* active = nullptr;
    try {
        while (running_) {
            observation_.latest(obs);
            depth_.latest(depth);
            obs.depth = &depth;
            auto* selected = obs.enabled && fault_ == Fault::None ? policies_.at(obs.mode).get() : nullptr;
            if (selected != active || (selected && obs.session != session)) {
                if (active) active->reset(); // Discard queued/in-flight visual work on exit.
                active = selected;
                if (active) active->reset();
                session = obs.session;
                // One non-RT worker; each policy retains its trained sampling period.
                period.reset(active ? 1.0 / config_.policies.at(obs.mode).step_seconds
                                    : config_.inference_hz);
                sequence = 0;
                encoder_us_ = actor_us_ = feature_stamp_ = 0;
                encoder_calls_ = actor_calls_ = feature_age_at_action_ns_ = inference_calls_ = 0;
            }
            if (active && obs.feedback.sequence != sequence) {
                sequence = obs.feedback.sequence;
                const auto begin = monotonicNow();
                if (begin >= obs.feedback.stamp &&
                    begin - obs.feedback.stamp <= config_.control.feedback_timeout_ns) {
                    if (obs.mode == Mode::Perception && !config_.external_depth.topic.empty()) {
                        depth = external_depth_.sample(begin, obs.feedback.stamp);
                        depth_stamp_ = depth.stamp;
                    }
                    expireVelocity(obs.velocity, begin);
                    auto result = active->step(obs);
                    result.motion_generation = obs.motion_generation;
                    result.inference_sequence = obs.feedback.sequence;
                    const auto stats = active->inferenceStats();
                    encoder_us_ = stats.encoder_us;
                    actor_us_ = stats.actor_us;
                    feature_stamp_ = stats.feature_stamp;
                    encoder_calls_ = stats.encoder_calls;
                    actor_calls_ = stats.actor_calls;
                    feature_age_at_action_ns_ = stats.feature_age_at_action_ns;
                    for (std::size_t i = 0; i < 3; ++i)
                        policy_velocity_[i] = obs.velocity.value[i];
                    result.evidence.timing=stats;
                    result.evidence.sequence=inference_calls_.load()+1;
                    result.evidence.duration_us=(monotonicNow()-begin)/1000;
                    target_.push(result);
                    ++inference_calls_;
                    infer_us_.store((monotonicNow() - begin) / 1000);
                }
            }
            period.wait();
        }
    } catch (const std::exception &e) {
        fail(Fault::Inference, e.what(),FailureOrigin::Inference,obs.mode,obs.session,obs.feedback.stamp);
    } catch (...) {
        fail(Fault::Inference, "Unknown inference failure",FailureOrigin::Inference,obs.mode,obs.session,obs.feedback.stamp);
    }
    if (active) active->reset();
}
void Runner::ioLoop() {
    Period period(config_.io_hz);
    loco::State raw;
    Feedback f;
    Command cmd;
    uint64_t sequence = 0, received = 0, last_command = monotonicNow();
    uint64_t next_depth = 0, next_record=0;
    uint64_t last_warning=0;
    uint8_t last_mechanism=0;
    Mode recorded_mode = Mode::Fault;
    bool enabled = false;
    loco::Command motor(config_.control.count);
    Fault operation = Fault::Runtime;
    try {
        while (running_) {
            const auto now = monotonicNow();
            TorqueWarning frame_warning;
            uint8_t frame_mechanism=0;
            drainModeRecords();
            if (fault_ != Fault::None) {
                runtime_.protectiveStop(); // Inhibit first; the event is an application protection stop.
                command_.latest(cmd); // Include the final control/fault context before leaving IO.
                cmd.evidence.fault=fault_.load();
                if(cmd.mode!=Mode::Fault) cmd.evidence.fault_mode=cmd.mode;
                if(!cmd.evidence.fault_time) cmd.evidence.fault_time=now;
                if(config_.runtime.flight_recorder.enabled) {
                    auto event=controllerPacket(config_,cmd,f,mapping_.evidence(),frame_warning,
                                                loco::flight::Event::ApplicationFault);
                    const auto context=cmd.evidence.fault_origin ?
                        (cmd.evidence.fault_origin<<16)|cmd.evidence.fault_reason : failure_context_.load(std::memory_order_acquire);
                    if(context) {uint32_t cause[]={context>>16,context&0xffff};event.integers(530,cause,2);}
                    runtime_.recordApplication(event);
                }
                break;
            }
            if (runtime_.status().mode == loco::Mode::Estopped)
                throw std::runtime_error(runtime_.status().message);
            if (runtime_.read(raw)) {
                if (raw.sequence != sequence) {
                    sequence = raw.sequence;
                    received = monotonicNow();
                }
                operation = Fault::Feedback;
                mapping_.updateFeedback(raw);
                operation = Fault::Runtime;
                f = feedback(raw, received, config_.base_from_imu);
                feedback_.push(f);
            }
            if (config_.runtime.depth.enabled && now >= next_depth) {
                loco::DepthStack depth;
                // A snapshot writer may be busy; keep the last acquisition until its deadline.
                if (runtime_.readDepth(depth)) {
                    depth_stamp_ = depth.stamp;
                    depth_.push(depth);
                    if (!config_.depth_image_topic.empty()) {
                        std::lock_guard lock(depth_debug_mutex_);
                        depth_debug_ = depth;
                    }
                }
                next_depth = now + uint64_t(1e9 / config_.runtime.depth.frequency_hz);
            }
            if (command_.latest(cmd)) {
                if (monotonicNow() < cmd.stamp ||
                    monotonicNow() - cmd.stamp > config_.command_timeout * 1e9)
                    throw std::runtime_error("Stale control command");
                last_command = cmd.stamp;
                if (cmd.mode == Mode::Fault) {
                    runtime_.protectiveStop();
                    auto event=controllerPacket(config_,cmd,f,mapping_.evidence(),frame_warning,
                                                loco::flight::Event::ApplicationFault);
                    if(cmd.evidence.fault_origin) {
                        uint32_t cause[]={cmd.evidence.fault_origin,cmd.evidence.fault_reason};event.integers(530,cause,2);
                    }
                    runtime_.recordApplication(event);
                    break;
                }
                if (cmd.mode == Mode::Still) {
                    if (!runtime_.setDamping(true))
                        throw std::runtime_error(runtime_.status().message);
                    enabled = false;
                    motor = loco::Command(config_.control.count);
                    motor.kd = runtime_.config().motor_damping_kd;
                } else {
                    operation = Fault::Runtime;
                    TorqueWarning warning;
                    MechanismWarning mechanism;
                    motor = mapping_.command(cmd, f, &warning, &mechanism);
                    motor.hand_enabled = cmd.hand_enabled;
                    for (std::size_t i = 0; i < 12; ++i) motor.hand_position[i] = cmd.hand[i] * 100;
                    motor.hand_kp = config_.hand.kp;
                    motor.hand_kd = config_.hand.kd;
                    frame_warning=warning; frame_mechanism=mechanism.mask;
                    if (warning.mask) {
                        std::lock_guard lock(debug_mutex_);
                        torque_warning_.mask |= warning.mask;
                        for (std::size_t i = 0; i < config_.control.count; ++i)
                            if (std::abs(warning.requested[i]) >
                                std::abs(torque_warning_.requested[i]))
                                torque_warning_.requested[i] = warning.requested[i];
                    }
                    operation = Fault::Runtime;
                    // Mapping may be slow; recheck age/fault after it, before publishing.
                    if (fault_ != Fault::None ||
                        monotonicNow() - cmd.stamp > config_.command_timeout * 1e9)
                        throw std::runtime_error("Command expired during mapping");
                    if (!enabled) {
                        if (!runtime_.setDamping(false))
                            throw std::runtime_error(runtime_.status().message);
                        enabled = true;
                    }
                    if (!runtime_.write(motor))
                        throw std::runtime_error(runtime_.status().message);
                    if (mechanism.mask) {
                        std::lock_guard lock(debug_mutex_);
                        mechanism_warning_.mask |= mechanism.mask;
                        for (std::size_t i = 0; i < mechanism.detail.size(); ++i)
                            if (mechanism.mask & (uint8_t{1} << i))
                                mechanism_warning_.detail[i] = std::move(mechanism.detail[i]);
                    }
                    if (suspended_ && config_.release_after_ready && cmd.ready_complete &&
                        fault_ == Fault::None &&
                        monotonicNow() - cmd.stamp <= config_.command_timeout * 1e9) {
                        if (!runtime_.releaseSuspension())
                            throw std::runtime_error(runtime_.status().message);
                        suspended_ = false;
                    }
                }
            } else if (now - last_command > config_.command_timeout * 1e9)
                throw std::runtime_error("Control worker stopped producing commands");
            {
                std::lock_guard lock(debug_mutex_);
                debug_ = {raw, f, cmd, motor};
            }
            if (config_.runtime.flight_recorder.enabled) {
                const bool changed=cmd.mode!=recorded_mode;
                const bool warning_changed=frame_warning.mask!=last_warning || frame_mechanism!=last_mechanism;
                if(changed || warning_changed || now>=next_record) {
                    const auto event=changed ? loco::flight::Event::ModeChanged :
                        warning_changed ? loco::flight::Event::Warning : loco::flight::Event::None;
                    auto record=controllerPacket(config_,cmd,f,mapping_.evidence(),frame_warning,event);
                    uint32_t mechanism[]={frame_mechanism};record.integers(517,mechanism,1);
                    const auto drop_count=recording_drops_.load();
                    uint32_t drops[]={uint32_t(drop_count),uint32_t(drop_count>>32)};
                    record.integers(535,drops,2);
                    runtime_.recordApplication(record);
                    next_record=now+uint64_t(1e9/config_.runtime.flight_recorder.application_hz);
                }
                recorded_mode=cmd.mode;last_warning=frame_warning.mask;last_mechanism=frame_mechanism;
            }
            period.wait();
        }
    } catch (const std::exception &e) {
        fail(operation, e.what(),FailureOrigin::Io,cmd.mode,cmd.session,cmd.stamp);
        runtime_.protectiveStop();
    } catch (...) {
        fail(Fault::Runtime, "Unknown SDK worker failure",FailureOrigin::Io,cmd.mode,cmd.session,cmd.stamp);
        runtime_.protectiveStop();
    }
}
} // namespace eir::ctrl
