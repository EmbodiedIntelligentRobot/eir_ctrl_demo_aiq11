#include <eir_ctrl/remote_node.hpp>
#include <ehr_message/srv/locomotion_mode_switch.hpp>
#include <prime_msgs/msg/debug_joint_array.hpp>
#include <prime_msgs/msg/debug_motor_array.hpp>
#include <prime_msgs/msg/debug_runtime.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/quaternion_stamped.hpp>
#include <gtest/gtest.h>
#include <deque>
#include <set>
#include <cmath>
#include <thread>
#include <std_srvs/srv/trigger.hpp>
#include <filesystem>
#include <fstream>
#include <map>

using namespace eir::ctrl;
using namespace std::chrono_literals;
using Joint = prime_msgs::msg::DebugJointArray;
using Motor = prime_msgs::msg::DebugMotorArray;
using RuntimeMessage = prime_msgs::msg::DebugRuntime;
using Service = ehr_message::srv::LocomotionModeSwitch;
namespace {
// SDK boundary oracle: unique, fixed feedback per slot and independently recorded
// accepted wire commands. This backend never connects to hardware.
class DiagnosticBackend final : public eir::loco::Backend {
    std::atomic<bool> running_{false};
    std::thread worker_;
    std::mutex mutex_;
    std::deque<RobotCmd> accepted_;
public:
    RobotState raw{};
    std::atomic<bool> paused{false}, stopped{false}, inject_fault{false};
    std::atomic<int> sbus_mode{0}; // 0 preserves fixture raw data; 1 neutral, 2 held STILL.
    ~DiagnosticBackend() override { stop(); }
    void start(const eir::loco::Config&, StateCallback cb) override {
        running_ = true;
        cb(raw);
        worker_ = std::thread([this, cb] {
            while (running_) {
                if (!paused) {
                    auto sample = raw;
                    if (const auto mode = sbus_mode.load()) {
                        sample.remote_state.sbus_ch.assign(12, 1002);
                        if (mode == 2) {
                            sample.remote_state.sbus_ch[5] = 1722;
                            sample.remote_state.sbus_ch[6] = 1722;
                        }
                    }
                    if (inject_fault) sample.motor_states.error_code.back() = 513;
                    cb(sample);
                }
                std::this_thread::sleep_for(2ms);
            }
        });
    }
    void stop() noexcept override {
        running_ = false;
        if (worker_.joinable()) worker_.join();
    }
    bool write(const RobotCmd& command) override {
        if (stopped) return false;
        std::lock_guard lock(mutex_);
        accepted_.push_back(command);
        if (accepted_.size() > 2048) accepted_.pop_front();
        return !stopped;
    }
    bool writeHand(const EECmd&) override { return !stopped; }
    bool estop() override { stopped = true; return true; }
    bool received(const Motor& message, const std::vector<std::string>& names) {
        std::lock_guard lock(mutex_);
        for (const auto& wire : accepted_) {
            const auto& m = wire.motor_cmds;
            bool equal = m.names == names && m.q.size() == message.q_des.size();
            for (std::size_t i = 0; equal && i < m.q.size(); ++i)
                equal = float(m.q[i]) == message.q_des[i] && float(m.dq[i]) == message.v_des[i] &&
                        float(m.tau[i]) == message.tau_ff[i] && float(m.Kp[i]) == message.kp[i] &&
                        float(m.Kd[i]) == message.kd[i] && m.control_mode[i] == 1;
            if (equal) return true;
        }
        return false;
    }
};
class Diagnostics : public testing::Test {
protected:
    Config config;
    DiagnosticBackend* backend{};
    std::unique_ptr<Runner> runner;
    std::shared_ptr<rclcpp::Node> remote, observer;
    rclcpp::executors::SingleThreadedExecutor executor;
    rclcpp::Client<Service>::SharedPtr client;
    rclcpp::Subscription<Joint>::SharedPtr joints, hands;
    rclcpp::Subscription<Motor>::SharedPtr motors;
    rclcpp::Subscription<RuntimeMessage>::SharedPtr runtimes;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_states;
    rclcpp::Subscription<geometry_msgs::msg::QuaternionStamped>::SharedPtr orientations;
    std::vector<Joint> joint_messages, hand_messages;
    std::vector<Motor> motor_messages;
    std::vector<RuntimeMessage> runtime_messages;
    std::vector<sensor_msgs::msg::JointState> joint_state_messages;
    std::vector<geometry_msgs::msg::QuaternionStamped> orientation_messages;
    void SetUp() override {
        config = loadConfig(PRO_CONFIG);
        config.runtime.flight_recorder.post_seconds=0;
        config.runtime.flight_recorder.directory =
            (std::filesystem::temp_directory_path() /
             ("eir-controller-diagnostics-" + std::to_string(monotonicNow()))).string();
        config.debug["enabled"] = true;
        config.remote_gui = false;
        config.runtime.headless = true;
        config.release_after_ready = false;
        auto fake = std::make_unique<DiagnosticBackend>();
        backend = fake.get();
        const auto n = config.control.count;
        auto& raw = backend->raw;
        raw.motor_states.q.resize(2*n);
        raw.motor_states.dq.resize(2*n);
        raw.motor_states.tau.resize(2*n);
        raw.motor_states.current.resize(2*n);
        raw.motor_states.motor_T.resize(2*n);
        raw.motor_states.status_code.resize(2*n);
        raw.motor_states.error_code.resize(2*n);
        for (std::size_t i = 0; i < n; ++i) {
            raw.motor_states.q[i] = config.control.ready[i];
            raw.motor_states.dq[i] = .0001 * (i+1);
            raw.motor_states.tau[i] = -.25 * (i+1);
            raw.motor_states.q[n+i] = .002 * (i+1);
            raw.motor_states.dq[n+i] = -.003 * (i+1);
            raw.motor_states.tau[n+i] = .125 * (i+1);
            raw.motor_states.current[n+i] = .0625 * (i+1);
            raw.motor_states.motor_T[n+i] = 20 + i;
            raw.motor_states.status_code[n+i] = i == n-1 ? 300 : i+1;
        }
        // Distinct sensor mounting, non-unit wxyz input and nonzero base roll
        // catch raw IMU publication, wrong quaternion order and missing normalization.
        const auto h = std::sqrt(.5), c = std::cos(.1), s = std::sin(.1);
        config.base_from_imu = {h, 0, 0, h};
        raw.imu_state.resize(1);
        raw.imu_state[0].quat[0] = 2*c*h;
        raw.imu_state[0].quat[1] = 2*s*h;
        raw.imu_state[0].quat[2] = -2*s*h;
        raw.imu_state[0].quat[3] = 2*c*h;
        raw.hand_states.q.assign(12, 25);
        raw.hand_states.q[5] = raw.hand_states.q[11] = 0;
        raw.hand_states.tau.assign(12, .2);
        raw.hand_states.code = {1, 1};
        runner = std::make_unique<Runner>(config, std::move(fake), true);
        remote = makeRemoteNode(config, *runner);
        observer = std::make_shared<rclcpp::Node>("diagnostics_observer");
        executor.add_node(remote); executor.add_node(observer);
        client = observer->create_client<Service>(config.mode_service);
        joint_states = observer->create_subscription<sensor_msgs::msg::JointState>(
            "joint_states", rclcpp::SensorDataQoS(),
            [this](sensor_msgs::msg::JointState::ConstSharedPtr m) { joint_state_messages.push_back(*m); });
        orientations = observer->create_subscription<geometry_msgs::msg::QuaternionStamped>(
            "base_orientation", rclcpp::SensorDataQoS(),
            [this](geometry_msgs::msg::QuaternionStamped::ConstSharedPtr m) { orientation_messages.push_back(*m); });
        joints = observer->create_subscription<Joint>("debug/joint", rclcpp::SensorDataQoS(),
            [this](Joint::ConstSharedPtr m) { joint_messages.push_back(*m); });
        motors = observer->create_subscription<Motor>("debug/motor", rclcpp::SensorDataQoS(),
            [this](Motor::ConstSharedPtr m) { motor_messages.push_back(*m); });
        hands = observer->create_subscription<Joint>("debug/hand", rclcpp::SensorDataQoS(),
            [this](Joint::ConstSharedPtr m) { hand_messages.push_back(*m); });
        runtimes = observer->create_subscription<RuntimeMessage>("debug/runtime", rclcpp::SensorDataQoS(),
            [this](RuntimeMessage::ConstSharedPtr m) { runtime_messages.push_back(*m); });
        runner->start();
        spin(1.0);
        ASSERT_FALSE(joint_messages.empty()); ASSERT_FALSE(motor_messages.empty());
        ASSERT_FALSE(runtime_messages.empty()); ASSERT_FALSE(hand_messages.empty());
        ASSERT_FALSE(joint_state_messages.empty());
        ASSERT_FALSE(orientation_messages.empty());
    }
    void TearDown() override {
        if (runner) runner->stop();
        std::filesystem::remove_all(config.runtime.flight_recorder.directory);
    }
    void spin(double seconds) {
        const auto until = monotonicNow() + uint64_t(seconds * 1e9);
        do { executor.spin_some(); std::this_thread::sleep_for(1ms); } while (monotonicNow() < until);
    }
    Service::Response::SharedPtr call(int mode) {
        auto request = std::make_shared<Service::Request>(); request->cmd = mode;
        auto future = client->async_send_request(request);
        const auto until = monotonicNow() + 2000000000;
        while (future.wait_for(0s) != std::future_status::ready && monotonicNow() < until) spin(.005);
        if (future.wait_for(0s) != std::future_status::ready) throw std::runtime_error("Mode service timeout");
        return future.get();
    }
    void checkFields(bool policy) {
        const auto n = config.control.count;
        for (const auto &pose : orientation_messages) {
            EXPECT_EQ(pose.header.frame_id, "imu_world");
            EXPECT_GT(rclcpp::Time(pose.header.stamp).nanoseconds(), 0);
            EXPECT_NEAR(pose.quaternion.w, std::cos(.1), 1e-12);
            EXPECT_NEAR(pose.quaternion.x, std::sin(.1), 1e-12);
            EXPECT_NEAR(pose.quaternion.y, 0, 1e-12);
            EXPECT_NEAR(pose.quaternion.z, 0, 1e-12);
        }
        for (const auto &state : joint_state_messages) {
            EXPECT_EQ(state.name, config.runtime.joint_names);
            ASSERT_EQ(state.position.size(), n);
            ASSERT_EQ(state.velocity.size(), n);
            ASSERT_EQ(state.effort.size(), n);
            for (std::size_t i = 0; i < n; ++i) {
                // Distinct fake anatomical and motor coordinates catch accidental raw-motor publication.
                EXPECT_DOUBLE_EQ(state.position[i], config.control.ready[i]);
                EXPECT_DOUBLE_EQ(state.velocity[i], .0001 * (i + 1));
                EXPECT_DOUBLE_EQ(state.effort[i], -.25 * (i + 1));
            }
        }
        ASSERT_GE(joint_messages.size(), 5U); ASSERT_GE(motor_messages.size(), 5U);
        for (const auto& m : motor_messages) {
            ASSERT_EQ(m.motor_num, n);
            for (const auto* values : {&m.q_act,&m.v_act,&m.tau_act,&m.current,&m.temperature,
                                      &m.q_des,&m.v_des,&m.tau_des,&m.tau_ff,&m.kp,&m.kd}) ASSERT_EQ(values->size(), n);
            ASSERT_EQ(m.status_code.size(), n); ASSERT_EQ(m.error_code.size(), n);
            // Compare all five command arrays to one whole command accepted by the backend,
            // not to a second copy of the debug calculation.
            EXPECT_TRUE(backend->received(m, config.runtime.motor_names));
            for (std::size_t i = 0; i < n; ++i) {
                EXPECT_FLOAT_EQ(m.q_act[i], .002*(i+1)); EXPECT_FLOAT_EQ(m.v_act[i], -.003*(i+1));
                EXPECT_FLOAT_EQ(m.tau_act[i], .125*(i+1)); EXPECT_FLOAT_EQ(m.current[i], .0625*(i+1));
                EXPECT_FLOAT_EQ(m.temperature[i], 20+i);
                EXPECT_EQ(m.status_code[i], i == n-1 ? 255 : i+1); EXPECT_EQ(m.error_code[i], 0);
                const double estimate = double(m.kp[i])*(double(m.q_des[i])-m.q_act[i]) +
                                        double(m.kd[i])*(double(m.v_des[i])-m.v_act[i]) + m.tau_ff[i];
                EXPECT_NEAR(m.tau_des[i], estimate, 2e-5 * std::max(1., std::abs(estimate)));
            }
        }
        for (const auto& j : joint_messages) {
            ASSERT_EQ(j.joint_num, n);
            for (const auto* values : {&j.q_act,&j.v_act,&j.tau_act,&j.q_des,&j.v_des,&j.tau_des,&j.tau_des_fake})
                ASSERT_EQ(values->size(), n);
            for (std::size_t i = 0; i < n; ++i) {
                EXPECT_FLOAT_EQ(j.q_act[i], config.control.ready[i]);
                EXPECT_FLOAT_EQ(j.v_act[i], .0001*(i+1)); EXPECT_FLOAT_EQ(j.tau_act[i], -.25*(i+1));
                EXPECT_EQ(j.tau_des[i], 0);
                const auto kp = policy ? config.policies.at(Mode::Walk).kp[i] : config.control.ready_kp[i];
                const auto kd = policy ? config.policies.at(Mode::Walk).kd[i] : config.control.ready_kd[i];
                const auto estimate = kp*(double(j.q_des[i])-j.q_act[i]) + kd*(double(j.v_des[i])-j.v_act[i]);
                EXPECT_NEAR(j.tau_des_fake[i], estimate, 5e-5 * std::max(1., std::abs(estimate)));
            }
        }
        for (const auto& h : hand_messages) {
            EXPECT_EQ(h.joint_num, 12); ASSERT_EQ(h.q_act.size(), 12U);
            EXPECT_TRUE(h.v_act.empty()); EXPECT_TRUE(h.v_des.empty());
            for (std::size_t i = 0; i < 12; ++i) EXPECT_FLOAT_EQ(h.q_act[i], i == 5 || i == 11 ? 0 : .25);
        }
        for (const auto& r : runtime_messages) {
            EXPECT_TRUE(r.feedback_valid); EXPECT_GT(r.feedback_sequence, 0U);
            EXPECT_GE(r.feedback_age_ns, 0); EXPECT_LE(r.feedback_age_ns, 100000000);
            EXPECT_GT(r.feedback_stamp_ns, 0U); EXPECT_GT(r.command_stamp_ns, 0U);
            EXPECT_EQ(r.controller_fault, 0);
            EXPECT_EQ(r.prime_ctl_duration, r.ctrl_loop_duration);
            EXPECT_GE(r.ctrl_loop_duration, 0); EXPECT_GE(r.infer_time_us, 0);
            EXPECT_EQ(r.wbc_solve_duration, 0); EXPECT_EQ(r.mpc_solve_duration, 0);
        }
    }
};
TEST_F(Diagnostics, All32SlotsUnitsWireCommandsAndTorqueEstimates) {
    ASSERT_TRUE(call(2)->result);
    spin(4.0);
    ASSERT_TRUE(runner->status().ready);
    joint_messages.clear(); motor_messages.clear(); hand_messages.clear(); runtime_messages.clear();
    spin(.5); checkFields(false);
    ASSERT_TRUE(call(28)->result);
    spin(.3);
    joint_messages.clear(); motor_messages.clear(); hand_messages.clear(); runtime_messages.clear();
    spin(.8); checkFields(true);
    EXPECT_EQ(runner->status().fault, Fault::None);
    ControlTiming timing;
    std::size_t samples = 0;
    std::set<uint64_t> execution_us;
    uint64_t previous = 0;
    while (runner->popTiming(timing)) {
        EXPECT_GT(timing.start_ns, 0U); EXPECT_GT(timing.execution_ns, 0U); ++samples;
        if (previous) EXPECT_EQ(timing.interval_ns, timing.start_ns - previous);
        previous = timing.start_ns;
        execution_us.insert(timing.execution_ns / 1000);
    }
    EXPECT_GT(samples, 1000U); EXPECT_EQ(runner->timingDrops(), 0U);
    for (const auto& message : runtime_messages) {
        // Published seconds must correspond to a measured microsecond value.
        EXPECT_TRUE(execution_us.count(std::llround(message.ctrl_loop_duration * 1e6)));
        EXPECT_GT(message.infer_time_us, 0);
    }
}
TEST_F(Diagnostics, FrozenFeedbackStopsBodyAndHandTelemetryButKeepsLatchedFault) {
    backend->paused = true;
    spin(.015);
    const auto first = runner->debug().feedback;
    spin(.025);
    const auto duplicate = runner->debug().feedback;
    EXPECT_EQ(first.sequence, duplicate.sequence); EXPECT_EQ(first.stamp, duplicate.stamp);
    spin(.3); // Drain previously queued DDS data beyond the 100 ms feedback deadline.
    const auto joint_count = joint_messages.size(), motor_count = motor_messages.size();
    const auto hand_count = hand_messages.size(), runtime_count = runtime_messages.size();
    const auto joint_state_count = joint_state_messages.size();
    const auto orientation_count = orientation_messages.size();
    spin(.6);
    EXPECT_EQ(joint_messages.size(), joint_count); EXPECT_EQ(motor_messages.size(), motor_count);
    EXPECT_EQ(hand_messages.size(), hand_count); EXPECT_GT(runtime_messages.size(), runtime_count);
    EXPECT_EQ(joint_state_messages.size(), joint_state_count);
    EXPECT_EQ(orientation_messages.size(), orientation_count);
    ASSERT_FALSE(runtime_messages.empty()); EXPECT_EQ(runtime_messages.back().ctrl_name, "FAULT");
    EXPECT_FALSE(runtime_messages.back().feedback_valid);
    EXPECT_GT(runtime_messages.back().feedback_age_ns, 100000000);
    EXPECT_EQ(runtime_messages.back().feedback_sequence, duplicate.sequence);
    EXPECT_EQ(runtime_messages.back().feedback_stamp_ns, duplicate.stamp);
    EXPECT_NE(runtime_messages.back().controller_fault, 0);
    auto response = call(0);
    EXPECT_NE(response->fb_msg.find("FAULT (latched)"), std::string::npos);
    EXPECT_NE(response->fb_msg.find("Feedback timeout"), std::string::npos);
    EXPECT_TRUE(backend->stopped);
    backend->paused = false; spin(.3);
    for (int mode : {1,2,28}) EXPECT_FALSE(call(mode)->result);
    EXPECT_EQ(joint_messages.size(), joint_count); EXPECT_EQ(motor_messages.size(), motor_count);
    EXPECT_EQ(joint_state_messages.size(), joint_state_count);
    EXPECT_EQ(orientation_messages.size(), orientation_count);
}
TEST_F(Diagnostics, BackendMotorFaultIsReportedEvenWhenLastMotorDebugLooksHealthy) {
    backend->inject_fault = true;
    spin(.4);
    const auto response = call(0);
    EXPECT_NE(response->fb_msg.find("FAULT (latched)"), std::string::npos);
    EXPECT_NE(response->fb_msg.find("Motor fault or offline status"), std::string::npos);
    EXPECT_TRUE(backend->stopped);
    EXPECT_EQ(runtime_messages.back().ctrl_name, "FAULT");
    const auto count = motor_messages.size(); spin(.3);
    EXPECT_EQ(motor_messages.size(), count);
    EXPECT_FALSE(call(2)->result);
}
} // namespace
int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    rclcpp::init(argc, argv);
    const int result = RUN_ALL_TESTS();
    rclcpp::shutdown();
    return result;
}

TEST_F(Diagnostics, FlightRecordServiceQueuesSaveWithoutChangingMode) {
    auto service=observer->create_client<std_srvs::srv::Trigger>("flight_recorder/save");
    ASSERT_TRUE(service->wait_for_service(1s));
    auto future=service->async_send_request(std::make_shared<std_srvs::srv::Trigger::Request>());
    const auto deadline=monotonicNow()+2000000000;
    while(future.wait_for(0s)!=std::future_status::ready && monotonicNow()<deadline)spin(.005);
    ASSERT_EQ(future.wait_for(0s),std::future_status::ready);EXPECT_TRUE(future.get()->success);
    for(int i=0;i<100 && !runner->flightStatistics().completed;++i)spin(.01);
    EXPECT_EQ(runner->flightStatistics().completed,1u);EXPECT_EQ(runner->status().mode,Mode::Still);
}

TEST_F(Diagnostics, AcceptedManualSaveIsDrainedWhenWorkersStopImmediately) {
    ASSERT_TRUE(runner->saveFlightRecord());
    runner->stop();
    const auto statistics = runner->flightStatistics();
    EXPECT_EQ(statistics.completed + statistics.incomplete, 1u);
    EXPECT_EQ(statistics.write_failures, 0u);
    EXPECT_EQ(statistics.dropped, 0u);
    EXPECT_TRUE(backend->stopped);
    bool found = false;
    for (const auto &run : std::filesystem::directory_iterator(config.runtime.flight_recorder.directory)) {
        if (!run.is_directory()) continue;
        for (const auto &file : std::filesystem::directory_iterator(run.path())) {
            if (file.path().extension() != ".json" ||
                file.path().filename().string().rfind("incident_", 0) != 0) continue;
            std::ifstream stream(file.path());
            const std::string text((std::istreambuf_iterator<char>(stream)), {});
            EXPECT_NE(text.find("manual_save"), std::string::npos);
            EXPECT_EQ(text.find("\"estop\""), std::string::npos);
            found = true;
        }
    }
    EXPECT_TRUE(found);
    runner->stop(); // Idempotent teardown cannot create a second incident.
    EXPECT_EQ(runner->flightStatistics().completed + runner->flightStatistics().incomplete, 1u);
}

// Production ROS SBUS -> request queues -> control worker -> runtime writer.
TEST(RecorderSbus, HeldStillKeepsHistoryAndStillPreemptsWithoutUnlockingFaults) {
    namespace fs = std::filesystem;
    for (const auto *file : {PRO_CONFIG, SPORT_CONFIG}) {
        auto c = loadConfig(file);
        c.remote_source = "sbus";
        c.remote_gui = false;
        c.hand.enabled = false;
        c.runtime.hand_control = false;
        c.runtime.headless = true;
        c.release_after_ready = false;
        c.debug["enabled"] = false;
        const auto directory = fs::temp_directory_path() /
                               ("eir-fr01-sbus-" + c.robot + "-" + std::to_string(monotonicNow()));
        c.runtime.flight_recorder.directory = directory.string();
        auto fake = std::make_unique<DiagnosticBackend>();
        auto *backend = fake.get();
        const auto n = c.control.count;
        auto &raw = backend->raw;
        raw.motor_states.q.assign(2 * n, 0);
        raw.motor_states.dq.assign(2 * n, 0);
        raw.motor_states.tau.assign(2 * n, 0);
        raw.motor_states.error_code.assign(2 * n, 0);
        std::copy_n(c.control.ready.begin(), n, raw.motor_states.q.begin());
        raw.imu_state.resize(1);
        raw.imu_state[0].quat[0] = 1;
        backend->sbus_mode = 2;
        Runner runner(c, std::move(fake));
        auto remote = makeRemoteNode(c, runner);
        auto observer = std::make_shared<rclcpp::Node>("fr01_sbus_observer");
        rclcpp::executors::SingleThreadedExecutor executor;
        executor.add_node(remote);
        executor.add_node(observer);
        auto client = observer->create_client<Service>(c.mode_service);
        auto spin = [&](double seconds) {
            const auto until = monotonicNow() + uint64_t(seconds * 1e9);
            do {
                executor.spin_some();
                std::this_thread::sleep_for(1ms);
            } while (monotonicNow() < until);
        };
        auto call = [&](Mode mode) {
            auto request = std::make_shared<Service::Request>();
            request->cmd = unsigned(mode);
            auto future = client->async_send_request(request);
            const auto until = monotonicNow() + 2000000000ULL;
            while (future.wait_for(0s) != std::future_status::ready && monotonicNow() < until)
                spin(.005);
            if (future.wait_for(0s) != std::future_status::ready)
                throw std::runtime_error("FR01 service timeout");
            return future.get();
        };
        runner.start();
        EXPECT_TRUE(client->wait_for_service(1s));
        spin(12.0); // More than the complete configured ten-second history.
        EXPECT_EQ(runner.status().mode, Mode::Still);
        EXPECT_EQ(runner.status().fault, Fault::None);
        EXPECT_EQ(runner.flightStatistics().dropped, 0u);
        for (int i = 0; i < 3; ++i)
            EXPECT_TRUE(call(Mode::Still)->result);
        // SBUS must continue to request STILL rather than becoming edge-only.
        backend->sbus_mode = 1;
        spin(.1);
        EXPECT_TRUE(call(Mode::Ready)->result);
        for (unsigned i = 0;
             i < 600 && !runner.status().ready && runner.status().fault == Fault::None; ++i)
            spin(.01);
        EXPECT_TRUE(runner.status().ready);
        backend->sbus_mode = 2;
        spin(.15);
        EXPECT_EQ(runner.status().mode, Mode::Still);
        EXPECT_FALSE(call(Mode::Walk)->result); // STILL cannot skip READY; still hold persists.
        EXPECT_TRUE(runner.saveFlightRecord());
        spin(.1);
        backend->inject_fault = true;
        for (unsigned i = 0; i < 100 && runner.status().fault == Fault::None; ++i)
            spin(.01);
        EXPECT_NE(runner.status().fault, Fault::None);
        EXPECT_FALSE(call(Mode::Still)->result); // Cannot clear a latched motor fault.
        runner.stop();
        const auto stats = runner.flightStatistics();
        EXPECT_EQ(stats.dropped, 0u);
        EXPECT_EQ(stats.write_failures, 0u);
        EXPECT_GE(stats.completed + stats.incomplete, 1u);
        // Verify operation records survived as journal rows, not fake state changes.
        size_t held = 0, service_requests = 0, transitions = 0, rejected = 0, diagnostic_frames = 0;
        for (const auto &entry : fs::recursive_directory_iterator(directory)) {
            const auto name = entry.path().filename().string();
            if (name != "events.bin" && name.rfind("incident_", 0) != 0)
                continue;
            if (entry.path().extension() != ".bin")
                continue;
            std::ifstream in(entry.path(), std::ios::binary);
            in.seekg(64);
            std::array<uint8_t, 8> prefix{};
            auto read_le = [](const uint8_t *b, size_t n) {
                uint64_t v = 0;
                for (size_t i = 0; i < n; ++i)
                    v |= uint64_t(b[i]) << (8 * i);
                return v;
            };
            while (in.read(reinterpret_cast<char *>(prefix.data()), 8)) {
                const auto size = read_le(prefix.data(), 4);
                if (size < 52 || size > 12340) {
                    ADD_FAILURE() << "Invalid record length";
                    break;
                }
                std::vector<uint8_t> body(size);
                in.read(reinterpret_cast<char *>(body.data()), size);
                if (!in) {
                    ADD_FAILURE() << "Truncated record";
                    break;
                }
                if (read_le(body.data(), 2) != unsigned(eir::loco::flight::Source::Application))
                    continue;
                std::map<unsigned, std::vector<uint32_t>> fields;
                for (size_t at = 52; at + 8 <= size;) {
                    const auto tag = read_le(body.data() + at, 2),
                               kind = read_le(body.data() + at + 2, 2),
                               count = read_le(body.data() + at + 4, 4);
                    at += 8;
                    const auto bytes = count * (kind == 1 ? 8 : 4);
                    if (at + bytes > size)
                        break;
                    if (kind == 2)
                        for (size_t i = 0; i < count; ++i)
                            fields[tag].push_back(read_le(body.data() + at + i * 4, 4));
                    at += bytes;
                }
                if (name == "events.bin" && fields.count(532)) {
                    const auto &request = fields.at(532);
                    const auto origin = fields.at(534).front();
                    if (origin == 2 && request[0] == 1 && request[1] == 1 && request[2] == 1 &&
                        request[3] == 1) {
                        ++held;
                        EXPECT_EQ(read_le(body.data() + 2, 2),
                                  unsigned(eir::loco::flight::Event::None));
                    }
                    if (origin == 1)
                        ++service_requests;
                    if (read_le(body.data() + 2, 2) ==
                        unsigned(eir::loco::flight::Event::ModeChanged))
                        ++transitions;
                    if (request[3] == 0)
                        ++rejected;
                    EXPECT_TRUE(read_le(body.data() + 4, 4) & eir::loco::flight::Packet::EventOnly);
                }
                if (name != "events.bin" && fields.count(500))
                    ++diagnostic_frames;
            }
        }
        EXPECT_GT(held, 500u);
        EXPECT_GE(service_requests, 5u);
        EXPECT_GE(transitions, 2u);
        EXPECT_GE(rejected, 2u);
        EXPECT_GT(diagnostic_frames,
                  450u); // Detailed prior-ten-second diagnostics remain available.
        fs::remove_all(directory);
    }
}
