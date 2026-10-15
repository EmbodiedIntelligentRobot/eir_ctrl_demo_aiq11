#include <eir_ctrl/runner.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <thread>

using namespace eir::ctrl;
using namespace std::chrono_literals;
namespace {
// Never creates a hardware API. All 32 SDK slots and both encoders are injected.
class HardwareFake final : public eir::loco::Backend {
    std::atomic<bool> running_{false};
    std::thread worker_;
    std::mutex mutex_;
public:
    RobotState initial{};
    std::vector<ImuState> startup_imu;
    std::chrono::milliseconds imu_delay{0};
    std::atomic<bool> paused{false}, inhibited{false}, stopped{false}, invalid_encoder{false};
    std::atomic<bool> invalid_imu{false}, imu_ready{false}, hands_offline{false};
    std::atomic<unsigned> writes{0}, hand_writes{0}, enabled_before_imu{0};
    RobotCmd accepted{};
    ~HardwareFake() override { stop(); }
    void start(const eir::loco::Config &c, StateCallback cb) override {
        EXPECT_EQ(c.backend, eir::loco::BackendType::Hardware);
        EXPECT_EQ(c.joint_names, c.motor_names);
        EXPECT_TRUE(c.encoder_names.empty());
        running_ = true;
        const auto imu_at = std::chrono::steady_clock::now() + imu_delay;
        auto first = initial;
        if (imu_delay.count()) first.imu_state = startup_imu;
        imu_ready = !imu_delay.count();
        cb(first);
        worker_ = std::thread([this, cb, imu_at] {
            while (running_) {
                if (!paused) {
                    auto sample = initial;
                    imu_ready = std::chrono::steady_clock::now() >= imu_at;
                    if (!imu_ready) sample.imu_state = startup_imu;
                    if (invalid_imu) sample.imu_state.clear();
                    if (hands_offline) { sample.hand_states.code = {0, 0}; sample.hand_states.q.assign(12, 0); }
                    if (invalid_encoder) sample.encoder_states.status[0] = 2;
                    cb(sample);
                }
                std::this_thread::sleep_for(2ms);
            }
        });
    }
    bool write(const RobotCmd &cmd) override {
        if (inhibited) return false;
        std::lock_guard lock(mutex_);
        if (inhibited) return false;
        accepted = cmd;
        if (!imu_ready && std::any_of(cmd.motor_cmds.Kp.begin(), cmd.motor_cmds.Kp.end(),
                                    [](double kp) { return kp != 0; }))
            ++enabled_before_imu;
        ++writes;
        return true;
    }
    bool writeHand(const EECmd &) override { ++hand_writes; return !inhibited; }
    bool estop() override { inhibited = true; return true; }
    void stop() noexcept override {
        running_ = false;
        if (worker_.joinable()) worker_.join();
        stopped = true;
    }
    RobotCmd command() { std::lock_guard lock(mutex_); return accepted; }
};

class ProHardware : public testing::Test {
protected:
    std::filesystem::path directory;
    YAML::Node root, sdk;
    void SetUp() override {
        auto pattern = (std::filesystem::temp_directory_path() / "eir-pro-hardware-XXXXXX").string();
        const char *created = mkdtemp(pattern.data());
        ASSERT_NE(created, nullptr);
        directory = created;
        const auto base = std::filesystem::path(PRO_HARDWARE_CONFIG).parent_path();
        root = YAML::LoadFile(PRO_HARDWARE_CONFIG);
        root["flight_recorder"] = YAML::LoadFile((base / "flight_recorder.yaml").string());
        root["flight_recorder"]["directory"] = (directory / "records").string();
#if !EIR_CTRL_WITH_ONNX
        root["inference"]["backend"] = "libtorch";
#endif
        root["robot"] = (base / root["robot"].as<std::string>()).string();
        for (auto entry : root["policies"])
            entry.second = (base / entry.second.as<std::string>()).lexically_normal().string();
        sdk = YAML::LoadFile(PRO_SDK_CONFIG);
        sdk["humanoid_hardware_node"]["RobotVersion"] = "aiq11-pro";
        sdk["humanoid_hardware_node"]["Versions"]["aiq11-pro"]["EndEffectorMapping"]["type"] = "LEGACY";
        root["runtime"]["hardware_config"] = (directory / "sdk.yaml").string();
    }
    void TearDown() override { if (!directory.empty()) std::filesystem::remove_all(directory); }
    Config load() {
        std::ofstream(directory / "sdk.yaml") << sdk;
        std::ofstream(directory / "app.yaml") << root;
        return loadConfig((directory / "app.yaml").string());
    }
    // The SDK API order is left leg, right leg, waist, left arm, right arm, neck.
    // It matches PRO motor_names; do not seed through the mapping under test.
    RobotState sample(const Config &c) {
        Mapping mapping(c);
        Feedback f; f.valid = true; f.q = c.control.ready;
        Command ready; ready.mode = Mode::Ready; ready.session = 1; ready.ready_blend = 1;
        ready.q = c.control.ready; ready.kp = c.control.ready_kp; ready.kd = c.control.ready_kd;
        const auto motors = mapping.command(ready, f);
        RobotState raw{};
        auto &m = raw.motor_states;
        m.q.resize(32); m.dq.resize(32); m.tau.resize(32);
        m.status_code.assign(32, 1); m.error_code.assign(32, 0);
        m.q = motors.position;
        raw.encoder_states.q = {c.control.ready[14], c.control.ready[13]};
        raw.encoder_states.dq = {0, 0}; raw.encoder_states.status = {0, 0};
        raw.imu_state.resize(1);
        for (int i = 0; i < 4; ++i) raw.imu_state[0].quat[i] = i == 0 ? 1 : 0;
        raw.hand_states.q.assign(12, 0); raw.hand_states.code = {1, 1};
        return raw;
    }
    template<class F> bool wait(F condition, std::chrono::milliseconds timeout = 2000ms) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline) {
            if (condition()) return true;
            std::this_thread::sleep_for(2ms);
        }
        return condition();
    }
};

TEST_F(ProHardware, ExplicitProfileLoadsWithoutSimulationModelOrFingerResources) {
    auto c = load();
    EXPECT_EQ(c.runtime.backend, eir::loco::BackendType::Hardware);
    EXPECT_EQ(c.runtime.hardware_robot, "aiq11-pro");
    EXPECT_TRUE(c.runtime.model_path.empty());
    EXPECT_TRUE(c.runtime.finger_hold_position.empty());
    EXPECT_TRUE(c.runtime.finger_coupling_csv.empty());
    EXPECT_FALSE(c.release_after_ready);
    EXPECT_FALSE(c.runtime.suspend);
    EXPECT_FALSE(c.hand.enabled);
    EXPECT_FALSE(c.runtime.hand_control);
    EXPECT_EQ(c.runtime.motor_names, std::vector<std::string>({
        "l_leg_1", "l_leg_2", "l_leg_3", "l_leg_4", "l_leg_41", "l_leg_43",
        "r_leg_1", "r_leg_2", "r_leg_3", "r_leg_4", "r_leg_41", "r_leg_43",
        "torso_1", "torso_31", "torso_33",
        "l_arm_1", "l_arm_2", "l_arm_3", "l_arm_4", "l_arm_5", "l_arm_51", "l_arm_7",
        "r_arm_1", "r_arm_2", "r_arm_3", "r_arm_4", "r_arm_5", "r_arm_51", "r_arm_7",
        "neck_1", "neck_11", "neck_13"}));
    EXPECT_EQ(c.runtime.motor_slots, std::vector<std::size_t>({0,1,2,3,4,5,6,7,8,9,10,11,
        12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}));
    EXPECT_EQ(c.runtime.encoder_names, std::vector<std::string>({"torso_3", "torso_2"}));
    eir::loco::Runtime runtime(runtimeConfig(c), std::make_unique<HardwareFake>());
}
TEST_F(ProHardware, HardwareHandOverrideIsExplicitAndPreservesLegacyProfiles) {
    root["runtime"].remove("hand_control");
    EXPECT_TRUE(load().hand.enabled); // Existing configurations retain their hand behavior.
    root["runtime"]["hand_control"] = true;
    auto c = load();
    EXPECT_TRUE(c.hand.enabled);
    EXPECT_TRUE(c.runtime.hand_control);
    root["runtime"]["hand_control"] = "invalid";
    EXPECT_THROW(load(), YAML::Exception);
    root["runtime"]["hand_control"] = false;
    sdk["humanoid_hardware_node"]["Versions"]["aiq11-pro"].remove("EndEffectorMapping");
    EXPECT_THROW(load(), std::invalid_argument); // Auto detection needs a declared protocol.
    root.remove("mimic_variant"); root.remove("mimic_detection");
    c = load();
    EXPECT_FALSE(c.hand.enabled);
    EXPECT_FALSE(c.runtime.hand_control);
}
TEST_F(ProHardware, StillOnlyConfigurationPreservesDampingAndDoesNotEnablePassiveObservation) {
    const auto normal = load();
    EXPECT_FALSE(normal.control.still_only);
    root["still_only"] = true;
    const auto locked = load();
    EXPECT_TRUE(locked.control.still_only);
    EXPECT_FALSE(locked.runtime.passive_observation);
    EXPECT_EQ(locked.runtime.damping_kd, normal.runtime.damping_kd);
    EXPECT_EQ(locked.runtime.motor_damping_kd, normal.runtime.motor_damping_kd);
    root["still_only"] = "invalid";
    EXPECT_THROW(load(), YAML::Exception);
}
TEST_F(ProHardware, StillOnlyRejectsEveryOtherModeWithoutStartingHardware) {
    root["still_only"] = true;
    Control control(load().control);
    Feedback feedback{};
    for (unsigned mode = 0; mode <= 255; ++mode) {
        if (mode == static_cast<unsigned>(Mode::Still)) continue;
        const auto result = control.request({mode + 1, 100, static_cast<Mode>(mode)}, feedback, 1);
        EXPECT_FALSE(result.success) << mode;
        EXPECT_EQ(control.status().mode, Mode::Still) << mode;
    }
    EXPECT_TRUE(control.request({257, 100, Mode::Still}, feedback, 1).success);
}
TEST_F(ProHardware, StillHealthTracksFeedbackAndRetainsFirstFaultSnapshot) {
    root["still_only"] = true;
    Control control(load().control);
    Feedback feedback{};
    feedback.valid = true;
    for (uint64_t sequence = 1; sequence <= 2; ++sequence) {
        feedback.sequence = sequence;
        feedback.stamp = sequence * 100;
        EXPECT_EQ(control.step(feedback, {}, feedback.stamp).mode, Mode::Still);
        EXPECT_EQ(control.status().feedback_stamp, feedback.stamp);
        EXPECT_EQ(control.status().feedback_sequence, sequence);
    }
    feedback.valid = false;
    feedback.stamp = 300;
    feedback.sequence = 3;
    EXPECT_EQ(control.step(feedback, {}, 300).mode, Mode::Fault);
    feedback.valid = true;
    feedback.stamp = 400;
    feedback.sequence = 4;
    EXPECT_EQ(control.step(feedback, {}, 400).mode, Mode::Fault);
    EXPECT_EQ(control.status().feedback_stamp, 300U);
    EXPECT_EQ(control.status().feedback_sequence, 3U);
}
TEST_F(ProHardware, StartupAndStillUseLowDampingOnlyForDistalArmsAndNeck) {
    const auto c = load();
    auto fake = std::make_unique<HardwareFake>(); auto *wire = fake.get();
    fake->initial = sample(c);
    eir::loco::Runtime runtime(runtimeConfig(c), std::move(fake));
    const std::array<std::size_t, 9> reduced{19, 20, 21, 26, 27, 28, 29, 30, 31};
    auto check = [&] {
        const auto m = wire->command().motor_cmds;
        for (std::size_t slot = 0; slot < 32; ++slot) {
            const bool small = std::find(reduced.begin(), reduced.end(), slot) != reduced.end();
            EXPECT_DOUBLE_EQ(m.Kd[slot], small ? .5 : 5) << "SDK slot " << slot;
            EXPECT_DOUBLE_EQ(m.Kp[slot], 0);
            EXPECT_DOUBLE_EQ(m.dq[slot], 0);
            EXPECT_DOUBLE_EQ(m.tau[slot], 0);
        }
    };
    ASSERT_TRUE(runtime.start());
    check();
    ASSERT_TRUE(runtime.setDamping(false));
    ASSERT_TRUE(runtime.setDamping(true));
    check();
    runtime.stop();
}
TEST_F(ProHardware, RejectsUnknownDuplicateAndInvalidDampingOverrides) {
    const auto saved = YAML::Clone(root);
    for (const auto *bad : {"l_arm_6", "neck_2", "no_motor"}) {
        root = YAML::Clone(saved);
        root["runtime"]["damping_kd_overrides"][bad] = .5;
        EXPECT_THROW(load(), std::invalid_argument);
    }
    for (double bad : {0., -1., std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::quiet_NaN()}) {
        root = YAML::Clone(saved);
        root["runtime"]["damping_kd_overrides"]["l_arm_5"] = bad;
        EXPECT_THROW(load(), std::invalid_argument);
    }
    root = YAML::Clone(saved);
    root["runtime"]["damping_kd_overrides"] = YAML::Load("{l_arm_5: 0.5, l_arm_5: 1.0}");
    EXPECT_THROW(load(), std::invalid_argument);
    root = YAML::Clone(saved);
    root["runtime"]["damping_kd_overrides"] = YAML::Load("[0.5, 5.0]");
    EXPECT_THROW(load(), std::invalid_argument);
    root = YAML::Clone(saved);
    root["runtime"].remove("damping_kd_overrides");
    const auto c = load();
    eir::loco::Runtime runtime(runtimeConfig(c), std::make_unique<HardwareFake>());
    EXPECT_EQ(runtime.config().motor_damping_kd, std::vector<double>(32, 5));
}
TEST_F(ProHardware, JointObservationKeepsAll32MotorsPassiveAndRejectsMotionRequests) {
    auto c = load();
    enableJointObservation(c);
    EXPECT_TRUE(c.runtime.passive_observation);
    EXPECT_FALSE(c.hand.enabled);
    EXPECT_FALSE(c.runtime.hand_control);
    EXPECT_TRUE(c.motion.motions.empty());
    EXPECT_EQ(c.remote_source, "external");
    auto fake = std::make_unique<HardwareFake>(); auto *wire = fake.get();
    fake->initial = sample(c);
    Runner runner(c, std::move(fake));
    ASSERT_NO_THROW(runner.start());
    ASSERT_TRUE(wait([&] { return wire->writes > 3; }));
    for (auto mode : {Mode::Ready, Mode::Walk, Mode::MimicA}) {
        ASSERT_TRUE(runner.requests.push({1, monotonicNow() + 500000000, mode}));
        Completion reply;
        ASSERT_TRUE(wait([&] { return runner.completions.pop(reply); }));
        EXPECT_FALSE(reply.success);
        EXPECT_EQ(runner.status().mode, Mode::Still);
    }
    const auto cmd = wire->command().motor_cmds;
    const auto debug = runner.debug();
    for (std::size_t i = 0; i < 32; ++i) {
        EXPECT_DOUBLE_EQ(cmd.Kp[i], 0);
        EXPECT_DOUBLE_EQ(cmd.Kd[i], 0);
        EXPECT_DOUBLE_EQ(cmd.tau[i], 0);
        EXPECT_DOUBLE_EQ(cmd.dq[i], 0);
        EXPECT_DOUBLE_EQ(debug.motor_command.kp[i], 0);
        EXPECT_DOUBLE_EQ(debug.motor_command.kd[i], 0);
    }
    EXPECT_EQ(runner.inferenceCalls(), 0U);
    wire->paused = true;
    ASSERT_TRUE(wait([&] { return wire->inhibited.load(); }));
    EXPECT_TRUE(wire->inhibited); // Feedback timeout still locks the backend.
}
TEST_F(ProHardware, RejectsWrongRobotAndInvalidOrMissingSlotMapping) {
    const auto saved = YAML::Clone(root);
    root["runtime"]["motor_slots"]["neck_1"] = 0;
    EXPECT_THROW(load(), std::invalid_argument);
    root = YAML::Clone(saved); root["runtime"]["motor_slots"]["neck_1"] = 32;
    EXPECT_THROW(load(), std::invalid_argument);
    root = YAML::Clone(saved);
    root["runtime"]["motor_slots"]["neck_1"] = 30;
    root["runtime"]["motor_slots"]["neck_11"] = 29;
    EXPECT_THROW(load(), std::invalid_argument); // Still a permutation, but the identities are wrong.
    root = YAML::Clone(saved); root["runtime"]["motor_slots"].remove("l_leg_1");
    EXPECT_THROW(load(), std::invalid_argument);
    root = YAML::Clone(saved); root["runtime"]["encoder_names"][1] = "torso_3";
    EXPECT_THROW(load(), std::invalid_argument);
    root = YAML::Clone(saved); root["runtime"]["suspend"] = true;
    EXPECT_THROW(load(), std::invalid_argument);
    root = YAML::Clone(saved); sdk["humanoid_hardware_node"]["RobotVersion"] = "aiq11-sport";
    EXPECT_THROW(load(), std::invalid_argument);
}
TEST_F(ProHardware, RejectsInvalidSdkCalibrationAndHandTypeBeforeAnyIo) {
    root["runtime"]["hand_control"] = true;
    const auto saved = YAML::Clone(sdk);
    for (const auto *field : {"dir", "gear_ratio"}) {
        sdk = YAML::Clone(saved);
        sdk["humanoid_hardware_node"]["Versions"]["aiq11-pro"]["JointMapping"]["LeftArm"][0][field] = 0;
        EXPECT_THROW(load(), std::invalid_argument);
    }
    sdk = YAML::Clone(saved);
    sdk["humanoid_hardware_node"]["Versions"]["aiq11-pro"]["EndEffectorMapping"]["type"] = "EIRGRIPPER";
    EXPECT_THROW(load(), std::invalid_argument);
    sdk = YAML::Clone(saved);
    sdk["humanoid_hardware_node"]["Versions"]["aiq11-pro"]["JointMapping"]["Neck"][29]["type"] = "NO_Type";
    EXPECT_THROW(load(), std::invalid_argument);
    sdk = YAML::Clone(saved);
    auto profile = sdk["humanoid_hardware_node"]["Versions"]["aiq11-pro"];
    profile.remove("EndEffectorMapping");
    EXPECT_THROW(load(), std::invalid_argument); // Older robot YAML must be made explicit.
    sdk = YAML::Clone(saved);
    auto joints = sdk["humanoid_hardware_node"]["Versions"]["aiq11-pro"]["JointMapping"];
    joints["UnknownBoard"] = YAML::Clone(joints["Neck"]);
    joints.remove("Neck");
    EXPECT_THROW(load(), std::invalid_argument);
    sdk = YAML::Clone(saved);
    joints = sdk["humanoid_hardware_node"]["Versions"]["aiq11-pro"]["JointMapping"];
    joints["LeftArm"][29] = YAML::Clone(joints["Neck"][29]);
    joints["LeftArm"][29]["can_id"] = 8;
    joints["Neck"].remove(29);
    EXPECT_THROW(load(), std::invalid_argument);
}
TEST_F(ProHardware, RoutesAllSdkLogicalIndicesAndReconstructsParallelJoints) {
    const auto c = load();
    auto fake = std::make_unique<HardwareFake>(); auto *wire = fake.get();
    fake->initial = sample(c);
    for (std::size_t slot = 0; slot < 32; ++slot) {
        fake->initial.motor_states.dq[slot] = .001 * slot;
        fake->initial.motor_states.tau[slot] = slot + .5;
    }
    const auto sdk_motors = fake->initial.motor_states;
    fake->initial.encoder_states.q = {.13, -.07};
    fake->initial.encoder_states.dq = {.11, -.09};
    eir::loco::Runtime runtime(runtimeConfig(c), std::move(fake));
    ASSERT_TRUE(runtime.start()) << runtime.status().message;
    eir::loco::State state;
    ASSERT_TRUE(runtime.read(state));
    for (std::size_t i = 0; i < 32; ++i) {
        EXPECT_DOUBLE_EQ(state.motors[i].position, sdk_motors.q[i]) << c.runtime.motor_names[i];
        EXPECT_DOUBLE_EQ(state.motors[i].velocity, .001 * i) << c.runtime.motor_names[i];
        EXPECT_DOUBLE_EQ(state.motors[i].torque, i + .5) << c.runtime.motor_names[i];
    }
    Mapping mapping(c);
    ASSERT_NO_THROW(mapping.updateFeedback(state));
    // Anatomical ankle/waist/neck position must be reconstructed from motors.
    for (auto i : {4, 5, 10, 11, 13, 14, 30, 31}) {
        EXPECT_NEAR(state.joints[i].position, c.control.ready[i], .04);
        EXPECT_DOUBLE_EQ(state.joints[i].torque, 0);
    }
    eir::loco::Command command(32);
    for (std::size_t i = 0; i < 32; ++i) {
        command.position[i] = .01 * i; command.velocity[i] = -.02 * i;
        command.torque[i] = .03 * i; command.kp[i] = i + 1; command.kd[i] = i + .1;
    }
    ASSERT_TRUE(runtime.setDamping(false));
    ASSERT_TRUE(runtime.write(command));
    const auto actual = wire->command().motor_cmds;
    for (std::size_t i = 0; i < 32; ++i) {
        SCOPED_TRACE(c.runtime.motor_names[i]);
        EXPECT_DOUBLE_EQ(actual.q[i], command.position[i]);
        EXPECT_DOUBLE_EQ(actual.dq[i], command.velocity[i]);
        EXPECT_DOUBLE_EQ(actual.tau[i], command.torque[i]);
        EXPECT_DOUBLE_EQ(actual.Kp[i], command.kp[i]);
        EXPECT_DOUBLE_EQ(actual.Kd[i], command.kd[i]);
        EXPECT_EQ(actual.control_mode[i], 1);
    }
    state.sdk.encoder_states.q[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(mapping.updateFeedback(state), std::runtime_error);
    state.sdk.encoder_states.q[0] = .13;
    state.motors[4].position = 100;
    EXPECT_THROW(mapping.updateFeedback(state), std::runtime_error);
}
TEST_F(ProHardware, WaistUsesMotorPitchRollReconstructionEvenWhenPitchEncoderIsZero) {
    const auto c = load();
    struct Fixture { double pitch, roll, m33, m31, v33, v31; };
    // Fixed SDK waist IK fixtures, independent of Mapping::command and slot routing.
    // Motor input is [torso_33, torso_31]; anatomical output is [torso_3, torso_2].
    const Fixture fixtures[] = {
        {.2, -.1, .25035446546213236, -.44894699015094996,
         .14297202584560603, -.29161103663804089},
        {-.15, .08, -.1864168587892352, .35904227395913124,
         .14762275305187478, -.3224789434884312}};
    for (const auto &f : fixtures) {
        SCOPED_TRACE(f.pitch);
        auto fake = std::make_unique<HardwareFake>();
        fake->initial = sample(c);
        fake->initial.motor_states.q[14] = f.m33;
        fake->initial.motor_states.q[13] = f.m31;
        fake->initial.motor_states.dq[14] = f.v33;
        fake->initial.motor_states.dq[13] = f.v31;
        fake->initial.encoder_states.q = {0, -.14};
        fake->initial.encoder_states.dq = {0, -.09};
        eir::loco::Runtime runtime(runtimeConfig(c), std::move(fake));
        ASSERT_TRUE(runtime.start()) << runtime.status().message;
        eir::loco::State state;
        ASSERT_TRUE(runtime.read(state));
        Mapping mapping(c);
        ASSERT_NO_THROW(mapping.updateFeedback(state));
        EXPECT_NEAR(state.joints[14].position, f.pitch, 1e-5);
        EXPECT_NEAR(state.joints[13].position, f.roll, 1e-5);
        EXPECT_NEAR(state.joints[14].velocity, .12, 1e-5);
        EXPECT_NEAR(state.joints[13].velocity, -.07, 1e-5);
        EXPECT_DOUBLE_EQ(state.motors[14].position, f.m33);
        EXPECT_DOUBLE_EQ(state.motors[13].position, f.m31);
        EXPECT_DOUBLE_EQ(state.sdk.encoder_states.q[0], 0);
        EXPECT_DOUBLE_EQ(state.sdk.encoder_states.q[1], -.14);
        // Finite input whose polynomial evaluation overflows must not reuse a
        // cached valid waist pose as fresh feedback.
        state.motors[14].position = std::numeric_limits<double>::max();
        EXPECT_THROW(mapping.updateFeedback(state), std::runtime_error);
    }
}
TEST_F(ProHardware, StartupFailureImmediatelyStopsAndInhibitsBackend) {
    auto c = load();
    auto fake = std::make_unique<HardwareFake>(); auto *wire = fake.get();
    fake->initial = sample(c); fake->initial.encoder_states.status[1] = 2;
    fake->imu_delay = 10s; // A delayed IMU must not hide an encoder fault.
    Runner runner(c, std::move(fake));
    const auto begin = std::chrono::steady_clock::now();
    EXPECT_THROW(runner.start(), std::runtime_error);
    EXPECT_LT(std::chrono::steady_clock::now() - begin, 1s);
    EXPECT_TRUE(wire->inhibited);
    EXPECT_TRUE(wire->stopped); // Assert before Runner destruction.
}
TEST_F(ProHardware, WaitsForDelayedImuWithoutEnablingControl) {
    for (int kind = 0; kind < 3; ++kind) {
        SCOPED_TRACE(kind);
        auto c = load();
        auto fake = std::make_unique<HardwareFake>(); auto *wire = fake.get();
        fake->initial = sample(c);
        if (kind != 0) {
            fake->startup_imu.resize(1); // Zero quaternion is not valid feedback.
            if (kind == 2) {
                fake->startup_imu[0].quat[0] = 1;
                fake->startup_imu[0].gyro[0] = std::numeric_limits<double>::quiet_NaN();
            }
        }
        fake->imu_delay = 60ms;
        Runner runner(c, std::move(fake));
        const auto begin = std::chrono::steady_clock::now();
        ASSERT_NO_THROW(runner.start());
        EXPECT_GE(std::chrono::steady_clock::now() - begin, 60ms);
        EXPECT_TRUE(wire->imu_ready);
        EXPECT_EQ(wire->enabled_before_imu, 0U);
        EXPECT_EQ(runner.inferenceCalls(), 0U);
        ASSERT_TRUE(wait([&] { return wire->writes > 3; }));
        EXPECT_EQ(runner.status().mode, Mode::Still);
        runner.stop();
        EXPECT_TRUE(wire->inhibited);
    }
}
TEST_F(ProHardware, PersistentInvalidImuTimesOutWithDiagnosticsAndStopsBackend) {
    for (bool missing : {true, false}) {
        auto c = load(); c.runtime.startup_timeout = 80ms;
        auto fake = std::make_unique<HardwareFake>(); auto *wire = fake.get();
        fake->initial = sample(c);
        fake->imu_delay = 10s;
        if (!missing) fake->startup_imu.resize(1);
        Runner runner(c, std::move(fake));
        const auto begin = std::chrono::steady_clock::now();
        try {
            runner.start();
            FAIL() << "Accepted invalid IMU";
        } catch (const std::runtime_error &error) {
            const std::string message = error.what();
            EXPECT_NE(message.find("Timed out waiting for valid base IMU"), std::string::npos);
            EXPECT_NE(message.find(missing ? "imu_count=0" : "quat(wxyz)=[0,0,0,0]"), std::string::npos);
        }
        EXPECT_GE(std::chrono::steady_clock::now() - begin, 80ms);
        EXPECT_LT(std::chrono::steady_clock::now() - begin, 1s);
        EXPECT_EQ(wire->enabled_before_imu, 0U);
        EXPECT_TRUE(wire->inhibited);
        EXPECT_TRUE(wire->stopped);
    }
}
TEST_F(ProHardware, FeedbackLossWhileWaitingForImuDoesNotWaitForStartupDeadline) {
    auto c = load(); c.runtime.startup_timeout = 2s;
    auto fake = std::make_unique<HardwareFake>(); auto *wire = fake.get();
    fake->initial = sample(c); fake->imu_delay = 10s; fake->paused = true;
    Runner runner(c, std::move(fake));
    const auto begin = std::chrono::steady_clock::now();
    EXPECT_THROW(runner.start(), std::runtime_error);
    EXPECT_LT(std::chrono::steady_clock::now() - begin, 1s);
    EXPECT_TRUE(wire->inhibited);
    EXPECT_TRUE(wire->stopped);
}
TEST_F(ProHardware, ImuLossAfterStartupStillLatchesFault) {
    auto c = load();
    auto fake = std::make_unique<HardwareFake>(); auto *wire = fake.get();
    fake->initial = sample(c);
    Runner runner(c, std::move(fake));
    ASSERT_NO_THROW(runner.start());
    ASSERT_TRUE(wait([&] { return wire->writes > 3; }));
    wire->invalid_imu = true;
    ASSERT_TRUE(wait([&] { return runner.status().mode == Mode::Fault && wire->inhibited; }));
    wire->invalid_imu = false;
    std::this_thread::sleep_for(30ms);
    EXPECT_EQ(runner.status().mode, Mode::Fault);
    EXPECT_TRUE(wire->inhibited);
}
TEST_F(ProHardware, AbsentHandsDoNotGateReadyOrPublishHandCommands) {
    for (int kind = 0; kind < 3; ++kind) {
        SCOPED_TRACE(kind);
        auto c = load();
        c.control.ready_seconds = .03; c.control.settle_seconds = .01;
        c.control.ready_timeout = 1; c.control.ready_error = .1;
        auto fake = std::make_unique<HardwareFake>(); auto *wire = fake.get();
        fake->initial = sample(c);
        auto &hand = fake->initial.hand_states;
        if (kind == 0) hand = {};
        else if (kind == 1) hand.code = {0, 0};
        else hand.q[0] = std::numeric_limits<double>::quiet_NaN();
        Runner runner(c, std::move(fake));
        ASSERT_NO_THROW(runner.start());
        ASSERT_TRUE(runner.requests.push({1, monotonicNow() + 500000000, Mode::Ready}));
        Completion reply;
        ASSERT_TRUE(wait([&] { return runner.completions.pop(reply); }));
        EXPECT_TRUE(reply.success);
        ASSERT_TRUE(wait([&] { return runner.debug().command.ready_complete; })) << runner.error();
        EXPECT_EQ(runner.status().fault, Fault::None);
        EXPECT_FALSE(runner.debug().command.hand_enabled);
        const auto command = wire->command().motor_cmds;
        EXPECT_TRUE(std::any_of(command.Kp.begin(), command.Kp.end(), [](double kp) { return kp > 0; }));
        runner.stop();
        EXPECT_EQ(wire->hand_writes, 0U);
    }
}
TEST_F(ProHardware, EnabledHandsStillRejectMissingFeedback) {
    root["runtime"]["hand_control"] = true;
    const auto c = load();
    auto fake = std::make_unique<HardwareFake>(); auto *wire = fake.get();
    fake->initial = sample(c);
    fake->initial.hand_states = {};
    Runner runner(c, std::move(fake));
    ASSERT_NO_THROW(runner.start());
    ASSERT_TRUE(runner.requests.push({1, monotonicNow() + 500000000, Mode::Ready}));
    Completion reply;
    ASSERT_TRUE(wait([&] { return runner.completions.pop(reply); }));
    EXPECT_FALSE(reply.success);
    EXPECT_EQ(runner.status().fault, Fault::Hand);
    ASSERT_TRUE(wait([&] { return wire->inhibited.load(); }));
    for (double kp : wire->command().motor_cmds.Kp) EXPECT_DOUBLE_EQ(kp, 0);
}
TEST_F(ProHardware, RunnerWithoutHandsRunsReadyAmpMimicStillAndRetainsFeedbackFaultLatch) {
    auto c = load();
    c.control.ready_seconds = .03; c.control.settle_seconds = .01;
    c.control.ready_timeout = 1; c.control.ready_error = .1;
    auto fake = std::make_unique<HardwareFake>(); auto *wire = fake.get();
    fake->initial = sample(c);
    fake->initial.hand_states = {};
    Runner runner(c, std::move(fake));
    ASSERT_NO_THROW(runner.start());
    ASSERT_TRUE(wait([&] { return runner.status().mode == Mode::Still && wire->writes > 3; }));
    ASSERT_TRUE(runner.requests.push({1, monotonicNow() + 500000000, Mode::Ready}));
    ASSERT_TRUE(wait([&] { return runner.debug().command.ready_complete; })) << runner.error();
    EXPECT_FALSE(runner.suspended());
    for (auto mode : {Mode::Walk, Mode::MimicA}) {
        ASSERT_TRUE(runner.requests.push({2, monotonicNow() + 500000000, mode}));
        ASSERT_TRUE(wait([&] { return runner.status().mode == mode && runner.debug().command.mode == mode; })) << runner.error();
        std::this_thread::sleep_for(80ms);
        EXPECT_GE(runner.inferenceCalls(), 3U);
        ASSERT_EQ(runner.status().fault, Fault::None) << runner.error();
    }
    ASSERT_TRUE(runner.still_requests.push({3, monotonicNow() + 500000000, Mode::Still}));
    ASSERT_TRUE(wait([&] { return runner.status().mode == Mode::Still; }));
    wire->paused = true;
    ASSERT_TRUE(wait([&] { return runner.status().mode == Mode::Fault && wire->inhibited; }));
    const auto count = wire->writes.load();
    wire->paused = false;
    ASSERT_TRUE(runner.requests.push({4, monotonicNow() + 500000000, Mode::Ready}));
    std::this_thread::sleep_for(30ms);
    EXPECT_EQ(runner.status().mode, Mode::Fault);
    EXPECT_EQ(wire->writes, count);
    runner.stop();
    EXPECT_TRUE(wire->stopped);
    EXPECT_EQ(wire->hand_writes, 0U);
}
TEST_F(ProHardware, InvalidEncoderDuringRunLatchesAndNormalStopInhibits) {
    for (bool fault : {false, true}) {
        auto c = load();
        auto fake = std::make_unique<HardwareFake>(); auto *wire = fake.get();
        fake->initial = sample(c);
        Runner runner(c, std::move(fake));
        ASSERT_NO_THROW(runner.start());
        ASSERT_TRUE(wait([&] { return wire->writes > 3; }));
        if (fault) {
            wire->invalid_encoder = true;
            ASSERT_TRUE(wait([&] { return runner.status().mode == Mode::Fault && wire->inhibited; }));
        }
        runner.stop();
        EXPECT_TRUE(wire->inhibited);
        EXPECT_TRUE(wire->stopped);
    }
}

// SDK indices are independently specified: SPORT's right arm starts at 22, not 19.
const std::vector<std::size_t> sport_sdk_indices{
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,22,23,24,25};
class SportHardware : public ProHardware {
protected:
    void SetUp() override {
        ProHardware::SetUp();
        const auto recording = YAML::Clone(root["flight_recorder"]);
        const auto base = std::filesystem::path(SPORT_HARDWARE_CONFIG).parent_path();
        root = YAML::LoadFile(SPORT_HARDWARE_CONFIG);
        root["flight_recorder"] = recording;
#if !EIR_CTRL_WITH_ONNX
        root["inference"]["backend"] = "libtorch";
#endif
        root["robot"] = (base / root["robot"].as<std::string>()).string();
        for (auto entry : root["policies"])
            entry.second = (base / entry.second.as<std::string>()).lexically_normal().string();
        root["runtime"]["hardware_config"] = (directory / "sdk.yaml").string();
        sdk["humanoid_hardware_node"]["RobotVersion"] = "aiq11-sport";
    }
    RobotState sample(const Config &c) {
        Mapping mapping(c);
        Feedback f; f.valid = true; f.q = c.control.ready;
        Command ready; ready.mode = Mode::Ready; ready.session = 1; ready.ready_blend = 1;
        ready.q = f.q; ready.kp = c.control.ready_kp; ready.kd = c.control.ready_kd;
        const auto motors = mapping.command(ready, f);
        RobotState raw;
        auto &m = raw.motor_states;
        m.q.assign(32, 0); m.dq.assign(32, 0); m.tau.assign(32, 0);
        m.status_code.assign(32, 0); m.error_code.assign(32, 0);
        for (std::size_t i = 0; i < sport_sdk_indices.size(); ++i) {
            m.q[sport_sdk_indices[i]] = motors.position[i];
            m.status_code[sport_sdk_indices[i]] = 1;
        }
        raw.encoder_states.q = {c.control.ready[14], c.control.ready[13]};
        raw.encoder_states.dq = {0, 0}; raw.encoder_states.status = {0, 0};
        raw.imu_state.resize(1); raw.imu_state[0].quat[0] = 1;
        return raw; // No hands or neck device.
    }
    void checkDisabled(const RobotCmd &wire) {
        ASSERT_EQ(wire.motor_cmds.q.size(), 32U);
        const auto &m = wire.motor_cmds;
        for (std::size_t slot = 0; slot < 32; ++slot) {
            if (std::find(sport_sdk_indices.begin(), sport_sdk_indices.end(), slot) != sport_sdk_indices.end())
                continue;
            EXPECT_EQ(m.control_mode[slot], 0) << slot;
            EXPECT_EQ(m.q[slot], 0); EXPECT_EQ(m.dq[slot], 0); EXPECT_EQ(m.tau[slot], 0);
            EXPECT_EQ(m.Kp[slot], 0); EXPECT_EQ(m.Kd[slot], 0); EXPECT_EQ(m.Ki[slot], 0);
            EXPECT_TRUE(m.names[slot].empty());
        }
    }
};
TEST_F(SportHardware, ProfilePreservesIdentityAndDefaultsToStillOnly) {
    const auto c = load();
    EXPECT_EQ(c.robot, "sport"); EXPECT_EQ(c.control.count, 23U);
    EXPECT_EQ(c.runtime.hardware_robot, "aiq11-sport");
    EXPECT_EQ(c.runtime.hardware_motor_slots, 32U);
    EXPECT_EQ(c.runtime.motor_slots, sport_sdk_indices);
    EXPECT_EQ(c.runtime.encoder_names, std::vector<std::string>({"torso_3", "torso_2"}));
    EXPECT_EQ(c.pairs.size(), 3U); EXPECT_EQ(c.policies.size(), 1U);
    EXPECT_EQ(c.policies.at(Mode::Walk).inference.input_size, 800U);
    EXPECT_DOUBLE_EQ(c.policies.at(Mode::Walk).step_seconds, .01);
    EXPECT_TRUE(c.control.still_only); EXPECT_FALSE(c.runtime.passive_observation);
    EXPECT_TRUE(c.runtime.model_path.empty()); EXPECT_TRUE(c.runtime.finger_joint_names.empty());
    EXPECT_FALSE(c.hand.enabled); EXPECT_FALSE(c.runtime.hand_control);
    EXPECT_FALSE(c.release_after_ready); EXPECT_FALSE(c.remote_gui);
    Control control(c.control); Feedback f; f.valid = true; f.stamp = 1; f.q = c.control.ready;
    EXPECT_FALSE(control.request({1, 100, Mode::Ready}, f, 1).success);
}
TEST_F(SportHardware, RejectsWrongVariantActiveSlotsAndIncompleteMappings) {
    const auto r = YAML::Clone(root), h = YAML::Clone(sdk);
    for (int kind = 0; kind < 10; ++kind) {
        SCOPED_TRACE(kind); root = YAML::Clone(r); sdk = YAML::Clone(h);
        auto profile = sdk["humanoid_hardware_node"]["Versions"]["aiq11-sport"];
        switch (kind) {
        case 0: sdk["humanoid_hardware_node"]["RobotVersion"] = "aiq11-pro"; break;
        case 1: profile["ActiveMotorCount"] = 32; break;
        case 2: profile["JointMapping"]["LeftArm"][4]["type"] = "ENCOS_P2_36H_4310"; break;
        case 3: profile["JointMapping"]["LeftLeg1"][17]["type"] = "NO_Type"; break;
        case 4: root["runtime"]["motor_slots"]["r_arm_1"] = 0; break;
        case 5: root["runtime"]["motor_slots"]["r_arm_1"] = 19; break;
        case 6: profile["JointMapping"]["Neck"].remove(29); break;
        case 7: root["runtime"]["hand_control"] = true; break;
        case 8: root["runtime"]["encoder_names"][1] = "torso_3"; break;
        case 9: profile["JointMapping"]["RightArm"][7]["dir"] = 0; break;
        }
        EXPECT_THROW(load(), std::invalid_argument);
    }
}
TEST_F(SportHardware, SparseSdkRoutingKeepsUnusedSlotsDisabledAndIgnoresTheirFaults) {
    auto c = load();
    auto fake = std::make_unique<HardwareFake>(); auto *wire = fake.get();
    fake->initial = sample(c);
    for (std::size_t i = 0; i < 32; ++i) {
        if (std::find(sport_sdk_indices.begin(), sport_sdk_indices.end(), i) != sport_sdk_indices.end()) continue;
        fake->initial.motor_states.q[i] = std::numeric_limits<double>::quiet_NaN();
        fake->initial.motor_states.error_code[i] = 17;
    }
    eir::loco::Runtime runtime(runtimeConfig(c), std::move(fake));
    ASSERT_TRUE(runtime.start()) << runtime.status().message;
    checkDisabled(wire->command());
    for (auto i : sport_sdk_indices) {
        EXPECT_EQ(wire->command().motor_cmds.Kp[i], 0);
        EXPECT_EQ(wire->command().motor_cmds.Kd[i], 5);
    }
    eir::loco::State state;
    ASSERT_TRUE(runtime.read(state));
    for (std::size_t i = 0; i < 23; ++i) {
        EXPECT_DOUBLE_EQ(state.motors[i].position, wire->initial.motor_states.q[sport_sdk_indices[i]]);
    }
    ASSERT_TRUE(runtime.setDamping(false));
    eir::loco::Command cmd(23);
    for (std::size_t i = 0; i < 23; ++i) {
        cmd.position[i] = .01 * i; cmd.velocity[i] = .02 * i;
        cmd.torque[i] = .03 * i; cmd.kp[i] = i + 1; cmd.kd[i] = i + .5;
    }
    ASSERT_TRUE(runtime.write(cmd));
    const auto accepted = wire->command(); checkDisabled(accepted);
    for (std::size_t i = 0; i < 23; ++i) {
        const auto slot = sport_sdk_indices[i];
        EXPECT_DOUBLE_EQ(accepted.motor_cmds.q[slot], cmd.position[i]);
        EXPECT_DOUBLE_EQ(accepted.motor_cmds.dq[slot], cmd.velocity[i]);
        EXPECT_DOUBLE_EQ(accepted.motor_cmds.tau[slot], cmd.torque[i]);
        EXPECT_DOUBLE_EQ(accepted.motor_cmds.Kp[slot], cmd.kp[i]);
        EXPECT_DOUBLE_EQ(accepted.motor_cmds.Kd[slot], cmd.kd[i]);
    }
    runtime.stop(); EXPECT_TRUE(wire->inhibited); EXPECT_EQ(wire->hand_writes, 0U);
}
TEST_F(SportHardware, JointReconstructionUsesSparseMotorsAndRejectsInvalidEncoder) {
    const auto c = load();
    auto fake = std::make_unique<HardwareFake>(); auto *wire = fake.get();
    fake->initial = sample(c);
    for (std::size_t i = 0; i < 23; ++i) {
        fake->initial.motor_states.dq[sport_sdk_indices[i]] = .001 * i;
        fake->initial.motor_states.tau[sport_sdk_indices[i]] = .5 + i;
    }
    eir::loco::Runtime runtime(runtimeConfig(c), std::move(fake));
    ASSERT_TRUE(runtime.start()); eir::loco::State state;
    ASSERT_TRUE(runtime.read(state)); Mapping mapping(c); mapping.updateFeedback(state);
    for (std::size_t i = 0; i < 23; ++i) {
        EXPECT_DOUBLE_EQ(state.motors[i].torque, .5 + i);
        EXPECT_DOUBLE_EQ(state.motors[i].velocity, .001 * i);
        const bool pair = i == 4 || i == 5 || i == 10 || i == 11 || i == 13 || i == 14;
        if (!pair) EXPECT_DOUBLE_EQ(state.joints[i].position, state.motors[i].position);
        else EXPECT_NEAR(state.joints[i].position, c.control.ready[i], .01);
    }
    state.sdk.encoder_states.status[0] = 2;
    EXPECT_THROW(mapping.updateFeedback(state), std::runtime_error);
    runtime.stop();
}
TEST_F(SportHardware, ActiveMotorOfflineOrErrorLatchesRuntime) {
    const auto c = load();
    for (bool offline : {false, true}) {
        auto fake = std::make_unique<HardwareFake>(); fake->initial = sample(c);
        if (offline) fake->initial.motor_states.status_code[22] = 0;
        else fake->initial.motor_states.error_code[22] = 1;
        eir::loco::Runtime runtime(runtimeConfig(c), std::move(fake));
        EXPECT_FALSE(runtime.start()); EXPECT_EQ(runtime.status().mode, eir::loco::Mode::Estopped);
    }
}
TEST_F(SportHardware, RunnerReadyAmpStillAndFeedbackLossKeepProtection) {
    const auto r = YAML::Clone(root);
    for (const auto *name : {
#if EIR_CTRL_WITH_ONNX
        "onnx",
#endif
#if EIR_CTRL_WITH_TORCH
        "libtorch",
#endif
    }) {
        root = YAML::Clone(r); root["still_only"] = false; root["inference"]["backend"] = name;
        auto c = load(); c.control.ready_seconds = .03; c.control.settle_seconds = .01;
        c.control.ready_timeout = 1;
        auto fake = std::make_unique<HardwareFake>(); auto *wire = fake.get(); fake->initial = sample(c);
        Runner runner(c, std::move(fake)); ASSERT_NO_THROW(runner.start());
        auto change = [&](Mode mode, uint64_t id) {
            EXPECT_TRUE(runner.requests.push({id, monotonicNow() + 500000000, mode}));
            Completion reply;
            EXPECT_TRUE(wait([&] { return runner.completions.pop(reply); }));
            EXPECT_TRUE(reply.success) << runner.error();
        };
        EXPECT_TRUE(wait([&] { return runner.status().mode == Mode::Still && wire->writes > 3; }));
        checkDisabled(wire->command()); change(Mode::Ready, 1);
        ASSERT_TRUE(wait([&] { return runner.debug().command.ready_complete; })) << runner.error();
        change(Mode::Walk, 2);
        ASSERT_TRUE(wait([&] { return runner.inferenceCalls() >= 4; })) << runner.error();
        EXPECT_EQ(runner.status().fault, Fault::None); checkDisabled(wire->command());
        change(Mode::Still, 3); EXPECT_EQ(wire->hand_writes, 0U);
        wire->paused = true;
        ASSERT_TRUE(wait([&] { return runner.status().mode == Mode::Fault && wire->inhibited; }));
        wire->paused = false;
        EXPECT_TRUE(runner.requests.push({4, monotonicNow() + 500000000, Mode::Ready}));
        std::this_thread::sleep_for(30ms); EXPECT_EQ(runner.status().mode, Mode::Fault);
        runner.stop(); EXPECT_TRUE(wire->stopped);
    }
}
} // namespace

TEST_F(ProHardware, AutoMimicSelectsOnceFromStableOnlineOrDefaultHandFeedback) {
    root["mimic_detection"]["stable_seconds"] = .02;
    root["mimic_detection"]["timeout_seconds"] = .06;
    for (int presence : {1,0}) {
        auto c = load(); auto backend = std::make_unique<HardwareFake>(); auto *hw = backend.get();
        hw->initial = sample(c); hw->initial.hand_states.code = {int16_t(presence),int16_t(presence)};
        Runner runner(c,std::move(backend)); runner.start();
        const auto &chosen = runner.effectiveConfig();
        EXPECT_EQ(chosen.mimic_variant,presence ? "gripper" : "fist");
        EXPECT_FALSE(chosen.mimic_fallback);
        EXPECT_EQ(chosen.policies.at(Mode::MimicA).mimic_variant,chosen.mimic_variant);
        EXPECT_NE(chosen.policies.at(Mode::MimicA).inference.model.find("policy_"+chosen.mimic_variant),std::string::npos);
        EXPECT_TRUE(chosen.gripper_policies.empty());
        hw->hands_offline = true; std::this_thread::sleep_for(30ms);
        EXPECT_EQ(runner.effectiveConfig().mimic_variant,presence ? "gripper" : "fist");
        runner.stop(); EXPECT_TRUE(hw->stopped);
    }
}
TEST_F(ProHardware, AutoMimicAmbiguousOrInvalidHandsKeepGripperDanceProfilesEnabled) {
    root["mimic_detection"]["stable_seconds"] = .02;
    root["mimic_detection"]["timeout_seconds"] = .04;
    for (int scenario : {0,1,2,3}) {
        auto c = load(); auto backend = std::make_unique<HardwareFake>(); auto *hw = backend.get();
        hw->initial = sample(c);
        if (scenario==0) hw->initial.hand_states.code={1,0};
        if (scenario==1) hw->initial.hand_states.q[0]=std::numeric_limits<double>::quiet_NaN();
        if (scenario==2) {hw->initial.hand_states.code={0,0};hw->initial.hand_states.q[0]=1;}
        if (scenario==3) hw->initial.hand_states.code.clear();
        Runner runner(c,std::move(backend)); runner.start();
        const auto &chosen=runner.effectiveConfig();
        EXPECT_EQ(chosen.mimic_variant,"gripper"); EXPECT_TRUE(chosen.mimic_fallback);
        for (int mode=30;mode<=34;++mode) {
            EXPECT_TRUE(chosen.policies.count(static_cast<Mode>(mode)));
            EXPECT_TRUE(chosen.control.policy_mask&(uint64_t{1}<<mode));
        }
        std::this_thread::sleep_for(10ms);EXPECT_EQ(runner.status().fault,Fault::None);
        runner.stop();
    }
}
TEST_F(ProHardware, AutoMimicCannotInferPresenceFromStaleRepeatedFeedback) {
    root["mimic_detection"]["stable_seconds"]=.02;
    root["mimic_detection"]["timeout_seconds"]=.2;
    auto c=load();auto backend=std::make_unique<HardwareFake>();auto *hw=backend.get();
    hw->initial=sample(c);hw->paused=true;
    Runner runner(c,std::move(backend));EXPECT_THROW(runner.start(),std::runtime_error);
    EXPECT_TRUE(hw->stopped);EXPECT_TRUE(hw->inhibited);
}
TEST_F(ProHardware, AutoMimicRetainsTheSelectedMotorGainsAndModeServiceBehavior) {
    root["mimic_detection"]["stable_seconds"]=.01;
    root["mimic_detection"]["timeout_seconds"]=.1;
    auto c=load();c.control.ready_seconds=.01;c.control.settle_seconds=.01;
    auto backend=std::make_unique<HardwareFake>();backend->initial=sample(c);
    Runner runner(c,std::move(backend));runner.start();
    auto request=[&](uint64_t id,Mode mode) {
        const auto end=monotonicNow()+2000000000;
        EXPECT_TRUE(runner.requests.push({id,end,mode}));Completion result;
        while(monotonicNow()<end&&!runner.completions.pop(result))std::this_thread::sleep_for(1ms);
        EXPECT_EQ(result.id,id);return result.success;
    };
    ASSERT_TRUE(request(1,Mode::Ready));auto end=monotonicNow()+1000000000;
    while(!runner.status().ready&&monotonicNow()<end)std::this_thread::sleep_for(1ms);
    ASSERT_TRUE(runner.status().ready);ASSERT_TRUE(request(2,Mode::MimicA));
    end=monotonicNow()+1000000000;DebugSnapshot d;
    do {d=runner.debug();if(d.command.mode==Mode::MimicA&&d.command.policy_target)break;std::this_thread::sleep_for(1ms);}while(monotonicNow()<end);
    ASSERT_EQ(d.command.mode,Mode::MimicA);ASSERT_TRUE(d.command.policy_target);
    const auto &chosen=runner.effectiveConfig().policies.at(Mode::MimicA);
    EXPECT_EQ(chosen.mimic_variant,"gripper");
    EXPECT_DOUBLE_EQ(d.motor_command.kd[3],chosen.motor_kd[3]);
    ASSERT_TRUE(request(3,Mode::Still));runner.stop();
}
TEST_F(ProHardware, AutoMimicRejectsWrongProtocolOrUnknownVariantBeforeHardwareInit) {
    sdk["humanoid_hardware_node"]["Versions"]["aiq11-pro"]["EndEffectorMapping"]["type"]="EIRGRIPPER";
    EXPECT_THROW(load(),std::invalid_argument);
    root["mimic_variant"]="invalid";EXPECT_THROW(load(),std::invalid_argument);
}
