#include <cmath>
#include <eir_ctrl/config.hpp>
#include <eir_ctrl/mapping.hpp>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <random>
#include <unistd.h>
using namespace eir::ctrl;
namespace {
class Capture final : public Inference {
  public:
    std::vector<float> input;
    std::vector<float> run(const std::vector<float> &value) override {
        input = value;
        return std::vector<float>(29, .5);
    }
};
struct MotionFile {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
                                 ("eir-motion-test-" + std::to_string(getpid()) + ".csv");
    ~MotionFile() {
        std::filesystem::remove(path);
    }
    void write() {
        std::ofstream file(path);
        // Identity orientation, torso neutral, distinguish q from dq and neck data.
        for (int row = 0; row < 3; ++row) {
            file << "1,0,0,0";
            for (int joint = 0; joint < 32; ++joint)
                file << ',' << (joint >= 12 && joint <= 14 ? 0 : .01 * (joint + row));
            for (int joint = 0; joint < 32; ++joint)
                file << ',' << .1 * (joint + row);
            file << '\n';
        }
    }
};
} // namespace
TEST(Mimic, TermMajorHistoryTimingNeckAndReset) {
    auto c = loadConfig(PRO_CONFIG).policies.at(Mode::MimicA);
    MotionFile file;
    file.write();
    c.mimic.motion = file.path;
    c.mimic.first_frame_hold_seconds = .06; // exactly three observations
    c.mimic.warmup_steps = 2;
    auto inference = std::make_unique<Capture>();
    auto *capture = inference.get();
    auto policy = makePolicy(c, std::move(inference));
    Observation o;
    o.session = 9;
    o.feedback.stamp = 123;
    o.feedback.q = c.hold;
    for (std::size_t j = 0; j < c.indices.size(); ++j)
        o.feedback.q[c.indices[j]] = c.defaults[j];
    o.feedback.dq.fill(2);
    o.feedback.omega = {1, 2, 3};
    auto first = policy->step(o);
    const auto initial = capture->input;
    ASSERT_EQ(initial.size(), 664U);
    for (int age = 0; age < 4; ++age) {
        for (int joint = 0; joint < 29; ++joint) {
            EXPECT_NEAR(initial[age * 58 + joint], joint >= 12 && joint <= 14 ? 0 : .01 * joint,
                        1e-6);
            EXPECT_EQ(initial[age * 58 + 29 + joint], 0);  // frozen dq_ref, with 32-column stride
            EXPECT_EQ(initial[316 + age * 29 + joint], 0); // q-default
            const bool ankle = joint == 4 || joint == 5 || joint == 10 || joint == 11;
            EXPECT_EQ(initial[432 + age * 29 + joint], ankle ? 0 : 2);
            EXPECT_EQ(initial[548 + age * 29 + joint], 0);
        }
        const std::array<float, 6> identity{1, 0, 0, 1, 0, 0};
        for (int i = 0; i < 6; ++i)
            EXPECT_EQ(initial[232 + age * 6 + i], identity[i]);
        for (int group = 0; group < 4; ++group) {
            EXPECT_NEAR(initial[256 + group * 12 + age * 3], 0, 1e-6);
            EXPECT_NEAR(initial[256 + group * 12 + age * 3 + 1], 0, 1e-6);
            EXPECT_NEAR(initial[256 + group * 12 + age * 3 + 2], -1, 1e-6);
        }
        for (int i = 0; i < 3; ++i)
            EXPECT_EQ(initial[304 + age * 3 + i], i + 1);
    }
    EXPECT_EQ(first.motion_frame, 0U);
    EXPECT_EQ(first.motion_frames, 3U);
    EXPECT_EQ(first.source_stamp, 123U);
    EXPECT_EQ(first.session, 9U);
    EXPECT_NEAR(first.q[29], .29, 1e-9);
    EXPECT_EQ(first.dq[29], 0);
    for (std::size_t j = 0; j < 29; ++j)
        EXPECT_NEAR(first.q[c.indices[j]], c.defaults[j] + .5 * c.scales[j], 1e-9);
    policy->step(o);
    policy->step(o); // final frozen first frame, last_action now enabled
    EXPECT_EQ(capture->input.back(), .5);
    auto fourth = policy->step(o);
    EXPECT_NEAR(capture->input[3 * 58], .01, 1e-6); // frame 1
    EXPECT_NEAR(capture->input[3 * 58 + 29], .1, 1e-6);
    EXPECT_NEAR(fourth.q[29], .30, 1e-9);
    EXPECT_NEAR(fourth.dq[29], 3, 1e-9);
    policy->step(o);
    auto end = policy->step(o);
    EXPECT_NEAR(end.q[29], .31, 1e-9); // final frame remains held
    policy->reset();
    policy->step(o);
    EXPECT_EQ(capture->input, initial);
    c.mimic.first_frame_hold_seconds = 0;
    c.mimic.loop = true;
    auto recorder = std::make_unique<Capture>();
    capture = recorder.get();
    policy = makePolicy(c, std::move(recorder));
    policy->step(o);
    policy->step(o);
    policy->step(o);
    policy->step(o);
    policy->step(o);
    EXPECT_NEAR(capture->input[3 * 58], 0, 1e-6); // wrapped to frame zero
}
TEST(Mimic, RejectsMalformedAndNonfiniteMotionAtStartup) {
    auto c = loadConfig(PRO_CONFIG).policies.at(Mode::MimicA);
    MotionFile file;
    c.mimic.motion = file.path;
    for (const auto &text : {"1,0,0,0\n", "nan,0,0,0\n", "1junk,0,0,0\n", ""}) {
        {
            std::ofstream stream(file.path);
            stream << text;
        }
        EXPECT_THROW(makePolicy(c, std::make_unique<Capture>()), std::runtime_error);
    }
    file.write();
    {
        std::ofstream stream(file.path, std::ios::app);
        stream << "0,0,0,0";
        for (int i = 0; i < 64; ++i)
            stream << ",0";
        stream << '\n';
    }
    EXPECT_THROW(makePolicy(c, std::make_unique<Capture>()), std::runtime_error);
}
TEST(Mimic, RelativeHeadingAndTorsoGravity) {
    auto c = loadConfig(PRO_CONFIG).policies.at(Mode::MimicA);
    MotionFile file;
    file.write();
    c.mimic.motion = file.path;
    auto recorder = std::make_unique<Capture>();
    auto *capture = recorder.get();
    auto policy = makePolicy(c, std::move(recorder));
    Observation o;
    const auto half = std::sqrt(.5);
    o.feedback.orientation = {half, 0, 0, half}; // 90 deg world yaw, aligned at entry
    o.feedback.q[13] = M_PI / 2; // torso roll; yaw must not change projected gravity
    policy->step(o);
    EXPECT_NEAR(capture->input[232], 1, 1e-6);
    EXPECT_NEAR(capture->input[235], 1, 1e-6);
    EXPECT_NEAR(capture->input[281], -1, 1e-6);
    EXPECT_NEAR(capture->input[282], 0, 1e-6);
    o.feedback.orientation = {1, 0, 0, 0}; // current heading changed relative to entry
    policy->step(o);
    EXPECT_NEAR(capture->input[232 + 18], 0, 1e-6);
    EXPECT_NEAR(capture->input[233 + 18], -1, 1e-6);
    EXPECT_NEAR(capture->input[234 + 18], 1, 1e-6);
}
TEST(ControlMimic, DirectPolicySwitchEnabledModesAndStaleTargets) {
    const auto sport = (std::filesystem::path(SPORT_CONFIG).parent_path() / "sport_mimic.yaml").string();
    for (const auto &profile : {std::string(PRO_CONFIG), sport}) {
        SCOPED_TRACE(profile);
        auto c = loadConfig(profile);
        Control control(c.control);
        Feedback f;
        f.valid = true;
        f.gravity = {0, 0, -1};
        f.q = c.control.ready;
        f.stamp = 1000000000;
        auto request = [&](Mode mode) {
            return control.request({1, f.stamp + 100, mode}, f, f.stamp).success;
        };
        auto ready = [&] {
            ASSERT_TRUE(request(Mode::Still));
            ASSERT_TRUE(request(Mode::Ready));
            f.stamp += uint64_t(c.control.ready_seconds * 1e9);
            control.step(f, {}, f.stamp);
            f.stamp += uint64_t(c.control.settle_seconds * 1e9) + 1;
            control.step(f, {}, f.stamp);
            ASSERT_TRUE(control.status().ready);
        };
        EXPECT_FALSE(request(Mode::MimicA));
        ready();
        ASSERT_TRUE(request(Mode::Walk));
        EXPECT_TRUE(request(Mode::MimicA));
        Target old;
        old.valid = true;
        old.q = c.control.ready;
        old.session = control.status().session;
        ready();
        ASSERT_TRUE(request(Mode::MimicA));
        old.source_stamp = f.stamp;
        EXPECT_FALSE(control.step(f, old, f.stamp).policy_target);
        old.session = control.status().session;
        EXPECT_TRUE(control.step(f, old, f.stamp).policy_target);
        EXPECT_TRUE(request(Mode::MimicB));
        EXPECT_TRUE(request(Mode::Still));
        c.control.policy_mask &= ~(uint64_t{1} << 31);
        Control disabled(c.control);
        EXPECT_FALSE(disabled.request({1, f.stamp + 100, Mode::MimicB}, f, f.stamp).success);
    }
}
#if EIR_CTRL_WITH_TORCH && EIR_CTRL_WITH_ONNX
TEST(Mimic, AllFiveBackendsAgreeAcrossTrajectoryAndReset) {
    const auto sport = (std::filesystem::path(SPORT_CONFIG).parent_path() / "sport_mimic.yaml").string();
    for (const auto &profile : {std::string(PRO_CONFIG),
            (std::filesystem::path(PRO_CONFIG).parent_path() / "pro_mimic_gripper.yaml").string(), sport}) {
        SCOPED_TRACE(profile);
        const auto config = loadConfig(profile);
        for (int mode = 30; mode <= 34; ++mode) {
            SCOPED_TRACE(mode);
            auto c = config.policies.at(static_cast<Mode>(mode));
            const auto model = std::filesystem::path(c.inference.model);
            c.inference.backend = "onnx";
            c.inference.model = std::filesystem::path(model).replace_extension(".onnx").string();
            auto onnx = makePolicy(c);
            c.inference.backend = "libtorch";
            c.inference.model = std::filesystem::path(model).replace_extension(".pt").string();
            auto torch = makePolicy(c);
            Observation obs;
            obs.feedback.q = config.control.ready;
            // Includes hold, warmup, history shifts, and PRO-only neck trajectories.
            EXPECT_EQ(c.inference.input_size, config.robot == "pro" ? 664U : 544U);
            EXPECT_EQ(c.indices.size(), config.robot == "pro" ? 29U : 23U);
            EXPECT_EQ(c.mimic.tracked_indices.size(), config.robot == "pro" ? 3U : 0U);
            for (int step = 0; step < 160; ++step) {
                obs.feedback.stamp = 1000 + step;
                obs.session = 42;
                obs.feedback.omega = {.01 * std::sin(step), .02, 0};
                const auto a = onnx->step(obs), b = torch->step(obs);
                for (std::size_t j = 0; j < config.control.count; ++j) {
                    EXPECT_NEAR(a.q[j], b.q[j], 2e-4);
                    EXPECT_DOUBLE_EQ(a.dq[j], b.dq[j]);
                }
            }
            onnx->reset();
            torch->reset();
            const auto a = onnx->step(obs), b = torch->step(obs);
            for (std::size_t j = 0; j < config.control.count; ++j)
                EXPECT_NEAR(a.q[j], b.q[j], 2e-4);
        }
    }
}
#endif
TEST(Mimic, InstancesDoNotChangeAmpHistoryOrGains) {
    auto config = loadConfig(PRO_CONFIG);
#if !EIR_CTRL_WITH_ONNX
    for (auto &[mode, profile] : config.policies) {
        profile.inference.backend = "libtorch";
        profile.inference.model =
            std::filesystem::path(profile.inference.model).replace_extension(".pt").string();
    }
#endif
    const auto &amp_config = config.policies.at(Mode::Walk);
    auto amp = makePolicy(amp_config), reference = makePolicy(amp_config);
    auto mimic = makePolicy(config.policies.at(Mode::MimicA));
    Observation observation;
    observation.feedback.q = config.control.ready;
    observation.feedback.gravity = {0, 0, -1};
    for (int step = 0; step < 12; ++step) {
        observation.feedback.omega = {.001 * step, 0, 0};
        mimic->step(observation);
        if (step % 3 == 0)
            mimic->reset();
        const auto actual = amp->step(observation), expected = reference->step(observation);
        EXPECT_EQ(actual.q, expected.q);
    }
    Mapping mapping(config);
    Feedback f;
    f.q = config.control.ready;
    Command cmd;
    cmd.q = f.q;
    cmd.q[3] += .01; // projected knee torque, with profile-specific motor damping
    cmd.policy_target = true;
    for (const auto mode : {Mode::Walk, Mode::MimicA, Mode::MimicB, Mode::Walk}) {
        cmd.mode = mode;
        const auto motor = mapping.command(cmd, f);
        const auto &profile = config.policies.at(mode);
        EXPECT_NEAR(motor.torque[3], profile.kp[3] * .01, 1e-9);
        EXPECT_DOUBLE_EQ(motor.kd[3], profile.motor_kd[3]);
        EXPECT_DOUBLE_EQ(motor.kp[18], profile.kp[18]);
    }
}
TEST(Mimic, NeckVelocityMatchesPositionMappingDerivative) {
    const auto c = loadConfig(PRO_CONFIG);
    Mapping mapping(c);
    Feedback f;
    f.q = c.control.ready;
    Command cmd;
    cmd.mode = Mode::MimicA;
    cmd.policy_target = true;
    cmd.q = f.q;
    cmd.q[30] = .02;
    cmd.q[31] = .04;
    cmd.dq[29] = .2;
    cmd.dq[30] = .05;
    cmd.dq[31] = -.08;
    const auto output = mapping.command(cmd, f);
    constexpr double dt = 1e-5;
    auto plus = cmd, minus = cmd;
    for (std::size_t i = 29; i < 32; ++i) {
        plus.q[i] += dt * cmd.dq[i];
        minus.q[i] -= dt * cmd.dq[i];
    }
    const auto a = mapping.command(plus, f), b = mapping.command(minus, f);
    for (std::size_t i = 29; i < 32; ++i)
        EXPECT_NEAR(output.velocity[i], (a.position[i] - b.position[i]) / (2 * dt), 1e-4);
}
TEST(ControlMimic, DirectSwitchBlendsFromMeasuredAndRejectsOldSession) {
    const auto c = loadConfig(PRO_CONFIG);
    Control control(c.control);
    Feedback f;
    f.valid = true;
    f.gravity = {0, 0, -1};
    f.q = c.control.ready;
    f.stamp = 1000000000;
    ASSERT_TRUE(control.request({1, f.stamp + 100, Mode::Ready}, f, f.stamp).success);
    f.stamp += uint64_t(c.control.ready_seconds * 1e9);
    control.step(f, {}, f.stamp);
    f.stamp += uint64_t(c.control.settle_seconds * 1e9) + 1;
    control.step(f, {}, f.stamp);
    ASSERT_TRUE(control.request({2, f.stamp + 100, Mode::Walk}, f, f.stamp).success);
    Target old;
    old.valid = true;
    old.session = control.status().session;
    old.source_stamp = f.stamp;
    old.q = f.q;
    ASSERT_TRUE(control.step(f, old, f.stamp).policy_target);
    ASSERT_TRUE(control.request({3, f.stamp + 100, Mode::MimicA}, f, f.stamp).success);
    EXPECT_FALSE(control.step(f, old, f.stamp).policy_target);
    auto target = old;
    target.session = control.status().session;
    target.q[0] += .2;
    EXPECT_DOUBLE_EQ(control.step(f, target, f.stamp).q[0], f.q[0]);
    f.stamp += uint64_t(c.control.policy_transition_seconds * 1e9 / 2);
    target.source_stamp = f.stamp;
    EXPECT_NEAR(control.step(f, target, f.stamp).q[0], f.q[0] + .1, 1e-9);
    f.stamp += uint64_t(c.control.policy_transition_seconds * 1e9 / 2);
    target.source_stamp = f.stamp;
    EXPECT_NEAR(control.step(f, target, f.stamp).q[0], target.q[0], 1e-9);
    const auto session = control.status().session;
    ASSERT_TRUE(control.request({4, f.stamp + 100, Mode::MimicA}, f, f.stamp).success);
    EXPECT_GT(control.status().session, session); // explicit restart resets the motion
    EXPECT_FALSE(control.step(f, target, f.stamp).policy_target);
    ASSERT_TRUE(control.request({5, f.stamp + 100, Mode::Still}, f, f.stamp).success);
}

TEST(ControlMimic, ReadyRequiresStillAndPolicyRejectionKeepsSessionAndOutput) {
    const auto c = loadConfig(PRO_CONFIG);
    Control control(c.control);
    Feedback f;
    f.valid = true; f.gravity = {0, 0, -1}; f.q = c.control.ready; f.stamp = 1000000000;
    auto request = [&](Mode mode) { return control.request({1, f.stamp + 100, mode}, f, f.stamp).success; };
    ASSERT_TRUE(request(Mode::Ready));
    f.stamp += uint64_t(c.control.ready_seconds * 1e9);
    control.step(f, {}, f.stamp);
    f.stamp += uint64_t(c.control.settle_seconds * 1e9) + 1;
    control.step(f, {}, f.stamp);
    for (auto mode : {Mode::Walk, Mode::MimicA, Mode::MimicB, Mode::MimicC, Mode::MimicD, Mode::MimicE}) {
        ASSERT_TRUE(request(mode));
        Target target;
        target.valid = true; target.q = f.q;
        target.session = control.status().session; target.source_stamp = f.stamp;
        ASSERT_TRUE(control.step(f, target, f.stamp).policy_target);
        EXPECT_FALSE(request(Mode::Ready));
        EXPECT_EQ(control.status().mode, mode);
        EXPECT_EQ(control.status().session, target.session);
        EXPECT_TRUE(control.step(f, target, f.stamp).policy_target);
    }
    ASSERT_TRUE(request(Mode::Still));
    EXPECT_FALSE(request(Mode::Walk));
    EXPECT_TRUE(request(Mode::Ready));
}

namespace {
constexpr std::array<Mode, 6> special_modes{Mode::LiedownSupine, Mode::LiedownProne,
    Mode::GetupSupine, Mode::GetupProne, Mode::StandToSquat, Mode::SquatToStand};
std::array<double, 4> worldYaw(const std::array<double, 4> &q, double angle) {
    const double c = std::cos(angle/2), s = std::sin(angle/2);
    return {c*q[0]-s*q[3], c*q[1]-s*q[2], c*q[2]+s*q[1], c*q[3]+s*q[0]};
}
}
TEST(Mimic, ProjectedHeadingHandlesLyingPosesAndRejectsUnsafeEntry) {
    const auto config = loadConfig(PRO_SPECIAL_CONFIG);
    for (auto mode : {Mode::LiedownSupine, Mode::LiedownProne, Mode::GetupSupine, Mode::GetupProne}) {
        SCOPED_TRACE(int(mode));
        const auto &h = config.policies.at(mode).mimic.heading;
        Feedback f;
        std::array<double, 4> yaw;
        for (double angle : {-2.8, -1., 0., 1.5, 3.}) {
            f.orientation = worldYaw(h.reference, angle);
            ASSERT_TRUE(alignMimicHeading(h, f, yaw));
            EXPECT_NEAR(yaw[0], std::cos(angle/2), 1e-9);
            EXPECT_NEAR(yaw[3], -std::sin(angle/2), 1e-9);
        }
        f.orientation = h.reference;
        f.omega[0] = h.max_angular_velocity + .001;
        EXPECT_FALSE(alignMimicHeading(h, f, yaw));
        f.omega = {};
        f.orientation = {};
        EXPECT_FALSE(alignMimicHeading(h, f, yaw));
        f.orientation[0] = std::numeric_limits<double>::quiet_NaN();
        EXPECT_FALSE(alignMimicHeading(h, f, yaw));
        f.orientation.fill(1e308);
        EXPECT_FALSE(alignMimicHeading(h, f, yaw));
    }
    auto h = config.policies.at(Mode::GetupSupine).mimic.heading;
    Feedback standing;
    std::array<double, 4> yaw;
    EXPECT_FALSE(alignMimicHeading(h, standing, yaw));
    h.reference = {1, 0, 0, 0};
    standing.orientation = {std::sqrt(.5), 0, std::sqrt(.5), 0};
    EXPECT_FALSE(alignMimicHeading(h, standing, yaw)); // Selected x axis is vertical.
}
TEST(ControlMimic, SpecialEntryRequiresSourceCompletionAndPostureWithoutMutatingOnRejection) {
    const auto c = loadConfig(PRO_SPECIAL_CONFIG);
    for (const auto pair : {std::pair{Mode::LiedownSupine, Mode::GetupSupine},
                            std::pair{Mode::LiedownProne, Mode::GetupProne}}) {
        Control control(c.control);
        Feedback f;
        f.valid = true; f.gravity = {0, 0, -1}; f.q = c.control.ready; f.stamp = 1000000000;
        auto request = [&](Mode m) { return control.request({1, f.stamp + 100, m}, f, f.stamp); };
        for (auto mode : special_modes) EXPECT_FALSE(request(mode).success);
        ASSERT_TRUE(request(Mode::Ready).success);
        f.stamp += uint64_t(c.control.ready_seconds * 1e9);
        control.step(f, {}, f.stamp);
        f.stamp += uint64_t(c.control.settle_seconds * 1e9) + 1;
        control.step(f, {}, f.stamp);
        ASSERT_TRUE(control.status().ready);
        EXPECT_EQ(request(pair.second).rejection, EntryRejection::SourceMode);
        ASSERT_TRUE(request(Mode::Walk).success);
        EXPECT_FALSE(request(Mode::Ready).success);
        const auto &entry = c.control.entries[unsigned(pair.first)].heading;
        if (entry.projected) f.orientation = entry.reference;
        ASSERT_TRUE(request(pair.first).success);
        const auto session = control.status().session;
        EXPECT_EQ(request(pair.second).rejection, EntryRejection::MotionIncomplete);
        EXPECT_EQ(control.status().session, session);
        Target t;
        t.valid = true; t.q = f.q; t.session = session; t.source_stamp = f.stamp;
        t.motion_frames = 10; t.motion_frame = 8;
        control.step(f, t, f.stamp);
        EXPECT_EQ(request(pair.second).rejection, EntryRejection::MotionIncomplete);
        t.motion_frame = 9;
        control.step(f, t, f.stamp);
        const auto &h = c.control.entries[unsigned(pair.second)].heading;
        if (h.projected) {
            EXPECT_EQ(request(pair.second).rejection, EntryRejection::Posture);
            EXPECT_EQ(control.status().session, session);
            EXPECT_EQ(control.status().mode, pair.first);
            f.orientation = worldYaw(h.reference, 1.2);
        }
        ASSERT_TRUE(request(pair.second).success);
        EXPECT_EQ(control.status().session, session + 1);
        EXPECT_FALSE(control.step(f, t, f.stamp).policy_target); // Previous policy target is stale.
        EXPECT_EQ(control.status().entry_yaw_valid, h.projected);
        ASSERT_TRUE(request(Mode::Still).success);
        EXPECT_FALSE(request(pair.second).success);
    }
}
TEST(Mimic, UsesAcceptedHeadingSnapshotWithoutRecheckingLaterAngularVelocity) {
    const auto c = loadConfig(PRO_SPECIAL_CONFIG).policies.at(Mode::GetupProne);
    auto policy = makePolicy(c, std::make_unique<Capture>());
    Observation o;
    o.feedback.orientation = c.mimic.heading.reference;
    o.feedback.q = c.hold;
    o.entry_yaw_valid = alignMimicHeading(c.mimic.heading, o.feedback, o.entry_yaw);
    ASSERT_TRUE(o.entry_yaw_valid);
    o.feedback.omega[0] = 1; // The request was already accepted with a quiet base.
    EXPECT_NO_THROW(policy->step(o));
    policy->reset();
    o.entry_yaw_valid = false;
    EXPECT_THROW(policy->step(o), std::runtime_error);
}
TEST(Mapping, SpecialHybridProDampingIsMappedOnceAndSessionIsReset) {
    auto c = loadConfig(PRO_SPECIAL_CONFIG);
    Mapping mapping(c);
    Feedback f;
    f.q = c.control.ready;
    Command cmd;
    cmd.q = f.q; cmd.policy_target = true;
    for (auto mode : {Mode::LiedownSupine, Mode::LiedownProne, Mode::GetupSupine, Mode::GetupProne}) {
        SCOPED_TRACE(int(mode));
        cmd.mode = mode; ++cmd.session;
        MechanismWarning warning;
        f.dq.fill(0);
        const auto rest = mapping.command(cmd, f, nullptr, &warning);
        ASSERT_EQ(warning.mask, 0);
        for (auto j : {4, 5, 10, 11}) f.dq[j] = .7;
        const auto moving = mapping.command(cmd, f, nullptr, &warning);
        ASSERT_EQ(warning.mask, 0);
        for (auto j : {4, 5, 10, 11}) {
            EXPECT_DOUBLE_EQ(rest.torque[j], moving.torque[j]); // No virtual ankle Kd.
            EXPECT_EQ(moving.kp[j], 0);
            EXPECT_GE(moving.kd[j], 0);
            EXPECT_TRUE(c.policies.at(mode).motor_only_damping_mask & (uint64_t{1} << j));
        }
        Mapping fresh(c);
        ++cmd.session;
        const auto reset = mapping.command(cmd, f);
        const auto expected = fresh.command(cmd, f);
        EXPECT_EQ(reset.kd, expected.kd);
    }
}
TEST(Mimic, SpecialModelsKeepMaskedAnkleVelocityOutOfEveryHistoryFrame) {
    const auto config = loadConfig(PRO_SPECIAL_CONFIG);
    for (auto mode : special_modes) {
        SCOPED_TRACE(int(mode));
        const auto &c = config.policies.at(mode);
        auto inference = std::make_unique<Capture>();
        auto *capture = inference.get();
        auto policy = makePolicy(c, std::move(inference));
        Observation o;
        o.feedback.q = c.hold;
        o.feedback.orientation = c.mimic.heading.reference;
        // All six bundled normalizers were trained with zero ankle velocity.
        // Check real profile configuration, including seeding and history shifts.
        for (int step = 0; step < 8; ++step) {
            for (std::size_t i = 0; i < c.indices.size(); ++i)
                o.feedback.dq[c.indices[i]] = (step + 1) * (i + 1) * .01;
            policy->step(o);
            for (int age = 0; age < 4; ++age) {
                for (auto joint : {4, 5, 10, 11})
                    EXPECT_EQ(capture->input[432 + age * 29 + joint], 0);
                EXPECT_GT(capture->input[432 + age * 29], 0); // Hip velocity remains available.
            }
        }
    }
}
#if EIR_CTRL_WITH_TORCH && EIR_CTRL_WITH_ONNX
TEST(Mimic, SixSpecialBackendsAgreeThroughFullTrajectoryFinalHoldAndReset) {
    const auto config = loadConfig(PRO_SPECIAL_CONFIG);
    for (auto mode : special_modes) {
        SCOPED_TRACE(int(mode));
        auto c = config.policies.at(mode);
        const auto folder = std::filesystem::path(c.inference.model).parent_path();
        c.inference.backend = "onnx"; c.inference.model = (folder / "policy.onnx").string();
        auto onnx = makePolicy(c);
        c.inference.backend = "libtorch"; c.inference.model = (folder / "policy.pt").string();
        auto torch = makePolicy(c);
        Observation o;
        o.session = 100;
        o.feedback.q = c.hold;
        for (std::size_t j = 0; j < c.indices.size(); ++j) o.feedback.q[c.indices[j]] = c.defaults[j];
        o.feedback.orientation = c.mimic.heading.reference;
        const auto initial = onnx->step(o);
        auto b = torch->step(o);
        const int steps = initial.motion_frames + int(c.mimic.first_frame_hold_seconds / c.step_seconds) + 5;
        for (int i = 1; i < steps; ++i) {
            o.feedback.stamp = i;
            o.feedback.omega = {.001 * std::sin(i), .002, 0};
            auto a = onnx->step(o); b = torch->step(o);
            ASSERT_EQ(a.motion_frame, b.motion_frame);
            for (std::size_t j = 0; j < 32; ++j) {
                ASSERT_NEAR(a.q[j], b.q[j], 3e-4) << "step=" << i << " joint=" << j;
                ASSERT_DOUBLE_EQ(a.dq[j], b.dq[j]);
            }
        }
        EXPECT_EQ(b.motion_frame + 1, b.motion_frames);
        onnx->reset(); torch->reset(); o.feedback.omega = {};
        const auto a = onnx->step(o); b = torch->step(o);
        EXPECT_EQ(a.motion_frame, 0U);
        for (std::size_t j = 0; j < 32; ++j) {
            EXPECT_NEAR(a.q[j], b.q[j], 3e-4);
            EXPECT_EQ(a.q[j], initial.q[j]);
        }
    }
}
#endif
extern void beginAllocationTracking();
extern std::size_t endAllocationTracking();
TEST(ControlMimic, ProjectedEntryAndRejectPathsDoNotAllocate) {
    auto c = loadConfig(PRO_SPECIAL_CONFIG);
    Control control(c.control);
    Feedback f;
    f.valid=true; f.gravity={0,0,-1}; f.q=c.control.ready; f.stamp=1000000000;
    control.request({1, f.stamp+100, Mode::Ready}, f, f.stamp);
    f.stamp+=uint64_t(c.control.ready_seconds*1e9);
    control.step(f,{},f.stamp);
    f.stamp+=uint64_t(c.control.settle_seconds*1e9)+1;
    control.step(f,{},f.stamp);
    ASSERT_TRUE(control.status().ready);
    f.orientation=c.policies.at(Mode::LiedownSupine).mimic.heading.reference;
    beginAllocationTracking();
    const auto accepted=control.request({2,f.stamp+100,Mode::LiedownSupine},f,f.stamp);
    const auto rejected=control.request({3,f.stamp+100,Mode::GetupSupine},f,f.stamp);
    const auto allocations=endAllocationTracking();
    EXPECT_TRUE(accepted.success);
    EXPECT_FALSE(rejected.success);
    EXPECT_EQ(allocations,0U);
}
