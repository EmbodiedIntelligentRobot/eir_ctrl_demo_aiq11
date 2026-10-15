#include <eir_ctrl/config.hpp>
#include <eir_ctrl/mapping.hpp>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <unistd.h>
using namespace eir::ctrl;
void beginAllocationTracking();
std::size_t endAllocationTracking();
namespace {
struct Playback : ::testing::Test {
    MotionConfig c;
    PolicyConfig p;
    Command base;
    Target target;
    uint64_t now{1000000000};
    void SetUp() override {
        c.blend_seconds = .02;
        c.recovery_warning_seconds = .1;
        c.dt_cap = .01;
        c.observation_groups = {(uint64_t{1} << 15) | (uint64_t{1} << 16)};
        c.max_velocity.fill(5);
        c.kp.fill(20);
        c.kd.fill(2);
        p.kp.fill(10);
        p.kd.fill(1);
        OfflineMotion m;
        m.cmd = 11;
        m.mask = uint64_t{1} << 15;
        m.frames = {{0, {}}, {.05, {}}, {.1, {}}};
        m.frames[1].q[15] = .2;
        m.frames[2].q[15] = .4;
        c.motions = {m};
        m.cmd = 12;
        m.mask = uint64_t{1} << 16;
        m.frames[1].q[16] = -.2;
        m.frames[2].q[16] = -.4;
        c.motions.push_back(m);
        base.mode = Mode::Walk;
        base.policy_target = true;
        base.session = 8;
        base.q[0] = .123; // Explicit leg sentinel.
        target.valid = true;
        target.session = base.session;
    }
    Command tick(MotionPlayer &player, double dt = .01) {
        now += uint64_t(dt * 1e9);
        auto command = base;
        player.step(command, target, now);
        EXPECT_DOUBLE_EQ(command.q[0], base.q[0]);
        EXPECT_DOUBLE_EQ(command.dq[0], base.dq[0]);
        EXPECT_EQ(command.gain_mask & 0xfff, 0U);
        return command;
    }
    void start(MotionPlayer &player, int cmd = 11) {
        ASSERT_EQ(player.request({1, now + 100000000, uint8_t(cmd)}, base, now).error, 0U);
    }
};
class AmpCapture final : public Inference {
  public:
    std::vector<float> input;
    std::vector<float> run(const std::vector<float> &x) override {
        input = x;
        return std::vector<float>(23, .5);
    }
};
} // namespace

TEST_F(Playback, DisabledPreservesCommandAndRejectsUnsupportedRequests) {
    c.motions.clear();
    MotionPlayer player(c, p);
    auto cmd = tick(player);
    EXPECT_EQ(cmd.q, base.q);
    EXPECT_EQ(cmd.kp, base.kp);
    EXPECT_EQ(cmd.gain_mask, 0U);
    EXPECT_EQ(player.observationMask(), 0U);
    EXPECT_EQ(player.request({1, now + 1, 11}, base, now).error, 2U);
    EXPECT_EQ(player.request({1, now + 1, 3}, base, now).error, 1U);
    EXPECT_EQ(player.request({1, now - 1, 2}, base, now).error, 3U);
    base.mode = Mode::Perception;
    EXPECT_EQ(player.request({1, now + 1, 2}, base, now).error, 3U);
}
TEST_F(Playback, FingerInterpolationIsIndependentAndStopsOnModeOrSessionChange) {
    c.motions[0].mask = 0; // hand-only motion exercises session cancellation too
    c.motions[0].hand_mask = 1;
    c.motions[0].frames[1].hand[0] = .5;
    c.motions[0].frames[2].hand[0] = 1;
    MotionPlayer player(c, p);
    start(player);
    auto cmd = tick(player);
    EXPECT_EQ(cmd.hand_mask, 1); EXPECT_DOUBLE_EQ(cmd.hand[0], 0);
    cmd = tick(player); EXPECT_NEAR(cmd.hand[0], .1, 1e-9);
    EXPECT_EQ(cmd.q, base.q); EXPECT_EQ(player.observationMask(), 0U);
    player.request({2, now + 1, 2}, cmd, now);
    cmd = tick(player); EXPECT_EQ(cmd.hand_mask, 0);
    start(player); ++base.session;
    cmd = tick(player); EXPECT_EQ(player.phase(), MotionPhase::Idle); EXPECT_EQ(cmd.hand_mask, 0);
}
TEST_F(Playback, InterpolationSpeedLimitGroupMaskAndRestartAreContinuous) {
    MotionPlayer player(c, p);
    start(player);
    EXPECT_EQ(player.observationMask(), c.observation_groups[0]);
    auto a = tick(player);
    auto b = tick(player);
    auto d = tick(player);
    EXPECT_NEAR(b.q[15], .04, 1e-10);
    EXPECT_NEAR(d.q[15], .08, 1e-10);
    EXPECT_DOUBLE_EQ(d.kp[15], 20);
    EXPECT_DOUBLE_EQ(d.dq[15], 0);
    EXPECT_LT(std::abs(b.q[15] - a.q[15]), 5 * .01 + 1e-10);
    auto generation = player.generation();
    start(player, 12);
    auto restarted = base;
    player.step(restarted, target, now); // same timestamp: no jump
    EXPECT_EQ(restarted.q, d.q);
    EXPECT_EQ(restarted.kp[15], d.kp[15]);
    EXPECT_GT(player.generation(), generation);
    auto delayed = tick(player, .08); // progress is capped after a late cycle
    EXPECT_NEAR(player.elapsed(), .01, 1e-10);
    EXPECT_LE(std::abs(delayed.q[16] - restarted.q[16]), .05 + 1e-10);
}
TEST_F(Playback, StopWaitsForDistinctUnmaskedInferencesThenReturnsToPolicy) {
    MotionPlayer player(c, p);
    start(player);
    for (int i = 0; i < 5; ++i) tick(player);
    target.motion_generation = player.generation();
    const auto generation = player.generation();
    ASSERT_EQ(player.request({2, now + 1, 2}, base, now).error, 0U);
    EXPECT_EQ(player.phase(), MotionPhase::Recovering);
    EXPECT_EQ(player.observationMask(), 0U);
    EXPECT_GT(player.generation(), generation);
    target.source_stamp = now;
    target.inference_sequence = 1;
    auto held = tick(player);
    EXPECT_EQ(player.phase(), MotionPhase::Recovering); // old masked inference ignored
    target.motion_generation = player.generation();
    auto held2 = tick(player);
    EXPECT_EQ(held.q, held2.q);
    tick(player);
    EXPECT_EQ(player.phase(), MotionPhase::Recovering); // duplicate target is not a new inference
    target.inference_sequence = 2;
    tick(player);
    EXPECT_EQ(player.phase(), MotionPhase::Returning);
    for (int i = 0; i < 10; ++i) tick(player);
    EXPECT_EQ(player.phase(), MotionPhase::Idle);
    auto done = tick(player);
    EXPECT_EQ(done.q, base.q);
    EXPECT_EQ(done.gain_mask, 0U);
}
TEST_F(Playback, DelayedRecoveryHoldsUntilFreshInferenceAndModeCancellation) {
    MotionPlayer player(c, p);
    start(player);
    for (int i = 0; i < 12; ++i) tick(player);
    EXPECT_EQ(player.phase(), MotionPhase::Recovering);
    const auto held = tick(player);
    now += 200000000;
    auto cmd = base;
    player.step(cmd, target, now);
    EXPECT_EQ(player.phase(), MotionPhase::Recovering);
    EXPECT_TRUE(player.recoveryDelayed(now));
    EXPECT_EQ(cmd.q, held.q);
    EXPECT_EQ(cmd.kp, held.kp);
    EXPECT_EQ(cmd.kd, held.kd);
    // Delayed recovery still rejects old/duplicate inference generations.
    target.source_stamp = now;
    target.inference_sequence = 1;
    tick(player);
    EXPECT_EQ(player.phase(), MotionPhase::Recovering);
    target.motion_generation = player.generation();
    tick(player);
    tick(player);
    EXPECT_EQ(player.phase(), MotionPhase::Recovering);
    target.inference_sequence = 2;
    tick(player);
    EXPECT_EQ(player.phase(), MotionPhase::Returning);
    EXPECT_FALSE(player.recoveryDelayed(now));
    for (int i = 0; i < 10; ++i) tick(player);
    EXPECT_EQ(player.phase(), MotionPhase::Idle);
    EXPECT_EQ(tick(player).q, base.q);
    start(player);
    tick(player);
    base.mode = Mode::Still;
    tick(player);
    EXPECT_EQ(player.phase(), MotionPhase::Idle);
    base.mode = Mode::Walk;
    start(player);
    tick(player);
    ++base.session;
    auto reset = tick(player);
    EXPECT_EQ(player.phase(), MotionPhase::Idle);
    EXPECT_EQ(reset.q, base.q);
}
TEST_F(Playback, RealtimePlaybackRequestsRecoveryAndQueuesDoNotAllocate) {
    MotionPlayer player(c, p);
    Channel<MotionRequest> requests;
    Channel<Command> commands;
    beginAllocationTracking();
    for (int i = 0; i < 10000; ++i) {
        now += 2000000;
        if (i % 200 == 0) requests.push({1, now + 100000000, 11});
        if (i % 200 == 25) requests.push({2, now + 100000000, 12});
        if (i % 200 == 50) requests.push({3, now + 100000000, 2});
        MotionRequest request;
        if (requests.pop(request)) player.request(request, base, now);
        target.motion_generation = player.generation();
        target.source_stamp = now;
        ++target.inference_sequence;
        auto cmd = base;
        player.step(cmd, target, now);
        commands.push(cmd);
        commands.latest(cmd);
        if (i % 200 == 150) player.cancel();
    }
    const auto allocations = endAllocationTracking();
    EXPECT_EQ(allocations, 0U);
}
TEST(MotionConfig, BuiltinResourcesAndInvalidDataAreCheckedBeforeStart) {
    const auto file = std::filesystem::path(PRO_CONFIG).parent_path() / "pro_motion.yaml";
    const auto c = loadConfig(file);
    ASSERT_EQ(c.motion.motions.size(), 4U);
    EXPECT_TRUE(loadConfig(PRO_CONFIG).motion.motions.empty());
    for (const auto &motion : c.motion.motions) {
        EXPECT_EQ(motion.mask & 0xfff, 0U);
        EXPECT_EQ(motion.mask >> 29, 0U); // Fingers have their own mask; never mapped onto the neck.
        EXPECT_EQ(motion.hand_mask, 0x7df); // 5 active slots per side, reserved 5/11 ignored.
        for (const auto& frame : motion.frames) EXPECT_EQ(frame.hand, HandValues{});
        EXPECT_EQ(motion.frames.front().time, 0);
    }
    auto path = std::filesystem::temp_directory_path() / ("eir-upper-motion-" + std::to_string(getpid()));
    std::filesystem::create_directory(path);
    struct Cleanup { std::filesystem::path p; ~Cleanup() { std::filesystem::remove_all(p); } } cleanup{path};
    auto manifest = YAML::LoadFile((file.parent_path() / "../motions/pro/config.yaml").string());
    manifest["motions"] = YAML::Load("[{cmd: 11, name: test, file: test.csv}]");
    std::ofstream(path / "config.yaml") << manifest;
    auto load = [&] { return loadMotionConfig((path / "config.yaml").string(), c.robot,
                                               c.runtime.joint_names, c.control); };
    for (auto text : {"time,l_leg_1\n0,0\n1,0\n", "time,l_arm_1\n0,0\n0,0\n",
                      "time,l_arm_1\n0,0\n1,nan\n", "time,l_arm_1\n0,0\n1,999\n",
                      "time,l_arm_1\n0,0\n1,0,0\n", "time,l_arm_1,l_arm_1\n0,0,0\n1,0,0\n"}) {
        std::ofstream(path / "test.csv") << text;
        EXPECT_THROW(load(), std::exception);
    }
    std::ofstream(path / "test.csv") << "time,neck_1\n0,0\n1,0.1\n";
    EXPECT_EQ(load().motions[0].mask, uint64_t{1} << 29);
    EXPECT_THROW(loadMotionConfig((path / "config.yaml").string(), "sport", c.runtime.joint_names,
                                 c.control), std::invalid_argument);
    for (auto text : {"time,l_fg1_1\n0,0\n1,1.01\n", "time,l_fg1_1\n0,0\n1,nan\n",
                      "time,l_fg1_1,l_fg1_1\n0,0,0\n1,0,0\n", "time,l_fg4_1\n0,0\n1,0\n"}) {
        std::ofstream(path / "test.csv") << text;
        EXPECT_THROW(load(), std::exception);
    }
    std::ofstream(path / "test.csv") << "time,l_fg1_1\n0,0\n1,1\n";
    const auto hands = load();
    EXPECT_EQ(hands.motions[0].mask, 0U); EXPECT_EQ(hands.motions[0].hand_mask, 1);
}
TEST(MotionObservation, OnlyOwnedGroupsAreSuppressedAndLegHistorySurvives) {
    auto c = loadConfig(PRO_CONFIG).policies.at(Mode::Walk);
    auto backend = std::make_unique<AmpCapture>();
    auto *capture = backend.get();
    auto policy = makePolicy(c, std::move(backend));
    Observation obs;
    obs.session = 1;
    obs.feedback.q.fill(.1);
    obs.feedback.dq.fill(2);
    policy->step(obs);
    const auto original = capture->input;
    obs.motion_mask = (uint64_t{1} << 15) | (uint64_t{1} << 16);
    obs.motion_observation_position[15] = .3;
    obs.motion_observation_position[16] = 0;
    policy->step(obs);
    const std::size_t width = 78, offset = 9 * width;
    EXPECT_NEAR(capture->input[offset + 9], original[9], 1e-6); // same leg q
    EXPECT_EQ(capture->input[offset + 9 + 23], 2); // leg dq
    EXPECT_EQ(capture->input[offset + 9 + 46], .5); // leg previous action remains
    EXPECT_NEAR(capture->input[offset + 9 + 15], 0, 1e-6);
    EXPECT_EQ(capture->input[offset + 9 + 23 + 15], 0);
    EXPECT_EQ(capture->input[offset + 9 + 46 + 15], 0);
    EXPECT_NEAR(capture->input[offset + 9 + 17], .1, 1e-6); // unmasked arm slot remains
    EXPECT_EQ(capture->input[9 + 23 + 15], 2); // no wholesale history reset
    obs.motion_mask = 0;
    policy->step(obs);
    EXPECT_EQ(capture->input[offset + 9 + 23 + 15], 2);
    EXPECT_EQ(capture->input[offset + 9 + 46 + 15], .5);
}
TEST(MotionMapping, CompositeGainsReachMotorTorqueAndRetainMotorDamping) {
    const auto c = loadConfig(PRO_CONFIG);
    Mapping mapping(c);
    Feedback f;
    f.q = c.control.ready;
    Command cmd;
    cmd.mode = Mode::Walk;
    cmd.policy_target = true;
    cmd.q = f.q;
    auto baseline = mapping.command(cmd, f);
    cmd.q[12] += .1;
    cmd.q[15] += .1;
    cmd.gain_mask = (uint64_t{1} << 12) | (uint64_t{1} << 15);
    cmd.kp[12] = 10;
    cmd.kd[12] = 2;
    cmd.kp[15] = 10;
    cmd.kd[15] = 2;
    auto overlay = mapping.command(cmd, f);
    EXPECT_NEAR(overlay.torque[12], 1, 1e-10);
    EXPECT_EQ(overlay.kd[12], c.policies.at(Mode::Walk).motor_kd[12]);
    EXPECT_EQ(overlay.kp[15], 10); // Arms retain position PD, not joint torque mode.
    EXPECT_EQ(overlay.kd[15], 2);
    for (std::size_t i = 0; i < 12; ++i) {
        EXPECT_EQ(overlay.position[i], baseline.position[i]);
        EXPECT_EQ(overlay.torque[i], baseline.torque[i]);
        EXPECT_EQ(overlay.kp[i], baseline.kp[i]);
        EXPECT_EQ(overlay.kd[i], baseline.kd[i]);
    }
}
