#include <eir_ctrl/config.hpp>
#include <eir_ctrl/mapping.hpp>
#include <gtest/gtest.h>
#include <limits>
using namespace eir::ctrl;
void beginAllocationTracking();
std::size_t endAllocationTracking();
namespace {
struct HandTest : ::testing::Test {
    HandConfig config{loadConfig(PRO_CONFIG).hand};
    HandControl hand{config};
    Feedback f;
    Command cmd;
    uint64_t now{1000000000};
    void SetUp() override { f.hand_valid = true; cmd.mode = Mode::Ready; }
    void tick(double dt = .01) {
        now += uint64_t(dt * 1e9);
        cmd.hand_mask = 0;
        ASSERT_TRUE(hand.step(cmd, f, now));
        EXPECT_EQ(cmd.hand[5], 0); EXPECT_EQ(cmd.hand[11], 0);
    }
};
}
TEST_F(HandTest, OpenMainSecondarySequenceAndPolicySwitches) {
    f.hand.fill(.4);
    tick(); EXPECT_DOUBLE_EQ(cmd.hand[0], .4); // initialize from measured feedback
    for (int i = 0; i < 50; ++i) tick();
    EXPECT_NEAR(cmd.hand[0], 0, 1e-9);
    for (int i = 0; i < 25; ++i) tick();
    EXPECT_NEAR(cmd.hand[0], .5, 1e-9); EXPECT_EQ(cmd.hand[2], 0);
    cmd.mode = Mode::Walk;
    for (int i = 0; i < 25; ++i) tick();
    EXPECT_NEAR(cmd.hand[0], 1, 1e-9); EXPECT_NEAR(cmd.hand[2], 0, 1e-9);
    cmd.mode = Mode::MimicA;
    for (int i = 0; i < 50; ++i) tick();
    EXPECT_EQ(cmd.hand, config.positions[2]);
    cmd.mode = Mode::Perception; tick();
    EXPECT_EQ(cmd.hand, config.positions[2]);
}
TEST_F(HandTest, MotionRateLimitStopAndStillFault) {
    tick();
    for (int i = 0; i < 160; ++i) tick();
    auto before = cmd.hand;
    cmd.hand_mask = 1; cmd.hand[0] = 0;
    now += 200000000; ASSERT_TRUE(hand.step(cmd, f, now));
    EXPECT_NEAR(cmd.hand[0], before[0] - .02, 1e-9); // no catch-up after delay
    EXPECT_EQ(cmd.hand[2], before[2]); // partial ownership holds the other fingers
    const auto moving = cmd.hand;
    tick();
    EXPECT_LE(std::abs(cmd.hand[0] - moving[0]), .02 + 1e-9);
    cmd.mode = Mode::Still; tick(); EXPECT_FALSE(cmd.hand_enabled);
    const auto held = cmd.hand;
    for (int i = 0; i < 100; ++i) tick();
    EXPECT_EQ(cmd.hand, held);
    cmd.mode = Mode::Fault; tick(); EXPECT_FALSE(cmd.hand_enabled);
    cmd.mode = Mode::Ready; f.hand.fill(.2); tick();
    EXPECT_LE(std::abs(cmd.hand[0] - .2), .02 + 1e-9);
}
TEST_F(HandTest, InvalidFeedbackAndTargetsDisableOutput) {
    f.hand_valid = false; EXPECT_FALSE(hand.step(cmd, f, now)); EXPECT_FALSE(cmd.hand_enabled);
    f.hand_valid = true; tick();
    cmd.hand_mask = 1; cmd.hand[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(hand.step(cmd, f, now)); EXPECT_FALSE(cmd.hand_enabled);
    Control control({}); control.fault(Fault::Hand);
    EXPECT_NE(faultDescription(control.status(), {}).find("Hand feedback"), std::string::npos);
    EXPECT_FALSE(control.request({1, now+1, Mode::Still}, f, now).success);
}
TEST_F(HandTest, SlowRateReachesTargetBeforeNextPhase) {
    config.max_velocity = .2;
    tick();
    for (int i = 0; i < 150; ++i) tick();
    EXPECT_LT(cmd.hand[0], .3); EXPECT_EQ(cmd.hand[2], 0);
    for (int i = 0; i < 1000; ++i) tick();
    EXPECT_EQ(cmd.hand, config.positions.back());
}
TEST_F(HandTest, DisabledIsTransparentAndRealtimeDoesNotAllocate) {
    config.enabled = false; cmd.hand.fill(.3);
    EXPECT_TRUE(hand.step(cmd, {}, now)); EXPECT_FALSE(cmd.hand_enabled);
    EXPECT_EQ(cmd.hand[0], .3);
    config.enabled = true;
    Channel<Command> channel;
    beginAllocationTracking();
    bool ok = true;
    for (int i = 0; i < 10000; ++i) {
        now += 2000000;
        cmd.mode = i % 2000 == 0 ? Mode::Still : Mode::Walk;
        cmd.hand_mask = i % 100 < 50 ? 1 : 0;
        cmd.hand[0] = .3;
        ok = hand.step(cmd, f, now) && ok;
        channel.push(cmd); channel.latest(cmd);
    }
    const auto allocations = endAllocationTracking();
    EXPECT_TRUE(ok); EXPECT_EQ(allocations, 0U);
}
TEST(HandConfig, RejectsUnsafeParametersAndReservedSlots) {
    auto root = YAML::LoadFile(std::string(PRO_CONFIG).substr(0, std::string(PRO_CONFIG).find_last_of('/')) + "/robots/pro.yaml");
    const auto n = root["hand_control"];
    for (const auto* key : {"kp", "kd", "max_velocity", "dt_cap"}) {
        auto bad = YAML::Clone(n); bad[key] = "nan";
        EXPECT_THROW(loadHandConfig(bad), std::exception);
    }
    auto bad = YAML::Clone(n); bad["closed_position"][5] = 1;
    EXPECT_THROW(loadHandConfig(bad), std::exception);
    bad = YAML::Clone(n); bad["open_position"][0] = -1;
    EXPECT_THROW(loadHandConfig(bad), std::exception);
    bad = YAML::Clone(n); bad["durations"][0] = 0;
    EXPECT_THROW(loadHandConfig(bad), std::exception);
    bad = YAML::Clone(n); bad["unexpected"] = 0;
    EXPECT_THROW(loadHandConfig(bad), std::exception);
}
TEST(HandFeedback, ConvertsPercentClosureAndRejectsOfflineState) {
    eir::loco::State state;
    state.sdk.imu_state.resize(1); state.sdk.imu_state[0].quat[0] = 1;
    auto& hand = state.sdk.hand_states;
    hand.q.assign(12, 50); hand.code = {1,1};
    auto f = feedback(state, 1, {1,0,0,0});
    ASSERT_TRUE(f.hand_valid); EXPECT_EQ(f.hand[0], .5);
    hand.code[1] = 0; EXPECT_FALSE(feedback(state, 1, {1,0,0,0}).hand_valid);
    hand.code[1] = 1; hand.q[0] = 101; EXPECT_FALSE(feedback(state, 1, {1,0,0,0}).hand_valid);
    hand.q[0] = std::numeric_limits<double>::quiet_NaN(); EXPECT_FALSE(feedback(state, 1, {1,0,0,0}).hand_valid);
    hand.q.resize(11); EXPECT_FALSE(feedback(state, 1, {1,0,0,0}).hand_valid);
}
