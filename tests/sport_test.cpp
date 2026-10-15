#include <eir_ctrl/config.hpp>
#include <eir_ctrl/mapping.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <random>

using namespace eir::ctrl;
namespace {
struct Capture final : Inference {
    std::vector<float> input;
    std::vector<float> run(const std::vector<float>& x) override {
        input = x;
        return std::vector<float>(23, .25);
    }
};
struct SportObservation : ::testing::Test {
    Config c = loadConfig(SPORT_CONFIG);
    Capture* capture{};
    std::unique_ptr<Policy> policy;
    Observation observation;
    void SetUp() override {
        auto backend = std::make_unique<Capture>();
        capture = backend.get();
        policy = makePolicy(c.policies.at(Mode::Walk), std::move(backend));
        observation.feedback.valid = true;
        observation.feedback.gravity = {0, 0, -1};
        const auto& p = c.policies.at(Mode::Walk);
        for (std::size_t j = 0; j < p.indices.size(); ++j)
            observation.feedback.q[p.indices[j]] = p.defaults[j];
    }
    void tick() {
        observation.feedback.stamp = monotonicNow();
        policy->step(observation);
        ASSERT_EQ(capture->input.size(), 800U);
    }
};
TEST(Sport, TopologyAndReadyMapping) {
    const auto c = loadConfig(SPORT_CONFIG);
    EXPECT_EQ(c.robot, "sport");
    EXPECT_EQ(c.control.count, 23U);
    EXPECT_EQ(c.runtime.model_name, "aiq11");
    EXPECT_EQ(c.pairs.size(), 3U);
    EXPECT_TRUE(c.runtime.finger_joint_names.empty());
    EXPECT_FALSE(c.hand.enabled);
    EXPECT_EQ(c.control.policy_mask, uint64_t{1} << unsigned(Mode::Walk));
    EXPECT_EQ(c.inference_hz, 100);
    Mapping mapping(c);
    Feedback feedback;
    feedback.q = c.control.ready;
    Command cmd;
    cmd.mode = Mode::Ready;
    cmd.session = 1;
    cmd.ready_blend = 1;
    cmd.q = c.control.ready;
    cmd.kp = c.control.ready_kp;
    cmd.kd = c.control.ready_kd;
    const auto motors = mapping.command(cmd, feedback);
    ASSERT_EQ(motors.position.size(), 23U);
    for (auto q : motors.position) EXPECT_TRUE(std::isfinite(q));
}
TEST_F(SportObservation, FirstFrameHistoryIsStationaryAndRepeatable) {
    tick();
    const auto first = capture->input;
    for (int h = 0; h < 10; ++h) {
        EXPECT_FLOAT_EQ(first[h * 80 + 5], -1);
        EXPECT_FLOAT_EQ(first[h * 80 + 6], 0);
        EXPECT_FLOAT_EQ(first[h * 80 + 78], 0);
        EXPECT_FLOAT_EQ(first[h * 80 + 79], 0);
    }
    observation.velocity.value = {1, 0, 0};
    for (int step = 0; step < 20; ++step) tick();
    EXPECT_GT(capture->input[798], 0);
    policy->reset();
    tick();
    EXPECT_EQ(capture->input, first); // reset removes command, phase, history and last action
}
TEST_F(SportObservation, CommandRampsAndHistoryIsFrameMajorOldestFirst) {
    tick();
    observation.velocity.value = {10, 0, 0};
    tick();
    EXPECT_FLOAT_EQ(capture->input[6], 0);
    EXPECT_NEAR(capture->input[726], .015, 1e-7);
    EXPECT_FLOAT_EQ(capture->input[775], .25);
    // The reference holds phase at zero until the ramped speed reaches 0.05.
    EXPECT_FLOAT_EQ(capture->input[798], 0);
    for (int step = 0; step < 2; ++step) {
        tick();
        EXPECT_FLOAT_EQ(capture->input[798], 0);
    }
    tick(); // 0.06 m/s: first phase increment, swing=.4-.1*(.06/3)
    EXPECT_NEAR(capture->input[798], (1 - std::cos(2 * M_PI * .01 / .398)) / 2, 1e-6);
    for (int step = 0; step < 100; ++step) tick();
    EXPECT_FLOAT_EQ(capture->input[726], 1);
    observation.velocity.value = {};
    for (int step = 0; step < 400; ++step) tick();
    EXPECT_FLOAT_EQ(capture->input[726], 0);
    EXPECT_FLOAT_EQ(capture->input[798], 0);
    EXPECT_FLOAT_EQ(capture->input[799], 0);
}
TEST_F(SportObservation, DirectionChoosesFootAndVelocityFilterResets) {
    tick();
    observation.velocity.value = {0, -.6, 0};
    observation.feedback.dq[0] = 1;
    tick();
    EXPECT_FLOAT_EQ(capture->input[798], 0);
    EXPECT_FLOAT_EQ(capture->input[799], 0);
    EXPECT_NEAR(capture->input[752], 1 - std::exp(-2 * M_PI * .1), 1e-6);
    for (int step = 0; step < 3; ++step) tick();
    EXPECT_FLOAT_EQ(capture->input[798], 0);
    EXPECT_GT(capture->input[799], 0);
    policy->reset();
    tick();
    EXPECT_NEAR(capture->input[752], 1 - std::exp(-2 * M_PI * .1), 1e-6);
}
#if EIR_CTRL_WITH_TORCH && EIR_CTRL_WITH_ONNX
TEST(Sport, BothBackendsAgreeOnBoundedInputs) {
    auto config = loadConfig(SPORT_CONFIG).policies.at(Mode::Walk).inference;
    auto onnx = makeInference(config);
    config.backend = "libtorch";
    config.model.replace(config.model.size() - 4, 4, "pt");
    auto torch = makeInference(config);
    std::mt19937 random(711);
    std::uniform_real_distribution<float> uniform(-1, 1);
    std::vector<float> input(800);
    for (int sample = 0; sample < 100; ++sample) {
        for (auto& x : input) x = uniform(random);
        const auto a = torch->run(input), b = onnx->run(input);
        ASSERT_EQ(a.size(), 23U);
        for (std::size_t i = 0; i < a.size(); ++i) EXPECT_NEAR(a[i], b[i], 1e-4);
    }
}
#endif
} // namespace
