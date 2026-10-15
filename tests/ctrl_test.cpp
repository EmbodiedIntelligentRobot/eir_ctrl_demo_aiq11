#include <algorithm>
#include <cmath>
#include <filesystem>
#include <eir_ctrl/inference.hpp>
#include <gtest/gtest.h>
#include <limits>
#include <random>
using namespace eir::ctrl;
TEST(Inference, RejectsUnknownBackend) {
    EXPECT_THROW(makeInference({"invalid", "", 780, 23, 1, 1}), std::invalid_argument);
}
#if EIR_CTRL_WITH_TORCH && EIR_CTRL_WITH_ONNX
TEST(Inference, ProActorBackendsAgreeAndRejectBadInput) {
    auto torch = makeInference({"libtorch", MODEL_DIR "/policy.pt", 780, 23, 1, 2});
    auto onnx = makeInference({"onnx", MODEL_DIR "/policy.onnx", 780, 23, 1, 2});
    std::mt19937 rng(431);
    std::normal_distribution<float> normal(0, 1);
    std::vector<float> input(780);
    for (int sample = 0; sample < 100; ++sample) {
        for (auto &x : input)
            x = normal(rng);
        auto a = torch->run(input), b = onnx->run(input);
        for (std::size_t i = 0; i < a.size(); ++i)
            EXPECT_NEAR(a[i], b[i], 1e-4);
    }
    EXPECT_THROW(torch->run({}), std::runtime_error);
    EXPECT_THROW(onnx->run({}), std::runtime_error);
    input[0] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(torch->run(input), std::runtime_error);
    EXPECT_THROW(onnx->run(input), std::runtime_error);
}
#endif

#include <eir_ctrl/config.hpp>
#include <eir_ctrl/mapping.hpp>
#include <eir_ctrl/remote.hpp>
#include <eir_closed_kinematics/ankle_interface.h>
TEST(RemotePriority, PhysicalRemotePreemptsGuiUntilLeaseExpires) {
    VelocityPriority priority;
    EXPECT_TRUE(priority.accept(1, 200, 100));
    EXPECT_TRUE(priority.accept(0, 220, 110));
    EXPECT_FALSE(priority.accept(1, 500, 120));
    EXPECT_FALSE(priority.accept(1, 0, 130)); // GUI zero cannot cancel remote.
    EXPECT_FALSE(priority.accept(1, 500, 219));
    EXPECT_TRUE(priority.accept(1, 300, 220)); // Only a newly received command.
}
TEST(RemotePriority, InvalidRemoteClearsWithoutExtendingLeaseAndUnknownPriorityIsRejected) {
    VelocityPriority priority;
    EXPECT_TRUE(priority.accept(0, 200, 100));
    EXPECT_TRUE(priority.accept(0, 0, 120)); // Invalid/keep_time=0 input becomes zero.
    EXPECT_TRUE(priority.accept(1, 250, 121));
    EXPECT_FALSE(priority.accept(2, 999, 122));
    EXPECT_FALSE(priority.accept(255, 999, 123));
}
TEST(RemotePriority, SbusOwnsInputAndLossDoesNotContinuouslyOverwriteFallback) {
    VelocityPriority priority;
    EXPECT_FALSE(priority.updateSbus(false));
    EXPECT_TRUE(priority.accept(1, 200, 100));
    EXPECT_TRUE(priority.updateSbus(true));
    EXPECT_TRUE(priority.sbusActive());
    EXPECT_FALSE(priority.accept(0, 300, 120));
    EXPECT_FALSE(priority.accept(1, 300, 120));
    EXPECT_TRUE(priority.updateSbus(false)); // Emit one zero at loss.
    EXPECT_FALSE(priority.sbusActive());
    EXPECT_TRUE(priority.accept(0, 350, 150));
    EXPECT_FALSE(priority.updateSbus(false)); // Do not erase Xbox input.
    EXPECT_FALSE(priority.accept(1, 400, 160));
    EXPECT_TRUE(priority.accept(1, 450, 350));
}
TEST(Config, ProProfileNamesAndMechanismsAreValid) {
    const auto c = loadConfig(PRO_CONFIG);
    EXPECT_EQ(c.robot, "pro");
    EXPECT_EQ(c.control.count, 32U);
    EXPECT_EQ(c.policies.at(Mode::Walk).inference.input_size, 780U);
    EXPECT_EQ(c.policies.at(Mode::Walk).indices.size(), 23U);
    Mapping mapping(c);
    Feedback f;
    f.q = c.control.ready;
    Command cmd;
    cmd.mode = Mode::Ready;
    cmd.q = f.q;
    cmd.kp = c.control.ready_kp;
    cmd.kd = c.control.ready_kd;
    auto result = mapping.command(cmd, f);
    EXPECT_EQ(result.position.size(), 32U);
    for (auto q : result.position)
        EXPECT_TRUE(std::isfinite(q));
}
TEST(Remote, SbusEdgesStopPriorityAndInvalidFrames) {
    const auto c = loadConfig(PRO_CONFIG);
    Sbus sbus(c.sbus);
    Feedback f;
    f.valid = true;
    f.remote_valid = true;
    f.stamp = 100;
    f.remote.fill(1002);
    f.remote[5] = 282;
    f.remote[9] = 1722;
    auto out = sbus.update(f, 100, 100);
    EXPECT_TRUE(out.valid);
    EXPECT_TRUE(out.has_request);
    EXPECT_EQ(out.request, Mode::Ready);
    EXPECT_FALSE(sbus.update(f, 101, 100).has_request);
    f.remote[5] = 1722;
    f.remote[6] = 1722;
    EXPECT_EQ(sbus.update(f, 102, 100).request, Mode::Still);
    EXPECT_TRUE(sbus.update(f, 103, 100).has_request);
    f.remote[2] = 0;
    out = sbus.update(f, 104, 100);
    EXPECT_FALSE(out.valid);
    EXPECT_FALSE(out.has_request);
    EXPECT_EQ(out.velocity.value, (std::array<double, 3>{}));
}
TEST(Control, RequiresReadyArrivalRejectsOldSessionAndLatchesTimeout) {
    auto c = loadConfig(PRO_CONFIG);
    Control control(c.control);
    Feedback f;
    f.valid = true;
    f.gravity = {0, 0, -1};
    f.q = c.control.ready;
    f.stamp = 1000000000;
    EXPECT_FALSE(control.request({1, 2000000000, Mode::Walk}, f, f.stamp).success);
    EXPECT_TRUE(control.request({2, 2000000000, Mode::Ready}, f, f.stamp).success);
    f.stamp += uint64_t(c.control.ready_seconds * 1e9);
    EXPECT_FALSE(control.step(f, {}, f.stamp).ready_complete);
    EXPECT_FALSE(control.status().ready);
    f.stamp += uint64_t(c.control.settle_seconds * 1e9) + 1;
    EXPECT_TRUE(control.step(f, {}, f.stamp).ready_complete);
    EXPECT_TRUE(control.status().ready);
    EXPECT_TRUE(control.request({3, f.stamp + 100, Mode::Walk}, f, f.stamp).success);
    EXPECT_TRUE(control.step(f, {}, f.stamp).ready_complete);
    Target old;
    old.valid = true;
    old.session = control.status().session - 1;
    old.q = c.control.ready;
    old.source_stamp = f.stamp;
    f.stamp += c.control.first_inference_timeout_ns + 1;
    control.step(f, old, f.stamp);
    EXPECT_EQ(control.status().mode, Mode::Fault);
    EXPECT_EQ(control.status().fault, Fault::Inference);
    EXPECT_FALSE(control.step(f, old, f.stamp).ready_complete);
    EXPECT_FALSE(control.request({4, f.stamp + 100, Mode::Ready}, f, f.stamp).success);
}
TEST(Channel, BoundedFullQueuePreservesProducerConsumerOrder) {
    Channel<int, 4> queue;
    EXPECT_TRUE(queue.push(1));
    EXPECT_TRUE(queue.push(2));
    EXPECT_TRUE(queue.push(3));
    EXPECT_FALSE(queue.push(4));
    int x = 0;
    EXPECT_TRUE(queue.latest(x));
    EXPECT_EQ(x, 3);
    EXPECT_FALSE(queue.pop(x));
    EXPECT_TRUE(queue.push(5));
    EXPECT_TRUE(queue.pop(x));
    EXPECT_EQ(x, 5);
}
namespace {
Feedback *sampled_feedback;
Target *sampled_target;
Velocity *sampled_velocity;
bool clock_saw_inputs;
uint64_t inputClock() noexcept {
    clock_saw_inputs = sampled_feedback->stamp == 101 && sampled_target->stamp == 101 &&
                       sampled_velocity->deadline == 201;
    return 102;
}
} // namespace
TEST(Control, ValidationClockRunsAfterAllInputSnapshots) {
    auto c = loadConfig(PRO_CONFIG);
    Channel<Feedback> feedbacks;
    Channel<Target> targets;
    Channel<Velocity> velocities;
    Feedback f, published;
    Target t;
    Velocity v;
    // Worker began at 100; producer published at 101 before the worker consumed.
    published.valid = true;
    published.stamp = 101;
    feedbacks.push(published);
    Target target;
    target.stamp = 101;
    targets.push(target);
    Velocity velocity;
    velocity.deadline = 201;
    velocities.push(velocity);
    sampled_feedback = &f;
    sampled_target = &t;
    sampled_velocity = &v;
    clock_saw_inputs = false;
    const auto now = receiveControlInputs(feedbacks, targets, velocities, f, t, v, inputClock);
    EXPECT_TRUE(clock_saw_inputs);
    Control previous(c.control), fixed(c.control);
    EXPECT_EQ(previous.step(f, t, 100).mode, Mode::Fault); // Reproduces the old order.
    EXPECT_EQ(fixed.step(f, t, now).mode, Mode::Still);
}
TEST(Control, FeedbackFaultDetailsLatchAndKeepRealDeadlines) {
    const auto c = loadConfig(PRO_CONFIG);
    for (auto reason : {FeedbackFault::Invalid, FeedbackFault::Future, FeedbackFault::Stale}) {
        Control control(c.control);
        Feedback f;
        f.valid = reason != FeedbackFault::Invalid;
        f.sequence = 42;
        const uint64_t now = 1000000000;
        f.stamp =
            reason == FeedbackFault::Future ? now + 1 : now - c.control.feedback_timeout_ns - 1;
        EXPECT_EQ(control.step(f, {}, now).mode, Mode::Fault);
        const auto s = control.status();
        EXPECT_EQ(s.fault, Fault::Feedback);
        EXPECT_EQ(s.feedback_fault, reason);
        EXPECT_EQ(s.feedback_sequence, 42U);
        EXPECT_FALSE(faultDescription(s, c.control).empty());
        f.valid = true;
        f.stamp = now + 100;
        f.sequence++;
        control.step(f, {}, now + 100);
        EXPECT_EQ(control.status().feedback_stamp, s.feedback_stamp);
        EXPECT_EQ(control.status().feedback_fault, reason);
        EXPECT_FALSE(control.request({1, now + 200, Mode::Ready}, f, now + 100).success);
    }
    Control control(c.control);
    Feedback f;
    f.valid = true;
    f.stamp = 1;
    EXPECT_EQ(control.step(f, {}, 1 + c.control.feedback_timeout_ns).mode, Mode::Still);
}
TEST(Control, ExpiredAndPreSessionActionsNeverReachCommand) {
    const auto c = loadConfig(PRO_CONFIG);
    Control control(c.control);
    Feedback f;
    f.valid = true;
    f.gravity = {0, 0, -1};
    f.q = c.control.ready;
    f.stamp = 1000000000;
    EXPECT_FALSE(control.request({1, f.stamp - 1, Mode::Ready}, f, f.stamp).success);
    ASSERT_TRUE(control.request({2, f.stamp + 100, Mode::Ready}, f, f.stamp).success);
    f.stamp += uint64_t(c.control.ready_seconds * 1e9);
    control.step(f, {}, f.stamp);
    f.stamp += uint64_t(c.control.settle_seconds * 1e9) + 1;
    control.step(f, {}, f.stamp);
    ASSERT_TRUE(control.request({3, f.stamp + 100, Mode::Walk}, f, f.stamp).success);
    Target t;
    t.valid = true;
    t.session = control.status().session;
    t.source_stamp = f.stamp;
    t.q = f.q;
    EXPECT_EQ(control.step(f, t, f.stamp).mode, Mode::Walk);
    f.stamp += c.control.inference_timeout_ns + 1;
    EXPECT_EQ(control.step(f, t, f.stamp).mode, Mode::Fault);
}
TEST(Control, FinitePositionLimitsWarnInStillReadyAndWalkAndClearOnRecovery) {
    const auto c = loadConfig(PRO_CONFIG);
    Control control(c.control);
    Feedback f;
    f.valid = true;
    f.gravity = {0, 0, -1};
    f.q = c.control.ready;
    f.stamp = 1000000000;
    f.q[13] = c.control.upper[13] + .02;
    f.q[29] = c.control.upper[29] + .02;
    const uint64_t mask = (uint64_t{1} << 13) | (uint64_t{1} << 29);
    EXPECT_EQ(control.step(f, {}, f.stamp).mode, Mode::Still);
    EXPECT_EQ(control.status().position_limit_mask, mask);
    ASSERT_TRUE(control.request({1, f.stamp + 100, Mode::Ready}, f, f.stamp).success);
    auto cmd = control.step(f, {}, f.stamp);
    EXPECT_EQ(cmd.mode, Mode::Ready);
    EXPECT_EQ(control.status().fault, Fault::None);
    EXPECT_EQ(control.status().position_limit_mask, mask);
    EXPECT_EQ(control.status().target_limit_mask, mask);
    EXPECT_DOUBLE_EQ(cmd.q[29], f.q[29]); // No silent clipping of the measured start.
    f.q = c.control.ready;
    f.stamp += uint64_t(c.control.ready_seconds * 1e9);
    control.step(f, {}, f.stamp);
    EXPECT_EQ(control.status().position_limit_mask, 0U);
    EXPECT_EQ(control.status().target_limit_mask, 0U);
    f.stamp += uint64_t(c.control.settle_seconds * 1e9) + 1;
    control.step(f, {}, f.stamp);
    ASSERT_TRUE(control.status().ready);
    ASSERT_TRUE(control.request({2, f.stamp + 100, Mode::Walk}, f, f.stamp).success);
    Target t;
    t.valid = true;
    t.session = control.status().session;
    t.source_stamp = f.stamp;
    t.q = f.q;
    t.q[29] = c.control.lower[29] - .1;
    cmd = control.step(f, t, f.stamp);
    EXPECT_EQ(cmd.mode, Mode::Walk);
    EXPECT_EQ(control.status().fault, Fault::None);
    EXPECT_EQ(control.status().target_limit_mask, uint64_t{1} << 29);
    EXPECT_DOUBLE_EQ(cmd.q[29], t.q[29]);
    EXPECT_DOUBLE_EQ(control.status().limit_target[29], t.q[29]);
    t.q = f.q;
    control.step(f, t, f.stamp);
    EXPECT_EQ(control.status().target_limit_mask, 0U);
}
TEST(Control, PerceptionAnkleTargetsRemainUnclippedAndInvalidInputsFault) {
    const auto c = loadConfig((std::filesystem::path(PRO_CONFIG).parent_path() /
                               "pro_perception.yaml").string());
    for (const auto mode : {Mode::Perception, Mode::Walk}) {
        for (const double excess : {0.5, -0.5, std::numeric_limits<double>::infinity(),
                                   std::numeric_limits<double>::quiet_NaN()}) {
            SCOPED_TRACE(unsigned(mode));
            SCOPED_TRACE(excess);
            auto config = c.control;
            config.policy_transition_seconds = 0;
            Control control(config);
            Feedback f;
            f.valid = true;
            f.q = config.ready;
            f.stamp = 1000000000;
            ASSERT_TRUE(control.request({1, f.stamp + 100, Mode::Ready}, f, f.stamp).success);
            f.stamp += uint64_t(config.ready_seconds * 1e9);
            control.step(f, {}, f.stamp);
            f.stamp += uint64_t(config.settle_seconds * 1e9) + 1;
            control.step(f, {}, f.stamp);
            ASSERT_TRUE(control.request({2, f.stamp + 100, mode}, f, f.stamp).success);
            Target target;
            target.valid = true;
            target.session = control.status().session;
            target.source_stamp = f.stamp;
            target.q = f.q;
            for (const auto i : {4U, 5U, 10U, 11U}) {
                target.q[i] = (excess > 0 ? config.upper[i] : config.lower[i]) + excess;
                target.dq[i] = excess > 0 ? 1 : -1;
            }
            // Other joint limits remain diagnostic, and measured overruns remain visible.
            target.q[13] = config.upper[13] + .1;
            f.q[4] = config.upper[4] + .1;
            auto cmd = control.step(f, target, f.stamp);
            if (!std::isfinite(excess)) {
                EXPECT_EQ(cmd.mode, Mode::Fault);
                EXPECT_EQ(control.status().fault, Fault::JointLimit);
                EXPECT_EQ(control.status().fault_joint, 4U);
                EXPECT_FALSE(control.request({3, f.stamp + 100, Mode::Ready}, f, f.stamp).success);
                continue;
            }
            ASSERT_EQ(cmd.mode, mode);
            EXPECT_NE(control.status().position_limit_mask & (uint64_t{1} << 4), 0U);
            EXPECT_DOUBLE_EQ(cmd.q[13], target.q[13]);
            for (const auto i : {4U, 5U, 10U, 11U}) {
                EXPECT_DOUBLE_EQ(cmd.q[i], target.q[i]);
                EXPECT_DOUBLE_EQ(cmd.dq[i], target.dq[i]);
                EXPECT_NE(control.status().target_limit_mask & (uint64_t{1} << i), 0U);
                EXPECT_DOUBLE_EQ(control.status().limit_target[i], target.q[i]);
            }
            const auto original = cmd;
            control.validateCommand(cmd); // Runner revalidates after motion/hand overlays.
            EXPECT_EQ(cmd.q, original.q);
            EXPECT_EQ(cmd.dq, original.dq);
            EXPECT_EQ(control.status().fault, Fault::None);
        }
    }
}

TEST(Mapping, PerceptionAnklePdUsesOriginalTargetsBeforeMotorTorqueLimits) {
    const auto c = loadConfig((std::filesystem::path(PRO_CONFIG).parent_path() /
                               "pro_perception.yaml").string());
    const auto &p = c.policies.at(Mode::Perception);
    for (const double sign : {-1., 1.}) {
        Feedback f;
        f.q = c.control.ready;
        Command cmd;
        cmd.mode = Mode::Perception;
        cmd.policy_target = true;
        cmd.q = f.q;
        for (const auto i : {4U, 5U, 10U, 11U}) {
            cmd.q[i] = sign > 0 ? c.control.upper[i] + .25 : c.control.lower[i] - .25;
            f.dq[i] = sign * .1;
        }
        Mapping mapping(c);
        TorqueWarning torque;
        MechanismWarning mechanism;
        const auto out = mapping.command(cmd, f, &torque, &mechanism);
        ASSERT_EQ(mechanism.mask, 0);
        for (const auto &pair : c.pairs) {
            if (pair.kind != "left_ankle" && pair.kind != "right_ankle") continue;
            const auto resource = std::filesystem::path(pair.resources);
            AnkleInterface solver(ANKLE_MOTOR_POSITION_VELOCITY,
                (resource / "ang_search_param.txt").string(),
                (resource / "fit_pitch_ploy33.txt").string(),
                (resource / "fit_roll_ploy33.txt").string(),
                (resource / "pitch.txt").string(), (resource / "roll.txt").string(),
                (resource / "pitch_roll_rge.txt").string());
            const auto a = pair.indices[0], b = pair.indices[1];
            const Eigen::Vector2d actual(f.q[a], f.q[b]);
            const Eigen::Vector2d pd(p.kp[a] * (cmd.q[a] - f.q[a]) - p.kd[a] * f.dq[a],
                                     p.kp[b] * (cmd.q[b] - f.q[b]) - p.kd[b] * f.dq[b]);
            const bool left = pair.kind == "left_ankle";
            const auto position = left ? solver.left_ankle_IK(actual) : solver.right_ankle_IK(actual);
            const auto expected = left ? solver.left_ankle_IK_force2tau(actual, pd)
                                       : solver.right_ankle_IK_force2tau(actual, pd);
            ASSERT_TRUE(expected.allFinite());
            for (int axis = 0; axis < 2; ++axis) {
                const auto i = pair.indices[axis];
                EXPECT_NEAR(out.torque[i], std::clamp(expected[axis], -c.torque_limit[i],
                                                     c.torque_limit[i]), 1e-9);
                EXPECT_NEAR(out.position[i], position[axis], 1e-9);
                EXPECT_EQ(out.kp[i], 0);
                EXPECT_EQ(out.kd[i], p.motor_kd[i]);
            }
        }
    }
}

TEST(Control, NonfiniteJointDataStillFaultsAndKeepsFirstInvalidIndex) {
    const auto c = loadConfig(PRO_CONFIG);
    for (int kind = 0; kind < 3; ++kind) {
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
        Target t;
        t.valid = true;
        t.session = control.status().session;
        t.source_stamp = f.stamp;
        t.q = f.q;
        auto &values = kind == 0 ? f.q : kind == 1 ? f.dq : t.q;
        values[13] = std::numeric_limits<double>::quiet_NaN();
        values[29] = std::numeric_limits<double>::infinity();
        EXPECT_EQ(control.step(f, t, f.stamp).mode, Mode::Fault);
        EXPECT_EQ(control.status().fault, Fault::JointLimit);
        EXPECT_EQ(control.status().fault_joint, 13U);
        EXPECT_NE(faultDescription(control.status(), c.control).find("Nonfinite"),
                  std::string::npos);
        f.q = c.control.ready;
        f.dq = {};
        control.step(f, t, f.stamp);
        EXPECT_EQ(control.status().fault_joint, 13U);
        EXPECT_FALSE(control.request({3, f.stamp + 100, Mode::Ready}, f, f.stamp).success);
    }
}
TEST(Mapping, ReadyBeginsAtMeasuredMotorPositions) {
    const auto c = loadConfig(PRO_CONFIG);
    Mapping mapping(c);
    Feedback f;
    f.q = c.control.ready;
    for (std::size_t i = 0; i < c.control.count; ++i)
        f.motor[i] = double(i) * .001;
    Command cmd;
    cmd.mode = Mode::Ready;
    cmd.session = 1;
    cmd.q = c.control.ready;
    cmd.kp = c.control.ready_kp;
    cmd.kd = c.control.ready_kd;
    cmd.ready_blend = 0;
    const auto start = mapping.command(cmd, f);
    for (std::size_t i = 0; i < c.control.count; ++i)
        EXPECT_DOUBLE_EQ(start.position[i], f.motor[i]);
}
TEST(Mapping, ReadyNeedsTargetIkButNotForceMappingOfSaggedPassiveJoints) {
    const auto c = loadConfig(PRO_CONFIG);
    Mapping mapping(c);
    Feedback f;
    f.q = c.control.ready;
    f.q[13] = .298; // Observed torso roll after prolonged suspended STILL.
    f.q[29] = .514;
    f.q[31] = -.272;
    f.motor[13] = .732;
    f.motor[29] = .514;
    f.motor[30] = -.266;
    f.motor[31] = -.265;
    Command cmd;
    cmd.mode = Mode::Ready;
    cmd.session = 1;
    cmd.kp = c.control.ready_kp;
    cmd.kd = c.control.ready_kd;
    auto out = mapping.command(cmd, f);
    for (std::size_t i = 0; i < c.control.count; ++i) {
        EXPECT_DOUBLE_EQ(out.position[i], f.motor[i]);
        EXPECT_DOUBLE_EQ(out.torque[i], 0);
    }
    cmd.ready_blend = 1;
    EXPECT_NO_THROW(mapping.command(cmd, f));
    // Invalid geometry holds the previously validated motor command.
    cmd.mode = Mode::Walk;
    cmd.q = c.control.ready;
    cmd.q[31] = 10;
    MechanismWarning warning;
    EXPECT_NO_THROW(mapping.command(cmd, f, nullptr, &warning));
    EXPECT_EQ(warning.mask, 8);
    cmd.policy_target = true;
    cmd.q = c.control.ready;
    f.q[13] = 10;
    EXPECT_NO_THROW(mapping.command(cmd, f, nullptr, &warning));
    EXPECT_NE(warning.mask & 4, 0);
}

#include <atomic>
#include <cstdlib>
#include <new>
#include <thread>
namespace {
thread_local bool track_allocations = false;
thread_local std::size_t allocation_count = 0;
} // namespace
void beginAllocationTracking() {
    allocation_count = 0;
    track_allocations = true;
}
std::size_t endAllocationTracking() {
    track_allocations = false;
    return allocation_count;
}
void *operator new(std::size_t size) {
    if (track_allocations)
        ++allocation_count;
    if (auto *p = std::malloc(size ? size : 1))
        return p;
    throw std::bad_alloc();
}
void operator delete(void *p) noexcept {
    std::free(p);
}
void operator delete(void *p, std::size_t) noexcept {
    std::free(p);
}
TEST(Control, RealTimeStepAndChannelsDoNotAllocate) {
    const auto c = loadConfig(PRO_CONFIG);
    Control control(c.control);
    Channel<Command> queue;
    Snapshot<Command> snapshot;
    Feedback f;
    f.valid = true;
    f.gravity = {0, 0, -1};
    f.q = c.control.ready;
    f.stamp = 1000000000;
    allocation_count = 0;
    track_allocations = true;
    control.request({1, f.stamp + 100, Mode::Ready}, f, f.stamp);
    for (int i = 0; i < 10000; ++i) {
        f.stamp += 2000000;
        f.q[29] = i % 2 ? c.control.upper[29] + .1 : c.control.ready[29];
        auto cmd = control.step(f, {}, f.stamp);
        queue.push(cmd);
        queue.latest(cmd);
        snapshot.push(cmd);
        snapshot.latest(cmd);
    }
    track_allocations = false;
    EXPECT_EQ(allocation_count, 0U);
}
TEST(Channel, ConcurrentTransferNeverTearsSnapshots) {
    struct Frame {
        uint64_t sequence{};
        std::array<uint64_t, 128> values{};
    };
    Channel<Frame> queue;
    std::atomic<bool> finished{false};
    std::thread producer([&] {
        for (uint64_t i = 1; i <= 10000; ++i) {
            Frame f;
            f.sequence = i;
            f.values.fill(i);
            while (!queue.push(f))
                std::this_thread::yield();
        }
        finished = true;
    });
    Frame f;
    uint64_t last = 0;
    bool coherent = true;
    while (!finished || last < 10000) {
        if (queue.pop(f)) {
            coherent = coherent && f.sequence == last + 1;
            for (auto x : f.values)
                coherent = coherent && x == f.sequence;
            last = f.sequence;
        } else
            std::this_thread::yield();
    }
    producer.join();
    EXPECT_TRUE(coherent);
    EXPECT_EQ(last, 10000U);
}

TEST(Feedback, ImuMountRotatesGravityAndGyroIntoBase) {
    eir::loco::State state;
    state.joints.resize(1);
    state.motors.resize(1);
    state.sdk.imu_state.resize(1);
    auto &imu = state.sdk.imu_state[0];
    imu.quat[0] = 1;
    imu.quat[1] = imu.quat[2] = imu.quat[3] = 0;
    imu.gyro[0] = 0;
    imu.gyro[1] = 1;
    imu.gyro[2] = 0;
    const double half = std::sqrt(0.5);
    const auto f = feedback(state, 123, {half, half, 0, 0});
    ASSERT_TRUE(f.valid);
    EXPECT_NEAR(f.gravity[0], 0, 1e-12);
    EXPECT_NEAR(f.gravity[1], 1, 1e-12);
    EXPECT_NEAR(f.gravity[2], 0, 1e-12);
    EXPECT_NEAR(f.omega[1], 0, 1e-12);
    EXPECT_NEAR(f.omega[2], 1, 1e-12);
    imu.quat[0] = 0;
    EXPECT_FALSE(feedback(state, 123, {1, 0, 0, 0}).valid);
}
TEST(Policy, RejectsUnportedIdentityAndBadDimensions) {
    auto c = loadConfig(PRO_CONFIG).policies.at(Mode::Walk);
    c.robot = "sport";
    EXPECT_THROW(makePolicy(c), std::invalid_argument);
    c.robot = "pro";
    c.inference.input_size = 779;
    EXPECT_THROW(makePolicy(c), std::invalid_argument);
}
TEST(Policy, ProObservationHistoryAndActionContract) {
    // Compare the complete policy path with independently assembled actor inputs.
    std::vector<std::string> backends;
#if EIR_CTRL_WITH_ONNX
    backends.push_back("onnx");
#endif
#if EIR_CTRL_WITH_TORCH
    backends.push_back("libtorch");
#endif
    for (const auto &backend : backends) {
        auto config = loadConfig(PRO_CONFIG);
        auto c = config.policies.at(Mode::Walk);
        c.inference.backend = backend;
        c.inference.model = std::string(MODEL_DIR) +
                            (backend == "onnx" ? "/policy.onnx" : "/policy.pt");
        auto policy = makePolicy(c);
        auto actor = makeInference(c.inference);
        Observation obs;
        obs.session = 17;
        obs.feedback.q = c.hold;
        obs.feedback.gravity = {0, 0, -1};
        obs.velocity.value = {0.1, -0.1, 0.2};
        std::vector<float> expected(780), previous(23, 0);
        for (int step = 0; step < 3; ++step) {
            obs.feedback.stamp = 1000 + step;
            obs.feedback.omega = {0.01 * step, 0.02 * step, -0.03 * step};
            std::vector<float> frame(78);
            for (int i = 0; i < 3; ++i) {
                frame[i] = obs.feedback.omega[i];
                frame[3 + i] = obs.feedback.gravity[i];
                frame[6 + i] = obs.velocity.value[i];
            }
            for (std::size_t j = 0; j < 23; ++j) {
                const auto index = c.indices[j];
                obs.feedback.q[index] = c.defaults[j] + .001 * j * (step + 1);
                obs.feedback.dq[index] = .02 * j;
                frame[9 + j] = obs.feedback.q[index] - c.defaults[j];
                const auto &name = config.runtime.joint_names[index];
                const bool ankle = name == "l_leg_5" || name == "l_leg_6" || name == "r_leg_5" ||
                                   name == "r_leg_6";
                frame[32 + j] = ankle ? 0 : obs.feedback.dq[index];
                frame[55 + j] = previous[j];
            }
            if (!step)
                for (int k = 0; k < 10; ++k)
                    std::copy(frame.begin(), frame.end(), expected.begin() + 78 * k);
            else {
                const auto old = expected;
                for (std::size_t i = 0; i < 702; ++i)
                    expected.at(i) = old.at(i + 78);
                std::copy(frame.begin(), frame.end(), expected.end() - 78);
            }
            previous = actor->run(expected);
            const auto output = policy->step(obs);
            EXPECT_EQ(output.session, obs.session);
            EXPECT_EQ(output.source_stamp, obs.feedback.stamp);
            for (std::size_t j = 0; j < 23; ++j) {
                previous[j] = std::clamp(previous[j], -float(c.action_clip), float(c.action_clip));
                EXPECT_NEAR(output.q[c.indices[j]], c.defaults[j] + c.scales[j] * previous[j],
                            1e-6);
            }
        }
        policy->reset();
        auto fresh = makePolicy(c);
        const auto reset_output = policy->step(obs), fresh_output = fresh->step(obs);
        EXPECT_EQ(reset_output.q, fresh_output.q);
    }
}

TEST(Mapping, SaturatesFiniteMotorFeedforwardAfterParallelConversion) {
    auto c = loadConfig(PRO_CONFIG);
    Feedback f;
    f.q = c.control.ready;
    Command cmd;
    cmd.mode = Mode::Walk;
    cmd.policy_target = true;
    cmd.q = f.q;
    // Exercise direct knee and both ankle motors, positive and negative torques.
    for (double sign : {-1., 1.}) {
        f.dq[3] = sign * 100;
        f.dq[4] = sign * 100;
        auto unbounded = c;
        unbounded.torque_limit.fill(1e9);
        Mapping reference(unbounded);
        const auto expected = reference.command(cmd, f);
        Mapping limited(c);
        TorqueWarning warning;
        auto out = limited.command(cmd, f, &warning);
        EXPECT_NE(warning.mask & (uint64_t{1} << 3), 0U);
        EXPECT_NE(warning.mask & ((uint64_t{1} << 4) | (uint64_t{1} << 5)), 0U);
        for (std::size_t i = 0; i < c.control.count; ++i) {
            EXPECT_DOUBLE_EQ(out.position[i], expected.position[i]);
            EXPECT_DOUBLE_EQ(out.torque[i],
                             std::clamp(expected.torque[i], -c.torque_limit[i], c.torque_limit[i]));
            if (warning.mask & (uint64_t{1} << i))
                EXPECT_DOUBLE_EQ(warning.requested[i], expected.torque[i]);
        }
        f.dq.fill(0);
        limited.command(cmd, f, &warning);
        EXPECT_EQ(warning.mask, 0U);
    }
    Mapping mapping(c);
    f.dq[3] = std::numeric_limits<double>::infinity();
    EXPECT_THROW(mapping.command(cmd, f), std::runtime_error);
    f.dq[3] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(mapping.command(cmd, f), std::runtime_error);
}

TEST(Mapping, WaistMotorDampingAddsToPolicyJointDamping) {
    const auto c = loadConfig(PRO_CONFIG);
    const auto roll = std::find(c.runtime.motor_names.begin(), c.runtime.motor_names.end(),
                                "torso_31") - c.runtime.motor_names.begin();
    const auto pitch = std::find(c.runtime.motor_names.begin(), c.runtime.motor_names.end(),
                                 "torso_33") - c.runtime.motor_names.begin();
    ASSERT_LT(roll, c.control.count);
    ASSERT_LT(pitch, c.control.count);
    Feedback f;
    f.q = c.control.ready;
    f.dq[roll] = 0.2;
    f.dq[pitch] = -0.3;
    for (const auto mode : {Mode::Walk}) {
        const auto &policy = c.policies.at(mode);
        SCOPED_TRACE(unsigned(mode));
        EXPECT_DOUBLE_EQ(policy.motor_kd[roll], 1.0);
        EXPECT_DOUBLE_EQ(policy.motor_kd[pitch], 1.0);
        auto reference_config = c;
        reference_config.policies.at(mode).motor_kd[roll] = 0;
        reference_config.policies.at(mode).motor_kd[pitch] = 0;
        Command cmd;
        cmd.mode = mode;
        cmd.policy_target = true;
        cmd.q = f.q;
        Mapping reference(reference_config), mapping(c);
        const auto expected = reference.command(cmd, f);
        MechanismWarning warning;
        const auto actual = mapping.command(cmd, f, nullptr, &warning);
        ASSERT_EQ(warning.mask, 0);
        for (std::size_t i = 0; i < c.control.count; ++i) {
            EXPECT_DOUBLE_EQ(actual.torque[i], expected.torque[i]);
            EXPECT_DOUBLE_EQ(actual.position[i], expected.position[i]);
            EXPECT_DOUBLE_EQ(actual.velocity[i], expected.velocity[i]);
            EXPECT_DOUBLE_EQ(actual.kp[i], expected.kp[i]);
            EXPECT_DOUBLE_EQ(actual.kd[i], expected.kd[i] + (i == roll || i == pitch ? 1.0 : 0));
        }
        // Zero motor velocity target makes the additional torque -Kd * measured dq.
        EXPECT_DOUBLE_EQ(actual.kp[roll], 0);
        EXPECT_DOUBLE_EQ(actual.kp[pitch], 0);
        EXPECT_DOUBLE_EQ(actual.velocity[roll], 0);
        EXPECT_DOUBLE_EQ(actual.velocity[pitch], 0);
    }
}

TEST(Mapping, AnkleFallbackKeepsLastFinalCommandPerSideAndRecovers) {
    const auto c = loadConfig(PRO_CONFIG);
    Mapping mapping(c);
    Feedback f;
    f.q = c.control.ready;
    f.motor.fill(.03);
    Command cmd;
    cmd.mode = Mode::Walk;
    cmd.q = c.control.ready;
    cmd.q[4] = 10;
    // Never invent a fallback before the first successful mapping.
    EXPECT_THROW(mapping.command(cmd, f), std::runtime_error);
    cmd.mode = Mode::Ready;
    cmd.session = 1;
    cmd.ready_blend = .4;
    cmd.q = c.control.ready;
    cmd.kp = c.control.ready_kp;
    cmd.kd = c.control.ready_kd;
    const auto last = mapping.command(cmd, f);
    auto same = [](const auto &out, const auto &expected, int a) {
        for (int i = a; i < a + 2; ++i) {
            EXPECT_DOUBLE_EQ(out.position[i], expected.position[i]);
            EXPECT_DOUBLE_EQ(out.velocity[i], expected.velocity[i]);
            EXPECT_DOUBLE_EQ(out.torque[i], expected.torque[i]);
            EXPECT_DOUBLE_EQ(out.kp[i], expected.kp[i]);
            EXPECT_DOUBLE_EQ(out.kd[i], expected.kd[i]);
        }
    };
    cmd.mode = Mode::Walk;
    cmd.q[4] = 10;
    cmd.q[10] += .02;
    MechanismWarning warning;
    auto next = mapping.command(cmd, f, nullptr, &warning);
    EXPECT_EQ(warning.mask, 1);
    EXPECT_NE(warning.detail[0].find("left_ankle"), std::string::npos);
    same(next, last, 4); // Preserve the interpolated motor command, not the final READY target.
    EXPECT_NE(next.position[10], last.position[10]); // Right ankle continues updating.
    for (int repeat = 0; repeat < 10; ++repeat) {
        const auto held = mapping.command(cmd, f, nullptr, &warning);
        same(held, last, 4);
        EXPECT_EQ(warning.mask, 1);
    }
    cmd.q[10] = 10;
    const auto both = mapping.command(cmd, f, nullptr, &warning);
    EXPECT_EQ(warning.mask, 3);
    same(both, last, 4);
    same(both, next, 10);
    cmd.q = c.control.ready;
    const auto recovered = mapping.command(cmd, f, nullptr, &warning);
    EXPECT_EQ(warning.mask, 0);
    EXPECT_NE(recovered.position[4], last.position[4]);
    cmd.q[4] = 10;
    same(mapping.command(cmd, f, nullptr, &warning), recovered, 4);
    f.q[4] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(mapping.command(cmd, f), std::runtime_error);
}

TEST(Mapping, AnkleFallbackRetainsLimitedTorqueAndGains) {
    const auto c = loadConfig(PRO_CONFIG);
    Mapping mapping(c);
    Feedback f;
    f.q = c.control.ready;
    f.dq[4] = 100;
    Command cmd;
    cmd.mode = Mode::Walk;
    cmd.policy_target = true;
    cmd.q = f.q;
    TorqueWarning torque;
    const auto last = mapping.command(cmd, f, &torque);
    ASSERT_NE(torque.mask & ((uint64_t{1} << 4) | (uint64_t{1} << 5)), 0U);
    f.q[4] = 10; // Finite measured geometry outside the ankle solver domain.
    MechanismWarning warning;
    const auto held = mapping.command(cmd, f, &torque, &warning);
    EXPECT_EQ(warning.mask, 1);
    for (int i : {4, 5}) {
        EXPECT_DOUBLE_EQ(held.position[i], last.position[i]);
        EXPECT_DOUBLE_EQ(held.torque[i], last.torque[i]);
        EXPECT_LE(std::abs(held.torque[i]), c.torque_limit[i]);
        EXPECT_DOUBLE_EQ(held.kp[i], last.kp[i]);
        EXPECT_DOUBLE_EQ(held.kd[i], last.kd[i]);
    }
    f.q = c.control.ready;
    f.dq.fill(0);
    const auto recovered = mapping.command(cmd, f, &torque, &warning);
    EXPECT_EQ(warning.mask, 0);
    EXPECT_DOUBLE_EQ(recovered.torque[4], 0);
    EXPECT_DOUBLE_EQ(recovered.torque[5], 0);
}

TEST(Control, TiltDoesNotStopReadyOrAnyPolicyButFeedbackTimeoutStillLatches) {
    auto c = loadConfig(PRO_CONFIG);
    for (auto mode : {Mode::Walk, Mode::MimicA, Mode::MimicB, Mode::MimicC,
                      Mode::MimicD, Mode::MimicE, Mode::Perception}) {
        c.control.policy_mask |= uint64_t{1} << unsigned(mode);
        Control control(c.control);
        Feedback f;
        f.valid = true;
        f.q = c.control.ready;
        f.gravity = {0, 0, 1}; // Fully upside down, finite IMU data.
        f.stamp = 1000000000;
        ASSERT_TRUE(control.request({1, f.stamp + 1, Mode::Ready}, f, f.stamp).success);
        EXPECT_EQ(control.step(f, {}, f.stamp).mode, Mode::Ready);
        f.stamp += uint64_t(c.control.ready_seconds * 1e9);
        control.step(f, {}, f.stamp);
        f.stamp += uint64_t(c.control.settle_seconds * 1e9) + 1;
        control.step(f, {}, f.stamp);
        ASSERT_TRUE(control.status().ready);
        ASSERT_TRUE(control.request({2, f.stamp + 1, mode}, f, f.stamp).success);
        Target t;
        t.valid = true;
        t.session = control.status().session;
        t.source_stamp = f.stamp;
        t.q = f.q;
        EXPECT_EQ(control.step(f, t, f.stamp).mode, mode);
        EXPECT_EQ(control.status().fault, Fault::None);
        EXPECT_EQ(control.step(f, t, f.stamp + c.control.feedback_timeout_ns + 1).mode, Mode::Fault);
        EXPECT_EQ(control.status().fault, Fault::Feedback);
    }
}

TEST(Mapping, WaistAndNeckFallbackAreIndependentAndRecoverWithoutInvalidOutput) {
    const auto c = loadConfig(PRO_CONFIG);
    for (auto kind : {"waist", "neck"}) {
        const auto pair = *std::find_if(c.pairs.begin(), c.pairs.end(),
                                      [kind](const auto& p) { return p.kind == kind; });
        const auto mask = kind == std::string("waist") ? 4 : 8;
        const auto a = pair.indices[0], b = pair.indices[1];
        Mapping mapping(c);
        Feedback f;
        f.q = c.control.ready;
        f.motor.fill(.03);
        Command cmd;
        cmd.mode = Mode::Walk;
        cmd.q = c.control.ready;
        cmd.q[a] = 10;
        // Without history an invalid first command is never sent.
        EXPECT_THROW(mapping.command(cmd, f), std::runtime_error);
        cmd.mode = Mode::Ready;
        cmd.session = 1;
        cmd.ready_blend = .4;
        cmd.q = c.control.ready;
        cmd.kp = c.control.ready_kp;
        cmd.kd = c.control.ready_kd;
        const auto last = mapping.command(cmd, f);
        cmd.mode = Mode::Walk;
        cmd.q[a] = 10;
        cmd.q[0] += .02;
        MechanismWarning warning;
        for (int i = 0; i < 10; ++i) {
            const auto held = mapping.command(cmd, f, nullptr, &warning);
            ASSERT_EQ(warning.mask, mask);
            EXPECT_NE(warning.detail[mask == 4 ? 2 : 3].find(kind), std::string::npos);
            for (const auto j : {a, b}) {
                EXPECT_DOUBLE_EQ(held.position[j], last.position[j]);
                EXPECT_DOUBLE_EQ(held.velocity[j], last.velocity[j]);
                EXPECT_DOUBLE_EQ(held.torque[j], last.torque[j]);
                EXPECT_DOUBLE_EQ(held.kp[j], last.kp[j]);
                EXPECT_DOUBLE_EQ(held.kd[j], last.kd[j]);
            }
            EXPECT_DOUBLE_EQ(held.position[0], cmd.q[0]);
        }
        cmd.q = c.control.ready;
        const auto recovered = mapping.command(cmd, f, nullptr, &warning);
        ASSERT_EQ(warning.mask, 0);
        EXPECT_NE(recovered.position[a], last.position[a]);
        cmd.q[a] = 10;
        const auto held = mapping.command(cmd, f, nullptr, &warning);
        EXPECT_DOUBLE_EQ(held.position[a], recovered.position[a]);
        f.q[a] = std::numeric_limits<double>::quiet_NaN();
        EXPECT_THROW(mapping.command(cmd, f), std::runtime_error);
    }
}

TEST(Control, ReadyDeviationGetsOneIndependentRecoveryDeadline) {
    const auto c = loadConfig(PRO_CONFIG);
    Control control(c.control);
    Feedback f; f.valid = true; f.q = c.control.ready; f.stamp = 1000000000;
    ASSERT_TRUE(control.request({1, f.stamp + 1, Mode::Ready}, f, f.stamp).success);
    f.stamp += uint64_t(c.control.ready_seconds * 1e9);
    control.step(f, {}, f.stamp);
    f.stamp += uint64_t(c.control.settle_seconds * 1e9) + 1;
    control.step(f, {}, f.stamp);
    ASSERT_TRUE(control.status().ready);
    f.stamp += uint64_t(2 * c.control.ready_timeout * 1e9);
    f.q[0] += 2 * c.control.ready_error;
    control.step(f, {}, f.stamp);
    ASSERT_EQ(control.status().mode, Mode::Ready);
    EXPECT_FALSE(control.status().ready);
    EXPECT_FALSE(control.request({2, f.stamp + 1, Mode::Walk}, f, f.stamp).success);
    const auto deviated = f.stamp;
    // Brief arrival without full settling must not renew the deadline.
    f.stamp += uint64_t(c.control.ready_timeout * .5e9);
    f.q = c.control.ready; control.step(f, {}, f.stamp);
    f.stamp += 1; f.q[0] += 2 * c.control.ready_error; control.step(f, {}, f.stamp);
    f.stamp = deviated + uint64_t(c.control.ready_timeout * 1e9) + 1;
    control.step(f, {}, f.stamp);
    EXPECT_EQ(control.status().fault, Fault::ReadyTimeout);
    EXPECT_FALSE(control.request({3, f.stamp + 1, Mode::Still}, f, f.stamp).success);
}
TEST(Control, ReadyRecoverySettlesAndFinalHandFaultRejectsCompletion) {
    const auto c = loadConfig(PRO_CONFIG);
    Control control(c.control); HandControl hand(c.hand);
    Feedback f; f.valid = true; f.q = c.control.ready; f.stamp = 1000000000;
    auto result = control.request({1, f.stamp + 1, Mode::Ready}, f, f.stamp);
    ASSERT_TRUE(result.success);
    auto cmd = control.step(f, {}, f.stamp);
    f.hand_valid = false;
    ASSERT_FALSE(hand.step(cmd, f, f.stamp));
    control.fault(Fault::Hand); control.validateCommand(cmd);
    result = control.complete(result);
    EXPECT_FALSE(result.success); EXPECT_EQ(result.mode, Mode::Fault);
    EXPECT_FALSE(cmd.hand_enabled);

    Control recovery(c.control);
    ASSERT_TRUE(recovery.request({2, f.stamp + 1, Mode::Ready}, f, f.stamp).success);
    f.stamp += uint64_t(c.control.ready_seconds * 1e9); recovery.step(f, {}, f.stamp);
    f.stamp += uint64_t(c.control.settle_seconds * 1e9) + 1; recovery.step(f, {}, f.stamp);
    ASSERT_TRUE(recovery.status().ready);
    f.stamp += uint64_t(2 * c.control.ready_timeout * 1e9);
    f.q[0] += 2 * c.control.ready_error; recovery.step(f, {}, f.stamp);
    EXPECT_FALSE(recovery.status().ready);
    f.stamp += 1000000; f.q = c.control.ready; recovery.step(f, {}, f.stamp);
    f.stamp += uint64_t(c.control.settle_seconds * 1e9) + 1; recovery.step(f, {}, f.stamp);
    EXPECT_TRUE(recovery.status().ready);
    EXPECT_TRUE(recovery.request({3, f.stamp + 1, Mode::Walk}, f, f.stamp).success);
}

TEST(Snapshot, SlowConsumerReceivesNewestObservationAndNeverTornFields) {
    Snapshot<Feedback> snapshot;
    Feedback f; EXPECT_FALSE(snapshot.latest(f));
    for (uint64_t i = 1; i <= 100; ++i) {
        f.sequence = i; f.q.fill(double(i)); snapshot.push(f);
    }
    ASSERT_TRUE(snapshot.latest(f)); EXPECT_EQ(f.sequence, 100U);
    EXPECT_FALSE(snapshot.latest(f));
    std::atomic<bool> done{false};
    std::thread writer([&] {
        Feedback value;
        for (uint64_t i = 101; i <= 100000; ++i) {
            value.sequence = i; value.q.fill(double(i)); snapshot.push(value);
        }
        done.store(true);
    });
    uint64_t last = 100;
    auto consume = [&] {
        if (snapshot.latest(f)) {
            EXPECT_GT(f.sequence, last);
            for (auto q : f.q) EXPECT_EQ(q, double(f.sequence));
            last = f.sequence;
        }
    };
    while (!done.load()) consume();
    writer.join(); consume(); EXPECT_EQ(last, 100000U);
}
TEST(Velocity, ExpiryIsRecheckedAfterObservationWaitWithoutRenewingLease) {
    Velocity v{{.2, -.1, .5}, 100};
    expireVelocity(v, 99); EXPECT_EQ(v.value[0], .2);
    expireVelocity(v, 100); EXPECT_EQ(v.value[0], .2);
    expireVelocity(v, 101); EXPECT_EQ(v.value, (std::array<double, 3>{}));
    EXPECT_EQ(v.deadline, 100U);
    expireVelocity(v, 200); EXPECT_EQ(v.value, (std::array<double, 3>{}));
}

#if EIR_CTRL_WITH_TORCH
TEST(Inference, WorkerInitializesConfiguredThreadsBeforeFirstForward) {
    auto inference = makeInference({"libtorch", MODEL_DIR "/policy.pt", 780, 23, 1, 2});
    std::vector<float> input(780);
    const auto reference = inference->run(input);
    auto average_ms = [&] {
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 64; ++i) {
            const auto result = inference->run(input);
            EXPECT_EQ(result, reference);
        }
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count() / 64;
    };
    const double main_ms = average_ms();
    double worker_ms = 0;
    // No ATen getter before forward: it can itself initialize threads and hide the defect.
    std::thread worker([&] { worker_ms = average_ms(); }); worker.join();
    EXPECT_LT(worker_ms, std::max(5.0, main_ms * 8))
        << "Warmup must not leave worker BLAS using the default thread count";
}
#endif

#include <eir_ctrl/runner.hpp>
namespace {
class HandFaultBackend final : public eir::loco::Backend {
    std::atomic<bool> running_{false};
    std::thread publisher_;
  public:
    RobotState raw{};
    std::atomic<bool> active_command{false};
    ~HandFaultBackend() override { stop(); }
    void start(const eir::loco::Config&, StateCallback cb) override {
        running_ = true;
        cb(raw);
        publisher_ = std::thread([this, cb] {
            while (running_) { cb(raw); std::this_thread::sleep_for(std::chrono::milliseconds(2)); }
        });
    }
    void stop() noexcept override {
        running_ = false; if (publisher_.joinable()) publisher_.join();
    }
    bool write(const RobotCmd& command) override {
        for (auto kp : command.motor_cmds.Kp) if (kp != 0) active_command = true;
        return true;
    }
    bool writeHand(const EECmd&) override { return true; }
    bool estop() override { return true; }
};
}
TEST(Runner, InvalidHandsRejectReadyReplyAndNeverEnableBodyCommands) {
    for (int failure = 0; failure < 3; ++failure) {
        auto config = loadConfig(PRO_CONFIG);
        auto backend = std::make_unique<HandFaultBackend>();
        auto* fake = backend.get();
        const auto n = config.control.count;
        auto& state = fake->raw;
        state.motor_states.q.assign(n * 2, 0);
        state.motor_states.dq.assign(n * 2, 0);
        state.motor_states.tau.assign(n * 2, 0);
        std::copy_n(config.control.ready.begin(), n, state.motor_states.q.begin());
        state.imu_state.resize(1); state.imu_state[0].quat[0] = 1;
        state.hand_states.q.assign(12, 0); state.hand_states.code = {1, 1};
        if (failure < 2) state.hand_states.code[failure] = 0;
        else state.hand_states.q[0] = std::numeric_limits<double>::quiet_NaN();
        Runner runner(config, std::move(backend));
        runner.start();
        const auto now = monotonicNow();
        ASSERT_TRUE(runner.requests.push({1, now + 500000000, Mode::Ready}));
        Completion result;
        const auto deadline = now + 2000000000;
        while (!runner.completions.pop(result) && monotonicNow() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        EXPECT_EQ(result.id, 1U); EXPECT_FALSE(result.success);
        EXPECT_EQ(result.mode, Mode::Fault);
        EXPECT_EQ(runner.status().fault, Fault::Hand);
        const auto later = monotonicNow();
        ASSERT_TRUE(runner.still_requests.push({2, later + 500000000, Mode::Still}));
        result = {};
        while (!runner.completions.pop(result) && monotonicNow() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        EXPECT_EQ(result.id, 2U); EXPECT_FALSE(result.success);
        EXPECT_EQ(result.mode, Mode::Fault);
        runner.stop();
        EXPECT_FALSE(fake->active_command.load());
    }
}
