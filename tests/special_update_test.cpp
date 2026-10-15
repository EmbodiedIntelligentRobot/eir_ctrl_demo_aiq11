#include <eir_ctrl/config.hpp>
#include <eir_ctrl/mapping.hpp>
#include <eir_ctrl/runner.hpp>
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <unistd.h>

using namespace eir::ctrl;
namespace {
struct SportProfile {
    std::filesystem::path file = std::filesystem::temp_directory_path() /
        ("eir-sport-special-" + std::to_string(getpid()) + ".yaml");
    YAML::Node root = YAML::LoadFile(SPORT_SPECIAL_CONFIG);
    SportProfile() {
        root["flight_recorder"] = YAML::LoadFile(
            (std::filesystem::path(SPORT_SPECIAL_CONFIG).parent_path() / "flight_recorder.yaml")
                .string());

        const auto base = std::filesystem::path(SPORT_SPECIAL_CONFIG).parent_path();
        root["robot"] = (base / root["robot"].as<std::string>()).string();
        root["runtime"]["model"] = (base / root["runtime"]["model"].as<std::string>()).string();
        for (auto p : root["policies"])
            p.second = (base / p.second.as<std::string>()).string();
    }
    ~SportProfile() { std::filesystem::remove(file); }
    Config load(bool original = false) {
        if (original) root["policies"][38] = (std::filesystem::path(SPORT_SPECIAL_CONFIG).parent_path() /
            "../policies/mimic/sport/getup_prone_original/config.yaml").string();
        std::ofstream(file) << root;
        return loadConfig(file.string());
    }
};
}
TEST(SpecialUpdate, SportIdentityContractsAndBothGetupVariants) {
    for (bool original : {false, true}) {
        SportProfile profile;
        const auto c = profile.load(original);
        EXPECT_EQ(c.control.count, 23U);
        EXPECT_EQ(c.policies.size(), 7U);
        EXPECT_DOUBLE_EQ(c.policies.at(Mode::Walk).step_seconds, .01);
        for (auto mode : {Mode::LiedownSupine, Mode::LiedownProne, Mode::GetupSupine, Mode::GetupProne, Mode::StandToSquat, Mode::SquatToStand}) {
            const auto &p = c.policies.at(mode);
            EXPECT_EQ(p.robot, "sport");
            EXPECT_EQ(p.inference.input_size, 544U);
            EXPECT_EQ(p.indices.size(), 23U);
            EXPECT_TRUE(p.mimic.tracked_indices.empty());
            EXPECT_DOUBLE_EQ(p.step_seconds, .02);
        }
        const auto &p = c.policies.at(Mode::GetupProne);
        EXPECT_DOUBLE_EQ(p.mimic.first_frame_hold_seconds, original ? 2 : 0);
        EXPECT_EQ(c.control.entries[38].from_modes, uint64_t{1} << 36);
        EXPECT_TRUE(c.control.entries[38].require_motion_complete);
        EXPECT_TRUE(c.control.entries[38].heading.projected);
    }
    SportProfile profile;
    profile.root["timing"]["inference_timeout"] = .03;
    EXPECT_THROW(profile.load(), std::invalid_argument); // Two slow-policy periods required.
}
#if EIR_CTRL_WITH_TORCH && EIR_CTRL_WITH_ONNX
TEST(SpecialUpdate, SportBackendsAgreeThroughAllFramesHoldAndReset) {
    for (bool original : {false, true}) {
        SportProfile profile;
        const auto c = profile.load(original);
        for (auto mode : {Mode::LiedownSupine, Mode::LiedownProne, Mode::GetupSupine, Mode::GetupProne, Mode::StandToSquat, Mode::SquatToStand}) {
            if (original && mode != Mode::GetupProne) continue;
            SCOPED_TRACE(std::to_string(int(mode)) + (original ? " original" : " default"));
            auto p = c.policies.at(mode);
            const auto folder = std::filesystem::path(p.inference.model).parent_path();
            p.inference.backend = "onnx"; p.inference.model = (folder / "policy.onnx").string();
            auto a = makePolicy(p);
            p.inference.backend = "libtorch"; p.inference.model = (folder / "policy.pt").string();
            auto b = makePolicy(p);
            Observation o;
            o.feedback.orientation = p.mimic.heading.reference;
            for (std::size_t j = 0; j < p.indices.size(); ++j) o.feedback.q[p.indices[j]] = p.defaults[j];
            const auto initial = a->step(o); b->step(o);
            const auto steps = initial.motion_frames + std::size_t(p.mimic.first_frame_hold_seconds / p.step_seconds) + 10;
            Target x, y;
            for (std::size_t step = 1; step < steps; ++step) {
                o.feedback.omega = {.001 * std::sin(step), .002, 0};
                x = a->step(o); y = b->step(o);
                ASSERT_EQ(x.motion_frame, y.motion_frame);
                for (std::size_t j = 0; j < c.control.count; ++j)
                    ASSERT_NEAR(x.q[j], y.q[j], 3e-4) << "step=" << step << " joint=" << j;
            }
            EXPECT_EQ(x.motion_frame + 1, x.motion_frames);
            a->reset(); b->reset(); o.feedback.omega = {};
            x = a->step(o); y = b->step(o);
            EXPECT_EQ(x.motion_frame, 0U);
            for (std::size_t j = 0; j < c.control.count; ++j) {
                EXPECT_DOUBLE_EQ(x.q[j], initial.q[j]);
                EXPECT_NEAR(x.q[j], y.q[j], 3e-4);
            }
        }
    }
}
#endif
TEST(SpecialUpdate, FinalPoseGainsRampOnceBeforeProjectionAndKeepAnkleStiffness) {
    const auto c = loadConfig(SPORT_SPECIAL_CONFIG);
    Mapping mapping(c);
    Feedback f; f.q = c.control.ready;
    Command cmd; cmd.q = f.q; cmd.policy_target = true; cmd.mode = Mode::StandToSquat;
    cmd.session = 1; cmd.motion_finished_stamp = 1000000000;
    const auto &p = c.policies.at(cmd.mode);
    for (double elapsed : {0., .02, .52, 1.02, 2.02}) {
        cmd.stamp = cmd.motion_finished_stamp + uint64_t(elapsed * 1e9);
        const auto gains = policyGains(p, cmd, 0);
        const double blend = std::clamp((elapsed - .02), 0., 1.);
        EXPECT_NEAR(gains[0], p.kp[0] * (1 - .7 * blend), 1e-8);
        EXPECT_NEAR(gains[1], p.kd[0] * (1 + .2 * blend), 1e-8);
        for (auto j : {4, 5, 10, 11}) EXPECT_EQ(policyGains(p, cmd, j)[0], p.kp[j]);
        f.dq[3] = .3; cmd.q[3] = f.q[3] + .01;
        const auto out = mapping.command(cmd, f);
        EXPECT_NEAR(out.torque[3], p.kp[3] * (1 - .7 * blend) * .01 - p.kd[3] * (1 + .2 * blend) * .3, 1e-8);
        EXPECT_EQ(out.kd[3], p.motor_kd[3]); // HYBRID motor damping is not multiplied.
        EXPECT_NEAR(out.kp[0], gains[0], 1e-8);
        EXPECT_NEAR(out.kd[0], gains[1], 1e-8);
    }
    cmd.motion_finished_stamp = 0; ++cmd.session;
    EXPECT_EQ(policyGains(p, cmd, 0)[0], p.kp[0]);
    EXPECT_EQ(policyGains(c.policies.at(Mode::Walk), cmd, 0)[0], c.policies.at(Mode::Walk).kp[0]);
}
namespace {
// Fixed body feedback isolates the real worker scheduler from robot dynamics.
class StationaryBackend final : public eir::loco::Backend {
    std::atomic<bool> running_{false};
    std::thread worker_;
  public:
    RobotState raw{};
    ~StationaryBackend() override { stop(); }
    void start(const eir::loco::Config &, StateCallback cb) override {
        running_ = true;
        cb(raw);
        worker_ = std::thread([this, cb] {
            while (running_) { cb(raw); std::this_thread::sleep_for(std::chrono::milliseconds(2)); }
        });
    }
    void stop() noexcept override { running_ = false; if (worker_.joinable()) worker_.join(); }
    bool write(const RobotCmd &) override { return true; }
    bool estop() override { return true; }
};
}
TEST(SpecialUpdate, WorkerSwitches100HzAmpAnd50HzMimicWithoutFaultOrCatchupBurst) {
    for (const auto *backend_name : {
#if EIR_CTRL_WITH_ONNX
             "onnx",
#endif
#if EIR_CTRL_WITH_TORCH
             "libtorch",
#endif
         }) {
    SCOPED_TRACE(backend_name);
    auto c = loadConfig(SPORT_SPECIAL_CONFIG);
    for (auto &[mode, policy] : c.policies) {
        const auto folder = std::filesystem::path(policy.inference.model).parent_path();
        policy.inference.backend = backend_name;
        policy.inference.model = (folder / (std::string(backend_name) == "onnx" ? "policy.onnx" : "policy.pt")).string();
    }
    c.control.ready_seconds = .02; c.control.settle_seconds = .01;
    c.runtime.suspend = false; c.release_after_ready = false;
    c.pairs.clear(); // Scheduling test: mechanism projection is covered independently.
    auto backend = std::make_unique<StationaryBackend>();
    auto &raw = backend->raw;
    const auto n = c.control.count;
    raw.motor_states.q.assign(n * 2, 0);
    raw.motor_states.dq.assign(n * 2, 0);
    raw.motor_states.tau.assign(n * 2, 0);
    std::copy_n(c.control.ready.begin(), n, raw.motor_states.q.begin());
    raw.imu_state.resize(1); raw.imu_state[0].quat[0] = 1;
    Runner runner(c, std::move(backend)); runner.start();
    uint64_t request_id = 0;
    auto change = [&](Mode mode) {
        const auto now = monotonicNow();
        EXPECT_TRUE(runner.requests.push({++request_id, now + 500000000, mode}));
        Completion result;
        while (monotonicNow() - now < 1000000000 && !runner.completions.pop(result))
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        EXPECT_EQ(result.id, request_id); EXPECT_TRUE(result.success) << runner.error();
    };
    change(Mode::Ready);
    const auto deadline = monotonicNow() + 1000000000;
    while (!runner.status().ready && monotonicNow() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    ASSERT_TRUE(runner.status().ready) << runner.error();
    for (auto mode : {Mode::Walk, Mode::LiedownProne, Mode::Walk, Mode::StandToSquat}) {
        change(mode);
        std::this_thread::sleep_for(std::chrono::milliseconds(60));
        const auto begin = monotonicNow(), calls = runner.inferenceCalls();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        const double rate = double(runner.inferenceCalls() - calls) * 1e9 / (monotonicNow() - begin);
        EXPECT_NEAR(rate, mode == Mode::Walk ? 100 : 50, mode == Mode::Walk ? 15 : 8);
        EXPECT_EQ(runner.status().fault, Fault::None) << runner.error();
    }
    runner.stop();
    }
}
TEST(SpecialUpdate, BundledParametersMatchSourceIncludingHybridProjection) {
    auto check = [](const PolicyConfig &p) {
        const auto folder = std::filesystem::path(p.inference.model).parent_path();
        SCOPED_TRACE(folder.string());
        const auto raw = YAML::LoadFile((folder / "source.yaml").string())["/**/main_controller"]["ros__parameters"];
        std::string prefix;
        for (const auto &item : raw) {
            const auto key = item.first.as<std::string>();
            const auto at = key.find("profile_id");
            if (at != std::string::npos) prefix = key.substr(0, at);
        }
        auto get = [&](const char *key) { return raw[prefix + key]; };
        EXPECT_EQ(p.defaults, get("default_jit_pos").as<std::vector<double>>());
        EXPECT_EQ(p.scales, get("act_scales").as<std::vector<double>>());
        EXPECT_EQ(p.history, get("obs_his_length").as<std::size_t>());
        EXPECT_EQ(p.step_seconds, get("step_dt").as<double>());
        EXPECT_EQ(p.action_clip, get("action_cfg")["jit_pos"]["clip_range"][1].as<double>());
        EXPECT_EQ(p.mimic.first_frame_hold_seconds, get("first_frame_hold_sec").as<double>());
        EXPECT_EQ(p.mimic.warmup_steps, get("warm_up_steps").as<std::size_t>());
        EXPECT_EQ(p.mimic.loop, get("motion_loop").as<bool>());
        EXPECT_EQ(get("obs_history_init").as<std::string>(), "repeat_first_frame");
        EXPECT_TRUE(get("ankle_hold_before_history").as<bool>());
        EXPECT_EQ(get("gain_transition")["kp_start_ratio"].as<double>(), 1);
        EXPECT_EQ(get("gain_transition")["kd_start_ratio"].as<double>(), 1);
        for (const auto &item : get("obs_cfg")) {
            EXPECT_EQ(item.second["obs_scale"].as<double>(), 1);
            const bool gravity = item.first.as<std::string>().find("gravity") != std::string::npos;
            EXPECT_EQ(item.second["clip_range"][1].as<double>(), gravity ? 1 : p.observation_clip);
            EXPECT_EQ(item.second["clip_range"][0].as<double>(), gravity ? -1 : -p.observation_clip);
        }
        const bool hybrid_pro = get("output_mode").as<std::string>() == "HYBRID_PRO";
        const bool legacy = !YAML::LoadFile((folder / "config.yaml").string())["output_mode"];
        const uint64_t ankles = (uint64_t{1} << 4) | (uint64_t{1} << 5) | (uint64_t{1} << 10) | (uint64_t{1} << 11);
        EXPECT_EQ(p.motor_only_damping_mask, !legacy && hybrid_pro ? ankles : 0);
        for (std::size_t j = 0; j < p.indices.size(); ++j) {
            const auto i = p.indices[j];
            const bool ankle = j == 4 || j == 5 || j == 10 || j == 11;
            EXPECT_DOUBLE_EQ(p.kp[i], get("jit_kp")[j].as<double>());
            EXPECT_DOUBLE_EQ(p.kd[i], legacy && hybrid_pro && ankle ? 0 : get("jit_kd")[j].as<double>());
            if (ankle) EXPECT_DOUBLE_EQ(p.motor_kd[i], legacy && hybrid_pro ? get("jit_kd")[j].as<double>() : get("hybrid_leg56_motor_kd").as<double>());
            if (j == 3 || j == 9) EXPECT_DOUBLE_EQ(p.motor_kd[i], get("hybrid_leg4_motor_kd").as<double>());
            if (j == 12) EXPECT_DOUBLE_EQ(p.motor_kd[i], get("hybrid_torso1_motor_kd").as<double>());
            if (j == 13 || j == 14) EXPECT_DOUBLE_EQ(p.motor_kd[i], get("hybrid_torso23_motor_kd").as<double>());
        }
        std::vector<std::size_t> mask;
        if (get("ankle_vel_mask")[0].as<bool>()) { mask.push_back(4); mask.push_back(5); }
        if (get("ankle_vel_mask")[1].as<bool>()) { mask.push_back(10); mask.push_back(11); }
        EXPECT_EQ(p.velocity_mask, mask);
        const bool map = get("ankle_kd_map") && get("ankle_kd_map").as<bool>();
        EXPECT_EQ(p.ankle_damping.enabled, map);
        if (map) {
            EXPECT_EQ(p.ankle_damping.soft_qp, get("ankle_kd_mode").as<std::string>() == "SOFT_QP");
            EXPECT_EQ(p.ankle_damping.pitch_weight, get("ankle_kd_wp").as<double>());
            EXPECT_EQ(p.ankle_damping.roll_weight, get("ankle_kd_wr").as<double>());
            EXPECT_EQ(p.ankle_damping.cross_weight, get("ankle_kd_wc").as<double>());
            EXPECT_EQ(p.ankle_damping.fallback_motor_kd, get("ankle_kd_fb").as<double>());
        }
        EXPECT_EQ(p.mimic.heading.projected, bool(get("heading_alignment")));
        if (p.mimic.heading.projected) {
            EXPECT_EQ(p.mimic.heading.min_horizontal_projection, get("alignment_min_horizontal_projection").as<double>());
            EXPECT_EQ(p.mimic.heading.max_angular_velocity, get("alignment_max_angular_velocity").as<double>());
            EXPECT_EQ(p.mimic.heading.max_residual_angle, get("alignment_max_residual_angle").as<double>());
        }
        if (get("post_motion_kp") && get("post_motion_kp")["enabled"].as<bool>()) {
            EXPECT_EQ(p.mimic.post_kp_ratio, get("post_motion_kp")["ratio"].as<double>());
            EXPECT_EQ(p.mimic.post_kd_ratio, get("post_motion_kp")["kd_ratio"].as<double>());
            EXPECT_EQ(p.mimic.post_gain_seconds, get("post_motion_kp")["duration"].as<double>());
            EXPECT_EQ(p.mimic.post_kp_keep_mask, (uint64_t{1} << 4) | (uint64_t{1} << 5) | (uint64_t{1} << 10) | (uint64_t{1} << 11));
        } else EXPECT_EQ(p.mimic.post_gain_seconds, 0);
    };
    const auto pro = loadConfig(PRO_SPECIAL_CONFIG);
    for (auto mode : {Mode::LiedownSupine, Mode::LiedownProne, Mode::GetupSupine, Mode::GetupProne, Mode::StandToSquat, Mode::SquatToStand})
        check(pro.policies.at(mode));
    SportProfile sport;
    for (const auto &[mode, p] : sport.load().policies) if (isMimic(mode)) check(p);
    check(sport.load(true).policies.at(Mode::GetupProne));
}
TEST(SpecialUpdate, ControlFinalFrameTimestampDoesNotRestartAndClearsOnModeChange) {
    auto c = loadConfig(SPORT_SPECIAL_CONFIG);
    Control ctl(c.control);
    Feedback f; f.valid = true; f.gravity = {0, 0, -1}; f.q = c.control.ready; f.stamp = 1000000000;
    auto request = [&](Mode mode) { return ctl.request({1, f.stamp + 100, mode}, f, f.stamp); };
    ASSERT_TRUE(request(Mode::Ready).success);
    f.stamp += uint64_t(c.control.ready_seconds * 1e9); ctl.step(f, {}, f.stamp);
    f.stamp += uint64_t(c.control.settle_seconds * 1e9) + 1; ctl.step(f, {}, f.stamp);
    ASSERT_TRUE(request(Mode::StandToSquat).success);
    Target target; target.valid = true; target.q = f.q; target.session = ctl.status().session;
    target.source_stamp = f.stamp; target.motion_frames = 3; target.motion_frame = 1;
    EXPECT_EQ(ctl.step(f, target, f.stamp).motion_finished_stamp, 0U);
    target.motion_frame = 2;
    const auto first = ctl.step(f, target, f.stamp).motion_finished_stamp;
    EXPECT_EQ(first, f.stamp);
    f.stamp += 20000000; target.source_stamp = f.stamp;
    EXPECT_EQ(ctl.step(f, target, f.stamp).motion_finished_stamp, first);
    ASSERT_TRUE(request(Mode::Walk).success);
    EXPECT_EQ(ctl.step(f, target, f.stamp).motion_finished_stamp, 0U); // Prior session ignored.
    target.session = ctl.status().session; target.motion_frames = 0; target.motion_frame = 0;
    EXPECT_EQ(ctl.step(f, target, f.stamp).motion_finished_stamp, 0U);
    ASSERT_TRUE(request(Mode::StandToSquat).success);
    target.session = ctl.status().session; target.motion_frames = 3; target.motion_frame = 0;
    EXPECT_EQ(ctl.step(f, target, f.stamp).motion_finished_stamp, 0U);
}
TEST(SpecialUpdate, OriginalAnkleKdIsAppliedOnceAccordingToOutputMode) {
    for (const auto *file : {PRO_SPECIAL_CONFIG, SPORT_SPECIAL_CONFIG}) {
        auto c = loadConfig(file);
        c.pairs.clear(); // Isolate the output projection; SDK geometry has separate tests.
        Mapping mapping(c);
        Feedback f; f.q = c.control.ready;
        Command cmd; cmd.policy_target = true; cmd.q = f.q;
        for (int i : {4, 5, 10, 11}) { f.dq[i] = .7; cmd.q[i] += .01; }
        for (auto mode : {Mode::LiedownSupine, Mode::LiedownProne, Mode::GetupSupine,
                          Mode::GetupProne, Mode::StandToSquat, Mode::SquatToStand}) {
            const auto &p = c.policies.at(mode);
            const auto source = YAML::LoadFile((std::filesystem::path(p.inference.model).parent_path() /
                                               "source.yaml").string())["/**/main_controller"]["ros__parameters"];
            std::string prefix;
            for (const auto &item : source) {
                const auto key = item.first.as<std::string>();
                const auto at = key.find("profile_id");
                if (at != std::string::npos) prefix = key.substr(0, at);
            }
            const bool pro = source[prefix + "output_mode"].as<std::string>() == "HYBRID_PRO";
            cmd.mode = mode; ++cmd.session;
            const auto out = mapping.command(cmd, f);
            for (int i : {4, 5, 10, 11}) {
                const double kp = source[prefix + "jit_kp"][i].as<double>();
                const double kd = source[prefix + "jit_kd"][i].as<double>();
                EXPECT_NEAR(out.torque[i], kp * .01 - (pro ? 0 : kd * .7), 1e-8);
                EXPECT_DOUBLE_EQ(out.kd[i], pro ? kd : source[prefix + "hybrid_leg56_motor_kd"].as<double>());
                EXPECT_EQ(out.kp[i], 0);
            }
        }
    }
}
TEST(SpecialUpdate, SquatCanSwitchImmediatelyToWalkOrStandOnlyFromSquat) {
    for (const auto *file : {PRO_SPECIAL_CONFIG, SPORT_SPECIAL_CONFIG}) {
        auto c = loadConfig(file);
        ASSERT_EQ(c.control.entries[43].from_modes, uint64_t{1} << 42);
        ASSERT_FALSE(c.control.entries[43].require_motion_complete);
        for (auto destination : {Mode::Walk, Mode::SquatToStand}) {
            Control ctl(c.control);
            Feedback f; f.valid = true; f.gravity = {0, 0, -1};
            f.q = c.control.ready; f.stamp = 1000000000;
            auto request = [&](Mode mode) { return ctl.request({1, f.stamp + 100, mode}, f, f.stamp); };
            ASSERT_TRUE(request(Mode::Ready).success);
            f.stamp += uint64_t(c.control.ready_seconds * 1e9); ctl.step(f, {}, f.stamp);
            f.stamp += uint64_t(c.control.settle_seconds * 1e9) + 1; ctl.step(f, {}, f.stamp);
            EXPECT_EQ(request(Mode::SquatToStand).rejection, EntryRejection::SourceMode);
            ASSERT_TRUE(request(Mode::StandToSquat).success);
            const auto session = ctl.status().session;
            ASSERT_EQ(ctl.status().motion_frames, 0U); // No inference or final frame needed.
            ASSERT_TRUE(request(destination).success);
            EXPECT_EQ(ctl.status().session, session + 1);
            Target old; old.valid = true; old.q = f.q; old.session = session; old.source_stamp = f.stamp;
            EXPECT_FALSE(ctl.step(f, old, f.stamp).policy_target);
            if (destination == Mode::SquatToStand)
                EXPECT_EQ(request(destination).rejection, EntryRejection::SourceMode);
        }
        // Source restriction is a control invariant, even if an entry configuration is loosened.
        for (const auto &item : c.policies) c.control.entries[unsigned(item.first)] = {};
        for (const auto &[source, p] : c.policies) {
            if (source == Mode::StandToSquat) continue;
            Control ctl(c.control);
            Feedback f; f.valid = true; f.gravity = {0, 0, -1};
            f.q = c.control.ready; f.stamp = 1000000000;
            auto request = [&](Mode mode) { return ctl.request({1, f.stamp + 100, mode}, f, f.stamp); };
            ASSERT_TRUE(request(Mode::Ready).success);
            f.stamp += uint64_t(c.control.ready_seconds * 1e9); ctl.step(f, {}, f.stamp);
            f.stamp += uint64_t(c.control.settle_seconds * 1e9) + 1; ctl.step(f, {}, f.stamp);
            if (source != Mode::SquatToStand) ASSERT_TRUE(request(source).success);
            const auto session = ctl.status().session;
            EXPECT_EQ(request(Mode::SquatToStand).rejection, EntryRejection::SourceMode);
            EXPECT_EQ(ctl.status().session, session);
        }
    }
}
