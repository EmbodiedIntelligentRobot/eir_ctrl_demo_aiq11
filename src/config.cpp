#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <eir_ctrl/config.hpp>
#include <eir_ctrl/recording.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <openssl/evp.h>
#include <set>
#include <stdexcept>
#include "hardware.hpp"
namespace eir::ctrl {
namespace {
std::string sha256(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::invalid_argument("Cannot read policy model: " + path);
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!ctx || EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) != 1)
        throw std::runtime_error("Cannot initialize SHA256");
    std::array<char, 65536> buffer{};
    while (file) {
        file.read(buffer.data(), buffer.size());
        if (file.gcount() && EVP_DigestUpdate(ctx.get(), buffer.data(), file.gcount()) != 1)
            throw std::runtime_error("Cannot hash policy model");
    }
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned length{};
    if (file.bad() || EVP_DigestFinal_ex(ctx.get(), digest.data(), &length) != 1 || length != 32)
        throw std::runtime_error("Cannot finish policy SHA256");
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (unsigned i = 0; i < length; ++i) out << std::setw(2) << unsigned(digest[i]);
    return out.str();
}
void require(bool b, const std::string &message) {
    if (!b)
        throw std::invalid_argument(message);
}
void keys(const YAML::Node &n, std::initializer_list<const char *> allowed) {
    require(n.IsMap(), "Expected configuration mapping");
    for (const auto &pair : n) {
        const auto key = pair.first.as<std::string>();
        bool found = false;
        for (auto a : allowed)
            found = found || key == a;
        require(found, "Unknown config key: " + key);
    }
}
double number(const YAML::Node &n, const char *name, double minimum) {
    const double x = n[name].as<double>();
    require(std::isfinite(x) && x >= minimum, std::string("Invalid ") + name);
    return x;
}
std::size_t index(const std::vector<std::string> &names, const std::string &name) {
    const auto it = std::find(names.begin(), names.end(), name);
    require(it != names.end(), "Unknown joint: " + name);
    return it - names.begin();
}
Values values(const YAML::Node &n, std::size_t count, const char *label) {
    auto list = n.as<std::vector<double>>();
    require(list.size() == count, std::string("Wrong size: ") + label);
    Values out{};
    for (std::size_t i = 0; i < count; ++i) {
        require(std::isfinite(list[i]), std::string("Nonfinite ") + label);
        out[i] = list[i];
    }
    return out;
}
} // namespace
void enableJointObservation(Config &c) {
    require(c.runtime.backend == loco::BackendType::Hardware,
            "Joint observation requires an explicit hardware profile");
    c.runtime.passive_observation = true;
    c.runtime.motor_damping_kd.assign(c.runtime.motor_names.size(), 0.0);
    c.runtime.hand_control = false;
    c.hand.enabled = false;
    c.control.still_only = true;
    c.motion = {};
    c.remote_gui = false;
    c.remote_source = "external";
    c.debug = YAML::Clone(c.debug);
    c.debug["enabled"] = true;
}
Config loadConfig(const std::string &file, const std::string &mimic_override) {
    const auto base = std::filesystem::absolute(file).parent_path();
    auto path = [&](const YAML::Node &n) {
        auto text = n.as<std::string>();
        if (text.size() > 3 && text.substr(0, 2) == "${" && text.back() == '}') {
            const auto name = text.substr(2, text.size() - 3);
            const char *value = std::getenv(name.c_str());
            require(value && *value,
                    "Set environment variable " + name + " to an explicit resource path");
            text = value;
        }
        auto p = std::filesystem::path(text);
        return (p.is_absolute() ? p : base / p).lexically_normal().string();
    };
    const auto root = YAML::LoadFile(file);
    keys(root, {"robot", "policies", "inference", "runtime", "timing", "threads", "remote", "debug",
                "motion_stream", "flight_recorder", "still_only", "mimic_variant", "mimic_detection"});
    Config c;
    if (root["mimic_variant"]) c.mimic_variant = root["mimic_variant"].as<std::string>();
    if (!mimic_override.empty()) c.mimic_variant = mimic_override;
    require(c.mimic_variant == "fist" || c.mimic_variant == "gripper" || c.mimic_variant == "auto",
            "mimic_variant must be fist, gripper or auto");
    if (const auto d = root["mimic_detection"]) {
        keys(d, {"stable_seconds", "timeout_seconds"});
        c.mimic_stable_seconds = number(d, "stable_seconds", 0.001);
        c.mimic_detection_seconds = number(d, "timeout_seconds", c.mimic_stable_seconds);
        require(c.mimic_detection_seconds <= 10, "MIMIC detection exceeds 10 seconds");
    }
    if (root["still_only"])
        c.control.still_only = root["still_only"].as<bool>();
    std::string recording_recipe;
    c.runtime.recorder_controllers={"1:STILL","2:READY","255:FAULT"};
    if (const auto node = root["flight_recorder"]) {
        const auto f = node.IsScalar() ? YAML::LoadFile(path(node)) : node;
        keys(f, {"enabled", "directory", "detailed_hz", "application_hz", "continuous_hz",
                 "pre_seconds", "post_seconds", "cooldown_seconds", "sync_ms", "queue_frames",
                 "max_pending", "max_memory_bytes", "max_storage_bytes", "min_free_bytes",
                 "continuous_file_bytes", "data", "triggers", "trigger_exemptions"});
        auto &recorder = c.runtime.flight_recorder;
        if (f["enabled"])
            recorder.enabled = f["enabled"].as<bool>();
        if (f["directory"])
            recorder.directory = f["directory"].as<std::string>();
        // Output is relative to launch cwd, unlike input resource paths.
        auto integer = [&](const char *name, auto &value) {
            if (f[name]) {
                const auto n = f[name].as<int64_t>();
                require(n >= 0, "Negative recorder parameter");
                value = static_cast<std::decay_t<decltype(value)>>(n);
                require(uint64_t(value) == uint64_t(n), "Recorder integer overflow");
            }
        };
        integer("detailed_hz", recorder.detailed_hz);
        integer("application_hz", recorder.application_hz);
        integer("continuous_hz", recorder.continuous_hz);
        integer("pre_seconds", recorder.pre_seconds);
        integer("post_seconds", recorder.post_seconds);
        integer("cooldown_seconds", recorder.cooldown_seconds);
        integer("sync_ms", recorder.sync_ms);
        integer("queue_frames", recorder.queue_frames);
        integer("max_pending", recorder.max_pending);
        integer("max_memory_bytes", recorder.max_memory_bytes);
        integer("max_storage_bytes", recorder.max_storage_bytes);
        integer("min_free_bytes", recorder.min_free_bytes);
        integer("continuous_file_bytes", recorder.continuous_file_bytes);
        if (const auto data = f["data"]) {
            keys(data, {"commands", "motors", "imu", "encoders", "bms", "hands", "force_sensors",
                        "remote", "alarms", "application", "actuator_torque", "head"});
            const std::pair<const char *, uint32_t> groups[] = {
                {"commands", loco::flight::Commands},
                {"motors", loco::flight::Motors},
                {"imu", loco::flight::Imu},
                {"encoders", loco::flight::Encoders},
                {"bms", loco::flight::Bms},
                {"hands", loco::flight::Hands},
                {"force_sensors", loco::flight::Forces},
                {"remote", loco::flight::Remote},
                {"alarms", loco::flight::Alarms},
                {"application", loco::flight::Application},
                {"actuator_torque", loco::flight::ActuatorTorque},
                {"head", loco::flight::Head}};
            for (auto [name, bit] : groups)
                if (data[name]) {
                    if (data[name].as<bool>())
                        recorder.data |= bit;
                    else
                        recorder.data &= ~bit;
                }
        }
        auto event = [](const std::string &name) {
            for (unsigned i = 0; i < unsigned(loco::flight::Event::Count); ++i) {
                auto e = static_cast<loco::flight::Event>(i);
                if (name == loco::flight::Recorder::eventName(e))
                    return e;
            }
            throw std::invalid_argument("Unknown recorder trigger: " + name);
        };
        if (f["triggers"]) {
            require(f["triggers"].IsSequence(), "Recorder triggers must be a sequence");
            recorder.triggers = 0;
            for (const auto &t : f["triggers"])
                recorder.triggers |= loco::flight::bit(event(t.as<std::string>()));
        }
        if (f["trigger_exemptions"]) {
            require(f["trigger_exemptions"].IsSequence(), "Trigger exemptions must be a sequence");
            for (const auto &rule : f["trigger_exemptions"]) {
                keys(rule, {"event", "hardware", "names"});
                loco::flight::Exemption exemption;
                exemption.event = event(rule["event"].as<std::string>());
                const auto hw = rule["hardware"].as<std::string>();
                require(hw == "motor" || hw == "head_servo", "Unknown exemption hardware");
                exemption.component = hw == "motor" ? loco::flight::Component::Motor
                                                    : loco::flight::Component::HeadServo;
                if (rule["names"])
                    exemption.names = rule["names"].as<std::vector<std::string>>();
                recorder.trigger_exemptions.push_back(std::move(exemption));
            }
        }
    }
    const auto rt = root["runtime"];
    const auto backend = rt["backend"].as<std::string>();
    require(backend == "simulation" || backend == "hardware", "Unknown runtime backend");
    c.runtime.backend = backend == "hardware" ? loco::BackendType::Hardware : loco::BackendType::Simulation;
    const auto r = YAML::LoadFile(path(root["robot"]));
    keys(r, {"identity",      "joint_names",    "motor_names",    "ready",        "ready_kp",
             "ready_kd",      "lower",          "upper",          "hold",         "hold_kp",
             "hold_kd",       "torque_indices", "motor_kd",       "torque_limit", "pairs",
             "ready_seconds", "settle_seconds", "ready_timeout",  "ready_error",  "ready_velocity",
             "base_from_imu",  "simulation_hand", "hand_control"});
    const auto imu = values(r["base_from_imu"], 4, "base_from_imu");
    double norm = 0;
    for (std::size_t i = 0; i < 4; ++i)
        norm += imu[i] * imu[i];
    require(std::isfinite(norm) && norm > 1e-12, "Invalid IMU mounting quaternion");
    for (std::size_t i = 0; i < 4; ++i)
        c.base_from_imu[i] = imu[i] / std::sqrt(norm);
    c.robot = r["identity"].as<std::string>();
    c.runtime.robot_identity = c.robot;
    c.runtime.recorder_config_digest = sha256(file);
    recording_recipe=sha256(file)+sha256(path(root["robot"]));
    if(root["flight_recorder"] && root["flight_recorder"].IsScalar())
        recording_recipe+=sha256(path(root["flight_recorder"]));
    require(c.robot == "pro" || c.robot == "sport", "Unknown robot identity");
    require(c.robot == "pro" || (!root["mimic_variant"] && !root["mimic_detection"] && mimic_override.empty()),
            "MIMIC gripper/fist selection is PRO-only");
    require(c.mimic_variant != "auto" || c.runtime.backend == loco::BackendType::Hardware,
            "Simulation requires explicit fist or gripper selection");
    c.runtime.joint_names = r["joint_names"].as<std::vector<std::string>>();
    c.runtime.motor_names = r["motor_names"].as<std::vector<std::string>>();
    if (const auto hand = r["simulation_hand"]; hand && c.runtime.backend == loco::BackendType::Simulation) {
        keys(hand, {"joint_names", "coupling_csv", "couplings", "hold_position", "kp", "kd",
                    "torque_limits"});
        c.runtime.finger_joint_names = hand["joint_names"].as<std::vector<std::string>>();
        const auto robot_base = std::filesystem::path(path(root["robot"])).parent_path();
        c.runtime.finger_coupling_csv =
            (robot_base / hand["coupling_csv"].as<std::string>()).lexically_normal().string();
        require(std::filesystem::is_regular_file(c.runtime.finger_coupling_csv),
                "Missing finger coupling table: " + c.runtime.finger_coupling_csv);
        for (const auto &entry : hand["couplings"]) {
            keys(entry, {"curve", "active_joint", "passive_joint"});
            c.runtime.finger_couplings.push_back({entry["curve"].as<std::string>(),
                                                  entry["active_joint"].as<std::string>(),
                                                  entry["passive_joint"].as<std::string>()});
        }
        c.runtime.finger_hold_position = hand["hold_position"].as<std::vector<double>>();
        c.runtime.finger_hold_kp = number(hand, "kp", 0);
        c.runtime.finger_hold_kd = number(hand, "kd", 0);
        c.runtime.finger_torque_limits = hand["torque_limits"].as<std::vector<double>>();
        require(c.runtime.finger_joint_names.size() == 12 &&
                    c.runtime.finger_hold_position.size() == 12 &&
                    !c.runtime.finger_couplings.empty(),
                "Simulation hand requires 12 slots and explicit couplings");
    }
    if (r["hand_control"]) {
        c.hand = loadHandConfig(r["hand_control"]);
        c.runtime.hand_control = c.hand.enabled;
        require(!c.hand.enabled || (c.robot == "pro" && (c.runtime.backend == loco::BackendType::Hardware || c.runtime.finger_joint_names ==
            std::vector<std::string>{"l_fg1_1", "l_fg2_0", "l_fg2_1", "l_fg3_0", "l_fg3_1", "",
                                     "r_fg1_1", "r_fg2_0", "r_fg2_1", "r_fg3_0", "r_fg3_1", ""})),
                "Three-finger control requires the verified PRO slot mapping");
    }
    const auto n = c.runtime.joint_names.size();
    require(n > 0 && n <= capacity && c.runtime.motor_names.size() == n, "Invalid robot dimension");
    require(
        std::set<std::string>(c.runtime.joint_names.begin(), c.runtime.joint_names.end()).size() ==
            n,
        "Duplicate robot joints");
    c.control.count = n;
    c.control.ready = values(r["ready"], n, "ready");
    c.control.ready_kp = values(r["ready_kp"], n, "ready_kp");
    c.control.ready_kd = values(r["ready_kd"], n, "ready_kd");
    c.control.lower = values(r["lower"], n, "lower");
    c.control.upper = values(r["upper"], n, "upper");
    const auto hold = values(r["hold"], n, "hold");
    const auto hold_kp = values(r["hold_kp"], n, "hold_kp");
    const auto hold_kd = values(r["hold_kd"], n, "hold_kd");
    c.motor_kd = values(r["motor_kd"], n, "motor_kd");
    c.torque_limit = values(r["torque_limit"], n, "torque_limit");
    require(std::set<std::string>(c.runtime.motor_names.begin(), c.runtime.motor_names.end()).size() == n,
            "Duplicate robot motors");
    require(r["torque_indices"].IsSequence(), "torque_indices must be a sequence");
    std::set<std::size_t> torque;
    for (const auto &name : r["torque_indices"]) {
        const auto i = index(c.runtime.joint_names, name.as<std::string>());
        require(torque.insert(i).second, "Duplicate torque joint: " + name.as<std::string>());
        c.torque_indices.push_back(i);
    }
    // Mechanical topology is an identity contract, not a tunable control parameter.
    // SPORT has three shared pairs, four joints per arm, and no neck or hand.
    require(c.robot == "pro" || (!r["simulation_hand"] && !c.hand.enabled),
            "SPORT does not support PRO hand mappings");
    struct Mechanism {
        const char *kind;
        std::array<const char *, 2> joints, motors;
    };
    std::vector<Mechanism> mechanisms{
        {"left_ankle", {"l_leg_5", "l_leg_6"}, {"l_leg_41", "l_leg_43"}},
        {"right_ankle", {"r_leg_5", "r_leg_6"}, {"r_leg_41", "r_leg_43"}},
        {"waist", {"torso_3", "torso_2"}, {"torso_33", "torso_31"}},
    };
    if (c.robot == "pro")
        mechanisms.push_back({"neck", {"neck_3", "neck_2"}, {"neck_13", "neck_11"}});
    require(r["pairs"].IsSequence() && r["pairs"].size() == mechanisms.size(),
            c.robot == "pro" ? "PRO requires left_ankle, right_ankle, waist and neck mappings"
                             : "SPORT requires left_ankle, right_ankle and waist mappings");
    std::set<std::size_t> mapped;
    std::set<std::string> kinds;
    for (const auto &p : r["pairs"]) {
        keys(p, {"kind", "resources", "joints"});
        PairMapping pair;
        pair.kind = p["kind"].as<std::string>();
        const auto expected = std::find_if(mechanisms.begin(), mechanisms.end(),
            [&](const auto &m) { return pair.kind == m.kind; });
        require(expected != mechanisms.end(), "Unknown pair mapping: " + pair.kind);
        require(kinds.insert(pair.kind).second, "Duplicate mechanism: " + pair.kind);
        // Resource paths are relative to the robot profile, not the process cwd.
        const auto robot_base = std::filesystem::path(path(root["robot"])).parent_path();
        pair.resources =
            (robot_base / p["resources"].as<std::string>()).lexically_normal().string();
        require(p["joints"].size() == 2, "Pair needs two joints");
        for (int i = 0; i < 2; ++i) {
            const auto name = p["joints"][i].as<std::string>();
            require(name == expected->joints[i], "Wrong joint/order for mechanism: " + pair.kind);
            pair.indices[i] = index(c.runtime.joint_names, name);
            require(c.runtime.motor_names[pair.indices[i]] == expected->motors[i],
                    "Wrong motor/order for mechanism: " + pair.kind);
            require(mapped.insert(pair.indices[i]).second, "Overlapping mechanisms");
        }
        require(torque.count(pair.indices[0]) == torque.count(pair.indices[1]),
                "Both mechanism joints must use the same torque mode: " + pair.kind);
        c.pairs.push_back(pair);
    }
    // Check every motor slot, including direct drives, against the selected identity.
    std::set<std::string> expected_joints;
    for (const auto* side : {"l", "r"}) {
        for (int axis = 1; axis <= 6; ++axis)
            expected_joints.insert(std::string(side) + "_leg_" + std::to_string(axis));
        for (int axis = 1; axis <= (c.robot == "pro" ? 7 : 4); ++axis)
            expected_joints.insert(std::string(side) + "_arm_" + std::to_string(axis));
    }
    for (int axis = 1; axis <= 3; ++axis) {
        expected_joints.insert("torso_" + std::to_string(axis));
        if (c.robot == "pro") expected_joints.insert("neck_" + std::to_string(axis));
    }
    require(std::set<std::string>(c.runtime.joint_names.begin(), c.runtime.joint_names.end()) ==
                expected_joints, c.robot == "pro" ? "PRO joint identity contract mismatch"
                                                  : "SPORT joint identity contract mismatch");
    for (std::size_t i = 0; i < n; ++i) {
        auto expected_motor = c.runtime.joint_names[i];
        if (expected_motor == "l_arm_6") expected_motor = "l_arm_51";
        if (expected_motor == "r_arm_6") expected_motor = "r_arm_51";
        for (const auto& mechanism : mechanisms)
            for (int axis = 0; axis < 2; ++axis)
                if (c.runtime.joint_names[i] == mechanism.joints[axis])
                    expected_motor = mechanism.motors[axis];
        require(c.runtime.motor_names[i] == expected_motor,
                "Wrong motor identity for joint: " + c.runtime.joint_names[i]);
    }
    c.control.ready_seconds = number(r, "ready_seconds", 0.01);
    c.control.settle_seconds = number(r, "settle_seconds", 0.01);
    c.control.ready_timeout =
        number(r, "ready_timeout", c.control.ready_seconds + c.control.settle_seconds);
    c.control.ready_error = number(r, "ready_error", 0);
    c.control.ready_velocity = number(r, "ready_velocity", 0);
    c.control.policy_mask = 0;
    require(root["policies"].IsMap(), "policies must map wire mode numbers to profile paths");
    for (const auto &entry : root["policies"]) {
        const auto wire_mode = entry.first.as<int>();
        require(wire_mode >= 0 && wire_mode < 64 && isPolicy(static_cast<Mode>(wire_mode)),
                "Unsupported policy mode");
        const auto mode = static_cast<Mode>(wire_mode);
        require(!c.policies.count(mode), "Duplicate policy mode");
        PolicyConfig policy;
        YAML::Node perception_manifest;
        policy.hold = hold;
        policy.kp = hold_kp;
        policy.kd = hold_kd;
        policy.motor_kd = c.motor_kd;
        auto policy_path = path(entry.second);
        if (c.robot == "pro" && wire_mode >= 30 && wire_mode <= 34 &&
            (root["mimic_variant"] || !mimic_override.empty())) {
            const auto variant = c.mimic_variant == "gripper" ? "gripper" : "fist";
            policy_path = (std::filesystem::path(policy_path).parent_path() /
                           (std::string("config_") + variant + ".yaml")).string();
        }
        const auto policy_base = std::filesystem::path(policy_path).parent_path().string();
        const auto p = YAML::LoadFile(policy_path);
        keys(p, {"robot", "kind", "joints", "defaults", "scales", "kp", "kd", "velocity_mask",
                 "history", "observation_clip", "action_clip", "velocity_min", "velocity_max",
                 "models", "step_seconds", "motor_kd", "mimic", "perception", "entry", "ankle_damping", "amp_gait", "output_mode", "contract", "mimic_variant"});
        policy.robot = p["robot"].as<std::string>();
        require(policy.robot == c.robot, "Robot/policy identity mismatch");
        policy.kind = p["kind"].as<std::string>();
        if (p["mimic_variant"]) {
            policy.mimic_variant = p["mimic_variant"].as<std::string>();
            require(c.robot == "pro" && wire_mode >= 30 && wire_mode <= 34 &&
                    policy.mimic_variant == (c.mimic_variant == "gripper" ? "gripper" : "fist"),
                    "MIMIC attachment variant mismatch");
        }
        require((mode == Mode::Walk && policy.kind == "amp_humanoid_walk") ||
                    (c.robot == "pro" && mode == Mode::Perception && policy.kind == "perception_walk_v2") ||
                    (isMimic(mode) && policy.kind == "mimic_gravity_v1"),
                "Robot/mode/policy contract mismatch");
        policy.step_seconds = number(p, "step_seconds", 0.001);
        require(bool(p["amp_gait"]) == (c.robot == "sport" && mode == Mode::Walk),
                "SPORT AMP requires phase observation; PRO uses its existing contract");
        if (const auto g = p["amp_gait"]) {
            keys(g, {"term_clips", "velocity_filter_hz", "velocity_reference", "yaw_weight",
                     "swing_min", "swing_max", "double_support_min", "double_support_max",
                     "command_threshold", "direction_threshold", "stop_threshold",
                     "linear_accel", "linear_decel", "angular_accel", "angular_decel",
                     "extra_supports", "default_left"});
            auto &a = policy.amp_gait;
            a.enabled = true;
            const auto clips = values(g["term_clips"], 7, "AMP term clips");
            for (std::size_t i = 0; i < a.term_clips.size(); ++i) {
                require(clips[i] > 0, "Invalid AMP term clip");
                a.term_clips[i] = clips[i];
            }
            a.velocity_filter_hz = number(g, "velocity_filter_hz", 0);
            a.velocity_reference = number(g, "velocity_reference", 0.001);
            a.yaw_weight = number(g, "yaw_weight", 0);
            a.swing_min = number(g, "swing_min", policy.step_seconds);
            a.swing_max = number(g, "swing_max", a.swing_min);
            a.double_support_min = number(g, "double_support_min", 0);
            a.double_support_max = number(g, "double_support_max", a.double_support_min);
            a.command_threshold = number(g, "command_threshold", 0.0001);
            a.direction_threshold = number(g, "direction_threshold", 0);
            a.stop_threshold = number(g, "stop_threshold", 0.0001);
            a.linear_accel = number(g, "linear_accel", 0.001);
            a.linear_decel = number(g, "linear_decel", 0.001);
            a.angular_accel = number(g, "angular_accel", 0.001);
            a.angular_decel = number(g, "angular_decel", 0.001);
            require(a.stop_threshold < 1 && a.velocity_filter_hz <= 0.5 / policy.step_seconds,
                    "Invalid AMP phase/filter bounds");
            a.extra_supports = g["extra_supports"].as<int>();
            require(a.extra_supports >= 0 && a.extra_supports <= 10, "Invalid AMP stop supports");
            a.default_left = g["default_left"].as<bool>();
        }
        if (p["motor_kd"])
            policy.motor_kd = values(p["motor_kd"], n, "motor_kd");
        if (p["ankle_damping"]) {
            const auto d = p["ankle_damping"];
            keys(d, {"strategy", "pitch_weight", "roll_weight", "cross_weight", "fallback_motor_kd"});
            const auto strategy = d["strategy"].as<std::string>();
            require(strategy == "zero_cross" || strategy == "soft_qp", "Invalid ankle damping strategy");
            policy.ankle_damping = {true, strategy == "soft_qp", number(d, "pitch_weight", 0.0001),
                number(d, "roll_weight", 0.0001), number(d, "cross_weight", 0.0001),
                number(d, "fallback_motor_kd", 0)};
        }
        auto &gate = c.control.entries[wire_mode];
        if (p["entry"]) {
            keys(p["entry"], {"from_modes", "require_motion_complete"});
            for (int source : p["entry"]["from_modes"].as<std::vector<int>>()) {
                require(source > 0 && source < 64 && (source == 2 || isPolicy(static_cast<Mode>(source))),
                        "Invalid policy entry source");
                require(!(gate.from_modes & (uint64_t{1} << source)), "Duplicate policy entry source");
                gate.from_modes |= uint64_t{1} << source;
            }
            gate.require_motion_complete = p["entry"]["require_motion_complete"].as<bool>();
        }
        const auto names = p["joints"].as<std::vector<std::string>>();
        require(!names.empty() && names.size() <= n, "Invalid policy joint count");
        require(std::set<std::string>(names.begin(), names.end()).size() == names.size(),
                "Duplicate policy joint");
        for (const auto &name : names)
            policy.indices.push_back(index(c.runtime.joint_names, name));
        if (p["output_mode"]) {
            const auto output = p["output_mode"].as<std::string>();
            require(isMimic(mode) && (output == "HYBRID_MODE" || output == "HYBRID_PRO"),
                    "Invalid MIMIC output mode");
            if (output == "HYBRID_PRO")
                for (const auto *name : {"l_leg_5", "l_leg_6", "r_leg_5", "r_leg_6"})
                    policy.motor_only_damping_mask |= uint64_t{1} << index(c.runtime.joint_names, name);
        }
        policy.defaults = p["defaults"].as<std::vector<double>>();
        policy.scales = p["scales"].as<std::vector<double>>();
        const auto kp = values(p["kp"], names.size(), "policy kp"),
                   kd = values(p["kd"], names.size(), "policy kd");
        require(policy.defaults.size() == names.size() && policy.scales.size() == names.size(),
                "Policy vectors differ");
        for (std::size_t j = 0; j < names.size(); ++j) {
            const auto i = policy.indices[j];
            policy.kp[i] = kp[j];
            policy.kd[i] = kd[j];
            require(policy.defaults[j] >= c.control.lower[i] &&
                        policy.defaults[j] <= c.control.upper[i],
                    "Policy default outside robot limits");
            require(std::isfinite(policy.defaults[j]) && std::isfinite(policy.scales[j]) &&
                        policy.scales[j] > 0,
                    "Invalid action scaling/default");
        }
        for (const auto &name : p["velocity_mask"])
            policy.velocity_mask.push_back(index(c.runtime.joint_names, name.as<std::string>()));
        policy.history = p["history"].as<std::size_t>();
        require(policy.history > 0 && policy.history <= 100, "Invalid history length");
        if (mode != Mode::Perception)
            policy.observation_clip = number(p, "observation_clip", 0.01);
        policy.action_clip = number(p, "action_clip", 0.01);
        if (mode == Mode::Walk || mode == Mode::Perception) {
            const auto lo = values(p["velocity_min"], 3, "velocity_min"),
                       hi = values(p["velocity_max"], 3, "velocity_max");
            for (int i = 0; i < 3; ++i) {
                require(lo[i] <= 0 && hi[i] >= 0 && hi[i] > lo[i], "Invalid velocity bounds");
                policy.velocity_min[i] = lo[i];
                policy.velocity_max[i] = hi[i];
            }
        } else {
            const auto m = p["mimic"];
            keys(m, {"motion", "tracked_joints", "tracked_kp", "tracked_kd", "torso_joints",
                     "first_frame_hold_seconds", "warmup_steps", "loop", "heading", "post_motion_gains"});
            policy.mimic.motion =
                (std::filesystem::path(policy_base) / m["motion"].as<std::string>())
                    .lexically_normal()
                    .string();
            if (m["heading"]) {
                const auto h = m["heading"];
                keys(h, {"mode", "min_horizontal_projection", "max_angular_velocity", "max_residual_angle"});
                require(h["mode"].as<std::string>() == "projected_body_axis", "Invalid MIMIC heading mode");
                auto &heading = policy.mimic.heading;
                heading.projected = true;
                heading.min_horizontal_projection = number(h, "min_horizontal_projection", 0.0001);
                heading.max_angular_velocity = number(h, "max_angular_velocity", 0);
                heading.max_residual_angle = number(h, "max_residual_angle", 0.0001);
                require(heading.min_horizontal_projection <= 1 && heading.max_residual_angle <= M_PI,
                        "Invalid MIMIC alignment bounds");
                std::ifstream motion(policy.mimic.motion);
                std::string line, cell;
                while (std::getline(motion, line) && line.find_first_not_of(" \t\r") == std::string::npos) {}
                std::istringstream first(line);
                double norm = 0;
                for (auto &q : heading.reference) {
                    require(bool(std::getline(first, cell, ',')), "Missing initial MIMIC orientation");
                    std::size_t end{};
                    q = std::stod(cell, &end);
                    require(std::isfinite(q) && cell.find_first_not_of(" \t\r", end) == std::string::npos,
                            "Invalid initial MIMIC orientation");
                    norm += q * q;
                }
                require(std::isfinite(norm) && norm > 1e-12, "Invalid initial MIMIC orientation");
                gate.heading = heading;
            }
            std::set<std::size_t> used(policy.indices.begin(), policy.indices.end());
            for (const auto &name : m["tracked_joints"]) {
                auto i = index(c.runtime.joint_names, name.as<std::string>());
                require(used.insert(i).second, "Overlapping motion joint");
                policy.mimic.tracked_indices.push_back(i);
            }
            const auto count = policy.mimic.tracked_indices.size();
            const auto tracked_kp = values(m["tracked_kp"], count, "tracked_kp"),
                       tracked_kd = values(m["tracked_kd"], count, "tracked_kd");
            for (std::size_t j = 0; j < count; ++j) {
                const auto i = policy.mimic.tracked_indices[j];
                policy.kp[i] = tracked_kp[j];
                policy.kd[i] = tracked_kd[j];
            }
            require(m["torso_joints"].size() == 3, "MIMIC requires torso yaw/roll/pitch joints");
            for (std::size_t j = 0; j < 3; ++j)
                policy.mimic.torso[j] = index(names, m["torso_joints"][j].as<std::string>());
            policy.mimic.first_frame_hold_seconds = number(m, "first_frame_hold_seconds", 0);
            require(policy.mimic.first_frame_hold_seconds <= 60, "MIMIC hold exceeds 60 seconds");
            policy.mimic.warmup_steps = m["warmup_steps"].as<std::size_t>();
            require(policy.mimic.warmup_steps <= 100, "Invalid MIMIC warmup_steps");
            policy.mimic.loop = m["loop"].as<bool>();
            if (const auto g = m["post_motion_gains"]) {
                keys(g, {"kp_ratio", "kd_ratio", "duration", "kp_keep_joints"});
                require(mode == Mode::StandToSquat && !policy.mimic.loop,
                        "Post-motion gains require non-looping StandToSquat");
                auto &mc = policy.mimic;
                mc.post_kp_ratio = number(g, "kp_ratio", 0.0001);
                mc.post_kd_ratio = number(g, "kd_ratio", 1);
                mc.post_gain_seconds = number(g, "duration", 0.0001);
                require(mc.post_kp_ratio <= 1 && mc.post_kd_ratio <= 2,
                        "Invalid post-motion gain ratios");
                for (const auto &name : g["kp_keep_joints"].as<std::vector<std::string>>()) {
                    const auto i = index(c.runtime.joint_names, name);
                    require(used.count(i) && !(mc.post_kp_keep_mask & (uint64_t{1} << i)),
                            "Unknown or duplicate post-motion gain joint");
                    mc.post_kp_keep_mask |= uint64_t{1} << i;
                }
            }
        }
        if (mode == Mode::Perception) {
            const auto d = p["perception"];
            keys(d, {"manifest", "angular_velocity_scale", "joint_velocity_scale",
                     "depth_timeout", "inference_timeout"});
            const auto manifest = YAML::LoadFile((std::filesystem::path(policy_base) /
                d["manifest"].as<std::string>()).string());
            perception_manifest = manifest;
            require(manifest["schema_version"].as<int>() == 1 &&
                    manifest["input_dim"].as<int>() == 5232 &&
                    manifest["action_dim"].as<int>() == 23 &&
                    manifest["depth_frames"].as<int>() == 8 &&
                    manifest["depth_height"].as<int>() == 18 &&
                    manifest["depth_width"].as<int>() == 32 &&
                    manifest["dtype"].as<std::string>() == "float32" &&
                    manifest["observation_order"].as<std::string>() == "term-major, oldest-to-newest" &&
                    manifest["camera_preprocessing_version"].as<std::string>() == "parkour_depth_v2" &&
                    manifest["policy_joint_names"].as<std::vector<std::string>>() == names &&
                    manifest["default_position"].as<std::vector<double>>() == policy.defaults &&
                    manifest["action_scale"].as<std::vector<double>>() == policy.scales &&
                    manifest["pd_kp"].as<std::vector<double>>() == p["kp"].as<std::vector<double>>() &&
                    manifest["pd_kd"].as<std::vector<double>>() == p["kd"].as<std::vector<double>>() &&
                    policy.history == 8 && policy.velocity_mask.empty(),
                    "Perception manifest/profile contract mismatch");
            require(policy.action_clip == 10 && manifest["output_action_clip"].IsNull(),
                    "Perception requires unclipped output and previous-action clip 10");
            policy.angular_velocity_scale = number(d, "angular_velocity_scale", 0.0001);
            policy.joint_velocity_scale = number(d, "joint_velocity_scale", 0.0001);
            policy.depth_timeout_ns = number(d, "depth_timeout", 0.001) * 1e9;
            policy.execution_timeout_ns = number(d, "inference_timeout", 0.001) * 1e9;
        }
        const auto inf = root["inference"];
        keys(inf, {"backend", "threads", "warmup_steps", "perception_execution", "encoder_hz"});
        policy.inference.backend = inf["backend"].as<std::string>();
        require(policy.inference.backend == "libtorch" || policy.inference.backend == "onnx",
                "Unknown inference backend");
        keys(p["models"], {"libtorch", "onnx"});
        policy.inference.model = (std::filesystem::path(policy_base) /
                                  p["models"][policy.inference.backend].as<std::string>())
                                     .lexically_normal()
                                     .string();
        policy.inference.threads = inf["threads"].as<int>();
        if (mode == Mode::Perception && (!inf["perception_execution"] ||
            inf["perception_execution"].as<std::string>() != "split")) {
            const auto hash_key = policy.inference.backend == "onnx" ? "onnx_sha256" : "model_sha256";
            require(sha256(policy.inference.model) == perception_manifest[hash_key].as<std::string>(),
                    "Perception model SHA256 differs from manifest");
        }
        if (mode != Mode::Perception) {
            const auto manifest_path = p["contract"] ?
                std::filesystem::path(policy_base) / p["contract"].as<std::string>() :
                std::filesystem::path(policy.inference.model).parent_path() / "model_contract.yaml";
            policy.contract_path = manifest_path.lexically_normal().string();
            require(std::filesystem::is_regular_file(manifest_path),
                    "Missing model contract: " + manifest_path.string());
            const auto manifest = YAML::LoadFile(manifest_path.string());
            require(policy.mimic_variant == manifest["mimic_variant"].as<std::string>(""),
                    "MIMIC attachment differs from model contract");
            require(bool(p["output_mode"]) == bool(manifest["output_mode"]) &&
                    (!p["output_mode"] || p["output_mode"].as<std::string>() == manifest["output_mode"].as<std::string>()),
                    "Policy output mode differs from contract");
            require(manifest["robot"].as<std::string>() == policy.robot &&
                    manifest["kind"].as<std::string>() == policy.kind &&
                    manifest["joints"].as<std::vector<std::string>>() == names &&
                    manifest["defaults"].as<std::vector<double>>() == policy.defaults &&
                    manifest["scales"].as<std::vector<double>>() == policy.scales &&
                    manifest["history"].as<std::size_t>() == policy.history &&
                    manifest["velocity_mask"].as<std::vector<std::string>>() ==
                        p["velocity_mask"].as<std::vector<std::string>>() &&
                    manifest["observation_clip"].as<double>() == policy.observation_clip &&
                    manifest["action_clip"].as<double>() == policy.action_clip &&
                    manifest["step_seconds"].as<double>() == policy.step_seconds,
                    "Model observation/action contract mismatch");
            require(sha256(policy.inference.model) ==
                    manifest["models"][policy.inference.backend]["sha256"].as<std::string>(),
                    "Policy model SHA256 differs from contract");
            require(bool(manifest["amp_gait"]) == policy.amp_gait.enabled,
                    "AMP phase contract mismatch");
            if (policy.amp_gait.enabled) {
                const auto a = p["amp_gait"], b = manifest["amp_gait"];
                require(a.size() == b.size(), "AMP phase contract mismatch");
                for (const auto &item : a) {
                    const auto key = item.first.as<std::string>();
                    bool equal = false;
                    if (key == "term_clips") equal = item.second.as<std::vector<double>>() == b[key].as<std::vector<double>>();
                    else if (key == "default_left") equal = item.second.as<bool>() == b[key].as<bool>();
                    else equal = item.second.as<double>() == b[key].as<double>();
                    require(equal, "AMP phase contract mismatch: " + key);
                }
            }
            if (mode != Mode::Walk) {
                require(manifest["mimic"]["tracked_joints"].as<std::vector<std::string>>() ==
                            p["mimic"]["tracked_joints"].as<std::vector<std::string>>() &&
                        manifest["mimic"]["torso_joints"].as<std::vector<std::string>>() ==
                            p["mimic"]["torso_joints"].as<std::vector<std::string>>() &&
                        sha256(policy.mimic.motion) == manifest["mimic"]["motion_sha256"].as<std::string>(),
                        "MIMIC reference/joint contract mismatch");
                if (mode > Mode::MimicE) {
                    require(manifest["mode"].as<int>() == wire_mode && p["entry"] &&
                            manifest["entry"]["from_modes"].as<std::vector<int>>() ==
                                p["entry"]["from_modes"].as<std::vector<int>>() &&
                            manifest["entry"]["require_motion_complete"].as<bool>() == gate.require_motion_complete,
                            "Special MIMIC mode/entry contract mismatch");
                    const auto h = manifest["mimic"]["heading"];
                    require((h && !h.IsNull()) == policy.mimic.heading.projected,
                            "MIMIC heading contract mismatch");
                    if (policy.mimic.heading.projected) {
                        require(h["mode"].as<std::string>() == "projected_body_axis" &&
                                h["min_horizontal_projection"].as<double>() == gate.heading.min_horizontal_projection &&
                                h["max_angular_velocity"].as<double>() == gate.heading.max_angular_velocity &&
                                h["max_residual_angle"].as<double>() == gate.heading.max_residual_angle,
                                "MIMIC heading contract mismatch");
                    }
                }
            }
        }
        policy.inference.warmup_steps = inf["warmup_steps"].as<int>();
        policy.inference.input_size =
            (mode == Mode::Walk || mode == Mode::Perception ? 9 + 3 * names.size() +
                 (policy.amp_gait.enabled ? 2 : 0) : 21 + 5 * names.size()) * policy.history +
            (mode == Mode::Perception ? 4608 : 0);
        policy.inference.output_size = names.size();
        if (mode == Mode::Perception) {
            const auto execution = inf["perception_execution"] ?
                inf["perception_execution"].as<std::string>() : "combined";
            require(execution == "combined" || execution == "split",
                    "perception_execution must be combined or split");
            policy.split_perception = execution == "split";
            if (policy.split_perception) {
                require(perception_manifest["latent_dim"].as<int>() == 128,
                        "Split perception requires 128 visual features");
                policy.encoder_hz = inf["encoder_hz"] ? number(inf, "encoder_hz", 1) : 50;
                require(policy.encoder_hz <= 50 && 1e9 / policy.encoder_hz < policy.depth_timeout_ns,
                        "Encoder frequency exceeds 50 Hz or cannot satisfy the feature age budget");
                const auto component = [&](const char* name, std::size_t input, std::size_t output) {
                    auto config = policy.inference;
                    const auto resource = perception_manifest["split_models"][name][config.backend];
                    require(resource && resource["file"] && resource["sha256"],
                            "Split model resources missing from perception manifest");
                    config.model = (std::filesystem::path(policy_base) /
                        resource["file"].as<std::string>()).lexically_normal().string();
                    require(sha256(config.model) == resource["sha256"].as<std::string>(),
                            std::string("Split ") + name + " SHA256 differs from manifest");
                    config.input_size = input;
                    config.output_size = output;
                    return config;
                };
                policy.encoder = component("encoder", 4608, 128);
                policy.encoder.input_shape = {1, 8, 18, 32};
                policy.actor = component("actor", 752, 23);
            } else require(!inf["encoder_hz"], "encoder_hz only applies to split perception");
        }
        for (std::size_t i = 0; i < n; ++i)
            require(policy.kp[i] >= 0 && policy.kd[i] >= 0 && policy.motor_kd[i] >= 0,
                    "Invalid policy gains");
        const std::string identity=std::to_string(wire_mode)+":"+policy.kind+":"+policy.inference.backend;
        const auto model_digest=policy.split_perception ?
            sha256(policy.encoder.model)+":"+sha256(policy.actor.model) : sha256(policy.inference.model);
        c.runtime.recorder_controllers.push_back(identity+":"+
            (policy.split_perception?"split:":"combined:")+model_digest);
        recording_recipe+=sha256(policy_path)+model_digest;
        if (!policy.contract_path.empty()) recording_recipe+=sha256(policy.contract_path);
        if(isMimic(mode)) recording_recipe+=sha256(policy.mimic.motion);
        c.policies.emplace(mode, std::move(policy));
        c.control.policy_mask |= uint64_t{1} << wire_mode;
    }
    for (const auto &entry : c.control.entries)
        require(!(entry.from_modes & ~(c.control.policy_mask | (uint64_t{1} << unsigned(Mode::Ready)))),
                "Policy entry source is not enabled");
    require(c.policies.count(Mode::Walk), "AMP WALK profile is required for velocity limits");
    if (c.runtime.backend == loco::BackendType::Hardware) {
        keys(rt, {"backend", "hardware_config", "motor_slots", "encoder_names", "damping_kd", "damping_kd_overrides", "external_depth", "hand_control"});
        if (const auto hand = rt["hand_control"]) {
            const bool enabled = hand.as<bool>();
            require(!enabled || c.hand.enabled, "Hand control requires an enabled robot hand profile");
            c.hand.enabled = enabled;
            c.runtime.hand_control = enabled;
        }
        c.runtime.hardware_config = path(rt["hardware_config"]);
        c.runtime.suspend = false;
        c.runtime.headless = true;
        loadHardware(c, rt);
    } else {
        keys(rt, {"backend", "model", "headless", "suspend", "release_after_ready", "damping_kd", "damping_kd_overrides", "depth", "external_depth"});
        c.runtime.model_path = path(rt["model"]);
        c.runtime.model_name = c.robot == "pro" ? "aiq11_pro" : "aiq11";
        c.runtime.headless = rt["headless"].as<bool>();
        c.runtime.suspend = rt["suspend"].as<bool>();
        if (rt["release_after_ready"])
            c.release_after_ready = rt["release_after_ready"].as<bool>();
    }
    c.runtime.damping_kd = number(rt, "damping_kd", 0.01);
    if (const auto overrides = rt["damping_kd_overrides"]) {
        require(overrides.IsMap(), "damping_kd_overrides must map motor names to gains");
        c.runtime.motor_damping_kd.assign(c.runtime.motor_names.size(), c.runtime.damping_kd);
        std::set<std::string> seen;
        for (const auto &entry : overrides) {
            const auto name = entry.first.as<std::string>();
            const auto motor = std::find(c.runtime.motor_names.begin(), c.runtime.motor_names.end(), name);
            require(motor != c.runtime.motor_names.end() && seen.insert(name).second,
                    "Unknown or duplicate damping motor: " + name);
            c.runtime.motor_damping_kd[motor - c.runtime.motor_names.begin()] =
                number(overrides, name.c_str(), 0.01);
        }
    }
    if (const auto d = rt["depth"]) {
        keys(d, {"site", "frequency_hz", "min_distance", "max_distance",
                 "sensor_min_distance", "sensor_max_distance",
                 "horizontal_fov_deg", "vertical_fov_deg", "preprocessing_version", "raw_width", "raw_height",
                 "crop_up", "crop_down", "crop_left", "crop_right"});
        c.runtime.depth.enabled = true;
        c.runtime.depth.site = d["site"].as<std::string>();
        c.runtime.depth.preprocessing_version = d["preprocessing_version"].as<std::string>();
        c.runtime.depth.raw_width = d["raw_width"].as<int>();
        c.runtime.depth.raw_height = d["raw_height"].as<int>();
        c.runtime.depth.crop_up = d["crop_up"].as<int>();
        c.runtime.depth.crop_down = d["crop_down"].as<int>();
        c.runtime.depth.crop_left = d["crop_left"].as<int>();
        c.runtime.depth.crop_right = d["crop_right"].as<int>();
        c.runtime.depth.frequency_hz = d["frequency_hz"].as<int>();
        c.runtime.depth.min_distance = number(d, "min_distance", 0);
        c.runtime.depth.max_distance = number(d, "max_distance", c.runtime.depth.min_distance);
        if (d["sensor_min_distance"])
            c.runtime.depth.sensor_min_distance = number(d, "sensor_min_distance", 0);
        if (d["sensor_max_distance"])
            c.runtime.depth.sensor_max_distance = number(d, "sensor_max_distance", 0);
        require(c.runtime.depth.sensor_max_distance > c.runtime.depth.sensor_min_distance,
                "Sensor maximum depth must exceed minimum depth");
        c.runtime.depth.horizontal_fov_deg = number(d, "horizontal_fov_deg", 0.001);
        c.runtime.depth.vertical_fov_deg = number(d, "vertical_fov_deg", 0.001);
    }
    if (const auto d = rt["external_depth"]) {
        keys(d, {"topic", "frame_id", "timeout", "max_body_skew", "clock_tolerance"});
        c.external_depth.topic = d["topic"].as<std::string>();
        c.external_depth.frame_id = d["frame_id"].as<std::string>();
        const auto timeout = number(d, "timeout", 0.001);
        const auto skew = number(d, "max_body_skew", 0.001);
        const auto tolerance = number(d, "clock_tolerance", 0.000001);
        require(timeout <= 1.0 && skew <= timeout && tolerance <= skew,
                "External depth deadlines exceed the policy contract");
        c.external_depth.timeout_ns = uint64_t(timeout * 1e9);
        c.external_depth.max_skew_ns = uint64_t(skew * 1e9);
        c.external_depth.clock_tolerance_ns = uint64_t(tolerance * 1e9);
        require(!c.external_depth.topic.empty() && !c.external_depth.frame_id.empty(),
                "External depth topic/frame_id must not be empty");
        require(!c.runtime.depth.enabled, "Choose simulation depth OR external depth, not both");
        require(c.policies.count(Mode::Perception), "External depth requires a perception policy");
        require(c.external_depth.timeout_ns <= c.policies.at(Mode::Perception).depth_timeout_ns &&
                    c.external_depth.max_skew_ns <= c.external_depth.timeout_ns &&
                    c.external_depth.clock_tolerance_ns <= c.external_depth.max_skew_ns,
                "External depth deadlines exceed the policy contract");
    }
    if (c.policies.count(Mode::Perception) && c.external_depth.topic.empty())
        require(c.runtime.depth.enabled && c.runtime.depth.preprocessing_version == "parkour_depth_v2" &&
                    c.runtime.depth.frequency_hz == 50 &&
                    c.runtime.depth.max_distance == 2.5,
                "parkour_depth_v2 policy requires 50 Hz depth normalized by 2.5 m");
    const auto t = root["timing"];
    keys(t, {"control_hz", "io_hz", "inference_hz", "ros_hz", "feedback_timeout",
             "inference_timeout", "first_inference_timeout", "command_timeout", "velocity_timeout",
             "request_timeout", "limit_warning_period", "policy_transition_seconds"});
    c.control.policy_transition_seconds = number(t, "policy_transition_seconds", 0);
    require(c.control.policy_transition_seconds <= 2, "Policy transition exceeds two seconds");
    c.control_hz = number(t, "control_hz", 1);
    c.io_hz = number(t, "io_hz", 1);
    c.inference_hz = number(t, "inference_hz", 1);
    c.ros_hz = number(t, "ros_hz", 1);
    double slowest_step = 1.0 / c.inference_hz;
    for (const auto &[mode, policy] : c.policies) {
        require(policy.step_seconds * c.inference_hz >= 1 - 1e-6,
                "Policy rate exceeds inference_hz");
        slowest_step = std::max(slowest_step, policy.step_seconds);
    }
    require(c.control_hz <= 2000 && c.io_hz <= 2000 && c.ros_hz <= 1000 &&
                c.inference_hz <= c.control_hz,
            "Invalid worker rates");
    c.command_timeout = number(t, "command_timeout", 2.0 / c.control_hz);
    c.velocity_timeout = number(t, "velocity_timeout", 0.01);
    c.request_timeout = number(t, "request_timeout", 0.01);
    if (t["limit_warning_period"])
        c.limit_warning_period = number(t, "limit_warning_period", 0.1);
    c.control.feedback_timeout_ns = number(t, "feedback_timeout", 2.0 / c.io_hz) * 1e9;
    c.control.inference_timeout_ns = number(t, "inference_timeout", 2.0 * slowest_step) * 1e9;
    c.control.first_inference_timeout_ns =
        number(t, "first_inference_timeout", 2.0 * slowest_step) * 1e9;
    c.runtime.feedback_timeout = std::chrono::milliseconds(c.control.feedback_timeout_ns / 1000000);
    c.runtime.command_timeout =
        std::chrono::milliseconds(static_cast<int>(c.command_timeout * 1000));
    const auto th = root["threads"];
    keys(th, {"control_cpu", "control_priority", "inference_cpu"});
    c.control_cpu = th["control_cpu"].as<int>();
    c.control_priority = th["control_priority"].as<int>();
    c.inference_cpu = th["inference_cpu"].as<int>();
    require(c.control_cpu >= -1 && c.inference_cpu >= -1 && c.control_priority >= 0 &&
                c.control_priority <= 99,
            "Invalid thread settings");
    const auto remote = root["remote"];
    keys(remote, {"source", "velocity_topic", "mode_service", "raw_topic", "sbus", "gui"});
    c.remote_gui = remote["gui"] ? remote["gui"].as<bool>() : true;
    c.remote_source = remote["source"].as<std::string>();
    require(c.remote_source == "sbus" || c.remote_source == "external", "Invalid remote source");
    c.velocity_topic = remote["velocity_topic"].as<std::string>();
    c.mode_service = remote["mode_service"].as<std::string>();
    c.raw_topic = remote["raw_topic"].as<std::string>();
    c.sbus = remote["sbus"];
    const auto d = root["debug"];
    keys(d, {"enabled", "rate_hz", "runtime_topic", "joint_topic", "motor_topic", "hand_topic", "depth_image_topic"});
    if (d["depth_image_topic"]) {
        c.depth_image_topic = d["depth_image_topic"].as<std::string>();
        require(c.depth_image_topic.empty() || c.runtime.depth.enabled,
                "Depth image publication requires a depth camera");
    }
    c.debug = d;
    require(number(d, "rate_hz", 0.1) <= c.ros_hz, "Debug rate exceeds ROS worker rate");
    require(!c.velocity_topic.empty() && !c.mode_service.empty() && !c.raw_topic.empty(),
            "Empty ROS interface name");
    for (std::size_t i = 0; i < n; ++i) {
        require(c.control.lower[i] < c.control.upper[i], "Invalid position bounds");
        require(c.control.ready[i] >= c.control.lower[i] &&
                    c.control.ready[i] <= c.control.upper[i],
                "READY outside bounds");
        require(hold[i] >= c.control.lower[i] && hold[i] <= c.control.upper[i],
                "Hold outside bounds");
        require(c.control.ready_kp[i] >= 0 && c.control.ready_kd[i] >= 0 && hold_kp[i] >= 0 &&
                    hold_kd[i] >= 0 && c.motor_kd[i] >= 0 && c.torque_limit[i] > 0,
                "Invalid gains/torque limit");
    }
    if (root["motion_stream"]) {
        c.motion = loadMotionConfig(path(root["motion_stream"]), c.robot, c.runtime.joint_names, c.control);
        require(c.motion.service != c.mode_service, "Motion and mode services must differ");
        require(c.motion.dt_cap >= 1.0 / c.control_hz, "Motion dt_cap below control period");
    }
    if (c.mimic_variant == "auto") {
        const auto other = loadConfig(file, "gripper");
        for (const auto &[mode, policy] : other.policies) {
            if (mode < Mode::MimicA || mode > Mode::MimicE) continue;
            const auto &fist = c.policies.at(mode);
            require(fist.indices == policy.indices && fist.step_seconds == policy.step_seconds &&
                    fist.mimic_variant == "fist" && policy.mimic_variant == "gripper",
                    "MIMIC variants must share topology and policy frequency");
            c.gripper_policies.emplace(mode, policy);
        }
        require(!c.gripper_policies.empty(), "Auto MIMIC requires enabled PRO A-E profiles");
        recording_recipe += other.runtime.recorder_config_digest;
    }
    if(c.runtime.flight_recorder.enabled) {
        require(n<=32 && c.pairs.size()<=4,"Controller recording supports verified 32-axis / 4-mechanism layouts");
        for(const auto &pair:c.pairs) for(const auto *name:
            {"ang_search_param.txt","fit_pitch_ploy33.txt","fit_roll_ploy33.txt","pitch.txt","roll.txt","pitch_roll_rge.txt"})
            recording_recipe+=sha256((std::filesystem::path(pair.resources)/name).string());
        if(root["motion_stream"]) {
            const auto motion_path=path(root["motion_stream"]);
            recording_recipe+=sha256(motion_path);
            const auto motion=YAML::LoadFile(motion_path);
            for(const auto &m:motion["motions"])
                recording_recipe+=sha256((std::filesystem::path(motion_path).parent_path()/m["file"].as<std::string>()).string());
        }
        std::array<unsigned char,32> digest{};unsigned length=0;
        require(EVP_Digest(recording_recipe.data(),recording_recipe.size(),digest.data(),&length,EVP_sha256(),nullptr)==1 && length==32,
                "Cannot hash effective recorder configuration");
        std::ostringstream text;text<<std::hex<<std::setfill('0');
        for(auto b:digest) text<<std::setw(2)<<unsigned(b);
        c.runtime.recorder_config_digest=text.str();
        c.runtime.recorder_controllers.push_back("application:" EIR_CTRL_REVISION);
    }
    return c;
}
Config recordingConfig(Config c) {
    if(!c.runtime.flight_recorder.enabled) return c;
    std::ostringstream effective;effective<<std::setprecision(17)<<c.runtime.recorder_config_digest;
    const auto &r=c.runtime;
    effective<<':'<<unsigned(r.backend)<<':'<<r.model_path<<':'<<r.hardware_config<<':'<<r.headless
        <<':'<<r.suspend<<':'<<r.passive_observation<<':'<<r.hand_control<<':'<<c.control.still_only
        <<':'<<c.control.policy_mask<<':'<<c.control_hz<<':'<<c.io_hz<<':'<<c.inference_hz
        <<':'<<c.control_cpu<<':'<<c.control_priority<<':'<<c.inference_cpu
        <<':'<<c.external_depth.topic<<':'<<c.external_depth.frame_id<<':'<<c.external_depth.timeout_ns
        <<':'<<c.external_depth.max_skew_ns<<':'<<c.release_after_ready
        <<':'<<r.depth.enabled<<':'<<r.depth.preprocessing_version<<':'<<r.depth.site
        <<':'<<r.depth.frequency_hz<<':'<<r.depth.sensor_min_distance<<':'<<r.depth.sensor_max_distance;
    const auto &recorder=r.flight_recorder;
    effective<<':'<<recorder.enabled<<':'<<recorder.directory<<':'<<recorder.data
        <<':'<<recorder.detailed_hz<<':'<<recorder.application_hz<<':'<<recorder.continuous_hz
        <<':'<<recorder.pre_seconds<<':'<<recorder.post_seconds<<':'<<recorder.queue_frames
        <<':'<<recorder.max_pending<<':'<<recorder.max_memory_bytes<<':'<<recorder.max_storage_bytes;
    auto numbers=[&](const auto &v){for(const auto x:v)effective<<':'<<x;};
    numbers(c.control.ready);numbers(c.control.ready_kp);numbers(c.control.ready_kd);
    numbers(c.runtime.motor_damping_kd);numbers(c.torque_limit);
    c.runtime.recorder_controllers={"1:STILL","2:READY","255:FAULT","application:" EIR_CTRL_REVISION};
    std::string logical="logical_joint_names:";
    for(const auto &name:c.runtime.joint_names) {logical+=name;logical+=',';}
    c.runtime.recorder_controllers.push_back(logical);
    for(const auto &[mode,p]:c.policies) {
        const auto models=p.split_perception?sha256(p.encoder.model)+":"+sha256(p.actor.model):sha256(p.inference.model);
        c.runtime.recorder_controllers.push_back(std::to_string(unsigned(mode))+":"+p.kind+":"+
            p.inference.backend+":"+(p.split_perception?"split:":"combined:")+models);
        effective<<':'<<unsigned(mode)<<':'<<p.inference.backend<<':'<<p.split_perception<<':'<<models
            <<':'<<p.step_seconds<<':'<<p.observation_clip<<':'<<p.action_clip<<':'<<p.encoder_hz;
        numbers(p.indices);numbers(p.defaults);numbers(p.scales);numbers(p.kp);numbers(p.kd);numbers(p.motor_kd);
    }
    effective << ":mimic:" << c.mimic_variant << ':' << c.mimic_hand_type
              << ':' << c.mimic_stable_seconds << ':' << c.mimic_detection_seconds;
    c.runtime.recorder_controllers.push_back("mimic_a_e_variant:" + c.mimic_variant);
    for (const auto &[mode, p] : c.gripper_policies) {
        const auto digest = sha256(p.inference.model);
        c.runtime.recorder_controllers.push_back("gripper_candidate:" + std::to_string(unsigned(mode)) +
            ":" + p.inference.backend + ":" + digest);
        effective << ':' << unsigned(mode) << ':' << digest << ':' << sha256(p.mimic.motion);
    }
    const auto text=effective.str();std::array<unsigned char,32> digest{};unsigned length=0;
    require(EVP_Digest(text.data(),text.size(),digest.data(),&length,EVP_sha256(),nullptr)==1 && length==32,
            "Cannot hash effective runtime recording configuration");
    std::ostringstream hex;hex<<std::hex<<std::setfill('0');for(auto b:digest)hex<<std::setw(2)<<unsigned(b);
    c.runtime.recorder_config_digest=hex.str();return c;
}
} // namespace eir::ctrl
