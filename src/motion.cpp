#include <eir_ctrl/motion.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

namespace eir::ctrl {
namespace {
void require(bool ok, const std::string &message) {
    if (!ok) throw std::invalid_argument("Motion stream: " + message);
}
void keys(const YAML::Node &n, std::initializer_list<const char *> allowed) {
    require(n.IsMap(), "expected a mapping");
    for (const auto &pair : n) {
        auto key = pair.first.as<std::string>();
        require(std::any_of(allowed.begin(), allowed.end(), [&](auto x) { return key == x; }),
                "unknown key: " + key);
    }
}
double positive(const YAML::Node &n) {
    const auto x = n.as<double>();
    require(std::isfinite(x) && x > 0, "expected a positive finite number");
    return x;
}
std::vector<std::string> columns(const std::string &line) {
    std::vector<std::string> out;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) {
        const auto first = field.find_first_not_of(" \t\r");
        out.push_back(first == std::string::npos ? "" :
                      field.substr(first, field.find_last_not_of(" \t\r") - first + 1));
    }
    return out;
}
double numeric(const std::string &text) {
    std::size_t used{};
    const auto x = std::stod(text, &used);
    require(used == text.size() && std::isfinite(x), "invalid CSV number: " + text);
    return x;
}
} // namespace
MotionConfig loadMotionConfig(const std::string &file, const std::string &robot,
                             const std::vector<std::string> &joints, const ControlConfig &control) {
    auto n = YAML::LoadFile(file);
    keys(n, {"robot", "service", "blend_seconds", "recovery_warning_seconds", "recovery_inferences",
             "dt_cap", "groups", "motions"});
    require(n["robot"].as<std::string>() == robot, "robot identity mismatch");
    MotionConfig c;
    c.service = n["service"].as<std::string>();
    require(!c.service.empty(), "empty service name");
    c.blend_seconds = positive(n["blend_seconds"]);
    c.recovery_warning_seconds = positive(n["recovery_warning_seconds"]);
    c.dt_cap = positive(n["dt_cap"]);
    require(c.dt_cap <= 0.02, "dt_cap must be <= 0.02 s");
    const auto count = n["recovery_inferences"].as<int>();
    require(count > 0 && count <= 100, "invalid recovery inference count");
    c.recovery_inferences = count;
    c.kp = control.ready_kp;
    c.kd = control.ready_kd;
    auto joint = [&](const std::string &name) {
        const auto it = std::find(joints.begin(), joints.end(), name);
        require(it != joints.end(), "unknown joint: " + name);
        // This identity contract prevents configuration from taking over a leg.
        require(name.rfind("torso_", 0) == 0 || name.rfind("l_arm_", 0) == 0 ||
                    name.rfind("r_arm_", 0) == 0 || name.rfind("neck_", 0) == 0,
                "only torso, arms and neck are allowed: " + name);
        return std::size_t(it - joints.begin());
    };
    uint64_t allowed = 0;
    require(n["groups"].IsSequence() && n["groups"].size() > 0, "groups must be nonempty");
    for (const auto &g : n["groups"]) {
        keys(g, {"joints", "observation_position", "max_velocity"});
        auto names = g["joints"].as<std::vector<std::string>>();
        auto positions = g["observation_position"].as<std::vector<double>>();
        require(!names.empty() && names.size() == positions.size(), "invalid observation group");
        const auto speed = positive(g["max_velocity"]);
        uint64_t mask = 0;
        for (std::size_t k = 0; k < names.size(); ++k) {
            const auto i = joint(names[k]), bit = uint64_t{1} << i;
            require(!(allowed & bit), "duplicate grouped joint: " + names[k]);
            require(std::isfinite(positions[k]) && positions[k] >= control.lower[i] &&
                        positions[k] <= control.upper[i], "invalid observation posture: " + names[k]);
            c.observation_position[i] = positions[k];
            c.max_velocity[i] = speed;
            allowed |= bit;
            mask |= bit;
        }
        c.observation_groups.push_back(mask);
    }
    const auto base = std::filesystem::absolute(file).parent_path();
    std::set<int> commands;
    require(n["motions"].IsSequence() && n["motions"].size() > 0 && n["motions"].size() <= 245,
            "expected 1..245 motion entries");
    for (const auto &entry : n["motions"]) {
        keys(entry, {"cmd", "name", "file"});
        const auto cmd = entry["cmd"].as<int>();
        require(cmd >= 11 && cmd <= 255 && commands.insert(cmd).second, "invalid/duplicate command");
        OfflineMotion m;
        m.cmd = cmd;
        m.name = entry["name"].as<std::string>();
        require(!m.name.empty(), "empty motion name");
        auto path = base / entry["file"].as<std::string>();
        std::ifstream stream(path);
        require(bool(stream), "cannot open " + path.string());
        std::string line;
        require(bool(std::getline(stream, line)), "empty CSV");
        auto header = columns(line);
        require(header.size() > 1 && header[0] == "time", "CSV header must be time,joint,...");
        std::vector<std::size_t> indices;
        const std::array<std::string, 12> hand_names{"l_fg1_1", "l_fg2_0", "l_fg2_1", "l_fg3_0", "l_fg3_1", "",
                                                   "r_fg1_1", "r_fg2_0", "r_fg2_1", "r_fg3_0", "r_fg3_1", ""};
        for (std::size_t k = 1; k < header.size(); ++k) {
            const auto hand = std::find(hand_names.begin(), hand_names.end(), header[k]);
            if (!header[k].empty() && hand != hand_names.end()) {
                const auto i = std::size_t(hand - hand_names.begin());
                require(robot == "pro" && !(m.hand_mask & (uint16_t{1} << i)),
                        "unsupported/duplicate CSV finger: " + header[k]);
                m.hand_mask |= uint16_t{1} << i;
                indices.push_back(capacity + i);
                continue;
            }
            const auto i = joint(header[k]), bit = uint64_t{1} << i;
            require((allowed & bit) && !(m.mask & bit), "ungrouped/duplicate CSV joint: " + header[k]);
            m.mask |= bit;
            indices.push_back(i);
        }
        while (std::getline(stream, line)) {
            require(m.frames.size() < 100000, "CSV exceeds 100000 frames");
            const auto fields = columns(line);
            require(fields.size() == header.size(), "wrong CSV column count: " + path.string());
            MotionFrame frame;
            frame.time = numeric(fields[0]);
            require(m.frames.empty() ? frame.time == 0 : frame.time > m.frames.back().time,
                    "CSV time must start at zero and increase strictly");
            for (std::size_t k = 0; k < indices.size(); ++k) {
                const auto i = indices[k];
                if (i >= capacity) {
                    frame.hand[i - capacity] = numeric(fields[k + 1]);
                    require(frame.hand[i - capacity] >= 0 && frame.hand[i - capacity] <= 1,
                            "CSV finger closure outside [0,1]: " + header[k + 1]);
                    continue;
                }
                frame.q[i] = numeric(fields[k + 1]);
                require(frame.q[i] >= control.lower[i] && frame.q[i] <= control.upper[i],
                        "CSV target outside joint limits: " + header[k + 1]);
            }
            m.frames.push_back(frame);
        }
        require(stream.eof() && m.frames.size() >= 2, "incomplete trajectory");
        c.motions.push_back(std::move(m));
    }
    return c;
}

void MotionPlayer::cancel() noexcept {
    if (phase_ != MotionPhase::Idle) ++generation_;
    phase_ = MotionPhase::Idle;
    motion_ = nullptr;
    mask_ = 0;
    elapsed_ = blend_ = 0;
    last_at_ = 0;
}
void MotionPlayer::recover(uint64_t now) noexcept {
    phase_ = MotionPhase::Recovering;
    recovery_at_ = now;
    fresh_inferences_ = 0;
    inference_stamp_ = 0;
    hold_ = last_.q;
    ++generation_; // In-flight masked policy results cannot satisfy recovery.
}
uint64_t MotionPlayer::observationMask() const noexcept {
    if (phase_ != MotionPhase::Playing) return 0;
    uint64_t mask = 0;
    for (auto group : config_.observation_groups)
        if (group & mask_) mask |= group;
    return mask;
}
MotionCompletion MotionPlayer::request(MotionRequest r, const Command &cmd, uint64_t now) noexcept {
    if (r.cmd != 2 && r.cmd < 11) return {r.id, 1};
    const OfflineMotion *selected = nullptr;
    if (r.cmd != 2) {
        for (const auto &m : config_.motions)
            if (m.cmd == r.cmd) { selected = &m; break; }
        if (!selected) return {r.id, 2};
    }
    if (r.deadline < now || cmd.mode != Mode::Walk || !cmd.policy_target ||
        (phase_ != MotionPhase::Idle && session_ != cmd.session)) return {r.id, 3};
    if (r.cmd == 2) {
        if (phase_ == MotionPhase::Playing) recover(now);
        return {r.id, 0};
    }
    // Restart from the last composite command, including already blended gains.
    start_ = last_.policy_target && last_.session == cmd.session ? last_ : cmd;
    for (std::size_t i = 0; i < capacity; ++i)
        if (!(start_.gain_mask & (uint64_t{1} << i))) {
            start_.kp[i] = policy_.kp[i];
            start_.kd[i] = policy_.kd[i];
        }
    mask_ |= selected->mask;
    motion_ = selected;
    session_ = cmd.session;
    phase_ = MotionPhase::Playing;
    elapsed_ = blend_ = 0;
    last_at_ = now;
    last_ = start_;
    hold_ = start_.q;
    ++generation_;
    return {r.id, 0};
}
void MotionPlayer::step(Command &cmd, const Target &target, uint64_t now) noexcept {
    cmd.hand_mask = 0;
    if (config_.motions.empty()) return;
    if (cmd.mode != Mode::Walk || (phase_ != MotionPhase::Idle && session_ != cmd.session)) {
        cancel();
        return;
    }
    if (phase_ == MotionPhase::Idle) { last_ = cmd; return; }
    if (!cmd.policy_target) { cancel(); return; }
    const auto dt = last_at_ && now >= last_at_
                        ? std::min((now - last_at_) * 1e-9, config_.dt_cap) : 0;
    last_at_ = now;
    if (phase_ == MotionPhase::Recovering) {
        if (target.valid && target.session == session_ && target.motion_generation == generation_ &&
            target.source_stamp >= recovery_at_ && target.inference_sequence > inference_stamp_) {
            inference_stamp_ = target.inference_sequence;
            ++fresh_inferences_;
        }
        if (fresh_inferences_ >= config_.recovery_inferences) {
            phase_ = MotionPhase::Returning;
            start_ = last_;
            blend_ = 0;
        }
    }
    Values desired = hold_;
    if (phase_ == MotionPhase::Playing) {
        // Binary search over a startup-validated, bounded immutable trajectory.
        const auto &frames = motion_->frames;
        const auto upper = std::upper_bound(frames.begin(), frames.end(), elapsed_,
                                           [](double t, const MotionFrame &f) { return t < f.time; });
        const auto &b = upper == frames.end() ? frames.back() : *upper;
        const auto &a = upper == frames.begin() ? b : *(upper - 1);
        const auto ratio = b.time > a.time ? (elapsed_ - a.time) / (b.time - a.time) : 0;
        for (std::size_t i = 0; i < capacity; ++i)
            if (motion_->mask & (uint64_t{1} << i)) desired[i] = a.q[i] + ratio * (b.q[i] - a.q[i]);
        cmd.hand_mask = motion_->hand_mask;
        for (std::size_t i = 0; i < 12; ++i)
            if (cmd.hand_mask & (uint16_t{1} << i))
                cmd.hand[i] = a.hand[i] + ratio * (b.hand[i] - a.hand[i]);
    } else if (phase_ == MotionPhase::Returning) desired = cmd.q;
    if (phase_ == MotionPhase::Playing || phase_ == MotionPhase::Returning)
        blend_ = std::min(1.0, blend_ + dt / config_.blend_seconds);
    const auto alpha = blend_ * blend_ * (3 - 2 * blend_);
    bool returned = phase_ == MotionPhase::Returning && blend_ >= 1;
    for (std::size_t i = 0; i < capacity; ++i) {
        const auto bit = uint64_t{1} << i;
        if (!(mask_ & bit)) continue;
        const auto q = phase_ == MotionPhase::Recovering ? hold_[i] :
                           start_.q[i] + alpha * (desired[i] - start_.q[i]);
        cmd.q[i] = last_.q[i] + std::clamp(q - last_.q[i], -config_.max_velocity[i] * dt,
                                                                config_.max_velocity[i] * dt);
        const bool returning = phase_ == MotionPhase::Returning;
        if (returning) returned = returned && std::abs(cmd.q[i] - desired[i]) < 1e-9;
        cmd.dq[i] = returning ? alpha * cmd.dq[i] : 0; // Reference offline actions use position PD.
        cmd.kp[i] = phase_ == MotionPhase::Recovering ? last_.kp[i] :
                       start_.kp[i] + alpha * ((returning ? policy_.kp[i] : config_.kp[i]) - start_.kp[i]);
        cmd.kd[i] = phase_ == MotionPhase::Recovering ? last_.kd[i] :
                       start_.kd[i] + alpha * ((returning ? policy_.kd[i] : config_.kd[i]) - start_.kd[i]);
        cmd.gain_mask |= bit;
    }
    last_ = cmd;
    if (phase_ == MotionPhase::Playing) {
        if (elapsed_ >= motion_->frames.back().time) recover(now);
        else elapsed_ = std::min(motion_->frames.back().time, elapsed_ + dt);
    }
    if (returned) cancel();
}
} // namespace eir::ctrl
