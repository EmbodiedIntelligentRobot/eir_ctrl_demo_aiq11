#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <eir_ctrl/control.hpp>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace eir::ctrl {
bool alignMimicHeading(const MimicHeadingConfig &c, const Feedback &f,
                       std::array<double, 4> &yaw) noexcept {
    Eigen::Quaterniond reference(c.reference[0], c.reference[1], c.reference[2], c.reference[3]);
    Eigen::Quaterniond current(f.orientation[0], f.orientation[1], f.orientation[2], f.orientation[3]);
    const Eigen::Vector3d omega(f.omega[0], f.omega[1], f.omega[2]);
    if (!reference.coeffs().allFinite() || !std::isfinite(reference.norm()) || reference.norm() < 1e-6 ||
        !current.coeffs().allFinite() || !std::isfinite(current.norm()) || current.norm() < 1e-6 || !omega.allFinite() ||
        omega.norm() > c.max_angular_velocity)
        return false;
    reference.normalize(); current.normalize();
    const auto r = reference.toRotationMatrix(), q = current.toRotationMatrix();
    int axis = 0;
    for (int i = 1; i < 3; ++i)
        if (r.block<2, 1>(0, i).squaredNorm() > r.block<2, 1>(0, axis).squaredNorm()) axis = i;
    const Eigen::Vector2d a = r.block<2, 1>(0, axis), b = q.block<2, 1>(0, axis);
    if (a.norm() < c.min_horizontal_projection || b.norm() < c.min_horizontal_projection)
        return false;
    const double angle = std::atan2(b.x() * a.y() - b.y() * a.x(), a.dot(b));
    const Eigen::Quaterniond offset(Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ()));
    const auto residual = (offset * current).conjugate() * reference;
    if (2 * std::acos(std::clamp(std::abs(residual.w()), 0.0, 1.0)) > c.max_residual_angle)
        return false;
    yaw = {offset.w(), offset.x(), offset.y(), offset.z()};
    return true;
}
namespace {
Eigen::Quaterniond quaternion(const double *q) {
    Eigen::Quaterniond value(q[0], q[1], q[2], q[3]);
    if (!value.coeffs().allFinite() || value.norm() < 1e-6)
        throw std::runtime_error("Invalid MIMIC orientation");
    return value.normalized();
}
Eigen::Matrix3d torso(const Eigen::Quaterniond &base, double yaw, double roll, double pitch) {
    return (base * Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()) *
            Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()))
        .toRotationMatrix();
}
class Mimic final : public Policy {
    PolicyConfig c_;
    std::unique_ptr<Inference> inference_;
    std::vector<std::vector<double>> motion_;
    std::vector<float> history_, last_;
    std::vector<std::size_t> groups_;
    std::size_t frame_{}, steps_{}, hold_steps_{};
    bool initialized_{}, align_{true};
    Eigen::Quaterniond yaw_{Eigen::Quaterniond::Identity()};

  public:
    Mimic(PolicyConfig c, std::unique_ptr<Inference> inference)
        : c_(std::move(c)), inference_(std::move(inference)), history_(c_.inference.input_size),
          last_(c_.indices.size()),
          groups_{2 * c_.indices.size(), 6, 3, 3, 3, 3, 3, c_.indices.size(), c_.indices.size(),
                  c_.indices.size()} {
        if (!std::isfinite(c_.step_seconds) || c_.step_seconds <= 0)
            throw std::invalid_argument("Invalid MIMIC step_seconds");
        for (auto i : c_.mimic.torso)
            if (i >= c_.indices.size())
                throw std::invalid_argument("Invalid torso slot");
        hold_steps_ = std::llround(c_.mimic.first_frame_hold_seconds / c_.step_seconds);
        const auto columns = 4 + 2 * (c_.indices.size() + c_.mimic.tracked_indices.size());
        std::ifstream file(c_.mimic.motion);
        if (!file)
            throw std::runtime_error("Cannot open motion: " + c_.mimic.motion);
        std::string line;
        std::size_t row = 0;
        while (std::getline(file, line)) {
            ++row;
            if (line.find_first_not_of(" \t\r") == std::string::npos)
                continue;
            std::vector<double> values;
            std::istringstream stream(line);
            std::string cell;
            try {
                while (std::getline(stream, cell, ',')) {
                    std::size_t end{};
                    const auto value = std::stod(cell, &end);
                    if (!std::isfinite(value) ||
                        cell.find_first_not_of(" \t\r", end) != std::string::npos)
                        throw std::invalid_argument("Invalid number");
                    values.push_back(value);
                }
                if (values.size() != columns || line.back() == ',')
                    throw std::invalid_argument("Wrong column count");
                quaternion(values.data());
            } catch (const std::exception &e) {
                throw std::runtime_error(c_.mimic.motion + ":" + std::to_string(row) + ": " +
                                         e.what());
            }
            motion_.push_back(std::move(values));
        }
        if (!file.eof() || motion_.empty())
            throw std::runtime_error("Empty/unreadable MIMIC motion");
    }
    void reset() override {
        frame_ = steps_ = 0;
        initialized_ = false;
        align_ = true;
        std::fill(last_.begin(), last_.end(), 0);
    }
    Target step(const Observation &input) override {
        const auto &f = input.feedback;
        const auto &ref = motion_[frame_];
        const auto n = c_.indices.size(), joints = n + c_.mimic.tracked_indices.size();
        const bool freeze = steps_ < hold_steps_;
        const auto current = quaternion(f.orientation.data()), reference = quaternion(ref.data());
        if (align_) {
            if (c_.mimic.heading.projected) {
                // Use the accepted request's alignment: feedback can change before
                // the inference worker receives its first observation.
                auto yaw = input.entry_yaw;
                if (!input.entry_yaw_valid && !alignMimicHeading(c_.mimic.heading, f, yaw))
                    throw std::runtime_error("MIMIC initial posture does not match reference");
                yaw_ = quaternion(yaw.data());
            } else {
                const auto relative = current.conjugate() * reference;
                const auto yaw =
                    std::atan2(2 * (relative.w() * relative.z() + relative.x() * relative.y()),
                               1 - 2 * (relative.y() * relative.y() + relative.z() * relative.z()));
                yaw_ = Eigen::Quaterniond(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
            }
            align_ = false;
        }
        std::vector<float> observation;
        observation.reserve(21 + 5 * n);
        for (std::size_t i = 0; i < n; ++i)
            observation.push_back(ref[4 + i]);
        for (std::size_t i = 0; i < n; ++i)
            observation.push_back(freeze ? 0 : ref[4 + joints + i]);
        const auto relative = ((yaw_ * current).conjugate() * reference).toRotationMatrix();
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 2; ++col)
                observation.push_back(relative(row, col));
        auto gravity = [&](const Eigen::Matrix3d &rotation) {
            const Eigen::Vector3d g = rotation.transpose() * Eigen::Vector3d(0, 0, -1);
            for (int i = 0; i < 3; ++i)
                observation.push_back(std::clamp(g[i], -1.0, 1.0));
        };
        gravity(current.toRotationMatrix());
        gravity(reference.toRotationMatrix());
        const auto &t = c_.mimic.torso;
        gravity(
            torso(current, f.q[c_.indices[t[0]]], f.q[c_.indices[t[1]]], f.q[c_.indices[t[2]]]));
        gravity(torso(reference, ref[4 + t[0]], ref[4 + t[1]], ref[4 + t[2]]));
        for (auto value : f.omega)
            observation.push_back(value);
        for (std::size_t i = 0; i < n; ++i)
            observation.push_back(f.q[c_.indices[i]] - c_.defaults[i]);
        for (auto i : c_.indices)
            observation.push_back(std::find(c_.velocity_mask.begin(), c_.velocity_mask.end(), i) !=
                                          c_.velocity_mask.end()
                                      ? 0
                                      : f.dq[i]);
        for (auto value : last_)
            observation.push_back(steps_ < c_.mimic.warmup_steps ? 0 : value);
        for (auto &value : observation) {
            if (!std::isfinite(value))
                throw std::runtime_error("Invalid MIMIC observation");
            value = std::clamp(value, -float(c_.observation_clip), float(c_.observation_clip));
        }
        // Each term owns an oldest-to-newest history; q_ref + dq_ref form one group.
        std::size_t source = 0, destination = 0;
        for (auto width : groups_) {
            auto begin = history_.begin() + destination;
            auto end = begin + width * c_.history;
            if (!initialized_) {
                for (std::size_t age = 0; age < c_.history; ++age)
                    std::copy_n(observation.begin() + source, width, begin + age * width);
            } else {
                std::move(begin + width, end, begin);
                std::copy_n(observation.begin() + source, width, end - width);
            }
            source += width;
            destination += width * c_.history;
        }
        const auto output = inference_->run(history_);
        Target result;
        result.source_stamp = f.stamp;
        result.session = input.session;
        result.motion_frame = frame_;
        result.motion_frames = motion_.size();
        result.q = c_.hold;
        result.evidence.count = n;
        result.evidence.valid = result.evidence.history_ready = true;
        result.evidence.holding = freeze;
        result.evidence.warmup = steps_ < c_.mimic.warmup_steps;
        for (std::size_t i = 0; i < n; ++i) {
            result.evidence.raw_action[i] = output.at(i);
            last_[i] = std::clamp(output.at(i), -float(c_.action_clip), float(c_.action_clip));
            result.evidence.applied_action[i] = last_[i];
            result.q[c_.indices[i]] = c_.defaults[i] + c_.scales[i] * last_[i];
        }
        for (std::size_t i = 0; i < c_.mimic.tracked_indices.size(); ++i) {
            const auto joint = c_.mimic.tracked_indices[i];
            result.q[joint] = ref[4 + n + i];
            result.dq[joint] = freeze ? 0 : ref[4 + joints + n + i];
        }
        // Match the reference controller: seed history before advancing frame zero.
        // A non-looping motion continues inference against its last reference frame.
        if (initialized_ && steps_ + 1 >= hold_steps_) {
            if (frame_ + 1 < motion_.size())
                ++frame_;
            else if (c_.mimic.loop) {
                frame_ = 0;
                align_ = true;
            }
        }
        initialized_ = true;
        ++steps_;
        result.stamp = monotonicNow();
        result.valid = true;
        return result;
    }
};
} // namespace
std::unique_ptr<Policy> makeMimic(PolicyConfig c, std::unique_ptr<Inference> inference) {
    return std::make_unique<Mimic>(std::move(c), std::move(inference));
}
} // namespace eir::ctrl
