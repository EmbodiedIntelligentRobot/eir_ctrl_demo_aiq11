#include <eir_ctrl/control.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace eir::ctrl {
namespace {
class PerceptionWalk final : public Policy {
    PolicyConfig c_;
    std::unique_ptr<Inference> inference_;
    std::vector<float> observation_;
    std::array<float, 23> last_{};
    bool initialized_{};
    uint32_t depth_sequence_{};
    uint64_t depth_stamp_{};
public:
    PerceptionWalk(PolicyConfig c, std::unique_ptr<Inference> inference)
        : c_(std::move(c)), inference_(std::move(inference)), observation_(5232) {}
    InferenceStats inferenceStats() const override { return inference_->stats(); }
    void reset() override {
        inference_->reset();
        initialized_ = false;
        last_.fill(0);
        depth_sequence_ = 0;
        depth_stamp_ = 0;
    }
    Target step(const Observation& input) override {
        const auto begin = monotonicNow();
        const auto* d = input.depth;
        if (!d || !d->valid || !d->stamp || d->stamp > begin ||
            begin - d->stamp > c_.depth_timeout_ns)
            throw std::runtime_error("Perception depth missing, invalid, future or expired");
        if (depth_stamp_ && (d->stamp < depth_stamp_ ||
            (d->sequence == depth_sequence_ && d->stamp != depth_stamp_) ||
            (d->sequence != depth_sequence_ &&
             (uint32_t(d->sequence - depth_sequence_) >= 0x80000000u || d->stamp <= depth_stamp_))))
            throw std::runtime_error("Perception depth sequence/timestamp discontinuity");
        for (float x : d->data)
            if (!std::isfinite(x) || x < 0 || x > 1)
                throw std::runtime_error("Perception depth outside normalized [0,1]");

        const auto& f = input.feedback;
        std::array<float, 78> frame{};
        for (std::size_t i = 0; i < 3; ++i) {
            frame[i] = float(f.omega[i]) * float(c_.angular_velocity_scale);
            frame[3 + i] = f.gravity[i];
            frame[6 + i] = std::clamp(input.velocity.value[i], c_.velocity_min[i], c_.velocity_max[i]);
        }
        for (std::size_t i = 0; i < 23; ++i) {
            frame[9 + i] = f.q[c_.indices[i]] - c_.defaults[i];
            frame[32 + i] = float(f.dq[c_.indices[i]]) * float(c_.joint_velocity_scale);
            frame[55 + i] = last_[i];
        }
        for (float x : frame)
            if (!std::isfinite(x)) throw std::runtime_error("Invalid perception proprioception");
        // Six independent histories, oldest to newest within each term.
        // This is intentionally different from AMP's frame-major history.
        std::size_t src = 0, dst = 0;
        for (std::size_t size : {3, 3, 3, 23, 23, 23}) {
            auto start = observation_.begin() + dst;
            if (!initialized_) {
                for (std::size_t k = 0; k < c_.history; ++k)
                    std::copy_n(frame.begin() + src, size, start + k * size);
            } else {
                std::move(start + size, start + size * c_.history, start);
                std::copy_n(frame.begin() + src, size, start + size * (c_.history - 1));
            }
            src += size;
            dst += size * c_.history;
        }
        std::copy(d->data.begin(), d->data.end(), observation_.begin() + dst);
        auto action = inference_->runPerception(observation_, d->stamp);
        const auto end = monotonicNow();
        if (end - begin > c_.execution_timeout_ns || end - d->stamp > c_.depth_timeout_ns)
            throw std::runtime_error("Perception inference/depth deadline exceeded");
        if (action.size() != last_.size()) throw std::runtime_error("Invalid perception action size");
        Target out;
        out.q = c_.hold;
        out.evidence.count = action.size();
        out.evidence.valid = out.evidence.history_ready = true;
        out.evidence.split = c_.split_perception;
        out.evidence.velocity = input.velocity.value;
        out.evidence.depth_stamp = d->stamp;
        for (std::size_t i = 0; i < action.size(); ++i) {
            if (!std::isfinite(action[i])) throw std::runtime_error("Nonfinite perception action");
            // Training clips previous-action observations only, never output actions.
            out.evidence.raw_action[i] = out.evidence.applied_action[i] = action[i];
            out.q[c_.indices[i]] = c_.defaults[i] + c_.scales[i] * action[i];
            last_[i] = std::clamp(action[i], -float(c_.action_clip), float(c_.action_clip));
        }
        initialized_ = true;
        depth_sequence_ = d->sequence;
        depth_stamp_ = d->stamp;
        out.stamp = end;
        out.source_stamp = f.stamp;
        out.session = input.session;
        out.valid = true;
        return out;
    }
};
} // namespace
std::unique_ptr<Policy> makePerception(PolicyConfig c, std::unique_ptr<Inference> inference) {
    return std::make_unique<PerceptionWalk>(std::move(c), std::move(inference));
}
} // namespace eir::ctrl
