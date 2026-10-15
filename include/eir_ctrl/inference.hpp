#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace eir::ctrl {
struct InferenceConfig {
    std::string backend, model;
    std::size_t input_size{}, output_size{};
    int threads{}, warmup_steps{};
    std::vector<int64_t> input_shape; // empty: [1,input_size]; encoder: [1,8,18,32]
};
struct InferenceStats {
    uint64_t encoder_us{}, actor_us{}, feature_stamp{}, encoder_calls{}, actor_calls{}, feature_age_at_action_ns{};
};
// Each instance belongs to one ordinary worker; library types stay behind this API.
class Inference {
  public:
    virtual ~Inference() = default;
    virtual void reset() {}
    virtual InferenceStats stats() const { return {}; }
    virtual std::vector<float> runPerception(const std::vector<float>& input, uint64_t depth_stamp) {
        (void)depth_stamp;
        return run(input);
    }
    virtual std::vector<float> run(const std::vector<float> &observation) = 0;
};
std::unique_ptr<Inference> makeInference(const InferenceConfig &config);
std::unique_ptr<Inference> makeSplitInference(const InferenceConfig& encoder,
    const InferenceConfig& actor, double encoder_hz, uint64_t timeout_ns, uint64_t max_age_ns);
// Same scheduler with injected implementations, useful for deadline/fault tests.
std::unique_ptr<Inference> makeSplitInference(std::unique_ptr<Inference> encoder,
    std::unique_ptr<Inference> actor, double encoder_hz, uint64_t timeout_ns, uint64_t max_age_ns);
} // namespace eir::ctrl
