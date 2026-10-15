#include <cmath>
#include <eir_ctrl/inference.hpp>
#include <stdexcept>

namespace eir::ctrl {
std::unique_ptr<Inference> makeTorch(const InferenceConfig &);
std::unique_ptr<Inference> makeOnnx(const InferenceConfig &);
namespace {
class CheckedInference final : public Inference {
    InferenceConfig config_;
    std::unique_ptr<Inference> impl_;

  public:
    CheckedInference(InferenceConfig config, std::unique_ptr<Inference> impl)
        : config_(std::move(config)), impl_(std::move(impl)) {}
    std::vector<float> run(const std::vector<float> &input) override {
        if (input.size() != config_.input_size)
            throw std::runtime_error("Policy input size mismatch");
        for (auto x : input)
            if (!std::isfinite(x))
                throw std::runtime_error("Nonfinite policy input");
        auto output = impl_->run(input);
        if (output.size() != config_.output_size)
            throw std::runtime_error("Policy output size mismatch");
        for (auto x : output)
            if (!std::isfinite(x))
                throw std::runtime_error("Nonfinite policy output");
        return output;
    }
};
} // namespace
std::unique_ptr<Inference> makeInference(const InferenceConfig &config) {
    if (!config.input_size || !config.output_size || config.threads < 1 || config.warmup_steps < 1)
        throw std::invalid_argument("Invalid inference dimensions/thread count/warmup count");
    if (!config.input_shape.empty()) {
        std::size_t product = 1;
        for (auto dimension : config.input_shape) {
            if (dimension <= 0 || std::size_t(dimension) > config.input_size / product)
                throw std::invalid_argument("Invalid inference input shape");
            product *= std::size_t(dimension);
        }
        if (config.input_shape.front() != 1 || product != config.input_size)
            throw std::invalid_argument("Inference shape/size mismatch");
    }
    std::unique_ptr<Inference> impl;
    if (config.backend == "libtorch") {
#if EIR_CTRL_WITH_TORCH
        impl = makeTorch(config);
#else
        throw std::runtime_error("LibTorch backend was not compiled");
#endif
    } else if (config.backend == "onnx") {
#if EIR_CTRL_WITH_ONNX
        impl = makeOnnx(config);
#else
        throw std::runtime_error("ONNX backend was not compiled");
#endif
    } else
        throw std::invalid_argument("Unknown inference backend: " + config.backend);
    auto checked = std::make_unique<CheckedInference>(config, std::move(impl));
    for (int i = 0; i < config.warmup_steps; ++i)
        checked->run(std::vector<float>(config.input_size, 0));
    return checked;
}
} // namespace eir::ctrl
