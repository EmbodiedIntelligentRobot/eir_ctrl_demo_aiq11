#include <ATen/Parallel.h>
#include <eir_ctrl/inference.hpp>
#include <stdexcept>
#include <torch/script.h>

namespace eir::ctrl {
namespace {
class TorchInference final : public Inference {
    torch::jit::Module model_;
    std::size_t output_size_;
    int threads_;
    std::vector<int64_t> shape_;

  public:
    explicit TorchInference(const InferenceConfig &c)
        : model_(torch::jit::load(c.model, torch::kCPU)), output_size_(c.output_size), threads_(c.threads),
          shape_(c.input_shape.empty() ? std::vector<int64_t>{1, int64_t(c.input_size)} : c.input_shape) {
        at::set_num_threads(c.threads);
        model_.eval();
        model_.to(torch::kFloat32);
    }
    std::vector<float> run(const std::vector<float> &input) override {
        // OpenMP/MKL settings are thread-local. Constructor warmup on the main
        // thread does not initialize the actual actor/encoder worker.
        thread_local int configured_threads = 0;
        if (configured_threads != threads_) {
            at::set_num_threads(threads_);
            configured_threads = threads_;
        }
        c10::InferenceMode guard;
        const auto tensor =
            torch::from_blob(const_cast<float *>(input.data()),
                             shape_, torch::kFloat32);
        const auto result = model_.forward({tensor});
        if (!result.isTensor())
            throw std::runtime_error("Actor must return one tensor");
        const auto output = result.toTensor().contiguous();
        if (output.device().type() != torch::kCPU || output.scalar_type() != torch::kFloat32 ||
            output.dim() != 2 || output.size(0) != 1 ||
            output.size(1) != static_cast<int64_t>(output_size_))
            throw std::runtime_error("Actor output must be float32 [1, actions] on CPU");
        return {output.data_ptr<float>(), output.data_ptr<float>() + output_size_};
    }
};
} // namespace
std::unique_ptr<Inference> makeTorch(const InferenceConfig &c) {
    return std::make_unique<TorchInference>(c);
}
} // namespace eir::ctrl
