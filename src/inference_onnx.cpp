#include <algorithm>
#include <eir_ctrl/inference.hpp>
#include <onnxruntime_cxx_api.h>
#include <stdexcept>

namespace eir::ctrl {
namespace {
class OnnxInference final : public Inference {
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "eir_ctrl"};
    Ort::Session session_{nullptr};
    Ort::MemoryInfo memory_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    std::string input_name_, output_name_;
    std::vector<int64_t> shape_;
    std::size_t output_size_;

  public:
    explicit OnnxInference(const InferenceConfig &c)
        : shape_(c.input_shape.empty() ? std::vector<int64_t>{1, int64_t(c.input_size)} : c.input_shape),
          output_size_(c.output_size) {
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(c.threads);
        options.SetInterOpNumThreads(1);
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options.AddConfigEntry("session.intra_op.allow_spinning", "0");
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_ = Ort::Session(env_, c.model.c_str(), options);
        if (session_.GetInputCount() != 1 || session_.GetOutputCount() != 1)
            throw std::runtime_error("Actor must have exactly one input and one output");
        Ort::AllocatorWithDefaultOptions allocator;
        input_name_ = session_.GetInputNameAllocated(0, allocator).get();
        output_name_ = session_.GetOutputNameAllocated(0, allocator).get();
        const auto input_type = session_.GetInputTypeInfo(0);
        const auto info = input_type.GetTensorTypeAndShapeInfo();
        const auto shape = info.GetShape();
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || shape.size() != shape_.size() ||
            (shape[0] != 1 && shape[0] != -1) ||
            !std::equal(shape.begin() + 1, shape.end(), shape_.begin() + 1))
            throw std::runtime_error("ONNX input shape/dtype differs from model contract");
    }
    std::vector<float> run(const std::vector<float> &input) override {
        auto tensor = Ort::Value::CreateTensor<float>(memory_, const_cast<float *>(input.data()),
                                                      input.size(), shape_.data(), shape_.size());
        const char *inputs[] = {input_name_.c_str()};
        const char *outputs[] = {output_name_.c_str()};
        auto result = session_.Run(Ort::RunOptions{nullptr}, inputs, &tensor, 1, outputs, 1);
        if (!result[0].IsTensor())
            throw std::runtime_error("ONNX actor output is not a tensor");
        const auto info = result[0].GetTensorTypeAndShapeInfo();
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            info.GetShape() != std::vector<int64_t>{1, static_cast<int64_t>(output_size_)})
            throw std::runtime_error("ONNX output must be float32 [1, actions]");
        const auto *data = result[0].GetTensorData<float>();
        return {data, data + output_size_};
    }
};
} // namespace
std::unique_ptr<Inference> makeOnnx(const InferenceConfig &c) {
    return std::make_unique<OnnxInference>(c);
}
} // namespace eir::ctrl
