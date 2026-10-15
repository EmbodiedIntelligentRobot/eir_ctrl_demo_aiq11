#include <eir_ctrl/inference.hpp>
#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace eir::ctrl {
namespace {
using Clock = std::chrono::steady_clock;
uint64_t nowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}
// Encoder alone owns its worker. Actor never waits for a new feature after the
// first step; it consumes a complete, age-checked snapshot. Reset invalidates
// in-flight work using a generation number, without cancelling a library call.
class SplitInference final : public Inference {
    std::unique_ptr<Inference> encoder_, actor_;
    std::chrono::nanoseconds period_;
    uint64_t timeout_, max_age_;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::thread worker_;
    bool stopping_{}, pending_{}, valid_{};
    uint64_t generation_{}, input_stamp_{}, feature_stamp_{}, encoder_us_{}, encoder_calls_{};
    std::vector<float> pending_depth_ = std::vector<float>(4608);
    std::array<float,128> feature_{};
    std::exception_ptr error_;
    InferenceStats stats_; // actor worker only
    void encode() noexcept {
        auto next = Clock::now();
        try {
            while (true) {
                std::unique_lock lock(mutex_);
                condition_.wait(lock, [this] { return stopping_ || pending_; });
                if (stopping_) return;
                if (!valid_) next = Clock::now(); // First feature after reset starts immediately.
                condition_.wait_until(lock, next, [this] { return stopping_ || !pending_; });
                if (stopping_) return;
                if (!pending_) { next = Clock::now(); continue; }
                auto input = pending_depth_;
                const auto stamp = input_stamp_, generation = generation_;
                pending_ = false;
                lock.unlock();
                const auto begin = nowNs();
                std::vector<float> latent;
                std::exception_ptr error;
                try {
                    if (begin < stamp || begin - stamp > max_age_)
                        throw std::runtime_error("Split encoder input depth expired");
                    latent = encoder_->run(input);
                    if (latent.size() != 128 || std::any_of(latent.begin(), latent.end(),
                        [](float x) { return !std::isfinite(x); }))
                        throw std::runtime_error("Split encoder feature shape/value invalid");
                    const auto end = nowNs();
                    if (end - begin > timeout_ || end - stamp > max_age_)
                        throw std::runtime_error("Split encoder inference/depth deadline exceeded");
                } catch (...) { error = std::current_exception(); }
                const auto duration = (nowNs() - begin) / 1000;
                lock.lock();
                if (generation == generation_) {
                    error_ = error;
                    valid_ = !error;
                    if (!error) {
                        std::copy(latent.begin(), latent.end(), feature_.begin());
                        feature_stamp_ = stamp;
                        encoder_us_ = duration;
                        ++encoder_calls_;
                    }
                    if (error) pending_ = false;
                }
                condition_.notify_all();
                next = Clock::time_point(std::chrono::nanoseconds(begin)) + period_;
                if (next < Clock::now()) next = Clock::now() + period_;
            }
        } catch (...) {
            std::lock_guard lock(mutex_);
            error_ = std::current_exception();
            valid_ = false;
            condition_.notify_all();
        }
    }
public:
    SplitInference(std::unique_ptr<Inference> encoder, std::unique_ptr<Inference> actor,
                   double hz, uint64_t timeout, uint64_t age)
        : encoder_(std::move(encoder)), actor_(std::move(actor)),
          period_(int64_t(1e9 / hz)), timeout_(timeout), max_age_(age) {
        // Start only after every shared field has been initialized.
        worker_ = std::thread(&SplitInference::encode, this);
    }
    ~SplitInference() override {
        { std::lock_guard lock(mutex_); stopping_ = true; }
        condition_.notify_all();
        worker_.join();
    }
    void reset() override {
        std::lock_guard lock(mutex_);
        ++generation_;
        pending_ = valid_ = false;
        error_ = nullptr;
        stats_ = {};
        encoder_calls_ = 0;
        condition_.notify_all();
    }
    InferenceStats stats() const override { return stats_; }
    std::vector<float> run(const std::vector<float>& input) override {
        return runPerception(input, nowNs());
    }
    std::vector<float> runPerception(const std::vector<float>& input, uint64_t stamp) override {
        if (input.size() != 5232) throw std::runtime_error("Split policy input size mismatch");
        std::vector<float> actor_input(752);
        std::copy_n(input.begin(),624,actor_input.begin());
        {
            std::unique_lock lock(mutex_);
            if (error_) std::rethrow_exception(error_);
            std::copy(input.begin()+624,input.end(),pending_depth_.begin());
            input_stamp_ = stamp;
            pending_ = true; // Latest wins; no backlog of stale camera frames.
            condition_.notify_all();
            if (!valid_ && !condition_.wait_for(lock, std::chrono::nanoseconds(timeout_),
                                               [this] { return valid_ || error_; }))
                throw std::runtime_error("Split encoder first feature deadline exceeded");
            if (error_) std::rethrow_exception(error_);
            const auto now = nowNs();
            if (!feature_stamp_ || now < feature_stamp_ || now - feature_stamp_ > max_age_)
                throw std::runtime_error("Split encoder visual feature expired");
            std::copy(feature_.begin(),feature_.end(),actor_input.begin()+624);
            stats_.encoder_us = encoder_us_;
            stats_.feature_stamp = feature_stamp_;
            stats_.encoder_calls = encoder_calls_;
        }
        const auto begin = nowNs();
        auto action = actor_->run(actor_input);
        const auto end = nowNs();
        if (end - begin > timeout_ || end - stats_.feature_stamp > max_age_)
            throw std::runtime_error("Split actor inference/feature deadline exceeded");
        stats_.actor_us = (end - begin)/1000;
        ++stats_.actor_calls;
        stats_.feature_age_at_action_ns = end - stats_.feature_stamp;
        return action;
    }
};
} // namespace
std::unique_ptr<Inference> makeSplitInference(const InferenceConfig& encoder,
    const InferenceConfig& actor, double hz, uint64_t timeout, uint64_t age) {
    if (!std::isfinite(hz) || hz < 1 || hz > 50 || !timeout || !age ||
        encoder.input_size != 4608 || encoder.output_size != 128 ||
        encoder.input_shape != std::vector<int64_t>{1,8,18,32} ||
        actor.input_size != 752 || actor.output_size != 23)
        throw std::invalid_argument("Invalid split CNN/actor contract");
    return makeSplitInference(makeInference(encoder), makeInference(actor), hz, timeout, age);
}
std::unique_ptr<Inference> makeSplitInference(std::unique_ptr<Inference> encoder,
    std::unique_ptr<Inference> actor, double hz, uint64_t timeout, uint64_t age) {
    if (!encoder || !actor || !std::isfinite(hz) || hz < 1 || hz > 50 ||
        !timeout || timeout > age || double(age) <= 1e9 / hz)
        throw std::invalid_argument("Invalid split inference scheduler/deadlines");
    return std::make_unique<SplitInference>(std::move(encoder),std::move(actor),hz,timeout,age);
}
} // namespace eir::ctrl
