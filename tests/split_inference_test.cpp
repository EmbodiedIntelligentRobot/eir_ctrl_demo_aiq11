#include <eir_ctrl/control.hpp>
#include <gtest/gtest.h>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <limits>
using namespace eir::ctrl;
namespace {
class Encoder final : public Inference {
public:
    std::mutex mutex;
    std::condition_variable condition;
    bool blocked{}, entered{};
    int delay_ms{};
    bool invalid{};
    std::vector<float> run(const std::vector<float>& input) override {
        {
            std::unique_lock lock(mutex);
            entered = true;
            condition.notify_all();
            condition.wait(lock,[this]{ return !blocked; });
        }
        if(delay_ms) std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        return std::vector<float>(128,invalid ? std::numeric_limits<float>::quiet_NaN() : input.front());
    }
    void block() { std::lock_guard lock(mutex); blocked=true; entered=false; }
    void unblock() { {std::lock_guard lock(mutex); blocked=false;} condition.notify_all(); }
    bool waitEntered() {
        std::unique_lock lock(mutex);
        return condition.wait_for(lock,std::chrono::seconds(1),[this]{return entered;});
    }
};
struct UnblockOnExit { Encoder* encoder; ~UnblockOnExit() { encoder->unblock(); } };
class Actor final : public Inference {
public:
    std::vector<float> run(const std::vector<float>& input) override {
        if(input.size()!=752) throw std::runtime_error("wrong actor input");
        return std::vector<float>(23,input[624]);
    }
};
TEST(SplitInference, ReusesFreshCompleteFeatureWhileEncoderBusyAndRejectsExpiry) {
    auto encoder=std::make_unique<Encoder>(); auto* e=encoder.get();
    auto split=makeSplitInference(std::move(encoder),std::make_unique<Actor>(),50,50000000,100000000);
    std::vector<float> input(5232,.2F);
    ASSERT_FLOAT_EQ(split->runPerception(input,monotonicNow()).front(),.2F);
    UnblockOnExit cleanup{e};
    e->block();
    input[624]=.8F;
    const auto original=split->stats().feature_stamp;
    EXPECT_FLOAT_EQ(split->runPerception(input,monotonicNow()).front(),.2F);
    ASSERT_TRUE(e->waitEntered());
    const auto begin=monotonicNow();
    EXPECT_FLOAT_EQ(split->runPerception(input,monotonicNow()).front(),.2F);
    EXPECT_LT(monotonicNow()-begin,10000000U); // Actor does not wait for blocked CNN.
    EXPECT_EQ(split->stats().feature_stamp,original);
    std::this_thread::sleep_for(std::chrono::milliseconds(110));
    EXPECT_THROW(split->runPerception(input,monotonicNow()),std::runtime_error);
    e->unblock();
}
TEST(SplitInference, ResetDiscardsFeatureAndInFlightResults) {
    auto encoder=std::make_unique<Encoder>(); auto* e=encoder.get();
    auto split=makeSplitInference(std::move(encoder),std::make_unique<Actor>(),50,50000000,100000000);
    std::vector<float> input(5232,.2F);
    ASSERT_FLOAT_EQ(split->runPerception(input,monotonicNow()).front(),.2F);
    UnblockOnExit cleanup{e};
    e->block();
    input[624]=.4F;
    split->runPerception(input,monotonicNow());
    ASSERT_TRUE(e->waitEntered());
    split->reset();
    EXPECT_EQ(split->stats().feature_stamp,0U);
    e->unblock();
    input[624]=.8F;
    EXPECT_FLOAT_EQ(split->runPerception(input,monotonicNow()).front(),.8F);
}
TEST(SplitInference, FirstFeatureTimeoutAndInvalidLatentAreRejected) {
    for(bool invalid:{false,true}) {
        auto e=std::make_unique<Encoder>();
        e->delay_ms=invalid?0:25; e->invalid=invalid;
        auto split=makeSplitInference(std::move(e),std::make_unique<Actor>(),50,10000000,100000000);
        EXPECT_THROW(split->runPerception(std::vector<float>(5232,.2F),monotonicNow()),std::runtime_error);
    }
}
} // namespace
