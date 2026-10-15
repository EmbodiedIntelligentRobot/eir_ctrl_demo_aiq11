#include <eir_ctrl/config.hpp>
#include <gtest/gtest.h>
#include <filesystem>
#include <limits>
#include <thread>
#include <cmath>

using namespace eir::ctrl;
namespace {
Config profile() {
    return loadConfig((std::filesystem::path(PRO_CONFIG).parent_path() / "pro_perception.yaml").string());
}
class CaptureDepth final : public Inference {
public:
    std::vector<float> input;
    bool slow{};
    std::vector<float> run(const std::vector<float>& x) override {
        input = x;
        if (slow) std::this_thread::sleep_for(std::chrono::milliseconds(25));
        return std::vector<float>(23, 12);
    }
};
}
TEST(Perception, TermMajorScalingActionHistoryAndReset) {
    auto c = profile().policies.at(Mode::Perception);
    auto inference = std::make_unique<CaptureDepth>();
    auto* capture = inference.get();
    auto policy = makePolicy(c, std::move(inference));
    eir::loco::DepthStack depth;
    depth.valid = true; depth.sequence = 1; depth.stamp = monotonicNow(); depth.data.fill(.4f);
    Observation o; o.depth = &depth; o.session = 3; o.feedback.stamp = monotonicNow();
    o.feedback.omega = {4,8,12}; o.feedback.gravity = {0,0,-1}; o.feedback.dq.fill(2);
    for (std::size_t i=0;i<23;++i) o.feedback.q[c.indices[i]] = c.defaults[i] + .1;
    const auto target = policy->step(o);
    ASSERT_EQ(capture->input.size(), 5232U);
    for (int age=0;age<8;++age) {
        EXPECT_FLOAT_EQ(capture->input[age*3],1);
        EXPECT_FLOAT_EQ(capture->input[24+age*3+2],-1);
        for (int j=0;j<23;++j) {
            EXPECT_NEAR(capture->input[72+age*23+j],.1,1e-6);
            EXPECT_FLOAT_EQ(capture->input[256+age*23+j],.1f);
            EXPECT_EQ(capture->input[440+age*23+j],0);
        }
    }
    EXPECT_NEAR(target.q[c.indices[0]],c.defaults[0]+3,1e-6); // Output 12 is not clipped to 10.
    EXPECT_EQ(target.session,3U);
    EXPECT_FLOAT_EQ(capture->input[624],.4f);
    o.feedback.omega[0]=8;
    policy->step(o);
    EXPECT_FLOAT_EQ(capture->input[0],1); EXPECT_FLOAT_EQ(capture->input[21],2);
    EXPECT_EQ(capture->input[440],0); EXPECT_EQ(capture->input[601],10);
    policy->reset(); policy->step(o);
    EXPECT_EQ(capture->input[440],0); EXPECT_EQ(capture->input[601],0);
}
TEST(Perception, RejectsMissingExpiredNonfiniteAndReplayedDepth) {
    auto c=profile().policies.at(Mode::Perception);
    auto p=makePolicy(c,std::make_unique<CaptureDepth>());
    Observation o;
    EXPECT_THROW(p->step(o),std::runtime_error);
    eir::loco::DepthStack d; o.depth=&d; d.valid=true; d.sequence=5; d.stamp=monotonicNow();
    EXPECT_NO_THROW(p->step(o));
    const auto stamp=d.stamp;
    d.stamp=monotonicNow()-c.depth_timeout_ns-1;
    EXPECT_THROW(p->step(o),std::runtime_error);
    d.stamp=monotonicNow()+1000000000;
    EXPECT_THROW(p->step(o),std::runtime_error);
    d.stamp=stamp; d.data[3]=std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(p->step(o),std::runtime_error);
    d.data[3]=1.1f; EXPECT_THROW(p->step(o),std::runtime_error);
    d.data[3]=.5f; d.sequence=4; EXPECT_THROW(p->step(o),std::runtime_error);
    d.sequence=5; d.stamp=monotonicNow(); EXPECT_THROW(p->step(o),std::runtime_error);
}
TEST(Perception, LateInferenceProducesNoTarget) {
    auto c=profile().policies.at(Mode::Perception);
    auto inference=std::make_unique<CaptureDepth>(); inference->slow=true;
    auto p=makePolicy(c,std::move(inference));
    eir::loco::DepthStack d; d.valid=true; d.stamp=monotonicNow(); d.sequence=1;
    Observation o; o.depth=&d;
    EXPECT_THROW(p->step(o),std::runtime_error);
}
TEST(Perception, CompiledBackendsLoadRealPolicy) {
    auto c=profile().policies.at(Mode::Perception);
    std::vector<std::string> backends;
#if EIR_CTRL_WITH_ONNX
    backends.emplace_back("onnx");
#endif
#if EIR_CTRL_WITH_TORCH
    backends.emplace_back("libtorch");
#endif
    std::vector<float> previous;
    for (const auto& name:backends) {
        c.inference.backend=name;
        auto path=std::filesystem::path(c.inference.model);
        path.replace_extension(name=="onnx"?".onnx":".pt"); c.inference.model=path.string();
        auto inference=makeInference(c.inference);
        std::vector<float> input(5232,.15f);
        for (std::size_t i=624;i<input.size();++i) input[i]=float(i%100)/100;
        const auto out=inference->run(input);
        ASSERT_EQ(out.size(),23U);
        for(std::size_t i=0;i<out.size();++i) {
            EXPECT_TRUE(std::isfinite(out[i]));
            if(!previous.empty()) EXPECT_NEAR(out[i],previous[i],2e-5);
        }
        previous=out;
    }
}

TEST(Perception, SplitExportsMatchCombinedPolicyAndReset) {
    auto c=profile().policies.at(Mode::Perception);
    // Only the reference policy ships encoder/actor exports for split parity checks.
    const auto directory=(std::filesystem::path(PRO_CONFIG).parent_path() /
                          "../policies/perception/pro/walk").lexically_normal();
    std::vector<std::string> backends;
#if EIR_CTRL_WITH_ONNX
    backends.emplace_back("onnx");
#endif
#if EIR_CTRL_WITH_TORCH
    backends.emplace_back("libtorch");
#endif
    for (const auto& backend:backends) {
        const auto extension=backend=="onnx" ? ".onnx" : ".pt";
        auto unified_config=c.inference;
        unified_config.backend=backend;
        unified_config.model=(directory/(std::string("policy")+extension)).string();
        auto unified=makeInference(unified_config);
        auto encoder_config=unified_config, actor_config=unified_config;
        encoder_config.model=(directory/(std::string("encoder")+extension)).string();
        encoder_config.input_size=4608; encoder_config.output_size=128;
        encoder_config.input_shape={1,8,18,32};
        actor_config.model=(directory/(std::string("actor")+extension)).string();
        actor_config.input_size=752;
        auto split=makeSplitInference(encoder_config,actor_config,50,18000000,100000000);
        for (int trial=0;trial<8;++trial) {
            split->reset();
            std::vector<float> input(5232);
            for (std::size_t i=0;i<input.size();++i)
                input[i]=i<624 ? std::sin(double(i+trial))*.2F : float((i+trial*17)%101)/100;
            const auto expected=unified->run(input);
            const auto stamp=monotonicNow();
            const auto actual=split->runPerception(input,stamp);
            ASSERT_EQ(actual.size(),23U);
            for (std::size_t i=0;i<actual.size();++i)
                EXPECT_NEAR(actual[i],expected[i],2e-5) << backend << " trial=" << trial;
            EXPECT_EQ(split->stats().feature_stamp,stamp);
        }
    }
}
