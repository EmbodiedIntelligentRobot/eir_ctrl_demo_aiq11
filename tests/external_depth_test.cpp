#include <eir_ctrl/external_depth.hpp>
#include <gtest/gtest.h>
#include <limits>
using namespace eir::ctrl;
namespace {
constexpr int64_t epoch = 1791400000000000000LL;
constexpr uint64_t now = 10000000000ULL;
struct Input {
    std::vector<float> data = std::vector<float>(4608, .3F);
    std::string version{"parkour_depth_v2"}, frame{"camera_optical"};
    uint32_t frames{8}, height{18}, width{32};
    int64_t stamp{epoch - 20000000}, ros_now{epoch};
    uint64_t seq{1}, steady_now{now};
    std::string send(ExternalDepth& d) {
        return d.receive(frames, height, width, data, version, frame, stamp, seq, ros_now, steady_now);
    }
};
ExternalDepthConfig config() { return {"/perception/depth_stack", "camera_optical"}; }
TEST(ExternalDepth, PreservesCaptureAgeAndNormalizedPayload) {
    ExternalDepth d(config()); Input m;
    m.data.front() = 0; m.data.back() = 1;
    EXPECT_TRUE(m.send(d).empty());
    auto sample = d.sample(now, now - 2000000);
    EXPECT_EQ(sample.stamp, now - 20000000);
    EXPECT_EQ(sample.data.front(), 0); EXPECT_EQ(sample.data.back(), 1);
    EXPECT_EQ(d.sample(now + 10000000, now).stamp, sample.stamp); // reuse does not freshen data
    EXPECT_THROW(d.sample(now + 81000000, now), std::runtime_error);
}
TEST(ExternalDepth, RejectsContractAndDataErrorsImmediately) {
    for (int kind = 0; kind < 9; ++kind) {
        ExternalDepth d(config()); Input m;
        ASSERT_TRUE(m.send(d).empty());
        ++m.seq; ++m.stamp;
        switch(kind) {
        case 0: m.frames=1; break;
        case 1: m.height=32; break;
        case 2: m.width=18; break;
        case 3: m.version="depth_v1_masked"; break;
        case 4: m.frame="other_camera"; break;
        case 5: m.data.pop_back(); break;
        case 6: m.data[0]=std::numeric_limits<float>::quiet_NaN(); break;
        case 7: m.data[0]=-0.1F; break;
        case 8: m.data[0]=1.1F; break;
        }
        EXPECT_FALSE(m.send(d).empty()) << kind;
        EXPECT_THROW(d.sample(now,now),std::runtime_error) << kind;
    }
}
TEST(ExternalDepth, RejectsSequenceClockAndBodySkew) {
    ExternalDepth d(config()); Input m;
    EXPECT_THROW(d.sample(now,now),std::runtime_error);
    ASSERT_TRUE(m.send(d).empty());
    EXPECT_THROW(d.sample(now,now-100000000),std::runtime_error);
    EXPECT_FALSE(m.send(d).empty());
    m.seq=2; m.stamp+=10000000; m.ros_now+=10000000; m.steady_now+=10000000;
    ASSERT_TRUE(m.send(d).empty());
    m.seq=1; m.stamp+=1000000;
    EXPECT_FALSE(m.send(d).empty());
    m.seq=3; m.ros_now+=1000000000;
    EXPECT_FALSE(m.send(d).empty());
    m.stamp+=1000000000; // fresh but clock jumps relative to steady clock
    EXPECT_NE(m.send(d).find("clock discontinuity"),std::string::npos);
}
TEST(ExternalDepth, RejectsMissingFutureAndExpiredAcquisitionTime) {
    for (auto stamp : {int64_t{0}, epoch+1, epoch-100000001}) {
        ExternalDepth d(config()); Input m; m.stamp=stamp;
        EXPECT_FALSE(m.send(d).empty());
    }
}
} // namespace
