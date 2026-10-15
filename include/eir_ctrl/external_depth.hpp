#pragma once
#include <eir_loco_runtime/depth.hpp>
#include <mutex>
#include <vector>

namespace eir::ctrl {
struct ExternalDepthConfig {
    std::string topic, frame_id;
    uint64_t timeout_ns{100000000}, max_skew_ns{50000000}, clock_tolerance_ns{5000000};
};
// Non-RT ROS producer and inference consumer. No image processing here.
// The source clock must be the same host's system clock (use_sim_time=false).
class ExternalDepth {
    ExternalDepthConfig config_;
    mutable std::mutex mutex_;
    loco::DepthStack latest_;
    int64_t clock_offset_{}, source_stamp_{};
    uint64_t source_sequence_{};
    bool clock_bound_{};
    std::string error_{"External depth has not arrived"};
public:
    explicit ExternalDepth(ExternalDepthConfig config) : config_(std::move(config)) {}
    // Returns a diagnostic; invalid input invalidates the snapshot immediately.
    std::string receive(uint32_t frames, uint32_t height, uint32_t width,
                        const std::vector<float>& data, const std::string& version,
                        const std::string& frame_id, int64_t stamp, uint64_t sequence,
                        int64_t ros_now, uint64_t steady_now);
    uint64_t acquisitionStamp() const {
        std::lock_guard lock(mutex_);
        return latest_.valid ? latest_.stamp : 0;
    }
    loco::DepthStack sample(uint64_t now, uint64_t feedback_stamp) const;
};
} // namespace eir::ctrl
