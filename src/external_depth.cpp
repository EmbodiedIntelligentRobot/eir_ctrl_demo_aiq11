#include <eir_ctrl/external_depth.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace eir::ctrl {
std::string ExternalDepth::receive(uint32_t frames, uint32_t height, uint32_t width,
                                  const std::vector<float>& data, const std::string& version,
                                  const std::string& frame_id, int64_t stamp, uint64_t sequence,
                                  int64_t ros_now, uint64_t steady_now) {
    std::lock_guard lock(mutex_);
    const auto reject = [this](const char* reason) {
        latest_.valid = false;
        error_ = reason;
        return error_;
    };
    if (frames != 8 || height != 18 || width != 32 || data.size() != latest_.data.size() ||
        version != "parkour_depth_v2" || frame_id != config_.frame_id)
        return reject("External depth contract/frame_id mismatch (expected parkour_depth_v2 8x18x32)");
    if (std::any_of(data.begin(), data.end(), [](float x) { return !std::isfinite(x) || x < 0 || x > 1; }))
        return reject("External depth contains nonfinite/out-of-range values");
    if (stamp <= 0 || ros_now < stamp || uint64_t(ros_now - stamp) > config_.timeout_ns)
        return reject("External depth capture timestamp is missing, future or expired");
    const auto offset = ros_now - int64_t(steady_now);
    if (clock_bound_ && uint64_t(std::abs(offset - clock_offset_)) > config_.clock_tolerance_ns)
        return reject("External depth system/steady clock discontinuity; restart after clock synchronization");
    if (!sequence || (source_stamp_ && (stamp <= source_stamp_ || sequence <= source_sequence_)))
        return reject("External depth capture timestamp/sequence moved backwards or repeated; restart sender and controller together");
    const auto capture = stamp - (clock_bound_ ? clock_offset_ : offset);
    if (capture <= 0 || uint64_t(capture) > steady_now ||
        steady_now - uint64_t(capture) > config_.timeout_ns)
        return reject("External depth mapped acquisition time is invalid");
    if (!clock_bound_) {
        clock_offset_ = offset;
        clock_bound_ = true;
    }
    source_stamp_ = stamp; // Preserve acquisition time; never replace by receipt time.
    source_sequence_ = sequence;
    std::copy(data.begin(), data.end(), latest_.data.begin());
    ++latest_.sequence; // Local uint32 sequence; source uint64 was validated before conversion.
    latest_.stamp = uint64_t(capture);
    latest_.valid = true;
    error_.clear();
    return {};
}
loco::DepthStack ExternalDepth::sample(uint64_t now, uint64_t feedback_stamp) const {
    std::lock_guard lock(mutex_);
    if (!latest_.valid) throw std::runtime_error(error_);
    if (now < latest_.stamp || now - latest_.stamp > config_.timeout_ns)
        throw std::runtime_error("External depth stream expired");
    const auto skew = feedback_stamp >= latest_.stamp ? feedback_stamp - latest_.stamp : latest_.stamp - feedback_stamp;
    if (!feedback_stamp || skew > config_.max_skew_ns)
        throw std::runtime_error("External depth/body timestamp skew exceeds configured limit");
    return latest_;
}
} // namespace eir::ctrl
