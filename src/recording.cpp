#include <eir_ctrl/recording.hpp>
#include <algorithm>
namespace eir::ctrl {
namespace {
// Preserve full integer timestamps and identities, rather than round them through f64.
template<std::size_t N> void wide(loco::flight::Packet &p, uint16_t tag,
                                 const std::array<uint64_t,N> &input) {
    std::array<uint32_t,2*N> data{};
    for(std::size_t i=0;i<N;++i) { data[2*i]=uint32_t(input[i]); data[2*i+1]=uint32_t(input[i]>>32); }
    p.integers(tag,data.data(),data.size());
}
}
FailureReason failureReason(Fault fault, FeedbackFault feedback, std::string_view text) noexcept {
    // Existing exceptions are classified on the non-RT failure path. Unknown stays explicit.
    if (text.find("contract") != text.npos || text.find("frame_id") != text.npos) return FailureReason::DepthContract;
    if (text.find("encoder") != text.npos) return FailureReason::Encoder;
    if (text.find("actor") != text.npos) return FailureReason::Actor;
    if (text.find("skew") != text.npos) return FailureReason::DepthSkew;
    if (text.find("expired") != text.npos && text.find("depth") != text.npos) return FailureReason::DepthExpired;
    if (text.find("depth deadline") != text.npos) return FailureReason::PerceptionDeadline;
    if (text.find("nonfinite") != text.npos || text.find("Nonfinite") != text.npos) return FailureReason::Nonfinite;
    switch(fault) {
    case Fault::None: return FailureReason::None;
    case Fault::Feedback: return feedback==FeedbackFault::Stale ? FailureReason::FeedbackExpired :
        feedback==FeedbackFault::Future ? FailureReason::FeedbackFuture : FailureReason::FeedbackInvalid;
    case Fault::Inference: return text.empty() ? FailureReason::TargetExpired : FailureReason::Unknown;
    case Fault::ReadyTimeout: return FailureReason::ReadyDeadline;
    case Fault::JointLimit: return FailureReason::JointNonfinite;
    case Fault::Runtime: return FailureReason::Runtime;
    case Fault::Scheduling: return FailureReason::Scheduling;
    case Fault::Hand: return FailureReason::Hand;
    }
    return FailureReason::Unknown;
}
void captureControlEvidence(Command &cmd, const Status &s, const Feedback &f, const Target &t,
                            uint64_t generation) noexcept {
    auto &r=cmd.evidence;
    r.feedback_sequence=f.sequence; r.feedback_stamp=f.stamp;
    r.target_sequence=t.inference_sequence; r.target_stamp=t.stamp;
    r.target_session=t.session; r.target_source_stamp=t.source_stamp;
    r.motion_frame=s.motion_frame; r.motion_frames=s.motion_frames;
    r.fault=s.fault; r.fault_mode=s.fault_mode; r.fault_time=s.fault_time ? s.fault_time : (s.fault!=Fault::None?cmd.stamp:0);
    r.fault_joint=s.fault_joint; r.feedback_fault=s.feedback_fault;
    r.measured_limit_mask=s.position_limit_mask; r.target_limit_mask=s.target_limit_mask;
    r.motion_cmd=s.upper_motion_cmd; r.motion_phase=s.upper_motion_phase;
    r.motion_seconds=s.upper_motion_seconds; r.recovery_delayed=s.upper_motion_recovery_delayed;
    r.motion_generation=generation;
    if (!cmd.policy_target) r.policy={}; // Never attach an unaccepted/previous-session action.
}
loco::flight::Packet controllerPacket(const Config &c, const Command &cmd, const Feedback &f,
    const std::array<MechanismEvidence,4> &mechanisms, const TorqueWarning &warning,
    loco::flight::Event event) {
    loco::flight::Packet p; p.event=event; p.source_ns=cmd.stamp; p.correlation=f.sequence;
    const auto &r=cmd.evidence; const auto &a=r.policy;
    uint32_t codes[]={unsigned(cmd.mode),unsigned(r.fault),cmd.hand_enabled?1u:0u};
    p.integers(500,codes,3);
    if (!(c.runtime.flight_recorder.data & loco::flight::Application)) return p;
    if (c.robot == "pro") {
        uint32_t variant[] = {c.mimic_variant == "gripper" ? 2u : 1u, unsigned(c.mimic_fallback)};
        p.integers(537, variant, 2); // Actual selection: 1 fist / 2 gripper; fallback flag.
    }
    const auto n=c.control.count;
    double timing[]={double(r.control_us),double(a.duration_us),
        a.depth_stamp && cmd.stamp>=a.depth_stamp ? (cmd.stamp-a.depth_stamp)/1e6 : -1.,
        a.timing.feature_stamp && cmd.stamp>=a.timing.feature_stamp ? (cmd.stamp-a.timing.feature_stamp)/1e6 : -1.,
        double(a.sequence),double(a.timing.encoder_us),double(a.timing.actor_us)};
    p.doubles(501,timing,7);
    if(a.valid && (cmd.mode==Mode::Walk || cmd.mode==Mode::Perception || r.fault_mode==Mode::Walk || r.fault_mode==Mode::Perception))
        p.doubles(502,a.velocity.data(),3);
    p.doubles(503,cmd.q.data(),n); p.doubles(504,f.q.data(),n); p.doubles(505,f.dq.data(),n);
    p.doubles(506,cmd.dq.data(),n);
    // Effective anatomical gains, before motor-only damping and Jacobian projection.
    Values kp=cmd.kp,kd=cmd.kd;
    if(cmd.policy_target && c.policies.count(cmd.mode==Mode::Fault?r.fault_mode:cmd.mode)) for(std::size_t i=0;i<n;++i) if(!(cmd.gain_mask&(uint64_t{1}<<i))) {
        const auto gains=policyGains(c.policies.at(cmd.mode==Mode::Fault?r.fault_mode:cmd.mode),cmd,i);kp[i]=gains[0];kd[i]=gains[1];
    }
    p.doubles(507,kp.data(),n);p.doubles(508,kd.data(),n);
    if(a.valid) {
        p.doubles(509,r.policy_q.data(),n);
        p.doubles(520,a.raw_action.data(),a.count);p.doubles(521,a.applied_action.data(),a.count);
        if(a.gait) p.doubles(522,a.phase.data(),2);
        const auto mode=cmd.mode==Mode::Fault?r.fault_mode:cmd.mode;
        if(c.policies.count(mode)) {
            std::array<uint32_t,capacity> indices{};const auto &order=c.policies.at(mode).indices;
            for(std::size_t i=0;i<order.size();++i) indices[i]=order[i];
            p.integers(523,indices.data(),order.size());
        }
    }
    wide(p,512,std::array<uint64_t,14>{cmd.session,r.feedback_sequence,r.feedback_stamp,
        r.target_sequence,r.target_stamp,r.target_session,r.target_source_stamp,a.sequence,
        a.depth_stamp,a.timing.feature_stamp,a.timing.encoder_calls,a.timing.actor_calls,f.sequence,f.stamp});
    uint32_t state[]={a.count,unsigned(a.valid),unsigned(a.history_ready),unsigned(a.holding),
        unsigned(a.warmup),unsigned(a.split),unsigned(cmd.policy_target),unsigned(r.feedback_fault),
        r.fault_joint,unsigned(r.fault_mode),r.motion_cmd,r.motion_phase,unsigned(r.recovery_delayed),unsigned(a.gait)};
    p.integers(513,state,14);
    wide(p,514,std::array<uint64_t,10>{r.fault_time,r.measured_limit_mask,r.target_limit_mask,
        r.motion_frame,r.motion_frames,r.motion_generation,cmd.gain_mask,cmd.hand_mask,
        r.interval_ns,r.overruns});
    double ages[]={a.timing.feature_stamp?double(a.timing.feature_age_at_action_ns)/1e6:-1.,r.motion_seconds,cmd.ready_blend};
    p.doubles(515,ages,3);
    uint32_t masks[]={uint32_t(warning.mask),uint32_t(warning.mask>>32)};
    p.integers(510,masks,2);
    if(warning.mask) p.doubles(516,warning.requested.data(),n);
    for(std::size_t i=0;i<mechanisms.size();++i) if(mechanisms[i].kind) {
        const auto &m=mechanisms[i];
        uint32_t meta[]={m.kind,m.validity,m.dk_flags,m.ik_flags,uint32_t(m.sequence),uint32_t(m.sequence>>32),m.dk_bad_samples,m.ik_bad_samples,m.warning_samples};
        p.integers(540+2*i,meta,9);p.doubles(541+2*i,m.values.data(),12);
    }
    return p;
}
loco::flight::Packet failurePacket(Fault fault, FailureOrigin origin, FailureReason reason,
                                   Mode mode, uint64_t session, uint64_t stamp) {
    loco::flight::Packet p;p.event=loco::flight::Event::ApplicationFault;p.source_ns=stamp;
    uint32_t codes[]={unsigned(mode),unsigned(fault),0};p.integers(500,codes,3);
    uint32_t cause[]={unsigned(origin),unsigned(reason)};p.integers(530,cause,2);
    wide(p,531,std::array<uint64_t,1>{session});return p;
}
loco::flight::Packet modePacket(const ModeRecord &r) {
    loco::flight::Packet p;
    p.flags |= loco::flight::Packet::EventOnly;
    p.event = !r.success             ? loco::flight::Event::Warning
              : r.actual != r.before ? loco::flight::Event::ModeChanged
                                     : loco::flight::Event::None;
    p.source_ns = r.stamp;
    uint32_t detail[] = {unsigned(r.before),
                         unsigned(r.requested),
                         unsigned(r.actual),
                         unsigned(r.success),
                         unsigned(r.rejection),
                         r.id ? 1u : 0u,
                         r.reason};
    p.integers(532, detail, 7);
    wide(p, 533, std::array<uint64_t, 2>{r.id, r.session});
    uint32_t origin[] = {unsigned(r.origin)};
    p.integers(534, origin, 1);
    return p;
}
} // namespace eir::ctrl
