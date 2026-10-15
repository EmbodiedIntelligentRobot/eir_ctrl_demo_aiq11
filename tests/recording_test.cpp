#include <eir_ctrl/recording.hpp>
#include <gtest/gtest.h>
#include <cstring>
#include <map>
#include <cmath>
#include <filesystem>
using namespace eir::ctrl;
void beginAllocationTracking();
std::size_t endAllocationTracking();
namespace {
struct Field { uint16_t kind{};std::vector<uint8_t> bytes;uint32_t count{}; };
std::map<uint16_t,Field> fields(const eir::loco::flight::Packet &p) {
    std::map<uint16_t,Field> result;
    for(std::size_t at=0;at<p.used;) {
        uint16_t tag,kind;uint32_t count;
        std::memcpy(&tag,p.payload.data()+at,2);std::memcpy(&kind,p.payload.data()+at+2,2);
        std::memcpy(&count,p.payload.data()+at+4,4);at+=8;
        const auto size=count*(kind==1?8:4);
        EXPECT_FALSE(result.count(tag));
        result[tag]={kind,{p.payload.begin()+at,p.payload.begin()+at+size},count};at+=size;
    }
    return result;
}
template<class T> T value(const Field &f,unsigned i=0) { T out;std::memcpy(&out,f.bytes.data()+sizeof(T)*i,sizeof(T));return out; }
class Constant final:public Inference {
    std::size_t n_;
public:
    explicit Constant(std::size_t n):n_(n){}
    std::vector<float> run(const std::vector<float>&) override {return std::vector<float>(n_,20.f);}
};
}
TEST(ControllerRecording, MaximumProAndSportPacketsFitExistingCacheStride) {
    for(const auto *file:{PRO_SPECIAL_CONFIG,SPORT_SPECIAL_CONFIG}) {
        auto c=loadConfig(file);Command cmd;cmd.mode=Mode::Walk;cmd.policy_target=true;
        auto &e=cmd.evidence;e.policy.valid=true;e.policy.count=32;
        std::array<MechanismEvidence,4> m{};for(unsigned i=0;i<4;++i)m[i].kind=i+1;
        TorqueWarning w;w.mask=~uint64_t{};Feedback f;
        auto p=controllerPacket(c,cmd,f,m,w,eir::loco::flight::Event::ApplicationFault);
        EXPECT_EQ(p.flags&1,0);EXPECT_LT(p.used+56+4,4096u);
        auto fs=fields(p);EXPECT_EQ(fs.at(503).count,c.control.count);
        EXPECT_EQ(fs.at(520).count,32u);EXPECT_TRUE(fs.count(547));
    }
}
TEST(ControllerRecording, IntegerNanosecondsAndSessionRemainExact) {
    auto c=loadConfig(PRO_CONFIG);Command cmd;Feedback f;f.sequence=123;f.stamp=987;
    cmd.session=0xfedcba9876543210ULL;cmd.evidence.feedback_stamp=9007199254740993ULL;
    auto fs=fields(controllerPacket(c,cmd,f,{},{}));
    const auto &x=fs.at(512);
    EXPECT_EQ(value<uint32_t>(x,24),123u);EXPECT_EQ(value<uint32_t>(x,26),987u);
    EXPECT_EQ(uint64_t(value<uint32_t>(x))|(uint64_t(value<uint32_t>(x,1))<<32),cmd.session);
    EXPECT_EQ(uint64_t(value<uint32_t>(x,4))|(uint64_t(value<uint32_t>(x,5))<<32),cmd.evidence.feedback_stamp);
}
TEST(ControllerRecording, DisabledApplicationDataDoesNotLeakJointOrActionValues) {
    auto c=loadConfig(PRO_CONFIG);c.runtime.flight_recorder.data&=~eir::loco::flight::Application;
    Command cmd;cmd.evidence.policy.valid=true;cmd.evidence.policy.count=23;
    const auto fs=fields(controllerPacket(c,cmd,{}, {},{},eir::loco::flight::Event::Warning));
    EXPECT_EQ(fs.size(),1u);EXPECT_TRUE(fs.count(500));
}
TEST(ControllerRecording, NonfiniteEvidenceIsNotSanitizedOrDropped) {
    auto c=loadConfig(PRO_CONFIG);Command cmd;cmd.q[3]=NAN;
    const auto fs=fields(controllerPacket(c,cmd,{}, {},{}));
    EXPECT_TRUE(std::isnan(value<double>(fs.at(503),3)));
}
TEST(ControllerRecording, PolicyActionsKeepAmpMimicAndPerceptionSemantics) {
    auto pro=loadConfig(PRO_SPECIAL_CONFIG);
    for(const auto mode:{Mode::Walk,Mode::LiedownProne}) {
        auto c=pro.policies.at(mode);auto p=makePolicy(c,std::make_unique<Constant>(c.indices.size()));
        Observation o;o.session=5;o.feedback.orientation=c.mimic.heading.reference;
        const auto result=p->step(o);
        ASSERT_TRUE(result.evidence.valid);ASSERT_EQ(result.evidence.count,c.indices.size());
        for(std::size_t i=0;i<c.indices.size();++i) {
            EXPECT_EQ(result.evidence.raw_action[i],20);
            EXPECT_DOUBLE_EQ(result.evidence.applied_action[i],std::min(20.,c.action_clip));
            EXPECT_DOUBLE_EQ(result.q[c.indices[i]],c.defaults[i]+c.scales[i]*std::min(20.,c.action_clip));
        }
    }
}
TEST(ControllerRecording, RejectedTargetDoesNotAttachAnotherSessionsAction) {
    auto c=loadConfig(PRO_CONFIG);Command cmd;cmd.evidence.policy.valid=true;
    Status s;Feedback f;Target t;t.session=999;t.evidence.valid=true;
    captureControlEvidence(cmd,s,f,t,3);
    EXPECT_FALSE(cmd.evidence.policy.valid);EXPECT_EQ(cmd.evidence.target_session,999u);
    cmd.policy_target=true;cmd.evidence.policy=t.evidence;
    captureControlEvidence(cmd,s,f,t,4);EXPECT_TRUE(cmd.evidence.policy.valid);
}
TEST(ControllerRecording, FaultSnapshotAndMechanismAvailabilityAreExplicit) {
    auto c=loadConfig(SPORT_CONFIG);Mapping mapping(c);
    auto m=mapping.evidence();EXPECT_EQ(m[3].kind,0u);
    for(const auto &x:m) if(x.kind==1 || x.kind==2) EXPECT_EQ(x.validity&8,0u);
    Command cmd;cmd.stamp=777;Status s;s.fault=Fault::Feedback;s.feedback_fault=FeedbackFault::Stale;
    s.fault_mode=Mode::Walk;s.fault_time=111;s.fault_joint=7;
    captureControlEvidence(cmd,s,{}, {},0);
    EXPECT_EQ(cmd.evidence.fault_time,111u);EXPECT_EQ(cmd.evidence.fault_mode,Mode::Walk);
    EXPECT_EQ(cmd.evidence.fault_joint,7u);
}
TEST(ControllerRecording, RequestAndFailurePacketsNeverInventAnEstop) {
    ModeRecord r{12,345,7,Mode::Walk,Mode::Ready,Mode::Walk,EntryRejection::SourceMode,false};
    EXPECT_EQ(modePacket(r).event,eir::loco::flight::Event::Warning);
    EXPECT_EQ(failureReason(Fault::Inference,FeedbackFault::None,"Split encoder inference/depth deadline exceeded"),FailureReason::Encoder);
    EXPECT_EQ(failureReason(Fault::Inference,FeedbackFault::None,"Split actor inference/feature deadline exceeded"),FailureReason::Actor);
    EXPECT_EQ(failureReason(Fault::Inference,FeedbackFault::None,"External depth stream expired"),FailureReason::DepthExpired);
    EXPECT_EQ(failurePacket(Fault::Inference,FailureOrigin::Inference,FailureReason::Encoder,Mode::Perception,8,999).event,
              eir::loco::flight::Event::ApplicationFault);
}

TEST(ControllerRecording, SameModeSuccessIsAnOperationNotAStateChange) {
    ModeRecord r{1, 100, 7, Mode::Still, Mode::Still, Mode::Still,
                 EntryRejection::None, true};
    auto p = modePacket(r);
    EXPECT_EQ(p.event, eir::loco::flight::Event::None);
    EXPECT_TRUE(p.flags & eir::loco::flight::Packet::EventOnly);
    EXPECT_EQ(value<uint32_t>(fields(p).at(532), 3), 1u);
    r.actual = Mode::Ready;
    EXPECT_EQ(modePacket(r).event, eir::loco::flight::Event::ModeChanged);
    r.success = false;
    EXPECT_EQ(modePacket(r).event, eir::loco::flight::Event::Warning);
}

TEST(ControllerRecording, SbusOriginIsPreservedEvenWithoutServiceRequestId) {
    ModeRecord record;
    record.origin = RequestOrigin::Sbus;
    const auto packet = fields(modePacket(record));
    EXPECT_EQ(value<uint32_t>(packet.at(534)), 2u);
    EXPECT_EQ(value<uint32_t>(packet.at(532), 5), 0u);
    record.origin = RequestOrigin::ModeService;
    EXPECT_EQ(value<uint32_t>(fields(modePacket(record)).at(534)), 1u);
}

TEST(ControllerRecording, PerceptionOutputIsNotConfusedWithClippedPreviousAction) {
    auto c=loadConfig((std::filesystem::path(PRO_CONFIG).parent_path()/"pro_perception.yaml").string()).policies.at(Mode::Perception);
    auto p=makePolicy(c,std::make_unique<Constant>(23));
    eir::loco::DepthStack d;d.valid=true;d.sequence=1;d.stamp=monotonicNow();d.data.fill(.5);
    Observation o;o.depth=&d;o.feedback.stamp=d.stamp;o.session=77;
    const auto t=p->step(o);
    for(unsigned i=0;i<23;++i) {
        EXPECT_EQ(t.evidence.raw_action[i],20);EXPECT_EQ(t.evidence.applied_action[i],20);
        EXPECT_DOUBLE_EQ(t.q[c.indices[i]],c.defaults[i]+c.scales[i]*20);
    }
    EXPECT_EQ(t.evidence.depth_stamp,d.stamp);
}
TEST(ControllerRecording, StartupDigestReflectsOperationalOverridesAndActualBackend) {
    auto original=loadConfig(PRO_CONFIG);
    const auto baseline=recordingConfig(original).runtime.recorder_config_digest;
    auto changed=original;changed.control.still_only=true;changed.control.policy_mask=0;
    EXPECT_NE(recordingConfig(changed).runtime.recorder_config_digest,baseline);
    changed=original;changed.runtime.suspend=!original.runtime.suspend;
    EXPECT_NE(recordingConfig(changed).runtime.recorder_config_digest,baseline);
    changed=original;auto &p=changed.policies.at(Mode::Walk);
    p.inference.backend="libtorch";
    p.inference.model=(std::filesystem::path(p.inference.model).parent_path()/"policy.pt").string();
    const auto refreshed=recordingConfig(changed);
    EXPECT_NE(refreshed.runtime.recorder_config_digest,baseline);
    EXPECT_TRUE(std::any_of(refreshed.runtime.recorder_controllers.begin(),refreshed.runtime.recorder_controllers.end(),
        [](const auto &s){return s.find("28:amp_humanoid_walk:libtorch:")!=s.npos;}));
}

TEST(ControllerRecording, RealTimeCaptureAndHandoffDoNotAllocate) {
    Command cmd;
    cmd.policy_target = true;
    cmd.evidence.policy.valid = true;
    Status status;
    Feedback feedback;
    Target target;
    Channel<Command> channel;
    Channel<ModeRecord, 32> events;
    uint64_t received = 0;
    beginAllocationTracking();
    for (uint64_t i = 1; i <= 10000; ++i) {
        feedback.sequence = i;
        feedback.stamp = i * 2000000;
        captureControlEvidence(cmd, status, feedback, target, i);
        channel.push(cmd);
        Command latest;
        channel.latest(latest);
        received = latest.evidence.feedback_sequence;
        events.push({i, feedback.stamp});
        ModeRecord event;
        events.pop(event);
    }
    const auto allocations = endAllocationTracking();
    EXPECT_EQ(allocations, 0u);
    EXPECT_EQ(received, 10000u);
}

TEST(ControllerRecording, MimicVariantAndFallbackDescribeTheActualStartupSelection) {
    auto c=loadConfig(PRO_CONFIG);c.mimic_variant="gripper";c.mimic_fallback=true;
    auto f=fields(controllerPacket(c,{}, {},{},{}));
    ASSERT_EQ(f.at(537).count,2u);EXPECT_EQ(value<uint32_t>(f.at(537)),2u);EXPECT_EQ(value<uint32_t>(f.at(537),1),1u);
    c.mimic_variant="fist";c.mimic_fallback=false;f=fields(controllerPacket(c,{}, {},{},{}));
    EXPECT_EQ(value<uint32_t>(f.at(537)),1u);EXPECT_EQ(value<uint32_t>(f.at(537),1),0u);
}
