// Explicit recorder acceptance on the production Runner and MuJoCo backend.
#include <eir_ctrl/runner.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <algorithm>
#include <cmath>
using namespace eir::ctrl;
using namespace std::chrono_literals;
int main(int argc,char **argv) {
    if(argc!=5) {std::cerr<<"usage: ctrl_recording_simulation CONFIG onnx|libtorch DIRECTORY suspended|ground|depth_fault|motion\n";return 2;}
    try {
        auto c=loadConfig(argv[1]);const std::string backend=argv[2],scope=argv[4];
        if(c.runtime.backend!=eir::loco::BackendType::Simulation ||
           (backend!="onnx" && backend!="libtorch") || (scope!="suspended" && scope!="ground" && scope!="depth_fault" && scope!="motion"))
            throw std::runtime_error("Simulation-only explicit acceptance arguments required");
        if(scope=="motion" && c.motion.motions.empty())throw std::runtime_error("Motion profile is required");
        c.runtime.headless=true;c.runtime.suspend=true;c.remote_gui=false;
        c.release_after_ready=scope=="ground";c.remote_source="external";
        if(scope=="depth_fault") {
            if(!c.policies.count(Mode::Perception))throw std::runtime_error("Depth fault requires perception");
            c.external_depth.topic="test/depth_stack";
            c.runtime.depth.enabled=false;c.depth_image_topic.clear();
        }
        c.runtime.flight_recorder.enabled=true;c.runtime.flight_recorder.directory=argv[3];
        for(const auto &[mode,p]:c.policies)
            if(p.inference.backend!=backend)
                throw std::runtime_error("Entry profile backend must match acceptance argument");
        std::filesystem::create_directories(argv[3]);
        std::ofstream trace(std::filesystem::path(argv[3])/"acceptance.csv");
        trace<<"stamp_ns,mode,fault,session,feedback_sequence,inference_calls,tilt_rad,suspended\n";
        Runner runner(c,{},true);runner.start();uint64_t id=0;
        uint64_t sent_depth=0,sent_stamp=0;
        auto snapshot=[&] {
            if(scope=="depth_fault") {
                const auto steady=monotonicNow();
                if(steady-sent_stamp>=16666667) {
                    // Synthetic external camera: generate a fresh constant scene. This
                    // tests contract failure, not MuJoCo rendering or camera calibration.
                    const std::vector<float> frame(8*18*32,.5f);
                    const auto stamp=eir::loco::flight::Recorder::systemNow();
                    // Simulate ordinary capture-to-callback latency. A zero-age stamp
                    // can map a few microseconds into the future after initial binding.
                    std::this_thread::sleep_for(2ms);
                    const auto received=eir::loco::flight::Recorder::systemNow();
                    const auto error=runner.receiveDepth(8,18,32,frame,
                        "parkour_depth_v2",c.external_depth.frame_id,stamp,++sent_depth,received);
                    if(!error.empty())throw std::runtime_error(error);
                    sent_stamp=steady;
                }
            }
            const auto s=runner.status();const auto f=runner.debug().feedback;
            const double tilt=std::acos(std::clamp(1-2*(f.orientation[1]*f.orientation[1]+f.orientation[2]*f.orientation[2]),-1.,1.));
            trace<<monotonicNow()<<','<<unsigned(s.mode)<<','<<unsigned(s.fault)<<','<<s.session<<','
                 <<s.feedback_sequence<<','<<runner.inferenceCalls()<<','<<tilt<<','<<runner.suspended()<<'\n';
            if(s.fault!=Fault::None) throw std::runtime_error(runner.error());
            return s;
        };
        auto wait=[&](double seconds) {
            const auto until=monotonicNow()+uint64_t(seconds*1e9);
            while(monotonicNow()<until){snapshot();std::this_thread::sleep_for(10ms);}
        };
        auto mode=[&](Mode target,bool expected=true) {
            const auto now=monotonicNow();const auto request=++id;
            if(!runner.requests.push({request,now+2000000000,target}))throw std::runtime_error("Request queue full");
            Completion result;
            while(monotonicNow()-now<2000000000 && !runner.completions.pop(result))std::this_thread::sleep_for(1ms);
            if(result.id!=request || result.success!=expected)throw std::runtime_error("Unexpected mode result: "+std::to_string(unsigned(target)));
        };
        wait(.3);mode(Mode::Walk,false);mode(Mode::Ready);
        const auto deadline=monotonicNow()+uint64_t((c.control.ready_timeout+2)*1e9);
        while(!snapshot().ready && monotonicNow()<deadline)std::this_thread::sleep_for(10ms);
        if(!runner.status().ready)throw std::runtime_error("READY did not settle");
        mode(Mode::Walk);wait(1.5);mode(Mode::Ready,false);
        if(scope=="motion") {
            auto motion=[&](uint8_t cmd) {
                const auto now=monotonicNow();const auto request=++id;
                const MotionRequest entry{request,now+2000000000,cmd};
                const bool queued=cmd==2?runner.motion_stops.push(entry):runner.motion_requests.push(entry);
                if(!queued)throw std::runtime_error("Motion queue full");
                MotionCompletion result;
                while(monotonicNow()-now<2000000000 && !runner.motion_completions.pop(result))
                    std::this_thread::sleep_for(1ms);
                if(result.id!=request || result.error)throw std::runtime_error("Unexpected motion result: cmd="+
                    std::to_string(cmd)+" error="+std::to_string(result.error));
            };
            for(const auto &resource:c.motion.motions) {
                motion(resource.cmd);wait(.3);motion(2); // Existing CMD_STOP protocol.
                const auto deadline=monotonicNow()+4000000000;
                while(snapshot().upper_motion_phase!=unsigned(MotionPhase::Idle) && monotonicNow()<deadline)
                    std::this_thread::sleep_for(10ms);
                if(runner.status().upper_motion_phase!=unsigned(MotionPhase::Idle))
                    throw std::runtime_error("Motion did not return to AMP");
            }
        }
        if(scope=="suspended" || scope=="depth_fault") {
            if(scope=="suspended") {
                for(const auto target:{Mode::MimicA,Mode::MimicB,Mode::MimicC,Mode::MimicD,Mode::MimicE}) {
                    if(!c.policies.count(target)) continue;
                    mode(target);wait(.3);
                }
                mode(Mode::Walk);wait(.3);
            }
            if(c.policies.count(Mode::Perception)) {mode(Mode::Perception);wait(1.);mode(Mode::Walk);wait(.4);}
            else if(c.policies.count(Mode::StandToSquat)) {
                mode(Mode::StandToSquat);wait(.25);mode(Mode::SquatToStand);wait(.25);
                mode(Mode::SquatToStand,false);mode(Mode::Walk);wait(.4);
            }
        }
        if(scope=="depth_fault") {
            mode(Mode::Perception);wait(.3);
            const auto ros=eir::loco::flight::Recorder::systemNow();
            if(runner.receiveDepth(0,0,0,{},"invalid",c.external_depth.frame_id,ros,++sent_depth,ros).empty())
                throw std::runtime_error("Invalid depth was not rejected");
            const auto deadline=monotonicNow()+1000000000;
            while(runner.status().fault==Fault::None && monotonicNow()<deadline)std::this_thread::sleep_for(1ms);
            if(runner.status().fault!=Fault::Inference)throw std::runtime_error("Depth fault was not latched");
        }
        if(!runner.saveFlightRecord())throw std::runtime_error("Manual save was not accepted");
        if(scope=="depth_fault")std::this_thread::sleep_for(500ms);
        else {wait(scope=="suspended" || scope=="motion"?c.runtime.flight_recorder.post_seconds+.1:.5);mode(Mode::Still);wait(.2);}
        runner.stop();const auto stats=runner.flightStatistics();
        if(!stats.healthy || stats.write_failures || stats.quota_rejections || stats.completed+stats.incomplete!=1)
            throw std::runtime_error("Recorder acceptance statistics invalid");
        ControlTiming t;uint64_t n=0,sum=0,maximum=0;
        while(runner.popTiming(t)){++n;sum+=t.execution_ns;maximum=std::max(maximum,t.execution_ns);}
        std::ofstream result(std::filesystem::path(argv[3])/"acceptance.json");
        result<<"{\"robot\":\""<<c.robot<<"\",\"backend\":\""<<backend<<"\",\"scope\":\""<<scope
              <<"\",\"completed\":"<<stats.completed<<",\"incomplete\":"<<stats.incomplete
              <<",\"drops\":"<<stats.dropped<<",\"memory_bytes\":"<<stats.memory_bytes
              <<",\"control_samples\":"<<n<<",\"control_mean_ns\":"<<(n?sum/n:0)
              <<",\"control_max_ns\":"<<maximum<<"}\n";
        std::cout<<"PASS "<<c.robot<<' '<<backend<<' '<<scope<<" records="<<stats.completed<<" drops="<<stats.dropped<<'\n';
        return 0;
    } catch(const std::exception &e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
