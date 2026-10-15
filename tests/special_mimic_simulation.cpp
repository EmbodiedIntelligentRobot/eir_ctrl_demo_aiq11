// Full PRO dynamics, production Runner threads and mode queues; never connects hardware.
#include <eir_ctrl/runner.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
using namespace eir::ctrl;
using namespace std::chrono_literals;
namespace {
void require(bool value, const std::string &message) {
    if (!value) throw std::runtime_error(message);
}
std::array<double, 3> gravity(const std::array<double, 4> &q) {
    const auto w=q[0], x=q[1], y=q[2], z=q[3];
    const auto n=w*w+x*x+y*y+z*z;
    return {2*(w*y-x*z)/n, -2*(y*z+w*x)/n, -(w*w-x*x-y*y+z*z)/n};
}
std::array<double, 36> finalPose(const PolicyConfig &c) {
    std::ifstream stream(c.mimic.motion);
    std::string line, last, cell;
    while (std::getline(stream, line)) if (!line.empty()) last=line;
    std::istringstream row(last);
    std::array<double, 36> q{}; // wxyz + the 32 configured reference joints
    for (auto &v : q) { require(bool(std::getline(row, cell, ',')), "Missing final frame"); v=std::stod(cell); }
    return q;
}
}
int main(int argc, char **argv) {
    try {
        require(argc == 5 || argc == 6, "usage: ctrl_special_simulation_test CONFIG onnx|libtorch DOWN_MODE UP_MODE [trace.csv]");
        std::ofstream trace;
        if (argc == 6) {
            trace.open(argv[5]);
            require(bool(trace), "Cannot open trace");
            trace << "time,mode,frame,gx,gy,gz,wx,wy,wz";
            for (int i=0;i<32;++i) trace << ",q" << i << ",v" << i << ",target" << i << ",tau" << i << ",kd" << i;
            trace << '\n';
        }
        const auto down=static_cast<Mode>(std::stoi(argv[3])), up=static_cast<Mode>(std::stoi(argv[4]));
        auto config=loadConfig(argv[1]);
        config.runtime.model_path=PRO_TEST_MODEL;
        config.runtime.headless=true; config.runtime.suspend=true;
        config.release_after_ready=true; config.remote_gui=false;
        for (auto &[mode, p] : config.policies) {
            p.inference.backend=argv[2];
            p.inference.model=std::filesystem::path(p.inference.model).replace_extension(
                p.inference.backend=="onnx" ? ".onnx" : ".pt").string();
        }
        Runner runner(config);
        runner.start();
        uint64_t request_id=0;
        auto healthy=[&] {
            const auto s=runner.status();
            require(s.fault==Fault::None, "FAULT: " + runner.error());
            return s;
        };
        auto request=[&](Mode mode) {
            const auto id=++request_id, deadline=monotonicNow()+2000000000;
            require(runner.requests.push({id,deadline,mode}), "Request queue full");
            Completion c;
            while (monotonicNow()<deadline) {
                healthy();
                if (runner.completions.pop(c)) { require(c.id==id, "Wrong completion"); return c; }
                std::this_thread::sleep_for(2ms);
            }
            throw std::runtime_error("Request timed out");
        };
        auto hold=[&](Mode mode, double seconds) {
            const auto until=monotonicNow()+uint64_t(seconds*1e9);
            while (monotonicNow()<until) {
                require(healthy().mode==mode, "Unexpected mode");
                std::this_thread::sleep_for(20ms);
            }
        };
        require(!request(up).success, "STILL bypassed READY");
        require(request(Mode::Ready).success, "READY rejected");
        auto deadline=monotonicNow()+12000000000;
        while (!healthy().ready || runner.suspended()) {
            require(monotonicNow()<deadline, "READY arrival timeout");
            std::this_thread::sleep_for(20ms);
        }
        require(!request(up).success, "READY entered recovery directly");
        deadline=monotonicNow()+2000000000;
        while (!request(Mode::Walk).success) require(monotonicNow()<deadline, "AMP entry timeout");
        hold(Mode::Walk, 2);
        auto enter=[&](Mode mode) {
            const auto until=monotonicNow()+8000000000;
            Completion c;
            do {
                c=request(mode);
                if (c.success) return;
                std::this_thread::sleep_for(50ms);
            } while (monotonicNow()<until);
            const auto f=runner.debug().feedback;
            std::cerr << "Entry rejected reason=" << int(c.rejection) << " q=";
            for (auto x:f.orientation) std::cerr << x << ',';
            std::cerr << " omega="; for (auto x:f.omega) std::cerr << x << ',';
            std::cerr << std::endl;
            throw std::runtime_error("Mode " + std::to_string(int(mode)) + " entry rejected");
        };
        auto complete=[&](Mode mode) {
            const auto until=monotonicNow()+25000000000;
            uint64_t final_at=0, next_log=0;
            while (true) {
                const auto s=healthy();
                require(s.mode==mode, "Unexpected motion mode");
                const auto now=monotonicNow();
                if (trace.is_open()) {
                    const auto d=runner.debug(); const auto &f=d.feedback;
                    trace << now << ',' << int(mode) << ',' << s.motion_frame;
                    for (auto v:f.gravity) trace << ',' << v;
                    for (auto v:f.omega) trace << ',' << v;
                    for (int i=0;i<32;++i)
                        trace << ',' << f.q[i] << ',' << f.dq[i] << ',' << d.command.q[i]
                              << ',' << d.motor_command.torque[i] << ',' << d.motor_command.kd[i];
                    trace << '\n';
                }
                if (s.motion_frames && s.motion_frame+1==s.motion_frames) {
                    if (!final_at) final_at=now;
                    if (now-final_at>2000000000) break;
                }
                if (now>next_log) {
                    const auto f=runner.debug().feedback;
                    const auto warning=runner.takeMechanismWarning();
                    for (int i=0; i<4; ++i)
                        if (warning.mask & (1<<i))
                            std::cout << "MECHANISM " << warning.detail[i] << std::endl;
                    const auto torque=runner.takeTorqueWarning();
                    if (torque.mask) {
                        std::cout << "TORQUE";
                        for (int i=0; i<32; ++i)
                            if (torque.mask & (uint64_t{1}<<i))
                                std::cout << " " << i << ":" << torque.requested[i];
                        std::cout << std::endl;
                    }
                    std::cout << "mode=" << int(mode) << " frame=" << s.motion_frame << '/' << s.motion_frames
                              << " gravity=" << f.gravity[0] << ',' << f.gravity[1] << ',' << f.gravity[2]
                              << " knees=" << f.q[3] << ',' << f.q[9] << std::endl;
                    next_log=now+1000000000;
                }
                require(now<until, "Motion completion timeout");
                std::this_thread::sleep_for(20ms);
            }
            const auto f=runner.debug().feedback;
            const auto reference=finalPose(config.policies.at(mode));
            const auto expected=gravity({reference[0],reference[1],reference[2],reference[3]});
            const auto actual=gravity(f.orientation);
            double dot=0; for (int i=0;i<3;++i) dot+=expected[i]*actual[i];
            const double tilt_error=std::acos(std::clamp(dot,-1.,1.));
            std::cout << "FINAL mode=" << int(mode) << " tilt_error_rad=" << tilt_error
                      << " knees=" << f.q[3] << ',' << f.q[9] << std::endl;
            require(tilt_error<.523599, "Final base tilt differs from reference by >30 degrees");
            if (mode==Mode::StandToSquat) require(f.q[3]>1 && f.q[9]>1, "Did not squat");
            if (mode==up) {
                require(actual[2]<-.85, "Did not stand upright");
                // Supine get-up ends at ~0.8 rad knees. Compare with its own
                // reference instead of rejecting that posture at a fixed cutoff.
                for (auto i : {3,9})
                    require(std::abs(f.q[i]-reference[4+i])<.3,
                            "Final knee posture differs from reference by >0.3 rad");
            }
        };
        enter(down);
        require(!request(up).success, "Recovery entered before motion completion");
        complete(down);
        enter(up);
        complete(up);
        require(request(Mode::Walk).success, "Return to AMP rejected");
        hold(Mode::Walk, 3);
        const auto final=runner.debug().feedback;
        require(final.gravity[2]<-.85, "AMP did not maintain upright posture");
        require(request(Mode::Still).success, "STILL rejected");
        runner.stop();
        std::cout << argv[2] << " chain " << int(down) << " -> " << int(up) << " -> AMP -> STILL: PASS" << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
