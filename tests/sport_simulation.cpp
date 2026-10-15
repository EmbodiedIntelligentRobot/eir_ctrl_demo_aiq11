// SPORT flat-ground acceptance using production Runner, mapping and SDK physics.
// A test-only Backend exposes world pose; production feedback intentionally does not.
#include <eir_ctrl/runner.hpp>
#include <eir_mujoco/mj_bridge.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>

using namespace eir::ctrl;
using namespace std::chrono_literals;
namespace {
void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
struct Pose {
    double time{}, x{}, y{}, z{}, yaw{}, tilt{}, vx{}, vy{}, wz{}, ground_force{};
    int warnings{};
};
class MeasuredSimulation final : public eir::loco::Backend {
    std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model_{nullptr, mj_deleteModel};
    std::unique_ptr<mjData, decltype(&mj_deleteData)> data_{nullptr, mj_deleteData};
    std::unique_ptr<MJ_BRIDGE::MJ_Bridge> bridge_;
    std::atomic<bool> running_{false};
    std::thread physics_;
  public:
    Snapshot<Pose> poses;
    ~MeasuredSimulation() override { stop(); }
    void start(const eir::loco::Config& c, StateCallback callback) override {
        require(c.backend == eir::loco::BackendType::Simulation && c.headless &&
                c.model_name == "aiq11" && c.motor_names.size() == 23,
                "This fixture only accepts headless SPORT simulation");
        require(mj_version() == mjVERSION_HEADER, "MuJoCo SDK version mismatch");
        char error[2048]{};
        model_.reset(mj_loadXML(c.model_path.c_str(), nullptr, error, sizeof(error)));
        require(bool(model_), error);
        require(c.model_name == model_->names, "Incorrect MJCF identity");
        auto* m = model_.get();
        const int base = mj_name2id(m, mjOBJ_BODY, "base_link");
        const int ground = mj_name2id(m, mjOBJ_GEOM, "ground");
        require(base > 0 && ground >= 0, "Missing base or ground");
        // Explicit flat-ground fixture: disable only non-robot obstacle collisions.
        // Keep original floor, robot, timestep, integrator and controller parameters.
        for (int g = 0; g < m->ngeom; ++g) {
            int body = m->geom_bodyid[g];
            while (body && body != base) body = m->body_parentid[body];
            if (body != base && g != ground) m->geom_contype[g] = m->geom_conaffinity[g] = 0;
        }
        data_.reset(mj_makeData(m));
        require(bool(data_), "mj_makeData failed");
        mj_forward(m, data_.get());
        bridge_ = std::make_unique<MJ_BRIDGE::MJ_Bridge>(c.model_path);
        bridge_->SubscribeRobotState([callback](const RobotStatePtr s) { if (s) callback(*s); });
        bridge_->setSuspendEnable(c.suspend);
        bridge_->setup(m, data_.get());
        require(bridge_->setJointSequence(c.joint_names), "Invalid joint mapping");
        bridge_->setSensorDelay(c.sensor_delay_ms);
        require(bridge_->setMotorSequence(c.motor_names), "Invalid motor mapping");
        const auto period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(m->opt.timestep));
        running_ = true;
        physics_ = std::thread([this, base, ground, period] {
            auto next = std::chrono::steady_clock::now();
            unsigned step = 0;
            try {
                while (running_) {
                    bridge_->bridgeStep();
                    if (++step % 20 == 0) {
                        auto* m = model_.get(); auto* d = data_.get();
                        const auto* p = d->xpos + 3 * base;
                        const auto* r = d->xmat + 9 * base;
                        mjtNum velocity[6]{};
                        // BODY uses the rotated inertial frame. XBODY is the
                        // robot base frame, matching the policy command axes.
                        mj_objectVelocity(m, d, mjOBJ_XBODY, base, velocity, 1);
                        Pose s{d->time, p[0], p[1], p[2], std::atan2(r[3], r[0]),
                               std::acos(std::clamp(r[8], -1.0, 1.0)),
                               velocity[3], velocity[4], velocity[2], 0, 0};
                        for (int k = 0; k < d->ncon; ++k) {
                            if (d->contact[k].geom[0] != ground && d->contact[k].geom[1] != ground) continue;
                            mjtNum force[6]{}; mj_contactForce(m, d, k, force);
                            s.ground_force += std::abs(force[0]);
                        }
                        for (int k = 0; k < mjNWARNING; ++k) s.warnings += d->warning[k].number;
                        poses.push(s); // Fixed-size snapshot; logging stays on the test thread.
                    }
                    next += period;
                    if (next < std::chrono::steady_clock::now()) next = std::chrono::steady_clock::now();
                    std::this_thread::sleep_until(next);
                }
            } catch (...) { running_ = false; }
        });
    }
    void stop() noexcept override {
        running_ = false;
        if (physics_.joinable()) physics_.join();
        if (bridge_) { bridge_->stop_sim(); bridge_.reset(); }
        data_.reset(); model_.reset();
    }
    bool write(const RobotCmd& c) override { return bridge_ && bridge_->PublishRobotCmd(c); }
    bool releaseSuspension() override {
        if (!bridge_) return false;
        bridge_->setSuspendEnable(false); return true;
    }
    bool estop() override {
        running_ = false;
        if (!bridge_) return false;
        bridge_->requestEstop(); return true;
    }
};
}
int main(int argc, char** argv) {
    try {
        require(argc == 5, "usage: ctrl_sport_simulation_test CONFIG onnx|libtorch ready|zero|forward|back|left|right|ccw|cw TRACE.csv");
        const std::map<std::string, std::array<double, 3>> cases{
            {"ready", {0,0,0}}, {"zero", {0,0,0}}, {"forward", {.3,0,0}}, {"back", {-.3,0,0}},
            {"left", {0,.2,0}}, {"right", {0,-.2,0}}, {"ccw", {0,0,.5}}, {"cw", {0,0,-.5}}};
        const auto command = cases.at(argv[3]);
        const std::string backend = argv[2];
        require(backend == "onnx" || backend == "libtorch", "Unknown inference backend");
        auto c = loadConfig(argv[1]);
        require(c.robot == "sport", "SPORT config required");
        c.runtime.headless = true; c.runtime.suspend = true;
        c.release_after_ready = true; c.remote_gui = false;
        for (auto& [mode, p] : c.policies) {
            p.inference.backend = backend;
            p.inference.model = std::filesystem::path(p.inference.model).replace_extension(
                backend == "onnx" ? ".onnx" : ".pt").string();
        }
        std::ofstream trace(argv[4]);
        require(bool(trace), "Cannot open trace file");
        trace << "wall_s,sim_s,phase,mode,x,y,z,yaw,tilt,vx_body,vy_body,wz_body,ground_force_N,infer_us,control_us,mechanism_warning_mask,torque_warning_mask\n";
        auto measured = std::make_unique<MeasuredSimulation>();
        auto* sim = measured.get();
        Runner runner(c, std::move(measured));
        runner.start();
        const auto begin = monotonicNow();
        Pose pose;
        int phase = 0;
        unsigned mechanism_warnings = 0, torque_warnings = 0;
        auto sample = [&] {
            const auto status = runner.status();
            require(status.fault == Fault::None, "FAULT: " + runner.error());
            if (sim->poses.latest(pose)) {
                const auto mechanism = runner.takeMechanismWarning();
                const auto torque = runner.takeTorqueWarning();
                mechanism_warnings += mechanism.mask != 0;
                torque_warnings += torque.mask != 0;
                trace << double(monotonicNow()-begin)/1e9 << ',' << pose.time << ',' << phase << ',' << int(status.mode)
                      << ',' << pose.x << ',' << pose.y << ',' << pose.z << ',' << pose.yaw << ',' << pose.tilt
                      << ',' << pose.vx << ',' << pose.vy << ',' << pose.wz << ',' << pose.ground_force
                      << ',' << runner.inferenceUs() << ',' << runner.controlUs()
                      << ',' << unsigned(mechanism.mask) << ',' << torque.mask << '\n';
                require(std::isfinite(pose.z) && std::isfinite(pose.tilt) && std::isfinite(pose.ground_force), "Nonfinite simulation");
                require(pose.warnings == 0, "MuJoCo numerical warning");
                require(pose.z > .4 && pose.tilt < 1.0472, "FALL: base height <=0.4 m or tilt >=60 degrees");
            }
            return status;
        };
        uint64_t id = 0;
        auto request = [&](Mode mode) {
            const auto until = monotonicNow() + 2000000000ULL;
            require(runner.requests.push({++id, until, mode}), "Mode queue full");
            Completion response;
            while (monotonicNow() < until) {
                sample();
                if (runner.completions.pop(response)) {
                    require(response.id == id, "Mismatched response"); return response.success;
                }
                std::this_thread::sleep_for(2ms);
            }
            throw std::runtime_error("Mode request timeout");
        };
        require(request(Mode::Ready), "READY rejected");
        auto until = monotonicNow() + 12000000000ULL;
        while (!sample().ready || runner.suspended()) {
            require(monotonicNow() < until, "READY timeout");
            std::this_thread::sleep_for(5ms);
        }
        if (std::string(argv[3]) == "ready") {
            const auto released = monotonicNow();
            double min_z = 100, max_tilt = 0;
            while (monotonicNow() - released < 5000000000ULL) {
                const auto status = sample();
                require(status.mode == Mode::Ready && !runner.suspended(), "READY/release lost");
                if (monotonicNow() - released > 2000000000ULL) {
                    require(status.ready, "READY posture lost after landing");
                    min_z = std::min(min_z, pose.z); max_tilt = std::max(max_tilt, pose.tilt);
                }
                std::this_thread::sleep_for(10ms);
            }
            std::cout << "backend=" << backend << " case=ready min_z=" << min_z
                      << " max_tilt_rad=" << max_tilt << std::endl;
            require(min_z > .5 && max_tilt < .523599, "READY ground posture acceptance failed");
            require(request(Mode::Still), "STILL rejected");
            runner.stop();
            std::cout << "PASS\n"; return 0;
        }
        until = monotonicNow() + 2000000000ULL;
        while (!request(Mode::Walk)) require(monotonicNow() < until, "AMP entry timeout");
        double min_z = 100, max_tilt = 0, peak_force = 0;
        double speed_sum[3]{}, stop_sum[3]{};
        unsigned speed_n = 0, stop_n = 0;
        Pose start, finish;
        for (phase = 1; phase <= 3; ++phase) {
            const auto start_time = monotonicNow();
            const double duration = phase == 2 ? 15 : 5;
            if (phase == 2) start = pose;
            while (double(monotonicNow()-start_time)/1e9 < duration) {
                const double elapsed = double(monotonicNow()-start_time)/1e9;
                require(runner.velocities.push({phase == 2 ? command : std::array<double,3>{}, monotonicNow()+200000000}), "Velocity queue full");
                require(sample().mode == Mode::Walk, "Left AMP unexpectedly");
                if (phase != 1 || elapsed > 2) {
                    min_z = std::min(min_z, pose.z); max_tilt = std::max(max_tilt, pose.tilt);
                    peak_force = std::max(peak_force, pose.ground_force);
                }
                const double v[]{pose.vx, pose.vy, pose.wz};
                if (phase == 2 && elapsed >= 3) { for (int k=0;k<3;++k) speed_sum[k]+=v[k]; ++speed_n; }
                if (phase == 3 && elapsed >= 3) { for (int k=0;k<3;++k) stop_sum[k]+=std::abs(v[k]); ++stop_n; }
                std::this_thread::sleep_for(10ms);
            }
            if (phase == 2) finish = pose;
        }
        require(speed_n && stop_n, "Missing samples");
        std::cout << "backend=" << backend << " case=" << argv[3] << " flat_fixture=true"
                  << " min_z=" << min_z << " max_tilt_rad=" << max_tilt
                  << " dx=" << finish.x-start.x << " dy=" << finish.y-start.y << " peak_ground_force_N=" << peak_force
                  << " mechanism_warning_samples=" << mechanism_warnings << " torque_warning_samples=" << torque_warnings;
        for (int k=0;k<3;++k) {
            speed_sum[k] /= speed_n; stop_sum[k] /= stop_n;
            std::cout << " mean_v" << k << '=' << speed_sum[k] << " stop_abs_v" << k << '=' << stop_sum[k];
        }
        std::cout << std::endl;
        require(min_z > .5 && max_tilt < .523599, "Posture acceptance: z>0.5m, tilt<30 degrees");
        require(stop_sum[0] < .1 && stop_sum[1] < .1 && stop_sum[2] < .2, "Did not stop within 3 seconds");
        if (std::string(argv[3]) == "zero") {
            require(std::hypot(finish.x-start.x, finish.y-start.y) < .3, "Zero command drift >=0.3m/15s");
        } else {
            for (int k=0;k<3;++k) if (command[k])
                require(speed_sum[k] / command[k] > .1, "Motion did not follow the commanded direction");
        }
        require(!request(Mode::Ready), "AMP incorrectly allowed READY");
        require(request(Mode::Still), "STILL rejected");
        runner.stop();
        std::cout << "PASS\n";
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n'; return 1;
    }
}
