#include <algorithm>
#include <cmath>
#include <filesystem>
#include <eir_ctrl/config.hpp>
#include <eir_ctrl/mapping.hpp>
#include <eir_mujoco/mj_bridge.h>
#include <gtest/gtest.h>
#include <iostream>
#include <limits>
#include <set>
#include <thread>
#include <yaml-cpp/yaml.h>

using namespace eir::ctrl;
using namespace std::chrono_literals;

TEST(ProSimulation, PolicyCameraVisibilityMatchesTrainingSceneDuringLegLift) {
    const auto config = loadConfig(PRO_PERCEPTION_CONFIG);
    const auto directory = std::filesystem::path(config.policies.at(Mode::Perception).inference.model).parent_path();
    const auto env = YAML::LoadFile((directory / "deployment_env.yaml").string());
    std::set<std::string> visible;
    for (const auto &entry : env["scene"]["camera"]["mesh_prim_paths"]) {
        auto path = entry.IsScalar() ? entry.as<std::string>() : entry["prim_expr"].as<std::string>();
        if (path.size() >= 8 && path.substr(path.size() - 8) == "/visuals") path.resize(path.size() - 8);
        visible.insert(path.substr(path.find_last_of('/') + 1));
    }
    char error[2048]{};
    std::unique_ptr<mjModel, decltype(&mj_deleteModel)> m{
        mj_loadXML(PRO_TEST_MODEL, nullptr, error, sizeof(error)), mj_deleteModel};
    ASSERT_TRUE(m) << error;
    std::vector<int> leg_visuals;
    for (int g = 0; g < m->ngeom; ++g) {
        const auto *raw_name = mj_id2name(m.get(), mjOBJ_BODY, m->geom_bodyid[g]);
        const std::string body = raw_name ? raw_name : "";
        if (!m->geom_bodyid[g]) {
            EXPECT_EQ(m->geom_group[g], 0);
            continue;
        }
        const bool visual = !m->geom_contype[g] && !m->geom_conaffinity[g];
        EXPECT_EQ(m->geom_group[g], visual ? (visible.count(body) ? 1 : 2) : 5) << body;
        if (visual && visible.count(body) &&
            (body.find("l_leg_") == 0 || body.find("r_leg_") == 0 ||
             body == "l_foot_l0" || body == "r_foot_l0"))
            leg_visuals.push_back(g);
    }
    ASSERT_EQ(leg_visuals.size(), 14U);
    std::unique_ptr<mjData, decltype(&mj_deleteData)> d{mj_makeData(m.get()), mj_deleteData};
    ASSERT_TRUE(d);
    for (std::size_t i = 0; i < config.control.count; ++i) {
        const int j = mj_name2id(m.get(), mjOBJ_JOINT, config.runtime.joint_names[i].c_str());
        ASSERT_GE(j, 0);
        d->qpos[m->jnt_qposadr[j]] = config.control.ready[i];
    }
    // Offline lifted-leg fixture: catches visibility defects absent in READY.
    for (const auto &[name, q] : std::vector<std::pair<const char *, double>>{
             {"l_leg_1", -1.0}, {"l_leg_4", 1.2}, {"l_leg_5", -.2},
             {"r_leg_1", -.7}, {"r_leg_4", .9}, {"r_leg_5", -.1}}) {
        const int j = mj_name2id(m.get(), mjOBJ_JOINT, name);
        ASSERT_GE(j, 0);
        d->qpos[m->jnt_qposadr[j]] = q;
    }
    mj_forward(m.get(), d.get());
    MJ_BRIDGE::DepthCameraConfig camera_config;
    camera_config.preprocessing_version = "parkour_depth_v2";
    camera_config.camera_site_name = config.runtime.depth.site;
    camera_config.raw_width = 64; camera_config.raw_height = 36;
    camera_config.crop_up = 18; camera_config.crop_down = 0;
    camera_config.crop_left = camera_config.crop_right = 16;
    camera_config.min_distance = 0; camera_config.max_distance = 2.5;
    camera_config.horizontal_fov_deg = config.runtime.depth.horizontal_fov_deg;
    camera_config.vertical_fov_deg = config.runtime.depth.vertical_fov_deg;
    MJ_BRIDGE::DepthCamera camera(camera_config);
    ASSERT_TRUE(camera.bind(m.get()));
    MJ_BRIDGE::DepthStack correct, hidden;
    ASSERT_TRUE(camera.capture(m.get(), d.get(), 1, 0, correct));
    for (auto g : leg_visuals) m->geom_group[g] = 2;
    camera.reset();
    ASSERT_TRUE(camera.capture(m.get(), d.get(), 2, 0, hidden));
    std::size_t changed = 0;
    for (std::size_t i = 0; i < MJ_BRIDGE::DepthStack::kFrameSize; ++i)
        changed += std::abs(correct.data[i] - hidden.data[i]) > 1e-4;
    EXPECT_GT(changed, 100U);
}

TEST(ProSimulation, RobotFrontDepthCameraSeesStairTerrain) {
    const auto c = loadConfig(PRO_PERCEPTION_CONFIG).runtime.depth;
    ASSERT_TRUE(c.enabled);
    ASSERT_EQ(c.preprocessing_version, "parkour_depth_v2");
    char error[2048]{};
    std::unique_ptr<mjModel, decltype(&mj_deleteModel)> m{
        mj_loadXML(PRO_TEST_MODEL, nullptr, error, sizeof(error)), mj_deleteModel};
    ASSERT_TRUE(m) << error;
    std::unique_ptr<mjData, decltype(&mj_deleteData)> d{mj_makeData(m.get()), mj_deleteData};
    ASSERT_TRUE(d);
    // Offline pose in front of the stairs; do not drive the user's simulation.
    const int x = mj_name2id(m.get(), mjOBJ_JOINT, "floating_base_joint_x");
    const int y = mj_name2id(m.get(), mjOBJ_JOINT, "floating_base_joint_y");
    ASSERT_GE(x, 0);
    ASSERT_GE(y, 0);
    const int first_stair = mj_name2id(m.get(), mjOBJ_GEOM, "stair1");
    ASSERT_GE(first_stair, 0);
    d->qpos[m->jnt_qposadr[x]] = m->geom_pos[3 * first_stair] -
                               m->geom_size[3 * first_stair] - .7;
    d->qpos[m->jnt_qposadr[y]] = m->geom_pos[3 * first_stair + 1];
    mj_forward(m.get(), d.get());
    MJ_BRIDGE::DepthCameraConfig depth;
    depth.camera_site_name = c.site;
    depth.preprocessing_version = c.preprocessing_version;
    depth.raw_width = c.raw_width;
    depth.raw_height = c.raw_height;
    depth.crop_up = c.crop_up;
    depth.crop_down = c.crop_down;
    depth.crop_left = c.crop_left;
    depth.crop_right = c.crop_right;
    depth.capture_frequency_hz = c.frequency_hz;
    depth.min_distance = c.min_distance;
    depth.max_distance = c.max_distance;
    depth.horizontal_fov_deg = c.horizontal_fov_deg;
    depth.vertical_fov_deg = c.vertical_fov_deg;
    MJ_BRIDGE::DepthCamera camera(depth);
    ASSERT_TRUE(camera.bind(m.get()));
    const int site = camera.siteId();
    ASSERT_EQ(m->site_bodyid[site], mj_name2id(m.get(), mjOBJ_BODY, "gemini_336_lf"));
    // Parkour optical +X equals the emitter's +Z: forward and down, not the viewer.
    const auto* rotation = d->site_xmat + 9 * site;
    EXPECT_GT(rotation[2], .5);
    EXPECT_LT(rotation[8], -.3);
    std::vector<int> terrain;
    for (int i = 1; i <= 11; ++i) {
        const auto name = i <= 10 ? "stair" + std::to_string(i) : "center2";
        const int geom = mj_name2id(m.get(), mjOBJ_GEOM, name.c_str());
        ASSERT_GE(geom, 0) << name;
        ASSERT_EQ(m->geom_group[geom], 0) << name;
        terrain.push_back(geom);
    }
    MJ_BRIDGE::DepthStack visible, excluded;
    ASSERT_TRUE(camera.capture(m.get(), d.get(), 1, 0, visible));
    // Reproduce the old group-2 bug using the actual policy sampling code.
    for (const auto geom : terrain)
        m->geom_group[geom] = 2;
    camera.reset();
    ASSERT_TRUE(camera.capture(m.get(), d.get(), 2, 0, excluded));
    std::size_t changed = 0;
    float max_difference = 0;
    for (std::size_t i = 0; i < MJ_BRIDGE::DepthStack::kFrameSize; ++i) {
        ASSERT_TRUE(std::isfinite(visible.data[i]));
        ASSERT_GE(visible.data[i], 0);
        ASSERT_LE(visible.data[i], 1);
        const float difference = std::abs(visible.data[i] - excluded.data[i]);
        changed += difference > 1e-4;
        max_difference = std::max(max_difference, difference);
    }
    EXPECT_GT(changed, 50U); // A terrain detection check, not a climbing criterion.
    std::cout << "Front-camera terrain regression: changed_pixels=" << changed
              << "/576 max_normalized_difference=" << max_difference << '\n';
}

TEST(ProSimulation, HandsWristsAndLegsStayBoundedInProModel) {
    const auto c = loadConfig(PRO_CONFIG).runtime;
    char error[2048]{};
    std::unique_ptr<mjModel, decltype(&mj_deleteModel)> m{
        mj_loadXML(PRO_TEST_MODEL, nullptr, error, sizeof(error)), mj_deleteModel};
    ASSERT_TRUE(m) << error;
    std::array<int, 12> leg_joints{}, leg_motors{}, leg_outputs{};
    constexpr std::array<int, 6> driven_suffix{1, 2, 3, 4, 41, 43};
    constexpr std::array<double, 6> gear{18, 18, 18, 36, 36, 36};
    constexpr std::array<double, 6> armature{.063828, .063828, .031752, .255312, .0324, .0324};
    constexpr std::array<double, 6> torque_limit{130, 130, 90, 130, 75, 75};
    for (int i = 0; i < 12; ++i) {
        const int axis = i % 6;
        const std::string prefix = i < 6 ? "l_leg_" : "r_leg_";
        const auto motor_name = prefix + std::to_string(driven_suffix[axis]);
        const auto output_name = prefix + std::to_string(axis + 1);
        ASSERT_EQ(c.motor_names[i], motor_name);
        ASSERT_EQ(c.joint_names[i], output_name);
        leg_joints[i] = mj_name2id(m.get(), mjOBJ_JOINT, motor_name.c_str());
        leg_outputs[i] = mj_name2id(m.get(), mjOBJ_JOINT, output_name.c_str());
        leg_motors[i] = mj_name2id(m.get(), mjOBJ_ACTUATOR, motor_name.c_str());
        ASSERT_GE(leg_joints[i], 0);
        ASSERT_GE(leg_outputs[i], 0);
        ASSERT_GE(leg_motors[i], 0);
        const int a = leg_motors[i], j = leg_joints[i];
        ASSERT_EQ(m->actuator_trntype[a], mjTRN_JOINT);
        ASSERT_EQ(m->actuator_trnid[2 * a], j);
        ASSERT_EQ(m->jnt_type[j], mjJNT_HINGE);
        EXPECT_DOUBLE_EQ(m->actuator_gear[6 * a], gear[axis]);
        EXPECT_NEAR(m->dof_armature[m->jnt_dofadr[j]], armature[axis], 1e-12);
        EXPECT_NEAR(m->actuator_ctrlrange[2 * a] * gear[axis], -torque_limit[axis], 2e-5);
        EXPECT_NEAR(m->actuator_ctrlrange[2 * a + 1] * gear[axis], torque_limit[axis], 2e-5);
        if (axis >= 4)
            EXPECT_DOUBLE_EQ(m->dof_armature[m->jnt_dofadr[leg_outputs[i]]], 0);
    }
    std::array<int, 2> wrists{}, outputs{}, motors{}, slots{};
    for (int side = 0; side < 2; ++side) {
        const std::string prefix = side ? "r" : "l";
        wrists[side] = mj_name2id(m.get(), mjOBJ_JOINT, (prefix + "_arm_51").c_str());
        outputs[side] = mj_name2id(m.get(), mjOBJ_JOINT, (prefix + "_arm_6").c_str());
        motors[side] = mj_name2id(m.get(), mjOBJ_ACTUATOR, (prefix + "_arm_51").c_str());
        ASSERT_GE(wrists[side], 0);
        ASSERT_GE(outputs[side], 0);
        ASSERT_GE(motors[side], 0);
        const auto a = motors[side];
        ASSERT_EQ(m->actuator_trntype[a], mjTRN_JOINT);
        ASSERT_EQ(m->actuator_trnid[2 * a], wrists[side]);
        ASSERT_EQ(m->jnt_type[wrists[side]], mjJNT_HINGE);
        EXPECT_DOUBLE_EQ(m->actuator_gear[6 * a], 36);
        EXPECT_NEAR(m->actuator_ctrlrange[2 * a] * 36, -12, 2e-5);
        EXPECT_NEAR(m->actuator_ctrlrange[2 * a + 1] * 36, 12, 2e-5);
        ASSERT_NEAR(m->dof_armature[m->jnt_dofadr[wrists[side]]], .00552, 1e-12)
            << "PRO MJCF must put wrist rotor inertia on driven arm_51, not passive arm_6";
        ASSERT_EQ(m->dof_armature[m->jnt_dofadr[outputs[side]]], 0);
        slots[side] = std::find(c.motor_names.begin(), c.motor_names.end(), prefix + "_arm_51") -
                      c.motor_names.begin();
        ASSERT_LT(slots[side], int(c.motor_names.size()));
        ASSERT_EQ(c.joint_names[slots[side]], prefix + "_arm_6");
    }
    std::unique_ptr<mjData, decltype(&mj_deleteData)> d{mj_makeData(m.get()), mj_deleteData};
    ASSERT_TRUE(d);
    mj_forward(m.get(), d.get());
    MJ_BRIDGE::MJ_Bridge bridge(m.get(), d.get());
    bridge.setSuspendEnable(c.suspend);
    ASSERT_TRUE(bridge.setFingerJointSequence(c.finger_joint_names));
    std::vector<MJ_BRIDGE::FingerCouplingJoint> couplings;
    for (const auto &pair : c.finger_couplings)
        couplings.push_back({pair.curve, pair.active_joint, pair.passive_joint});
    ASSERT_TRUE(bridge.setFingerCouplingTable(c.finger_coupling_csv, couplings));
    ASSERT_TRUE(bridge.setJointSequence(c.joint_names));
    ASSERT_TRUE(bridge.setFingerImpedanceControl(true));
    ASSERT_TRUE(bridge.setFingerTorqueLimits(c.finger_torque_limits));
    EECmd hand{};
    hand.q = c.finger_hold_position;
    hand.dq.assign(12, 0);
    hand.tau.assign(12, 0);
    hand.Kp.assign(12, c.finger_hold_kp);
    hand.Kd.assign(12, c.finger_hold_kd);
    ASSERT_TRUE(bridge.PublishEndEffectorCmd(hand));
    ASSERT_TRUE(bridge.setMotorSequence(c.motor_names));
    RobotCmd command{};
    const auto n = c.motor_names.size();
    command.motor_cmds.q.assign(n, 0);
    command.motor_cmds.dq.assign(n, 0);
    command.motor_cmds.tau.assign(n, 0);
    command.motor_cmds.Kp.assign(n, 0);
    command.motor_cmds.Kd.assign(n, c.damping_kd);
    ASSERT_TRUE(bridge.PublishRobotCmd(command));
    MJ_BRIDGE::FingerCouplingTable table;
    ASSERT_TRUE(table.loadCsv(c.finger_coupling_csv));
    std::vector<int> joints, active, passive;
    for (const auto &name : c.finger_joint_names)
        if (!name.empty())
            joints.push_back(mj_name2id(m.get(), mjOBJ_JOINT, name.c_str()));
    for (const auto &pair : c.finger_couplings) {
        active.push_back(mj_name2id(m.get(), mjOBJ_JOINT, pair.active_joint.c_str()));
        passive.push_back(mj_name2id(m.get(), mjOBJ_JOINT, pair.passive_joint.c_str()));
        joints.push_back(passive.back());
    }
    for (auto id : joints)
        ASSERT_GE(id, 0);
    std::vector<double> lo(joints.size(), 1e9), hi(joints.size(), -1e9);
    double max_velocity = 0, max_coupling_error = 0, max_torque = 0;
    std::array<double, 2> wrist_velocity{}, wrist_torque{}, wrist_jump{}, previous_torque{},
        wrist_mapping_error{};
    std::array<double, 12> leg_velocity{}, leg_torque{}, leg_jump{}, previous_leg_torque{};
    double passive_ankle_velocity = 0;
    // 10 simulated seconds; measure the last five after the startup transient.
    const int steps = static_cast<int>(10.0 / m->opt.timestep);
    for (int step = 0; step < steps; ++step) {
        bridge.bridgeStep();
        for (int i = 0; i < 12; ++i) {
            const int a = leg_motors[i];
            const double tau = d->actuator_force[a] * m->actuator_gear[6 * a];
            ASSERT_TRUE(std::isfinite(tau));
            ASSERT_LE(std::abs(tau), torque_limit[i % 6]);
            if (step >= steps / 2) {
                leg_velocity[i] =
                    std::max(leg_velocity[i], std::abs(d->qvel[m->jnt_dofadr[leg_joints[i]]]));
                leg_torque[i] = std::max(leg_torque[i], std::abs(tau));
                leg_jump[i] = std::max(leg_jump[i], std::abs(tau - previous_leg_torque[i]));
                if (i % 6 >= 4)
                    passive_ankle_velocity = std::max(
                        passive_ankle_velocity, std::abs(d->qvel[m->jnt_dofadr[leg_outputs[i]]]));
            }
            previous_leg_torque[i] = tau;
        }
        for (int side = 0; side < 2; ++side) {
            const auto a = motors[side];
            const double tau = d->actuator_force[a] * m->actuator_gear[6 * a];
            ASSERT_LE(std::abs(tau), 12.0);
            if (step >= steps / 2) {
                wrist_velocity[side] =
                    std::max(wrist_velocity[side], std::abs(d->qvel[m->jnt_dofadr[wrists[side]]]));
                wrist_torque[side] = std::max(wrist_torque[side], std::abs(tau));
                wrist_jump[side] =
                    std::max(wrist_jump[side], std::abs(tau - previous_torque[side]));
                wrist_mapping_error[side] = std::max(
                    wrist_mapping_error[side], std::abs(d->qpos[m->jnt_qposadr[wrists[side]]] -
                                                        d->qpos[m->jnt_qposadr[outputs[side]]]));
            }
            previous_torque[side] = tau;
        }
        for (int j = 0; j < m->nq; ++j)
            ASSERT_TRUE(std::isfinite(d->qpos[j]));
        for (std::size_t i = 0; i < passive.size(); ++i) {
            const auto value =
                table.evaluate(c.finger_couplings[i].curve, d->qpos[m->jnt_qposadr[active[i]]]);
            ASSERT_TRUE(value);
            max_coupling_error =
                std::max(max_coupling_error,
                         std::abs(d->qpos[m->jnt_qposadr[passive[i]]] - value->passive_position));
            EXPECT_NEAR(d->qvel[m->jnt_dofadr[passive[i]]],
                        value->slope * d->qvel[m->jnt_dofadr[active[i]]], 1e-12);
        }
        for (std::size_t i = 0; i < c.finger_joint_names.size(); ++i) {
            if (c.finger_joint_names[i].empty())
                continue;
            const int id = mj_name2id(m.get(), mjOBJ_ACTUATOR, c.finger_joint_names[i].c_str());
            ASSERT_GE(id, 0);
            const double torque = std::abs(d->actuator_force[id] * m->actuator_gear[6 * id]);
            ASSERT_LE(torque, c.finger_torque_limits[i / 6] + 1e-9);
            max_torque = std::max(max_torque, torque);
        }
        if (step < steps / 2)
            continue;
        for (std::size_t i = 0; i < joints.size(); ++i) {
            const auto j = joints[i];
            const double q = d->qpos[m->jnt_qposadr[j]];
            lo[i] = std::min(lo[i], q);
            hi[i] = std::max(hi[i], q);
            max_velocity = std::max(max_velocity, std::abs(d->qvel[m->jnt_dofadr[j]]));
        }
    }
    double max_amplitude = 0;
    for (std::size_t i = 0; i < joints.size(); ++i) {
        SCOPED_TRACE(mj_id2name(m.get(), mjOBJ_JOINT, joints[i]));
        EXPECT_LT(hi[i] - lo[i], .01); // <0.6 degree; previous missing mapping moved ~0.3 rad.
        max_amplitude = std::max(max_amplitude, hi[i] - lo[i]);
    }
    EXPECT_LT(max_velocity, .2);
    EXPECT_LT(max_coupling_error, 1e-12);
    for (int side = 0; side < 2; ++side) {
        EXPECT_LT(wrist_velocity[side], .05);
        EXPECT_LT(wrist_torque[side], .25);
        EXPECT_LT(wrist_jump[side], .01);
        EXPECT_LT(wrist_mapping_error[side], .01); // Near neutral, not full-range IK validation.
        std::cout << "Wrist regression side=" << side << " velocity_rad_s=" << wrist_velocity[side]
                  << " torque_Nm=" << wrist_torque[side] << " step_jump_Nm=" << wrist_jump[side]
                  << " neutral_mapping_error_rad=" << wrist_mapping_error[side] << '\n';
    }
    for (int i = 0; i < mjNWARNING; ++i)
        EXPECT_EQ(d->warning[i].number, 0);
    // Settled suspended STILL only: these are regression bounds, not walking criteria.
    for (int i = 0; i < 12; ++i) {
        SCOPED_TRACE(c.motor_names[i]);
        EXPECT_LT(leg_velocity[i], .01);
        EXPECT_LT(leg_torque[i], .05);
        EXPECT_LT(leg_jump[i], .01);
    }
    EXPECT_LT(passive_ankle_velocity, .01);
    std::cout << "Leg STILL regression: velocity_rad_s="
              << *std::max_element(leg_velocity.begin(), leg_velocity.end())
              << " torque_Nm=" << *std::max_element(leg_torque.begin(), leg_torque.end())
              << " step_jump_Nm=" << *std::max_element(leg_jump.begin(), leg_jump.end())
              << " passive_ankle_velocity_rad_s=" << passive_ankle_velocity << '\n';
    std::cout << "Finger regression: amplitude_rad=" << max_amplitude
              << " max_velocity_rad_s=" << max_velocity
              << " coupling_error_rad=" << max_coupling_error << " max_torque_Nm=" << max_torque
              << '\n';
}

TEST(ProSimulation, RuntimeLoadsAndMaintainsConfiguredHandPosture) {
    auto c = loadConfig(PRO_CONFIG).runtime;
    c.model_path = PRO_TEST_MODEL;
    c.headless = true;
    eir::loco::Runtime runtime(c);
    ASSERT_TRUE(runtime.start()) << runtime.status().message;
    eir::loco::State state;
    uint64_t sequence = 0;
    const auto end = std::chrono::steady_clock::now() + 6s;
    const auto settled = std::chrono::steady_clock::now() + 3s;
    while (std::chrono::steady_clock::now() < end) {
        ASSERT_EQ(runtime.status().mode, eir::loco::Mode::Damping) << runtime.status().message;
        ASSERT_TRUE(runtime.read(state));
        sequence = state.sequence;
        const auto &hand = state.sdk.hand_states;
        ASSERT_EQ(hand.q.size(), 12U);
        ASSERT_EQ(hand.dq.size(), 12U);
        ASSERT_EQ(hand.tau.size(), 12U);
        ASSERT_EQ(hand.code.size(), 2U);
        ASSERT_EQ(hand.code[0], 1);
        ASSERT_EQ(hand.code[1], 1);
        for (std::size_t i = 0; i < hand.q.size(); ++i) {
            ASSERT_TRUE(std::isfinite(hand.q[i]));
            ASSERT_LE(std::abs(hand.tau[i]), c.finger_torque_limits[i / 6] + 1e-9);
            if (std::chrono::steady_clock::now() > settled) {
                EXPECT_LT(hand.q[i], 1.0); // Percent closure, target is zero.
                EXPECT_LT(std::abs(hand.dq[i]), .2);
            }
        }
        std::this_thread::sleep_for(10ms);
    }
    EXPECT_GT(sequence, 100U);
}
TEST(ProSimulation, RuntimeExecutesDynamicHandTargetsWithTorqueLimits) {
    const auto app = loadConfig(PRO_CONFIG);
    auto c = app.runtime;
    c.model_path = PRO_TEST_MODEL; c.headless = true;
    eir::loco::Runtime runtime(c);
    ASSERT_TRUE(runtime.start()) << runtime.status().message;
    ASSERT_TRUE(runtime.setDamping(false));
    HandControl hand(app.hand);
    Command target; target.mode = Mode::Ready;
    eir::loco::Command command(c.motor_names.size());
    std::fill(command.kd.begin(), command.kd.end(), c.damping_kd);
    command.hand_kp = app.hand.kp; command.hand_kd = app.hand.kd;
    std::array<double, 2> peak_main{}, peak_secondary{};
    double peak_torque = 0;
    const auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start < 5s) {
        eir::loco::State state;
        ASSERT_TRUE(runtime.read(state));
        const auto now = monotonicNow();
        const auto f = feedback(state, now, app.base_from_imu);
        ASSERT_TRUE(f.hand_valid);
        target.hand_mask = 0;
        // After the default closed sequence, an offline motion opens the fingers.
        if (std::chrono::steady_clock::now() - start > 3s) {
            target.mode = Mode::Walk; target.hand_mask = 0x7df; target.hand.fill(0);
        }
        ASSERT_TRUE(hand.step(target, f, now));
        command.hand_enabled = target.hand_enabled;
        for (std::size_t i = 0; i < 12; ++i) command.hand_position[i] = target.hand[i] * 100;
        ASSERT_TRUE(runtime.write(command)) << runtime.status().message;
        for (std::size_t side = 0; side < 2; ++side) {
            peak_main[side] = std::max(peak_main[side], state.sdk.hand_states.q[6 * side]);
            peak_secondary[side] = std::max(peak_secondary[side], state.sdk.hand_states.q[6 * side + 2]);
        }
        for (std::size_t i = 0; i < 12; ++i) {
            const auto tau = state.sdk.hand_states.tau[i];
            ASSERT_TRUE(std::isfinite(tau));
            ASSERT_LE(std::abs(tau), c.finger_torque_limits[i / 6] + 1e-9);
            peak_torque = std::max(peak_torque, std::abs(tau));
        }
        std::this_thread::sleep_for(2ms);
    }
    eir::loco::State state; ASSERT_TRUE(runtime.read(state));
    for (std::size_t side = 0; side < 2; ++side) {
        EXPECT_GT(peak_main[side], 70); EXPECT_GT(peak_secondary[side], 70);
        EXPECT_LT(state.sdk.hand_states.q[6 * side], 5); EXPECT_LT(state.sdk.hand_states.q[6 * side + 2], 5);
    }
    ASSERT_TRUE(runtime.setDamping(true));
    std::cout << "Dynamic hand: peak_main_percent=" << peak_main[0] << '/' << peak_main[1]
              << " peak_secondary_percent=" << peak_secondary[0] << '/' << peak_secondary[1]
              << " peak_torque_Nm=" << peak_torque << '\n';
}
