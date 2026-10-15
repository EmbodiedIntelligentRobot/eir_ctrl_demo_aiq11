#include <eir_ctrl/config.hpp>
#include <gtest/gtest.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>

using namespace eir::ctrl;
namespace {
class ConfigValidation : public ::testing::Test {
protected:
    std::filesystem::path directory;
    YAML::Node root, robot;
    void SetUp() override {
        auto pattern = (std::filesystem::temp_directory_path() / "eir-config-XXXXXX").string();
        const auto created = mkdtemp(pattern.data());
        ASSERT_NE(created, nullptr);
        directory = created;
        const auto base = std::filesystem::path(PRO_CONFIG).parent_path();
        root = YAML::LoadFile(PRO_CONFIG);
        root["flight_recorder"] = YAML::LoadFile((base / "flight_recorder.yaml").string());
        const auto robot_path = base / root["robot"].as<std::string>();
        robot = YAML::LoadFile(robot_path.string());
        for (auto entry : root["policies"])
            entry.second = (base / entry.second.as<std::string>()).lexically_normal().string();
        for (auto pair : robot["pairs"])
            pair["resources"] = (robot_path.parent_path() / pair["resources"].as<std::string>())
                                    .lexically_normal().string();
        robot["simulation_hand"]["coupling_csv"] =
            (robot_path.parent_path() / robot["simulation_hand"]["coupling_csv"].as<std::string>())
                .lexically_normal().string();
        root["robot"] = (directory / "robot.yaml").string();
    }
    void useSport() {
        const auto base = std::filesystem::path(PRO_CONFIG).parent_path();
        root = YAML::LoadFile((base / "sport.yaml").string());
        root["flight_recorder"] = YAML::LoadFile((base / "flight_recorder.yaml").string());
        const auto profile = base / root["robot"].as<std::string>();
        robot = YAML::LoadFile(profile.string());
        for (auto entry : root["policies"])
            entry.second = (base / entry.second.as<std::string>()).lexically_normal().string();
        for (auto pair : robot["pairs"])
            pair["resources"] = (profile.parent_path() / pair["resources"].as<std::string>()).string();
        root["robot"] = (directory / "robot.yaml").string();
    }
    void TearDown() override { if (!directory.empty()) std::filesystem::remove_all(directory); }
    Config load() {
        std::ofstream(directory / "robot.yaml") << robot;
        std::ofstream(directory / "config.yaml") << root;
        return loadConfig((directory / "config.yaml").string());
    }
    void rejects(const std::string &reason) {
        try { (void)load(); FAIL() << "Unsafe configuration accepted"; }
        catch (const std::invalid_argument &e) {
            EXPECT_NE(std::string(e.what()).find(reason), std::string::npos) << e.what();
        }
    }
};
TEST_F(ConfigValidation, SensorRangeIsOptionalAndValidated) {
    const auto base = std::filesystem::path(PRO_CONFIG).parent_path();
    root["runtime"]["depth"] = YAML::LoadFile((base / "pro_perception.yaml").string())["runtime"]["depth"];
    root["runtime"]["depth"].remove("sensor_min_distance");
    root["runtime"]["depth"].remove("sensor_max_distance");
    EXPECT_DOUBLE_EQ(load().runtime.depth.sensor_min_distance, .25);
    root["runtime"]["depth"]["sensor_min_distance"] = .3;
    root["runtime"]["depth"]["sensor_max_distance"] = 2.;
    EXPECT_DOUBLE_EQ(load().runtime.depth.sensor_min_distance, .3);
    EXPECT_DOUBLE_EQ(load().runtime.depth.sensor_max_distance, 2.);
    root["runtime"]["depth"]["sensor_max_distance"] = .1;
    rejects("Sensor maximum depth must exceed minimum depth");
}
TEST_F(ConfigValidation, RemoteGuiDefaultsOnAndCanBeDisabled) {
    root["remote"].remove("gui");
    EXPECT_TRUE(load().remote_gui);
    root["remote"]["gui"] = false;
    EXPECT_FALSE(load().remote_gui);
    root["remote"]["gui"] = true;
    EXPECT_TRUE(load().remote_gui);
}
TEST_F(ConfigValidation, ExternalDepthRejectsMixedSourcesAndUnsafeDeadlines) {
    const auto base = std::filesystem::path(PRO_CONFIG).parent_path();
    const auto camera = YAML::LoadFile((base / "pro_perception_camera.yaml").string());
    root["policies"][41] = (base / camera["policies"][41].as<std::string>()).lexically_normal().string();
    root["runtime"]["external_depth"] = YAML::Clone(camera["runtime"]["external_depth"]);
    EXPECT_FALSE(load().runtime.depth.enabled);
    root["runtime"]["depth"] = YAML::LoadFile((base / "pro_perception.yaml").string())["runtime"]["depth"];
    rejects("Choose simulation depth OR external depth");
    root["runtime"].remove("depth");
    root["runtime"]["external_depth"]["max_body_skew"] = 0.2;
    rejects("deadlines exceed");
    root["runtime"]["external_depth"]["timeout"] = 1e100;
    rejects("deadlines exceed");
}
TEST_F(ConfigValidation, SplitPerceptionChecksComponentHashesAndFrequency) {
    const auto base = std::filesystem::path(PRO_CONFIG).parent_path();
    const auto root_profile = YAML::LoadFile((base / "pro_perception.yaml").string());
    // Split resources belong to the reference policy, independent of the default model.
    const auto profile_path = (base / "../policies/perception/pro/walk/config.yaml").lexically_normal();
    auto profile = YAML::LoadFile(profile_path.string());
    for (auto entry : profile["models"])
        entry.second = (profile_path.parent_path() / entry.second.as<std::string>()).string();
    auto manifest = YAML::LoadFile((profile_path.parent_path() / "manifest.json").string());
    for (const auto* name : {"encoder", "actor"})
        for (const auto* backend : {"libtorch", "onnx"}) {
            auto resource = manifest["split_models"][name][backend];
            resource["file"] = (profile_path.parent_path() / resource["file"].as<std::string>()).string();
        }
    profile["perception"]["manifest"] = (directory / "manifest.json").string();
    root["policies"][41] = (directory / "perception.yaml").string();
    root["runtime"]["depth"] = YAML::Clone(root_profile["runtime"]["depth"]);
    root["inference"]["perception_execution"] = "split";
    std::ofstream(directory / "perception.yaml") << profile;
    std::ofstream(directory / "manifest.json") << manifest;
    EXPECT_TRUE(load().policies.at(Mode::Perception).split_perception);
    root["inference"]["encoder_hz"] = 10;
    rejects("feature age budget");
    root["inference"]["encoder_hz"] = 50;
    const auto backend = root["inference"]["backend"].as<std::string>();
    manifest["split_models"]["encoder"][backend]["sha256"] = std::string(64,'0');
    std::ofstream(directory / "manifest.json") << manifest;
    rejects("Split encoder SHA256 differs");
    root["inference"]["perception_execution"] = "unknown";
    rejects("perception_execution must be");
}
TEST_F(ConfigValidation, ValidProfileSurvivesRelocation) { EXPECT_EQ(load().pairs.size(), 4U); }
TEST_F(ConfigValidation, RejectsEveryMissingMechanism) {
    const auto original = YAML::Clone(robot);
    for (std::size_t removed = 0; removed < 4; ++removed) {
        robot = YAML::Clone(original);
        YAML::Node pairs(YAML::NodeType::Sequence);
        for (std::size_t i = 0; i < 4; ++i)
            if (i != removed) pairs.push_back(robot["pairs"][i]);
        robot["pairs"] = pairs;
        rejects("PRO requires");
    }
    robot["pairs"] = YAML::Node(YAML::NodeType::Sequence);
    rejects("PRO requires");
}
TEST_F(ConfigValidation, RejectsDuplicateTorqueJoint) {
    robot["torque_indices"].push_back("l_leg_4");
    rejects("Duplicate torque joint: l_leg_4");
}
TEST_F(ConfigValidation, RejectsDuplicateMechanismKind) {
    robot["pairs"][1]["kind"] = "left_ankle";
    rejects("Duplicate mechanism");
}
TEST_F(ConfigValidation, RejectsWrongJointOrderAndMechanismKind) {
    robot["pairs"][0]["joints"][0] = "l_leg_6";
    robot["pairs"][0]["joints"][1] = "l_leg_5";
    rejects("Wrong joint/order");
    robot["pairs"][0]["kind"] = "unknown";
    rejects("Unknown pair mapping");
}
TEST_F(ConfigValidation, RejectsWrongMotorOrder) {
    robot["motor_names"][4] = "l_leg_43";
    robot["motor_names"][5] = "l_leg_41";
    rejects("Wrong motor/order");
}
TEST_F(ConfigValidation, RejectsHalfTorqueControlledMechanism) {
    robot["torque_indices"].push_back("neck_2");
    rejects("Both mechanism joints");
}
TEST_F(ConfigValidation, RejectsProProfileRelabeledAsSport) {
    robot["identity"] = "sport";
    rejects("Three-finger control requires");
}

TEST_F(ConfigValidation, PerceptionRequiresDepthAndMatchingModelManifest) {
    const auto base = std::filesystem::path(PRO_CONFIG).parent_path();
    const auto perception = YAML::LoadFile((base / "pro_perception.yaml").string());
    const auto policy_path = (base / perception["policies"][41].as<std::string>()).lexically_normal();
    auto policy = YAML::LoadFile(policy_path.string());
    for (auto entry : policy["models"])
        entry.second = (policy_path.parent_path() / entry.second.as<std::string>()).string();
    const auto manifest_path = policy_path.parent_path() / policy["perception"]["manifest"].as<std::string>();
    auto manifest = YAML::LoadFile(manifest_path.string());
    policy["perception"]["manifest"] = (directory / "manifest.json").string();
    root["policies"][41] = (directory / "perception.yaml").string();
    std::ofstream(directory / "perception.yaml") << policy;
    std::ofstream(directory / "manifest.json") << manifest;
    rejects("requires 50 Hz depth");
    root["runtime"]["depth"] = YAML::Clone(perception["runtime"]["depth"]);
    EXPECT_NO_THROW(load());
    const auto key = root["inference"]["backend"].as<std::string>() == "onnx" ? "onnx_sha256" : "model_sha256";
    manifest[key] = std::string(64, '0');
    std::ofstream(directory / "manifest.json") << manifest;
    rejects("SHA256 differs");
    manifest = YAML::LoadFile(manifest_path.string());
    manifest["observation_order"] = "frame-major";
    std::ofstream(directory / "manifest.json") << manifest;
    rejects("manifest/profile contract mismatch");
}
} // namespace

TEST_F(ConfigValidation, RejectsSwappedDirectDriveMotorsAndUnknownIdentity) {
    const auto left = robot["motor_names"][3].as<std::string>();
    robot["motor_names"][3] = robot["motor_names"][9].as<std::string>();
    robot["motor_names"][9] = left;
    rejects("Wrong motor identity for joint: l_leg_4");
    robot["motor_names"][3] = left;
    robot["motor_names"][9] = "r_leg_4";
    robot["joint_names"][0] = "unknown_leg";
    rejects("PRO joint identity contract mismatch");
}
TEST_F(ConfigValidation, RejectsPolicyJointReorderingWithUnchangedModelsInBothBackends) {
    for (const auto* backend : {"onnx", "libtorch"}) {
        root["inference"]["backend"] = backend;
        const auto path = std::filesystem::path(root["policies"][28].as<std::string>());
        auto policy = YAML::LoadFile(path.string());
        for (auto entry : policy["models"])
            entry.second = (path.parent_path() / entry.second.as<std::string>()).string();
        const auto first = policy["joints"][0].as<std::string>();
        policy["joints"][0] = policy["joints"][6].as<std::string>();
        policy["joints"][6] = first;
        std::ofstream(directory / "policy.yaml") << policy;
        root["policies"][28] = (directory / "policy.yaml").string();
        rejects("Model observation/action contract mismatch");
        root["policies"][28] = path.string();
    }
}

TEST_F(ConfigValidation, RejectsMimicReferenceJointReordering) {
    const auto source = std::filesystem::path(root["policies"][30].as<std::string>());
    auto profile = YAML::LoadFile(source.string());
    if (profile["contract"])
        profile["contract"] = (source.parent_path() / profile["contract"].as<std::string>()).string();
    for (auto model : profile["models"])
        model.second = (source.parent_path() / model.second.as<std::string>()).string();
    profile["mimic"]["motion"] = (source.parent_path() / profile["mimic"]["motion"].as<std::string>()).string();
    root["policies"][30] = (directory / "mimic.yaml").string();
    for (const auto* key : {"tracked_joints", "torso_joints"}) {
        auto bad = YAML::Clone(profile);
        const auto first = bad["mimic"][key][0].as<std::string>();
        bad["mimic"][key][0] = bad["mimic"][key][1].as<std::string>();
        bad["mimic"][key][1] = first;
        std::ofstream(directory / "mimic.yaml") << bad;
        rejects("MIMIC reference/joint contract mismatch");
    }
}
TEST_F(ConfigValidation, SpecialMimicRejectsMissingSourceAndChangedEntryContract) {
    const auto special = YAML::LoadFile(PRO_SPECIAL_CONFIG);
    const auto base = std::filesystem::path(PRO_SPECIAL_CONFIG).parent_path();
    for (auto entry : special["policies"])
        root["policies"][entry.first.as<int>()] = (base / entry.second.as<std::string>()).string();
    const auto original = YAML::Clone(root);
    root["policies"].remove(35);
    rejects("Policy entry source is not enabled");
    root = YAML::Clone(original);
    const auto source = std::filesystem::path(root["policies"][37].as<std::string>());
    auto policy = YAML::LoadFile(source.string());
    for (auto item : policy["models"])
        item.second = (source.parent_path() / item.second.as<std::string>()).string();
    policy["mimic"]["motion"] = (source.parent_path() / policy["mimic"]["motion"].as<std::string>()).string();
    root["policies"][37] = (directory / "special.yaml").string();
    auto save = [&] { std::ofstream(directory / "special.yaml") << policy; };
    save();
    EXPECT_NO_THROW(load());
    policy["robot"] = "sport";
    save(); rejects("Robot/policy identity mismatch");
    policy["robot"] = "pro";
    policy["entry"]["from_modes"][0] = 28;
    save(); rejects("Special MIMIC mode/entry contract mismatch");
    policy["entry"]["from_modes"][0] = 35;
    policy["mimic"]["heading"]["max_residual_angle"] = 1;
    save(); rejects("MIMIC heading contract mismatch");
    policy["mimic"]["heading"]["max_residual_angle"] = .523599;
    policy["ankle_damping"]["strategy"] = "invalid";
    save(); rejects("Invalid ankle damping strategy");
}

TEST_F(ConfigValidation, SportRejectsProMechanismsAndMismatchedPolicy) {
    useSport();
    EXPECT_EQ(load().pairs.size(), 3U);
    const auto original = YAML::Clone(robot);
    robot["pairs"][2]["kind"] = "neck";
    rejects("Unknown pair mapping");
    robot = YAML::Clone(original);
    const auto base = std::filesystem::path(PRO_CONFIG).parent_path();
    root["policies"][28] = (base / "../policies/amp/pro/walk/config.yaml").string();
    rejects("Robot/policy identity mismatch");
}
TEST_F(ConfigValidation, SportRequiresOriginalPhaseContractAnd100Hz) {
    useSport();
    root["timing"]["inference_hz"] = 50;
    rejects("Policy rate exceeds inference_hz");
    root["timing"]["inference_hz"] = 100;
    const auto source = std::filesystem::path(root["policies"][28].as<std::string>());
    auto p = YAML::LoadFile(source.string());
    for (auto model : p["models"])
        model.second = (source.parent_path() / model.second.as<std::string>()).string();
    root["policies"][28] = (directory / "walk.yaml").string();
    p["amp_gait"]["velocity_filter_hz"] = 5;
    std::ofstream(directory / "walk.yaml") << p;
    rejects("AMP phase contract mismatch");
    p.remove("amp_gait");
    std::ofstream(directory / "walk.yaml") << p;
    rejects("SPORT AMP requires phase observation");
}
TEST_F(ConfigValidation, PostMotionGainsRejectUnsafeRatiosLoopsAndJointSelectors) {
    useSport();
    const auto base = std::filesystem::path(SPORT_CONFIG).parent_path();
    const auto source = base / "../policies/mimic/sport/stand_to_squat/config.yaml";
    auto p = YAML::LoadFile(source.string());
    for (auto model : p["models"])
        model.second = (source.parent_path() / model.second.as<std::string>()).string();
    p["mimic"]["motion"] = (source.parent_path() / p["mimic"]["motion"].as<std::string>()).string();
    root["policies"][42] = (directory / "squat.yaml").string();
    const auto original = YAML::Clone(p);
    auto save = [&] { std::ofstream(directory / "squat.yaml") << p; };
    save(); EXPECT_NO_THROW(load());
    for (const auto &key : {"kp_ratio", "kd_ratio", "duration"}) {
        p = YAML::Clone(original); p["mimic"]["post_motion_gains"][key] = -1;
        save(); rejects(std::string("Invalid ") + key);
    }
    for (const auto &key : {"kp_ratio", "kd_ratio"}) {
        p = YAML::Clone(original); p["mimic"]["post_motion_gains"][key] = 3;
        save(); rejects("Invalid post-motion gain ratios");
    }
    p = YAML::Clone(original); p["mimic"]["loop"] = true;
    save(); rejects("Post-motion gains require non-looping StandToSquat");
    p = YAML::Clone(original); p["mimic"]["post_motion_gains"]["kp_keep_joints"].push_back("l_leg_5");
    save(); rejects("Unknown or duplicate post-motion gain joint");
    p = YAML::Clone(original); p["mimic"]["post_motion_gains"]["kp_keep_joints"].push_back("neck_1");
    save(); rejects("Unknown joint");
}
