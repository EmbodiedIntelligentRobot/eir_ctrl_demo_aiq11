#include "hardware.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <set>
#include <stdexcept>

namespace eir::ctrl {
namespace {
void require(bool condition, const std::string &message) {
    if (!condition) throw std::invalid_argument(message);
}
}
void loadHardware(Config &c, const YAML::Node &rt) {
    const bool pro = c.robot == "pro";
    require((pro && c.control.count == 32) || (c.robot == "sport" && c.control.count == 23),
            "Hardware robot/joint identity mismatch");
    const std::string version = "aiq11-" + c.robot;
    require(std::filesystem::is_regular_file(c.runtime.hardware_config), "Missing hardware SDK configuration");
    const auto sdk = YAML::LoadFile(c.runtime.hardware_config)["humanoid_hardware_node"];
    require(sdk && sdk["RobotVersion"].as<std::string>("") == version,
            "Hardware SDK RobotVersion must be " + version);
    const auto profile = sdk["Versions"][version];
    require(profile && profile["TotalMotorCount"].as<int>(0) == 32 &&
            profile["ActiveMotorCount"].as<std::size_t>(0) == c.control.count,
            "SDK active motor count does not match the robot");
    std::set<int> sdk_slots;
    std::set<std::string> sdk_names;
    std::array<std::string, 32> slot_names;
    const std::map<std::string, std::pair<int, int>> board_slots{
        {"LeftArm", {0, 6}}, {"RightArm", {7, 13}}, {"Waist", {14, 16}},
        {"LeftLeg1", {17, 19}}, {"LeftLeg2", {20, 22}},
        {"RightLeg1", {23, 25}}, {"RightLeg2", {26, 28}}, {"Neck", {29, 31}}};
    require(profile["JointMapping"].IsMap() && profile["JointMapping"].size() == board_slots.size(),
            "SDK JointMapping requires all eight AIQ board groups");
    for (const auto &group : profile["JointMapping"]) {
        require(group.second.IsMap(), "Invalid SDK motor group");
        const auto board = board_slots.find(group.first.as<std::string>());
        require(board != board_slots.end(), "Unknown SDK motor board");
        std::set<int> addresses;
        for (const auto &entry : group.second) {
            const int slot = entry.first.as<int>();
            const auto motor = entry.second;
            require(slot >= 0 && slot < 32 && sdk_slots.insert(slot).second, "Invalid or duplicate SDK motor slot");
            require(slot >= board->second.first && slot <= board->second.second,
                    "SDK motor slot is assigned to the wrong board");
            const auto name = motor["name"].as<std::string>("");
            require(!name.empty() && sdk_names.insert(name).second, "Invalid or duplicate SDK motor name");
            slot_names[slot] = name;
            const auto type = motor["type"].as<std::string>("");
            // SPORT keeps the SDK's full layout; distal arms and neck are disabled.
            const bool active = pro || slot < 4 || (slot >= 7 && slot <= 10) ||
                                (slot >= 14 && slot <= 28);
            require(!type.empty() && (type != "NO_Type") == active,
                    "SDK active/disabled slot does not match the robot");
            const auto address = motor["can_id"].as<int>();
            require(address > 0 && addresses.insert(address).second, "Invalid or duplicate SDK CAN address");
            const auto direction = motor["dir"].as<int>();
            const auto ratio = motor["gear_ratio"].as<double>();
            const auto offset = motor["offset"].as<double>();
            require((direction == -1 || direction == 1) && std::isfinite(ratio) && ratio > 0 &&
                    std::isfinite(offset), "Invalid SDK direction, gear ratio or offset");
        }
    }
    require(sdk_slots.size() == 32, "Incomplete SDK motor slots");
    const auto slots = rt["motor_slots"];
    require(slots.IsMap() && slots.size() == c.control.count, "Explicit motor_slots mapping must cover every active motor");
    // Public AIQ topology: application actuator names versus SDK board-local names.
    const std::map<std::string, std::string> sdk_aliases{
        {"l_arm_1", "la_1"}, {"l_arm_2", "la_2"}, {"l_arm_3", "la_3"}, {"l_arm_4", "la_4"},
        {"l_arm_5", "la_5"}, {"l_arm_51", "la_6"}, {"l_arm_7", "la_7"},
        {"r_arm_1", "ra_1"}, {"r_arm_2", "ra_2"}, {"r_arm_3", "ra_3"}, {"r_arm_4", "ra_4"},
        {"r_arm_5", "ra_5"}, {"r_arm_51", "ra_6"}, {"r_arm_7", "ra_7"},
        {"torso_1", "waist_1"}, {"torso_31", "waist_2"}, {"torso_33", "waist_3"},
        {"l_leg_1", "ll1_1"}, {"l_leg_2", "ll1_2"}, {"l_leg_3", "ll1_3"},
        {"l_leg_4", "ll2_4"}, {"l_leg_41", "ll2_5"}, {"l_leg_43", "ll2_6"},
        {"r_leg_1", "rl1_1"}, {"r_leg_2", "rl1_2"}, {"r_leg_3", "rl1_3"},
        {"r_leg_4", "rl2_4"}, {"r_leg_41", "rl2_5"}, {"r_leg_43", "rl2_6"},
        {"neck_1", "n_1"}, {"neck_11", "n_2"}, {"neck_13", "n_3"}};
    std::set<std::string> mapped_names;
    std::set<int> mapped_slots;
    for (const auto &entry : slots) {
        const auto name = entry.first.as<std::string>();
        const auto slot = entry.second.as<int>();
        require(mapped_names.insert(name).second &&
                std::find(c.runtime.motor_names.begin(), c.runtime.motor_names.end(), name) != c.runtime.motor_names.end(),
                "Unknown or duplicate motor_slots name: " + name);
        require(slot >= 0 && slot < 32 && mapped_slots.insert(slot).second, "Invalid or duplicate motor_slots value");
        require(slot_names[slot] == sdk_aliases.at(name), "Motor/SDK slot identity mismatch: " + name);
    }
    // JointMapping/YAML uses physical board slots; RobotState/RobotCmd already
    // use SDK logical indices: left leg, right leg, waist, left arm, right arm, neck.
    // Convert once at the application boundary for both feedback and commands.
    constexpr std::array<std::size_t, 32> physical_to_logical{
        15,16,17,18,19,20,21, 22,23,24,25,26,27,28, 12,13,14,
        0,1,2,3,4,5, 6,7,8,9,10,11, 29,30,31};
    for (const auto &name : c.runtime.motor_names)
        c.runtime.motor_slots.push_back(physical_to_logical.at(slots[name].as<std::size_t>()));
    c.runtime.encoder_names = rt["encoder_names"].as<std::vector<std::string>>();
    require(c.runtime.encoder_names.size() == 2 &&
            std::set<std::string>(c.runtime.encoder_names.begin(), c.runtime.encoder_names.end()) ==
                std::set<std::string>{"torso_3", "torso_2"}, "AIQ encoder_names must map waist pitch and roll");
    std::set<int> encoders;
    std::set<int> encoder_ids;
    require(profile["EncoderMapping"].IsMap() && profile["EncoderMapping"].size() == 1,
            "SDK EncoderMapping requires the waist board");
    for (const auto &group : profile["EncoderMapping"]) {
        require(group.first.as<std::string>() == "Waist" && group.second.IsMap(), "Invalid SDK encoder group");
        for (const auto &entry : group.second) {
            const int slot = entry.first.as<int>();
            require(slot >= 0 && slot < 2 && encoders.insert(slot).second, "Invalid or duplicate SDK encoder slot");
            const auto direction = entry.second["dir"].as<int>();
            require(direction == -1 || direction == 1, "Invalid SDK encoder direction");
            const auto id = entry.second["id"].as<int>();
            require(id >= 0 && id < 2 && encoder_ids.insert(id).second, "Invalid or duplicate SDK encoder ID");
            const auto name = entry.second["name"].as<std::string>("");
            require(c.runtime.encoder_names[slot] == "torso_3" ? name == "waist_p" :
                    (name == "wasit_r" || name == "waist_r"), "Waist encoder/SDK identity mismatch");
        }
    }
    require(encoders.size() == 2, "AIQ requires two SDK encoders");
    if (pro && c.mimic_variant == "auto") {
        const auto ee = profile["EndEffectorMapping"];
        c.mimic_hand_type = ee ? ee["type"].as<std::string>("") : "";
        require(c.mimic_hand_type == "LEGACY" || c.mimic_hand_type == "NO_TYPE",
                "Auto MIMIC requires explicit LEGACY or NO_TYPE SDK mapping");
        for (const auto *side : {"Left", "Right"})
            require(!ee[side] || !ee[side]["type"] ||
                    ee[side]["type"].as<std::string>() == c.mimic_hand_type,
                    "Conflicting MIMIC attachment SDK side type");
    }
    if (c.hand.enabled) {
        const auto ee = profile["EndEffectorMapping"];
        require(ee && ee["type"].as<std::string>("") == "LEGACY", "PRO hands require explicit LEGACY SDK mapping");
        for (const auto *side : {"Left", "Right"})
            require(!ee[side] || !ee[side]["type"] || ee[side]["type"].as<std::string>() == "LEGACY",
                    "Conflicting PRO hand SDK type");
    }
    c.runtime.hardware_family = loco::HardwareFamily::Aiq;
    c.runtime.hardware_robot = version;
    c.runtime.hardware_motor_slots = 32;
    c.runtime.encoder_count = 2;
}
loco::Config runtimeConfig(const Config &config) {
    auto runtime = config.runtime;
    if (runtime.backend == loco::BackendType::Hardware) {
        runtime.joint_names = runtime.motor_names;
        runtime.encoder_names.clear(); // Mapping reads and validates the original SDK encoders.
    }
    return runtime;
}
} // namespace eir::ctrl
