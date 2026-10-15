// Offline manager contract: no Runtime, ROS, control threads or hardware SDK Init.
#include <eir_ctrl/config.hpp>
#include <eir_ctrl/mapping.hpp>
#include <eir_service/health.hpp>
#include <openssl/evp.h>
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;
namespace {
std::string hex(const unsigned char *bytes, unsigned length) {
    std::ostringstream result;
    for (unsigned i = 0; i < length; ++i)
        result << std::hex << std::setfill('0') << std::setw(2) << unsigned(bytes[i]);
    return result.str();
}
std::string hashBody(const std::string &body) {
    unsigned char bytes[EVP_MAX_MD_SIZE];
    unsigned length{};
    if (!EVP_Digest(body.data(), body.size(), bytes, &length, EVP_sha256(), nullptr) || length != 32)
        throw std::runtime_error("Cannot hash resource recipe");
    return hex(bytes, length);
}
std::string hash(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!in || !ctx || EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) != 1)
        throw std::runtime_error("Cannot hash resource: " + path.string());
    std::array<char, 65536> buffer{};
    while (in) {
        in.read(buffer.data(), buffer.size());
        if (in.gcount() && EVP_DigestUpdate(ctx.get(), buffer.data(), in.gcount()) != 1)
            throw std::runtime_error("Cannot hash resource: " + path.string());
    }
    unsigned char bytes[EVP_MAX_MD_SIZE];
    unsigned length{};
    if (in.bad() || EVP_DigestFinal_ex(ctx.get(), bytes, &length) != 1 || length != 32)
        throw std::runtime_error("Cannot finish resource hash: " + path.string());
    return hex(bytes, length);
}
bool within(const fs::path &path, const fs::path &root) {
    const auto relative = path.lexically_relative(root);
    return !relative.empty() && *relative.begin() != "..";
}
struct Resources {
    fs::path release, config;
    std::string backend;
    std::set<fs::path> visited, files;

    fs::path check(const fs::path &path, bool immutable = false) {
        const auto real = fs::canonical(path);
        if (!within(real, release) && (immutable || !within(real, config)))
            throw std::runtime_error("Resource outside immutable release/config: " + path.string());
        if (fs::is_regular_file(real)) files.insert(real);
        return real;
    }
    // Follow YAML inputs before loadConfig; bound recursion and deny external includes.
    void yaml(const fs::path &path, unsigned depth = 0) {
        if (depth > 12) throw std::runtime_error("Resource YAML nesting exceeds 12");
        const auto real = check(path);
        if (!visited.insert(real).second) return;
        const auto root = YAML::LoadFile(real.string());
        scan(root, real.parent_path(), depth);
        if (root["kind"] && root["models"] && !root["perception"] &&
            (root["models"]["libtorch"].IsScalar() || root["models"]["onnx"].IsScalar()))
            check(real.parent_path() / (root["contract"] ? root["contract"].as<std::string>() : "model_contract.yaml"), true);
    }
    void scan(const YAML::Node &node, const fs::path &base, unsigned depth) {
        if (node.IsSequence()) {
            for (const auto &value : node) scan(value, base, depth);
            return;
        }
        if (!node.IsMap()) return;
        for (const auto &item : node) {
            const auto key = item.first.as<std::string>();
            const auto value = item.second;
            if ((key == "robot" || key == "flight_recorder" || key == "motion_stream" ||
                 key == "manifest" || key == "contract" || key == "file" || key == "resources" ||
                 key == "hardware_config" || key == "model" || key == "motion") && value.IsScalar()) {
                const auto text = value.as<std::string>();
                // These keys can also contain plain robot identities or enums.
                if (text.find('/') != std::string::npos || fs::path(text).has_extension()) {
                    const auto resource = check(base / text);
                    if (resource.extension() == ".yaml" || resource.extension() == ".yml")
                        yaml(resource, depth + 1);
                }
            } else if (key == "policies" && value.IsMap()) {
                for (const auto &policy : value) {
                    auto profile = base / policy.second.as<std::string>();
                    const int wire = policy.first.as<int>();
                    if (wire >= 30 && wire <= 34 && node["mimic_variant"]) {
                        const auto variant = node["mimic_variant"].as<std::string>();
                        if (variant == "auto")
                            yaml(check(profile.parent_path() / "config_gripper.yaml", true), depth + 1);
                        profile = profile.parent_path() / (variant == "gripper" ? "config_gripper.yaml" : "config_fist.yaml");
                    }
                    yaml(check(profile, true), depth + 1);
                }
            } else if (key == "models" && value.IsMap()) {
                if (value[backend] && value[backend].IsScalar())
                    check(base / value[backend].as<std::string>(), true);
            } else {
                scan(value, base, depth);
            }
        }
    }
};
} // namespace
int main(int argc, char **argv) {
    try {
        if (argc != 4) throw std::invalid_argument("usage: resource-check CONFIG RELEASE LOG_ROOT");
        const auto config = fs::canonical(argv[1]);
        const auto root = YAML::LoadFile(config.string());
        const auto backend = root["inference"]["backend"].as<std::string>();
        Resources resources{fs::canonical(argv[2]), config.parent_path(), backend, {}, {}};
        resources.yaml(config);
        const auto c = eir::ctrl::loadConfig(config.string());
        const bool hardware = c.runtime.backend == eir::loco::BackendType::Hardware;
        const auto runtime_resource = resources.check(
            hardware ? c.runtime.hardware_config : c.runtime.model_path, true);
        for (const auto &pair : c.pairs) resources.check(pair.resources, true);
        if (c.runtime.flight_recorder.enabled) {
            const auto log = fs::weakly_canonical(argv[3]);
            const auto output = fs::weakly_canonical(c.runtime.flight_recorder.directory);
            if (!within(output, log))
                throw std::runtime_error("Recorder output must remain under instance log/");
        }
        eir::ctrl::Mapping mapping(c);
        // Validate both declared attachments even when startup chooses only one.
        bool dual = false;
        for (const auto &[mode, p] : c.policies) {
            if (p.mimic_variant.empty()) continue;
            dual = true;
            const auto folder = (config.parent_path() / root["policies"][int(mode)].as<std::string>()).parent_path();
            resources.yaml(folder / "config_fist.yaml");
            resources.yaml(folder / "config_gripper.yaml");
        }
        const auto fist = dual ? eir::ctrl::loadConfig(config.string(), "fist") : c;
        const auto gripper = dual ? eir::ctrl::loadConfig(config.string(), "gripper") : c;
        auto fingerprint = [&](const eir::ctrl::PolicyConfig &p, const fs::path &profile) {
            Resources policy{resources.release, resources.config, backend, {}, {}};
            policy.yaml(profile);
            if (p.split_perception) {
                policy.check(p.encoder.model, true); policy.check(p.actor.model, true);
            } else policy.check(p.inference.model, true);
            if (!p.mimic.motion.empty()) policy.check(p.mimic.motion, true);
            auto checked = p;
            checked.inference.warmup_steps = std::max(1, checked.inference.warmup_steps);
            checked.encoder.warmup_steps = std::max(1, checked.encoder.warmup_steps);
            checked.actor.warmup_steps = std::max(1, checked.actor.warmup_steps);
            auto loaded = eir::ctrl::makePolicy(checked);
            std::string recipe;
            for (const auto &file : policy.files)
                recipe += file.lexically_relative(resources.release).string() + ":" + hash(file) + "\n";
            return hashBody(recipe);
        };
        std::ostringstream fingerprints, variants;
        bool first = true, first_variant = true;
        for (const auto &[mode, p] : c.policies) {
            const auto name = "mode" + std::to_string(int(mode));
            const auto profile = config.parent_path() / root["policies"][int(mode)].as<std::string>();
            std::string digest;
            if (!p.mimic_variant.empty()) {
                const auto a = fingerprint(fist.policies.at(mode), profile.parent_path() / "config_fist.yaml");
                const auto b = fingerprint(gripper.policies.at(mode), profile.parent_path() / "config_gripper.yaml");
                digest = hashBody("fist:" + a + ":gripper:" + b);
                if (!first_variant) variants << ',';
                first_variant = false;
                variants << eir::service::quote(name) << ":{\"fist\":" << eir::service::quote(a)
                         << ",\"gripper\":" << eir::service::quote(b) << '}';
            } else digest = fingerprint(p, profile);
            if (!first) fingerprints << ',';
            first = false;
            fingerprints << eir::service::quote(name) << ':' << eir::service::quote(digest);
        }
        std::cout << "{\"protocol\":1,\"robot\":" << eir::service::quote("aiq11-" + c.robot)
                  << ",\"backend\":" << eir::service::quote(hardware ? "hardware" : "simulation")
                  << ",\"config_sha256\":" << eir::service::quote(hash(config))
                  << ",\"runtime_resource\":" << eir::service::quote(runtime_resource.string())
                  << ",\"models\":\"loaded\",\"policies\":{" << fingerprints.str()
                  << "},\"policy_variants\":{" << variants.str() << "}}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
