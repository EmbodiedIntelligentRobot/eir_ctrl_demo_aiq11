#include <eir_ctrl/control.hpp>
#include <stdexcept>
namespace eir::ctrl {
std::unique_ptr<Policy> makeAmp(PolicyConfig, std::unique_ptr<Inference>);
std::unique_ptr<Policy> makeMimic(PolicyConfig, std::unique_ptr<Inference>);
std::unique_ptr<Policy> makePerception(PolicyConfig, std::unique_ptr<Inference>);
std::unique_ptr<Policy> makePolicy(PolicyConfig c, std::unique_ptr<Inference> inference) {
    const bool mimic = c.kind == "mimic_gravity_v1";
    const bool perception = c.kind == "perception_walk_v2";
    if ((c.robot != "pro" && !(c.robot == "sport" &&
          (mimic || (c.kind == "amp_humanoid_walk" && c.amp_gait.enabled)))) ||
        (!mimic && !perception && c.kind != "amp_humanoid_walk"))
        throw std::invalid_argument("Policy profile has not been implemented/validated");
    const auto n = c.indices.size();
    const auto frame = mimic ? 21 + 5 * n : 9 + 3 * n + (c.amp_gait.enabled ? 2 : 0);
    if (!n || !c.history || c.defaults.size() != n || c.scales.size() != n ||
        c.inference.output_size != n ||
        c.inference.input_size != frame * c.history + (perception ? 4608 : 0))
        throw std::invalid_argument("Policy observation/action contract mismatch");
    if (perception && (n != 23 || c.history != 8 || !c.depth_timeout_ns || !c.execution_timeout_ns))
        throw std::invalid_argument("Perception requires 23 joints, 8 histories and bounded timeouts");
    if (!inference) {
        if (perception && c.split_perception)
            inference = makeSplitInference(c.encoder, c.actor, c.encoder_hz,
                                            c.execution_timeout_ns, c.depth_timeout_ns);
        else inference = makeInference(c.inference);
    }
    if (perception) return makePerception(std::move(c), std::move(inference));
    return mimic ? makeMimic(std::move(c), std::move(inference))
                 : makeAmp(std::move(c), std::move(inference));
}
} // namespace eir::ctrl
