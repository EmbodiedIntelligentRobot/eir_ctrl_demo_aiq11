#pragma once
#include <eir_ctrl/config.hpp>
namespace eir::ctrl {
// Configuration validation only; never constructs the hardware SDK or opens IO.
void loadHardware(Config &config, const YAML::Node &runtime);
}
