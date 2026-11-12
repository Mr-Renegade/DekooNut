// SPDX-License-Identifier: GPL-3.0-or-later
// Standalone compiler settings: no emulator configuration or game-specific visual patches.
#pragma once
#include <cstdint>
namespace Settings {
struct Boolean {
    bool value{};
    bool GetValue() const { return value; }
    operator bool() const { return value; }
};
struct Resolution {
    bool active{};
    std::uint32_t up_scale{1}, down_shift{};
    float up_factor{1}, down_factor{1};
};
struct Values {
    Boolean disable_shader_loop_safety_checks{}, fix_bloom_effects{}, renderer_debug{};
    Resolution resolution_info{};
};
inline Values values;
}
