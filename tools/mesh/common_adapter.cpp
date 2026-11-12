// SPDX-License-Identifier: GPL-3.0-or-later
#include <cstdio>
#include <stdexcept>
#include "common/logging.h"

void AssertFailSoftImpl() { throw std::runtime_error("shader compiler assertion failed"); }
[[noreturn]] void AssertFatalImpl() { throw std::runtime_error("shader compiler reached an unsupported path"); }
namespace Common::Log {
bool IsLevelEnabled(Class, Level level) { return level >= Level::Warning; }
void FmtLogMessageImpl(Class, Level level, const char*, unsigned, const char*, fmt::string_view format,
                       const fmt::format_args& args) {
    if (level >= Level::Warning) {
        const auto message = fmt::vformat(format, args);
        if (message.find("(STUBBED)") != std::string::npos)
            throw std::runtime_error("incomplete instruction translation: " + message);
        fmt::print(stderr, "mesh-xlate: {}\n", message);
    }
}
}
