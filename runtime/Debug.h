#pragma once

#include <spdlog/spdlog.h>

template<typename... Args>
void DebugVerbose(const spdlog::format_string_t<Args...> &format, Args &&...args) {
    spdlog::trace(format, std::forward<Args>(args)...);
}

template<typename... Args>
void DebugInfo(const spdlog::format_string_t<Args...> &format, Args &&...args) {
    spdlog::info(format, std::forward<Args>(args)...);
}

template<typename... Args>
void DebugWarning(const spdlog::format_string_t<Args...> &format, Args &&...args) {
    spdlog::warn(format, std::forward<Args>(args)...);
}

template<typename... Args>
void DebugError(const spdlog::format_string_t<Args...> &format, Args &&...args) {
    spdlog::error(format, std::forward<Args>(args)...);
}

template<typename... Args>
bool DebugCheck(const bool succeeded, const spdlog::format_string_t<Args...> &failMessage, Args &&...args) {
    if (succeeded) return true;
    DebugWarning(failMessage, std::forward<Args>(args)...);
    return false;
}

template<typename... Args>
void DebugCheckCritical(const bool succeeded, const spdlog::format_string_t<Args...> &failMessage, Args &&...args) {
    if (succeeded) return;
    DebugError(failMessage, std::forward<Args>(args)...);
    exit(EXIT_FAILURE);
}
