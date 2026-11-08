// Finds the tiles from where the program is, so a download works from any folder and no build path is compiled in.
#pragma once

#include <filesystem>
#include <string>
#include <system_error>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace streetkit {

inline std::filesystem::path executablePath(const char* argv0) {
    std::error_code ec;
#ifdef _WIN32
    wchar_t buffer[32768];
    const DWORD length = GetModuleFileNameW(nullptr, buffer, 32768);
    if (length > 0 && length < 32768) return std::filesystem::path(buffer, buffer + length);
#else
    const auto self = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (!ec) return self;
#endif
    const auto fallback = std::filesystem::absolute(argv0, ec);
    return ec ? std::filesystem::path(argv0) : fallback;
}

// The first `tiles` folder holding a manifest, beside `exe` or up to two folders above it; empty if none does.
inline std::string tilesNear(const std::filesystem::path& exe) {
    std::error_code ec;
    auto dir = exe.parent_path();
    for (int up = 0; up <= 2; ++up) {
        const auto tiles = dir / "tiles";
        if (std::filesystem::exists(tiles / "manifest.txt", ec)) return tiles.string();
        dir = dir.parent_path();
    }
    return {};
}

inline std::string findTiles(const char* argv0) { return tilesNear(executablePath(argv0)); }

}  // namespace streetkit
