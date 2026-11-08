// The handlebars ahead of a rider, as lines in the world, for the desktop viewer and the browser alike.
#pragma once

#include <array>
#include <cmath>
#include <vector>

#include "walker.h"

namespace streetkit {

struct BikeSegment {
    std::array<float, 3> from, to;
    float grey, width;
};

inline std::vector<BikeSegment> handlebars(const Bike& bike, float x, float eyeY, float z) {
    const float pi = 3.14159265f, yaw = bike.heading * pi / 180.0f, bars = yaw + bike.steer * 25.0f * pi / 180.0f;
    // Low in a level view: 0.8 m ahead and 0.4 m down is 27 degrees below the eye, inside the default field of view.
    const float cx = x + std::sin(yaw) * 0.8f, cy = eyeY - 0.4f, cz = z - std::cos(yaw) * 0.8f;
    const float rx = std::cos(bars), rz = std::sin(bars);
    auto along = [&](float reach) { return std::array<float, 3>{cx + rx * reach, cy, cz + rz * reach}; };
    return {{{cx, cy, cz}, {cx + std::sin(bars) * 0.14f, cy - 0.8f, cz - std::cos(bars) * 0.14f}, 0.3f, 0.05f},
            {along(-0.2f), along(0.2f), 0.5f, 0.028f},
            {along(-0.3f), along(-0.2f), 0.02f, 0.038f},
            {along(0.2f), along(0.3f), 0.02f, 0.038f}};
}

}  // namespace streetkit
