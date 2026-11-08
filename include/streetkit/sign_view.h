// The street signs and street name for this moment, for the desktop viewer and the browser alike.
#pragma once

#include <cmath>
#include <string>
#include <vector>

#include <strata/strata.h>

#include "streets.h"
#include "tile_stream.h"
#include "walker.h"

namespace streetkit {

constexpr double kStreetReach = 35;              // metres from a road's middle that still counts as being on it
constexpr double kSignsOnFoot = 80, kSignsFlying = 220;  // how far away a sign shows
constexpr size_t kMostSignsOnFoot = 6, kMostSignsFlying = 10;
constexpr float kPostHeight = 3.0f, kSignPixels = 16;

class SignView {
public:
    struct Label {
        float at[3];
        std::string text;
    };

    // Looks again. Finding them costs a walk over every road, so this runs a few times a second and not each frame.
    void update(const Streets& streets, const Stream& stream, const Walker& w, bool flying) {
        const double east = stream.east(w.x), north = stream.north(w.z);
        const int id = streets.streetAt(east, north, kStreetReach);
        street_ = id >= 0 ? streets.names[static_cast<size_t>(id)] : "";
        items_.clear();
        labels_.clear();
        const double originEast = stream.east(0), originNorth = stream.north(0);
        auto worldX = [&](const StreetSign& s) { return static_cast<float>(s.east - originEast); };
        auto worldZ = [&](const StreetSign& s) { return static_cast<float>(originNorth - s.north); };
        // A sign behind a building stays hidden on foot. From the air the names are a map, and the buildings are below.
        auto seen = [&](const StreetSign& s) {
            const float x = worldX(s), z = worldZ(s);
            if (!stream.walkable(x, z)) return false;  // its tile has not arrived
            if (flying) return true;
            const float dx = x - w.x, dz = z - w.z;
            const int steps = static_cast<int>(std::hypot(dx, dz) / 3.0f);
            for (int i = 1; i < steps; ++i) {
                const float u = static_cast<float>(i) / static_cast<float>(steps);
                if (stream.solids().inside(w.x + dx * u, w.z + dz * u)) return false;
            }
            return true;
        };
        for (const StreetSign* s : streets.signsNear(east, north, flying ? kSignsFlying : kSignsOnFoot,
                                                     flying ? kMostSignsFlying : kMostSignsOnFoot, seen)) {
            const float x = worldX(*s), z = worldZ(*s), y = stream.groundAt(x, z);
            const float green[4] = {0.03f, 0.42f, 0.17f, 1}, grey[4] = {0.3f, 0.3f, 0.3f, 1};
            add(STRATA_PRIMITIVE_CYLINDER, x, y + kPostHeight * 0.5f, z, 0.1f, kPostHeight * 0.5f, 0.1f, grey);
            add(STRATA_PRIMITIVE_CUBE, x, y + kPostHeight, z, 1.5f, 0.28f, 0.05f, green);
            add(STRATA_PRIMITIVE_CUBE, x, y + kPostHeight - 0.32f, z, 0.05f, 0.28f, 1.5f, green);
            labels_.push_back({{x, y + kPostHeight + 0.35f, z}, streets.signText(*s)});
        }
    }

    const std::string& street() const { return street_; }
    const std::vector<StrataRenderItem>& items() const { return items_; }
    const std::vector<Label>& labels() const { return labels_; }

private:
    void add(int kind, float x, float y, float z, float sx, float sy, float sz, const float c[4]) {
        StrataRenderItem item{};
        item.kind = kind;
        item.position[0] = x;
        item.position[1] = y;
        item.position[2] = z;
        item.scale[0] = sx;
        item.scale[1] = sy;
        item.scale[2] = sz;
        for (int i = 0; i < 4; ++i) item.color[i] = c[i];
        items_.push_back(item);
    }

    std::string street_;
    std::vector<StrataRenderItem> items_;
    std::vector<Label> labels_;
};

}  // namespace streetkit
