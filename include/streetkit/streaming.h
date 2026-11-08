// Which tiles should be in memory for a viewer at a position. Pure: tested without a GPU.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <vector>

#include "tile.h"

namespace streetkit {

struct StreamConfig {
    float loadRadius = 700;    // a tile this close to the viewer gets loaded
    float unloadRadius = 900;  // and stays until it is this far, so walking along an edge does not thrash
    int maxTiles = 60;         // four meshes a tile, and Strata holds STRATA_MAX_MESHES (256)
};

struct StreamPlan {
    std::vector<TileKey> load;    // nearest first
    std::vector<TileKey> unload;
};

// Distance from a point to the nearest edge of a tile; zero inside it.
inline double tileDistance(TileKey k, double east, double north, double size) {
    const double west = k.ix * size, south = k.iy * size;
    const double dx = std::max({west - east, 0.0, east - (west + size)});
    const double dy = std::max({south - north, 0.0, north - (south + size)});
    return std::hypot(dx, dy);
}

// When the next tile may load. A load costs a frame, so they are spaced out unless the viewer has nothing to stand on.
inline bool loadDue(float secondsSinceLoad, bool tileUnderViewerMissing) {
    return tileUnderViewerMissing || secondsSinceLoad >= 0.12f;
}

// Near and far planes for an eye this far above the ground, for a map with an overview. The near plane lifts with
// height, which keeps depth precise enough at distance to tell the tiles from the overview just under them.
inline void clipPlanes(float aboveGround, float& nearPlane, float& farPlane) {
    nearPlane = std::clamp(aboveGround * 0.05f, 0.1f, 20.0f);
    farPlane = 40000.0f;
}

inline StreamPlan planStreaming(const std::vector<TileKey>& available, const std::set<TileKey>& loaded, double east,
                                double north, double size, const StreamConfig& c) {
    StreamPlan plan;
    for (const TileKey& k : loaded) {
        if (tileDistance(k, east, north, size) > c.unloadRadius) plan.unload.push_back(k);
    }
    std::vector<std::pair<double, TileKey>> wanted;
    for (const TileKey& k : available) {
        const double d = tileDistance(k, east, north, size);
        if (d <= c.loadRadius && loaded.count(k) == 0) wanted.push_back({d, k});
    }
    std::sort(wanted.begin(), wanted.end());
    int room = c.maxTiles - (static_cast<int>(loaded.size()) - static_cast<int>(plan.unload.size()));
    for (const auto& w : wanted) {
        if (room-- <= 0) break;
        plan.load.push_back(w.second);
    }
    return plan;
}

// Copies the items of `all` near (x, z) into `out`, in order: every one within `full` metres, and one in three of
// those out to `reach`, picked by where they stand so a tree's trunk and crown go together and nothing flickers.
template <typename Item>  // anything with a position[3]: Strata's render items, or a test's
void nearItems(const std::vector<Item>& all, float x, float z, float full, float reach, std::vector<Item>& out) {
    out.clear();
    const float f2 = full * full, r2 = reach * reach;
    for (const Item& item : all) {
        const float dx = item.position[0] - x, dz = item.position[2] - z, d2 = dx * dx + dz * dz;
        if (d2 > r2) continue;
        const uint32_t spot = static_cast<uint32_t>(std::lround(item.position[0] * 4.0f)) * 73856093u ^
                              static_cast<uint32_t>(std::lround(item.position[2] * 4.0f)) * 19349663u;
        if (d2 <= f2 || (spot ^ (spot >> 13)) % 3 == 0) out.push_back(item);
    }
}

// How far trees are drawn: far enough on foot to fill a street, further from the air where more of the map shows.
inline float treeReach(float aboveGround) { return std::clamp(300.0f + aboveGround * 2.0f, 300.0f, 800.0f); }
constexpr float kTreesFull = 180.0f;   // every tree this close is drawn

}  // namespace streetkit
