// Street lamps and lit windows, worked out from the sun and the map. Pure: tested without a GPU.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "streets.h"
#include "tile.h"

namespace streetkit {

constexpr float kLampSpacing = 45;     // metres along a road between one lamp and the next
constexpr float kLampSide = 7;         // metres from the road's middle, to the verge
constexpr float kLampKerb = 5;         // no lamp this close to the middle of any named road
constexpr float kLampGap = 20;         // two roads running side by side do not each get a lamp here
constexpr size_t kMostLamps = 1800;
constexpr float kLampHeight = 4.5f;
constexpr float kPoolRadius = 18;      // the light on the ground around a lamp
constexpr float kPoolLift = 0.35f;     // above the road, which is 0.12 above the ground, and paths at 0.2
constexpr int kPoolGrid = 5;           // vertices along a side, so that a pool follows the ground

constexpr float kFloorHeight = 3.3f, kBayWidth = 3.6f;  // how a wall divides into windows
constexpr float kWindowWidth = 1.3f, kWindowHeight = 1.5f, kWindowSill = 1.0f;
constexpr float kWindowLift = 0.2f;    // off the wall, so that it still sorts in front from far away
constexpr float kShortestLitWall = 3;  // metres along, and eaves lower than the next are sheds
constexpr float kLowestLitEave = 2.8f;
constexpr uint32_t kLitBuildings = 45, kLitWindows = 40;  // in a hundred: buildings with anyone home, and their windows
constexpr size_t kMostWindows = 1500;  // in a tile
constexpr int kWindowStages = 8;       // windows come on in this many groups

inline float smoothBetween(float from, float to, float x) {
    const float t = std::clamp((x - from) / (to - from), 0.0f, 1.0f);
    return t * t * (3 - 2 * t);
}

// Degrees above the horizon. Strata's time-of-day sun: no season, up at 0.25 and down at 0.75.
inline float sunElevation(float timeOfDay, float latitudeDegrees) {
    constexpr float kPi = 3.14159265f;
    const float t = timeOfDay - std::floor(timeOfDay), hour = (t - 0.5f) * 2 * kPi, latitude = latitudeDegrees * kPi / 180;
    const float side = std::sin(hour), up = std::cos(latitude) * std::cos(hour), along = std::sin(latitude) * std::cos(hour);
    return std::asin(std::clamp(up / std::sqrt(side * side + up * up + along * along), -1.0f, 1.0f)) * 180 / kPi;
}

// 0 by day, 1 at night: the lamps come on from a sun 6 degrees up and are full once it is 6 degrees down.
inline float nightLevel(float elevationDegrees) { return smoothBetween(6, -6, elevationDegrees); }

struct Lamp {
    double east = 0, north = 0;
};

// Whether a point is on a named road's verge or closer. Streets::streetAt answers this too, but this runs thousands of
// times as the map opens, on a phone as well, so a box around each road rules out nearly all of them.
class RoadBoxes {
public:
    explicit RoadBoxes(const Streets& streets) : streets_(streets) {
        for (const StreetRoad& r : streets.roads) {
            Box b{r.en.empty() ? 0 : r.en[0], r.en.empty() ? 0 : r.en[1], r.en.empty() ? 0 : r.en[0], r.en.empty() ? 0 : r.en[1]};
            for (size_t i = 0; i + 1 < r.en.size(); i += 2) {
                b.west = std::min(b.west, r.en[i]);
                b.east = std::max(b.east, r.en[i]);
                b.south = std::min(b.south, r.en[i + 1]);
                b.north = std::max(b.north, r.en[i + 1]);
            }
            boxes_.push_back(b);
        }
    }
    bool within(double east, double north, double reach) const {
        for (size_t k = 0; k < boxes_.size(); ++k) {
            const Box& b = boxes_[k];
            const StreetRoad& r = streets_.roads[k];
            if (east < b.west - reach || east > b.east + reach || north < b.south - reach || north > b.north + reach || !streets_.named(r.street)) continue;
            for (size_t i = 0; i + 3 < r.en.size(); i += 2) {
                const double ax = r.en[i], ay = r.en[i + 1], dx = r.en[i + 2] - ax, dy = r.en[i + 3] - ay;
                const double length = dx * dx + dy * dy;
                const double t = length > 0 ? std::clamp(((east - ax) * dx + (north - ay) * dy) / length, 0.0, 1.0) : 0.0;
                const double px = east - ax - dx * t, py = north - ay - dy * t;
                if (px * px + py * py < reach * reach) return true;
            }
        }
        return false;
    }

private:
    struct Box {
        double west, south, east, north;
    };
    const Streets& streets_;
    std::vector<Box> boxes_;
};

// Lamps along the named roads, a lamp every kLampSpacing metres, swapping sides as real ones do, never on a roadway.
inline std::vector<Lamp> placeLamps(const Streets& streets) {
    std::vector<Lamp> lamps;
    const RoadBoxes roads(streets);
    auto clear = [&](double east, double north) {
        if (roads.within(east, north, kLampKerb)) return false;
        for (const Lamp& l : lamps)
            if (std::hypot(l.east - east, l.north - north) < kLampGap) return false;
        return true;
    };
    for (const StreetRoad& r : streets.roads) {
        if (!streets.named(r.street)) continue;
        double next = 0, walked = 0;  // from the road's first metre: ways are split at every crossing, many shorter than a gap
        int side = 1;
        for (size_t i = 0; i + 3 < r.en.size(); i += 2) {
            const double ax = r.en[i], ay = r.en[i + 1], dx = r.en[i + 2] - ax, dy = r.en[i + 3] - ay, length = std::hypot(dx, dy);
            if (length <= 0) continue;
            const double ux = dx / length, uy = dy / length;
            for (; next <= walked + length; next += kLampSpacing, side = -side) {
                const double along = next - walked, x = ax + ux * along, y = ay + uy * along;
                for (int s : {side, -side}) {  // the far verge when this one is taken
                    const double east = x - uy * s * kLampSide, north = y + ux * s * kLampSide;
                    if (lamps.size() < kMostLamps && clear(east, north)) {
                        lamps.push_back({east, north});
                        break;
                    }
                }
            }
            walked += length;
        }
    }
    return lamps;
}

// Triangles for strata_add_textured_mesh.
struct GlowMesh {
    std::vector<float> positions, normals, colors, uvs;
    std::vector<uint32_t> indices;

    void vertex(float x, float y, float z, const float n[3], const float c[3], float u, float v) {
        positions.insert(positions.end(), {x, y, z});
        normals.insert(normals.end(), n, n + 3);
        colors.insert(colors.end(), c, c + 3);
        uvs.insert(uvs.end(), {u, v});
    }
    size_t vertices() const { return positions.size() / 3; }
    bool empty() const { return indices.empty(); }
};

// The warm disc of ground around a lamp at world x, z. height(x, z) is the ground. It is a texture on a patch of mesh,
// so that every lamp on the map has one for one draw a tile, where real lights are few.
template <typename Height>
inline void addPool(GlowMesh& out, float x, float z, Height&& height) {
    const float up[3] = {0, 1, 0}, warm[3] = {1.0f, 0.68f, 0.34f};
    const uint32_t first = static_cast<uint32_t>(out.vertices()), n = kPoolGrid;
    for (int row = 0; row < kPoolGrid; ++row) {
        for (int col = 0; col < kPoolGrid; ++col) {
            const float u = static_cast<float>(col) / (kPoolGrid - 1), v = static_cast<float>(row) / (kPoolGrid - 1);
            const float px = x + (u * 2 - 1) * kPoolRadius, pz = z + (v * 2 - 1) * kPoolRadius;
            out.vertex(px, height(px, pz) + kPoolLift, pz, up, warm, u, v);
        }
    }
    for (uint32_t row = 0; row + 1 < n; ++row) {
        for (uint32_t col = 0; col + 1 < n; ++col) {
            const uint32_t a = first + row * n + col, b = a + 1, c = a + n, d = c + 1;
            out.indices.insert(out.indices.end(), {a, c, b, b, c, d});
        }
    }
}

inline uint32_t mixHash(uint32_t hash, uint32_t value) {
    for (int i = 0; i < 4; ++i, value >>= 8) hash = (hash ^ (value & 255)) * 16777619u;
    return hash ^ (hash >> 15);
}

// The lit windows of a tile's buildings mesh, on its walls, world positions being tile positions plus (ox, oz). A
// building is a run of walls ending at its roof, as streetkit/mesh.py writes them, and its first corner names it,
// so the same windows light on every run. The texture's u picks a stage, for them not all to come on at once.
inline GlowMesh litWindows(const TileMesh& buildings, float ox, float oz) {
    GlowMesh out;
    const auto& p = buildings.positions;
    const auto& idx = buildings.indices;
    uint32_t building = 0;
    bool afterRoof = true;
    uint32_t wallNumber = 0;
    for (size_t t = 0; t + 2 < idx.size();) {
        const uint32_t first = idx[t];  // a face is a fan: its triangles share their first vertex
        for (t += 3; t + 2 < idx.size() && idx[t] == first;) t += 3;
        if ((static_cast<size_t>(first) + 1) * 4 > buildings.normals.size() || (static_cast<size_t>(first) + 3) * 3 > p.size()) continue;
        const int8_t* n = &buildings.normals[first * 4];
        if (std::abs(static_cast<int>(n[1])) > 12) {  // a roof
            afterRoof = true;
            continue;
        }
        const float nx = static_cast<float>(n[0]) / 127.0f, nz = static_cast<float>(n[2]) / 127.0f;
        const float ax = p[first * 3], base = p[first * 3 + 1], az = p[first * 3 + 2];
        const float bx = p[(first + 1) * 3], bz = p[(first + 1) * 3 + 2], eave = p[(first + 2) * 3 + 1];
        if (afterRoof) {
            building = mixHash(mixHash(2166136261u, static_cast<uint32_t>(std::lround(ax * 100))), static_cast<uint32_t>(std::lround(az * 100)));
            afterRoof = false;
            wallNumber = 0;
        }
        const uint32_t wall = mixHash(building, wallNumber++);
        const float length = std::hypot(bx - ax, bz - az), tall = eave - base;
        if (building % 100 >= kLitBuildings || length < kShortestLitWall || tall < kLowestLitEave) continue;
        const int bays = std::max(1, static_cast<int>(length / kBayWidth)), floors = std::max(1, static_cast<int>(tall / kFloorHeight));
        const float ux = (bx - ax) / length, uz = (bz - az) / length;
        const float normal[3] = {nx, 0, nz};
        for (int floor = 0; floor < floors; ++floor) {
            for (int bay = 0; bay < bays; ++bay) {
                const uint32_t pick = mixHash(mixHash(wall, static_cast<uint32_t>(floor)), static_cast<uint32_t>(bay));
                if (pick % 100 >= kLitWindows || out.indices.size() / 6 >= kMostWindows) continue;
                const float along = length * (static_cast<float>(bay) + 0.5f) / static_cast<float>(bays);
                const float bottom = base + kWindowSill + static_cast<float>(floor) * kFloorHeight, top = bottom + kWindowHeight;
                const float cx = ax + ux * along + nx * kWindowLift + ox, cz = az + uz * along + nz * kWindowLift + oz;
                const float half = kWindowWidth * 0.5f, warmth = static_cast<float>(pick >> 8 & 63) / 63.0f;
                const float color[3] = {1.0f, 0.62f + 0.2f * warmth, 0.26f + 0.3f * warmth};
                const float u = (static_cast<float>(pick >> 16 & (kWindowStages - 1)) + 0.5f) / kWindowStages;
                const uint32_t v = static_cast<uint32_t>(out.vertices());
                out.vertex(cx - ux * half, bottom, cz - uz * half, normal, color, u, 0.5f);
                out.vertex(cx + ux * half, bottom, cz + uz * half, normal, color, u, 0.5f);
                out.vertex(cx + ux * half, top, cz + uz * half, normal, color, u, 0.5f);
                out.vertex(cx - ux * half, top, cz - uz * half, normal, color, u, 0.5f);
                out.indices.insert(out.indices.end(), {v, v + 1, v + 2, v, v + 2, v + 3});
            }
        }
    }
    return out;
}

// How lit a stage of windows is at a night level: the first stage comes on early, the last well into the dusk.
inline float windowStageLevel(int stage, float level) {
    return std::clamp((level - 0.5f * static_cast<float>(stage) / kWindowStages) * 2.5f, 0.0f, 1.0f);
}

// How strongly a pool shows at a distance from its lamp, as a share of its radius.
inline float poolShape(float r) {
    const float inside = std::max(0.0f, 1 - r * r);
    return inside * inside;
}

// The few lamps given a real light. A lamp only takes a free slot and is given up by fading out first, so walking
// along a street hands the lights on without one ever snapping on or off.
class LightSlots {
public:
    struct Slot {
        int lamp = -1;
        float level = 0;
        bool wanted = false;
    };
    static constexpr float kFadeSeconds = 1.2f;

    // `nearest` is the lamps that should have lights, best first; `budget` is how many lights there may be.
    void choose(const std::vector<int>& nearest, size_t budget) {
        slots_.resize(budget);
        const size_t count = std::min(nearest.size(), budget);
        for (Slot& s : slots_) s.wanted = s.lamp >= 0 && std::find(nearest.begin(), nearest.begin() + static_cast<std::ptrdiff_t>(count), s.lamp) != nearest.begin() + static_cast<std::ptrdiff_t>(count);
        for (size_t i = 0; i < count; ++i) {
            const int lamp = nearest[i];
            if (std::any_of(slots_.begin(), slots_.end(), [&](const Slot& s) { return s.lamp == lamp; })) continue;
            auto open = std::find_if(slots_.begin(), slots_.end(), [](const Slot& s) { return s.lamp < 0; });
            if (open == slots_.end()) break;  // until one has faded out
            *open = {lamp, 0, true};
        }
    }
    void tick(float dt) {
        for (Slot& s : slots_) {
            if (s.lamp < 0) continue;
            s.level = std::clamp(s.level + (s.wanted ? dt : -dt) / kFadeSeconds, 0.0f, 1.0f);
            if (!s.wanted && s.level <= 0) s = Slot{};
        }
    }
    void settle() {  // the first look, with nothing on screen yet to pop
        for (Slot& s : slots_) {
            if (s.wanted) s.level = 1;
            else s = Slot{};
        }
    }
    void clear() { slots_.clear(); }
    const std::vector<Slot>& slots() const { return slots_; }

private:
    std::vector<Slot> slots_;
};

}  // namespace streetkit
