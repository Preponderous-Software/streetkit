// A person on the ground. Everything solid is kept per tile, so a tile takes its walls with it when it unloads.
#pragma once

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

#include "tile.h"

namespace streetkit {

constexpr float kEyeHeight = 1.65f, kBodyRadius = 0.3f, kWalkSpeed = 1.4f, kRunSpeed = 5.0f, kTrunkRadius = 0.35f;
constexpr float kFlySpeed = 20.0f, kFlyFastSpeed = 80.0f;
constexpr float kJumpSpeed = 4.7f, kGravity = 9.8f;  // leaves the ground at this speed, for a hop of about 1.1 m
constexpr float kFlyClearance = 2.0f, kFlyCeiling = 400.0f;  // metres above the ground

struct SolidOutline {
    std::vector<float> xz;  // world x, z
    float minX, minZ, maxX, maxZ;
};
struct SolidTrunk {
    float x, z;
};

class Solids {
public:
    void add(const Tile& t, float offsetX, float offsetZ) {
        Set& s = tiles_[t.key];
        for (const Footprint& f : t.footprints) {
            SolidOutline o{f.xz, 1e30f, 1e30f, -1e30f, -1e30f};
            for (size_t i = 0; i + 1 < o.xz.size(); i += 2) {
                o.xz[i] += offsetX;
                o.xz[i + 1] += offsetZ;
                o.minX = std::min(o.minX, o.xz[i]);
                o.maxX = std::max(o.maxX, o.xz[i]);
                o.minZ = std::min(o.minZ, o.xz[i + 1]);
                o.maxZ = std::max(o.maxZ, o.xz[i + 1]);
            }
            s.outlines.push_back(std::move(o));
        }
        for (const Tree& tree : t.trees) s.trunks.push_back({tree.x + offsetX, tree.z + offsetZ});
    }
    void remove(TileKey k) { tiles_.erase(k); }

    bool inside(float x, float z) const {
        for (const auto& t : tiles_) {
            for (const SolidOutline& o : t.second.outlines) {
                if (x < o.minX || x > o.maxX || z < o.minZ || z > o.maxZ) continue;
                bool in = false;
                const size_t n = o.xz.size() / 2;
                for (size_t i = 0, j = n - 1; i < n; j = i++) {
                    const float xi = o.xz[i * 2], zi = o.xz[i * 2 + 1], xj = o.xz[j * 2], zj = o.xz[j * 2 + 1];
                    if ((zi > z) != (zj > z) && x < (xj - xi) * (z - zi) / (zj - zi) + xi) in = !in;
                }
                if (in) return true;
            }
        }
        return false;
    }

    void pushOut(float& x, float& z) const {
        for (const auto& t : tiles_) {
            for (const SolidOutline& o : t.second.outlines) {
                if (x < o.minX - kBodyRadius || x > o.maxX + kBodyRadius || z < o.minZ - kBodyRadius || z > o.maxZ + kBodyRadius)
                    continue;
                const size_t n = o.xz.size() / 2;
                for (size_t i = 0, j = n - 1; i < n; j = i++) {
                    const float ax = o.xz[j * 2], az = o.xz[j * 2 + 1], ex = o.xz[i * 2] - ax, ez = o.xz[i * 2 + 1] - az;
                    const float len2 = ex * ex + ez * ez;
                    const float u = len2 > 0 ? std::clamp(((x - ax) * ex + (z - az) * ez) / len2, 0.0f, 1.0f) : 0.0f;
                    clear(x, z, ax + ex * u, az + ez * u, kBodyRadius);
                }
            }
            for (const SolidTrunk& tree : t.second.trunks) clear(x, z, tree.x, tree.z, kBodyRadius + kTrunkRadius);
        }
    }

private:
    struct Set {
        std::vector<SolidOutline> outlines;
        std::vector<SolidTrunk> trunks;
    };
    static void clear(float& x, float& z, float px, float pz, float distance) {
        const float dx = x - px, dz = z - pz, d = std::hypot(dx, dz);
        if (d >= distance || d < 1e-5f) return;
        x = px + dx / d * distance;
        z = pz + dz / d * distance;
    }
    std::map<TileKey, Set> tiles_;
};

constexpr float kBikeSpeed = 12.0f, kBikeSprint = 2.0f, kBikeReverse = 1.5f;  // metres a second, and Shift's multiple
constexpr float kBikeAccel = 6.0f, kBikeBrake = 12.0f, kBikeCoast = 0.8f;  // metres a second, each second
constexpr float kBikeEye = 1.75f, kBikeTurnRadius = 4.0f, kBikeMaxTurn = 70.0f, kBikeMaxLean = 2.0f;  // m, m, deg/s, deg

// Yaw 0 faces north (-z), 90 faces east (+x).
struct Walker {
    float x = 0, z = 0, yawDeg = 0, pitchDeg = 0, eyeY = 0;
    float hop = 0, hopSpeed = 0;  // height of a jump above the ground, kept apart from eyeY, which eases over the terrain

    // A jump starts only from the ground, so holding the key hops rather than climbs.
    void jump() {
        if (hop <= 0) hopSpeed = kJumpSpeed;
    }
    void stepHop(float dt) {
        if (hop <= 0 && hopSpeed <= 0) return;
        hop += hopSpeed * dt;
        hopSpeed -= kGravity * dt;
        if (hop <= 0) hop = hopSpeed = 0;
    }

    void look(float dyaw, float dpitch) {
        yawDeg = std::fmod(yawDeg + dyaw + 360.0f, 360.0f);
        pitchDeg = std::clamp(pitchDeg + dpitch, -85.0f, 85.0f);
    }
    void forward(float out[3]) const {
        const float yaw = yawDeg * 3.14159265f / 180.0f, pitch = pitchDeg * 3.14159265f / 180.0f;
        out[0] = std::sin(yaw) * std::cos(pitch);
        out[1] = std::sin(pitch);
        out[2] = -std::cos(yaw) * std::cos(pitch);
    }
    // Moves by ahead/right (-1..1), sliding along solids and stopping where `walkable(x, z)` says no.
    template <typename Walkable>
    void step(const Solids& solids, Walkable&& walkable, float ahead, float right, bool run, float dt, float speed = kWalkSpeed) {
        const float length = std::hypot(ahead, right);
        if (length <= 0) return;
        const float yaw = yawDeg * 3.14159265f / 180.0f, scale = (run ? speed * kRunSpeed / kWalkSpeed : speed) * dt / std::max(1.0f, length);
        const float fx = std::sin(yaw), fz = -std::cos(yaw);
        const float dx = (ahead * fx - right * fz) * scale, dz = (ahead * fz + right * fx) * scale;
        const int steps = std::max(1, static_cast<int>(std::ceil(std::hypot(dx, dz) / (kBodyRadius * 0.5f))));
        for (int i = 0; i < steps; ++i) {
            float nx = x + dx / static_cast<float>(steps), nz = z + dz / static_cast<float>(steps);
            for (int pass = 0; pass < 3; ++pass) solids.pushOut(nx, nz);
            if (walkable(nx, nz)) {
                x = nx;
                z = nz;
            }
        }
    }
    // Moves freely toward the view: ahead and right follow the look, up follows the world. Solids do not
    // stop a flight, but `walkable(x, z)` does, one axis at a time so the edge of the map can be slid along.
    template <typename Walkable, typename Ground>
    void fly(Walkable&& walkable, Ground&& ground, float ahead, float right, float up, bool fast, float dt, float speed = kFlySpeed) {
        const float length = std::sqrt(ahead * ahead + right * right + up * up);
        if (length > 0) {
            float f[3];
            forward(f);
            const float yaw = yawDeg * 3.14159265f / 180.0f, scale = (fast ? speed * kFlyFastSpeed / kFlySpeed : speed) * dt / std::max(1.0f, length);
            const float dx = (ahead * f[0] + right * std::cos(yaw)) * scale, dz = (ahead * f[2] + right * std::sin(yaw)) * scale;
            if (walkable(x + dx, z)) x += dx;
            if (walkable(x, z + dz)) z += dz;
            eyeY += (ahead * f[1] + up) * scale;
        }
        const float floor = ground(x, z);
        eyeY = std::clamp(eyeY, floor + kFlyClearance, floor + kFlyCeiling);
    }
    // Steps sideways out of any building, for landing on a roof.
    template <typename Walkable>
    void clearOfBuildings(const Solids& solids, Walkable&& walkable) {
        for (float r = 1; r <= 80 && solids.inside(x, z); r += 1) {
            for (int i = 0; i < 8; ++i) {
                const float a = static_cast<float>(i) * 0.7854f, nx = x + r * std::cos(a), nz = z + r * std::sin(a);
                if (walkable(nx, nz) && !solids.inside(nx, nz)) {
                    x = nx;
                    z = nz;
                    return;
                }
            }
        }
    }
};

// A bike goes where its wheels point: it turns in arcs, cannot move sideways, and keeps rolling when you stop pedalling.
// The view turns with the bike and otherwise stays where the rider left it: nothing but the rider moves it.
struct Bike {
    bool riding = false;
    float speed = 0, heading = 0, steer = 0, lean = 0;

    void mount(Walker& w) {
        *this = Bike{};
        riding = true;
        heading = w.yawDeg;
        w.hop = w.hopSpeed = 0;
    }
    void dismount() { *this = Bike{}; }

    // pedal is -1 (brake, then a slow reverse) to 1; turn is -1 (left) to 1.
    template <typename Walkable>
    void ride(Walker& w, const Solids& solids, Walkable&& walkable, float pedal, float turn, bool sprint, float dt, float cruise = kBikeSpeed) {
        const float top = sprint ? cruise * kBikeSprint : cruise;
        if (pedal > 0) {
            if (speed < 0) speed = std::min(0.0f, speed + kBikeBrake * dt);
            else if (speed < top) speed = std::min(top, speed + kBikeAccel * (sprint ? kBikeSprint : 1.0f) * pedal * dt);
            else speed = std::max(top, speed - kBikeCoast * 4 * dt);
        } else if (pedal < 0) {
            if (speed > 0) speed = std::max(0.0f, speed + kBikeBrake * pedal * dt);
            else speed = std::max(-kBikeReverse, speed + kBikeAccel * pedal * dt);
        } else {
            speed = speed > 0 ? std::max(0.0f, speed - kBikeCoast * dt) : std::min(0.0f, speed + kBikeCoast * dt);
        }
        steer += (std::clamp(turn, -1.0f, 1.0f) - steer) * std::min(1.0f, dt * 8);
        const float pace = std::abs(speed);
        const float rate = std::min(pace / kBikeTurnRadius * 180.0f / 3.14159265f, kBikeMaxTurn) * (speed < 0 ? -1.0f : 1.0f);
        const float turned = steer * rate * dt;
        heading = std::fmod(heading + turned + 360.0f, 360.0f);
        w.yawDeg = std::fmod(w.yawDeg + turned + 360.0f, 360.0f);
        const float target = steer * std::min(pace / kBikeSpeed, 1.0f) * kBikeMaxLean;
        lean += (target - lean) * std::min(1.0f, dt * 3);

        const float yaw = heading * 3.14159265f / 180.0f, want = speed * dt, dx = std::sin(yaw) * want, dz = -std::cos(yaw) * want;
        const float x0 = w.x, z0 = w.z;
        const int steps = std::max(1, static_cast<int>(std::ceil(std::abs(want) / (kBodyRadius * 0.5f))));
        for (int i = 0; i < steps; ++i) {
            float nx = w.x + dx / static_cast<float>(steps), nz = w.z + dz / static_cast<float>(steps);
            for (int pass = 0; pass < 3; ++pass) solids.pushOut(nx, nz);
            if (walkable(nx, nz)) {
                w.x = nx;
                w.z = nz;
            }
        }
        // Whatever the wheels could not cover was a wall: a glancing touch slows you, and a head-on one stops you.
        if (std::abs(want) > 1e-4f) speed *= std::min(1.0f, std::hypot(w.x - x0, w.z - z0) / std::abs(want));
    }
};

// Which way a move is going across the map, from the keys or stick. A rider goes where the wheels point, whichever
// way the rider looks, and keeps rolling with nothing held; the length of (dx, dz) is zero when nothing is moving.
inline void travelDirection(const Walker& w, const Bike& bike, float ahead, float right, float& dx, float& dz) {
    const float degrees = bike.riding ? bike.heading : w.yawDeg, yaw = degrees * 3.14159265f / 180.0f;
    const float fx = std::sin(yaw), fz = -std::cos(yaw);
    if (bike.riding) {
        const float way = bike.speed > 0 ? 1.0f : bike.speed < 0 ? -1.0f : 0.0f;
        dx = fx * way;
        dz = fz * way;
        return;
    }
    dx = ahead * fx - right * fz;
    dz = ahead * fz + right * fx;
}

// The speed to show: a rider's real one, or what a walker or a flight is going at with Shift or a full stick.
inline float shownSpeed(bool flying, const Bike& bike, float cruise, bool hurrying) {
    if (bike.riding) return std::abs(bike.speed);
    if (!hurrying) return cruise;
    return cruise * (flying ? kFlyFastSpeed / kFlySpeed : kRunSpeed / kWalkSpeed);
}

}  // namespace streetkit
