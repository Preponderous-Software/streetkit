// The street lamps for this moment, for the desktop viewer and the browser alike.
#pragma once

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include <strata/strata.h>

#include "night.h"
#include "tile_stream.h"
#include "walker.h"

namespace streetkit {

constexpr float kLampsOnFoot = 320, kLampsFlying = 800;  // how far away a lamp shows
constexpr float kPolesOnFoot = 160, kPolesFlying = 320;  // and its pole, too thin to see from further
constexpr size_t kMostLampsShown = 360, kMostPoles = 48;  // a post is two items, a lamp shown two points
constexpr float kPoleWidth = 0.09f, kHeadWidth = 0.34f, kHeadDepth = 0.22f;  // metres
constexpr float kLightRange = 34, kLightIntensity = 0.85f;
constexpr float kLightReach = 110;   // real lights go to lamps this close to the player
constexpr float kLightCeiling = 40;  // and above this height the pools do the work alone
constexpr float kLightAhead = 12;    // measured from this far ahead: what is behind the player shows nothing
constexpr float kLightHold = 0.8f;   // a lamp with a light counts as this much nearer, so that two do not trade places
constexpr int kNightLooks = 15;      // frames between looks

class NightView {
public:
    void setOn(bool on) { on_ = on; }
    bool on() const { return on_; }
    void setSun(float elevationDegrees) { elevation_ = elevationDegrees; }

    // Looks again. Going over every lamp is cheap, but it need not be done each frame. `budget` is how many point
    // lights a frame may have.
    void update(Stream& stream, const Walker& w, bool flying, size_t budget) {
        level_ = on_ ? nightLevel(elevation_) : 0;
        stream.setNightLevel(level_);
        items_.clear();
        points_.clear();
        at_.clear();
        const bool lit = level_ > 0;  // the posts stand all day, and everything else waits for dusk
        if (!lit) {
            slots_.clear();
            lights_.clear();
            started_ = false;
        }
        const std::vector<Lamp>& lamps = stream.lamps();
        const double originEast = stream.east(0), originNorth = stream.north(0);
        auto spot = [&](int lamp, float& x, float& z) {
            x = static_cast<float>(lamps[static_cast<size_t>(lamp)].east - originEast);
            z = static_cast<float>(originNorth - lamps[static_cast<size_t>(lamp)].north);
        };
        const float reach = flying ? kLampsFlying : kLampsOnFoot, poleReach = flying ? kPolesFlying : kPolesOnFoot;
        float ahead[3];
        w.forward(ahead);
        const float aheadX = w.x + ahead[0] * kLightAhead, aheadZ = w.z + ahead[2] * kLightAhead;
        const bool lightsWanted = w.eyeY - stream.groundAt(w.x, w.z) < kLightCeiling;
        const auto& slots = slots_.slots();
        shown_.clear();
        closest_.clear();
        for (int i = 0; i < static_cast<int>(lamps.size()); ++i) {
            float x, z;
            spot(i, x, z);
            const float d = std::hypot(x - w.x, z - w.z);
            if (d > reach || !stream.walkable(x, z) || stream.solids().inside(x, z)) continue;
            shown_.push_back({d, i});
            const float toLight = std::hypot(x - aheadX, z - aheadZ);
            if (!lit || !lightsWanted || toLight > kLightReach) continue;
            const bool held = std::any_of(slots.begin(), slots.end(), [&](const LightSlots::Slot& s) { return s.lamp == i; });
            closest_.push_back({held ? toLight * kLightHold : toLight, i});
        }
        std::sort(shown_.begin(), shown_.end());
        std::sort(closest_.begin(), closest_.end());
        if (shown_.size() > kMostLampsShown) shown_.resize(kMostLampsShown);
        if (lit) {
            chosen_.clear();
            for (const auto& c : closest_) chosen_.push_back(c.second);
            slots_.choose(chosen_, budget);
            if (!started_) slots_.settle();
            started_ = true;
        }

        size_t posts = 0;
        for (const auto& s : shown_) {
            float x, z;
            spot(s.second, x, z);
            const float ground = stream.groundAt(x, z), head = ground + kLampHeight;
            const float fade = level_ * smoothBetween(reach, reach * 0.8f, s.first);  // no edge to the lamps shown
            if (s.first < poleReach && posts < kMostPoles) {
                ++posts;
                StrataRenderItem item{};
                item.kind = STRATA_PRIMITIVE_CYLINDER;
                item.position[0] = x;
                item.position[1] = ground + kLampHeight * 0.5f;
                item.position[2] = z;
                item.scale[0] = item.scale[2] = kPoleWidth;
                item.scale[1] = kLampHeight * 0.5f;
                item.color[0] = item.color[1] = 0.05f;
                item.color[2] = 0.055f;
                item.color[3] = 1;
                items_.push_back(item);
                item.kind = STRATA_PRIMITIVE_CUBE;  // the lamp itself, dark by day, so a post does not read as a bare stick
                item.position[1] = head;
                item.scale[0] = item.scale[2] = kHeadWidth;
                item.scale[1] = kHeadDepth;
                items_.push_back(item);
            }
            if (!lit) continue;
            points_.push_back({{x, head, z}, 14, 5.5f, {1.0f, 0.66f, 0.32f, 0.5f * fade}, STRATA_POINT_SOFT});
            points_.push_back({{x, head, z}, 3, 0.5f, {1.0f, 0.9f, 0.72f, fade}, 0});
        }
        if (!lit) return;
        for (const LightSlots::Slot& s : slots) {
            float x = 0, z = 0;
            if (s.lamp >= 0) spot(s.lamp, x, z);
            at_.push_back({x, s.lamp >= 0 ? stream.groundAt(x, z) + kLampHeight : 0, z});
        }
        buildLights();
    }

    // Every frame: eases the real lights in and out.
    void tick(float dt) {
        if (level_ <= 0) return;
        slots_.tick(dt);
        buildLights();
    }

    float level() const { return level_; }
    const std::vector<StrataRenderItem>& items() const { return items_; }
    const std::vector<StrataPoint>& points() const { return points_; }
    const std::vector<StrataPointLight>& lights() const { return lights_; }
    // For the ambient: moonlight enough to tell a street from a block where no lamp is near.
    void lift(float out[3]) const {
        out[0] = 0.075f * level_;
        out[1] = 0.085f * level_;
        out[2] = 0.13f * level_;
    }

private:
    struct At {
        float x, y, z;
    };

    void buildLights() {
        lights_.clear();
        const auto& slots = slots_.slots();
        for (size_t i = 0; i < slots.size() && i < at_.size(); ++i) {
            if (slots[i].lamp < 0 || slots[i].level <= 0) continue;
            StrataPointLight l{};
            l.position[0] = at_[i].x;
            l.position[1] = at_[i].y;
            l.position[2] = at_[i].z;
            l.range = kLightRange;
            l.color[0] = 1.0f;
            l.color[1] = 0.72f;
            l.color[2] = 0.42f;
            l.intensity = kLightIntensity * slots[i].level * level_;
            lights_.push_back(l);
        }
    }

    bool on_ = true, started_ = false;
    float elevation_ = 90, level_ = 0;
    LightSlots slots_;
    std::vector<StrataRenderItem> items_;
    std::vector<StrataPoint> points_;
    std::vector<StrataPointLight> lights_;
    std::vector<At> at_;
    std::vector<std::pair<float, int>> shown_, closest_;
    std::vector<int> chosen_;
};

}  // namespace streetkit
