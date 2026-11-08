// Street names: which street a point is on, and which signs stand near it. Pure: tested without a GPU.
#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace streetkit {

struct StreetRoad {
    int street = 0;
    std::vector<double> en;  // UTM east, north pairs
};
struct StreetSign {
    double east = 0, north = 0;
    int a = 0, b = 0;  // the two streets it names
};

struct Streets {
    std::vector<std::string> names;
    std::vector<StreetRoad> roads;
    std::vector<StreetSign> signs;

    void addName(size_t id, std::string name) {
        if (names.size() <= id) names.resize(id + 1);
        names[id] = std::move(name);
    }
    bool named(int id) const { return id >= 0 && static_cast<size_t>(id) < names.size(); }

    // The nearest named road within `reach` metres, or -1 when there is none.
    int streetAt(double east, double north, double reach) const {
        int best = -1;
        double nearest = reach;
        for (const StreetRoad& r : roads) {
            for (size_t i = 0; i + 3 < r.en.size(); i += 2) {
                const double ax = r.en[i], ay = r.en[i + 1], dx = r.en[i + 2] - ax, dy = r.en[i + 3] - ay;
                const double length = dx * dx + dy * dy;
                const double t = length > 0 ? std::clamp(((east - ax) * dx + (north - ay) * dy) / length, 0.0, 1.0) : 0.0;
                const double d = std::hypot(east - ax - dx * t, north - ay - dy * t);
                if (d <= nearest && named(r.street)) {
                    nearest = d;
                    best = r.street;
                }
            }
        }
        return best;
    }

    // Signs within `radius` metres that `seen(sign)` allows, nearest first, at most `most` of them.
    template <typename Seen>
    std::vector<const StreetSign*> signsNear(double east, double north, double radius, size_t most, Seen&& seen) const {
        std::vector<std::pair<double, const StreetSign*>> nearby;
        for (const StreetSign& s : signs) {
            const double d = std::hypot(s.east - east, s.north - north);
            if (d <= radius) nearby.push_back({d, &s});
        }
        std::sort(nearby.begin(), nearby.end(), [](const auto& l, const auto& r) { return l.first < r.first; });
        std::vector<const StreetSign*> out;
        for (const auto& n : nearby) {
            if (out.size() == most) break;
            if (seen(*n.second)) out.push_back(n.second);
        }
        return out;
    }

    // What a sign says: `WALNUT ST / LINDEN ST`.
    std::string signText(const StreetSign& s) const {
        std::string text = (named(s.a) ? names[static_cast<size_t>(s.a)] : "") + " / " + (named(s.b) ? names[static_cast<size_t>(s.b)] : "");
        std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
        return text;
    }
};

}  // namespace streetkit
