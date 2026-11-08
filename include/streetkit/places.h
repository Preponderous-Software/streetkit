// Named places on the map: which one the player has come to, and the words to show. Pure: tested without a GPU.
#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace streetkit {

struct Place {
    double east = 0, north = 0;  // UTM
    std::string name, sentence;
};

constexpr double kPlaceReach = 30;  // metres from a place that count as having come to it
constexpr double kPlaceLeave = 50;  // and how far to go before it can greet you again, so a step back and forth stays quiet

inline double placeDistance(const Place& p, double east, double north) { return std::hypot(p.east - east, p.north - north); }

// Places within `radius` metres, nearest first, at most `most` of them.
inline std::vector<const Place*> placesNear(const std::vector<Place>& places, double east, double north, double radius, size_t most) {
    std::vector<std::pair<double, const Place*>> nearby;
    for (const Place& p : places) {
        const double d = placeDistance(p, east, north);
        if (d <= radius) nearby.push_back({d, &p});
    }
    std::sort(nearby.begin(), nearby.end(), [](const auto& l, const auto& r) { return l.first < r.first; });
    std::vector<const Place*> out;
    for (const auto& n : nearby) {
        if (out.size() == most) break;
        out.push_back(n.second);
    }
    return out;
}

// Follows the player from place to place, so a place greets them once, when they arrive.
class PlaceWatch {
public:
    // The place just arrived at, or nullptr. Wherever the player begins is not an arrival. `places` must not change.
    const Place* update(const std::vector<Place>& places, double east, double north) {
        const bool first = !started_;
        started_ = true;
        if (here_ && placeDistance(*here_, east, north) <= kPlaceReach) return nullptr;
        const auto nearest = placesNear(places, east, north, kPlaceReach, 1);
        if (!nearest.empty()) {
            here_ = nearest[0];
            if (first) return nullptr;
            ++arrivals_;
            return here_;
        }
        if (here_ && placeDistance(*here_, east, north) > kPlaceLeave) here_ = nullptr;
        return nullptr;
    }

    const Place* here() const { return here_; }
    int arrivals() const { return arrivals_; }

private:
    const Place* here_ = nullptr;
    bool started_ = false;
    int arrivals_ = 0;
};

// Breaks text at spaces so no line runs past `columns` characters.
inline std::string wrapped(const std::string& text, size_t columns) {
    std::string out;
    size_t line = 0, from = 0;
    while (from <= text.size()) {
        const size_t space = std::min(text.find(' ', from), text.size());
        const size_t word = space - from;
        if (line > 0 && line + 1 + word > columns) {
            out += '\n';
            line = 0;
        } else if (line > 0) {
            out += ' ';
            ++line;
        }
        out.append(text, from, word);
        line += word;
        from = space + 1;
    }
    return out;
}

// The longest line's length in characters.
inline size_t widest(const std::string& text) {
    size_t longest = 0, line = 0;
    for (char c : text) {
        line = c == '\n' ? 0 : line + 1;
        longest = std::max(longest, line);
    }
    return longest;
}

}  // namespace streetkit
