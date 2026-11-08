// The labels for the places around the player, for the desktop viewer and the browser alike.
#pragma once

#include <string>
#include <vector>

#include "places.h"
#include "tile_stream.h"
#include "walker.h"

namespace streetkit {

constexpr double kPlacesOnFoot = 120, kPlacesFlying = 320;  // how far away a place's name shows
constexpr size_t kMostPlacesOnFoot = 3, kMostPlacesFlying = 8;
constexpr float kPlaceLabelHeight = 7.0f, kPlacePixels = 16, kPlaceNoteSeconds = 8;

class PlaceView {
public:
    struct Label {
        float at[3];
        std::string text;
    };

    // Looks again, as often as the signs do. Returns the place just arrived at, or nullptr. A label goes by distance
    // and not line of sight: unlike a street sign, a place is worth knowing about from the other side of a block.
    const Place* update(const std::vector<Place>& places, const Stream& stream, const Walker& w, bool flying) {
        const double east = stream.east(w.x), north = stream.north(w.z), originEast = stream.east(0), originNorth = stream.north(0);
        const Place* arrived = watch_.update(places, east, north);
        labels_.clear();
        for (const Place* p : placesNear(places, east, north, flying ? kPlacesFlying : kPlacesOnFoot, flying ? kMostPlacesFlying : kMostPlacesOnFoot)) {
            const float x = static_cast<float>(p->east - originEast), z = static_cast<float>(originNorth - p->north);
            if (stream.walkable(x, z)) labels_.push_back({{x, stream.groundAt(x, z) + kPlaceLabelHeight, z}, p->name});
        }
        return arrived;
    }

    const std::string& here() const {
        static const std::string nowhere;
        return watch_.here() ? watch_.here()->name : nowhere;
    }
    int arrivals() const { return watch_.arrivals(); }
    const std::vector<Label>& labels() const { return labels_; }

private:
    PlaceWatch watch_;
    std::vector<Label> labels_;
};

}  // namespace streetkit
