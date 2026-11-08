// A tile and the manifest, as written by streetkit/tiles.py. Little-endian. No GPU here.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "places.h"
#include "streets.h"

namespace streetkit {

struct TileKey {
    int ix = 0, iy = 0;
    bool operator<(const TileKey& o) const { return iy != o.iy ? iy < o.iy : ix < o.ix; }
    bool operator==(const TileKey& o) const { return ix == o.ix && iy == o.iy; }
};

struct TileMesh {
    uint32_t kind = 0;
    std::vector<float> positions;     // xyz, tile-local
    std::vector<int8_t> normals;      // xyz + pad, scaled to 127
    std::vector<uint8_t> colors;      // sRGB + pad
    std::vector<uint32_t> indices;
};

struct Footprint {
    std::vector<float> xz;  // tile-local, closed implicitly
};

struct Tree {
    float x, y, z, height, crown, r, g, b;
};

struct Tile {
    TileKey key;
    float size = 0, cell = 0;
    int samples = 0;                  // per side, not counting the border
    std::vector<float> heights;       // (samples + 2)^2, row by row from the north
    std::vector<uint8_t> colors;      // samples^2 * 3 sRGB
    std::vector<TileMesh> meshes;
    std::vector<Footprint> footprints;
    std::vector<Tree> trees;

    // Height at a sample; -1 and samples are the border row.
    float height(int col, int row) const {
        return heights[static_cast<size_t>(row + 1) * static_cast<size_t>(samples + 2) + static_cast<size_t>(col + 1)];
    }
    // Bilinear height at tile-local x, z.
    float heightAt(float x, float z) const;
};

struct Manifest {
    std::string name, dir;
    float tileSize = 0, cell = 0, datum = 0;
    double latitude = 0, startEast = 0, startNorth = 0;
    std::vector<TileKey> tiles;
    Streets streets;
    std::vector<Place> places;
    bool has(TileKey k) const;
    std::string path(TileKey k) const;
};

bool loadManifest(const std::string& dir, Manifest& out, std::string* error);
bool loadTile(const std::string& path, Tile& out, std::string* error);
// The same from bytes already in memory: a browser downloads what a desktop reads.
bool parseManifest(const std::string& text, const std::string& dir, Manifest& out);
bool parseTile(const uint8_t* data, size_t size, Tile& out, std::string* error);

}  // namespace streetkit
