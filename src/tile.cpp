#include "streetkit/tile.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>

namespace streetkit {
namespace {

// Reads a tile's bytes in order; running out sets ok to false.
struct Reader {
    const uint8_t* at;
    size_t left;
    bool ok = true;
    bool bytes(void* out, size_t n) {
        if (!ok || n > left) {
            ok = false;
            return false;
        }
        if (n) std::memcpy(out, at, n);
        at += n;
        left -= n;
        return true;
    }
    template <typename T>
    T get() {
        T v{};
        bytes(&v, sizeof v);
        return v;
    }
    template <typename T>
    void array(std::vector<T>& v, size_t n) {
        if (!ok || n > left / sizeof(T)) {
            ok = false;
            return;
        }
        v.resize(n);
        bytes(v.data(), n * sizeof(T));
    }
    void pad(size_t count) {
        char skip[4];
        bytes(skip, (4 - count % 4) % 4);
    }
};

constexpr int kMaxMeshes = 16, kMaxVertices = 1 << 24, kMaxIndices = 1 << 26, kMaxCount = 1000000;

}  // namespace

float Tile::heightAt(float x, float z) const {
    const float fx = std::clamp(x / cell, 0.0f, static_cast<float>(samples - 1));
    const float fz = std::clamp(z / cell, 0.0f, static_cast<float>(samples - 1));
    const int c = std::min(static_cast<int>(fx), samples - 2), r = std::min(static_cast<int>(fz), samples - 2);
    const float u = fx - static_cast<float>(c), v = fz - static_cast<float>(r);
    const float north = height(c, r) + (height(c + 1, r) - height(c, r)) * u;
    const float south = height(c, r + 1) + (height(c + 1, r + 1) - height(c, r + 1)) * u;
    return north + (south - north) * v;
}

float Overview::heightAt(double e, double n) const {
    const float fx = std::clamp(static_cast<float>((e - east) / cell), 0.0f, static_cast<float>(cols - 1));
    const float fz = std::clamp(static_cast<float>((north - n) / cell), 0.0f, static_cast<float>(rows - 1));
    const int c = std::min(static_cast<int>(fx), cols - 2), r = std::min(static_cast<int>(fz), rows - 2);
    const float u = fx - static_cast<float>(c), v = fz - static_cast<float>(r);
    const float* a = &heights[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c)];
    const float* b = a + cols;
    const float top = a[0] + (a[1] - a[0]) * u, bottom = b[0] + (b[1] - b[0]) * u;
    return top + (bottom - top) * v;
}

bool loadOverview(const std::string& path, Overview& o, std::string* error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        if (error) *error = path + ": cannot open";
        return false;
    }
    const std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (parseOverview(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size(), o, error)) return true;
    if (error) *error = path + ": " + *error;
    return false;
}

// overview.bin, as streetkit/tiles.py writes it: "PDROVER1", version, cols, rows, cell, east, north, heights, colours.
bool parseOverview(const uint8_t* data, size_t size, Overview& o, std::string* error) {
    auto fail = [&](const char* what) {
        if (error) *error = what;
        return false;
    };
    Reader r{data, size};
    char magic[8];
    if (!r.bytes(magic, 8) || std::memcmp(magic, "PDROVER1", 8) != 0) return fail("not an overview");
    o = Overview{};
    if (r.get<uint32_t>() != 1) return fail("unsupported version");
    o.cols = r.get<int32_t>();
    o.rows = r.get<int32_t>();
    o.cell = r.get<float>();
    o.east = r.get<double>();
    o.north = r.get<double>();
    if (!r.ok || o.cols < 2 || o.rows < 2 || o.cols > 1025 || o.rows > 1025 || !(o.cell > 0)) return fail("bad header");
    const size_t n = static_cast<size_t>(o.cols) * static_cast<size_t>(o.rows);
    r.array(o.heights, n);
    r.array(o.colors, n * 3);
    if (!r.ok) {
        o = Overview{};
        return fail("truncated");
    }
    return true;
}

bool Manifest::has(TileKey k) const { return std::find(tiles.begin(), tiles.end(), k) != tiles.end(); }

std::string Manifest::path(TileKey k) const {
    return dir + "/" + std::to_string(k.ix) + "_" + std::to_string(k.iy) + ".tile";
}

bool loadManifest(const std::string& dir, Manifest& m, std::string* error) {
    std::ifstream in(dir + "/manifest.txt");
    if (!in) {
        if (error) *error = "cannot open " + dir + "/manifest.txt (run `streetkit build` first)";
        return false;
    }
    std::stringstream text;
    text << in.rdbuf();
    if (parseManifest(text.str(), dir, m)) return true;
    if (error) *error = dir + "/manifest.txt is not a manifest this viewer reads";
    return false;
}

bool parseManifest(const std::string& text, const std::string& dir, Manifest& m) {
    std::istringstream in(text);
    m = Manifest{};
    m.dir = dir;
    std::string line, word;
    bool header = false;
    while (std::getline(in, line)) {
        std::istringstream s(line);
        s >> word;
        // The older names are of the games streetkit came from, so tiles built before it still load.
        if (word == "streetkit-manifest" || word == "old-town-walk-manifest" || word == "ponderosa-manifest") {
            int version = 0;
            s >> version;
            header = version == 1;
        } else if (word == "name") s >> m.name;
        else if (word == "tile_size") s >> m.tileSize;
        else if (word == "cell") s >> m.cell;
        else if (word == "datum") s >> m.datum;
        else if (word == "latitude") s >> m.latitude;
        else if (word == "overview") s >> m.overview;
        else if (word == "start") s >> m.startEast >> m.startNorth;
        else if (word == "tile") {
            TileKey k;
            s >> k.ix >> k.iy;
            if (s) m.tiles.push_back(k);
        } else if (word == "street") {
            size_t id = 0;
            std::string name;
            s >> id;
            std::getline(s >> std::ws, name);
            if (!name.empty()) m.streets.addName(id, name);
        } else if (word == "road") {
            StreetRoad road;
            s >> road.street;
            double value = 0;
            while (s >> value) road.en.push_back(value);
            if (road.en.size() >= 4) m.streets.roads.push_back(std::move(road));
        } else if (word == "sign") {
            StreetSign sign;
            s >> sign.east >> sign.north >> sign.a >> sign.b;
            if (s) m.streets.signs.push_back(sign);
        } else if (word == "place") {
            Place place;
            std::string text;
            s >> place.east >> place.north;
            std::getline(s >> std::ws, text);
            const size_t bar = text.find('|');
            if (!s.fail() && bar != std::string::npos) {
                place.name = text.substr(0, bar);
                place.sentence = text.substr(bar + 1);
                m.places.push_back(std::move(place));
            }
        }
    }
    return header && m.tileSize > 0 && !m.tiles.empty();
}

bool loadTile(const std::string& path, Tile& t, std::string* error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        if (error) *error = path + ": cannot open";
        return false;
    }
    const std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (parseTile(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size(), t, error)) return true;
    if (error) *error = path + ": " + *error;
    return false;
}

bool parseTile(const uint8_t* data, size_t size, Tile& t, std::string* error) {
    auto fail = [&](const char* what) {
        if (error) *error = what;
        return false;
    };
    Reader r{data, size};
    char magic[8];
    // PDRTILE1 is the same layout under Ponderosa's name, so tiles it built before streetkit still load.
    if (!r.bytes(magic, 8) || (std::memcmp(magic, "OTWTILE1", 8) != 0 && std::memcmp(magic, "PDRTILE1", 8) != 0))
        return fail("not a tile");
    t = Tile{};
    if (r.get<uint32_t>() != 1) return fail("unsupported version");
    t.key.ix = r.get<int32_t>();
    t.key.iy = r.get<int32_t>();
    t.size = r.get<float>();
    t.cell = r.get<float>();
    t.samples = r.get<int32_t>();
    const auto meshCount = r.get<int32_t>();
    const auto footprintCount = r.get<int32_t>();
    const auto treeCount = r.get<int32_t>();
    if (!r.ok || t.samples < 2 || t.samples > 2049 || meshCount < 0 || meshCount > kMaxMeshes || footprintCount < 0 ||
        footprintCount > kMaxCount || treeCount < 0 || treeCount > kMaxCount)
        return fail("bad header");
    const size_t ring = static_cast<size_t>(t.samples) + 2, n = static_cast<size_t>(t.samples);
    r.array(t.heights, ring * ring);
    r.array(t.colors, n * n * 3);
    r.pad(n * n * 3);
    t.meshes.resize(static_cast<size_t>(meshCount));
    for (TileMesh& m : t.meshes) {
        m.kind = r.get<uint32_t>();
        const auto vc = r.get<int32_t>();
        const auto ic = r.get<int32_t>();
        if (!r.ok || vc <= 0 || ic <= 0 || vc > kMaxVertices || ic > kMaxIndices) return fail("bad mesh header");
        const size_t verts = static_cast<size_t>(vc), count = static_cast<size_t>(ic);
        r.array(m.positions, verts * 3);
        r.array(m.normals, verts * 4);
        r.array(m.colors, verts * 4);
        if (vc < 65536) {
            std::vector<uint16_t> small;
            r.array(small, count);
            r.pad(count * 2);
            m.indices.assign(small.begin(), small.end());
        } else {
            r.array(m.indices, count);
        }
        if (!r.ok) return fail("truncated mesh");
        for (uint32_t i : m.indices) {
            if (i >= static_cast<uint32_t>(vc)) return fail("mesh index out of range");
        }
    }
    t.footprints.resize(static_cast<size_t>(footprintCount));
    for (Footprint& f : t.footprints) {
        const auto points = r.get<int32_t>();
        if (!r.ok || points < 3 || points > (1 << 20)) return fail("bad footprint");
        r.array(f.xz, static_cast<size_t>(points) * 2);
    }
    t.trees.resize(static_cast<size_t>(treeCount));
    for (Tree& tree : t.trees) tree = r.get<Tree>();
    return r.ok || fail("truncated");
}

}  // namespace streetkit
