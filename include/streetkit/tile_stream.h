// The tiles in memory and on the GPU, shared by the desktop viewer and the browser page.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "night.h"
#include "streaming.h"
#include "strata/strata.h"
#include "tile.h"
#include "walker.h"

namespace streetkit {

using Clock = std::chrono::steady_clock;
inline double msSince(Clock::time_point from) { return std::chrono::duration<double, std::milli>(Clock::now() - from).count(); }

// Where a tile comes from: a file on the desktop, a download in a browser. Pending means not yet: the
// stream asks again on a later update.
enum class TileFetch { Ready, Pending, Failed };
using TileSource = std::function<TileFetch(TileKey, Tile&, std::string*)>;

inline bool check(StrataStatus s, const char* what) {
    if (s == STRATA_OK) return true;
    std::fprintf(stderr, "streetkit: %s: %s (%s)\n", what, strata_status_string(s), strata_last_error());
    return false;
}

inline float linearOf(uint8_t srgb) { return std::pow(static_cast<float>(srgb) / 255.0f, 2.2f); }

// The tiles in memory and on the GPU. World x, z are metres from the north-west corner of the start tile.
class Stream {
public:
    Stream(StrataEngine* engine, const Manifest& m, TileKey origin, StreamConfig c, TileSource source)
        : engine_(engine), m_(m), origin_(origin), config_(c), source_(std::move(source)) {
        for (int i = 0; i < 256; ++i) linear_[i] = linearOf(static_cast<uint8_t>(i));
        for (const Lamp& l : placeLamps(m_.streets)) {  // roads run on past the map's edge
            const double tile = m_.tileSize;
            if (m_.has({static_cast<int>(std::floor(l.east / tile)), static_cast<int>(std::floor(l.north / tile))})) lamps_.push_back(l);
        }
    }

    float offsetX(TileKey k) const { return static_cast<float>(k.ix - origin_.ix) * m_.tileSize; }
    float offsetZ(TileKey k) const { return static_cast<float>(origin_.iy - k.iy) * m_.tileSize; }
    double east(float x) const { return origin_.ix * static_cast<double>(m_.tileSize) + x; }
    double north(float z) const { return (origin_.iy + 1) * static_cast<double>(m_.tileSize) - z; }
    TileKey keyAt(float x, float z) const {
        return {static_cast<int>(std::floor(east(x) / m_.tileSize)), static_cast<int>(std::floor(north(z) / m_.tileSize))};
    }

    // Applies the plan for a position, loading at most maxLoads tiles.
    bool update(float x, float z, int maxLoads) {
        std::set<TileKey> loaded;
        for (const auto& t : tiles_) loaded.insert(t.first);
        const StreamPlan plan =
            planStreaming(m_.tiles, loaded, east(x), north(z), m_.tileSize, config_);
        for (const TileKey& k : plan.unload) drop(k);
        int count = 0;
        waiting_ = static_cast<int>(plan.load.size());
        for (const TileKey& k : plan.load) {
            if (count >= maxLoads) break;
            const TileFetch got = load(k);
            if (got == TileFetch::Failed) return false;
            if (got == TileFetch::Ready) {
                ++count;
                --waiting_;
            }
        }
        return true;
    }

    // After the GPU was lost: everything on it is gone, so the tiles are fetched and uploaded again.
    void restart(StrataEngine* engine) {
        engine_ = engine;
        tiles_.clear();
        solids_ = Solids{};
        trees_.clear();
        meshCount_ = 0;
        poolTexture_ = windowTexture_ = 0;
        if (overview_) showOverview(*overview_);   // nothing survives a lost context, the terrain included
    }

    // Draws an area's overview under the tiles as Strata's terrain, and lets ground and flight reach over it.
    // The overview must outlive the stream. Returns false when Strata refuses it.
    bool showOverview(const Overview& o) {
        overview_ = &o;
        if (o.empty()) return true;
        // Strata lays samples along +x then +z, which is how the file runs: from the north-west, rows to the south.
        std::vector<float> colors(o.colors.size());
        for (size_t i = 0; i < colors.size(); ++i) colors[i] = linear_[o.colors[i]];
        StrataTerrainTile tile{};
        tile.origin[0] = static_cast<float>(o.east - origin_.ix * static_cast<double>(m_.tileSize));
        tile.origin[2] = static_cast<float>((origin_.iy + 1) * static_cast<double>(m_.tileSize) - o.north);
        tile.cell_size = o.cell;
        tile.samples_x = o.cols;
        tile.samples_z = o.rows;
        tile.heights = o.heights.data();
        tile.colors = colors.data();
        return check(strata_set_terrain(engine_, &tile), "overview");
    }
    // Flight goes wherever there is land to see, which with an overview is further than the tiles in memory.
    bool flyable(float x, float z) const { return walkable(x, z) || (overview_ && overview_->covers(east(x), north(z))); }
    void setLoadRadius(float radius) {
        config_.loadRadius = radius;
        config_.unloadRadius = radius + 200;
    }

    bool walkable(float x, float z) const { return tiles_.count(keyAt(x, z)) != 0; }
    float groundAt(float x, float z) const {
        const TileKey k = keyAt(x, z);
        const auto t = tiles_.find(k);
        if (t == tiles_.end()) return overview_ && !overview_->empty() ? overview_->heightAt(east(x), north(z)) : 0;
        return t->second.tile.heightAt(x - offsetX(k), z - offsetZ(k));
    }
    const Solids& solids() const { return solids_; }
    const std::vector<StrataRenderItem>& trees() const { return trees_; }
    // The trees within `reach` metres of (x, z): all of them is tens of thousands, and far ones are a few pixels.
    const std::vector<StrataRenderItem>& treesNear(float x, float z, float reach) {
        nearItems(trees_, x, z, kTreesFull, reach, near_);
        return near_;
    }
    const std::vector<Lamp>& lamps() const { return lamps_; }

    // How far the lamps' pools and the windows have come on, 0 by day to 1. They are meshes kept with the tile and
    // the alpha of their textures fades them, so this is a small upload through the dusk and nothing otherwise. By day
    // they are moved out of sight as well: invisible pixels are still paid for.
    bool setNightLevel(float level) {
        const int step = static_cast<int>(std::lround(std::clamp(level, 0.0f, 1.0f) * kNightSteps));
        if (step == nightStep_) return true;
        const bool turned = (step > 0) != (nightStep_ > 0);
        nightStep_ = step;
        if (turned)
            for (const auto& entry : tiles_)
                for (uint32_t id : entry.second.night) moveNight(id);
        return !poolTexture_ || writeNightTextures(false);
    }
    int meshes() const { return meshCount_; }
    int peakMeshes() const { return peakMeshes_; }
    int tileCount() const { return static_cast<int>(tiles_.size()); }
    int waiting() const { return waiting_; }
    float viewRadius() const { return config_.loadRadius; }
    bool overTile(float x, float z) const { const TileKey k = keyAt(x, z); return std::find(m_.tiles.begin(), m_.tiles.end(), k) != m_.tiles.end(); }
    int loads() const { return loads_; }
    int unloads() const { return unloads_; }
    double readMs() const { return readMs_; }
    double uploadMs() const { return uploadMs_; }
    double worstLoadMs() const { return worstLoadMs_; }

private:
    struct Loaded {
        Tile tile;
        std::vector<uint32_t> ids, night;  // every mesh, and those that are only for the dark
    };

    bool add(const StrataMesh& mesh, Loaded& l) {
        uint32_t id = 0;
        if (!check(strata_add_mesh(engine_, &mesh, &id), "mesh")) return false;
        l.ids.push_back(id);
        ++meshCount_;
        peakMeshes_ = std::max(peakMeshes_, meshCount_);
        return true;
    }

    bool addTerrain(Loaded& l, float ox, float oz) {
        const Tile& t = l.tile;
        const int n = t.samples;
        const size_t count = static_cast<size_t>(n) * static_cast<size_t>(n);
        std::vector<float> pos(count * 3), nrm(count * 3), col(count * 3);
        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < n; ++c) {
                const size_t i = (static_cast<size_t>(r) * static_cast<size_t>(n) + static_cast<size_t>(c)) * 3;
                pos[i] = ox + static_cast<float>(c) * t.cell;
                pos[i + 1] = t.height(c, r);
                pos[i + 2] = oz + static_cast<float>(r) * t.cell;
                const float dx = t.height(c - 1, r) - t.height(c + 1, r), dz = t.height(c, r - 1) - t.height(c, r + 1);
                const float up = 2 * t.cell, len = std::sqrt(dx * dx + up * up + dz * dz);
                nrm[i] = dx / len;
                nrm[i + 1] = up / len;
                nrm[i + 2] = dz / len;
                for (int k = 0; k < 3; ++k) col[i + static_cast<size_t>(k)] = linear_[t.colors[i + static_cast<size_t>(k)]];
            }
        }
        std::vector<uint32_t> idx;
        idx.reserve(static_cast<size_t>(n - 1) * static_cast<size_t>(n - 1) * 6);
        for (int r = 0; r + 1 < n; ++r) {
            for (int c = 0; c + 1 < n; ++c) {
                const uint32_t a = static_cast<uint32_t>(r * n + c), b = a + 1, d = a + static_cast<uint32_t>(n), e = d + 1;
                idx.insert(idx.end(), {a, d, b, b, d, e});
            }
        }
        StrataMesh mesh{};
        mesh.positions = pos.data();
        mesh.normals = nrm.data();
        mesh.colors = col.data();
        mesh.indices = idx.data();
        mesh.vertex_count = static_cast<int32_t>(count);
        mesh.index_count = static_cast<int32_t>(idx.size());
        return add(mesh, l);
    }

    static constexpr int kNightSteps = 64, kPoolTexels = 32;
    static constexpr float kPoolStrength = 0.5f;

    // Both textures are alpha over white, and the meshes' colours supply the warmth.
    bool writeNightTextures(bool create) {
        const float level = static_cast<float>(nightStep_) / kNightSteps;
        std::vector<uint8_t> pool(kPoolTexels * kPoolTexels * 4, 255), windows(kWindowStages * 4, 255);
        for (int y = 0; y < kPoolTexels; ++y) {
            for (int x = 0; x < kPoolTexels; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) / kPoolTexels * 2 - 1, v = (static_cast<float>(y) + 0.5f) / kPoolTexels * 2 - 1;
                pool[static_cast<size_t>(y * kPoolTexels + x) * 4 + 3] = static_cast<uint8_t>(255 * poolShape(std::hypot(u, v)) * kPoolStrength * level);
            }
        }
        for (int i = 0; i < kWindowStages; ++i) windows[static_cast<size_t>(i) * 4 + 3] = static_cast<uint8_t>(255 * windowStageLevel(i, level));
        if (create)
            return check(strata_create_texture_rgba8(engine_, kPoolTexels, kPoolTexels, pool.data(), STRATA_TEXTURE_CLAMP_U | STRATA_TEXTURE_CLAMP_V, &poolTexture_), "pool texture") &&
                   check(strata_create_texture_rgba8(engine_, kWindowStages, 1, windows.data(), STRATA_TEXTURE_NEAREST, &windowTexture_), "window texture");
        return check(strata_update_texture_rgba8(engine_, poolTexture_, 0, 0, kPoolTexels, kPoolTexels, pool.data()), "pool texture") &&
               check(strata_update_texture_rgba8(engine_, windowTexture_, 0, 0, kWindowStages, 1, windows.data()), "window texture");
    }

    void moveNight(uint32_t id) {
        const float away[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, -100000, 0, 1};
        check(strata_set_mesh_transform(engine_, id, nightStep_ > 0 ? nullptr : away), "night mesh");
    }

    bool addGlow(const GlowMesh& g, uint32_t texture, Loaded& l) {
        if (g.empty()) return true;
        StrataTexturedMesh mesh{};
        mesh.mesh.positions = g.positions.data();
        mesh.mesh.normals = g.normals.data();
        mesh.mesh.colors = g.colors.data();
        mesh.mesh.indices = g.indices.data();
        mesh.mesh.vertex_count = static_cast<int32_t>(g.vertices());
        mesh.mesh.index_count = static_cast<int32_t>(g.indices.size());
        mesh.mesh.color[0] = mesh.mesh.color[1] = mesh.mesh.color[2] = mesh.mesh.color[3] = 1;
        mesh.uvs = g.uvs.data();
        mesh.texture = texture;
        mesh.flags = STRATA_MESH_UNLIT | STRATA_MESH_TRANSLUCENT;
        uint32_t id = 0;
        if (!check(strata_add_textured_mesh(engine_, &mesh, &id), "night mesh")) return false;
        l.ids.push_back(id);
        l.night.push_back(id);
        if (nightStep_ == 0) moveNight(id);
        ++meshCount_;
        peakMeshes_ = std::max(peakMeshes_, meshCount_);
        return true;
    }

    // A tile's lit windows and lamp pools, while its buildings are still in memory. The four meshes of every tile still
    // to come fit Strata's limit first, so on a map too big to hold in six meshes a tile, the nearest tiles get these.
    bool addNight(Loaded& l, TileKey k, float ox, float oz) {
        constexpr int kTileMeshes = 4, kNightMeshes = 2;
        const int most = std::min(config_.maxTiles, static_cast<int>(m_.tiles.size())), others = std::max(0, most - tileCount() - 1);
        if (meshCount_ + kNightMeshes + others * kTileMeshes > STRATA_MAX_MESHES) return true;
        if (!poolTexture_ && !writeNightTextures(true)) return false;
        for (const TileMesh& m : l.tile.meshes)
            if (m.kind == 0 && !addGlow(litWindows(m, ox, oz), windowTexture_, l)) return false;
        GlowMesh pools;
        const float size = m_.tileSize;
        auto ground = [&](float x, float z) { return l.tile.heightAt(std::clamp(x - ox, 0.0f, size), std::clamp(z - oz, 0.0f, size)); };
        for (const Lamp& lamp : lamps_) {
            const float x = static_cast<float>(lamp.east - east(0)), z = static_cast<float>(north(0) - lamp.north);
            if (keyAt(x, z) == k && !solids_.inside(x, z)) addPool(pools, x, z, ground);
        }
        return addGlow(pools, poolTexture_, l);
    }

    TileFetch load(TileKey k) {
        const auto t0 = Clock::now();
        Loaded l;
        std::string err;
        const TileFetch got = source_(k, l.tile, &err);
        if (got == TileFetch::Failed) std::fprintf(stderr, "streetkit: %s\n", err.c_str());
        if (got != TileFetch::Ready) return got;
        const double read = msSince(t0);
        const auto t1 = Clock::now();
        const float ox = offsetX(k), oz = offsetZ(k);
        if (!addTerrain(l, ox, oz)) return TileFetch::Failed;
        for (const TileMesh& m : l.tile.meshes) {
            const size_t count = m.positions.size() / 3;
            std::vector<float> pos(m.positions), nrm(count * 3), col(count * 3);
            for (size_t i = 0; i < count; ++i) {
                pos[i * 3] += ox;
                pos[i * 3 + 2] += oz;
                for (size_t c = 0; c < 3; ++c) {
                    nrm[i * 3 + c] = static_cast<float>(m.normals[i * 4 + c]) / 127.0f;
                    col[i * 3 + c] = linear_[m.colors[i * 4 + c]];
                }
            }
            StrataMesh mesh{};
            mesh.positions = pos.data();
            mesh.normals = nrm.data();
            mesh.colors = col.data();
            mesh.indices = m.indices.data();
            mesh.vertex_count = static_cast<int32_t>(count);
            mesh.index_count = static_cast<int32_t>(m.indices.size());
            if (!add(mesh, l)) return TileFetch::Failed;
        }
        solids_.add(l.tile, ox, oz);
        if (!addNight(l, k, ox, oz)) return TileFetch::Failed;
        // The GPU has these now; heights, outlines and trees are what stays useful.
        std::vector<TileMesh>().swap(l.tile.meshes);
        std::vector<uint8_t>().swap(l.tile.colors);
        std::vector<Footprint>().swap(l.tile.footprints);
        tiles_[k] = std::move(l);
        rebuildTrees();
        const double upload = msSince(t1);
        readMs_ += read;
        uploadMs_ += upload;
        worstLoadMs_ = std::max(worstLoadMs_, read + upload);
        ++loads_;
        return TileFetch::Ready;
    }

    void drop(TileKey k) {
        const auto t = tiles_.find(k);
        if (t == tiles_.end()) return;
        for (uint32_t id : t->second.ids) {
            check(strata_remove_mesh(engine_, id), "remove mesh");
            --meshCount_;
        }
        solids_.remove(k);
        tiles_.erase(t);
        ++unloads_;
        rebuildTrees();
    }

    void rebuildTrees() {
        trees_.clear();
        for (const auto& entry : tiles_) {
            const float ox = offsetX(entry.first), oz = offsetZ(entry.first);
            for (const Tree& t : entry.second.tile.trees) {
                const float trunk = t.height * 0.45f, crown = t.height * 0.7f;
                trees_.push_back({STRATA_PRIMITIVE_CYLINDER, {t.x + ox, t.y + trunk * 0.5f, t.z + oz},
                                  {kTrunkRadius, trunk * 0.5f, kTrunkRadius}, {0.105f, 0.065f, 0.034f, 1}});
                trees_.push_back({STRATA_PRIMITIVE_SPHERE, {t.x + ox, t.y + t.height - crown * 0.5f, t.z + oz},
                                  {t.crown * 2, crown, t.crown * 2}, {t.r, t.g, t.b, 1}});   // a sphere is 1 across
            }
        }
    }

    StrataEngine* engine_;
    const Manifest& m_;
    TileKey origin_;
    StreamConfig config_;
    TileSource source_;
    float linear_[256];
    std::map<TileKey, Loaded> tiles_;
    Solids solids_;
    std::vector<StrataRenderItem> trees_, near_;
    std::vector<Lamp> lamps_;
    uint32_t poolTexture_ = 0, windowTexture_ = 0;
    const Overview* overview_ = nullptr;
    int nightStep_ = 0;
    int waiting_ = 0, meshCount_ = 0, peakMeshes_ = 0, loads_ = 0, unloads_ = 0;
    double readMs_ = 0, uploadMs_ = 0, worstLoadMs_ = 0;
};


}  // namespace streetkit
