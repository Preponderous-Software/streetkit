// Unit tests for streetkit's C++ half, which needs no GPU: the tile and manifest readers, streaming, walking,
// riding and the map's views. The real-tile checks read a fixture built by tests/make_fixture.py.
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "bike_view.h"
#include "exe_dir.h"
#include "places.h"
#include "night.h"
#include "streaming.h"
#include "tile.h"
#include "walker.h"

static int failures = 0;
static std::string tilesDir = "build/fixture";
#define CHECK(c)                                                                \
    do {                                                                        \
        if (!(c)) {                                                             \
            std::fprintf(stderr, "%s:%d: CHECK(%s)\n", __FILE__, __LINE__, #c); \
            ++failures;                                                         \
        }                                                                       \
    } while (0)

using streetkit::TileKey;

static std::vector<TileKey> grid(int across) {
    std::vector<TileKey> out;
    for (int y = 0; y < across; ++y) {
        for (int x = 0; x < across; ++x) out.push_back({x, y});
    }
    return out;
}

static bool has(const std::vector<TileKey>& v, TileKey k) { return std::find(v.begin(), v.end(), k) != v.end(); }

static void testStreaming() {
    CHECK(streetkit::tileDistance({0, 0}, 250, 250, 500) == 0);
    CHECK(std::fabs(streetkit::tileDistance({1, 0}, 250, 250, 500) - 250) < 1e-9);
    CHECK(std::fabs(streetkit::tileDistance({1, 1}, 400, 400, 500) - std::hypot(100.0, 100.0)) < 1e-9);

    streetkit::StreamConfig c;
    c.loadRadius = 300;
    c.unloadRadius = 400;
    c.maxTiles = 60;
    const auto all = grid(10);
    // From nothing: the tile underfoot comes first, then only what is in range.
    auto p = streetkit::planStreaming(all, {}, 2250, 2250, 500, c);
    CHECK(!p.load.empty() && p.load[0] == (TileKey{4, 4}));
    CHECK(p.load.size() == 5 && p.unload.empty());  // the tile and its four sides; corners are 353 m off
    CHECK(!has(p.load, {2, 4}));

    // Loaded tiles are not asked for again.
    std::set<TileKey> loaded(p.load.begin(), p.load.end());
    CHECK(streetkit::planStreaming(all, loaded, 2250, 2250, 500, c).load.empty());

    // Hysteresis: a tile past the load radius but inside the unload radius stays, so an edge walk does not thrash.
    std::set<TileKey> one{{0, 0}};
    CHECK(streetkit::planStreaming(all, one, 850, 250, 500, c).unload.empty());
    CHECK(has(streetkit::planStreaming(all, one, 901, 250, 500, c).unload, {0, 0}));

    // The budget caps what loads, keeps the nearest, and counts tiles being unloaded as free.
    c.maxTiles = 2;
    p = streetkit::planStreaming(all, {}, 2250, 2250, 500, c);
    CHECK(p.load.size() == 2 && p.load[0] == (TileKey{4, 4}));
    std::set<TileKey> full{{9, 9}, {4, 4}};
    p = streetkit::planStreaming(all, full, 2250, 2250, 500, c);
    CHECK(p.unload.size() == 1 && p.load.size() == 1);

    // Off the data: nothing to load, and no crash.
    c.maxTiles = 60;
    CHECK(streetkit::planStreaming(all, {}, -5000, -5000, 500, c).load.empty());
}

static streetkit::Tile squareTile() {
    streetkit::Tile t;
    t.key = {3, 7};
    t.size = 20;
    t.cell = 1;
    t.samples = 21;
    for (int r = -1; r <= 21; ++r) {
        for (int c = -1; c <= 21; ++c) t.heights.push_back(static_cast<float>(c) * 0.1f);
    }
    t.footprints.push_back({{8, 8, 12, 8, 12, 12, 8, 12}});
    t.trees.push_back({4, 0, 4, 9, 3, 0, 1, 0});
    return t;
}

static void testWalking() {
    const streetkit::Tile t = squareTile();
    CHECK(std::fabs(t.heightAt(5, 5) - 0.5f) < 1e-5f);
    CHECK(std::fabs(t.heightAt(5.5f, 2) - 0.55f) < 1e-5f);
    streetkit::Solids solids;
    solids.add(t, 100, 200);
    CHECK(solids.inside(110, 210) && !solids.inside(107, 210));
    auto anywhere = [](float, float) { return true; };

    {
        // Flight: over walls, toward the view, never into the ground, and only as far as the map goes.
        auto ground = [](float x, float) { return x * 0.01f; };
        auto mapped = [](float x, float) { return x < 1000.0f; };
        streetkit::Walker f;
        f.x = 100;
        f.z = 210;
        f.eyeY = 50;
        f.yawDeg = 90;
        f.fly(mapped, ground, 1, 0, 0, false, 1.0f);
        CHECK(std::fabs(f.x - (100 + streetkit::kFlySpeed)) < 1e-3f && std::fabs(f.eyeY - 50) < 1e-3f);
        CHECK(solids.inside(110, 210));  // the flight above went straight through this building
        f.fly(mapped, ground, 1, 0, 0, true, 1.0f);
        CHECK(std::fabs(f.x - (120 + streetkit::kFlyFastSpeed)) < 1e-3f);
        f.fly(mapped, ground, 0, 1, 0, false, 1.0f);  // right of east is south
        CHECK(std::fabs(f.z - (210 + streetkit::kFlySpeed)) < 1e-3f);
        f.pitchDeg = -45;  // diving goes down as well as along
        const float before = f.eyeY;
        f.fly(mapped, ground, 1, 0, 0, false, 0.5f);
        CHECK(f.eyeY < before);
        f.pitchDeg = 0;
        f.fly(mapped, ground, 0, 0, -1, true, 100.0f);
        CHECK(std::fabs(f.eyeY - (ground(f.x, f.z) + streetkit::kFlyClearance)) < 1e-3f);
        f.fly(mapped, ground, 0, 0, 1, true, 100.0f);
        CHECK(std::fabs(f.eyeY - (ground(f.x, f.z) + streetkit::kFlyCeiling)) < 1e-3f);
        f.fly(mapped, ground, 1, 0, 0, true, 100.0f);
        CHECK(f.x < 1000);  // the edge of the map stops a flight
        const float z0 = f.z;
        f.yawDeg = 135;  // and it can be slid along
        f.fly(mapped, ground, 1, 0, 0, false, 1.0f);
        CHECK(f.z > z0 && f.x < 1000);

        {
            // A jump rises about a metre, comes down, and cannot be repeated in the air.
            streetkit::Walker j;
            j.jump();
            float top = 0;
            int frames = 0;
            for (; frames < 300 && (frames == 0 || j.hop > 0); ++frames) {
                if (frames == 10) j.jump();
                j.stepHop(1.0f / 60.0f);
                top = std::max(top, j.hop);
            }
            CHECK(top > 1.0f && top < 1.3f);
            CHECK(j.hop == 0 && j.hopSpeed == 0 && frames > 40 && frames < 70);
            j.jump();
            CHECK(j.hopSpeed == streetkit::kJumpSpeed);
        }
        streetkit::Walker quick;  // A chosen speed, and Shift scaling it.
        quick.x = 100;
        quick.yawDeg = 90;
        quick.fly(mapped, ground, 1, 0, 0, false, 1.0f, 200.0f);
        CHECK(std::fabs(quick.x - 300) < 1e-2f);
        quick.fly(mapped, ground, 1, 0, 0, true, 1.0f, 100.0f);
        CHECK(std::fabs(quick.x - 700) < 1e-2f);
        {
            // The longest view fits Strata's mesh limit anywhere on a full map, with nothing dropped by the budget.
            std::vector<streetkit::TileKey> all;
            for (int y = 0; y < 40; ++y)
                for (int x = 0; x < 40; ++x) all.push_back({x, y});
            streetkit::StreamConfig longest;
            longest.loadRadius = 1500.0f;
            longest.unloadRadius = 1500.0f + 200;
            const auto plan = streetkit::planStreaming(all, {}, 10000, 10000, 500, longest);
            size_t reach = 0;
            for (const auto& k : all) reach += streetkit::tileDistance(k, 10000, 10000, 500) <= 1500.0f;
            CHECK(reach <= static_cast<size_t>(longest.maxTiles) && plan.load.size() == reach);
        }

        streetkit::Walker land;  // Landing on a roof steps off it.
        land.x = 110;
        land.z = 210;
        CHECK(solids.inside(land.x, land.z));
        land.clearOfBuildings(solids, anywhere);
        CHECK(!solids.inside(land.x, land.z));
    }

    streetkit::Walker p;
    p.x = 102;
    p.z = 210;
    p.yawDeg = 90;
    float f[3];
    p.forward(f);
    CHECK(f[0] > 0.99f);
    for (int i = 0; i < 60 * 10; ++i) p.step(solids, anywhere, 1, 0, false, 1.0f / 60.0f);
    CHECK(std::fabs(p.x - (108 - streetkit::kBodyRadius)) < 1e-3f);  // stopped at the wall
    p.step(solids, anywhere, 1, 1, true, 0.3f);                   // a long frame slides along it
    CHECK(p.z > 210.5f && p.x < 108 && !solids.inside(p.x, p.z));

    streetkit::Walker q;
    q.x = 101;
    q.z = 204;
    q.yawDeg = 90;
    for (int i = 0; i < 60 * 5; ++i) q.step(solids, anywhere, 1, 0, false, 1.0f / 60.0f);
    CHECK(std::fabs(q.x - (104 - streetkit::kTrunkRadius - streetkit::kBodyRadius)) < 1e-3f);  // a trunk

    streetkit::Walker e;  // Pace, and the edge of what is loaded.
    e.x = 0;
    e.yawDeg = 90;
    for (int i = 0; i < 60; ++i) e.step(solids, anywhere, 1, 0, false, 1.0f / 60.0f);
    CHECK(std::fabs(e.x - streetkit::kWalkSpeed) < 1e-3f);
    for (int i = 0; i < 60; ++i) e.step(solids, [](float x, float) { return x < 2.0f; }, 1, 0, true, 1.0f / 60.0f);
    CHECK(e.x < 2.0f && e.x > 1.8f);

    // A tile's walls leave with it.
    solids.remove(t.key);
    CHECK(!solids.inside(110, 210));
}

static void testShownSpeed() {
    streetkit::Bike bike;
    CHECK(streetkit::shownSpeed(false, bike, 1.4f, false) == 1.4f);
    CHECK(std::fabs(streetkit::shownSpeed(false, bike, 1.4f, true) - 5.0f) < 1e-4f);
    CHECK(streetkit::shownSpeed(true, bike, 20, true) == 80.0f);
    bike.riding = true;
    bike.speed = -1.5f;
    CHECK(streetkit::shownSpeed(false, bike, 12, true) == 1.5f);
}

static void testTravelDirection() {
    streetkit::Walker w;
    w.yawDeg = 90;  // looking east
    streetkit::Bike bike;
    float dx = 0, dz = 0;
    streetkit::travelDirection(w, bike, 1, 0, dx, dz);
    CHECK(std::fabs(dx - 1) < 1e-4f && std::fabs(dz) < 1e-4f);
    bike.mount(w);
    bike.heading = 0;  // riding north, still looking east
    streetkit::travelDirection(w, bike, 1, 1, dx, dz);
    CHECK(dx == 0 && dz == 0);  // nothing rolling, whatever is held
    bike.speed = 5;
    streetkit::travelDirection(w, bike, 0, 0, dx, dz);
    CHECK(std::fabs(dx) < 1e-4f && std::fabs(dz + 1) < 1e-4f);  // north, coasting
    bike.speed = -1;
    streetkit::travelDirection(w, bike, 0, 0, dx, dz);
    CHECK(std::fabs(dz - 1) < 1e-4f);
}

static void testBike() {
    streetkit::Solids none;
    auto open = [](float, float) { return true; };
    streetkit::Walker w;
    streetkit::Bike b;
    b.mount(w);
    CHECK(b.riding);
    for (int i = 0; i < 60; ++i) b.ride(w, none, open, 1, 0, false, 0.05f);
    CHECK(b.speed > 4 && b.speed <= streetkit::kBikeSpeed + 0.01f);
    CHECK(std::abs(w.x) < 0.01f && w.z < -8);  // north, straight ahead
    const float rolling = b.speed;
    b.ride(w, none, open, 0, 0, false, 0.5f);
    CHECK(b.speed < rolling && b.speed > 0);  // coasts
    for (int i = 0; i < 40; ++i) b.ride(w, none, open, 1, 0, true, 0.05f);
    CHECK(b.speed > streetkit::kBikeSpeed);  // Shift goes faster
    streetkit::Walker turning;
    streetkit::Bike t;
    t.mount(turning);
    t.ride(turning, none, open, 0, 1, false, 1);
    CHECK(t.heading == 0);  // a stopped bike does not turn on the spot
    for (int i = 0; i < 40; ++i) t.ride(turning, none, open, 1, 1, false, 0.05f);
    CHECK(t.heading > 20 && t.heading < 180 && t.lean > 0);  // an arc to the right, leaning into it
    CHECK(turning.x > 0.1f);
    for (int i = 0; i < 60; ++i) b.ride(w, none, open, -1, 0, false, 0.05f);
    CHECK(b.speed < 0 && b.speed >= -streetkit::kBikeReverse);  // brakes, then a slow reverse
    streetkit::Walker looking;
    streetkit::Bike v;
    v.mount(looking);
    looking.look(30, 0);
    for (int i = 0; i < 40; ++i) v.ride(looking, none, open, 1, 0, false, 0.05f);
    CHECK(std::abs(looking.yawDeg - 30) < 0.01f);  // only the rider moves the view
    for (int i = 0; i < 20; ++i) v.ride(looking, none, open, 1, 1, false, 0.05f);
    CHECK(std::abs(std::remainder(looking.yawDeg - v.heading, 360.0f) - 30) < 0.01f);  // and it turns with the bike
    streetkit::Walker atEdge;
    streetkit::Bike e;
    e.mount(atEdge);
    e.speed = 5;
    e.ride(atEdge, none, [](float, float) { return false; }, 1, 0, false, 0.1f);
    CHECK(e.speed == 0);  // a wall stops it
    CHECK(streetkit::handlebars(b, 0, 2, 0).size() == 4);
}

static void testStreets() {
    streetkit::Manifest m;
    CHECK(streetkit::parseManifest("old-town-walk-manifest 1\ntile_size 500\ntile 1 2 3 abc\nstreet 0 Linden St\nstreet 1 E Mountain Ave\n"
                             "road 0 0 0 100 0\nroad 1 50 -40 50 40\nsign 54.0 6.0 0 1\n", "x", m));
    const streetkit::Streets& s = m.streets;
    CHECK(s.names.size() == 2 && s.roads.size() == 2 && s.signs.size() == 1);
    CHECK(s.names[1] == "E Mountain Ave");
    CHECK(s.streetAt(10, 5, 35) == 0);
    CHECK(s.streetAt(52, 30, 35) == 1);
    CHECK(s.streetAt(10, 100, 35) == -1);
    CHECK(s.signText(s.signs[0]) == "LINDEN ST / E MOUNTAIN AVE");
    auto always = [](const streetkit::StreetSign&) { return true; };
    auto never = [](const streetkit::StreetSign&) { return false; };
    CHECK(s.signsNear(50, 0, 20, 6, always).size() == 1);
    CHECK(s.signsNear(500, 0, 20, 6, always).empty());
    CHECK(s.signsNear(50, 0, 20, 6, never).empty());
}

static void testPlaces() {
    streetkit::Manifest m;
    CHECK(streetkit::parseManifest("old-town-walk-manifest 1\ntile_size 500\ntile 1 2 3 abc\n"
                             "place 100.0 0.0 Old Town Square|Once a street, it became a plaza.\n"
                             "place 170.5 0.0 Avery Block|A corner block.\n"
                             "place 900.0 0.0 No sentence\nplace broken\nsomething new the viewer has not met\n", "x", m));
    CHECK(m.places.size() == 2);
    CHECK(m.tiles.size() == 1);
    const streetkit::Place& square = m.places[0];
    CHECK(square.name == "Old Town Square" && square.sentence == "Once a street, it became a plaza.");
    CHECK(m.places[1].east == 170.5);

    const auto close = streetkit::placesNear(m.places, 150, 0, 60, 5);
    CHECK(close.size() == 2 && close[0]->name == "Avery Block");
    CHECK(streetkit::placesNear(m.places, 150, 0, 60, 1).size() == 1);
    CHECK(streetkit::placesNear(m.places, 0, 900, 60, 5).empty());

    streetkit::PlaceWatch beginning;  // starting in a place is not arriving at it, and it stays the place you are at
    CHECK(beginning.update(m.places, 100, 5) == nullptr);
    CHECK(beginning.here() == &m.places[0] && beginning.arrivals() == 0);

    streetkit::PlaceWatch watch;
    CHECK(watch.update(m.places, -500, 0) == nullptr);  // nowhere in particular
    CHECK(watch.here() == nullptr);
    CHECK(watch.update(m.places, 85, 0) == &m.places[0]);  // arrives once,
    CHECK(watch.update(m.places, 85, 0) == nullptr);       // and says nothing more while standing there
    CHECK(watch.update(m.places, 100, 10) == nullptr);
    CHECK(watch.update(m.places, 65, 0) == nullptr);  // just out of reach, and back: still the same visit
    CHECK(watch.here() == &m.places[0]);
    CHECK(watch.update(m.places, 85, 0) == nullptr);
    CHECK(watch.update(m.places, 170, 0) == &m.places[1]);  // on to the next
    CHECK(watch.here() == &m.places[1]);
    CHECK(watch.update(m.places, 400, 0) == nullptr);  // away
    CHECK(watch.here() == nullptr);
    CHECK(watch.update(m.places, 170, 0) == &m.places[1]);  // a second visit counts
    CHECK(watch.arrivals() == 3);

    CHECK(streetkit::wrapped("one two three", 7) == "one two\nthree");
    CHECK(streetkit::wrapped("one two three", 40) == "one two three");
    CHECK(streetkit::wrapped("unbreakable word", 4) == "unbreakable\nword");
    CHECK(streetkit::wrapped("", 8).empty());
    CHECK(streetkit::widest("one two\nthree") == 7);
    CHECK(streetkit::widest("") == 0);
}

static streetkit::Streets straightRoad(double length) {
    streetkit::Streets s;
    s.addName(0, "Linden St");
    s.roads.push_back({0, {0, 0, length, 0}});
    return s;
}

// A wall of a building, as old_town_walk/mesh.py writes one: a quad, then the roof that ends the building.
static void addBuilding(streetkit::TileMesh& m, float x, float z) {
    const uint32_t base = static_cast<uint32_t>(m.positions.size() / 3);
    const float wall[12] = {x, 0, z, x + 30, 0, z, x + 30, 7, z, x, 7, z}, roof[12] = {x, 7, z, x + 30, 7, z, x + 30, 7, z + 9, x, 7, z + 9};
    m.positions.insert(m.positions.end(), wall, wall + 12);
    m.positions.insert(m.positions.end(), roof, roof + 12);
    for (int i = 0; i < 4; ++i) m.normals.insert(m.normals.end(), {0, 0, -127, 0});
    for (int i = 0; i < 4; ++i) m.normals.insert(m.normals.end(), {0, 127, 0, 0});
    for (uint32_t face : {base, base + 4}) m.indices.insert(m.indices.end(), {face, face + 1, face + 2, face, face + 2, face + 3});
}

static void testNight() {
    // The sun: Strata's, which has no seasons, so noon stands 90 degrees less the latitude.
    CHECK(std::fabs(streetkit::sunElevation(0.5f, 40) - 50) < 0.01f);
    CHECK(std::fabs(streetkit::sunElevation(0.0f, 40) + 50) < 0.01f);
    CHECK(std::fabs(streetkit::sunElevation(0.25f, 40)) < 0.01f);
    CHECK(std::fabs(streetkit::sunElevation(0.75f, 40)) < 0.01f);
    CHECK(std::fabs(streetkit::sunElevation(1.25f, 40)) < 0.01f);  // a time past midnight wraps

    // The fade: nothing by day, everything at night, and no jump between.
    CHECK(streetkit::nightLevel(50) == 0 && streetkit::nightLevel(6) == 0);
    CHECK(streetkit::nightLevel(-6) == 1 && streetkit::nightLevel(-50) == 1);
    CHECK(std::fabs(streetkit::nightLevel(0) - 0.5f) < 0.001f);
    float last = 0;
    for (float t = 0.5f; t <= 1.0f; t += 0.0005f) {  // noon to midnight
        const float level = streetkit::nightLevel(streetkit::sunElevation(t, 40.6f));
        CHECK(level >= last && level - last < 0.02f);
        last = level;
    }
    CHECK(last == 1);
    CHECK(streetkit::nightLevel(streetkit::sunElevation(0.3f, 40)) == streetkit::nightLevel(streetkit::sunElevation(0.7f, 40)));  // as it set, it rises

    // Lamps stand beside a road, evenly, on alternate sides.
    const streetkit::Streets road = straightRoad(1000);
    const auto lamps = streetkit::placeLamps(road);
    const size_t expected = static_cast<size_t>(1000 / streetkit::kLampSpacing) + 1;
    CHECK(lamps.size() == expected);
    for (size_t i = 0; i < lamps.size(); ++i) {
        CHECK(std::fabs(std::fabs(lamps[i].north) - streetkit::kLampSide) < 0.01);
        if (i == 0) continue;
        CHECK(std::fabs(lamps[i].east - lamps[i - 1].east - streetkit::kLampSpacing) < 0.01);
        CHECK(lamps[i].north * lamps[i - 1].north < 0);
    }
    // One that would land on a road crossing nearby goes to the other side, and none stands on either.
    streetkit::Streets crossed = road;
    crossed.roads.push_back({0, {0, 7, 1000, 7}});
    for (const streetkit::Lamp& l : streetkit::placeLamps(crossed)) CHECK(crossed.streetAt(l.east, l.north, streetkit::kLampKerb) < 0);
    // The same road twice gets no second row, and nothing without a name gets lamps.
    streetkit::Streets twice = road;
    twice.roads.push_back(road.roads[0]);
    CHECK(streetkit::placeLamps(twice).size() == lamps.size());
    streetkit::Streets unnamed;
    unnamed.roads.push_back({5, {0, 0, 500, 0}});
    CHECK(streetkit::placeLamps(unnamed).empty());
    // However much road there is, the count stops at the bound. Far enough apart that no road is near another.
    streetkit::Streets sprawl;
    sprawl.addName(0, "Long Rd");
    for (int i = 0; i < 50; ++i) sprawl.roads.push_back({0, {0, i * 40.0, 2000, i * 40.0}});
    const auto many = streetkit::placeLamps(sprawl);
    CHECK(many.size() == streetkit::kMostLamps);
    for (size_t i = 0; i + 1 < many.size(); i += 97) CHECK(std::hypot(many[i].east - many[i + 1].east, many[i].north - many[i + 1].north) >= streetkit::kLampGap);

    // A pool is a patch of ground; its strength dies away to its edge.
    streetkit::GlowMesh pool;
    streetkit::addPool(pool, 10, 20, [](float x, float) { return x * 0.1f; });
    CHECK(pool.vertices() == static_cast<size_t>(streetkit::kPoolGrid * streetkit::kPoolGrid) && pool.uvs.size() == pool.vertices() * 2);
    CHECK(std::fabs(pool.positions[1] - ((10 - streetkit::kPoolRadius) * 0.1f + streetkit::kPoolLift)) < 0.001f);
    CHECK(streetkit::poolShape(0) == 1 && streetkit::poolShape(1) == 0 && streetkit::poolShape(1.4f) == 0 && streetkit::poolShape(0.5f) < 1);

    // Windows: the same building lights the same way every time, on the wall itself, and not every building lights.
    streetkit::TileMesh town;
    for (int i = 0; i < 200; ++i) addBuilding(town, static_cast<float>(i % 20) * 40, static_cast<float>(i / 20) * 30);
    const streetkit::GlowMesh windows = streetkit::litWindows(town, 500, 1000), again = streetkit::litWindows(town, 500, 1000);
    CHECK(windows.positions == again.positions && windows.uvs == again.uvs);
    const size_t quads = windows.indices.size() / 6;
    // 200 buildings of 8 bays on 2 floors, a share of buildings by a share of windows.
    CHECK(quads > 200 * 16 * streetkit::kLitBuildings * streetkit::kLitWindows / 10000 / 2 && quads < 200 * 16 * streetkit::kLitBuildings * streetkit::kLitWindows / 10000 * 2);
    size_t lit = 0;
    for (int i = 0; i < 200; ++i) {
        streetkit::TileMesh one;
        addBuilding(one, static_cast<float>(i % 20) * 40, static_cast<float>(i / 20) * 30);
        lit += streetkit::litWindows(one, 0, 0).empty() ? 0 : 1;
    }
    CHECK(lit > 40 && lit < 160);
    for (size_t i = 0; i < windows.vertices(); ++i) {
        CHECK(std::fabs(std::fmod(windows.positions[i * 3 + 2] - 1000 + streetkit::kWindowLift, 30.0f)) < 0.01f);  // just out from the wall
        CHECK(windows.positions[i * 3 + 1] >= streetkit::kWindowSill && windows.positions[i * 3 + 1] <= 7);        // and within it
    }
    streetkit::TileMesh shed;  // too low to live in
    addBuilding(shed, 0, 0);
    for (size_t i = 7; i < 24; i += 3) shed.positions[i] *= 0.3f;
    CHECK(streetkit::litWindows(shed, 0, 0).empty());
    streetkit::TileMesh crowded;
    for (int i = 0; i < 4000; ++i) addBuilding(crowded, static_cast<float>(i), static_cast<float>(i) * 0.5f);
    CHECK(streetkit::litWindows(crowded, 0, 0).indices.size() / 6 <= streetkit::kMostWindows);
    CHECK(streetkit::windowStageLevel(0, 0) == 0 && streetkit::windowStageLevel(0, 1) == 1 && streetkit::windowStageLevel(streetkit::kWindowStages - 1, 1) == 1);
    CHECK(streetkit::windowStageLevel(streetkit::kWindowStages - 1, 0.3f) < streetkit::windowStageLevel(0, 0.3f));

    // Real lights: within budget, and handed on by fading, never by snapping.
    streetkit::LightSlots slots;
    slots.choose({1, 2, 3}, 2);
    CHECK(slots.slots().size() == 2);
    slots.settle();
    CHECK(slots.slots()[0].lamp == 1 && slots.slots()[0].level == 1 && slots.slots()[1].lamp == 2);
    std::vector<int> wanted = {7, 2};
    float seven = 0;
    for (int frame = 0; frame < 300; ++frame) {
        if (frame % 15 == 0) slots.choose(wanted, 2);
        const auto before = slots.slots();
        slots.tick(1.0f / 60.0f);
        for (size_t i = 0; i < 2; ++i) {
            if (slots.slots()[i].lamp == before[i].lamp) CHECK(std::fabs(slots.slots()[i].level - before[i].level) <= 1.0f / 60.0f / streetkit::LightSlots::kFadeSeconds + 0.0001f);
            else CHECK(before[i].level < 0.02f && slots.slots()[i].level < 0.02f);  // a slot only changes hands dark
            if (slots.slots()[i].lamp == 7) seven = slots.slots()[i].level;
        }
    }
    CHECK(seven == 1);
    CHECK(slots.slots()[1].lamp == 2 && slots.slots()[1].level == 1);  // and the one that stayed wanted never flickered
}

static void testTreesNear() {
    struct Item { float position[3]; };
    std::vector<Item> all, out;
    for (int i = 0; i < 300; ++i) all.push_back({{static_cast<float>(i * 3), 0, 0}});   // a row of trees, 3 m apart
    streetkit::nearItems(all, 0, 0, 180, 600, out);
    int inner = 0, outer = 0;   // not near and far: Windows headers define those
    for (const Item& item : out) (item.position[0] <= 180 ? inner : outer)++;
    CHECK(inner == 61);                    // every tree out to 180 m
    CHECK(outer > 30 && outer < 110);          // about one in three of the 140 between 180 and 600 m
    for (const Item& item : out) CHECK(item.position[0] <= 600);
    streetkit::nearItems(all, 0, 0, 180, 600, out);
    CHECK(static_cast<int>(out.size()) == inner + outer);   // the same trees every frame
    CHECK(streetkit::treeReach(0) == 300 && streetkit::treeReach(1000) == 800);
}

static void testFiles() {
    streetkit::Manifest m;
    std::string err;
    CHECK(!streetkit::loadManifest("no-such-dir", m, &err) && !err.empty());
    streetkit::Tile t;
    CHECK(!streetkit::loadTile("no-such.tile", t, &err));

    const std::string dir = tilesDir;
    if (!streetkit::loadManifest(dir, m, &err)) {
        std::puts("no fixture tiles (tests/make_fixture.py); skipping the real-tile checks");
        return;
    }
    CHECK(m.tileSize == 100 && m.tiles.size() == 4 && std::fabs(m.latitude - 40.6) < 1);
    const TileKey k = m.tiles[0];
    CHECK(streetkit::loadTile(m.path(k), t, &err));
    CHECK(t.key == k && t.samples == 11 && t.meshes.size() >= 1);
    CHECK(t.heights.size() == 13u * 13u && t.colors.size() == 11u * 11u * 3);

    // A truncated copy is refused.
    std::FILE* in = std::fopen(m.path(k).c_str(), "rb");
    std::fseek(in, 0, SEEK_END);
    std::string bytes(static_cast<size_t>(std::ftell(in)), '\0');
    std::fseek(in, 0, SEEK_SET);
    CHECK(std::fread(&bytes[0], 1, bytes.size(), in) == bytes.size());
    std::fclose(in);
    std::FILE* out = std::fopen("truncated.tile", "wb");
    std::fwrite(bytes.data(), 1, bytes.size() / 2, out);
    std::fclose(out);
    streetkit::Tile bad;
    CHECK(!streetkit::loadTile("truncated.tile", bad, &err));
    // The browser reads the same bytes from memory: whole, they parse; cut short or foreign, they do not.
    streetkit::Tile fromMemory;
    CHECK(streetkit::parseTile(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size(), fromMemory, &err));
    CHECK(fromMemory.key == k && fromMemory.samples == t.samples && fromMemory.trees.size() == t.trees.size());
    CHECK(!streetkit::parseTile(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size() / 2, fromMemory, &err));
    CHECK(!streetkit::parseTile(reinterpret_cast<const uint8_t*>("not a tile at all"), 17, fromMemory, &err));
    streetkit::Manifest none;
    CHECK(!streetkit::parseManifest("nothing here", "tiles", none));
    CHECK(streetkit::parseManifest("old-town-walk-manifest 1\ntile_size 500\ntile 1 2\n", "tiles", none));
    CHECK(none.tiles.size() == 1 && none.path({1, 2}) == "tiles/1_2.tile");
    std::remove("truncated.tile");

    // Neighbours agree along their shared edge, where a seam would show.
    if (m.has({k.ix + 1, k.iy})) {
        streetkit::Tile east;
        CHECK(streetkit::loadTile(m.path({k.ix + 1, k.iy}), east, &err));
        float worst = 0;
        for (int r = 0; r < t.samples; ++r) worst = std::max(worst, std::fabs(t.height(t.samples - 1, r) - east.height(0, r)));
        CHECK(worst == 0);
    }
}

static void testPacing() {
    CHECK(streetkit::loadDue(0.2f, false) && !streetkit::loadDue(0.01f, false) && streetkit::loadDue(0.01f, true));
    float nearPlane, farPlane;
    streetkit::clipPlanes(1.65f, nearPlane, farPlane);
    CHECK(nearPlane == 0.1f);
    streetkit::clipPlanes(400, nearPlane, farPlane);
    CHECK(nearPlane == 20.0f && farPlane > 30000);
}

static void testOverview() {
    // An overview in memory, as tiles.py writes it: 3 x 2 samples 10 m apart from (1000, 2000), heights 0..5.
    std::string bytes("PDROVER1", 8);
    auto put = [&](const void* v, size_t n) { bytes.append(static_cast<const char*>(v), n); };
    const uint32_t version = 1;
    const int32_t cols = 3, rows = 2;
    const float cell = 10;
    const double east = 1000, north = 2000;
    put(&version, 4); put(&cols, 4); put(&rows, 4); put(&cell, 4); put(&east, 8); put(&north, 8);
    for (int i = 0; i < 6; ++i) { const float h = static_cast<float>(i); put(&h, 4); }
    for (int i = 0; i < 18; ++i) bytes.push_back(static_cast<char>(i));
    bytes.append(2, '\0');
    streetkit::Overview o;
    std::string err;
    CHECK(streetkit::parseOverview(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size(), o, &err));
    CHECK(o.cols == 3 && o.rows == 2 && o.colors.size() == 18 && o.colors[17] == 17);
    CHECK(o.covers(1000, 2000) && o.covers(1020, 1990) && !o.covers(1021, 1995) && !o.covers(1005, 1989));
    CHECK(std::fabs(o.heightAt(1010, 2000) - 1.0f) < 1e-5f);       // second sample of the first row
    CHECK(std::fabs(o.heightAt(1005, 1995) - 2.0f) < 1e-5f);       // the middle of the first square: (0+1+3+4)/4
    CHECK(!streetkit::parseOverview(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size() - 10, o, &err) && o.empty());
    CHECK(!streetkit::parseOverview(reinterpret_cast<const uint8_t*>("PDROVER2xxxxxxxx"), 16, o, &err));
    streetkit::Manifest m;
    CHECK(streetkit::parseManifest("ponderosa-manifest 1\ntile_size 500\noverview overview.bin\ntile 1 2 3 abc\n", "d", m));
    CHECK(m.overview == "overview.bin");
}

static void testFindTiles() {
    namespace fs = std::filesystem;
    const fs::path top = fs::temp_directory_path() / "old_town_walk_find_tiles";
    fs::remove_all(top);
    fs::create_directories(top / "unzipped" / "tiles");
    fs::create_directories(top / "build" / "app" / "Release");
    fs::create_directories(top / "build" / "tiles");
    std::ofstream(top / "unzipped" / "tiles" / "manifest.txt") << "x";
    std::ofstream(top / "build" / "tiles" / "manifest.txt") << "x";
    CHECK(fs::path(streetkit::tilesNear(top / "unzipped" / "old-town-walk.exe")) == top / "unzipped" / "tiles");
    CHECK(fs::path(streetkit::tilesNear(top / "build" / "app" / "old-town-walk")) == top / "build" / "tiles");
    CHECK(fs::path(streetkit::tilesNear(top / "build" / "app" / "Release" / "old-town-walk.exe")) == top / "build" / "tiles");
    fs::create_directories(top / "empty" / "a" / "b" / "c");
    CHECK(streetkit::tilesNear(top / "empty" / "a" / "b" / "c" / "old-town-walk").empty());
    fs::remove_all(top);
}

int main(int argc, char** argv) {
    if (argc > 1) tilesDir = argv[1];
    testFindTiles();
    testOverview();
    testPacing();
    testStreaming();
    testWalking();
    testFiles();
    testStreets();
    testPlaces();
    testNight();
    testBike();
    testTreesNear();
    testShownSpeed();
    testTravelDirection();
    if (failures) std::fprintf(stderr, "%d failure(s)\n", failures);
    else std::puts("test_streetkit: all checks passed");
    return failures ? 1 : 0;
}
