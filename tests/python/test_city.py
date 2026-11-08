"""Cities: tiles picked by a boundary, elevation in pieces, the overview, sampling, extracts and resumed builds.
Brought from Ponderosa, which built the first city."""

import numpy as np
import pytest

from streetkit.extract import read_extract
from streetkit.geo import Area, select_tiles
from streetkit.osm import Feature, OsmData
from streetkit.tiles import GRASS, OVERVIEW_HEADER, OVERVIEW_MAGIC, Dem, build_all, build_overview, overview_cell, read_manifest


def test_a_boundary_picks_tiles_that_are_not_a_rectangle():
    # An L: tile (5, 5) lies in the notch, more than the margin from the boundary.
    ring = np.array([[0.0, 0.0], [600.0, 0.0], [600.0, 200.0], [200.0, 200.0], [200.0, 600.0], [0.0, 600.0]])
    keys = select_tiles([ring], 100.0, 50.0)
    assert (0, 0) in keys and (5, 0) in keys and (0, 5) in keys
    assert (5, 5) not in keys and (3, 3) not in keys
    assert (1, 1) in keys                       # inside, touching no edge of the boundary
    assert (2, 2) in keys                       # outside, within the margin of the corner
    assert (-1, 0) in keys and (-2, 0) not in keys   # the margin reaches one tile out and no further
    assert keys == sorted(keys, key=lambda k: (k[1], k[0]))


def test_a_tile_inside_a_ring_is_kept_without_touching_it():
    ring = np.array([[0.0, 0.0], [1000.0, 0.0], [1000.0, 1000.0], [0.0, 1000.0]])
    assert (5, 5) in select_tiles([ring], 100.0, 0.0)


def test_an_area_within_a_boundary_uses_those_tiles():
    ring = np.array([[0.0, 0.0], [190.0, 0.0], [190.0, 90.0], [0.0, 90.0]])
    city = Area("city", 0.0, 0.0, 100.0, 0, 10.0).within([ring])
    assert city.tiles() == [(0, 0), (1, 0)] and city.bounds() == (0.0, 0.0, 200.0, 100.0)


def write_dem(path, east, north, heights):
    import tifffile
    tifffile.imwrite(path, np.asarray(heights, np.float32), extratags=[
        (33550, "d", 3, (2.0, 2.0, 0.0)), (33922, "d", 6, (0.0, 0.0, 0.0, east, north, 0.0))])


def test_dem_files_join_into_one_mosaic(tmp_path):
    write_dem(tmp_path / "a.tif", 100.0, 200.0, np.full((4, 4), 10.0))
    write_dem(tmp_path / "b.tif", 108.0, 200.0, np.full((4, 4), 20.0))
    write_dem(tmp_path / "c.tif", 100.0, 192.0, np.full((4, 4), 30.0))
    dem = Dem(tmp_path / "b.tif", tmp_path / "a.tif", tmp_path / "c.tif")
    assert float(dem.height(103.0, 197.0)) == 10.0 and float(dem.height(111.0, 197.0)) == 20.0
    assert float(dem.height(103.0, 189.0)) == 30.0
    assert 10.0 <= float(dem.height(111.0, 189.0)) <= 30.0     # the missing corner reads as the median


class RoughDem:
    sx = 2.0

    def height(self, east, north):
        return 1500.0 + 8.0 * np.sin(np.asarray(east, np.float64) / 37.0) * np.cos(np.asarray(north, np.float64) / 23.0)


def test_the_overview_stays_below_the_detailed_ground():
    area = Area("o", 0.0, 0.0, 100.0, 0, 10.0, overview_cell=20.0, keys=((0, 0), (1, 0), (0, 1)))
    dem = RoughDem()
    o = build_overview(area, OsmData(), [], dem, 1500.0)
    rows, cols = o.heights.shape
    assert (rows, cols) == (11, 11) and o.cell == 20.0
    e = np.linspace(1, 199, 300)
    n = np.linspace(1, 199, 300)
    ge, gn = np.meshgrid(e, n)
    fx, fy = ge / o.cell, (o.north - gn) / o.cell
    c, r = np.minimum(fx.astype(int), cols - 2), np.minimum(fy.astype(int), rows - 2)
    u, v = fx - c, fy - r
    h = o.heights
    surface = (h[r, c] * (1 - u) + h[r, c + 1] * u) * (1 - v) + (h[r + 1, c] * (1 - u) + h[r + 1, c + 1] * u) * v
    gap = dem.height(ge, gn) - 1500.0 - surface
    assert gap.min() > 0.5 and gap.max() < 12


def test_an_overview_too_large_for_one_terrain_tile_coarsens():
    area = Area("big", 0.0, 0.0, 500.0, 0, 2.0, overview_cell=20.0, keys=((0, 0), (49, 0)))
    assert overview_cell(area) == 25.0


def test_the_overview_tints_with_roofs_and_roads():
    area = Area("o", 0.0, 0.0, 100.0, 0, 10.0, overview_cell=20.0, keys=((0, 0),))
    road = Feature("w1", {"highway": "primary"}, line=np.array([[10.0, 50.0], [90.0, 50.0]]))
    roof = Feature("w2", {"building": "yes"}, rings=[np.array([[70.0, 10.0], [90.0, 10.0], [90.0, 30.0], [70.0, 30.0]])])
    o = build_overview(area, OsmData(areas=[roof], lines=[road]), [], RoughDem(), 1500.0)
    assert o.colours[0, 0].tolist() == list(GRASS)     # nothing near it
    assert o.colours[2, 2].tolist() != list(GRASS)     # the road
    assert o.colours[4, 4].tolist() != list(GRASS)     # the roof


def test_the_overview_file_and_manifest_line(tmp_path):
    area = Area("test", 0.0, 0.0, 100.0, 2, 10.0, overview_cell=20.0)
    assert build_all(area, OsmData(), RoughDem(), tmp_path)["overview_bytes"] > 0
    manifest = read_manifest(tmp_path / "manifest.txt")
    assert manifest["overview"] == "overview.bin"
    data = (tmp_path / "overview.bin").read_bytes()
    magic, version, cols, rows, cell, east, north = OVERVIEW_HEADER.unpack_from(data)
    assert magic == OVERVIEW_MAGIC and version == 1 and cell == 20.0
    assert len(data) == OVERVIEW_HEADER.size + cols * rows * 4 + ((cols * rows * 3 + 3) // 4) * 4


def test_sampling_writes_only_some_tiles(tmp_path):
    area = Area("test", 0.0, 0.0, 100.0, 4, 10.0)
    summary = build_all(area, OsmData(), RoughDem(), tmp_path, sample=4)
    assert summary["tiles"] == 16 and summary["written"] == 4
    assert len(list(tmp_path.glob("*.tile"))) == 4


def test_an_extract_keeps_geometry_but_not_names_or_addresses(tmp_path):
    osmium = pytest.importorskip("osmium")
    path = tmp_path / "x.osm.pbf"
    with osmium.SimpleWriter(str(path)) as w:
        square = [(1, -105.0, 40.0), (2, -104.999, 40.0), (3, -104.999, 40.001), (4, -105.0, 40.001)]
        for ref, lon, lat in square:
            w.add_node(osmium.osm.mutable.Node(id=ref, location=(lon, lat), tags={"addr:street": "Sample Street"}))
        w.add_node(osmium.osm.mutable.Node(id=5, location=(-105.0005, 40.0005), tags={"natural": "tree", "name": "Old Oak"}))
        w.add_node(osmium.osm.mutable.Node(id=6, location=(-90.0, 30.0), tags={"natural": "tree"}))
        w.add_way(osmium.osm.mutable.Way(id=10, nodes=[1, 2, 3, 4, 1], tags={"building": "yes", "name": "Sample Hall",
                                                                             "addr:housenumber": "12"}))
        w.add_way(osmium.osm.mutable.Way(id=11, nodes=[1, 2, 3], tags={}))
        w.add_way(osmium.osm.mutable.Way(id=12, nodes=[3, 4, 1], tags={}))
        w.add_relation(osmium.osm.mutable.Relation(id=20, members=[("w", 11, "outer"), ("w", 12, "outer")], tags={
            "type": "multipolygon", "landuse": "grass", "name": "Sample Green"}))
    data = read_extract(path, -105.01, 39.99, -104.99, 40.01)
    assert len(data.areas) == 2
    building = next(a for a in data.areas if "building" in a.tags)
    assert building.tags == {"building": "yes"} and len(building.rings[0]) == 5
    green = next(a for a in data.areas if "landuse" in a.tags)
    assert green.tags == {"type": "multipolygon", "landuse": "grass"} and len(green.rings[0]) == 5
    assert len(data.points) == 1 and data.points[0].tags == {"natural": "tree"}


def test_a_resumed_build_keeps_finished_tiles_and_ignores_partial_ones(tmp_path):
    area = Area("test", 0.0, 0.0, 100.0, 2, 10.0)
    build_all(area, OsmData(), RoughDem(), tmp_path)
    kept = sorted(tmp_path.glob("*.tile"))[0]
    gone = sorted(tmp_path.glob("*.tile"))[1]
    kept.write_bytes(kept.read_bytes() + b"marker")
    gone.unlink()
    (tmp_path / (gone.stem + ".part123")).write_bytes(b"half a tile")
    summary = build_all(area, OsmData(), RoughDem(), tmp_path, resume=True)
    assert kept.read_bytes().endswith(b"marker") and gone.exists() and summary["written"] == 4
    assert read_manifest(tmp_path / "manifest.txt")["tiles"][tuple(map(int, gone.stem.split("_")))][0] == gone.stat().st_size
