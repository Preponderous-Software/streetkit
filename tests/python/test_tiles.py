
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from streetkit.geo import Area, tile_north_west, tile_of
from streetkit.mesh import Mesh, clean_ring, points_in_ring, ribbon_quads, signed_area
from streetkit.osm import OsmData, _join
from streetkit.tiles import HEADER, MAGIC, build_all, read_manifest, parse_length, tile_name


class FlatDem:
    def height(self, east, north):
        return np.asarray(east, np.float64) * 0.0 + 1500.0 + (np.asarray(east, np.float64) - 1000.0) * 0.01


AREA = Area("test", 0.0, 0.0, 100.0, 2, 10.0)


def test_tile_names_do_not_depend_on_the_area():
    assert tile_of(1000.0, 2000.0, 500) == (2, 4)
    assert tile_of(999.9, 1999.9, 500) == (1, 3)
    assert tile_north_west(2, 4, 500) == (1000.0, 2500.0)


def test_area_is_a_square_of_tiles(monkeypatch):
    monkeypatch.setattr(Area, "centre_utm", lambda self: (1050.0, 2050.0))
    assert AREA.tiles() == [(9, 19), (10, 19), (9, 20), (10, 20)]
    assert AREA.bounds() == (900.0, 1900.0, 1100.0, 2100.0)


def test_ring_cleaning_and_area():
    ring = clean_ring([(0, 0), (0, 4), (4, 4), (4, 0), (0, 0)])
    assert len(ring) == 4 and signed_area(ring) == 16


def test_points_in_ring():
    ring = clean_ring([(0, 0), (4, 0), (4, 4), (0, 4)])
    assert points_in_ring(np.array([2.0, 5.0]), np.array([2.0, 2.0]), ring).tolist() == [True, False]


def test_ribbon_is_as_wide_as_asked():
    quads = list(ribbon_quads(np.array([(0.0, 0.0), (10.0, 0.0)]), 3.0, step=2.0))
    assert len(quads) == 5 * 2
    ys = [p[1] for q in quads for p in q]
    assert min(ys) == pytest.approx(-1.5) and max(ys) == pytest.approx(1.5)


def test_ways_join_into_rings():
    assert _join([["a", "b"], ["b", "c"], ["c", "a"]])[0][0] == _join([["a", "b"], ["b", "c"], ["c", "a"]])[0][-1]
    assert _join([["a", "b"], ["b", "c"]]) == []


def test_parse_length():
    assert parse_length("12") == 12 and parse_length("30'") == pytest.approx(9.144) and parse_length("many") is None


def read_tile(path):
    data = path.read_bytes()
    h = HEADER.unpack_from(data)
    assert h[0] == MAGIC and h[1] == 1
    samples = h[6]
    heights = np.frombuffer(data, np.float32, (samples + 2) ** 2, HEADER.size).reshape(samples + 2, samples + 2)
    return h, heights


@pytest.fixture
def built(tmp_path, monkeypatch):
    from streetkit.osm import Feature
    monkeypatch.setattr(Area, "centre_utm", lambda self: (1000.0, 2000.0))
    osm = OsmData()
    osm.areas.append(Feature("w1", {"building": "house"}, rings=[np.array([(950.0, 1950.0), (960.0, 1950.0), (960.0, 1956.0), (950.0, 1956.0), (950.0, 1950.0)])]))
    osm.lines.append(Feature("w2", {"highway": "residential"}, line=np.array([(910.0, 2010.0), (1090.0, 2010.0)])))
    osm.points.append(Feature("n3", {"natural": "tree"}, point=(1040.0, 1940.0)))
    summary = build_all(AREA, osm, FlatDem(), tmp_path)
    return tmp_path, summary


def test_every_tile_is_in_the_manifest_with_its_size(built):
    out, summary = built
    m = read_manifest(out / "manifest.txt")
    assert summary["tiles"] == 4 and len(m["tiles"]) == 4 and m["tile_size"] == "100"
    for (ix, iy), (size, _) in m["tiles"].items():
        assert (out / tile_name(ix, iy)).stat().st_size == size


def test_neighbouring_tiles_share_their_edge_heights(built):
    out, _ = built
    _, west = read_tile(out / tile_name(9, 19))
    _, east = read_tile(out / tile_name(10, 19))
    assert np.array_equal(west[1:-1, -2], east[1:-1, 1])
    assert np.array_equal(west[1:-1, -1], east[1:-1, 2])     # the border is the neighbour's first inner column


def test_a_road_crossing_a_tile_edge_lands_in_both(built):
    out, summary = built
    assert summary["buildings"] == 1 and summary["trees"] == 1
    meshes = [read_tile(out / tile_name(ix, 20))[0][7] for ix in (9, 10)]
    assert meshes == [1, 1]


FULL_DOWNLOAD = """<?xml version="1.0"?>
<osm version="0.6" generator="an editor" copyright="c">
 <bounds minlat="1" minlon="1" maxlat="2" maxlon="2"/>
 <node id="1" lat="1.0" lon="1.0" user="someone" uid="7" changeset="9" timestamp="2020-01-01T00:00:00Z" version="3" visible="true"/>
 <node id="2" lat="1.0" lon="1.1" user="someone" uid="7"/>
 <node id="3" lat="1.1" lon="1.1" user="someone" uid="7"/>
 <node id="4" lat="1.1" lon="1.0" user="someone" uid="7">
  <tag k="addr:street" v="Example Street"/><tag k="name" v="A Cafe"/>
 </node>
 <node id="5" lat="1.2" lon="1.2" user="someone"><tag k="natural" v="tree"/><tag k="name" v="Old Oak"/></node>
 <node id="6" lat="1.3" lon="1.3" user="someone"><tag k="amenity" v="cafe"/><tag k="phone" v="+1 555 0100"/></node>
 <way id="10" user="someone" uid="7" changeset="9" timestamp="2020-01-01T00:00:00Z" version="2">
  <nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="1"/>
  <tag k="building" v="yes"/><tag k="height" v="9"/><tag k="name" v="A Shop"/><tag k="addr:housenumber" v="12"/>
  <tag k="website" v="https://example.com"/><tag k="email" v="a@example.com"/><tag k="contact:phone" v="1"/>
  <tag k="operator" v="Acme"/><tag k="brand" v="Acme"/><tag k="ref" v="7"/>
 </way>
 <way id="11" user="someone"><nd ref="1"/><nd ref="2"/><tag k="name" v="Nothing but a name"/></way>
 <relation id="20" user="someone" uid="7">
  <member type="way" ref="10" role="outer"/><member type="node" ref="6" role="label"/>
  <tag k="type" v="multipolygon"/><tag k="landuse" v="grass"/><tag k="name" v="A Park"/>
 </relation>
 <relation id="21" user="someone"><member type="way" ref="10" role="x"/><tag k="type" v="route"/></relation>
</osm>
"""


def test_sanitize_keeps_geometry_and_drops_everything_personal(tmp_path):
    from streetkit.osm import parse
    from streetkit.sanitize import KEEP_TAGS, sanitize

    source = tmp_path / "osm-0-0.osm"
    source.write_text(FULL_DOWNLOAD)
    out = tmp_path / "osm-sample.osm"
    assert sanitize([source], out) == {"nodes": 5, "ways": 1, "relations": 1}
    text = out.read_text()
    for gone in ("user=", "uid=", "changeset=", "timestamp=", "version=\"2\"", "version=\"3\"", "visible=", "bounds",
                 "someone", "addr:", "Example Street", "A Cafe", "A Shop", "A Park", "Old Oak", "phone", "555", "website",
                 "example.com", "email", "contact:", "operator", "brand", "Acme", "k=\"name\"", "k=\"ref\"", "an editor"):
        assert gone not in text, gone
    root = ET.fromstring(text)
    assert {t.get("k") for t in root.iter("tag")} <= KEEP_TAGS
    data = parse([out])
    buildings = [a for a in data.areas if a.tags.get("building") == "yes"]
    assert len(buildings) == 1 and buildings[0].tags["height"] == "9"
    assert any(p.tags.get("natural") == "tree" for p in data.points)


