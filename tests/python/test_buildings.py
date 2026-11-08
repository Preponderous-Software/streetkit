import math

import numpy as np
import pytest

from streetkit import buildings as B
from streetkit.context import BuildingContext
from streetkit.lidar import Chunk, Lidar, edge_distance, fill_gaps, fill_holes
from streetkit.mesh import Mesh, clean_ring
from streetkit.osm import Feature, OsmData
from streetkit.roofs import as_paint, sample
from streetkit.trees import mapped_height, treetops


def fake_lidar(height, leafy=None, west=0.0, north=100.0, cell=1.0):
    chunk = Chunk.__new__(Chunk)
    chunk.height = np.asarray(height, np.float32)
    chunk.leafy = np.zeros_like(chunk.height) if leafy is None else np.asarray(leafy, np.float32)
    chunk.cell, chunk.w, chunk.n = cell, west, north
    chunk.e, chunk.s = west + chunk.height.shape[1] * cell, north - chunk.height.shape[0] * cell
    lid = Lidar.__new__(Lidar)
    lid.chunks, lid.cell = [chunk], cell
    lid.height, lid.leafy, lid.w, lid.n = chunk.height, chunk.leafy, west, north
    return lid


def surface(fn, size=100):
    """A lidar grid whose cell centres (east, north) take heights from fn."""
    e, n = np.meshgrid(np.arange(size) + 0.5, 100.0 - (np.arange(size) + 0.5))
    return fn(e, n)


def box(x0, y0, x1, y1):
    return clean_ring([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def windings_are_outward(mesh: Mesh):
    p, n = np.array(mesh.positions), np.array(mesh.normals)
    for t in np.array(mesh.indices).reshape(-1, 3):
        face = np.cross(p[t[1]] - p[t[0]], p[t[2]] - p[t[0]])
        if np.linalg.norm(face) > 1e-9:
            assert face @ n[t[0]] > 0


# ------------------------------------------------------------------ frames and rectangles

def test_turning_a_footprint_square_to_its_walls():
    turned = B.unrotate(np.array([[0, 0], [10, 0], [10, 6], [0, 6]], float), math.radians(30))
    angle = B.main_angle(clean_ring(turned))
    assert math.degrees(angle) == pytest.approx(30, abs=1e-6)
    assert B.squareness(clean_ring(turned), angle) == pytest.approx(1)
    assert np.allclose(B.rotate(B.unrotate(turned, 0.4), 0.4), turned)


def test_an_l_shaped_house_is_two_overlapping_wings():
    ring = clean_ring([(0, 0), (12, 0), (12, 6), (6, 6), (6, 14), (0, 14)])
    rects = B.footprint_rects(ring, B.main_angle(ring))
    assert len(rects) == 2
    areas = sorted((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in rects)
    assert areas == [72, 84]   # 12 x 6 and 6 x 14, overlapping in the corner


def test_inset_ring_moves_walls_in_and_refuses_to_fold():
    ring = box(0, 0, 10, 6)
    assert np.allclose(B.inset_ring(ring, 1)[0], [1, 1])
    assert B.inset_ring(ring, 3.5) is None


def test_colours_parse_names_and_hex():
    assert B.parse_colour("#ff8000") == (255, 128, 0)
    assert B.parse_colour("#f80") == (255, 136, 0)
    assert B.parse_colour("Light Grey") == (190, 190, 190)
    assert B.parse_colour("plaid") is None


def test_lengths_parse_metres_and_feet():
    assert B.parse_length("12") == 12 and B.parse_length("12 m") == 12
    assert B.parse_length("30'") == pytest.approx(9.144)
    assert B.parse_length("tall") is None


# ------------------------------------------------------------------ reading the lidar

def gable(e, n, x0=20, x1=32, y0=40, y1=48, eave=4.0, pitch=0.6):
    inside = (e > x0) & (e < x1) & (n > y0) & (n < y1)
    return np.where(inside, eave + pitch * np.minimum(n - y0, y1 - n), 0.0)


def hip(e, n, x0=20, x1=32, y0=40, y1=48, eave=4.0, pitch=0.6):
    inside = (e > x0) & (e < x1) & (n > y0) & (n < y1)
    edge = np.minimum(np.minimum(n - y0, y1 - n), np.minimum(e - x0, x1 - e))
    return np.where(inside, eave + pitch * edge, 0.0)


@pytest.mark.parametrize("shape,style", [(gable, "gable"), (hip, "hip")])
def test_a_pitched_roof_is_fitted_with_its_style_pitch_and_eave(shape, style):
    lid = fake_lidar(surface(shape))
    plan = B.plan_building(box(20, 40, 32, 48), [], {"building": "house"}, "w1", lid, lambda e, n: np.zeros(len(e)),
                           np.zeros(2), 0.0)
    assert plan.source == "lidar" and plan.style == style
    assert plan.pitch == pytest.approx(0.6, abs=0.12)
    assert plan.eave == pytest.approx(4.0, abs=0.5)


def test_a_flat_roof_is_flat_at_its_height_and_steps_where_it_rises():
    def block(e, n):
        inside = (e > 10) & (e < 50) & (n > 20) & (n < 60)
        upper = (e > 30) & (e < 48) & (n > 25) & (n < 55)
        return np.where(upper, 18.0, np.where(inside, 9.0, 0.0))

    lid = fake_lidar(surface(block))
    plan = B.plan_building(box(10, 20, 50, 60), [], {"building": "commercial"}, "w2", lid,
                           lambda e, n: np.zeros(len(e)), np.zeros(2), 0.0)
    assert plan.style == "flat" and plan.eave == pytest.approx(9, abs=0.5)
    assert len(plan.steps) >= 1 and plan.steps[0][1] == pytest.approx(18, abs=0.5)


def test_a_footprint_the_survey_did_not_see_falls_back_on_its_tags():
    lid = fake_lidar(np.zeros((100, 100)))
    plan = B.plan_building(box(20, 40, 32, 48), [], {"building": "apartments", "building:levels": "4"}, "w3", lid,
                           lambda e, n: np.zeros(len(e)), np.zeros(2), 0.0)
    assert plan.source == "tags" and plan.eave == pytest.approx(4 * B.STOREY + 0.6)


def test_a_tree_over_a_roof_is_not_read_as_roof():
    leafy = surface(lambda e, n: ((e > 20) & (e < 26) & (n > 40) & (n < 48)).astype(float))
    height = surface(lambda e, n: np.where((e > 20) & (e < 26) & (n > 40) & (n < 48), 14.0, gable(e, n)))
    plan = B.plan_building(box(20, 40, 32, 48), [], {"building": "house"}, "w4", fake_lidar(height, leafy),
                           lambda e, n: np.zeros(len(e)), np.zeros(2), 0.0)
    assert plan.eave < 6


def test_a_steeple_becomes_a_tower_on_a_church():
    def church(e, n):
        body = np.where((e > 10) & (e < 40) & (n > 20) & (n < 40), 10.0, 0.0)
        return np.where((e > 12) & (e < 16) & (n > 30) & (n < 34), 30.0, body)

    plan = B.plan_building(box(10, 20, 40, 40), [], {"building": "church"}, "w5", fake_lidar(surface(church)),
                           lambda e, n: np.zeros(len(e)), np.zeros(2), 0.0)
    assert len(plan.towers) == 1 and plan.towers[0][4] == pytest.approx(30, abs=0.5)


def test_fill_holes_spreads_from_known_cells():
    v = np.array([[1.0, 0, 3.0]])
    out = fill_holes(v, np.array([[True, False, True]]))
    assert out[0, 1] == pytest.approx(2.0)


def test_edge_distance():
    ring = box(0, 0, 10, 10)
    assert edge_distance(np.array([5.0, 1.0]), np.array([5.0, 5.0]), ring).tolist() == [5.0, 1.0]


# ------------------------------------------------------------------ drawing

@pytest.mark.parametrize("style", ["flat", "gable", "hip", "skeleton"])
def test_every_roof_style_faces_out_and_reaches_its_height(style):
    ring = clean_ring([(0, 0), (12, 0), (12, 6), (6, 6), (6, 14), (0, 14)])
    plan = B.Plan(ring, [], -0.4, 0.0, 5.0, style, pitch=0.6, angle=0.0)
    if style in ("gable", "hip"):
        plan.rects = B.footprint_rects(ring, 0.0)
    m = Mesh()
    B.draw(m, plan, B.Look((200, 190, 170), (90, 90, 90), (210, 200, 190), "house"), lambda a, b: False, lambda a, b: False)
    windings_are_outward(m)
    top = np.array(m.positions)[:, 1].max()
    if style == "flat":
        assert top == pytest.approx(5.0)
    elif style == "skeleton":
        assert 5.5 < top < 5.0 + 0.6 * 3   # up to the inner ring, which the 6 m arms keep close to the walls
    else:
        assert top == pytest.approx(5.0 + 0.6 * 3, abs=0.01)   # half of the 6 m wing, at the pitch


def test_shops_get_glass_fronts_on_the_street_and_no_windows_on_shared_walls():
    ring = box(0, 0, 10, 20)
    plan = B.Plan(ring, [], -0.4, 0.0, 9.0, "flat")
    street = lambda a, b: a[1] == 0 and b[1] == 0            # the south wall faces the street
    shared = lambda a, b: a[0] == 0 and b[0] == 0            # the west wall leans on a neighbour
    m = Mesh()
    B.draw(m, plan, B.Look((150, 84, 64), (150, 150, 150), (200, 190, 180), "shop", shopfront=True), shared, street)
    p, c = np.array(m.positions), np.array(m.colors)
    glass = (c == B.GLASS).all(axis=1)
    assert glass.any()
    assert not (glass & (np.abs(p[:, 0]) < 0.2)).any()        # nothing on the shared wall
    low_glass = glass & (p[:, 1] < 3.3) & (p[:, 2] > -0.2)    # the shop front, along the south wall
    assert low_glass.sum() >= 8


def test_a_courtyard_gets_walls_facing_into_it():
    plan = B.Plan(box(0, 0, 30, 30), [box(10, 10, 20, 20)], -0.4, 0.0, 9.0, "flat")
    m = Mesh()
    B.draw(m, plan, B.Look((150, 84, 64), (150, 150, 150), (200, 190, 180), "block"), lambda a, b: False, lambda a, b: False)
    windings_are_outward(m)
    p, n = np.array(m.positions), np.array(m.normals)
    yard_side = (np.abs(p[:, 0] - 10) < 0.05) & (n[:, 0] > 0.9)   # the wall at x=10 faces east, into the yard
    assert yard_side.any()


# ------------------------------------------------------------------ neighbours, trees, roof colours

def test_context_knows_shared_walls_streets_and_shop_districts():
    osm = OsmData()
    osm.areas = [Feature("w1", {"building": "yes"}, rings=[box(0, 0, 10, 10)]),
                 Feature("w2", {"building": "yes"}, rings=[box(10, 0, 20, 10)]),
                 Feature("w3", {"landuse": "retail"}, rings=[box(-50, -50, 50, 50)])]
    osm.lines = [Feature("w4", {"highway": "residential"}, line=np.array([[-50.0, -8.0], [50.0, -8.0]]))]
    ctx = BuildingContext(osm, (0, 0))
    party = ctx.party(osm.areas[0])
    assert party((10, 0), (10, 10)) and not party((0, 10), (0, 0))
    assert ctx.street_side((0, 0), (10, 0)) and not ctx.street_side((10, 10), (0, 10))
    assert ctx.kind(osm.areas[0], osm.areas[0].tags, box(0, 0, 10, 10)) == "shop"
    assert ctx.built(np.array([5.0, 30.0]), np.array([5.0, 30.0])).tolist() == [True, False]


def test_parts_are_drawn_instead_of_their_building():
    osm = OsmData()
    osm.areas = [Feature("w1", {"building": "church"}, rings=[box(0, 0, 20, 10)]),
                 Feature("w2", {"building:part": "yes", "roof:shape": "gabled"}, rings=[box(0, 0, 14, 10)]),
                 Feature("w3", {"building:part": "yes", "height": "30"}, rings=[box(14, 0, 20, 10)])]
    drawn = [(f.osm_id, tags.get("building")) for f, _, _, tags in BuildingContext(osm, (0, 0)).drawable()]
    assert drawn == [("w2", "church"), ("w3", "church")]


def test_treetops_one_a_crown_and_none_on_roofs():
    def canopy(e, n):
        a = 12 - np.hypot(e - 20, n - 20)          # a 12 m tree
        b = 8 - np.hypot(e - 60, n - 60)           # an 8 m tree
        return np.clip(np.maximum(a, b), 0, None)

    height = surface(canopy)
    lid = fake_lidar(height, (height > 0).astype(float))
    tops = treetops(lid, lambda e, n: np.zeros(len(e), bool))
    assert len(tops) == 2 and max(t[2] for t in tops) == pytest.approx(11.3, abs=0.8)
    big = max(tops, key=lambda t: t[2])
    assert big[3] == pytest.approx(6, abs=1.5)   # the 12 m cone falls to half its height 6 m out
    assert treetops(lid, lambda e, n: np.ones(len(e), bool)) == []
    assert mapped_height(lid, 21.0, 21.0) == pytest.approx(11.3, abs=0.8)
    assert mapped_height(lid, 90.0, 10.0) is None


def test_a_roof_colour_is_the_median_of_its_lit_pixels():
    image = np.zeros((100, 100, 3), np.uint8)
    image[30:70, 30:70] = (180, 60, 50)
    image[30:70, 30:40] = (20, 10, 10)                 # the shadowed slope
    rgb = sample(image, (0, 0, 100, 100), box(30, 30, 70, 70))
    assert rgb == (180, 60, 50)
    assert as_paint((128, 128, 128)) == (118, 118, 118)


# ------------------------------------------------------------------ cities: the grid in pieces

def test_fill_gaps_reaches_across_a_wide_gap_and_keeps_what_is_known():
    v = np.zeros((40, 40))
    known = np.zeros((40, 40), bool)
    known[:, :5], known[:, 35:] = True, True
    v[:, :5], v[:, 35:] = 10.0, 20.0
    out = fill_gaps(v, known)
    assert (out[:, :5] == 10).all() and (out[:, 35:] == 20).all()
    assert np.isfinite(out).all() and 10 <= out[:, 5:35].min() and out[:, 5:35].max() <= 20


def test_a_mosaic_samples_each_piece_and_reads_a_roof_across_the_seam(tmp_path):
    from streetkit.lidar import write
    write(tmp_path / "a.tif", (0, 0, 50, 50), np.full((50, 50), 6.0), np.zeros((50, 50)))
    write(tmp_path / "b.tif", (50, 0, 100, 50), np.full((50, 50), 9.0), np.zeros((50, 50)))
    lid = Lidar(tmp_path / "a.tif", tmp_path / "b.tif")
    h, leafy = lid.sample(np.array([10.0, 60.0, 500.0]), np.array([10.0, 10.0, 10.0]))
    assert h.tolist() == [6.0, 9.0, 0.0]          # nothing out of every piece
    e, n, hh, _ = lid.cells_in(box(45, 10, 55, 20))
    assert set(hh.tolist()) == {6.0, 9.0} and len(hh) == 100


def test_city_chunks_hold_their_tiles(monkeypatch):
    from streetkit.geo import Area
    from streetkit.lidar import city_chunks
    area = Area("c", 0, 0, 500.0, 0, 2.0, keys=((9, 9), (10, 9), (19, 0)))
    chunks = city_chunks(area)
    assert set(chunks) == {(0, 0), (1, 0)}   # tile (9, 9) is in square (0, 0); (10, 9) and (19, 0) in square (1, 0)
    assert chunks[(0, 0)] == [(4500.0, 4500.0, 5000.0, 5000.0)] and len(chunks[(1, 0)]) == 2


def test_surveys_are_newest_first_and_only_where_they_cover():
    from streetkit.lidar import surveys_over
    from streetkit.geo import utm_to_lonlat
    lon, lat = utm_to_lonlat(493000.0, 4492000.0)
    square = [[lon - 0.1, lat - 0.1], [lon + 0.1, lat - 0.1], [lon + 0.1, lat + 0.1], [lon - 0.1, lat + 0.1]]
    far = [[lon + 5, lat + 5], [lon + 6, lat + 5], [lon + 6, lat + 6]]
    index = [{"geometry": {"type": "Polygon", "coordinates": [square]}, "properties": {"name": "OLD_2013_LAS_2015"}},
             {"geometry": {"type": "Polygon", "coordinates": [square]}, "properties": {"name": "NEW_2021"}},
             {"geometry": {"type": "Polygon", "coordinates": [far]}, "properties": {"name": "ELSEWHERE_2023"}}]
    assert surveys_over([(492900.0, 4491900.0, 493100.0, 4492100.0)], index) == ["NEW_2021", "OLD_2013_LAS_2015"]
