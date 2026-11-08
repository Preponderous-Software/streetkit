"""What a building needs to know about its neighbours: which walls it shares, which face a street, whether it sits
in a shopping district, whether OSM splits it into parts, and its roof colour from the aerial sample."""

from __future__ import annotations

import collections
import math

import numpy as np

from .buildings import kind_of
from .mesh import clean_ring, points_in_ring, signed_area
from .roofs import as_paint

CELL = 25.0                      # metres a square of the spatial index
SHOP_LANDUSE = {"retail", "commercial"}
SHOP_ROADS = {"primary", "secondary", "tertiary", "unclassified", "residential", "pedestrian", "living_street"}
PARTS_COVER = 0.6                # a building drawn as its parts when they cover this much of it


class Grid:
    """Things with a bounding box, found by the squares they touch."""

    def __init__(self):
        self.cells = collections.defaultdict(list)

    def add(self, box, item) -> None:
        for i in range(math.floor(box[0] / CELL), math.floor(box[2] / CELL) + 1):
            for j in range(math.floor(box[1] / CELL), math.floor(box[3] / CELL) + 1):
                self.cells[(i, j)].append(item)

    def near(self, x: float, y: float):
        return self.cells.get((math.floor(x / CELL), math.floor(y / CELL)), ())


def box_of(points: np.ndarray):
    return points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()


def load_overrides(path) -> dict:
    """data/building-overrides.csv: facts about named buildings that a public source states (`source`), as OSM
    tags, for what neither OSM nor the lidar can tell: what a wall is made of, how many storeys it has."""
    import csv
    from pathlib import Path

    if path is None or not Path(path).exists():
        return {}
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[row["osm_id"]] = dict(pair.split("=", 1) for pair in row["tags"].split(";") if "=" in pair)
    return out


class BuildingContext:
    def __init__(self, osm, origin, lidar=None, roof_colours=None, overrides=None):
        self.origin = np.asarray(origin, float)
        self.overrides = overrides or {}
        self.lidar = lidar
        self.roof_colours = roof_colours or {}
        self.stats = collections.Counter()
        self.buildings = []
        self.parts = []
        self.shop_areas = Grid()
        for f in osm.areas:
            if "building" in f.tags and f.tags["building"] not in ("no", "construction", "proposed", "demolished"):
                rings = [clean_ring(r) for r in f.rings]
                if rings and len(rings[0]) >= 3:
                    self.buildings.append((f, rings))
            elif "building:part" in f.tags and f.tags["building:part"] != "no":
                ring = clean_ring(f.rings[0])
                if len(ring) >= 3:
                    self.parts.append((f, ring))
            elif f.tags.get("landuse") in SHOP_LANDUSE:
                for ring in f.rings:
                    ring = clean_ring(ring)
                    if len(ring) >= 3:
                        self.shop_areas.add(box_of(ring), ring)
        self.walls = Grid()
        for f, rings in self.buildings:
            for ring in rings:
                self.walls.add(box_of(ring), (f.osm_id, ring))
        self.roads = Grid()
        for f in osm.lines:
            if f.tags.get("highway") in SHOP_ROADS and f.tags.get("tunnel") not in ("yes",):
                for a, b in zip(f.line, f.line[1:]):
                    self.roads.add((min(a[0], b[0]) - 15, min(a[1], b[1]) - 15, max(a[0], b[0]) + 15, max(a[1], b[1]) + 15), (a, b))
        self.parts_of = collections.defaultdict(list)
        for part, ring in self.parts:
            c = ring.mean(axis=0)
            for fid, building in self.walls.near(*c):
                if points_in_ring(np.array([c[0]]), np.array([c[1]]), building)[0]:
                    self.parts_of[fid].append((part, ring))
                    break

    # -------------------------------------------------------------- what gets drawn

    def drawable(self):
        """(feature, outer ring, holes, tags) for every block to draw: a building, or its parts where OSM has them."""
        for f, rings in self.buildings:
            parts = self.parts_of.get(f.osm_id, [])
            whole = sum(abs(signed_area(r)) for r in rings)
            covered = sum(abs(signed_area(r)) for _, r in parts)
            if parts and covered >= PARTS_COVER * whole:
                self.stats["drawn as parts"] += 1
                for part, ring in parts:
                    tags = dict(f.tags)
                    tags.update(part.tags)
                    tags.setdefault("building", f.tags.get("building", "yes"))
                    yield part, ring, [], tags
                continue
            holes = [clean_ring(h) for h in getattr(f, "holes", []) if len(h) >= 3]
            tags = {**f.tags, **self.overrides.get(f.osm_id, {})}
            for i, ring in enumerate(rings):
                yield f, ring, holes if i == 0 else [], tags

    def footprints(self):
        """Every building, for walking into: whole outlines, never parts."""
        return [f for f, _ in self.buildings]

    def count(self, plan) -> None:
        self.stats[f"height from {plan.source}"] += 1
        self.stats[f"roof {plan.style}"] += 1
        self.stats["towers"] += len(plan.towers)
        self.stats["stepped"] += bool(plan.steps)

    # -------------------------------------------------------------- what it looks like

    def in_shop_district(self, ring) -> bool:
        c = ring.mean(axis=0)
        return any(points_in_ring(np.array([c[0]]), np.array([c[1]]), r)[0] for r in self.shop_areas.near(*c))

    def kind(self, f, tags, ring) -> str:
        k = kind_of(tags, abs(signed_area(ring)))
        if k in ("house", "block") and tags.get("building", "yes") == "yes" and self.in_shop_district(ring):
            return "shop"
        return k

    def shopfront(self, f, tags, ring) -> bool:
        return self.kind(f, tags, ring) == "shop" and abs(signed_area(ring)) > 60

    def roof(self, f):
        rgb = self.roof_colours.get(f.osm_id)
        return as_paint(rgb) if rgb else None

    def built(self, east: np.ndarray, north: np.ndarray) -> np.ndarray:
        """Which points stand inside a building."""
        out = np.zeros(len(east), bool)
        for i, (e, n) in enumerate(zip(east, north)):
            for _, ring in self.walls.near(e, n):
                if points_in_ring(np.array([e]), np.array([n]), ring)[0]:
                    out[i] = True
                    break
        return out

    # -------------------------------------------------------------- walls

    def party(self, f):
        """For one building: does a wall from a to b (local) lean on a neighbour?"""
        me = f.osm_id

        def shared(a, b) -> bool:
            a, b = np.asarray(a) + self.origin, np.asarray(b) + self.origin
            d = b - a
            length = float(np.hypot(*d))
            if length < 1e-6:
                return False
            out = np.array([d[1], -d[0]]) / length
            probe = (a + b) / 2 + out * 0.6
            for fid, ring in self.walls.near(*probe):
                if fid != me and points_in_ring(np.array([probe[0]]), np.array([probe[1]]), ring)[0]:
                    return True
            return False

        return shared

    def street_side(self, a, b) -> bool:
        """Does a wall from a to b (local) face a street within about 18 metres?"""
        a, b = np.asarray(a) + self.origin, np.asarray(b) + self.origin
        d = b - a
        length = float(np.hypot(*d))
        if length < 1e-6:
            return False
        out = np.array([d[1], -d[0]]) / length
        mid = (a + b) / 2
        for step in (4.0, 9.0, 14.0, 18.0):
            p = mid + out * step
            for s0, s1 in self.roads.near(*p):
                e = s1 - s0
                u = np.clip(np.dot(p - s0, e) / max(np.dot(e, e), 1e-9), 0, 1)
                if np.hypot(*(p - (s0 + e * u))) < 4.0:
                    return True
        return False
