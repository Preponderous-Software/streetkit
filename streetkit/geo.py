"""The tile grid. Tiles are squares of a fixed UTM zone, so a tile's name never depends on the area built."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from pyproj import Transformer

UTM_EPSG = 32613  # WGS 84 / UTM zone 13N


@dataclass(frozen=True)
class Area:
    name: str
    centre_lat: float
    centre_lon: float
    tile_size: float
    tiles_across: int
    cell: float
    overview_cell: float = 0.0            # metres per sample of the whole-area overview; 0 for none
    boundary_relation: int = 0            # an OSM boundary relation picks the tiles; 0 keeps the square
    margin: float = 0.0                   # metres kept beyond that boundary
    osm_extract: str = ""                 # a dated .osm.pbf URL for areas too large for the OSM API
    keys: tuple | None = None             # the tiles, once a boundary has picked them

    @staticmethod
    def load(path: Path) -> "Area":
        d = json.loads(Path(path).read_text())
        return Area(d["name"], d["centre_lat"], d["centre_lon"], float(d["tile_size_m"]), int(d.get("tiles_across", 0)),
                    float(d["cell_m"]), float(d.get("overview_cell_m", 0)), int(d.get("boundary_relation", 0)),
                    float(d.get("margin_m", 0)), d.get("osm_extract", ""))

    def within(self, rings: list) -> "Area":
        """This area with its tiles picked by boundary rings instead of a square."""
        return replace(self, keys=tuple(select_tiles(rings, self.tile_size, self.margin)))

    def centre_utm(self) -> tuple[float, float]:
        return Transformer.from_crs(4326, UTM_EPSG, always_xy=True).transform(self.centre_lon, self.centre_lat)

    def tiles(self) -> list[tuple[int, int]]:
        """Every (ix, iy) in the area, west to east, south to north."""
        if self.keys is not None:
            return list(self.keys)
        e, n = self.centre_utm()
        half = self.tiles_across // 2
        ix0, iy0 = tile_of(e, n, self.tile_size)
        return [(ix, iy) for iy in range(iy0 - half, iy0 - half + self.tiles_across)
                for ix in range(ix0 - half, ix0 - half + self.tiles_across)]

    def bounds(self) -> tuple[float, float, float, float]:
        """UTM west, south, east, north of the whole area."""
        keys = self.tiles()
        return (min(k[0] for k in keys) * self.tile_size, min(k[1] for k in keys) * self.tile_size,
                (max(k[0] for k in keys) + 1) * self.tile_size, (max(k[1] for k in keys) + 1) * self.tile_size)


def select_tiles(rings: list, size: float, margin: float) -> list[tuple[int, int]]:
    """Every tile within `margin` of the rings, or inside one, south to north. The result need not be a rectangle."""
    keep = set()
    reach = int(math.ceil(margin / size))
    for ring in rings:
        closed = np.vstack([ring, ring[:1]])
        for a, b in zip(closed, closed[1:]):
            for t in np.linspace(0, 1, max(2, int(np.linalg.norm(b - a) / (size / 2)) + 2)):
                ix, iy = tile_of(*(a + (b - a) * t), size)
                for dy in range(-reach, reach + 1):
                    for dx in range(-reach, reach + 1):
                        # The tile's nearest point to this edge point, so the margin is round, not square.
                        px, py = (a + (b - a) * t)
                        wx, wy = (ix + dx) * size, (iy + dy) * size
                        nx, ny = min(max(px, wx), wx + size), min(max(py, wy), wy + size)
                        if math.hypot(px - nx, py - ny) <= margin:
                            keep.add((ix + dx, iy + dy))
    # Tiles wholly inside a ring touch no edge, so fill by testing their centres.
    if rings:
        xs = [k[0] for k in keep]
        ys = [k[1] for k in keep]
        ix, iy = np.meshgrid(np.arange(min(xs), max(xs) + 1), np.arange(min(ys), max(ys) + 1))
        for ring in rings:
            inside = np.zeros(ix.shape, dtype=bool)
            px, py = (ix + 0.5) * size, (iy + 0.5) * size
            x0, y0 = ring[-1]
            for x1, y1 in ring:
                crosses = (y1 > py) != (y0 > py)
                with np.errstate(divide="ignore", invalid="ignore"):
                    inside ^= crosses & (px < (x0 - x1) * (py - y1) / (y0 - y1) + x1)
                x0, y0 = x1, y1
            keep |= {(int(a), int(b)) for a, b in zip(ix[inside], iy[inside])}
    return sorted(keep, key=lambda k: (k[1], k[0]))


def tile_of(east: float, north: float, size: float) -> tuple[int, int]:
    return math.floor(east / size), math.floor(north / size)


def tile_north_west(ix: int, iy: int, size: float) -> tuple[float, float]:
    """UTM east, north of a tile's corner; tile-local x runs east from it and z runs south."""
    return ix * size, (iy + 1) * size


def utm_to_lonlat(east, north):
    return Transformer.from_crs(UTM_EPSG, 4326, always_xy=True).transform(east, north)


def lonlat_to_utm(lon, lat):
    return Transformer.from_crs(4326, UTM_EPSG, always_xy=True).transform(lon, lat)
