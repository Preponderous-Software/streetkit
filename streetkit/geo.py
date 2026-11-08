"""The tile grid. Tiles are squares of a fixed UTM zone, so a tile's name never depends on the area built."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

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

    @staticmethod
    def load(path: Path) -> "Area":
        d = json.loads(Path(path).read_text())
        return Area(d["name"], d["centre_lat"], d["centre_lon"], float(d["tile_size_m"]), int(d["tiles_across"]),
                    float(d["cell_m"]))

    def centre_utm(self) -> tuple[float, float]:
        return Transformer.from_crs(4326, UTM_EPSG, always_xy=True).transform(self.centre_lon, self.centre_lat)

    def tiles(self) -> list[tuple[int, int]]:
        """Every (ix, iy) in the area, west to east, south to north."""
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


def tile_of(east: float, north: float, size: float) -> tuple[int, int]:
    return math.floor(east / size), math.floor(north / size)


def tile_north_west(ix: int, iy: int, size: float) -> tuple[float, float]:
    """UTM east, north of a tile's corner; tile-local x runs east from it and z runs south."""
    return ix * size, (iy + 1) * size


def utm_to_lonlat(east, north):
    return Transformer.from_crs(UTM_EPSG, 4326, always_xy=True).transform(east, north)


def lonlat_to_utm(lon, lat):
    return Transformer.from_crs(4326, UTM_EPSG, always_xy=True).transform(lon, lat)
