"""Heights of everything standing on the ground, measured by USGS 3DEP airborne lidar.

The fetch picks the newest survey covering the area (or the one area.json names as "lidar_survey"), walks its
Entwine Point Tile index on the USGS open-data bucket, downloads every point tile over the area (cached under
build/fetched/lidar, never committed), and grids the points to one-metre cells:

    lidar-ndsm.tif  page 0: height of the highest return above the ground, uint16 centimetres (0 where bare)
                    page 1: share of the cell's returns that came from a pulse split by leaves, uint8 percent

Only those two grids are committed. The build reads them offline (`Lidar`) to measure each building's walls and
roof and each tree's height. The ground under a roof has no returns of its own, so it is filled in from the
surrounding ground cells.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import urllib.request
from pathlib import Path

import numpy as np

from .geo import UTM_EPSG, Area

BUCKETS = "https://s3-us-west-2.amazonaws.com/usgs-lidar-public/"
SURVEYS = "https://raw.githubusercontent.com/hobuinc/usgs-lidar/master/boundaries/resources.geojson"
CELL = 1.0
MARGIN = 4.0          # the DEM's margin, so both grids line up
GROUND, NOISE_LOW, NOISE_HIGH = 2, 7, 18
FILE = "lidar-ndsm.tif"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "streetkit/0.1 (open data 3D map)"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def grid_bounds(area: Area) -> tuple[float, float, float, float]:
    w, s, e, n = area.bounds()
    return w - MARGIN, s - MARGIN, e + MARGIN, n + MARGIN


def _inside(rings, x: float, y: float) -> bool:
    inside = False
    for ring in rings[:1]:
        for (x1, y1), (x2, y2) in zip(ring, ring[-1:] + ring[:-1]):
            if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
                inside = not inside
    return inside


def find_survey(lon: float, lat: float) -> str:
    """The name of the newest USGS survey on the open bucket that covers a point, from the bucket's index of
    survey boundaries. A name ends in the year it was flown, then the year it was published."""
    import re

    found = []
    for f in json.loads(_get(SURVEYS))["features"]:
        g = f["geometry"]
        polygons = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        if any(_inside(p, lon, lat) for p in polygons):
            years = [int(y) for y in re.findall(r"(?<!\d)(19\d\d|20\d\d)(?!\d)", f["properties"]["name"])]
            found.append((years[0] if years else 0, f["properties"]["name"]))
    if not found:
        raise RuntimeError(f"no USGS lidar survey on the open bucket covers {lat:.4f}, {lon:.4f}")
    return max(found)[1]


def _nodes(bucket: str, box: tuple[float, float, float, float]) -> list[str]:
    """Every point tile of the survey's index that overlaps a Web Mercator box."""
    ept = json.loads(_get(bucket + "ept.json"))
    b = ept["bounds"]
    hierarchy: dict = {}

    def load(name: str) -> None:
        hierarchy.update(json.loads(_get(f"{bucket}ept-hierarchy/{name}.json")))

    load("0-0-0-0")
    found, stack = [], [(0, 0, 0, 0)]
    while stack:
        d, x, y, z = stack.pop()
        key = f"{d}-{x}-{y}-{z}"
        side = (b[3] - b[0]) / 2 ** d
        x0, y0 = b[0] + x * side, b[1] + y * side
        if x0 > box[2] or x0 + side < box[0] or y0 > box[3] or y0 + side < box[1]:
            continue
        count = hierarchy.get(key)
        if count is None:
            continue
        if count == -1:
            load(key)
            count = hierarchy[key]
        if count > 0:
            found.append(key)
        stack += [(d + 1, 2 * x + i, 2 * y + j, 2 * z + k) for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    return found


def _download(bucket: str, keys: list[str], cache: Path) -> list[Path]:
    cache.mkdir(parents=True, exist_ok=True)

    def one(key: str) -> Path:
        path = cache / f"{key}.laz"
        if not path.exists():
            part = path.with_suffix(".part")
            part.write_bytes(_get(f"{bucket}ept-data/{key}.laz"))
            os.replace(part, path)
        return path

    with cf.ThreadPoolExecutor(12) as pool:
        return list(pool.map(one, keys))


def fill_holes(values: np.ndarray, known: np.ndarray, rounds: int = 400) -> np.ndarray:
    """Spreads known cells into unknown ones, a ring at a time, averaging the known neighbours."""
    v, k = np.where(known, values, 0.0).astype(np.float64), known.copy()
    for _ in range(rounds):
        if k.all():
            break
        total, count = np.zeros_like(v), np.zeros_like(v)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            total += np.roll(np.roll(np.where(k, v, 0), dy, 0), dx, 1)
            count += np.roll(np.roll(k, dy, 0), dx, 1)
        grow = ~k & (count > 0)
        v[grow] = total[grow] / count[grow]
        k |= grow
    return v


def grid_points(files: list[Path], bounds: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Returns (height above ground in metres, share of returns from split pulses) on the area's one-metre grid."""
    import laspy
    from pyproj import Transformer

    w, s, e, n = bounds
    nx, ny = int(round((e - w) / CELL)), int(round((n - s) / CELL))
    to_utm = Transformer.from_crs(3857, UTM_EPSG, always_xy=True)
    top = np.full(nx * ny, -np.inf)
    ground_sum, ground_count = np.zeros(nx * ny), np.zeros(nx * ny)
    returns, split = np.zeros(nx * ny), np.zeros(nx * ny)
    for path in files:
        las = laspy.read(path)
        cls = np.asarray(las.classification)
        x, y = to_utm.transform(np.asarray(las.x), np.asarray(las.y))
        keep = (x >= w) & (x < e) & (y >= s) & (y < n) & (cls != NOISE_LOW) & (cls != NOISE_HIGH)
        if not keep.any():
            continue
        i = ((n - y[keep]) / CELL).astype(int) * nx + ((x[keep] - w) / CELL).astype(int)
        z, c = np.asarray(las.z, np.float64)[keep], cls[keep]
        np.maximum.at(top, i, z)
        np.add.at(returns, i, 1)
        np.add.at(split, i, (np.asarray(las.number_of_returns)[keep] > 1).astype(float))
        g = c == GROUND
        np.add.at(ground_sum, i[g], z[g])
        np.add.at(ground_count, i[g], 1)
    ground = fill_holes((ground_sum / np.maximum(ground_count, 1)).reshape(ny, nx), (ground_count > 0).reshape(ny, nx))
    seen = (returns > 0).reshape(ny, nx)
    surface = fill_holes(top.reshape(ny, nx), seen, rounds=3)   # a missed metre here and there, not a gap
    height = np.clip(surface - ground, 0, None)
    leafy = np.where(seen, split.reshape(ny, nx) / np.maximum(returns.reshape(ny, nx), 1), 0)
    return height, leafy


def write(path: Path, bounds, height: np.ndarray, leafy: np.ndarray) -> None:
    import tifffile

    w, _, _, n = bounds
    geo = {"extratags": [(33550, "d", 3, (CELL, CELL, 0.0)), (33922, "d", 6, (0.0, 0.0, 0.0, w, n, 0.0))]}
    with tifffile.TiffWriter(path) as tif:
        tif.write(np.round(np.clip(height, 0, 655) * 100).astype(np.uint16), compression="zlib", predictor=True, **geo)
        tif.write(np.round(np.clip(leafy, 0, 1) * 100).astype(np.uint8), compression="zlib", predictor=True, **geo)


def fetch_lidar(area: Area, raw: Path, cache: Path, survey: str | None = None) -> tuple[Path, str]:
    """Grids the area's lidar into raw/lidar-ndsm.tif. `survey` names one (area.json's "lidar_survey"); without
    it, the newest survey covering the area's centre is used. Returns the file and the survey's index URL."""
    from pyproj import Transformer

    survey = survey or find_survey(area.centre_lon, area.centre_lat)
    bucket = f"{BUCKETS}{survey}/"
    cache = cache / survey
    bounds = grid_bounds(area)
    to_web = Transformer.from_crs(UTM_EPSG, 3857, always_xy=True)
    xs, ys = to_web.transform([bounds[0], bounds[2], bounds[0], bounds[2]], [bounds[1], bounds[1], bounds[3], bounds[3]])
    keys = _nodes(bucket, (min(xs), min(ys), max(xs), max(ys)))
    height, leafy = grid_points(_download(bucket, keys, cache), bounds)
    path = raw / FILE
    write(path, bounds, height, leafy)
    return path, bucket + "ept.json"


class Lidar:
    """The committed grids, sampled at the nearest cell. Heights are metres above the ground."""

    def __init__(self, path: Path):
        import tifffile

        with tifffile.TiffFile(path) as tif:
            self.height = tif.pages[0].asarray().astype(np.float32) / 100
            self.leafy = tif.pages[1].asarray().astype(np.float32) / 100
            scale, tie = tif.pages[0].tags["ModelPixelScaleTag"].value, tif.pages[0].tags["ModelTiepointTag"].value
        self.cell, self.w, self.n = float(scale[0]), float(tie[3]), float(tie[4])

    def index(self, east, north):
        col = np.clip(((np.asarray(east) - self.w) / self.cell).astype(int), 0, self.height.shape[1] - 1)
        row = np.clip(((self.n - np.asarray(north)) / self.cell).astype(int), 0, self.height.shape[0] - 1)
        return row, col

    def at(self, east, north):
        r, c = self.index(east, north)
        return self.height[r, c]

    def cells_in(self, ring: np.ndarray, inset: float = 0.0):
        """East, north, height and leafiness of every cell centre inside a ring, at least `inset` from its edge."""
        from .mesh import points_in_ring

        e0, n0 = ring.min(axis=0)
        e1, n1 = ring.max(axis=0)
        es = np.arange(np.floor((e0 - self.w) / self.cell), np.ceil((e1 - self.w) / self.cell)) * self.cell + self.w + self.cell / 2
        ns = self.n - (np.arange(np.floor((self.n - n1) / self.cell), np.ceil((self.n - n0) / self.cell)) * self.cell + self.cell / 2)
        ge, gn = np.meshgrid(es, ns)
        ge, gn = ge.ravel(), gn.ravel()
        inside = points_in_ring(ge, gn, ring)
        if inset > 0 and inside.any():
            inside &= edge_distance(ge, gn, ring) >= inset
        ge, gn = ge[inside], gn[inside]
        r, c = self.index(ge, gn)
        return ge, gn, self.height[r, c], self.leafy[r, c]


def edge_distance(px: np.ndarray, py: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest edge of a ring."""
    best = np.full(px.shape, np.inf)
    for a, b in zip(ring, np.roll(ring, -1, axis=0)):
        d = b - a
        length2 = float(d @ d) or 1e-12
        u = np.clip(((px - a[0]) * d[0] + (py - a[1]) * d[1]) / length2, 0, 1)
        best = np.minimum(best, np.hypot(px - (a[0] + u * d[0]), py - (a[1] + u * d[1])))
    return best
