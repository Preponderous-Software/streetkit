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


_hierarchies: dict = {}


def _nodes(bucket: str, boxes: list) -> list[str]:
    """Every point tile of the survey's index that overlaps any of the Web Mercator boxes."""
    if bucket not in _hierarchies:
        _hierarchies[bucket] = (json.loads(_get(bucket + "ept.json")), {})
    ept, hierarchy = _hierarchies[bucket]
    b = ept["bounds"]

    def load(name: str) -> None:
        hierarchy.update(json.loads(_get(f"{bucket}ept-hierarchy/{name}.json")))

    if "0-0-0-0" not in hierarchy:
        load("0-0-0-0")
    found, stack = [], [(0, 0, 0, 0)]
    while stack:
        d, x, y, z = stack.pop()
        key = f"{d}-{x}-{y}-{z}"
        side = (b[3] - b[0]) / 2 ** d
        x0, y0 = b[0] + x * side, b[1] + y * side
        if not any(x0 <= box[2] and x0 + side >= box[0] and y0 <= box[3] and y0 + side >= box[1] for box in boxes):
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


def _download(bucket: str, keys: list[str], cache: Path, workers: int = 12) -> list[Path]:
    cache.mkdir(parents=True, exist_ok=True)

    def one(key: str) -> Path:
        path = cache / f"{key}.laz"
        if not path.exists():
            part = path.with_suffix(f".part{os.getpid()}")
            for attempt in range(5):
                try:
                    part.write_bytes(_get(f"{bucket}ept-data/{key}.laz"))
                    break
                except OSError:
                    if attempt == 4:
                        raise
            os.replace(part, path)
        return path

    with cf.ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, keys))


def fill_holes(values: np.ndarray, known: np.ndarray, rounds: int = 400) -> np.ndarray:
    """Spreads known cells into unknown ones, a ring at a time, averaging the known neighbours. For small gaps."""
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


def fill_gaps(values: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Every unknown cell from the known cells around it, however wide the gap: the ground under a large roof. Halves
    the grid until every coarse cell is known, then fills each level from the one above it, so it costs a few passes
    over the grid whatever the gap."""
    if known.all() or not known.any():
        return np.where(known, values, 0.0).astype(np.float64)
    h, w = values.shape
    ph, pw = h + h % 2, w + w % 2
    v = np.zeros((ph, pw))
    k = np.zeros((ph, pw))
    v[:h, :w] = np.where(known, values, 0.0)
    k[:h, :w] = known
    total = v.reshape(ph // 2, 2, pw // 2, 2).sum(axis=(1, 3))
    count = k.reshape(ph // 2, 2, pw // 2, 2).sum(axis=(1, 3))
    coarse = fill_gaps(np.where(count > 0, total / np.maximum(count, 1), 0.0), count > 0)
    up = np.repeat(np.repeat(coarse, 2, axis=0), 2, axis=1)[:h, :w]
    return np.where(known, values, up)


def grid_points(files: list[Path], bounds: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (height above ground in metres, share of returns from split pulses, cells with any return) on a
    one-metre grid over bounds."""
    import laspy
    from pyproj import Transformer

    w, s, e, n = bounds
    nx, ny = int(round((e - w) / CELL)), int(round((n - s) / CELL))
    to_utm = Transformer.from_crs(3857, UTM_EPSG, always_xy=True)
    top = np.full(nx * ny, -np.inf, np.float32)
    ground_sum, ground_count = np.zeros(nx * ny), np.zeros(nx * ny, np.int32)
    returns, split = np.zeros(nx * ny, np.int32), np.zeros(nx * ny, np.int32)
    for path in files:
        las = laspy.read(path)
        cls = np.asarray(las.classification)
        x, y = to_utm.transform(np.asarray(las.x), np.asarray(las.y))
        keep = (x >= w) & (x < e) & (y >= s) & (y < n) & (cls != NOISE_LOW) & (cls != NOISE_HIGH)
        if not keep.any():
            continue
        i = ((n - y[keep]) / CELL).astype(np.int64) * nx + ((x[keep] - w) / CELL).astype(np.int64)
        z, c = np.asarray(las.z, np.float64)[keep], cls[keep]
        np.maximum.at(top, i, z.astype(np.float32))
        np.add.at(returns, i, 1)
        np.add.at(split, i, (np.asarray(las.number_of_returns)[keep] > 1).astype(np.int32))
        g = c == GROUND
        np.add.at(ground_sum, i[g], z[g])
        np.add.at(ground_count, i[g], 1)
    ground = fill_gaps((ground_sum / np.maximum(ground_count, 1)).reshape(ny, nx), (ground_count > 0).reshape(ny, nx))
    seen = (returns > 0).reshape(ny, nx)
    surface = fill_holes(top.reshape(ny, nx), seen, rounds=3)   # a missed metre here and there, not a gap
    height = np.clip(surface - ground, 0, None)
    height[~seen & ~np.isfinite(surface)] = 0
    leafy = np.where(seen, split.reshape(ny, nx) / np.maximum(returns.reshape(ny, nx), 1), 0)
    return height, leafy, seen


def write(path: Path, bounds, height: np.ndarray, leafy: np.ndarray) -> None:
    import tifffile

    w, _, _, n = bounds
    geo = {"extratags": [(33550, "d", 3, (CELL, CELL, 0.0)), (33922, "d", 6, (0.0, 0.0, 0.0, w, n, 0.0))]}
    with tifffile.TiffWriter(path) as tif:
        tif.write(np.round(np.clip(np.nan_to_num(height), 0, 655) * 100).astype(np.uint16), compression="zlib", predictor=True, **geo)
        tif.write(np.round(np.clip(leafy, 0, 1) * 100).astype(np.uint8), compression="zlib", predictor=True, **geo)


def _web_box(bounds):
    from pyproj import Transformer

    to_web = Transformer.from_crs(UTM_EPSG, 3857, always_xy=True)
    xs, ys = to_web.transform([bounds[0], bounds[2], bounds[0], bounds[2]], [bounds[1], bounds[1], bounds[3], bounds[3]])
    return min(xs), min(ys), max(xs), max(ys)


def fetch_lidar(area: Area, raw: Path, cache: Path, survey: str | None = None) -> tuple[Path, str]:
    """Grids the area's lidar into raw/lidar-ndsm.tif. `survey` names one (area.json's "lidar_survey"); without
    it, the newest survey covering the area's centre is used. Returns the file and the survey's index URL."""
    survey = survey or find_survey(area.centre_lon, area.centre_lat)
    bucket = f"{BUCKETS}{survey}/"
    bounds = grid_bounds(area)
    keys = _nodes(bucket, [_web_box(bounds)])
    height, leafy, _ = grid_points(_download(bucket, keys, cache / survey), bounds)
    path = raw / FILE
    write(path, bounds, height, leafy)
    return path, bucket + "ept.json"


CHUNK = 5000.0   # a city's grid is kept in squares this wide, like its elevation


def city_chunks(area: Area) -> dict:
    """The CHUNK squares that hold any of a city's tiles, each with the boxes of those tiles."""
    size = area.tile_size
    out: dict = {}
    for ix, iy in area.tiles():
        box = (ix * size, iy * size, (ix + 1) * size, (iy + 1) * size)
        for ce in range(int(np.floor(box[0] / CHUNK)), int(np.floor((box[2] - 1e-6) / CHUNK)) + 1):
            for cn in range(int(np.floor(box[1] / CHUNK)), int(np.floor((box[3] - 1e-6) / CHUNK)) + 1):
                out.setdefault((ce, cn), []).append(box)
    return out


def chunk_name(ce: int, cn: int) -> str:
    return f"lidar-{ce}-{cn}.tif"


def surveys_over(boxes: list, index: list) -> list[str]:
    """Every survey in the index covering any tile centre or corner in `boxes`, newest first."""
    import re

    from .geo import utm_to_lonlat

    pts = []
    for w, s, e, n in boxes:
        pts += [((w + e) / 2, (s + n) / 2), (w + 1, s + 1), (e - 1, s + 1), (w + 1, n - 1), (e - 1, n - 1)]
    lon, lat = utm_to_lonlat(np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
    box = (lon.min(), lat.min(), lon.max(), lat.max())
    found = {}
    for f in index:
        g = f["geometry"]
        polygons = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        name = f["properties"]["name"]
        outer = np.array([pt for p in polygons for pt in p[0]])
        if outer[:, 0].max() < box[0] or outer[:, 0].min() > box[2] or outer[:, 1].max() < box[1] or outer[:, 1].min() > box[3]:
            continue   # nowhere near: most of the country
        if any(_inside(p, x, y) for p in polygons for x, y in zip(lon, lat)):
            years = [int(v) for v in re.findall(r"(?<!\d)(19\d\d|20\d\d)(?!\d)", name)]
            found[name] = years[0] if years else 0
    return sorted(found, key=lambda k: -found[k])


def grid_chunk(job) -> tuple:
    """One CHUNK square from every survey over it: the newest survey's cells where it saw anything, the older ones
    beneath. Runs in its own process."""
    (ce, cn), boxes, surveys, cache = job
    bounds = (ce * CHUNK, cn * CHUNK, (ce + 1) * CHUNK, (cn + 1) * CHUNK)
    height = leafy = have = None
    for survey in surveys:
        bucket = f"{BUCKETS}{survey}/"
        keys = _nodes(bucket, [_web_box(b) for b in boxes])
        if not keys:
            continue
        h, lf, seen = grid_points(_download(bucket, keys, cache / survey, workers=4), bounds)
        if height is None:
            height, leafy, have = h, lf, seen
            continue
        fill = ~have & seen
        height[fill], leafy[fill] = h[fill], lf[fill]
        have |= seen
    return (ce, cn), bounds, height, leafy


def fetch_lidar_city(area: Area, raw: Path, cache: Path, workers: int = 4, say=lambda text: print(text, flush=True)) -> list[tuple[Path, str]]:
    """Grids a city's lidar, one file a CHUNK square, into raw/. Returns each file with the surveys it used."""
    import multiprocessing

    index = json.loads(_get(SURVEYS))["features"]
    chunks = city_chunks(area)
    jobs = [(key, boxes, surveys_over(boxes, index), cache) for key, boxes in sorted(chunks.items())]
    out = []
    pending = []
    for job in jobs:
        path = raw / chunk_name(*job[0])
        if path.exists():
            out.append((path, " ".join(f"{BUCKETS}{s}/ept.json" for s in job[2])))
        else:
            pending.append(job)
    say(f"{len(jobs)} squares, {len(pending)} to grid, surveys: {sorted({s for j in jobs for s in j[2]})}")
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        for key, bounds, height, leafy in pool.imap_unordered(grid_chunk, pending):
            path = raw / chunk_name(*key)
            if height is not None:
                part = path.with_suffix(".part")
                write(part, bounds, height, leafy)
                os.replace(part, path)
                surveys = next(j[2] for j in jobs if j[0] == key)
                out.append((path, " ".join(f"{BUCKETS}{s}/ept.json" for s in surveys)))
            say(f"gridded square {key}")
    return sorted(out)


class Chunk:
    """One committed or fetched grid file, sampled at the nearest cell."""

    def __init__(self, path: Path):
        import tifffile

        with tifffile.TiffFile(path) as tif:
            self.height = tif.pages[0].asarray().astype(np.float32) / 100
            self.leafy = tif.pages[1].asarray().astype(np.float32) / 100
            scale, tie = tif.pages[0].tags["ModelPixelScaleTag"].value, tif.pages[0].tags["ModelTiepointTag"].value
        self.cell, self.w, self.n = float(scale[0]), float(tie[3]), float(tie[4])
        self.e = self.w + self.height.shape[1] * self.cell
        self.s = self.n - self.height.shape[0] * self.cell

    def index(self, east, north):
        col = np.clip(((np.asarray(east) - self.w) / self.cell).astype(int), 0, self.height.shape[1] - 1)
        row = np.clip(((self.n - np.asarray(north)) / self.cell).astype(int), 0, self.height.shape[0] - 1)
        return row, col

    def holds(self, east, north):
        east, north = np.asarray(east), np.asarray(north)
        return (east >= self.w) & (east < self.e) & (north > self.s) & (north <= self.n)


class Lidar:
    """The grids of an area: one file for a small area, a file a CHUNK square for a city. Heights are metres above
    the ground; a point no grid holds reads as bare ground with no leaves."""

    def __init__(self, *paths: Path):
        self.chunks = [Chunk(p) for p in paths]
        first = self.chunks[0]
        self.cell = first.cell
        if len(self.chunks) == 1:   # the arrays themselves, for code that reads one grid whole
            self.height, self.leafy, self.w, self.n = first.height, first.leafy, first.w, first.n

    def index(self, east, north):
        return self.chunks[0].index(east, north)

    def sample(self, east, north):
        east, north = np.atleast_1d(np.asarray(east, np.float64)), np.atleast_1d(np.asarray(north, np.float64))
        h, lf = np.zeros(east.shape, np.float32), np.zeros(east.shape, np.float32)
        for c in self.chunks:
            inside = c.holds(east, north) if len(self.chunks) > 1 else np.ones(east.shape, bool)
            if inside.any():
                r, col = c.index(east[inside], north[inside])
                h[inside], lf[inside] = c.height[r, col], c.leafy[r, col]
        return h, lf

    def at(self, east, north):
        return self.sample(east, north)[0]

    def cells_in(self, ring: np.ndarray, inset: float = 0.0):
        """East, north, height and leafiness of every cell centre inside a ring, at least `inset` from its edge."""
        from .mesh import points_in_ring

        e0, n0 = ring.min(axis=0)
        e1, n1 = ring.max(axis=0)
        w = self.chunks[0].w
        top = self.chunks[0].n
        es = w + (np.arange(np.floor((e0 - w) / self.cell), np.ceil((e1 - w) / self.cell)) + 0.5) * self.cell
        ns = top - (np.arange(np.floor((top - n1) / self.cell), np.ceil((top - n0) / self.cell)) + 0.5) * self.cell
        ge, gn = np.meshgrid(es, ns)
        ge, gn = ge.ravel(), gn.ravel()
        inside = points_in_ring(ge, gn, ring)
        if inset > 0 and inside.any():
            inside &= edge_distance(ge, gn, ring) >= inset
        ge, gn = ge[inside], gn[inside]
        h, lf = self.sample(ge, gn)
        return ge, gn, h, lf


def edge_distance(px: np.ndarray, py: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest edge of a ring."""
    best = np.full(px.shape, np.inf)
    for a, b in zip(ring, np.roll(ring, -1, axis=0)):
        d = b - a
        length2 = float(d @ d) or 1e-12
        u = np.clip(((px - a[0]) * d[0] + (py - a[1]) * d[1]) / length2, 0, 1)
        best = np.minimum(best, np.hypot(px - (a[0] + u * d[0]), py - (a[1] + u * d[1])))
    return best
