"""Writes one tile file per grid square, plus a manifest. app/tile.cpp reads the same layout.

A tile's coordinates are local to its north-west corner, so precision does not depend on where on the map the
tile is; the viewer adds the offset when it loads one. Heights are metres above the manifest's datum.

    tile     := header heights colors mesh* footprint* tree*
    header   := "OTWTILE1" u32 version  i32 ix iy  f32 size cell  i32 samples meshes footprints trees
    heights  := (samples + 2)^2 f32, row by row from the north, with one extra sample all round for normals
    colors   := samples^2 sRGB bytes * 3, padded to 4 bytes
    mesh     := u32 kind  i32 vertices indices  xyz f32  normal i8[3]+pad  colour u8[3]+pad  index u16 or u32, padded
    footprint:= i32 points  x z f32
    tree     := x y z height crown r g b, f32

Indices are u16 when the mesh has fewer than 65536 vertices. Ponderosa wrote the same layout as "PDRTILE1", which
the reader also accepts.
"""

from __future__ import annotations

import hashlib
import math
import multiprocessing
import os
import struct
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geo import Area, tile_north_west, tile_of
from .buildings import colours, draw, parse_length, pick, plan_building
from .context import BuildingContext, Grid
from .trees import mapped_height, treetops
from .mesh import Mesh, clean_ring, points_in_ring, ribbon_quads, signed_area
from .osm import OsmData

MAGIC = b"OTWTILE1"
VERSION = 1
HEADER = struct.Struct("<8sIiiffiiii")
TREE = struct.Struct("<8f")
KINDS = {"buildings": 0, "roads": 1, "paths": 2}

GRASS = (112, 142, 84)
AREA_COLOURS = [
    (lambda t: t.get("natural") == "water" or t.get("landuse") in ("reservoir", "basin") or "water" in t, (72, 108, 138)),
    (lambda t: t.get("natural") in ("wood", "scrub") or t.get("landuse") == "forest", (72, 104, 58)),
    (lambda t: t.get("leisure") in ("park", "garden", "pitch", "recreation_ground", "golf_course")
        or t.get("landuse") in ("grass", "recreation_ground", "cemetery", "meadow"), (100, 148, 76)),
    (lambda t: t.get("amenity") in ("parking", "fuel") or t.get("landuse") in ("commercial", "retail", "industrial", "railway", "construction")
        or t.get("place") == "square", (138, 136, 130)),
    (lambda t: t.get("landuse") == "residential", (118, 146, 90)),
]
ASPHALT, CONCRETE = (66, 66, 70), (186, 182, 174)
ROAD_WIDTHS = {"motorway": 12, "trunk": 11, "primary": 11, "secondary": 10, "tertiary": 9, "unclassified": 7,
               "residential": 7.5, "living_street": 6, "service": 4, "motorway_link": 6, "trunk_link": 6,
               "primary_link": 6, "secondary_link": 6, "tertiary_link": 6}
PATH_WIDTHS = {"footway": 1.8, "path": 1.5, "cycleway": 2.5, "pedestrian": 4, "steps": 1.5, "track": 3}


class Dem:
    """North-up float32 GeoTIFFs in the grid's UTM zone, sampled bilinearly at pixel centres.

    Several files of the same pixel size become one mosaic. They may overlap, and may leave gaps at the corners
    of an area that is not a rectangle; a gap reads as the median height.
    """

    def __init__(self, *paths: Path):
        import tifffile
        parts = []
        for path in paths:
            with tifffile.TiffFile(path) as tif:
                page = tif.pages[0]
                parts.append((page.asarray().astype(np.float32), page.tags["ModelPixelScaleTag"].value,
                              page.tags["ModelTiepointTag"].value))
        self.sx, self.sy = float(parts[0][1][0]), float(parts[0][1][1])
        self.e0, self.n0 = min(float(p[2][3]) for p in parts), max(float(p[2][4]) for p in parts)
        cols = max(round((float(p[2][3]) - self.e0) / self.sx) + p[0].shape[1] for p in parts)
        rows = max(round((self.n0 - float(p[2][4])) / self.sy) + p[0].shape[0] for p in parts)
        self.z = np.full((rows, cols), np.nan, dtype=np.float32)
        for z, _, tie in parts:
            c, r = round((float(tie[3]) - self.e0) / self.sx), round((self.n0 - float(tie[4])) / self.sy)
            self.z[r:r + z.shape[0], c:c + z.shape[1]] = z
        bad = ~(self.z > -1000)
        if bad.any():
            self.z[bad] = np.median(self.z[~bad])

    def height(self, east, north):
        col = np.clip((np.asarray(east, np.float64) - self.e0) / self.sx - 0.5, 0, self.z.shape[1] - 1.000001)
        row = np.clip((self.n0 - np.asarray(north, np.float64)) / self.sy - 0.5, 0, self.z.shape[0] - 1.000001)
        c, r = np.floor(col).astype(int), np.floor(row).astype(int)
        fc, fr = col - c, row - r
        top = self.z[r, c] * (1 - fc) + self.z[r, c + 1] * fc
        bottom = self.z[r + 1, c] * (1 - fc) + self.z[r + 1, c + 1] * fc
        return top * (1 - fr) + bottom * fr


@dataclass
class Overview:
    """The whole area at low resolution, always below the detailed tiles so they cover it where they are loaded."""
    east: float                  # UTM of the sample in the north-west corner
    north: float
    cell: float
    heights: np.ndarray          # rows from the north
    colours: np.ndarray          # sRGB bytes


OVERVIEW_DROP = 1.0              # metres below the lowest detailed ground in reach of a sample
OVERVIEW_MAX_SAMPLES = 1025      # what one Strata terrain tile holds per side
OVERVIEW_MAGIC = b"PDROVER1"     # the format's name from Ponderosa, where it began
OVERVIEW_ROOF = (120, 112, 104)  # roofs smaller than a sample tint it this colour
OVERVIEW_HEADER = struct.Struct("<8sIiifdd")


def overview_cell(area: Area) -> float:
    """The requested spacing, or the smallest whole number of metres that fits one terrain tile."""
    west, south, east, north = area.bounds()
    return max(area.overview_cell, float(math.ceil(max(east - west, north - south) / (OVERVIEW_MAX_SAMPLES - 1))))


def build_overview(area: Area, osm: OsmData, painted: list, dem: Dem, datum: float) -> Overview:
    west, south, east, north = area.bounds()
    cell = overview_cell(area)
    cols, rows = int(math.ceil((east - west) / cell)) + 1, int(math.ceil((north - south) / cell)) + 1
    ex, ny = west + np.arange(cols) * cell, north - np.arange(rows) * cell

    # A cell's lowest ground, sampled at the DEM's own spacing. Each corner takes the lowest of the cells around
    # it, so the interpolated surface stays below the detailed ground everywhere.
    per_cell = max(2, int(round(cell / dem.sx)))
    fine = (np.arange(per_cell) + 0.5) / per_cell * cell
    lowest = np.empty((rows - 1, cols - 1), dtype=np.float32)
    for r in range(rows - 1):
        gx, gy = np.meshgrid(ex[:-1, None] + fine, ny[r] - fine, sparse=False, indexing="xy")
        lowest[r] = dem.height(gx.reshape(-1), gy.reshape(-1)).reshape(gx.shape).min(axis=0).reshape(-1, per_cell).min(axis=1)
    pad = np.pad(lowest, 1, mode="edge")
    heights = np.minimum.reduce([pad[:-1, :-1], pad[:-1, 1:], pad[1:, :-1], pad[1:, 1:]]) - datum - OVERVIEW_DROP

    colours = np.empty((rows, cols, 3), dtype=np.float32)
    colours[:] = GRASS
    gx, gy = np.meshgrid(ex, ny)
    for box, ring, rgb in painted:
        c0, c1 = max(0, int((box[0] - west) // cell)), min(cols - 1, int((box[2] - west) // cell) + 1)
        r0, r1 = max(0, int((north - box[3]) // cell)), min(rows - 1, int((north - box[1]) // cell) + 1)
        if c0 > c1 or r0 > r1:
            continue
        window = (slice(r0, r1 + 1), slice(c0, c1 + 1))
        colours[window][points_in_ring(gx[window], gy[window], ring)] = rgb

    # Roofs and roads are smaller than a sample, so they tint it by how much of it they cover.
    roofs, roads = np.zeros((rows, cols), np.float32), np.zeros((rows, cols), np.float32)
    def cover(grid, e, n, amount):
        c, r = int(round((e - west) / cell)), int(round((north - n) / cell))
        if 0 <= c < cols and 0 <= r < rows:
            grid[r, c] += amount
    for f in osm.areas:
        if "building" in f.tags:
            ring = f.rings[0]
            cover(roofs, *ring.mean(axis=0), abs(signed_area(ring)))
    for f in osm.lines:
        hw = f.tags.get("highway")
        width = ROAD_WIDTHS.get(hw)
        if width is None or f.line is None:
            continue
        for a, b in zip(f.line, f.line[1:]):
            cover(roads, *((a + b) / 2), float(np.linalg.norm(b - a)) * width)
    for grid, rgb in ((roofs, OVERVIEW_ROOF), (roads, ASPHALT)):
        # Smoothed over its neighbours, so a suburb reads as an even tone rather than speckle.
        pad = np.pad(grid, 1, mode="edge")
        smooth = sum(pad[1 + dy:rows + 1 + dy, 1 + dx:cols + 1 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9
        share = np.minimum(1.0, smooth / (cell * cell))[..., None]
        colours = colours * (1 - share) + np.array(rgb, np.float32) * share
    return Overview(float(west), float(north), cell, heights.astype(np.float32), np.clip(colours, 0, 255).astype(np.uint8))


def write_overview(path: Path, o: Overview) -> None:
    rows, cols = o.heights.shape
    pad = b"\0" * (-o.colours.size % 4)
    with open(path, "wb") as f:
        f.write(OVERVIEW_HEADER.pack(OVERVIEW_MAGIC, 1, cols, rows, o.cell, o.east, o.north))
        f.write(o.heights.tobytes())
        f.write(o.colours.tobytes() + pad)


@dataclass
class Tile:
    ix: int
    iy: int
    meshes: dict = field(default_factory=lambda: {k: Mesh() for k in KINDS})
    footprints: list = field(default_factory=list)   # (N,2) UTM
    trees: list = field(default_factory=list)


def _inside(east: float, north: float, area: Area, keys: set) -> tuple[int, int] | None:
    key = tile_of(east, north, area.tile_size)
    return key if key in keys else None


_buildings_shared: dict = {}


def _buildings_of(key) -> tuple:
    """One tile's buildings, planned and drawn: (key, mesh, counts)."""
    import collections

    s = _buildings_shared
    context, mesh, counts = s["context"], Mesh(), collections.Counter()
    for f, ring, holes, tags in s["groups"][key]:
        plan = plan_building(ring, holes, tags, f.osm_id, s["lidar"], s["dem"].height, s["origin"], s["datum"])
        look = colours(tags, f.osm_id, context.kind(f, tags, ring), context.roof(f), context.shopfront(f, tags, ring), plan)
        draw(mesh, plan, look, context.party(f), context.street_side)
        counts[f"height from {plan.source}"] += 1
        counts[f"roof {plan.style}"] += 1
        counts["towers"] += len(plan.towers)
        counts["stepped"] += bool(plan.steps)
    return key, mesh, counts


def build_tiles(area: Area, osm: OsmData, dem: Dem, lidar=None, roof_colours=None, stats=None, overrides=None) -> tuple[dict, float]:
    """Assigns each feature to the tile holding its centre. Geometry may cross a tile edge, which hides seams."""
    stats = {} if stats is None else stats
    keys = set(area.tiles())
    tiles = {k: Tile(*k) for k in keys}
    west, south, east, north = area.bounds()
    datum = float(round(float(dem.height((west + east) / 2, (south + north) / 2))))
    ox, on = west, south   # meshes stay in small numbers while they are built

    context = BuildingContext(osm, (ox, on), lidar, roof_colours, overrides)
    groups: dict = {}
    for item in context.drawable():
        ce, cn = item[1].mean(axis=0)
        key = _inside(ce, cn, area, keys)
        if key is not None:
            groups.setdefault(key, []).append(item)
    _buildings_shared.update(groups=groups, context=context, lidar=lidar, dem=dem, origin=np.array([ox, on]), datum=datum)
    # Measuring a roof against the lidar is the slow part of a build, so tiles' buildings are planned in forked
    # workers that share everything parsed so far. A tile's buildings go in the same order either way.
    order = sorted(groups)
    workers = min(os.cpu_count() or 1, 16) if len(order) > 32 and hasattr(os, "fork") else 1
    if workers > 1:
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            done = list(pool.imap_unordered(_buildings_of, order, chunksize=2))
    else:
        done = [_buildings_of(k) for k in order]
    for key, mesh, counts in done:
        tiles[key].meshes["buildings"] = mesh
        for name, n in counts.items():
            context.stats[name] += n
    for f in context.footprints():
        ce, cn = f.rings[0].mean(axis=0)
        key = _inside(ce, cn, area, keys)
        if key is not None:
            tiles[key].footprints.extend(clean_ring(r) for r in f.rings)
    stats.update(context.stats)

    for f in osm.lines:
        hw = f.tags.get("highway")
        if hw is None or f.tags.get("tunnel") in ("yes", "culvert") or f.tags.get("area") == "yes":
            continue
        is_path = hw in PATH_WIDTHS
        width = parse_length(f.tags.get("width")) or PATH_WIDTHS.get(hw) or ROAD_WIDTHS.get(hw)
        if width is None:
            continue
        kind, rgb, lift = ("paths", CONCRETE, 0.20) if is_path else ("roads", ASPHALT, 0.12)
        for quad in ribbon_quads(f.line, width):
            ce, cn = np.mean(quad, axis=0)
            key = _inside(float(ce), float(cn), area, keys)
            if key is None:
                continue
            pts = np.array(quad)
            ys = dem.height(pts[:, 0], pts[:, 1]) - datum + lift
            corner = [(p[0] - ox, y, -(p[1] - on)) for p, y in zip(pts, ys)]
            a, b, d = (np.array(corner[i]) for i in (0, 1, 3))
            order = corner if np.cross(b - a, d - a)[1] > 0 else corner[::-1]
            tiles[key].meshes[kind].quad(*order, rgb)

    palette = [(84, 118, 62), (98, 132, 70), (74, 108, 58), (108, 136, 74), (88, 114, 70)]
    mapped = []
    for f in osm.points:
        if f.tags.get("natural") != "tree":
            continue
        key = _inside(*f.point, area, keys)
        if key is None:
            continue
        height = parse_length(f.tags.get("height")) or (mapped_height(lidar, *f.point) if lidar else None) or 9.0
        tiles[key].trees.append((f.point[0], f.point[1], height, pick(palette, f.osm_id)))
        mapped.append(f.point)
    if lidar is not None:
        near_mapped = Grid()
        for e, n in mapped:
            near_mapped.add((e - 4, n - 4, e + 4, n + 4), (e, n))
        roads = Grid()   # a crown over a road stands at the kerb: its top is no place for a trunk
        for f in osm.lines:
            width = ROAD_WIDTHS.get(f.tags.get("highway"))
            if width and f.tags.get("tunnel") not in ("yes", "culvert"):
                width = parse_length(f.tags.get("width")) or width
                for a, b in zip(f.line, f.line[1:]):
                    roads.add((min(a[0], b[0]) - width, min(a[1], b[1]) - width, max(a[0], b[0]) + width, max(a[1], b[1]) + width),
                              (a, b, width / 2 + 0.8))

        def on_road(e, n) -> bool:
            p = np.array([e, n])
            for a, b, half in roads.near(e, n):
                d = b - a
                u = np.clip(np.dot(p - a, d) / max(np.dot(d, d), 1e-9), 0, 1)
                if np.hypot(*(p - (a + d * u))) < half:
                    return True
            return False

        found = 0
        for e, n, height, crown in treetops(lidar, context.built):
            if on_road(e, n) or any(np.hypot(e - me, n - mn) < 4 for me, mn in near_mapped.near(e, n)):
                continue
            key = _inside(e, n, area, keys)
            if key is not None:
                tiles[key].trees.append((e, n, height, pick(palette, f"{e:.0f},{n:.0f}"), crown))
                found += 1
        stats["trees from lidar"] = found
    return tiles, datum


def terrain_grids(area: Area, tile: Tile, osm_painted: list, dem: Dem, datum: float):
    size, cell = area.tile_size, area.cell
    samples = int(round(size / cell)) + 1
    tw, tn = tile_north_west(tile.ix, tile.iy, size)
    ticks = np.arange(-1, samples + 1) * cell
    gx, gz = np.meshgrid(tw + ticks, tn - ticks)
    heights = dem.height(gx, gz) - datum
    east, north = gx[1:-1, 1:-1], gz[1:-1, 1:-1]
    colours = np.empty((samples, samples, 3), dtype=np.uint8)
    colours[:] = GRASS
    for box, ring, rgb in osm_painted:
        if box[2] < east.min() or box[0] > east.max() or box[3] < north.min() or box[1] > north.max():
            continue
        mask = points_in_ring(east, north, ring)
        colours[mask] = rgb
    return samples, heights.astype(np.float32), colours


def painted_areas(osm: OsmData) -> list:
    out = []
    for f in osm.areas:
        if "building" in f.tags:
            continue
        rgb = next((c for test, c in AREA_COLOURS if test(f.tags)), None)
        if rgb is None:
            continue
        for ring in f.rings:
            out.append((abs(signed_area(ring)), ring, rgb))
    out.sort(key=lambda t: -t[0])
    return [((r[:, 0].min(), r[:, 1].min(), r[:, 0].max(), r[:, 1].max()), r, c) for _, r, c in out]


def write_tile(path: Path, area: Area, tile: Tile, painted: list, dem: Dem, datum: float) -> None:
    samples, heights, colours = terrain_grids(area, tile, painted, dem, datum)
    size = area.tile_size
    tw, tn = tile_north_west(tile.ix, tile.iy, size)
    west, south, _, _ = area.bounds()
    meshes = [(k, m) for k, m in tile.meshes.items() if m.positions]
    pad = lambda n: b"\0" * (-n % 4)
    with open(path, "wb") as f:
        f.write(HEADER.pack(MAGIC, VERSION, tile.ix, tile.iy, size, area.cell, samples, len(meshes),
                            len(tile.footprints), len(tile.trees)))
        f.write(heights.tobytes())
        f.write(colours.tobytes() + pad(colours.size))
        for kind, m in meshes:
            pos = np.array(m.positions, np.float32)
            pos[:, 0] += float(west - tw)
            pos[:, 2] += float(tn - south)
            normal = np.zeros((len(pos), 4), np.int8)
            normal[:, :3] = np.round(np.array(m.normals) * 127).astype(np.int8)
            colour = np.zeros((len(pos), 4), np.uint8)
            colour[:, :3] = np.array(m.colors, np.uint8)
            index = np.array(m.indices, np.uint16 if len(pos) < 65536 else np.uint32)
            f.write(struct.pack("<Iii", KINDS[kind], len(pos), len(index)))
            f.write(pos.tobytes() + normal.tobytes() + colour.tobytes() + index.tobytes() + pad(index.nbytes))
        for ring in tile.footprints:
            f.write(struct.pack("<i", len(ring)))
            f.write(np.column_stack((ring[:, 0] - tw, tn - ring[:, 1])).astype(np.float32).tobytes())
        for east, north, height, rgb, *crown in tile.trees:
            y = float(dem.height(east, north)) - datum
            radius = crown[0] if crown else height * 0.32
            f.write(TREE.pack(east - tw, y, tn - north, height, radius, *[(c / 255) ** 2.2 for c in rgb]))


def write_manifest(path: Path, area: Area, datum: float, sizes: dict, streets: list = (), overview: str = "") -> None:
    e, n = area.centre_utm()
    lines = ["streetkit-manifest 1", f"name {area.name}", "epsg 32613", f"tile_size {area.tile_size:g}", f"cell {area.cell:g}",
             f"datum {datum:g}", f"latitude {area.centre_lat}", f"start {e:.1f} {n:.1f}", f"tiles {len(sizes)}"]
    if overview:
        lines.insert(-1, f"overview {overview}")
    lines += [f"tile {ix} {iy} {size} {digest}" for (ix, iy), (size, digest) in sorted(sizes.items())]
    lines += streets
    path.write_text("\n".join(lines) + "\n")


def read_manifest(path: Path) -> dict:
    out, tiles = {}, {}
    for line in Path(path).read_text().splitlines():
        word, _, rest = line.partition(" ")
        if word == "tile":
            ix, iy, size, digest = rest.split()
            tiles[(int(ix), int(iy))] = (int(size), digest)
        elif word not in ("streetkit-manifest", "old-town-walk-manifest", "ponderosa-manifest", "tiles"):
            out[word] = rest
    out["tiles"] = tiles
    return out


def tile_name(ix: int, iy: int) -> str:
    return f"{ix}_{iy}.tile"


_shared = {}


def _write_one(key) -> tuple:
    """One tile, under its name only once it is whole, so a tile file that exists is a finished tile."""
    s = _shared
    path = s["out"] / tile_name(*key)
    if not (s["resume"] and path.exists()):
        part = path.with_suffix(f".part{os.getpid()}")
        write_tile(part, s["area"], s["tiles"][key], s["painted"], s["dem"], s["datum"])
        part.replace(path)
    return key, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _say(start: float, text: str) -> None:
    print(f"[{time.time() - start:6.0f} s] {text}", flush=True)


def build_all(area: Area, osm: OsmData, dem: Dem, out: Path, streets: list = (), lidar=None, roof_colours=None,
              overrides=None, sample: int = 1, resume: bool = False, quiet: bool = True) -> dict:
    """Every tile, or every `sample`th to measure a large area. `resume` keeps the tile files already written. An
    area with an overview cell also gets overview.bin, the whole area at low resolution."""
    start = time.time()
    say = (lambda text: None) if quiet else (lambda text: _say(start, text))
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    tiles, datum = build_tiles(area, osm, dem, lidar, roof_colours, stats, overrides)
    say(f"planned {sum(len(t.footprints) for t in tiles.values())} buildings in {len(tiles)} tiles")
    painted = painted_areas(osm)
    _shared.update(out=out, area=area, tiles=tiles, painted=painted, dem=dem, datum=datum, resume=resume)
    keys = sorted(tiles)[::sample]
    # Forked workers share the parsed data; there is no fork on Windows, and a small area does not need it.
    workers = min(os.cpu_count() or 1, 16) if len(keys) > 32 and hasattr(os, "fork") else 1
    if workers > 1:
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            written = []
            for result in pool.imap_unordered(_write_one, keys, chunksize=4):
                written.append(result)
                if len(written) % 50 == 0:
                    say(f"{len(written)} of {len(keys)} tiles")
    else:
        written = [_write_one(k) for k in keys]
    sizes = {k: (size, digest) for k, size, digest in written}
    name = ""
    if area.overview_cell > 0:
        name = "overview.bin"
        write_overview(out / name, build_overview(area, osm, painted, dem, datum))
        say("overview written")
    write_manifest(out / "manifest.txt", area, datum, sizes, streets, name)
    return {"tiles": len(tiles), "written": len(sizes), "buildings": sum(len(t.footprints) for t in tiles.values()),
            "trees": sum(len(t.trees) for t in tiles.values()), "bytes": sum(size for size, _ in sizes.values()),
            "overview_bytes": (out / name).stat().st_size if name else 0, **stats}
