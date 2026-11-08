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

Indices are u16 when the mesh has fewer than 65536 vertices.
"""

from __future__ import annotations

import hashlib
import struct
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
    """A north-up float32 GeoTIFF in the grid's UTM zone, sampled bilinearly at pixel centres."""

    def __init__(self, path: Path):
        import tifffile
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            self.z = page.asarray().astype(np.float64)
            scale, tie = page.tags["ModelPixelScaleTag"].value, page.tags["ModelTiepointTag"].value
        bad = self.z < -1000
        if bad.any():
            self.z[bad] = np.median(self.z[~bad])
        self.sx, self.sy, self.e0, self.n0 = float(scale[0]), float(scale[1]), float(tie[3]), float(tie[4])

    def height(self, east, north):
        col = np.clip((np.asarray(east, np.float64) - self.e0) / self.sx - 0.5, 0, self.z.shape[1] - 1.000001)
        row = np.clip((self.n0 - np.asarray(north, np.float64)) / self.sy - 0.5, 0, self.z.shape[0] - 1.000001)
        c, r = np.floor(col).astype(int), np.floor(row).astype(int)
        fc, fr = col - c, row - r
        top = self.z[r, c] * (1 - fc) + self.z[r, c + 1] * fc
        bottom = self.z[r + 1, c] * (1 - fc) + self.z[r + 1, c + 1] * fc
        return top * (1 - fr) + bottom * fr


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


def build_tiles(area: Area, osm: OsmData, dem: Dem, lidar=None, roof_colours=None, stats=None, overrides=None) -> tuple[dict, float]:
    """Assigns each feature to the tile holding its centre. Geometry may cross a tile edge, which hides seams."""
    stats = {} if stats is None else stats
    keys = set(area.tiles())
    tiles = {k: Tile(*k) for k in keys}
    west, south, east, north = area.bounds()
    datum = float(round(float(dem.height((west + east) / 2, (south + north) / 2))))
    ox, on = west, south   # meshes stay in small numbers while they are built

    context = BuildingContext(osm, (ox, on), lidar, roof_colours, overrides)
    for f, ring, holes, tags in context.drawable():
        ce, cn = ring.mean(axis=0)
        key = _inside(ce, cn, area, keys)
        if key is None:
            continue
        plan = plan_building(ring, holes, tags, f.osm_id, lidar, dem.height, np.array([ox, on]), datum)
        look = colours(tags, f.osm_id, context.kind(f, tags, ring), context.roof(f), context.shopfront(f, tags, ring), plan)
        draw(tiles[key].meshes["buildings"], plan, look, context.party(f), context.street_side)
        context.count(plan)
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


def write_manifest(path: Path, area: Area, datum: float, sizes: dict, streets: list = ()) -> None:
    e, n = area.centre_utm()
    lines = ["streetkit-manifest 1", f"name {area.name}", "epsg 32613", f"tile_size {area.tile_size:g}", f"cell {area.cell:g}",
             f"datum {datum:g}", f"latitude {area.centre_lat}", f"start {e:.1f} {n:.1f}", f"tiles {len(sizes)}"]
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
        elif word not in ("streetkit-manifest", "old-town-walk-manifest", "tiles"):
            out[word] = rest
    out["tiles"] = tiles
    return out


def tile_name(ix: int, iy: int) -> str:
    return f"{ix}_{iy}.tile"


def build_all(area: Area, osm: OsmData, dem: Dem, out: Path, streets: list = (), lidar=None, roof_colours=None, overrides=None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    tiles, datum = build_tiles(area, osm, dem, lidar, roof_colours, stats, overrides)
    painted = painted_areas(osm)
    sizes = {}
    for key, tile in sorted(tiles.items()):
        path = out / tile_name(*key)
        write_tile(path, area, tile, painted, dem, datum)
        sizes[key] = (path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest()[:16])
    write_manifest(out / "manifest.txt", area, datum, sizes, streets)
    return {"tiles": len(tiles), "buildings": sum(len(t.footprints) for t in tiles.values()),
            "trees": sum(len(t.trees) for t in tiles.values()), "bytes": sum(s for s, _ in sizes.values()), **stats}
