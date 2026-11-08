"""Buildings: how tall, what roof, what colour, and what their walls look like.

Heights and roofs come from airborne lidar where the survey saw the building (lidar.py). Each footprint is read for
its eave height, its highest point, and the shape of its roof: a flat roof, or a pitched one fitted as gables or
hips over the rectangles that make up the footprint. Pitched roofs over rectangles that overlap meet the way cross
gables do. Flat roofs at more than one height become stepped blocks, and a small cluster standing well above the
roof becomes a tower with a spire. A footprint the survey did not see (built after 2013) falls back on its OSM tags,
and failing those on its kind and size.

Colours come from OSM tags when mappers gave them, from the roof colours sampled from aerial imagery (roofs.py),
and otherwise from palettes chosen by the kind of building. Walls get rows of windows by storey, and shops on a
commercial street get a glass front.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field

import mapbox_earcut
import numpy as np

from .mesh import Mesh, clean_ring, points_in_ring, signed_area

STOREY = 3.2                 # metres a floor, for levels tags and for windows
SHOP_STOREY = 4.4            # a shop's ground floor
MIN_FLAT_RISE = 0.9          # a roof that rises less than this from eave to ridge is drawn flat
MAX_PITCH = 1.1              # rise over half-width, about 48 degrees
LEAFY = 0.35                 # a lidar cell with more split returns than this is a tree over the roof
EDGE = 1.6                   # metres in from the wall where the eave is read
GLASS = (78, 92, 106)
FRAME_LIGHT = (214, 208, 196)

HOUSE_KINDS = {"house", "detached", "residential", "semidetached_house", "bungalow", "terrace", "duplex", "cabin"}
SHED_KINDS = {"garage", "garages", "shed", "carport", "roof", "hut", "greenhouse"}
SHOP_KINDS = {"retail", "commercial", "office", "hotel", "supermarket", "kiosk"}
CIVIC_KINDS = {"church", "chapel", "cathedral", "religious", "civic", "government", "public", "school", "university",
               "college", "library", "museum", "train_station", "transportation", "courthouse", "fire_station"}

BRICK = [(150, 84, 64), (166, 98, 72), (138, 76, 60), (176, 112, 82), (128, 70, 56), (184, 140, 104), (196, 168, 128)]
SIDING = [(226, 220, 204), (204, 194, 172), (172, 178, 164), (188, 162, 128), (152, 92, 72), (234, 230, 220),
          (142, 152, 160), (206, 186, 150), (118, 128, 112)]
MODERN = [(204, 200, 192), (176, 166, 154), (150, 146, 140), (190, 178, 160), (120, 116, 112), (214, 210, 200)]
SHEDS = [(170, 160, 146), (140, 128, 116), (200, 196, 186), (120, 104, 92)]
STONE = [(196, 184, 160), (170, 158, 138), (212, 200, 176)]
SHINGLE = [(92, 88, 86), (112, 100, 92), (78, 80, 86), (120, 110, 100), (96, 72, 64)]
FLAT_ROOFS = [(150, 148, 144), (128, 126, 122), (170, 168, 162), (110, 110, 112)]

NAMED_COLOURS = {
    "white": (236, 234, 228), "black": (40, 40, 42), "grey": (128, 128, 128), "gray": (128, 128, 128),
    "lightgrey": (190, 190, 190), "lightgray": (190, 190, 190), "darkgrey": (80, 80, 82), "darkgray": (80, 80, 82),
    "silver": (184, 186, 188), "red": (150, 60, 50), "darkred": (110, 40, 36), "maroon": (110, 40, 36),
    "brown": (120, 80, 56), "tan": (196, 170, 130), "beige": (214, 200, 168), "cream": (232, 222, 192),
    "yellow": (214, 190, 110), "orange": (196, 120, 60), "green": (80, 112, 72), "darkgreen": (56, 80, 52),
    "blue": (72, 96, 140), "darkblue": (48, 60, 96), "lightblue": (150, 176, 196), "navy": (40, 50, 80),
    "brick": (156, 86, 64), "sandstone": (200, 176, 136), "terracotta": (178, 96, 66), "copper": (96, 140, 120),
}


def pick(options, key: str):
    return options[zlib.crc32(key.encode()) % len(options)]


def parse_length(value: str | None) -> float | None:
    if not value:
        return None
    text = value.strip().lower().replace(",", ".")
    try:
        if text.endswith(("'", "ft")):
            return float(text.rstrip("'ft ")) * 0.3048
        return float(text.replace("m", "").strip())
    except ValueError:
        return None


def parse_colour(value: str | None):
    if not value:
        return None
    text = value.strip().lower().replace(" ", "").replace("_", "")
    if text.startswith("#") and len(text) in (4, 7):
        h = text[1:] if len(text) == 7 else "".join(c * 2 for c in text[1:])
        try:
            return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            return None
    return NAMED_COLOURS.get(text)


def kind_of(tags: dict, area: float) -> str:
    """house, shed, shop, civic or block (anything else large)."""
    k = tags.get("building", tags.get("building:part", "yes"))
    if k in HOUSE_KINDS:
        return "house"
    if k in SHED_KINDS:
        return "shed"
    if k in SHOP_KINDS:
        return "shop"
    if k in CIVIC_KINDS:
        return "civic"
    if k in ("apartments", "dormitory", "industrial", "warehouse", "parking", "hospital"):
        return "block"
    if area < 45:
        return "shed"
    return "house" if area < 320 else "block"


# ------------------------------------------------------------------------------------------------ geometry helpers

def rotate(points: np.ndarray, angle: float) -> np.ndarray:
    """Turns points by -angle, so a wall running at `angle` runs along x."""
    c, s = math.cos(angle), math.sin(angle)
    return points @ np.array([[c, -s], [s, c]])


def unrotate(points: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return points @ np.array([[c, s], [-s, c]])


def main_angle(ring: np.ndarray) -> float:
    """The direction the walls mostly run in, modulo a right angle, weighted by wall length."""
    d = np.roll(ring, -1, axis=0) - ring
    length = np.hypot(d[:, 0], d[:, 1])
    a = np.arctan2(d[:, 1], d[:, 0]) * 4   # fold the four right angles onto one
    return float(math.atan2((length * np.sin(a)).sum(), (length * np.cos(a)).sum()) / 4)


def squareness(ring: np.ndarray, angle: float) -> float:
    """Share of the walls' length that runs along or across the main direction, within 10 degrees."""
    d = rotate(np.roll(ring, -1, axis=0) - ring, angle)
    length = np.hypot(d[:, 0], d[:, 1])
    off = np.minimum(np.abs(d[:, 0]), np.abs(d[:, 1])) / np.maximum(length, 1e-9)
    return float(length[off < 0.18].sum() / max(length.sum(), 1e-9))


def rect_cover(mask: np.ndarray, xs: np.ndarray, ys: np.ndarray, most: int = 6, smallest: float = 6.0) -> list:
    """Rectangles (x0, y0, x1, y1) that together cover a grid of cells, each as large as it can be. Cells are
    the spans between consecutive xs and ys; the rectangles may overlap, as the wings of a house do."""
    left = mask.copy()
    found = []
    widths, heights = np.diff(xs), np.diff(ys)
    while left.any() and len(found) < most:
        best, best_area = None, 0.0
        for j, i in zip(*np.nonzero(left)):
            for order in (0, 1):
                j0 = j1 = j
                i0 = i1 = i
                for step in range(2):
                    if (step == 0) == (order == 0):   # grow along x
                        while i0 > 0 and mask[j0:j1 + 1, i0 - 1].all():
                            i0 -= 1
                        while i1 + 1 < mask.shape[1] and mask[j0:j1 + 1, i1 + 1].all():
                            i1 += 1
                    else:
                        while j0 > 0 and mask[j0 - 1, i0:i1 + 1].all():
                            j0 -= 1
                        while j1 + 1 < mask.shape[0] and mask[j1 + 1, i0:i1 + 1].all():
                            j1 += 1
                fresh = (left[j0:j1 + 1, i0:i1 + 1] * np.outer(heights[j0:j1 + 1], widths[i0:i1 + 1])).sum()
                if fresh > best_area:
                    best, best_area = (j0, j1, i0, i1), fresh
        if best is None:
            break
        j0, j1, i0, i1 = best
        left[j0:j1 + 1, i0:i1 + 1] = False
        if best_area >= smallest or not found:
            found.append((float(xs[i0]), float(ys[j0]), float(xs[i1 + 1]), float(ys[j1 + 1])))
    return found


def snap(values: np.ndarray, gap: float = 0.6) -> np.ndarray:
    v = np.sort(values)
    out = [v[0]]
    for x in v[1:]:
        if x - out[-1] > gap:
            out.append(x)
    return np.array(out)


def footprint_rects(ring: np.ndarray, angle: float) -> list:
    """The footprint, turned square to its walls, as covering rectangles in that turned frame."""
    q = rotate(ring, angle)
    xs, ys = snap(q[:, 0]), snap(q[:, 1])
    if len(xs) < 2 or len(ys) < 2 or len(xs) * len(ys) > 900:
        return []
    cx, cy = (xs[:-1] + xs[1:]) / 2, (ys[:-1] + ys[1:]) / 2
    gx, gy = np.meshgrid(cx, cy)
    mask = points_in_ring(gx.ravel(), gy.ravel(), q).reshape(gx.shape)
    return rect_cover(mask, xs, ys)


def inset_ring(ring: np.ndarray, d: float) -> np.ndarray | None:
    """The ring moved `d` inwards along its corners' bisectors, or None when that folds it over."""
    n = len(ring)
    out = []
    for i in range(n):
        a, b, c = ring[i - 1], ring[i], ring[(i + 1) % n]
        e0, e1 = b - a, c - b
        e0 /= max(np.linalg.norm(e0), 1e-9)
        e1 /= max(np.linalg.norm(e1), 1e-9)
        n0, n1 = np.array([-e0[1], e0[0]]), np.array([-e1[1], e1[0]])   # inward for a counter-clockwise ring
        m = n0 + n1
        cos_half = max(float(np.linalg.norm(m)) / 2, 0.25)
        m = m / max(np.linalg.norm(m), 1e-9)
        out.append(b + m * d / cos_half)
    out = np.array(out)
    if signed_area(out) <= 0.2 * signed_area(ring):
        return None
    for i in range(n):   # the moved walls must keep their directions
        if np.dot(out[(i + 1) % n] - out[i], ring[(i + 1) % n] - ring[i]) <= 0:
            return None
    return out


def triangulate(ring: np.ndarray, holes=()) -> list:
    pts = np.concatenate([ring] + list(holes)) if holes else ring
    ends = np.cumsum([len(ring)] + [len(h) for h in holes]).astype(np.uint32)
    flat = mapbox_earcut.triangulate_float64(np.ascontiguousarray(pts, dtype=np.float64), ends)
    return pts, [tuple(int(v) for v in flat[i:i + 3]) for i in range(0, len(flat), 3)]


class Builder:
    """Writes faces into a mesh with their fronts the right way out, in the mesh's (x east, y up, z south)."""

    def __init__(self, mesh: Mesh):
        self.mesh = mesh

    @staticmethod
    def p(xy, y):
        return (float(xy[0]), float(y), -float(xy[1]))

    def face(self, pts, tris, rgb, outward) -> None:
        pts = [tuple(v) for v in pts]
        if not tris:
            return
        a, b, c = (np.array(pts[i]) for i in tris[0])
        if np.dot(np.cross(b - a, c - a), outward) < 0:
            tris = [(t[0], t[2], t[1]) for t in tris]
        self.mesh.face(pts, tris, rgb)

    def wall(self, a, b, y0, y1, rgb, push: float = 0.0) -> None:
        """A wall from a to b, ring order (counter-clockwise), facing out; `push` moves it out by that much."""
        out = np.array([b[1] - a[1], -(b[0] - a[0])])
        length = float(np.linalg.norm(out))
        if length < 1e-6 or y1 - y0 < 1e-3:
            return
        out /= length
        a2, b2 = np.asarray(a) + out * push, np.asarray(b) + out * push
        self.face([self.p(a2, y0), self.p(b2, y0), self.p(b2, y1), self.p(a2, y1)], [(0, 1, 2), (0, 2, 3)], rgb,
                  (out[0], 0, -out[1]))

    def walls(self, ring, y0, y1, rgb, inward: bool = False) -> None:
        n = len(ring)
        for i in range(n):
            a, b = ring[i], ring[(i + 1) % n]
            if inward:
                a, b = b, a
            self.wall(a, b, y0, y1, rgb)

    def flat(self, ring, y, rgb, holes=(), down: bool = False) -> None:
        pts, tris = triangulate(ring, list(holes))
        self.face([self.p(v, y) for v in pts], tris, rgb, (0, -1 if down else 1, 0))


# ------------------------------------------------------------------------------------------------ measuring

@dataclass
class Reading:
    """What the lidar says about one footprint. Heights are metres above the ground."""
    cells: int            # cells inside the footprint
    seen: int             # of those, cells on a roof and not under a tree
    east: np.ndarray
    north: np.ndarray
    height: np.ndarray    # of the seen cells
    eave: float
    top: float

    @property
    def rise(self) -> float:
        return self.top - self.eave


def read(lidar, ring: np.ndarray) -> Reading | None:
    """None when the survey did not see a building here: too few cells, or the ground bare (built after 2013)."""
    if lidar is None:
        return None
    from .lidar import edge_distance

    e, n, h, leafy = lidar.cells_in(ring)
    if len(h) < 4:
        return None
    clear = leafy < LEAFY
    if clear.sum() < max(3, 0.3 * len(h)):
        return None
    e, n, h = e[clear], n[clear], h[clear]
    if np.median(h) < 1.8:
        return None
    standing = h > 1.2
    e, n, h = e[standing], n[standing], h[standing]
    edge = edge_distance(e, n, ring) < EDGE
    eave = float(np.median(h[edge])) if edge.sum() >= 3 else float(np.percentile(h, 15))
    top = float(np.percentile(h, 97))
    return Reading(len(standing), len(h), e, n, h, min(eave, top), top)


# ------------------------------------------------------------------------------------------------ roofs

def roof_shape(rects, angle, origin, pitch, style, px, py):
    """Height above the eave of a roof made of `rects` (turned frame, relative to `origin`) at points, or NaN outside
    every rectangle. Overlapping rectangles show the highest roof, as they would to the lidar."""
    q = rotate(np.column_stack([px, py]) - origin, angle)
    best = np.full(len(px), np.nan)
    for x0, y0, x1, y1 in rects:
        w, d = x1 - x0, y1 - y0
        inside = (q[:, 0] >= x0 - 0.5) & (q[:, 0] <= x1 + 0.5) & (q[:, 1] >= y0 - 0.5) & (q[:, 1] <= y1 + 0.5)
        if not inside.any():
            continue
        u, v = np.clip(q[inside, 0] - x0, 0, w), np.clip(q[inside, 1] - y0, 0, d)
        across, along = (np.minimum(v, d - v), np.minimum(u, w - u)) if w >= d else (np.minimum(u, w - u), np.minimum(v, d - v))
        h = pitch * (across if style == "gable" else np.minimum(across, along))
        cur = best[inside]
        best[inside] = np.where(np.isnan(cur), h, np.maximum(cur, h))
    return best


def misfit(residual: np.ndarray) -> float:
    """Mean absolute error with each point's share capped, so a chimney or a branch cannot outvote a roof."""
    return float(np.mean(np.minimum(np.abs(residual), 1.5)))


def fit_roof(reading: Reading, rects, angle, origin):
    """Picks flat, gable or hip, a pitch and an eave height, by which matches the lidar best.
    Returns (style, pitch, eave, misfit)."""
    level = float(np.median(reading.height))
    flat = ("flat", 0.0, level, misfit(reading.height - level))
    if reading.top - float(np.percentile(reading.height, 10)) < MIN_FLAT_RISE or not rects:
        return flat
    widest = max(min(x1 - x0, y1 - y0) for x0, y0, x1, y1 in rects)
    best = flat
    for style in ("gable", "hip"):
        for pitch in np.linspace(0.2, MAX_PITCH, 14):
            shape = roof_shape(rects, angle, origin, pitch, style, reading.east, reading.north)
            ok = ~np.isnan(shape)
            if ok.sum() < 0.7 * len(shape):
                continue
            eave = float(np.median(reading.height[ok] - shape[ok]))
            if eave < 2.0:
                continue
            err = misfit(reading.height[ok] - shape[ok] - eave)
            if err < best[3]:
                best = (style, float(pitch), eave, err)
    if best[0] != "flat" and (best[3] > 0.85 * flat[3] or best[1] * widest / 2 < MIN_FLAT_RISE):
        return flat   # pitched only when it explains the roof clearly better than a level one
    return best


UNTRUSTED = 0.95   # metres of misfit beyond which no roof explains the lidar: a building put up after the survey,
                   # under trees the survey saw instead, or one that has since changed


def towers(reading: Reading, roof_top: float):
    """Small clusters standing well above the roof: steeples, clock towers, a hotel's lift housing."""
    tall = reading.height > roof_top + 3.0
    if tall.sum() < 4:
        return []
    e, n, h = reading.east[tall], reading.north[tall], reading.height[tall]
    out, used = [], np.zeros(len(h), bool)
    for i in np.argsort(-h):
        if used[i]:
            continue
        near = np.hypot(e - e[i], n - n[i]) < 6
        used |= near
        if near.sum() > 60 or near.sum() < 4:
            continue
        half = max(1.5, math.sqrt(near.sum()) / 2 + 0.5)
        out.append((float(e[near].mean()), float(n[near].mean()), half, float(np.percentile(h[near], 50)), float(h[near].max())))
        if len(out) == 3:
            break
    return out


# ------------------------------------------------------------------------------------------------ the building

@dataclass
class Look:
    wall: tuple
    roof: tuple
    trim: tuple
    kind: str
    shopfront: bool = False


@dataclass
class Plan:
    """Everything decided about one footprint before it is drawn. Heights are above `ground` (metres, mesh y)."""
    ring: np.ndarray           # local x east, y north, counter-clockwise
    holes: list
    base: float                # mesh y of the wall bottoms
    ground: float              # mesh y the heights stand on
    eave: float                # above ground
    style: str                 # flat, gable, hip, pyramid, skeleton
    pitch: float = 0.0
    angle: float = 0.0
    rects: list = field(default_factory=list)
    steps: list = field(default_factory=list)    # (local ring, height above ground) of higher flat blocks
    towers: list = field(default_factory=list)   # (x, y, half, body height, spire height) local
    source: str = "guess"      # lidar, tags or guess
    error: float = 0.0


def guess_shape(tags: dict, kind: str, area: float) -> tuple[float, str, float]:
    """Eave, roof style and rise without lidar: from tags, then from the kind of building."""
    height, levels = parse_length(tags.get("height")), parse_length(tags.get("building:levels"))
    roof_h = parse_length(tags.get("roof:height")) or (parse_length(tags.get("roof:levels")) or 0) * 2.6
    shape = tags.get("roof:shape")
    style = {"gabled": "gable", "hipped": "hip", "pyramidal": "pyramid", "flat": "flat", "half-hipped": "hip",
             "skillion": "flat", "mansard": "hip", "gambrel": "gable"}.get(shape or "")
    if height:
        rise = roof_h if style and style != "flat" else 0.0
        return max(2.0, height - rise), style or "flat", rise
    if levels:
        eave = levels * STOREY + (0.0 if kind == "house" else 0.6)
        if style is None:
            style = "gable" if kind == "house" else "flat"
        return eave, style, roof_h or (2.4 if style != "flat" else 0.0)
    if kind == "shed":
        return 2.7, style or "flat", 0.0
    if kind == "house":
        return 3.6 if area < 140 else 4.4, style or "hip", roof_h or 2.4
    return (7.5 if kind != "civic" else 9.0), style or "flat", 0.0


def plan_building(ring_utm: np.ndarray, holes_utm: list, tags: dict, osm_id: str, lidar, ground_at, origin,
                  datum: float) -> Plan:
    """Decides a footprint's shape. `ground_at(e, n)` is the DEM; `origin` is the mesh's (east, north) zero."""
    ring = ring_utm - origin
    holes = [h - origin for h in holes_utm]
    area = abs(signed_area(ring))
    kind = kind_of(tags, area)
    ground = ground_at(ring_utm[:, 0], ring_utm[:, 1]) - datum
    plan = Plan(ring, holes, float(ground.min()) - 0.4, float(ground.mean()), 0.0, "flat")
    plan.angle = main_angle(ring)
    square = squareness(ring, plan.angle) > 0.88 and not holes
    rects = footprint_rects(ring, plan.angle) if square else []
    tagged_style = {"gabled": "gable", "hipped": "hip", "pyramidal": "pyramid", "flat": "flat"}.get(tags.get("roof:shape", ""))

    reading = read(lidar, ring_utm)
    if reading is not None:
        style, pitch, eave, err = fit_roof(reading, rects, plan.angle, origin)
        if err > UNTRUSTED:
            reading = None
    if reading is not None:
        plan.source = "lidar"
        if tagged_style == "flat":
            style, pitch, eave = "flat", 0.0, float(np.median(reading.height))
        plan.error = err
        if style == "flat":
            plan.eave, plan.style = eave, "flat"
            if area > 200 and square:
                plan.steps = stepped(reading, ring, plan.angle, eave, origin)
            roof_top = max([eave] + [h for _, h in plan.steps])
            if not square and reading.top - eave > 1.5 and kind in ("house", "shed"):
                plan.style, plan.pitch = "skeleton", min(MAX_PITCH, (reading.top - eave) / 3.0)
                plan.eave = float(np.percentile(reading.height, 20))
        else:
            plan.eave, plan.style, plan.pitch, plan.rects = eave, style, pitch, rects
            roof_top = eave + pitch * max(min(x1 - x0, y1 - y0) for x0, y0, x1, y1 in rects) / 2
        if kind == "civic" or tags.get("roof:shape") in ("pyramidal", "spire", "onion", "dome"):
            plan.towers = [(x - origin[0], y - origin[1], half, body, spire)
                           for x, y, half, body, spire in towers(reading, roof_top)]
        return plan

    eave, style, rise = guess_shape(tags, kind, area)
    plan.source = "tags" if any(k in tags for k in ("height", "building:levels")) else "guess"
    plan.eave = eave
    if style == "flat" or rise <= 0:
        plan.style = "flat"
    elif rects:
        widest = max(min(x1 - x0, y1 - y0) for x0, y0, x1, y1 in rects)
        plan.style, plan.pitch, plan.rects = ("hip" if style == "pyramid" else style), min(MAX_PITCH, rise / max(widest / 2, 1)), rects
    else:
        plan.style, plan.pitch = "skeleton", 0.7
    return plan


def stepped(reading: Reading, ring: np.ndarray, angle: float, level: float, origin) -> list:
    """Higher blocks on a flat roof, as (local ring, height) pairs: each run of roof well above `level`."""
    local = np.column_stack([reading.east - origin[0], reading.north - origin[1]])
    q = rotate(local, angle)
    high = reading.height > level + 2.5
    if high.sum() < 20:
        return []
    x0, y0 = q[:, 0].min(), q[:, 1].min()
    nx, ny = int(math.ceil(q[:, 0].max() - x0)) + 1, int(math.ceil(q[:, 1].max() - y0)) + 1
    grid = np.full((ny, nx), np.nan)
    gi, gj = (q[:, 0] - x0).astype(int), (q[:, 1] - y0).astype(int)
    grid[gj, gi] = reading.height
    mask = np.nan_to_num(grid, nan=0) > level + 2.5
    # Keep only cells with most neighbours also high, so a rooftop unit or a tree's edge does not become a block.
    pad = np.pad(mask, 1)
    count = sum(np.roll(np.roll(pad, dy, 0), dx, 1)[1:-1, 1:-1] for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    mask &= count >= 7
    if mask.sum() < 20:
        return []
    rects = rect_cover(mask, np.arange(nx + 1, dtype=float) + x0, np.arange(ny + 1, dtype=float) + y0, most=6, smallest=20)
    out = []
    for rx0, ry0, rx1, ry1 in rects:
        if (rx1 - rx0) < 3 or (ry1 - ry0) < 3:
            continue
        sel = (gi >= rx0 - x0) & (gi < rx1 - x0) & (gj >= ry0 - y0) & (gj < ry1 - y0) & high
        if sel.sum() < 6:
            continue
        h = float(np.percentile(reading.height[sel], 60))
        if float(np.mean(np.abs(reading.height[sel] - h))) > 1.2:
            continue   # a roof is level; a crown of leaves the survey saw as one return is not
        corners = unrotate(np.array([[rx0, ry0], [rx1, ry0], [rx1, ry1], [rx0, ry1]]), angle)
        out.append((corners, h))
    return out


# ------------------------------------------------------------------------------------------------ drawing

def draw(mesh: Mesh, plan: Plan, look: Look, party, street_side) -> int:
    """Adds the building to the mesh. `party(a, b)` says a wall is shared with a neighbour (no windows);
    `street_side(a, b)` says it faces a street (a shop window if the building has one). Returns triangles added."""
    before = len(mesh.indices)
    b = Builder(mesh)
    g = plan.ground
    eave_y = g + plan.eave
    b.walls(plan.ring, plan.base, eave_y, look.wall)
    for hole in plan.holes:
        b.walls(hole, plan.base, eave_y, look.wall, inward=True)
    if plan.style == "flat" or (plan.style in ("gable", "hip") and not plan.rects):
        b.flat(plan.ring, eave_y, look.roof, plan.holes)
        if look.kind in ("shop", "civic", "block") and abs(signed_area(plan.ring)) > 120 and plan.eave > 4:
            parapet(b, plan.ring, eave_y, look)
    elif plan.style == "skeleton":
        skeleton_roof(b, plan.ring, eave_y, plan.pitch, look)
    else:
        b.flat(plan.ring, eave_y, look.roof, plan.holes)   # a ceiling under the roof, for views from above the eave
        for rect in plan.rects:
            rect_roof(b, rect, plan.angle, np.zeros(2), eave_y, plan.pitch, plan.style, look)
    for corners, h in plan.steps:
        ring = clean_ring(corners)
        b.walls(ring, eave_y - 0.1, g + h, look.wall)
        b.flat(ring, g + h, look.roof)
        windows(b, ring, eave_y, g + h, look, party=lambda a, c: False, street_side=lambda a, c: False, start_floor=1)
    for x, y, half, body, spire in plan.towers:
        tower(b, x, y, half, eave_y, g + body, g + spire, look)
    windows(b, plan.ring, g, eave_y, look, party, street_side)
    return (len(mesh.indices) - before) // 3


def rect_roof(b: Builder, rect, angle, origin, eave_y, pitch, style, look: Look) -> None:
    x0, y0, x1, y1 = rect
    w, d = x1 - x0, y1 - y0
    along_x = w >= d
    half = (d if along_x else w) / 2
    rise = pitch * half
    top = eave_y + rise
    to_local = lambda pts: unrotate(np.array(pts, dtype=float), angle) + origin
    if along_x:
        ym = (y0 + y1) / 2
        inset = min(half, w / 2) if style == "hip" else 0.0
        ridge = [(x0 + inset, ym), (x1 - inset, ym)]
        c = to_local([(x0, y0), (x1, y0), (x1, y1), (x0, y1), ridge[0], ridge[1]])
        slopes = [([c[0], c[1], c[5], c[4]], (0, -1)), ([c[2], c[3], c[4], c[5]], (0, 1))]
        ends = [([c[3], c[0], c[4]], (-1, 0)), ([c[1], c[2], c[5]], (1, 0))]
    else:
        xm = (x0 + x1) / 2
        inset = min(half, d / 2) if style == "hip" else 0.0
        ridge = [(xm, y0 + inset), (xm, y1 - inset)]
        c = to_local([(x0, y0), (x1, y0), (x1, y1), (x0, y1), ridge[0], ridge[1]])
        slopes = [([c[1], c[2], c[5], c[4]], (1, 0)), ([c[3], c[0], c[4], c[5]], (-1, 0))]
        ends = [([c[0], c[1], c[4]], (0, -1)), ([c[2], c[3], c[5]], (0, 1))]
    ys = [eave_y, eave_y, top, top]
    for pts, (ox, oy) in slopes:
        out = unrotate(np.array([[ox, oy]], float), angle)[0]
        b.face([b.p(p, y) for p, y in zip(pts, ys)], [(0, 1, 2), (0, 2, 3)], look.roof, (out[0], 1.0, -out[1]))
    for pts, (ox, oy) in ends:
        out = unrotate(np.array([[ox, oy]], float), angle)[0]
        colour = look.roof if style == "hip" else look.wall
        tilt = 1.0 if style == "hip" else 0.0
        b.face([b.p(pts[0], eave_y), b.p(pts[1], eave_y), b.p(pts[2], top)], [(0, 1, 2)], colour, (out[0], tilt, -out[1]))


def skeleton_roof(b: Builder, ring, eave_y, pitch, look: Look) -> None:
    """A hip roof for any outline: slopes from every wall up to an inner ring, with a flat top inside it."""
    d = 0.0
    inner = None
    for step in (3.0, 2.4, 1.8, 1.2, 0.8):
        inner = inset_ring(ring, step)
        if inner is not None:
            d = step
            break
    if inner is None:
        b.flat(ring, eave_y, look.roof)
        return
    top = eave_y + pitch * d
    n = len(ring)
    for i in range(n):
        j = (i + 1) % n
        out = np.array([ring[j][1] - ring[i][1], -(ring[j][0] - ring[i][0])])
        b.face([b.p(ring[i], eave_y), b.p(ring[j], eave_y), b.p(inner[j], top), b.p(inner[i], top)],
               [(0, 1, 2), (0, 2, 3)], look.roof, (out[0], 1.0, -out[1]))
    b.flat(inner, top, look.roof)


def parapet(b: Builder, ring, eave_y, look: Look, height: float = 0.8, thick: float = 0.3) -> None:
    inner = inset_ring(ring, thick)
    if inner is None:
        return
    b.walls(ring, eave_y, eave_y + height, look.wall)
    b.walls(inner, eave_y, eave_y + height, look.wall, inward=True)
    n = len(ring)
    for i in range(n):
        j = (i + 1) % n
        b.face([b.p(ring[i], eave_y + height), b.p(ring[j], eave_y + height), b.p(inner[j], eave_y + height),
                b.p(inner[i], eave_y + height)], [(0, 1, 2), (0, 2, 3)], look.trim, (0, 1, 0))


def tower(b: Builder, x, y, half, from_y, body_y, spire_y, look: Look) -> None:
    ring = np.array([[x - half, y - half], [x + half, y - half], [x + half, y + half], [x - half, y + half]])
    body_y = max(body_y, from_y + 2.0)
    b.walls(ring, from_y - 0.2, body_y, look.wall)
    if spire_y > body_y + 2.0:
        tip = (x, y)
        for i in range(4):
            a, c = ring[i], ring[(i + 1) % 4]
            out = np.array([c[1] - a[1], -(c[0] - a[0])])
            b.face([b.p(a, body_y), b.p(c, body_y), b.p(tip, spire_y)], [(0, 1, 2)], look.roof, (out[0], 0.6, -out[1]))
    else:
        b.flat(ring, body_y, look.roof)


def windows(b: Builder, ring, floor_y, top_y, look: Look, party, street_side, start_floor: int = 0) -> None:
    """Rows of windows on every wall long enough, a row a storey, and a glass shop front at street level."""
    if look.kind == "shed" or top_y - floor_y < 2.4:
        return
    house = look.kind == "house"
    n = len(ring)
    ground_h = SHOP_STOREY if look.shopfront else STOREY
    rows, level = [], floor_y
    if start_floor == 0:
        rows.append(("ground", floor_y))
        level = floor_y + ground_h
    while level + 2.4 <= top_y + 0.2:
        rows.append(("upper", level))
        level += STOREY
    w_width, w_height, spacing = (1.0, 1.3, 3.4) if house else (1.2, 1.7, 2.8)
    for i in range(n):
        a, c = np.asarray(ring[i], float), np.asarray(ring[(i + 1) % n], float)
        length = float(np.linalg.norm(c - a))
        if length < 2.6 or party(a, c):
            continue
        along = (c - a) / length
        shop_here = look.shopfront and street_side(a, c) and length > 4
        for name, y in rows:
            if name == "ground" and shop_here:
                panes = max(1, int(length // 3.2))
                pane = (length - 0.6) / panes
                for k in range(panes):
                    p0 = a + along * (0.3 + k * pane + 0.2)
                    p1 = a + along * (0.3 + (k + 1) * pane - 0.2)
                    b.wall(p0, p1, y + 0.5, y + 3.2, GLASS, push=0.03)
                b.wall(a, c, y + 3.3, y + 3.6, look.trim, push=0.05)
                continue
            count = int((length - 1.0) // spacing)
            if count < 1:
                continue
            gap = length / count
            sill = y + (0.9 if house else 1.0)
            for k in range(count):
                mid = a + along * (gap * (k + 0.5))
                b.wall(mid - along * w_width / 2, mid + along * w_width / 2, sill, sill + w_height, GLASS, push=0.03)
    if look.kind in ("shop", "civic", "block") and top_y - floor_y > 5:
        for i in range(n):   # a cornice where the wall meets the roof
            a, c = ring[i], ring[(i + 1) % n]
            if np.linalg.norm(np.asarray(c) - a) > 2 and not party(a, c):
                b.wall(a, c, top_y - 0.45, top_y, look.trim, push=0.12)


def colours(tags: dict, osm_id: str, kind: str, roof_sample, shopfront: bool, plan: Plan) -> Look:
    wall = parse_colour(tags.get("building:colour"))
    roof = parse_colour(tags.get("roof:colour")) or roof_sample
    material = tags.get("building:material", "")
    if wall is None:
        if material in ("brick",):
            wall = pick(BRICK, osm_id)
        elif material in ("stone", "sandstone", "limestone"):
            wall = pick(STONE, osm_id)
        elif kind == "house":
            wall = pick(SIDING, osm_id)
        elif kind == "shed":
            wall = pick(SHEDS, osm_id)
        elif kind in ("shop", "civic") or shopfront:
            wall = pick(BRICK + STONE, osm_id)
        else:
            wall = pick(MODERN, osm_id)
    if roof is None:
        roof = pick(FLAT_ROOFS, osm_id + "r") if plan.style == "flat" else pick(SHINGLE, osm_id + "r")
    trim = tuple(int(0.45 * c + 0.55 * s) for c, s in zip(wall, (208, 200, 184)))   # a pale stone or paint
    return Look(wall, roof, trim, kind, shopfront)
