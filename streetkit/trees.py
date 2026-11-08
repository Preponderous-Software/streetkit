"""Trees from the lidar: the height of each tree OSM maps, and the trees it does not, found as the high points of
leafy canopy. A treetop stands where the canopy is higher than everything within a few metres of it."""

from __future__ import annotations

import math

import numpy as np

MIN_HEIGHT = 4.0       # metres; lower than this is a shrub
LEAFY = 0.3            # share of split returns that says leaves
SPACING = 0.22         # a treetop clears this share of its height of any higher one, so one crown is one tree
MAPPED_REACH = 3.0     # a mapped tree takes the highest canopy within this many metres


def window_max(h: np.ndarray, radius: int) -> np.ndarray:
    out = h.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                out = np.maximum(out, np.roll(np.roll(h, dy, 0), dx, 1))
    return out


def treetops(lidar, built) -> list[tuple[float, float, float, float]]:
    """(east, north, height, crown radius) of every treetop the survey saw, over every grid of the area.
    `built(east, north)` masks out roofs."""
    out = []
    for chunk in getattr(lidar, "chunks", [lidar]):
        out += _treetops(chunk, built)
    return out


def _treetops(lidar, built) -> list[tuple[float, float, float, float]]:
    h = np.where(lidar.leafy >= LEAFY, lidar.height, 0.0)
    peaks = (h >= MIN_HEIGHT) & (h >= window_max(h, 2))
    rows, cols = np.nonzero(peaks)
    east = lidar.w + (cols + 0.5) * lidar.cell
    north = lidar.n - (rows + 0.5) * lidar.cell
    height = h[rows, cols]
    keep = ~built(east, north)
    east, north, height, rows, cols = east[keep], north[keep], height[keep], rows[keep], cols[keep]
    order = np.argsort(-height)
    taken = np.zeros(len(order), bool)
    cell = 8.0
    index: dict = {}
    out = []
    for i in order:
        key = (int(east[i] // cell), int(north[i] // cell))
        clear = SPACING * height[i] + 1.5
        near = [j for dx in (-1, 0, 1) for dy in (-1, 0, 1) for j in index.get((key[0] + dx, key[1] + dy), ())]
        if any(np.hypot(east[i] - east[j], north[i] - north[j]) < max(clear, SPACING * height[j] + 1.5) for j in near):
            continue
        index.setdefault(key, []).append(i)
        out.append((float(east[i]), float(north[i]), float(height[i]), crown_radius(h, rows[i], cols[i], lidar.cell)))
    return out


def crown_radius(h: np.ndarray, row: int, col: int, cell: float) -> float:
    """How far the crown reaches: in eight directions, the distance to where the canopy falls below half the
    treetop's height, the median of them, held to a sensible share of the tree's height."""
    top = h[row, col]
    reach = []
    for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
        step = math.hypot(dr, dc) * cell
        d = 0.0
        r, c = row, col
        while 0 <= r + dr < h.shape[0] and 0 <= c + dc < h.shape[1] and d < top:
            r, c = r + dr, c + dc
            if h[r, c] < top * 0.5:
                break
            d += step
        reach.append(d + step / 2)
    return float(np.clip(np.median(reach), 0.18 * top, 0.5 * top))


def mapped_height(lidar, east: float, north: float) -> float | None:
    """The canopy height over a mapped tree, or None where the survey saw no leaves (planted since)."""
    lidar = next((c for c in getattr(lidar, "chunks", [lidar]) if not hasattr(c, "holds") or c.holds(east, north)), None)
    if lidar is None:
        return None
    r = int(MAPPED_REACH / lidar.cell)
    row, col = lidar.index(east, north)
    win = (slice(max(row - r, 0), row + r + 1), slice(max(col - r, 0), col + r + 1))
    h = np.where(lidar.leafy[win] >= LEAFY, lidar.height[win], 0)
    top = float(h.max()) if h.size else 0.0
    return top if top >= MIN_HEIGHT else None
