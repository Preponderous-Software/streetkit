"""Triangles in UTM-relative metres: (x east, y up, z south). Vertices are shared within a face."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

EPS = 1e-9


def signed_area(ring: np.ndarray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def clean_ring(ring) -> np.ndarray:
    """No closing duplicate, no repeated points, counter-clockwise on a north-up map."""
    pts = np.asarray(ring, dtype=np.float64)
    if len(pts) > 1 and np.linalg.norm(pts[0] - pts[-1]) < 1e-6:
        pts = pts[:-1]
    keep = [i for i in range(len(pts)) if np.linalg.norm(pts[i] - pts[i - 1]) > 1e-3]
    pts = pts[keep]
    if len(pts) >= 3 and signed_area(pts) < 0:
        pts = pts[::-1].copy()
    return pts


def points_in_ring(px: np.ndarray, py: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Even-odd test of arrays of points against one ring."""
    inside = np.zeros(px.shape, dtype=bool)
    x0, y0 = ring[-1]
    for x1, y1 in ring:
        crosses = (y1 > py) != (y0 > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            inside ^= crosses & (px < (x0 - x1) * (py - y1) / (y0 - y1) + x1)
        x0, y0 = x1, y1
    return inside


@dataclass
class Mesh:
    positions: list = field(default_factory=list)
    normals: list = field(default_factory=list)
    colors: list = field(default_factory=list)
    indices: list = field(default_factory=list)

    def face(self, points, triangles, rgb) -> None:
        """One flat face: its points, triangles indexing them, and one sRGB byte colour."""
        p = np.asarray(points, dtype=np.float64)
        a, b, c = p[triangles[0][0]], p[triangles[0][1]], p[triangles[0][2]]
        n = np.cross(b - a, c - a)
        length = float(np.linalg.norm(n))
        if length < EPS:
            return
        n = tuple(n / length)
        base = len(self.positions)
        self.positions += [tuple(v) for v in p]
        self.normals += [n] * len(p)
        self.colors += [tuple(rgb)] * len(p)
        for t in triangles:
            self.indices += [base + t[0], base + t[1], base + t[2]]

    def quad(self, a, b, c, d, rgb) -> None:
        """a, b, c, d counter-clockwise from outside."""
        self.face([a, b, c, d], [(0, 1, 2), (0, 2, 3)], rgb)


def ribbon_quads(line: np.ndarray, width: float, step: float = 2.0):
    """Yields quads (left0, right0, right1, left1) of (x, north) along a polyline, about `step` long, split across
    so that no piece is wider than `step` and the terrain cannot bulge through."""
    pts = [line[0]]
    for a, b in zip(line, line[1:]):
        length = float(np.linalg.norm(b - a))
        for j in range(1, max(1, math.ceil(length / step)) + 1):
            pts.append(a + (b - a) * (j / max(1, math.ceil(length / step))))
    pts = np.array(pts)
    pts = pts[np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-3]]
    if len(pts) < 2:
        return
    seg = np.diff(pts, axis=0)
    seg /= np.linalg.norm(seg, axis=1, keepdims=True)
    perp = np.stack([-seg[:, 1], seg[:, 0]], axis=1)
    side = [perp[0]]
    for i in range(1, len(pts) - 1):
        m = perp[i - 1] + perp[i]
        length = float(np.linalg.norm(m))
        m = perp[i] if length < 1e-6 else m / length
        side.append(m / max(float(m @ perp[i]), 0.5))
    side.append(perp[-1])
    lanes = max(1, math.ceil(width / step))
    for i in range(len(pts) - 1):
        for k in range(lanes):
            t0, t1 = k / lanes - 0.5, (k + 1) / lanes - 0.5
            yield (pts[i] - side[i] * width * t0, pts[i] - side[i] * width * t1,
                   pts[i + 1] - side[i + 1] * width * t1, pts[i + 1] - side[i + 1] * width * t0)
