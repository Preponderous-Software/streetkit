"""Street names for the signs and the on-screen street, from a list of OSM way ids and names.

The committed map carries no names (see sanitize.py). Streets are the one exception, kept apart in street-names.csv,
so what is public is plain: the names of the public roads inside the map.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from .geo import Area, lonlat_to_utm
from .osm import OsmData

ROADS = ("trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "living_street", "pedestrian")
LONG_TO_SHORT = {"Street": "St", "Avenue": "Ave", "Drive": "Dr", "Court": "Ct", "Lane": "Ln", "Road": "Rd",
                 "Place": "Pl", "Boulevard": "Blvd", "Parkway": "Pkwy", "Highway": "Hwy"}
COMPASS = {"East": "E", "West": "W", "North": "N", "South": "S"}
SAME_SIGN_M = 40.0   # one sign for two streets within this distance, as at a dual carriageway's two crossings
CORNER_M = (8.0, 11.0, 14.0)  # how far from the crossing's centre a post may stand, nearest tried first
KERB_M = 5.0         # a post stands at least this far from the middle of any named road


def short_name(name: str) -> str:
    """The name as a street sign writes it: `East Mountain Avenue` becomes `E Mountain Ave`."""
    words = name.split()
    if len(words) > 2 and words[0] in COMPASS:
        words[0] = COMPASS[words[0]]
    if len(words) > 1 and words[-1] in LONG_TO_SHORT:
        words[-1] = LONG_TO_SHORT[words[-1]]
    return " ".join(words)


def load_names(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as f:
        return {row["way"]: short_name(row["name"]) for row in csv.DictReader(f)}


def _distance_to_line(point, line) -> float:
    a, b = line[:-1], line[1:]
    d = b - a
    lengths = (d * d).sum(1)
    t = np.clip(((point - a) * d).sum(1) / np.where(lengths > 0, lengths, 1), 0, 1)
    return float(np.hypot(*(a + d * t[:, None] - point).T).min())


def _inside(point, ring) -> bool:
    x, y = point
    hit = False
    for (ax, ay), (bx, by) in zip(ring[:-1], ring[1:]):
        if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
            hit = not hit
    return hit


def _corner(centre, roads, rings):
    """A spot on the pavement by a crossing: off the road and outside every building, or None."""
    near_roads = [r for r in roads if np.abs(r - centre).max(axis=1).min() < 80]
    near_rings = [r for r in rings if np.abs(r - centre).max(axis=1).min() < 60]
    for reach in CORNER_M:
        for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            spot = centre + np.array([sx, sy]) * reach / math.sqrt(2)
            if all(_distance_to_line(spot, r) >= KERB_M for r in near_roads) and not any(_inside(spot, r) for r in near_rings):
                return spot
    return None


def load_places(path: Path) -> list:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def place_lines(area: Area, places: list) -> list:
    """The manifest's `place` lines, one for each place on the map. A place is `name|sentence` after its position: names
    have spaces, and the viewer puts both straight into JSON and on screen, so they stay plain ASCII."""
    west, south, east, north = area.bounds()
    lines = []
    for place in places:
        words = place["name"] + place["sentence"]
        if not words.isprintable() or not words.isascii() or set(words) & set('|"\\'):
            raise ValueError(f"not plain text: {place['name']}")
        e, n = lonlat_to_utm(float(place["lon"]), float(place["lat"]))
        if west <= e <= east and south <= n <= north:
            lines.append(f"place {e:.1f} {n:.1f} {place['name']}|{place['sentence']}")
    return lines


def manifest_lines(area: Area, osm: OsmData, names: dict) -> list:
    """The manifest's `street`, `road` and `sign` lines. Names and shapes come from the map, so a way the list does not
    know is left out, and a way the map does not have adds nothing."""
    west, south, east, north = area.bounds()
    named = [(names[f.osm_id[1:]], f.line) for f in osm.lines if f.osm_id[1:] in names and f.line is not None
             and f.tags.get("highway", "").removesuffix("_link") in ROADS and f.tags.get("tunnel") in (None, "no")]
    named = [(n, np.asarray(line)) for n, line in named]
    ids = {n: i for i, n in enumerate(sorted({n for n, _ in named}))}
    lines = [f"street {i} {n}" for n, i in ids.items()]
    at = {}
    for name, line in named:
        lines.append(f"road {ids[name]} " + " ".join(f"{e:.1f} {n:.1f}" for e, n in line))
        for e, n in line:
            at.setdefault((round(float(e), 2), round(float(n), 2)), set()).add(name)
    rings = [np.asarray(f.rings[0]) for f in osm.areas if "building" in f.tags]
    roads = [line for _, line in named]
    signs = []
    for (e, n), crossing in sorted(at.items()):
        if len(crossing) < 2 or not (west <= e <= east and south <= n <= north):
            continue
        pair = sorted(crossing)[:2]
        if any(s[0] == pair and math.hypot(s[1][0] - e, s[1][1] - n) < SAME_SIGN_M for s in signs):
            continue
        spot = _corner(np.array([e, n]), roads, rings)
        if spot is not None:
            signs.append((pair, spot))
    lines += [f"sign {s[1][0]:.1f} {s[1][1]:.1f} {ids[s[0][0]]} {ids[s[0][1]]}" for s in signs]
    return lines
