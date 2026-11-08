"""OpenStreetMap XML, from any number of overlapping blocks, into buildings, ways, areas and trees."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geo import lonlat_to_utm


@dataclass
class Feature:
    osm_id: str
    tags: dict
    rings: list = field(default_factory=list)   # (N,2) UTM east, north: the outer rings
    holes: list = field(default_factory=list)   # inner rings of a multipolygon: courtyards and light wells
    line: np.ndarray | None = None
    point: tuple | None = None


@dataclass
class OsmData:
    areas: list = field(default_factory=list)
    lines: list = field(default_factory=list)
    points: list = field(default_factory=list)


def _join(ways: list[list]) -> list[list]:
    """Joins open way segments end to end into closed rings."""
    rings, open_ways = [], [list(w) for w in ways]
    while open_ways:
        cur = open_ways.pop()
        grew = True
        while grew and cur[0] != cur[-1]:
            grew = False
            for i, w in enumerate(open_ways):
                if w[0] == cur[-1]:
                    cur += w[1:]
                elif w[-1] == cur[-1]:
                    cur += w[-2::-1]
                elif w[-1] == cur[0]:
                    cur = w[:-1] + cur
                elif w[0] == cur[0]:
                    cur = w[:0:-1] + cur
                else:
                    continue
                open_ways.pop(i)
                grew = True
                break
        if cur[0] == cur[-1] and len(cur) >= 4:
            rings.append(cur)
    return rings


def parse(paths: list[Path]) -> OsmData:
    nodes, node_tags, ways, relations = {}, {}, {}, {}
    for path in paths:
        for el in ET.parse(path).getroot():
            tags = {t.get("k"): t.get("v") for t in el.findall("tag")}
            if el.tag == "node":
                nodes[el.get("id")] = (float(el.get("lon")), float(el.get("lat")))
                if tags:
                    node_tags[el.get("id")] = tags
            elif el.tag == "way":
                ways[el.get("id")] = ([n.get("ref") for n in el.findall("nd")], tags)
            elif el.tag == "relation":
                relations[el.get("id")] = ([(m.get("type"), m.get("ref"), m.get("role")) for m in el.findall("member")], tags)

    ids = list(nodes)
    east, north = lonlat_to_utm(np.array([nodes[i][0] for i in ids]), np.array([nodes[i][1] for i in ids]))
    utm = {i: (float(e), float(n)) for i, e, n in zip(ids, east, north)}

    def coords(refs):
        return np.array([utm[r] for r in refs]) if all(r in utm for r in refs) else None

    out = OsmData()
    outer_only = set()
    for rid, (members, tags) in relations.items():
        if tags.get("type") != "multipolygon":
            continue
        outers = [ways[ref][0] for typ, ref, role in members if typ == "way" and role == "outer" and ref in ways]
        inners = [ways[ref][0] for typ, ref, role in members if typ == "way" and role == "inner" and ref in ways]
        rings = [c for c in (coords(r) for r in _join(outers)) if c is not None]
        holes = [c for c in (coords(r) for r in _join(inners)) if c is not None]
        if rings:
            out.areas.append(Feature("r" + rid, tags, rings=rings, holes=holes))
            outer_only |= {ref for typ, ref, role in members if typ == "way" and role == "outer" and ref in ways and not ways[ref][1]}
    area_keys = ("building", "building:part", "landuse", "leisure", "amenity", "natural", "water")
    for wid, (refs, tags) in ways.items():
        if not tags or wid in outer_only:
            continue
        c = coords(refs)
        if c is None or len(c) < 2:
            continue
        closed = refs[0] == refs[-1] and len(refs) >= 4
        if closed and "highway" not in tags and tags.get("area") != "no" and any(k in tags for k in area_keys):
            out.areas.append(Feature("w" + wid, tags, rings=[c]))
        else:
            out.lines.append(Feature("w" + wid, tags, line=c))
    for nid, tags in node_tags.items():
        if nid in utm:
            out.points.append(Feature("n" + nid, tags, point=utm[nid]))
    return out
