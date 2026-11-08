"""Reduces OpenStreetMap XML to what the tile build reads, for committing a sample.

An OSM download carries each object's editor name, user id, changeset and timestamps, along with names, addresses,
phone numbers and websites. None of that is needed to draw a block, so none of it is kept: only geometry, element ids
(which pick colours) and the tags in KEEP_TAGS. Street names live apart, in street-names.csv.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

KEEP_TAGS = frozenset({"amenity", "area", "building", "building:levels", "height", "highway", "landuse", "leisure",
                       "natural", "place", "tunnel", "type", "water", "width",
                       # how a building looks: its parts, roof and colours, none of which says who is inside
                       "building:part", "building:min_level", "min_height", "building:colour", "building:material",
                       "roof:shape", "roof:height", "roof:levels", "roof:colour", "roof:material", "roof:direction",
                       "roof:orientation"})
ROOT_ATTRIBUTES = {"version": "0.6", "generator": "streetkit sanitize", "copyright": "OpenStreetMap and contributors",
                   "attribution": "https://www.openstreetmap.org/copyright",
                   "license": "https://opendatacommons.org/licenses/odbl/1-0/"}


def _tags(el) -> dict:
    return {t.get("k"): t.get("v") for t in el.findall("tag") if t.get("k") in KEEP_TAGS}


def sanitize(paths: list[Path], out: Path) -> dict:
    """Merges overlapping downloads into one file at `out`, and returns how many of each element it kept."""
    nodes, ways, relations = {}, {}, {}
    for path in paths:
        for el in ET.parse(path).getroot():
            tags = _tags(el)
            if el.tag == "node":
                nodes[el.get("id")] = (el.get("lat"), el.get("lon"), tags)
            elif el.tag == "way":
                ways[el.get("id")] = ([n.get("ref") for n in el.findall("nd")], tags)
            elif el.tag == "relation" and tags.get("type") == "multipolygon" and set(tags) != {"type"}:
                members = [(m.get("type"), m.get("ref"), m.get("role")) for m in el.findall("member")]
                relations[el.get("id")] = (members, tags)

    member_ways = {ref for members, _ in relations.values() for typ, ref, _role in members if typ == "way"}
    kept_ways = {i: w for i, w in ways.items() if w[1] or i in member_ways}
    used = {ref for refs, _ in kept_ways.values() for ref in refs}
    kept_nodes = {i: n for i, n in nodes.items() if i in used or n[2].get("natural") == "tree"}

    root = ET.Element("osm", ROOT_ATTRIBUTES)
    for nid in sorted(kept_nodes, key=int):
        lat, lon, tags = kept_nodes[nid]
        el = ET.SubElement(root, "node", {"id": nid, "lat": lat, "lon": lon})
        for k in sorted(tags) if tags.get("natural") == "tree" else ():
            ET.SubElement(el, "tag", {"k": k, "v": tags[k]})
    for wid in sorted(kept_ways, key=int):
        refs, tags = kept_ways[wid]
        el = ET.SubElement(root, "way", {"id": wid})
        for ref in refs:
            ET.SubElement(el, "nd", {"ref": ref})
        for k in sorted(tags):
            ET.SubElement(el, "tag", {"k": k, "v": tags[k]})
    for rid in sorted(relations, key=int):
        members, tags = relations[rid]
        el = ET.SubElement(root, "relation", {"id": rid})
        for typ, ref, role in members:
            if typ == "way":
                ET.SubElement(el, "member", {"type": typ, "ref": ref, "role": role})
        for k in sorted(tags):
            ET.SubElement(el, "tag", {"k": k, "v": tags[k]})
    ET.indent(root, space="")
    out.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(out, encoding="utf-8", xml_declaration=True)
    return {"nodes": len(kept_nodes), "ways": len(kept_ways), "relations": len(relations)}
