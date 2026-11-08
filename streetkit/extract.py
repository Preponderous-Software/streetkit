"""An area out of a regional .osm.pbf extract, for areas too large for the OSM API.

Needs the `city` extra (pyosmium). Only the tags the build reads are kept, so names, addresses and contributor
details stop here and cannot reach a tile.
"""

from __future__ import annotations

from pathlib import Path

from .osm import OsmData, assemble
from .sanitize import KEEP_TAGS as SAMPLE_TAGS

KEEP_TAGS = SAMPLE_TAGS | {"service"}   # the same tags a committed sample keeps


def kept(tags) -> dict:
    return {t.k: t.v for t in tags if t.k in KEEP_TAGS}


def read_extract(path: Path, west: float, south: float, east: float, north: float) -> OsmData:
    """Everything with a node inside the lon/lat box. Ways keep every node, so they do not stop at the edge."""
    import osmium

    nodes, node_tags, ways, relations = {}, {}, {}, {}

    class Reader(osmium.SimpleHandler):
        def node(self, n):
            if n.tags.get("natural") == "tree" and west <= n.location.lon <= east and south <= n.location.lat <= north:
                nodes[str(n.id)] = (n.location.lon, n.location.lat)
                node_tags[str(n.id)] = kept(n.tags)

        def way(self, w):
            tags = kept(w.tags)
            located = [(str(n.ref), n.location.lon, n.location.lat) for n in w.nodes if n.location.valid()]
            if not located or not any(west <= lon <= east and south <= lat <= north for _, lon, lat in located):
                return
            for ref, lon, lat in located:
                nodes[ref] = (lon, lat)
            ways[str(w.id)] = ([ref for ref, _, _ in located], tags)

        def relation(self, r):
            if r.tags.get("type") == "multipolygon":
                relations[str(r.id)] = ([(m.type, str(m.ref), m.role) for m in r.members], kept(r.tags))

    Reader().apply_file(str(path), locations=True, idx="flex_mem")
    relations = {rid: ([("way" if t == "w" else t, ref, role) for t, ref, role in members], tags)
                 for rid, (members, tags) in relations.items()}
    return assemble(nodes, node_tags, ways, relations)
