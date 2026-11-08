"""Downloads the raw data once. A small area's raw files are committed; a city's are too large, so only their URLs
and SHA-256 are (fetch_city, verify_city). The build is offline either way.

OSM downloads carry contributor metadata, so they stay out of git: `streetkit sanitize` reduces them to data/raw/osm-sample.osm, and the build reads only that, offline."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .geo import UTM_EPSG, Area, lonlat_to_utm, utm_to_lonlat
from .osm import _join
from .streets import ROADS

OSM_MAP = "https://api.openstreetmap.org/api/0.6/map"
OVERPASS = ("https://overpass.private.coffee/api/interpreter", "https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter")
DEM_EXPORT = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
USER_AGENT = "streetkit/0.1 (open data 3D map)"
OSM_RELATION = "https://api.openstreetmap.org/api/0.6/relation"
DEM_CHUNK_M = 5000.0   # a city's elevation is fetched in squares this wide
DEM_BORDER_M = 8.0     # beyond a tile's edge, for the normals and for geometry that crosses it
BLOCK_M = 1000.0   # OSM is fetched in blocks this wide, which keeps each call under the API's node limit
MARGIN_M = 60.0    # roads and buildings that start just outside the area


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=240) as r:
        return r.read()


def _download(url: str, path: Path) -> None:
    """To a file, in pieces, and only under its name once it is whole."""
    part = path.with_suffix(path.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=240) as r, open(part, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            part.replace(path)
            return
        except (urllib.error.URLError, TimeoutError):
            if attempt == 5:
                raise
            time.sleep(2 ** attempt * 5)   # public services fail for a while under load


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(area: Area, raw: Path) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    w, s, e, n = area.bounds()
    sources = {}

    def save(name: str, url: str) -> None:
        path = raw / name
        if not path.exists():
            path.write_bytes(_get(url))
            time.sleep(1.1)
        sources[name] = {"url": url, "sha256": sha256(path), "bytes": path.stat().st_size}

    cols = int((e - w) // BLOCK_M + (1 if (e - w) % BLOCK_M else 0))
    rows = int((n - s) // BLOCK_M + (1 if (n - s) % BLOCK_M else 0))
    for r in range(rows):
        for c in range(cols):
            lon0, lat0 = utm_to_lonlat(w + c * BLOCK_M - MARGIN_M, s + r * BLOCK_M - MARGIN_M)
            lon1, lat1 = utm_to_lonlat(w + (c + 1) * BLOCK_M + MARGIN_M, s + (r + 1) * BLOCK_M + MARGIN_M)
            box = ",".join(f"{v:.6f}" for v in (lon0, lat0, lon1, lat1))
            save(f"osm-{r}-{c}.osm", OSM_MAP + "?bbox=" + box)

    border = 2 * area.cell
    px = int(round((e - w + 2 * border) / area.cell))
    query = {"bbox": f"{w - border},{s - border},{e + border},{n + border}", "bboxSR": UTM_EPSG, "imageSR": UTM_EPSG,
             "size": f"{px},{px}", "format": "tiff", "pixelType": "F32",
             "interpolation": "RSP_BilinearInterpolation", "noDataInterpretation": "esriNoDataMatchAny", "f": "image"}
    save("dem-3dep.tif", DEM_EXPORT + "?" + urllib.parse.urlencode(query))
    (raw / "SOURCES.json").write_text(json.dumps(sources, indent=1) + "\n")


def fetch_street_names(area: Area, raw: Path) -> Path:
    """Writes street-names.csv: the id and name of each public road on the map, and nothing else about it."""
    w, s, e, n = area.bounds()
    lon0, lat0 = utm_to_lonlat(w, s)
    lon1, lat1 = utm_to_lonlat(e, n)
    roads = "|".join(ROADS)
    query = (f'[out:json][timeout:60];way["highway"~"^({roads})(_link)?$"]["name"]'
             f"({lat0:.5f},{lon0:.5f},{lat1:.5f},{lon1:.5f});out tags;")
    ways = None
    for host in OVERPASS:  # the public servers are often busy
        req = urllib.request.Request(host, data=urllib.parse.urlencode({"data": query}).encode(),
                                     headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                ways = json.load(r)["elements"]
            break
        except OSError as failure:
            print(f"{host}: {failure}")
    if ways is None:
        raise RuntimeError("no Overpass server answered")
    path = raw / "street-names.csv"
    rows = sorted((w["id"], w["tags"]["name"]) for w in ways)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["way", "name"])
        writer.writerows(rows)
    sources = json.loads((raw / "SOURCES.json").read_text())
    sources[path.name] = {"url": host, "sha256": sha256(path), "bytes": path.stat().st_size}
    (raw / "SOURCES.json").write_text(json.dumps(sources, indent=1) + "\n")
    return path


def boundary_rings(path: Path) -> list:
    """The outer rings of a boundary relation, from the XML of `relation/<id>/full`, in UTM."""
    root = ET.parse(path).getroot()
    nodes = {n.get("id"): (float(n.get("lon")), float(n.get("lat"))) for n in root.findall("node")}
    ways = {w.get("id"): [nd.get("ref") for nd in w.findall("nd")] for w in root.findall("way")}
    outers = [ways[m.get("ref")] for m in root.find("relation").findall("member")
              if m.get("type") == "way" and m.get("role") == "outer" and m.get("ref") in ways]
    rings = []
    for refs in _join(outers):
        lon, lat = np.array([nodes[r] for r in refs]).T
        rings.append(np.column_stack(lonlat_to_utm(lon, lat))[:-1])
    return rings


def dem_chunks(area: Area, keys: list) -> list[tuple[int, int]]:
    """The DEM_CHUNK_M squares that hold any tile, each with the border its tiles' normals read."""
    size = area.tile_size
    out = set()
    for ix, iy in keys:
        for e in (ix * size - DEM_BORDER_M, (ix + 1) * size + DEM_BORDER_M):
            for n in (iy * size - DEM_BORDER_M, (iy + 1) * size + DEM_BORDER_M):
                out.add((math.floor(e / DEM_CHUNK_M), math.floor(n / DEM_CHUNK_M)))
    return sorted(out)


def fetch_city(area: Area, raw: Path, sources: Path) -> None:
    """The boundary, the regional extract and the DEM chunks. Writes their URLs and SHA-256 to `sources`."""
    raw.mkdir(parents=True, exist_ok=True)
    pinned = json.loads(sources.read_text()) if sources.exists() else {}
    record = {}

    def save(name: str, url: str) -> Path:
        path = raw / name
        if not path.exists():
            _download(url, path)
            time.sleep(1.1)
        digest = sha256(path)
        if name in pinned and pinned[name]["sha256"] != digest:
            raise SystemExit(f"{name} does not match {sources}")
        record[name] = {"url": url, "sha256": digest, "bytes": path.stat().st_size}
        return path

    boundary = save("boundary.osm", f"{OSM_RELATION}/{area.boundary_relation}/full")
    save("region.osm.pbf", area.osm_extract)
    city = area.within(boundary_rings(boundary))
    px = int(DEM_CHUNK_M / area.cell)
    for ce, cn in dem_chunks(area, city.tiles()):
        e, n = int(ce * DEM_CHUNK_M), int(cn * DEM_CHUNK_M)
        query = {"bbox": f"{e},{n},{e + DEM_CHUNK_M:.0f},{n + DEM_CHUNK_M:.0f}", "bboxSR": UTM_EPSG, "imageSR": UTM_EPSG,
                 "size": f"{px},{px}", "format": "tiff", "pixelType": "F32",
                 "interpolation": "RSP_BilinearInterpolation", "noDataInterpretation": "esriNoDataMatchAny", "f": "image"}
        save(f"dem-{ce}-{cn}.tif", DEM_EXPORT + "?" + urllib.parse.urlencode(query))
    sources.parent.mkdir(parents=True, exist_ok=True)
    sources.write_text(json.dumps(record, indent=1) + "\n")


def verify_city(raw: Path, sources: Path) -> list[str]:
    pinned = json.loads(sources.read_text())
    return [name for name, s in pinned.items() if not (raw / name).exists() or sha256(raw / name) != s["sha256"]]


def verify(raw: Path) -> list[str]:
    """Names of raw files that are missing or do not match SOURCES.json."""
    sources = json.loads((raw / "SOURCES.json").read_text())
    return [name for name, s in sources.items() if not (raw / name).exists() or sha256(raw / name) != s["sha256"]]
