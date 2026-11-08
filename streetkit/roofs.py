"""Roof colours, sampled from aerial imagery.

The imagery is used as a reference only: it is downloaded to build/fetched (never committed), and all that is kept
is one colour a roof, the median of the clear pixels inside the footprint, in data/raw/roof-colours.csv. Pixels
under a tree (by the lidar) and pixels in deep shadow are left out. Nothing from the image is drawn on a surface.
"""

from __future__ import annotations

import csv
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from .geo import UTM_EPSG, Area
from .lidar import grid_bounds
from .mesh import clean_ring, points_in_ring

NAIP = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/exportImage"
PIXEL = 0.6
FILE = "roof-colours.csv"


def fetch_image(area: Area, fetched: Path, pieces: int = 3) -> Path:
    """The aerial image over the area, fetched in pieces (the service times out on one large request)."""
    import tifffile

    w, s, e, n = grid_bounds(area)
    path = fetched / "naip.tif"
    if path.exists():
        return path
    fetched.mkdir(parents=True, exist_ok=True)
    side = int(round((e - w) / PIXEL / pieces))
    span = (e - w) / pieces
    rows = []
    for j in range(pieces):
        row = []
        for i in range(pieces):
            box = (w + i * span, n - (j + 1) * span, w + (i + 1) * span, n - j * span)
            query = {"bbox": ",".join(f"{v:.2f}" for v in box), "bboxSR": UTM_EPSG, "imageSR": UTM_EPSG,
                     "size": f"{side},{side}", "format": "tiff", "pixelType": "U8", "bandIds": "0,1,2",
                     "interpolation": "RSP_BilinearInterpolation", "f": "image"}
            piece = fetched / f"naip-{j}-{i}.tif"
            if not piece.exists():
                req = urllib.request.Request(NAIP + "?" + urllib.parse.urlencode(query),
                                             headers={"User-Agent": "streetkit/0.1 (open data 3D map)"})
                for attempt in range(4):   # the service is slow and often times out
                    try:
                        with urllib.request.urlopen(req, timeout=300) as r:
                            piece.write_bytes(r.read())
                        break
                    except urllib.error.HTTPError:
                        if attempt == 3:
                            raise
            image = tifffile.imread(piece)
            row.append(np.moveaxis(image, 0, -1) if image.shape[0] == 3 else image)
        rows.append(np.concatenate(row, axis=1))
    tifffile.imwrite(path, np.concatenate(rows, axis=0))
    return path


def sample(image: np.ndarray, bounds, ring: np.ndarray, lidar=None) -> tuple | None:
    """Median colour of a footprint's clear pixels, or None when too few are clear."""
    w, s, e, n = bounds
    px = (e - w) / image.shape[1]
    from .lidar import edge_distance

    c0, c1 = int((ring[:, 0].min() - w) / px), int((ring[:, 0].max() - w) / px) + 1
    r0, r1 = int((n - ring[:, 1].max()) / px), int((n - ring[:, 1].min()) / px) + 1
    c0, r0, c1, r1 = max(c0, 0), max(r0, 0), min(c1, image.shape[1]), min(r1, image.shape[0])
    if c1 <= c0 or r1 <= r0:
        return None
    cols, rows = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
    ge, gn = w + (cols.ravel() + 0.5) * px, n - (rows.ravel() + 0.5) * px
    inside = points_in_ring(ge, gn, ring)
    if inside.sum() >= 12:
        inside &= edge_distance(ge, gn, ring) > 0.7
    if lidar is not None and inside.any():
        inside &= lidar.sample(ge, gn)[1] < 0.35
    rgb = image[rows.ravel(), cols.ravel()][inside].astype(np.float64)
    if len(rgb) < 6:
        return None
    light = rgb.mean(axis=1)
    rgb = rgb[light >= np.percentile(light, 30)]   # leave out the shadowed side of a pitched roof
    return tuple(int(round(v)) for v in np.median(rgb, axis=0))


def sample_roofs(area: Area, osm, image_path: Path, raw: Path, lidar=None) -> Path:
    import tifffile

    image = tifffile.imread(image_path)
    if image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    bounds = grid_bounds(area)
    path = raw / FILE
    with open(path, "w", newline="") as f:
        out = csv.writer(f, lineterminator="\n")   # the hash in SOURCES.json must match a checkout on any system
        out.writerow(["osm_id", "r", "g", "b"])
        for feature in sorted(osm.areas, key=lambda x: x.osm_id):
            if "building" not in feature.tags and "building:part" not in feature.tags:
                continue
            ring = clean_ring(feature.rings[0])
            if len(ring) < 3:
                continue
            rgb = sample(image, bounds, ring, lidar)
            if rgb is not None:
                out.writerow([feature.osm_id, *rgb])
    return path


def load(path: Path) -> dict:
    if not Path(path).exists():
        return {}
    with open(path) as f:
        return {row["osm_id"]: (int(row["r"]), int(row["g"]), int(row["b"])) for row in csv.DictReader(f)}


def as_paint(rgb: tuple) -> tuple:
    """An aerial colour as a roof colour: haze lifts and greys what the camera saw from above, so the colour is
    pushed back a little toward its own hue and down in brightness."""
    v = np.array(rgb, np.float64)
    grey = v.mean()
    v = grey + (v - grey) * 1.25
    v = v * 0.92
    return tuple(int(x) for x in np.clip(np.round(v), 0, 255))
