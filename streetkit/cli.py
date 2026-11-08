"""streetkit fetch | names | sanitize | verify | lidar | roofs | accuracy | build

Run from a project folder holding area.json and data/: the defaults are relative to the working directory.

An area with a "boundary_relation" is a city: its tiles are the ones within "margin_m" of that OSM boundary, its
OpenStreetMap comes from the regional extract named in "osm_extract" (needs the `city` extra), and its raw data is too
large to commit, so `fetch` writes it to build/<name>-raw and pins URLs and hashes in city/SOURCES.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .context import load_overrides
from .fetch import boundary_rings, fetch, fetch_city, fetch_street_names, sha256, verify, verify_city
from .geo import Area, utm_to_lonlat
from .osm import parse
from .sanitize import sanitize
from .streets import load_names, load_places, manifest_lines, place_lines
from .lidar import FILE as LIDAR_FILE, Lidar, fetch_lidar, fetch_lidar_city
from .roofs import FILE as ROOFS_FILE, NAIP, fetch_image, load as load_roofs, sample_roofs
from .tiles import Dem, build_all

ROOT = Path.cwd()
BOX_MARGIN_M = 100.0   # roads and buildings that start just outside a city's tiles


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="streetkit")
    ap.add_argument("command", choices=["fetch", "names", "sanitize", "verify", "lidar", "roofs", "accuracy", "build"])
    ap.add_argument("--area", type=Path, default=ROOT / "area.json")
    ap.add_argument("--raw", type=Path, default=None, help="default: data/raw, or build/<area>-raw for a city")
    ap.add_argument("--places", type=Path, default=ROOT / "data" / "landmarks.csv")
    ap.add_argument("--out", type=Path, default=None, help="default: build/tiles, or build/<area> for a city")
    ap.add_argument("--fetched", type=Path, default=ROOT / "build" / "fetched", help="full downloads, never committed")
    ap.add_argument("--sources", type=Path, default=ROOT / "city" / "SOURCES.json", help="a city's pinned downloads")
    ap.add_argument("--sample", type=int, default=1, help="write every Nth tile, to measure a large area")
    ap.add_argument("--resume", action="store_true", help="keep tiles already written, after an interrupted build")
    ap.add_argument("--workers", type=int, default=4, help="processes for a city's lidar, each about 2 GB of memory")
    args = ap.parse_args(argv)
    area = Area.load(args.area)
    city = area.boundary_relation != 0
    args.raw = args.raw or (ROOT / "build" / f"{area.name}-raw" if city else ROOT / "data" / "raw")
    args.out = args.out or (ROOT / "build" / (area.name if city else "tiles"))
    if args.command == "fetch":
        fetch_city(area, args.raw, args.sources) if city else fetch(area, args.fetched)
        return 0
    if args.command == "names":
        print(f"wrote {fetch_street_names(area, args.raw)}")
        return 0
    if args.command == "sanitize":
        out = args.raw / "osm-sample.osm"
        print(sanitize(sorted(args.fetched.glob("osm-*.osm")), out))
        sources = json.loads((args.raw / "SOURCES.json").read_text())
        sources[out.name] = {"sha256": sha256(out), "bytes": out.stat().st_size}
        (args.raw / "SOURCES.json").write_text(json.dumps(sources, indent=1) + "\n")
        return 0
    if args.command == "lidar":
        if city:
            # Too large to commit, like the rest of a city's raw data: pinned by hash in the city's sources.
            city_area = area.within(boundary_rings(args.raw / "boundary.osm"))
            files = fetch_lidar_city(city_area, args.raw, args.fetched / "lidar", workers=args.workers)
            pinned = json.loads(args.sources.read_text()) if args.sources.exists() else {}
            for path, source in files:
                pinned[path.name] = {"url": source, "sha256": sha256(path), "bytes": path.stat().st_size}
            args.sources.write_text(json.dumps(pinned, indent=1) + "\n")
            print(f"wrote {len(files)} lidar squares to {args.raw}")
            return 0
        path, source = fetch_lidar(area, args.raw, args.fetched / "lidar", json.loads(args.area.read_text()).get("lidar_survey"))
        _record(args.raw, path, source)
        print(f"wrote {path}")
        return 0
    if args.command == "roofs":
        image = fetch_image(area, args.fetched)
        path = sample_roofs(area, parse([args.raw / "osm-sample.osm"]), image, args.raw, Lidar(args.raw / LIDAR_FILE))
        _record(args.raw, path, NAIP + " (sampled; the image is not kept)")
        print(f"wrote {path}")
        return 0
    if args.command == "accuracy":
        from .accuracy import measure

        for k, v in measure(parse([args.raw / "osm-sample.osm"]), Lidar(args.raw / LIDAR_FILE)).items():
            print(f"{k:28} {v}")
        return 0
    if args.command == "verify":
        bad = verify_city(args.raw, args.sources) if city else verify(args.raw)
        print("raw data OK" if not bad else f"mismatch: {bad}")
        return 1 if bad else 0
    if city:
        from .extract import read_extract

        area = area.within(boundary_rings(args.raw / "boundary.osm"))
        west, south, east, north = area.bounds()
        lon0, lat0 = utm_to_lonlat(west - BOX_MARGIN_M, south - BOX_MARGIN_M)
        lon1, lat1 = utm_to_lonlat(east + BOX_MARGIN_M, north + BOX_MARGIN_M)
        osm = read_extract(args.raw / "region.osm.pbf", lon0, lat0, lon1, lat1)
        dem = Dem(*sorted(args.raw.glob("dem-*.tif")))
    else:
        osm = parse(sorted(args.raw.glob("osm-*.osm")))
        dem = Dem(args.raw / "dem-3dep.tif")
    names = args.raw / "street-names.csv"
    streets = manifest_lines(area, osm, load_names(names)) if names.exists() else []
    streets += place_lines(area, load_places(args.places)) if args.places.exists() else []
    grids = sorted(args.raw.glob("lidar-*-*.tif")) if city else [args.raw / LIDAR_FILE]
    grids = [g for g in grids if g.exists()]
    lidar = Lidar(*grids) if grids else None
    summary = build_all(area, osm, dem, args.out, streets, lidar=lidar, roof_colours=load_roofs(args.raw / ROOFS_FILE),
                        overrides=load_overrides(args.places.parent / "building-overrides.csv"),
                        sample=args.sample, resume=args.resume, quiet=not city)
    print(f"wrote {args.out}: {summary}")
    return 0


def _record(raw: Path, path: Path, url: str) -> None:
    sources = json.loads((raw / "SOURCES.json").read_text())
    sources[path.name] = {"url": url, "sha256": sha256(path), "bytes": path.stat().st_size}
    (raw / "SOURCES.json").write_text(json.dumps(sources, indent=1) + "\n")


if __name__ == "__main__":
    sys.exit(main())
