"""streetkit fetch | names | sanitize | verify | lidar | roofs | accuracy | build

Run from a project folder holding area.json and data/: the defaults are relative to the working directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .context import load_overrides
from .fetch import fetch, fetch_street_names, sha256, verify
from .geo import Area
from .osm import parse
from .sanitize import sanitize
from .streets import load_names, load_places, manifest_lines, place_lines
from .lidar import FILE as LIDAR_FILE, Lidar, fetch_lidar
from .roofs import FILE as ROOFS_FILE, NAIP, fetch_image, load as load_roofs, sample_roofs
from .tiles import Dem, build_all

ROOT = Path.cwd()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="streetkit")
    ap.add_argument("command", choices=["fetch", "names", "sanitize", "verify", "lidar", "roofs", "accuracy", "build"])
    ap.add_argument("--area", type=Path, default=ROOT / "area.json")
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--places", type=Path, default=ROOT / "data" / "landmarks.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "tiles")
    ap.add_argument("--fetched", type=Path, default=ROOT / "build" / "fetched", help="full downloads, never committed")
    args = ap.parse_args(argv)
    area = Area.load(args.area)
    if args.command == "fetch":
        fetch(area, args.fetched)
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
        bad = verify(args.raw)
        print("raw data OK" if not bad else f"mismatch: {bad}")
        return 1 if bad else 0
    osm = parse(sorted(args.raw.glob("osm-*.osm")))
    streets = manifest_lines(area, osm, load_names(args.raw / "street-names.csv")) + place_lines(area, load_places(args.places))
    lidar_path = args.raw / LIDAR_FILE
    lidar = Lidar(lidar_path) if lidar_path.exists() else None
    summary = build_all(area, osm, Dem(args.raw / "dem-3dep.tif"), args.out, streets, lidar=lidar,
                        roof_colours=load_roofs(args.raw / ROOFS_FILE),
                        overrides=load_overrides(args.places.parent / "building-overrides.csv"))
    print(f"wrote {args.out}: {summary}")
    return 0


def _record(raw: Path, path: Path, url: str) -> None:
    sources = json.loads((raw / "SOURCES.json").read_text())
    sources[path.name] = {"url": url, "sha256": sha256(path), "bytes": path.stat().st_size}
    (raw / "SOURCES.json").write_text(json.dumps(sources, indent=1) + "\n")


if __name__ == "__main__":
    sys.exit(main())
