"""Builds a small tile set for the C++ tests: four 100 m tiles of 10 m cells over gently sloping ground, with a
house, a road across a tile edge and a tree. `python tests/make_fixture.py OUT`."""

import sys
from pathlib import Path

import numpy as np

from streetkit.geo import Area
from streetkit.osm import Feature, OsmData
from streetkit.tiles import build_all


class SlopingDem:
    def height(self, east, north):
        return 1500.0 + (np.asarray(east, np.float64) - 1000.0) * 0.01


class FixtureArea(Area):
    def centre_utm(self):
        return 1000.0, 2000.0


def main(out: Path) -> None:
    osm = OsmData()
    osm.areas.append(Feature("w1", {"building": "house"}, rings=[np.array([(950.0, 1950.0), (960.0, 1950.0), (960.0, 1956.0), (950.0, 1956.0)])]))
    osm.lines.append(Feature("w2", {"highway": "residential"}, line=np.array([(910.0, 2010.0), (1090.0, 2010.0)])))
    osm.points.append(Feature("n3", {"natural": "tree"}, point=(1040.0, 1940.0)))
    print(build_all(FixtureArea("fixture", 40.6, -105.0, 100.0, 2, 10.0), osm, SlopingDem(), out))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
