"""Street names and places reach the manifest, and every sign and place stands somewhere sensible."""

import re
from pathlib import Path

import numpy as np
import pytest

from streetkit.geo import Area
from streetkit.osm import parse
from streetkit.streets import KERB_M, _distance_to_line, _inside, load_names, load_places, manifest_lines, place_lines, short_name
from streetkit.tiles import read_manifest, write_manifest

ROOT = Path(__file__).resolve().parents[2]
AREA = ROOT / "tests" / "fixtures" / "area.json"   # a real area: Old Town Fort Collins


def test_names_are_written_as_on_a_sign():
    assert short_name("East Mountain Avenue") == "E Mountain Ave"
    assert short_name("North College Avenue") == "N College Ave"
    assert short_name("Linden Street") == "Linden St"
    assert short_name("East Drive") == "East Dr"
    assert short_name("Old Town Square") == "Old Town Square"


def test_a_place_outside_the_map_is_left_out():
    area = Area.load(AREA)
    assert place_lines(area, [{"name": "Far Off", "lat": "39.7", "lon": "-104.9", "sentence": "Elsewhere."}]) == []


@pytest.mark.parametrize("name", ['A "quoted" place', "A|B", "Caf\u00e9", "back\\slash"])
def test_a_place_must_be_plain_text(name):
    with pytest.raises(ValueError):
        place_lines(Area.load(AREA), [{"name": name, "lat": "40.5877", "lon": "-105.0761", "sentence": "Words."}])


def test_a_place_line_survives_the_manifest(tmp_path):
    area = Area.load(AREA)
    path = tmp_path / "manifest.txt"
    write_manifest(path, area, 1500, {(1, 2): (3, "abc")}, ["place 1.0 2.0 Old Town Square|A sentence."])
    assert read_manifest(path)["place"] == "1.0 2.0 Old Town Square|A sentence."
