"""How close the drawn buildings come to the measured ones: for every footprint the survey saw, the gap between the
roof as drawn and the lidar, cell by cell. `streetkit accuracy` prints it, for the model and for the guess it
replaced (heights from the kind of building alone), so a change to either can be judged by a number."""

from __future__ import annotations

import numpy as np

from .buildings import guess_shape, kind_of, plan_building, read, roof_shape
from .mesh import clean_ring, signed_area


def drawn_height(plan, east, north) -> np.ndarray:
    """Height above the ground of the roof as drawn, at points inside the footprint (`plan` in a zero origin)."""
    pts = np.column_stack([east, north])
    if plan.style in ("gable", "hip") and plan.rects:
        h = plan.eave + roof_shape(plan.rects, plan.angle, np.zeros(2), plan.pitch, plan.style, east, north)
        h = np.where(np.isnan(h), plan.eave, h)
    else:
        h = np.full(len(pts), plan.eave)
    from .mesh import points_in_ring

    for corners, top in plan.steps:
        inside = points_in_ring(east, north, clean_ring(corners))
        h = np.where(inside, np.maximum(h, top), h)
    return h


def old_guess(tags: dict, area: float) -> float:
    """The height the build used before lidar: one number by kind (house 4.5 m and a 2.5 m gable, else 7 m)."""
    kind = tags.get("building", "yes")
    house = kind in ("house", "detached", "residential", "semidetached_house", "bungalow") or (kind == "yes" and area < 320)
    if kind in ("garage", "garages", "shed", "carport"):
        return 2.7
    return 4.5 + 1.25 if house else 7.0   # the gable's mean height over its footprint


def measure(osm, lidar) -> dict:
    model_err, guess_err, tag_err, sizes = [], [], [], []
    for f in osm.areas:
        if "building" not in f.tags:
            continue
        ring = clean_ring(f.rings[0])
        if len(ring) < 3:
            continue
        reading = read(lidar, ring)
        if reading is None:
            continue
        area = abs(signed_area(ring))
        plan = plan_building(ring, [], f.tags, f.osm_id, lidar, lambda e, n: np.zeros(len(e)), np.zeros(2), 0.0)
        if plan.source != "lidar":
            continue
        model = np.abs(drawn_height(plan, reading.east, reading.north) - reading.height)
        model_err.append(float(np.median(model)))
        guess_err.append(float(np.median(np.abs(old_guess(f.tags, area) - reading.height))))
        eave, style, rise = guess_shape(f.tags, kind_of(f.tags, area), area)
        tag_err.append(float(np.median(np.abs(eave + rise / 2 - reading.height))))
        sizes.append(area)
    m, g, t = np.array(model_err), np.array(guess_err), np.array(tag_err)
    return {"buildings": len(m),
            "model median error m": round(float(np.median(m)), 2), "model within 1 m": round(float((m < 1).mean()), 3),
            "old guess median error m": round(float(np.median(g)), 2), "old guess within 1 m": round(float((g < 1).mean()), 3),
            "fallback median error m": round(float(np.median(t)), 2), "fallback within 1 m": round(float((t < 1).mean()), 3)}
