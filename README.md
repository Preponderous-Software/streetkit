# streetkit

Walkable 3D towns from open data. streetkit turns OpenStreetMap, USGS elevation and USGS airborne lidar into
streamed tiles, and gives a game the C++ to load them, walk, ride and fly through them, and draw them with
[Strata](https://github.com/Preponderous-Software/strata).

It was taken out of [Old Town Walk](https://github.com/dmccoystephenson/old-town-walk), which is built on it, and
the city features came from [Ponderosa](https://github.com/Preponderous-Software/ponderosa), which maps all of Fort
Collins.

## What it does

- **Data, offline.** `streetkit fetch` downloads OpenStreetMap and the USGS 3DEP elevation for an area, and
  `streetkit sanitize` keeps only geometry and the tags the build reads, so the committed sample carries no contributor
  names or business details. `streetkit build` then works from committed files alone, checked by hash with
  `streetkit verify`.
- **Buildings measured from lidar.** `streetkit lidar` finds the newest USGS lidar survey over the area, reads its
  point tiles and grids them to one metre. Each footprint's eave and roof are read from that grid: flat, gabled and
  hipped roofs are fitted over the rectangles that cover the footprint, stepped flat roofs and steeples are found, and
  footprints the survey did not see fall back on their OSM tags. `streetkit accuracy` reports how close the drawn roofs
  are to the survey.
- **Walls and colours.** Windows by storey, shop fronts on commercial streets, parapets and cornices. Roof colours can
  be sampled from aerial imagery (`streetkit roofs`), one colour a building, with the imagery itself never kept.
- **Trees** from the lidar, at their measured height and crown width.
- **Whole cities.** An area can name an OpenStreetMap boundary instead of a square: its tiles are the ones within a
  margin of it, its map comes from a dated regional `.osm.pbf` extract (the `city` extra), its elevation is fetched
  in 5 km pieces and joined, and its raw data is pinned by URL and hash in `city/SOURCES.json` instead of committed.
  Builds use every core, can write one tile in N to measure, and resume after an interruption.
- **An overview.** With `overview_cell_m`, the whole area is also written at low resolution, and the C++ draws it as
  Strata's terrain under the tiles, so flight sees land all the way to the horizon.
- **Tiles.** A global grid of UTM squares, each tile self-contained, with a manifest listing every tile by size and
  hash. The format is documented at the top of `streetkit/tiles.py` and read by `src/tile.cpp`.
- **C++.** `include/streetkit/`: the tile and manifest reader, which tiles to stream, a walker that walks, jumps, flies
  and rides a bike, and views for street signs, named places, night lights and the bike's handlebars.

## Using it

A game keeps its area and data and calls streetkit for the rest.

```
my-game/
  area.json        # {"name", "centre_lat", "centre_lon", "tile_size_m", "tiles_across", "cell_m"}, and optionally
                   # "overview_cell_m", "lidar_survey", or for a city "boundary_relation", "margin_m", "osm_extract"
  data/raw/        # what `streetkit fetch | sanitize | lidar | roofs` write, committed with SOURCES.json
  data/landmarks.csv, data/building-overrides.csv   # optional
```

Python, from the game's folder (needs [uv](https://docs.astral.sh/uv/) and Python 3.12):

```sh
uv add "streetkit @ git+https://github.com/Preponderous-Software/streetkit@<commit>"
uv run streetkit verify
uv run streetkit build          # -> build/tiles
```

For a city, add the `city` extra: `uv run --extra city streetkit fetch --area city.json`, then `verify` and `build`
(to `build/<name>`, with `--resume` after an interruption and `--sample N` to try one tile in N).

C++, with CMake 3.24 or newer:

```cmake
FetchContent_Declare(streetkit GIT_REPOSITORY https://github.com/Preponderous-Software/streetkit.git GIT_TAG <commit>)
FetchContent_MakeAvailable(streetkit)
target_link_libraries(my-game PRIVATE streetkit::streetkit strata::static)
```

`tile_stream.h` and the `*_view.h` headers draw with Strata, so the game provides `strata::static`; the rest needs no
GPU. Everything is in `namespace streetkit`.

## Building and testing it

```sh
uv run pytest -q
uv run python tests/make_fixture.py build/fixture     # a tiny tile set for the C++ tests
cmake -S . -B build && cmake --build build -j4 && ctest --test-dir build
```

## Limits

- Areas are in UTM zone 13N (EPSG:32613), which covers Colorado; other zones need `geo.py` to take the zone from
  `area.json`.
- Lidar must be on the USGS open bucket, and a survey older than a building cannot see it.
- Roofs are flat, gabled or hipped over rectangles; mansards, domes and curved roofs are not modelled.

## License

MIT (see `LICENSE`). The data a game builds with has its own terms: OpenStreetMap is under the ODbL, and USGS and
USDA NAIP data are in the public domain with attribution requested.
