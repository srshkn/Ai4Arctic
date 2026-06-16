"""
export_geojson_smooth.py — Production export of MAGT zones as multi-class GeoJSON
with continuous color gradient.

Output dir:  model/results/geojson_smooth/
    magt_smooth_2007.geojson ... magt_smooth_2035.geojson   (29 files, ~2 MB each)

Each GeoJSON: FeatureCollection with 15 MultiPolygon features (one per class).
Properties: { year, class_id, color (hex), magt_min, magt_max, label }.

Dependencies: rasterio shapely matplotlib

Usage:
    python3 model/scripts/export_geojson_smooth.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio.features
from matplotlib.colors import TwoSlopeNorm, to_hex
from rasterio.transform import from_origin
from shapely.geometry import mapping, shape
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union

BASE_DIR = Path(__file__).resolve().parent.parent  # .../model/
SRC_NPZ = BASE_DIR / "results" / "maps" / "magt_all_years_2007_2035.npz"
OUT_DIR = BASE_DIR / "results" / "geojson_smooth"

# Fixed scale across all years — same colors mean same temperature on slider.
VMIN, VCENTER, VMAX = -15.0, 0.0, 10.0
CMAP_NAME = "RdBu_r"

# 15 bins, narrower around 0°C (climate-critical boundary)
BIN_EDGES = [-np.inf, -10, -8, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 5, 8, np.inf]

SIMPLIFY_TOL = 0.05
COORD_PRECISION = 4


# ===================================================================
# Helpers
# ===================================================================
def precompute_class_colors() -> tuple[list[str], list[tuple]]:
    """Color per class from RdBu_r at TwoSlopeNorm-mapped class midpoint."""
    norm = TwoSlopeNorm(vmin=VMIN, vcenter=VCENTER, vmax=VMAX)
    cmap = plt.colormaps[CMAP_NAME]
    colors, ranges = [], []
    n = len(BIN_EDGES) - 1
    for i in range(n):
        low, high = BIN_EDGES[i], BIN_EDGES[i + 1]
        lo = VMIN if np.isinf(low) else low
        hi = VMAX if np.isinf(high) else high
        mid = max(min((lo + hi) / 2, VMAX), VMIN)
        colors.append(to_hex(cmap(norm(mid))))
        ranges.append((low, high))
    return colors, ranges


def round_geom(geom, precision: int = COORD_PRECISION):
    return shapely_transform(
        lambda x, y, z=None: (round(x, precision), round(y, precision)), geom
    )


def make_label(low: float, high: float) -> str:
    if np.isinf(low):
        return f"< {high:.1f}°C"
    if np.isinf(high):
        return f"> {low:.1f}°C"
    return f"{low:.1f}..{high:.1f}°C"


def classify(magt_year: np.ndarray) -> np.ndarray:
    out = np.zeros(magt_year.shape, dtype="uint8")
    valid = ~np.isnan(magt_year)
    for i in range(len(BIN_EDGES) - 1):
        low, high = BIN_EDGES[i], BIN_EDGES[i + 1]
        out[valid & (magt_year >= low) & (magt_year < high)] = i + 1
    return out


def export_year(
    magt_year: np.ndarray,
    transform,
    year: int,
    colors: list[str],
    ranges: list[tuple],
    out_path: Path,
) -> int:
    classes = classify(magt_year)
    polys_by_class: dict[int, list] = defaultdict(list)
    for geom, val in rasterio.features.shapes(
        classes, mask=classes > 0, transform=transform, connectivity=8
    ):
        polys_by_class[int(val)].append(shape(geom))

    features = []
    for class_id in sorted(polys_by_class):
        merged = unary_union(polys_by_class[class_id])
        merged = merged.simplify(SIMPLIFY_TOL, preserve_topology=True)
        merged = round_geom(merged)
        low, high = ranges[class_id - 1]
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(merged),
                "properties": {
                    "year": year,
                    "class_id": class_id,
                    "color": colors[class_id - 1],
                    "magt_min": None if np.isinf(low) else float(low),
                    "magt_max": None if np.isinf(high) else float(high),
                    "label": make_label(low, high),
                },
            }
        )

    fc = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "features": features,
    }
    with open(out_path, "w") as f:
        json.dump(fc, f, separators=(",", ":"))
    return len(features)


# ===================================================================
# Main
# ===================================================================
def main() -> int:
    if not SRC_NPZ.exists():
        print(f"ERROR: source not found: {SRC_NPZ}", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = np.load(SRC_NPZ, allow_pickle=False)
    magt = data["magt"]  # (29, 231, 1501)
    years = data["years"]  # (29,)
    lats = data["lats"]
    lons = data["lons"]

    dx = float(abs(lons[1] - lons[0]))
    dy = float(abs(lats[1] - lats[0]))
    flip = lats[0] < lats[-1]
    if flip:
        magt = magt[:, ::-1, :]
        lat_top = float(lats[-1]) + dy / 2
    else:
        lat_top = float(lats[0]) + dy / 2
    transform = from_origin(float(lons[0]) - dx / 2, lat_top, dx, dy)

    colors, ranges = precompute_class_colors()

    print(f"Input:    shape={magt.shape}")
    print(f"Years:    {int(years[0])}..{int(years[-1])}  ({len(years)} files)")
    print(f"Classes:  {len(BIN_EDGES) - 1}")
    print(f"Scale:    vmin={VMIN}, vcenter={VCENTER}, vmax={VMAX}, cmap={CMAP_NAME}")
    print(f"Output:   {OUT_DIR.relative_to(BASE_DIR.parent)}/\n")

    total_kb = 0.0
    for i, year in enumerate(years):
        out_path = OUT_DIR / f"magt_smooth_{int(year)}.geojson"
        n_feat = export_year(magt[i], transform, int(year), colors, ranges, out_path)
        size_kb = out_path.stat().st_size / 1024
        total_kb += size_kb
        print(
            f"  {out_path.name:35s}  {size_kb:8.1f} KB  ({n_feat}/{len(BIN_EDGES) - 1} cls)"
        )

    print(f"\nDone. {len(years)} files · {total_kb / 1024:.1f} MB total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
