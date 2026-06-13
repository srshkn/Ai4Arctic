"""
export_geojson_smooth.py — Production export of MAGT zones as multi-class GeoJSON
with continuous color gradient. Generates 29 GeoJSON files (2007–2035) and a
year-slider preview HTML for visual inspection / frontend reference.

Output dir:  model/results/geojson_smooth/
    magt_smooth_2007.geojson ... magt_smooth_2035.geojson   (29 files, ~2 MB each)
    preview_slider.html                                       (interactive slider)

Each GeoJSON: FeatureCollection with 15 MultiPolygon features (one per class).
Properties: { year, class_id, color (hex), magt_min, magt_max, label }.

Frontend usage:
    L.geoJSON(data, { style: f => ({ fillColor: f.properties.color, ... }) })

Dependencies: rasterio shapely matplotlib

Usage:
    python3 model/scripts/export_geojson_smooth.py
    cd model/results/geojson_smooth && python3 -m http.server 8000
    open http://localhost:8000/preview_slider.html
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


BASE_DIR = Path(__file__).resolve().parent.parent          # .../model/
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
        lo = VMIN if np.isinf(low)  else low
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
    if np.isinf(low):  return f"< {high:.1f}°C"
    if np.isinf(high): return f"> {low:.1f}°C"
    return f"{low:.1f}..{high:.1f}°C"


def classify(magt_year: np.ndarray) -> np.ndarray:
    out = np.zeros(magt_year.shape, dtype="uint8")
    valid = ~np.isnan(magt_year)
    for i in range(len(BIN_EDGES) - 1):
        low, high = BIN_EDGES[i], BIN_EDGES[i + 1]
        out[valid & (magt_year >= low) & (magt_year < high)] = i + 1
    return out


def export_year(magt_year: np.ndarray, transform, year: int,
                colors: list[str], ranges: list[tuple], out_path: Path) -> int:
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
        features.append({
            "type": "Feature",
            "geometry": mapping(merged),
            "properties": {
                "year":     year,
                "class_id": class_id,
                "color":    colors[class_id - 1],
                "magt_min": None if np.isinf(low)  else float(low),
                "magt_max": None if np.isinf(high) else float(high),
                "label":    make_label(low, high),
            },
        })

    fc = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "features": features,
    }
    with open(out_path, "w") as f:
        json.dump(fc, f, separators=(",", ":"))
    return len(features)


# ===================================================================
# Preview HTML with year slider
# ===================================================================
def build_preview_html(years: list[int], colors: list[str],
                        ranges: list[tuple]) -> str:
    """Build a standalone Leaflet page with year slider over all GeoJSONs."""
    # Legend HTML
    legend_rows = []
    for c, (low, high) in zip(colors, ranges):
        label = make_label(low, high)
        legend_rows.append(
            f'<div class="row"><span class="sw" style="background:{c}"></span>'
            f'<span class="lbl">{label}</span></div>'
        )
    legend_html = "\n".join(legend_rows)
    years_js = json.dumps(years)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>MAGT 2007–2035 — Permafrost Zones Slider</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    body, html {{ margin: 0; height: 100%; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }}
    #map {{ position: absolute; top: 0; left: 0; right: 0; bottom: 70px; }}
    #controls {{ position: absolute; bottom: 0; left: 0; right: 0; height: 70px;
                background: #fafafa; border-top: 1px solid #ddd; padding: 12px 24px;
                display: flex; align-items: center; gap: 16px; }}
    #year {{ font-size: 22px; font-weight: 600; min-width: 70px; }}
    #slider {{ flex: 1; }}
    #legend {{ position: absolute; top: 12px; right: 12px; z-index: 1000;
              background: rgba(255,255,255,0.95); padding: 10px 14px;
              border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,0.15);
              font-size: 11px; max-height: calc(100vh - 100px); overflow-y: auto; }}
    #legend h4 {{ margin: 0 0 6px 0; font-size: 12px; }}
    #legend .row {{ display: flex; align-items: center; margin: 2px 0; }}
    #legend .sw {{ width: 16px; height: 12px; margin-right: 8px; border: 1px solid #999; }}
    #info {{ position: absolute; top: 12px; left: 60px; z-index: 1000;
            background: rgba(255,255,255,0.95); padding: 6px 12px;
            border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.15);
            font-size: 12px; color: #555; }}
  </style>
</head>
<body>
  <div id="map"></div>
  <div id="info">Drag slider · {len(years)} years · {len(colors)} classes · EPSG:4326</div>
  <div id="legend">
    <h4>MAGT (°C)</h4>
    {legend_html}
  </div>
  <div id="controls">
    <div id="year">2025</div>
    <input id="slider" type="range" min="0" max="{len(years) - 1}" value="{years.index(2025) if 2025 in years else 0}" step="1" />
  </div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const YEARS = {years_js};
    const map = L.map('map').setView([67, 105], 3);
    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
                {{ attribution: '© OpenStreetMap' }}).addTo(map);

    let currentLayer = null;
    const cache = {{}};

    async function load(year) {{
      if (!cache[year]) {{
        const r = await fetch(`magt_smooth_${{year}}.geojson`);
        cache[year] = await r.json();
      }}
      return cache[year];
    }}

    async function show(year) {{
      const data = await load(year);
      if (currentLayer) map.removeLayer(currentLayer);
      currentLayer = L.geoJSON(data, {{
        style: f => ({{
          fillColor:   f.properties.color,
          fillOpacity: 0.78,
          color:       '#333',
          weight:      0.2,
          opacity:     0.4
        }}),
        onEachFeature: (f, layer) => {{
          layer.bindTooltip(f.properties.label, {{ sticky: true }});
        }}
      }}).addTo(map);
      document.getElementById('year').textContent = year;
    }}

    const slider = document.getElementById('slider');
    slider.addEventListener('input', () => show(YEARS[+slider.value]));
    show(YEARS[+slider.value]);

    // Prefetch neighbors after first paint
    setTimeout(() => YEARS.forEach((y, i) => {{
      if (Math.abs(i - +slider.value) <= 3) load(y);
    }}), 500);
  </script>
</body>
</html>
"""


# ===================================================================
# Main
# ===================================================================
def main() -> int:
    if not SRC_NPZ.exists():
        print(f"ERROR: source not found: {SRC_NPZ}", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = np.load(SRC_NPZ, allow_pickle=False)
    magt = data["magt"]                      # (29, 231, 1501)
    years = data["years"]                    # (29,)
    lats = data["lats"]; lons = data["lons"]

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
        print(f"  {out_path.name:35s}  {size_kb:8.1f} KB  ({n_feat}/{len(BIN_EDGES)-1} cls)")

    preview = OUT_DIR / "preview_slider.html"
    preview.write_text(build_preview_html(
        years=[int(y) for y in years], colors=colors, ranges=ranges,
    ))

    print(f"\nDone. {len(years)} files · {total_kb / 1024:.1f} MB total")
    print(f"Preview: {preview.relative_to(BASE_DIR.parent)}")
    print(f"\nTo view:")
    print(f"  cd model/results/geojson_smooth && python3 -m http.server 8000")
    print(f"  open http://localhost:8000/preview_slider.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
