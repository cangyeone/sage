---
name: terrain_3d_plotting
category: visualization
keywords: 3D, 三维, 三维地形, 3D terrain, terrain surface, topography, elevation, DEM, 四川, Sichuan, Chengdu, Longmenshan, Tibetan Plateau, 45度俯视角, 俯视角, camera view, plotly, three.js, threejs, webgl, pyvista, rasterio, xarray, 地形三维图, 三维地图, 地貌, 山地, relief
related_skills: cartopy_plotting, tabular_io
---

# 3D Terrain / Topography Plotting

Use this skill when the user asks for a three-dimensional terrain/topography map, a 3D geographic scene, a tilted perspective view, or phrases such as:

- “绘制四川地区三维图”
- “45度俯视角”
- “三维地形”
- “3D terrain / topography / relief”
- “带地形起伏的三维地图”

Default output should be a **self-contained Python script** that creates:

1. an interactive `.html` 3D scene, and
2. a `.png` static export if `kaleido` is available.

Prefer **Python + Plotly** for fast scientific output and browser-friendly HTML. Prefer **Three.js/WebGL** when the user asks for a web app, smooth interaction, custom lighting/materials, animated camera, or a more polished 3D scene. Use PyVista only if the user explicitly asks for mesh/VTK-style rendering.

Do **not** use GMT as the main implementation for 3D terrain. GMT is better for publication-style 2D maps, map frames, and classic geophysical cartography; it is usually not the best choice for interactive or visually rich 3D terrain.

## Implementation Choice

| User request | Best path |
|---|---|
| “帮我绘制一下四川地区三维图，45度俯视角，包含地形三维的” | Python + Plotly |
| “生成一个网页里的可交互三维地形” | Three.js |
| “要能旋转、缩放、加光照、浏览器展示” | Three.js |
| “论文里快速放一张三维地形图” | Python + Plotly, plus PNG export |
| “我要严肃 GIS 制图 / GMT 风格图框” | Use `gmt_plotting` for 2D, not this skill |

## Default Geographic Presets

For “四川地区”, use:

```python
REGION = {
    "west": 97.0,
    "east": 109.0,
    "south": 26.0,
    "north": 34.5,
}
```

For Chengdu / Longmenshan focus:

```python
REGION = {
    "west": 101.0,
    "east": 106.0,
    "south": 28.0,
    "north": 32.5,
}
```

Camera for “45度俯视角”:

```python
camera = dict(
    eye=dict(x=1.35, y=-1.35, z=0.95),
    center=dict(x=0, y=0, z=0),
    up=dict(x=0, y=0, z=1),
)
```

Use vertical exaggeration by default:

```python
Z_EXAGGERATION = 2.5
```

## Critical Grid Rules

The most common failure mode is a Plotly surface that becomes a **thin vertical wall or strip**. This means the DEM array and the longitude/latitude axes were transposed, flattened, or masked incorrectly.

Always enforce this invariant before plotting:

```python
assert elev.shape == (len(lat), len(lon))
```

For Plotly, prefer 1D axes:

```python
go.Surface(x=lon, y=lat, z=z_display, surfacecolor=elev)
```

Do not pass a mismatched `LON, LAT = np.meshgrid(...)` result. If a mesh is needed, use `np.meshgrid(lon, lat, indexing="xy")` and assert `LON.shape == elev.shape`.

For Sichuan, do **not** mask/crop the DEM by an administrative polygon in the first pass. Render the rectangular DEM extent first, then optionally overlay a verified boundary line. A bad polygon mask can leave only a narrow sliver and produce the wall-like result.

## Recommended Robust Pattern

```python
import os
from pathlib import Path
import numpy as np
import plotly.graph_objects as go

OUTDIR = Path(os.environ.get("SAGE_OUTDIR", "."))
OUTDIR.mkdir(parents=True, exist_ok=True)

REGION = {"west": 97.0, "east": 109.0, "south": 26.0, "north": 34.5}
Z_EXAGGERATION = 2.5


def validate_grid(lon, lat, elev, label="DEM"):
    lon = np.asarray(lon, dtype=float).ravel()
    lat = np.asarray(lat, dtype=float).ravel()
    elev = np.asarray(elev, dtype=float)
    if elev.ndim != 2:
        raise ValueError(f"{label} must be 2D, got shape {elev.shape}")
    if elev.shape != (len(lat), len(lon)):
        if elev.T.shape == (len(lat), len(lon)):
            elev = elev.T
        else:
            raise ValueError(
                f"{label} grid mismatch: elev={elev.shape}, "
                f"lat={len(lat)}, lon={len(lon)}"
            )
    if lon[0] > lon[-1]:
        lon = lon[::-1]
        elev = elev[:, ::-1]
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        elev = elev[::-1, :]
    if len(lon) < 20 or len(lat) < 20:
        raise ValueError(f"{label} grid too small: {len(lat)} x {len(lon)}")
    if max(len(lon), len(lat)) / max(1, min(len(lon), len(lat))) > 8:
        raise ValueError(f"{label} grid aspect suspicious: {len(lat)} x {len(lon)}")
    elev = np.nan_to_num(elev, nan=float(np.nanmedian(elev)))
    assert elev.shape == (len(lat), len(lon))
    return lon, lat, elev


def load_grd_regular(grd_path):
    """
    Read a GMT/NetCDF grid robustly as lon(1D), lat(1D), elev(nlat, nlon).
    This prevents the classic Plotly 'vertical strip' caused by transposed axes.
    """
    import xarray as xr

    ds = xr.open_dataset(grd_path)
    data_vars = list(ds.data_vars)
    if not data_vars:
        raise ValueError(f"No data variable found in {grd_path}")
    da = ds[data_vars[0]].squeeze()

    lon_name = next((n for n in ("lon", "longitude", "x") if n in da.coords), None)
    lat_name = next((n for n in ("lat", "latitude", "y") if n in da.coords), None)
    if lon_name is None or lat_name is None:
        raise ValueError(f"Cannot identify lon/lat coordinates in {grd_path}: {list(da.coords)}")

    da = da.transpose(lat_name, lon_name)
    lon = da[lon_name].values
    lat = da[lat_name].values
    elev = da.values
    return validate_grid(lon, lat, elev, label=str(grd_path))


def synthetic_sichuan_dem(region, nx=180, ny=140):
    """Offline fallback DEM-like relief for Sichuan / eastern Tibetan Plateau."""
    lon = np.linspace(region["west"], region["east"], nx)
    lat = np.linspace(region["south"], region["north"], ny)
    LON, LAT = np.meshgrid(lon, lat)

    # Broad west-high east-low trend plus Longmenshan-like ridge.
    west_to_east = (region["east"] - LON) / (region["east"] - region["west"])
    plateau = 4200 * west_to_east**1.7
    basin = -900 * np.exp(-(((LON - 104.2) / 1.8) ** 2 + ((LAT - 30.6) / 1.2) ** 2))
    ridge = 1200 * np.exp(-((LAT - (0.55 * (LON - 101.0) + 29.0)) ** 2) / 0.18)
    texture = 180 * np.sin(LON * 2.7) * np.cos(LAT * 2.2)
    elev = plateau + basin + ridge + texture
    elev = np.clip(elev, 200, 6500)
    return validate_grid(lon, lat, elev, label="synthetic DEM")


def try_fetch_dem(region):
    """
    Optional online DEM path.
    Uses OpenTopography globaldem if available; otherwise returns None.
    The fallback is deterministic, so the script still works offline.
    """
    try:
        import requests
        import rasterio
        from rasterio.io import MemoryFile
        url = (
            "https://portal.opentopography.org/API/globaldem"
            "?demtype=SRTMGL1"
            f"&south={region['south']}&north={region['north']}"
            f"&west={region['west']}&east={region['east']}"
            "&outputFormat=GTiff"
        )
        r = requests.get(url, timeout=30)
        if r.status_code != 200 or len(r.content) < 1000:
            return None
        with MemoryFile(r.content) as mem:
            with mem.open() as ds:
                arr = ds.read(1).astype(float)
                arr[arr < -1000] = np.nan
                # Downsample for responsive web rendering.
                step_y = max(1, arr.shape[0] // 160)
                step_x = max(1, arr.shape[1] // 200)
                arr = arr[::step_y, ::step_x]
                west, south, east, north = ds.bounds
                lon = np.linspace(west, east, arr.shape[1])
                lat = np.linspace(north, south, arr.shape[0])
                return validate_grid(lon, lat, arr, label="OpenTopography DEM")
    except Exception:
        return None


local_grd = next((p for p in (OUTDIR / "sichuan_topo.grd", Path("sichuan_topo.grd")) if p.exists()), None)
if local_grd:
    lon, lat, elev = load_grd_regular(local_grd)
    dem_source = f"GMT grid: {local_grd.name}"
else:
    dem = try_fetch_dem(REGION)
    if dem is None:
        lon, lat, elev = synthetic_sichuan_dem(REGION)
        dem_source = "synthetic offline relief fallback"
    else:
        lon, lat, elev = dem
        dem_source = "OpenTopography SRTMGL1"

lon, lat, elev = validate_grid(lon, lat, elev, label=dem_source)
print(
    f"[SAGE_TEST] DEM source = {dem_source}; "
    f"grid = {len(lat)} x {len(lon)}; "
    f"elevation = {np.nanmin(elev):.0f}..{np.nanmax(elev):.0f} m"
)

if max(len(lon), len(lat)) / max(1, min(len(lon), len(lat))) > 4:
    # A valid Sichuan grid should be rectangular, not a narrow strip.
    raise RuntimeError("DEM grid is suspiciously narrow; refusing to render a wall-like surface")

z = elev * Z_EXAGGERATION
xspan = float(np.nanmax(lon) - np.nanmin(lon))
yspan = float(np.nanmax(lat) - np.nanmin(lat))
aspect_y = max(0.55, min(1.15, yspan / max(xspan, 1e-9) * 1.45))

fig = go.Figure(data=[
    go.Surface(
        x=lon,
        y=lat,
        z=z,
        surfacecolor=elev,
        colorscale=[
            [0.00, "#234b9b"],
            [0.18, "#4f9bd7"],
            [0.32, "#6fbf73"],
            [0.55, "#d8c36a"],
            [0.78, "#9d6b3d"],
            [1.00, "#f2f2f2"],
        ],
        colorbar=dict(title="Elevation (m)"),
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True)
        ),
        lighting=dict(ambient=0.45, diffuse=0.75, specular=0.18, roughness=0.65),
        lightposition=dict(x=-100, y=100, z=8000),
    )
])

fig.update_layout(
    title=f"3D Terrain of Sichuan Region ({dem_source})",
    scene=dict(
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        zaxis_title=f"Elevation x{Z_EXAGGERATION}",
        aspectmode="manual",
        aspectratio=dict(x=1.45, y=aspect_y, z=0.34),
        camera=dict(
            eye=dict(x=1.35, y=-1.35, z=0.95),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=0, z=1),
        ),
    ),
    margin=dict(l=0, r=0, t=48, b=0),
)

html_path = OUTDIR / "sichuan_3d_terrain.html"
fig.write_html(str(html_path), include_plotlyjs=True)
print(f"[SAGE_TEST] wrote interactive 3D terrain: {html_path}")

png_path = OUTDIR / "sichuan_3d_terrain.png"
try:
    fig.write_image(str(png_path), scale=2)
    print(f"[FIGURE] {png_path}")
    print(f"[SAGE_TEST] wrote static PNG: {png_path}")
except Exception as exc:
    print(f"[SAGE_TEST] PNG export skipped; install kaleido for static image export: {exc}")

assert html_path.exists() and html_path.stat().st_size > 0
assert elev.shape == (len(lat), len(lon))
```

## Three.js / WebGL Pattern

Use this when the target is a web page or app. The code engine can generate a single self-contained HTML file. The DEM can be embedded as a synthetic grid or loaded from a JSON/CSV grid.

```html
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Sichuan 3D Terrain</title>
  <style>
    html, body { margin: 0; height: 100%; overflow: hidden; background: #0b1020; }
    #scene { width: 100vw; height: 100vh; }
    .label {
      position: fixed; left: 16px; top: 14px; color: white;
      font: 14px/1.35 Arial, sans-serif; background: rgba(0,0,0,.35);
      padding: 8px 10px; border-radius: 8px;
    }
  </style>
</head>
<body>
<div id="scene"></div>
<div class="label">Sichuan 3D terrain · 45 degree oblique view</div>
<script type="module">
import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.164.1/build/three.module.js";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.164.1/examples/jsm/controls/OrbitControls.js";

const root = document.getElementById("scene");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
root.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b1020);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(95, -95, 62); // 45-degree oblique view

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 10);
controls.enableDamping = true;

scene.add(new THREE.HemisphereLight(0xdcecff, 0x2d190d, 1.3));
const sun = new THREE.DirectionalLight(0xffffff, 2.2);
sun.position.set(-60, -80, 120);
scene.add(sun);

const nx = 180, ny = 140;
const width = 120, height = 86;
const geometry = new THREE.PlaneGeometry(width, height, nx - 1, ny - 1);
geometry.rotateX(-Math.PI / 2);

function relief(ix, iy) {
  const x = ix / (nx - 1);
  const y = iy / (ny - 1);
  const westHigh = 28 * Math.pow(1 - x, 1.7);
  const basin = -8 * Math.exp(-((x - 0.62) ** 2 / 0.025 + (y - 0.52) ** 2 / 0.018));
  const ridge = 11 * Math.exp(-((y - (0.55 * x + 0.23)) ** 2) / 0.003);
  const texture = 1.5 * Math.sin(ix * 0.22) * Math.cos(iy * 0.18);
  return Math.max(1, westHigh + basin + ridge + texture);
}

const pos = geometry.attributes.position;
const colors = [];
const color = new THREE.Color();
for (let iy = 0; iy < ny; iy++) {
  for (let ix = 0; ix < nx; ix++) {
    const i = iy * nx + ix;
    const z = relief(ix, iy);
    pos.setY(i, z);
    const t = Math.min(1, z / 34);
    color.setHSL(0.33 - 0.24 * t, 0.55, 0.34 + 0.34 * t);
    colors.push(color.r, color.g, color.b);
  }
}
geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
geometry.computeVertexNormals();

const material = new THREE.MeshStandardMaterial({
  vertexColors: true,
  roughness: 0.78,
  metalness: 0.02,
  side: THREE.DoubleSide
});
const terrain = new THREE.Mesh(geometry, material);
scene.add(terrain);

const grid = new THREE.GridHelper(130, 13, 0x6688aa, 0x223344);
grid.position.y = -0.15;
scene.add(grid);

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
</script>
</body>
</html>
```

When producing this from Python code, write it to:

```python
html_path = OUTDIR / "sichuan_3d_terrain_threejs.html"
html_path.write_text(html_text, encoding="utf-8")
print(f"[SAGE_TEST] wrote Three.js terrain HTML: {html_path}")
assert html_path.exists() and html_path.stat().st_size > 0
```

## If The User Provides Earthquake Catalog Data

Overlay hypocenters only if lon/lat/depth columns are available. Add a `Scatter3d` trace:

```python
import pandas as pd

df = pd.read_csv(catalog_path, sep=None, engine="python")
lon_col = next(c for c in df.columns if "lon" in c.lower())
lat_col = next(c for c in df.columns if "lat" in c.lower())
dep_col = next((c for c in df.columns if "dep" in c.lower() or "depth" in c.lower()), None)
mag_col = next((c for c in df.columns if c.lower() in ("mag", "ml", "mw", "magnitude")), None)

event_z = -df[dep_col].fillna(0).to_numpy() * 1000 if dep_col else np.zeros(len(df))
size = (df[mag_col].fillna(2).clip(lower=0.5) * 2.5).to_numpy() if mag_col else 4

fig.add_trace(go.Scatter3d(
    x=df[lon_col],
    y=df[lat_col],
    z=event_z,
    mode="markers",
    marker=dict(size=size, color=event_z, colorscale="Plasma", opacity=0.82),
    name="Earthquakes",
))
```

## Rules

- For the first version, always create the `.html` output even if PNG export fails.
- For web/app requests, prefer Three.js over Plotly.
- Never fail just because online DEM download fails; use the synthetic fallback.
- For “45度俯视角”, set `camera.eye` rather than trying to rotate data.
- Use moderate grids (`~200 x 160`) for browser responsiveness.
- Include `[SAGE_TEST]` lines and assert output files exist.
- If the user requests exact scientific topography, clearly state the DEM source in stdout and figure title.
