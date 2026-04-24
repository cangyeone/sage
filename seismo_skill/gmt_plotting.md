---
name: gmt_plotting
category: visualization
keywords: GMT, map, seismicity, epicenter distribution, station distribution, fault, topography, terrain, terrain background, 地形底图, focal mechanism, coastline, contour, cross-section, travel time, earthquake catalog, basemap, coast, pscoast, meca, coupe, grdimage, gmt begin, gmt end, run_gmt, 地图, 震中分布, 台站分布, 地形, 地形底图, 震源机制, 海岸线
---

# GMT Map Plotting

## Description

Use GMT (Generic Mapping Tools) to create professional seismology maps: epicenter distribution maps, station location maps, topographic maps, focal mechanism beachballs, cross-sections, etc.

> If the user explicitly requests `地形底图` or `terrain background`, always include a terrain grid (`gmt grdimage`) before coastlines and plot layers.

---

## ⚠️ Critical Rules

- `run_gmt(script, outname, title)` is pre-injected; **call directly, no import needed**
- `script` parameter is a **complete GMT6 bash script string** (multi-line string)
- Script must use `gmt begin <name> PNG` ... `gmt end` structure (GMT6 modern mode)
- **Write Chinese/non-ASCII titles and labels directly in the script** (e.g., `-BWSne+t"中国地形图"`); `run_gmt` auto-handles CJK characters without conversion needed
- After execution, automatically generates **PNG image** and **.sh script file**, both downloadable from interface
- Requires GMT >= 6.0 installed on system

## ⚠️ TOPOGRAPHY RULES — READ FIRST

**ALWAYS add terrain background when plotting geographic maps.**
The user almost always wants topography. Do NOT skip it.

### Layer order (mandatory):
1. `gmt grdimage` — terrain background (FIRST, before everything)
2. `gmt coast` — coastlines / borders on top of terrain (NO `-G` fill when using grdimage)
3. `gmt plot` / `gmt meca` etc. — data on top
4. `gmt colorbar` — color scale

### Robust topography snippet (copy-paste this every time):
```bash
# --- Download relief grid (try high→low resolution) ---
# Note: always check the file was actually created (grdcut may exit 0 without output)
if gmt grdcut @earth_relief_01m -R${R} -Gtopo.grd 2>/dev/null && [ -f topo.grd ]; then
  echo "01m OK"
elif gmt grdcut @earth_relief_02m -R${R} -Gtopo.grd 2>/dev/null && [ -f topo.grd ]; then
  echo "02m OK"
else
  gmt grdcut @earth_relief_05m -R${R} -Gtopo.grd
fi

# --- Verify grid exists before proceeding ---
if [ ! -f topo.grd ]; then
  echo "ERROR: topo.grd not created. Check region bounds: R=${R}" >&2
  exit 1
fi

# --- ALWAYS build the CPT explicitly — never rely on -Cetopo1 alone ---
Z_MIN=$(gmt grdinfo topo.grd -C 2>/dev/null | awk '{print $6}')
Z_MAX=$(gmt grdinfo topo.grd -C 2>/dev/null | awk '{print $7}')
if [ -z "${Z_MIN}" ] || [ -z "${Z_MAX}" ] || ! printf '%s\n' "${Z_MIN} ${Z_MAX}" | grep -Eq '^-?[0-9]+(\.[0-9]+)?\s+-?[0-9]+(\.[0-9]+)?$'; then
  echo "ERROR: invalid elevation range (${Z_MIN}, ${Z_MAX})" >&2
  exit 1
fi
gmt makecpt -Cgeo -T${Z_MIN}/${Z_MAX} -Z > topo.cpt
if [ ! -s topo.cpt ]; then
  echo "ERROR: topo.cpt empty" >&2
  exit 1
fi

# --- Render terrain with hillshading ---
gmt grdimage topo.grd -J${J} -R${R} -Ctopo.cpt -I+d

# --- Coast on top of terrain (NO -G fill — would hide terrain) ---
gmt coast -R${R} -J${J} -W0.6p,gray30 -N1/0.8p,gray50 -A500

# --- Elevation colorbar ---
gmt colorbar -DJBC+w7c/0.35c+o0/0.5c -Ctopo.cpt -Baf+l"Elevation"
```

### Why explicit makecpt is required:
- ❌ `-Cetopo1` — GMT looks for built-in CPT; if not found, falls back to **grayscale → land appears solid black**
- ✅ `gmt makecpt -Cgeo -T${Z_MIN}/${Z_MAX} -Z > topo.cpt` then `-Ctopo.cpt` — always works, correct colors
- `-Cgeo` is built into every GMT6 installation (land=tan/green/brown, ocean=blue/cyan)
- Use `gmt grdinfo -C` to get real min/max so the color range exactly fits the data

### Other common mistakes:
- ❌ `gmt coast -Gtan` / `-Gwhite` / `-G<ANY color>` — SOLID fill completely **hides terrain**. NEVER use -G with grdimage.
- ❌ `gmt basemap` before `gmt grdimage` — basemap frame gets buried under terrain
- ❌ Missing `-I+d` on grdimage — flat, washed-out colors without hillshading
- ❌ Hard-coding `-T-6000/6000` when region is all land — white/grey scale, looks wrong
- ❌ `gmt coast -Slightblue` alone without grdimage — ocean will be colored but land has no terrain

---

## Core Function

### `run_gmt(script, outname="gmt_map", title="GMT Map")`

**Parameters:**
- `script` : str — Complete GMT6 bash script (multi-line string)
- `outname` : str — Output file base name, e.g. `"seismicity_map"`
- `title` : str — Script comment title

**Returns:** str (PNG image path)

---

## GMT Basemap Skill

## Objective

Add and configure a `basemap` in GMT scripts to control map region, projection, axes, ticks, annotations, title, and overall layout.

## When to Use

Use this skill when the user needs to:
- Define map boundaries (`-R`)
- Set projection (`-J`)
- Add axes, ticks, gridlines (`-B`)
- Add map title or labels
- Standardize GMT plotting structure
- Prepare a base layer before plotting data (e.g., earthquakes, stations, faults)

---

## Core Command

```bash
gmt basemap -R<west>/<east>/<south>/<north> -J<projection> -B<axis_settings>
```

---

## Typical Example

```bash
gmt basemap -R100/110/20/30 -JM12c -Bxa2f1 -Bya2f1 -BWSen
```

### Meaning

```
-R100/110/20/30    Region: lon 100–110, lat 20–30
-JM12c             Mercator projection, 12 cm width
-Bxa2f1            X-axis: major ticks every 2°, minor every 1°
-Bya2f1            Y-axis: major ticks every 2°, minor every 1°
-BWSen             Draw west & south with annotations; east & north frame only
```

---

## Standard Script Template

```bash
#!/bin/bash

gmt begin map png

gmt basemap \
    -R100/110/20/30 \
    -JM12c \
    -Bxa2f1+l"Longitude" \
    -Bya2f1+l"Latitude" \
    -BWSen+t"Study Area"

gmt coast \
    -W0.5p,black \
    -Df \
    -N1/0.5p,gray \
    -A1000

gmt end show
```

---

## Recommended Workflow

Always plot `basemap` first, then overlay data layers:

```bash
gmt begin seismic_map pdf

gmt basemap -R100/110/20/30 -JM15c -Bxa2f1 -Bya2f1 -BWSen+t"Seismicity Map"

gmt coast -W0.5p -Df -N1/0.5p,gray -A1000

gmt plot faults.txt -W1p,red

gmt plot stations.txt -St0.25c -Gblue -W0.3p,black

gmt plot earthquakes.txt -Sc0.08c -Gred -W0.1p,black

gmt end show
```

---

## Key Parameters

### 1. Region (`-R`)

```bash
-Rwest/east/south/north
```

Example:

```bash
-R95/110/20/35
```

Cartesian:

```bash
-R0/100/0/50
```

---

### 2. Projection (`-J`)

Common options:

```bash
-JM12c       # Mercator (maps)
-JX12c/8c    # Cartesian plot
-JL...       # Lambert projection
```

Typical usage:

* Maps → `-JM`
* Profiles / sections → `-JX`

---

### 3. Axes & Frame (`-B`)

Example:

```bash
-Bxa1f0.5 -Bya1f0.5 -BWSen
```

Meaning:

```
a1      major ticks = 1
f0.5    minor ticks = 0.5
W S     draw + annotate west & south
e n     draw east & north frame only
```

Add labels:

```bash
-Bxa2f1+l"Longitude"
-Bya2f1+l"Latitude"
```

Add title:

```bash
-BWSen+t"Earthquake Distribution"
```

---

## Geographic Map Template

```bash
gmt basemap \
    -R${lon_min}/${lon_max}/${lat_min}/${lat_max} \
    -JM${width}c \
    -Bxa${x_major}f${x_minor}+l"Longitude" \
    -Bya${y_major}f${y_minor}+l"Latitude" \
    -BWSen+t"${title}"
```

---

## Cross-section (Depth Profile) Template

```bash
gmt basemap \
    -R0/100/0/40 \
    -JX12c/-6c \
    -Bxa20f10+l"Distance (km)" \
    -Bya10f5+l"Depth (km)" \
    -BWSen+t"Depth Section"
```

Note:

```
-JX12c/-6c
```

Negative height flips the Y-axis (depth increasing downward).

---

## Complete Working Example: Topographic Basemap

**Use this template directly.** It handles all common pitfalls: terrain download, CPT creation, hillshading, coastlines, and colorbar.

```bash
#!/usr/bin/env bash

# ============================================================================
# Complete Topographic Basemap Script
# Region: Western China (95–110°E, 20–35°N)
# ============================================================================

# Map settings
region=95/110/20/35
projection=M15c

    # --- Step 1: Download terrain grid with fallback resolution ---
    if gmt grdcut @earth_relief_01m -R${region} -Gtopo.grd 2>/dev/null && [ -f topo.grd ]; then
        echo "✓ Terrain resolution 01m downloaded"
    elif gmt grdcut @earth_relief_02m -R${region} -Gtopo.grd 2>/dev/null && [ -f topo.grd ]; then
        echo "✓ Terrain resolution 02m downloaded"
    else
        echo "⚠ Using fallback resolution 05m"
        gmt grdcut @earth_relief_05m -R${region} -Gtopo.grd
    fi

    # --- Step 2: Verify terrain file exists ---
    if [ ! -f topo.grd ]; then
        echo "ERROR: topo.grd not created. Check region: R=${region}" >&2
        exit 1
    fi

    # --- Step 3: Extract elevation range and create color palette ---
    Z_MIN=$(gmt grdinfo topo.grd -C 2>/dev/null | awk '{print $6}')
    Z_MAX=$(gmt grdinfo topo.grd -C 2>/dev/null | awk '{print $7}')
    if [ -z "${Z_MIN}" ] || [ -z "${Z_MAX}" ] || ! printf '%s\n' "${Z_MIN} ${Z_MAX}" | grep -Eq '^-?[0-9]+(\.[0-9]+)?\s+-?[0-9]+(\.[0-9]+)?$'; then
        echo "ERROR: invalid elevation range (${Z_MIN}, ${Z_MAX})" >&2
        exit 1
    fi
    echo "Elevation range: ${Z_MIN} — ${Z_MAX} m"

    gmt makecpt -Cgeo -T${Z_MIN}/${Z_MAX}/100 > topo.cpt
    if [ ! -s topo.cpt ]; then
        echo "ERROR: topo.cpt empty" >&2
        exit 1
    fi

gmt begin topo_map png

    # --- Step 4: Render terrain with hillshading ---
    gmt grdimage topo.grd -R${region} -J${projection} \
        -Ctopo.cpt -I+d

    # --- Step 5: Add coastlines and borders on top of terrain ---
    # NOTE: NO -G (fill) — it would completely hide the terrain!
    gmt coast -R${region} -J${projection} \
        -W0.6p,gray30 \
        -N1/0.8p,gray50 \
        -A500

    # --- Step 6: Add title and frame ---
    gmt basemap -R${region} -J${projection} \
        -Bxa2f1+l"Longitude (°E)" \
        -Bya2f1+l"Latitude (°N)" \
        -BWSen+t"Topographic Basemap"

    # --- Step 7: Add elevation colorbar ---
    # CRITICAL: -C topo.cpt and -D position must be present
    gmt colorbar -DJBC+w7c/0.35c+o0/0.5c \
        -Ctopo.cpt \
        -Baf+l"Elevation (m)"

    # --- Optional: Add stations/earthquakes as red dots ---
    # gmt plot stations.txt -Sc0.15c -Gred -W0.3p,black
    # gmt plot earthquakes.txt -Sc0.08c -Gred -W0.1p,darkred

gmt end show
```

### Key Features

| Feature | How It Works |
|---------|-------------|
| **Fallback resolution** | Tries 01m → 02m → 05m to ensure download succeeds |
| **Validation** | Checks `topo.grd` exists and elevation range is valid before continuing |
| **Explicit CPT** | Uses `gmt makecpt -Cgeo -T${Z_MIN}/${Z_MAX}` instead of relying on built-in colormap names |
| **Hillshading** | `-I+d` adds relief shading for 3D effect |
| **No terrain override** | `gmt coast` has NO `-G` fill; coastlines draw cleanly over terrain |
| **Colorbar** | Specifies `-D` position and `-C topo.cpt` explicitly |
| **Frame** | `gmt basemap` adds axes, ticks, labels, and title **after** terrain is rendered |

### Common Customizations

**Different region:**
```bash
region=100/115/25/40    # Japan & nearby
region=70/100/10/30     # India & Himalayas
```

**Different projection:**
```bash
projection=M20c         # Wider map
projection=L100/25/20/30/12c    # Lambert Conic (better for latitude range)
```

**Add data points (stations, earthquakes):**
```bash
# Before "gmt end show", add:
gmt plot stations.txt -Sc0.2c -Gblue -W0.3p,black
gmt plot earthquakes.txt -Sc0.08c -Gred -W0.1p,darkred
```

**Different color scheme:**
```bash
gmt makecpt -Cturbo -T${Z_MIN}/${Z_MAX} -Z > topo.cpt     # Turbo (viridis-like)
gmt makecpt -Cbilbao -T${Z_MIN}/${Z_MAX} -Z > topo.cpt    # Bilbao (blue→red)
gmt makecpt -Chot -T${Z_MIN}/${Z_MAX} -Z > topo.cpt       # Hot (white→red)
```

---

## Minimal Working Example

```bash
gmt begin test_basemap png

gmt basemap \
    -R100/110/20/30 \
    -JM12c \
    -Bxa2f1+l"Longitude" \
    -Bya2f1+l"Latitude" \
    -BWSen+t"GMT Basemap Example"

gmt coast -W0.5p -Df -N1/0.5p,gray -A1000

gmt end show
```

---

## Common Mistakes

### 1. Missing axis definition

❌

```bash
-B
```

✔

```bash
-Bxa2f1 -Bya2f1 -BWSen
```

---

### 2. Region mismatch

If data fall outside `-R`, nothing is plotted.

Check:

```bash
gmt info data.txt
```

---

### 3. Wrong depth direction

Fix with:

```bash
-JX12c/-6c
```

---

### 4. Wrong projection choice

* Maps → `-JM`
* Profiles → `-JX`

---

## Output Requirement (for agents)

When the user asks to “add basemap”:

* Return a **complete runnable GMT script**
* Auto-detect plot type:

  * Map → `-JM`
  * Section → `-JX`
  * Depth → negative Y
* Use proper labels:

  * Map → Longitude / Latitude
  * Section → Distance / Depth

---

## Note

If you want, I can extend this into a **full GMT seismic plotting skill set** (basemap + coast + fault + beachball + colorbar + legend), which is closer to your actual workflow.

---

## Example Collection

### 0. Plot Points from CSV with Non-Standard Column Names (e.g. lon1/lat1)

When your CSV has coordinate columns named `lon1`, `lat1` (not `lon`, `lat`), extract
them first in Python, write a temp GMT-format file, then call `run_gmt()`.

```python
import pandas as pd, numpy as np, os

# 1. Load CSV — use EXACT column names from FILE CONTEXT
df = pd.read_csv("/path/to/data.csv")
print("columns:", df.columns.tolist()); print(df.head(3))

lon_col = 'lon1'   # ← exact name from FILE CONTEXT (prefer lon1 over lon2)
lat_col = 'lat1'   # ← exact name from FILE CONTEXT (prefer lat1 over lat2)
lon = df[lon_col].values
lat = df[lat_col].values

# ⚠️ Validate BEFORE calling GMT — wrong column causes region error
assert -180 <= lon.min() and lon.max() <= 180, \
    f"lon out of range {lon.min():.2f}~{lon.max():.2f} — column '{lon_col}' is not longitude"
assert  -90 <= lat.min() and lat.max() <=  90, \
    f"lat out of range {lat.min():.2f}~{lat.max():.2f} — column '{lat_col}' is not latitude!"
print(f"lon: {lon.min():.4f}~{lon.max():.4f}  lat: {lat.min():.4f}~{lat.max():.4f}")

# 2. Write data file in Python (NEVER use bash loops in f-string — causes SyntaxError)
pts_file = os.path.join(os.environ.get('SAGE_OUTDIR', '/tmp'), 'points.txt')
np.savetxt(pts_file, np.column_stack([lon, lat]), fmt='%.6f')

# 3. Region as Python string (substituted directly into f-string)
R = f"{lon.min()-1:.2f}/{lon.max()+1:.2f}/{lat.min()-1:.2f}/{lat.max()+1:.2f}"
J = "M15c"
print(f"Region: {R}")

# 4. GMT f-string: Python vars use {R}, bash vars use ${{Z_MIN}}, awk uses {{print $6}}
#    Skip @earth_relief_01m — use 02m→05m to avoid download timeout
script = f"""
gmt begin location_map PNG
  if gmt grdcut @earth_relief_02m -R{R} -Gtopo.grd 2>/dev/null && [ -f topo.grd ]; then
    echo "02m OK"
  elif gmt grdcut @earth_relief_05m -R{R} -Gtopo.grd 2>/dev/null && [ -f topo.grd ]; then
    echo "05m OK"
  else
    echo "Terrain download failed" >&2; exit 1
  fi
  Z_MIN=$(gmt grdinfo topo.grd -C | awk '{{print $6}}')
  Z_MAX=$(gmt grdinfo topo.grd -C | awk '{{print $7}}')
  gmt makecpt -Cgeo -T${{Z_MIN}}/${{Z_MAX}} -Z > topo.cpt
  gmt grdimage topo.grd -J{J} -R{R} -Ctopo.cpt -I+d
  gmt coast -R{R} -J{J} -W0.6p,gray30 -N1/0.8p,gray50 -A500 -Baf -BWSne+t"Location Map"
  gmt plot {pts_file} -R{R} -J{J} -Sc0.2c -Gred -W0.3p,black
  gmt colorbar -DJBC+w7c/0.35c -Ctopo.cpt -Baf+l"Elevation (m)"
gmt end
"""
run_gmt(script, outname="location_map", title="Location map from CSV")
```

---

### 1. Epicenter Distribution Map (Most Common) — with topography

```python
R = "70/140/15/55"
J = "M15c"

gmt_script = f"""
gmt begin seismicity PNG

  # ── Topography first ────────────────────────────────────────────────────
  if gmt grdcut @earth_relief_02m -R{R} -Gtopo.grd 2>/dev/null; then
    echo "02m OK"
  else
    gmt grdcut @earth_relief_05m -R{R} -Gtopo.grd
  fi
  Z_MIN=$(gmt grdinfo topo.grd -C | awk '{{print $6}}')
  Z_MAX=$(gmt grdinfo topo.grd -C | awk '{{print $7}}')
  gmt makecpt -Cgeo -T${{Z_MIN}}/${{Z_MAX}} -Z > topo.cpt
  gmt grdimage topo.grd -J{J} -R{R} -Ctopo.cpt -I+d

  # ── Coast on top (no -G — would hide terrain) ───────────────────────────
  gmt coast -R{R} -J{J} -W0.5p,gray40 -N1/0.8p,gray60 -Slightblue@60 \\
      -Baf -BWSne+t"Seismicity Map"

  # ── Plot epicenters ─────────────────────────────────────────────────────
  # CSV format: lon lat [depth [mag]]
  # gmt plot catalog.csv -i0,1 -Sc0.2c -Gred -W0.3p,black -t30

gmt end
"""

run_gmt(gmt_script, outname="seismicity_map", title="Epicenter distribution map")
```

---

### 2. Station Location Map

```python
gmt_script = """
gmt begin station_map PNG
  gmt basemap -R100/115/25/42 -JM14c -Baf -BWSne+t"Station Map"
  gmt coast -Gwheat -Slightblue -W0.5p -N1/0.8p,gray

  # Plot stations (triangles), data file: lon lat sta_name
  gmt plot stations.txt -i0,1 -St0.5c -Gred -W1p,black

  # Station name annotation
  gmt text stations.txt -i0,1,2 -F+f7p,Helvetica,black+jBL -D0.1c/0.1c

gmt end
"""

run_gmt(gmt_script, outname="station_map", title="Station distribution map")
```

---

### 3. Focal Mechanism Beachballs

```python
gmt_script = """
gmt begin focal_map PNG
  gmt basemap -R95/105/28/38 -JM12c -Baf -BWSne+t"Focal Mechanisms"
  gmt coast -Gtan -Slightblue -W0.5p -N1/0.5p

  # Focal mechanism data: lon lat depth strike dip rake mag [event_name]
  # Use meca to plot (requires GMT seismology module)
  cat << 'EOF' | gmt meca -Sa1c -Gred -W0.5p,black
  100.5 33.2 15 120 60 -90 5.8 1 1
  102.1 30.5 20  45 75  30 4.9 1 1
  103.4 31.8 10  90 45 180 5.2 1 1
EOF

gmt end
"""

run_gmt(gmt_script, outname="focal_map", title="Focal mechanism map")
```

---

### 4. Topographic Map (ETOPO / SRTM) — Robust Version

```python
R = "100/115/25/40"
J = "M14c"

gmt_script = f"""
gmt begin topo_map PNG

  # --- Download relief (fallback resolutions) ---
  if gmt grdcut @earth_relief_01m -R{R} -Gtopo.grd 2>/dev/null; then
    echo "01m OK"
  elif gmt grdcut @earth_relief_02m -R{R} -Gtopo.grd 2>/dev/null; then
    echo "02m OK"
  else
    gmt grdcut @earth_relief_05m -R{R} -Gtopo.grd
  fi

  # REQUIRED: build CPT from actual data range (avoids black-terrain bug)
  Z_MIN=$(gmt grdinfo topo.grd -C | awk '{{print $6}}')
  Z_MAX=$(gmt grdinfo topo.grd -C | awk '{{print $7}}')
  gmt makecpt -Cgeo -T${{Z_MIN}}/${{Z_MAX}} -Z > topo.cpt

  # Render terrain with hillshading
  gmt grdimage topo.grd -J{J} -R{R} -Ctopo.cpt -I+d

  # Coastlines + borders (NO -G fill)
  gmt coast -R{R} -J{J} -W0.6p,gray30 -N1/0.8p,gray50 -A500 \\
      -Bxaf+l"Longitude" -Byaf+l"Latitude" -BWSne+t"Topographic Map"

  gmt colorbar -DJBC+w8c/0.4c+e -Ctopo.cpt -Baf+l"Elevation (m)"

gmt end
"""

run_gmt(gmt_script, outname="topo_map", title="Topographic map")
```

---

### 5. Cross-Section

```python
gmt_script = """
gmt begin cross_section PNG
  # Cross-section parameters: Point A (lon1, lat1) to Point B (lon2, lat2), depth 0-100 km, width ±50 km
  # First extract earthquakes within projection range
  # gmt project catalog.csv -C100/30 -E110/35 -W-50/50 -Fxyzpqrs > proj.txt

  gmt basemap -R0/1200/0/100 -JX15c/-8c -Bxaf+l"Distance (km)" -Byaf+l"Depth (km)" -BWSne+t"Cross Section A-B"

  # Plot projected epicenters (distance, depth)
  # gmt plot proj.txt -i4,2 -Sc0.15c -Cjet -W0.2p

  # Reference line
  gmt plot -W1p,red,- << 'EOF'
0 35
1200 35
EOF

gmt end
"""

run_gmt(gmt_script, outname="cross_section", title="Earthquake cross-section")
```

---

## Complete Chained Example: CSV Catalog to Epicenter Map

```python
import os

# Assume earthquake catalog CSV exists: lon,lat,depth,mag
catalog_file = "/data/catalog/eq_catalog.csv"

gmt_script = """
gmt begin seismicity_final PNG
  gmt basemap -R70/140/15/55 -JM16c -Baf+g245/245/240 -BWSne+t"2020-2024 Seismicity"
  gmt coast -Gtan -Slightblue -W0.5p,gray50 -N1/0.8p,gray70 -A500

  # Small earthquakes (M<4): small gray dots
  awk -F',' '$4<4 {print $1,$2}' """ + catalog_file + """ | gmt plot -Sc0.08c -Ggray60 -t50

  # Moderate earthquakes (4≤M<5): orange dots
  awk -F',' '$4>=4 && $4<5 {print $1,$2}' """ + catalog_file + """ | gmt plot -Sc0.18c -Gorange -W0.3p,black

  # Large earthquakes (M≥5): red dots
  awk -F',' '$4>=5 {print $1,$2,($4*0.15)"c"}' """ + catalog_file + """ | gmt plot -Sc -Gred -W0.5p,black

  gmt colorbar -DJBC+w6c/0.35c -Baf+l"Depth (km)"

gmt end
"""

run_gmt(gmt_script, outname="seismicity_final", title="Epicenter distribution map")
print("Earthquake count: " + str(sum(1 for _ in open(catalog_file)) - 1))
```

---

## Complete Chained Example: Read CSV Data and Plot Locations with GMT + Topography

This is the **canonical template** for any "read CSV → plot locations on map" request.
Always follow this pattern when the user provides a data file with lon/lat columns.

```python
import pandas as pd
import numpy as np
import os, tempfile

# --- 1. Load data and inspect ---
data_file = "/path/to/your/data.csv"   # replace with actual path
df = pd.read_csv(data_file)
print("Columns:", list(df.columns))
print("Shape:", df.shape)

# --- 2. Auto-detect lon/lat columns ---
lon_col = next((c for c in df.columns if 'lon' in c.lower()), df.columns[0])
lat_col = next((c for c in df.columns if 'lat' in c.lower()), df.columns[1])
print(f"Using lon={lon_col}, lat={lat_col}")

lons = pd.concat([df[lon_col]] + ([df['lon2']] if 'lon2' in df.columns else [])).dropna()
lats = pd.concat([df[lat_col]] + ([df['lat2']] if 'lat2' in df.columns else [])).dropna()

# --- 3. Compute map region (with 5% padding) ---
pad = 0.05
lon_min = lons.min(); lon_max = lons.max()
lat_min = lats.min(); lat_max = lats.max()
dlon = max(lon_max - lon_min, 1.0); dlat = max(lat_max - lat_min, 1.0)
R = f"{lon_min - pad*dlon:.2f}/{lon_max + pad*dlon:.2f}/{lat_min - pad*dlat:.2f}/{lat_max + pad*dlat:.2f}"
J = "M15c"
print(f"Region: {R}")

# --- 4. Write point coordinates to temp file ---
tmp = tempfile.mktemp(suffix='.txt')
points = pd.DataFrame({'lon': df[lon_col], 'lat': df[lat_col]}).drop_duplicates()
points.to_csv(tmp, sep=' ', index=False, header=False)
print(f"Plotting {len(points)} unique locations")

# --- 5. GMT script with robust topography ---
gmt_script = f"""
gmt begin locations_map PNG

  # ── Topography (try 01m → 02m → 05m) ──────────────────────────────────
  if gmt grdcut @earth_relief_01m -R{R} -Gtopo.grd 2>/dev/null; then
    echo "terrain 01m loaded"
  elif gmt grdcut @earth_relief_02m -R{R} -Gtopo.grd 2>/dev/null; then
    echo "terrain 02m loaded"
  else
    gmt grdcut @earth_relief_05m -R{R} -Gtopo.grd
    echo "terrain 05m loaded"
  fi

  # Render terrain with hillshading
  Z_MIN=$(gmt grdinfo topo.grd -C | awk '{{print $6}}')
  Z_MAX=$(gmt grdinfo topo.grd -C | awk '{{print $7}}')
  gmt makecpt -Cgeo -T${{Z_MIN}}/${{Z_MAX}} -Z > topo.cpt
  gmt grdimage topo.grd -J{J} -R{R} -Ctopo.cpt -I+d

  # ── Coastlines + borders on top (no -G fill — preserves terrain) ──────
  gmt coast -R{R} -J{J} \\
      -W0.6p,gray30 -N1/0.8p,gray50 -A500 \\
      -Bxaf+l"Longitude (°E)" -Byaf+l"Latitude (°N)" \\
      -BWSne+t"Seismic Station Locations"

  # ── Plot data points ───────────────────────────────────────────────────
  gmt plot {tmp} -R{R} -J{J} -St0.35c -Gred -W0.6p,black

  # ── Elevation color bar ────────────────────────────────────────────────
  gmt colorbar -DJBC+w7c/0.35c+o0/0.5c -Baf+l"Elevation (m)" -F+g255/255/255@40

gmt end
"""

run_gmt(gmt_script, outname="seismic_locations_topo", title="Seismic locations on topographic map")
os.remove(tmp)
print("Map with topography saved.")
```

---

## Complete Chained Example: Ray Path Visualization with Topography

```python
import pandas as pd
import os, tempfile

data_file = "/path/to/your/data.csv"   # replace with actual path
df = pd.read_csv(data_file)

# Auto-detect region from all four endpoint columns
all_lons = list(df['lon1']) + list(df['lon2'])
all_lats = list(df['lat1']) + list(df['lat2'])
pad = 0.5
R = f"{min(all_lons)-pad:.2f}/{max(all_lons)+pad:.2f}/{min(all_lats)-pad:.2f}/{max(all_lats)+pad:.2f}"
J = "M15c"

# Write ray path segments (GMT multi-segment format)
rays_file = tempfile.mktemp(suffix='_rays.txt')
pts_file  = tempfile.mktemp(suffix='_pts.txt')
ray_ids = df['ray_id'].unique()[:500]   # limit to 500 rays for clarity

with open(rays_file, 'w') as fr, open(pts_file, 'w') as fp:
    seen = set()
    for rid in ray_ids:
        sub = df[df['ray_id'] == rid].iloc[0]
        fr.write(f">\n{sub.lon1} {sub.lat1}\n{sub.lon2} {sub.lat2}\n")
        for pt in [(sub.lon1, sub.lat1), (sub.lon2, sub.lat2)]:
            if pt not in seen:
                fp.write(f"{pt[0]} {pt[1]}\n")
                seen.add(pt)

gmt_script = f"""
gmt begin ray_paths PNG

  # ── Topography ──────────────────────────────────────────────────────────
  if gmt grdcut @earth_relief_01m -R{R} -Gtopo.grd 2>/dev/null; then
    echo "01m OK"
  elif gmt grdcut @earth_relief_02m -R{R} -Gtopo.grd 2>/dev/null; then
    echo "02m OK"
  else
    gmt grdcut @earth_relief_05m -R{R} -Gtopo.grd
  fi
  Z_MIN=$(gmt grdinfo topo.grd -C | awk '{{print $6}}')
  Z_MAX=$(gmt grdinfo topo.grd -C | awk '{{print $7}}')
  gmt makecpt -Cgeo -T${{Z_MIN}}/${{Z_MAX}} -Z > topo.cpt
  gmt grdimage topo.grd -J{J} -R{R} -Ctopo.cpt -I+d

  # ── Coast + frame ───────────────────────────────────────────────────────
  gmt coast -R{R} -J{J} -W0.5p,gray30 -N1/0.6p,gray50 -A500 \\
      -Bxaf+l"Longitude" -Byaf+l"Latitude" -BWSne+t"Ray Paths"

  # ── Ray paths (semi-transparent blue) ──────────────────────────────────
  gmt plot {rays_file} -R{R} -J{J} -W0.4p,dodgerblue@60

  # ── Station / source points ─────────────────────────────────────────────
  gmt plot {pts_file} -R{R} -J{J} -Sc0.12c -Gred -W0.4p,black

  gmt colorbar -DJBC+w7c/0.35c+o0/0.5c -Baf+l"Elevation (m)"

gmt end
"""

run_gmt(gmt_script, outname="ray_paths_topo", title="Ray path visualization with topography")
os.remove(rays_file); os.remove(pts_file)
print(f"Plotted {len(ray_ids)} ray paths, {len(seen)} unique endpoints")
```

---

## Projection Rule: Rectangular Map vs Orthographic Projection

If the user asks for a normal rectangular geographic map, DO NOT use orthographic or oblique projections.

Use:

```bash
-JM15c
````

or, for regional maps:

```bash
-JL${lon0}/${lat0}/${lat1}/${lat2}/15c
```

Avoid these projections unless the user explicitly requests a globe-style or oblique map:

```bash
-JG    # Orthographic globe projection
-JO    # Oblique Mercator projection
-JA    # Azimuthal projection
```

Important:

* A normal longitude-latitude regional map should look rectangular.
* If the plotted map appears as a tilted parallelogram or curved globe view, the projection is probably wrong.
* For most seismic station maps, use `-JM` by default.

````

Stable station map template:

```python
gmt_script = f"""
gmt begin station_map PNG

  gmt set MAP_FRAME_TYPE plain
  gmt set FORMAT_GEO_MAP ddd.x

  REGION="-R{lon_min}/{lon_max}/{lat_min}/{lat_max}"
  PROJ="-JM15c"

  # Topography with fallback + explicit CPT (avoids black-terrain bug)
  if gmt grdcut @earth_relief_01m $REGION -Gtopo.grd 2>/dev/null; then
    echo "01m OK"
  elif gmt grdcut @earth_relief_02m $REGION -Gtopo.grd 2>/dev/null; then
    echo "02m OK"
  else
    gmt grdcut @earth_relief_05m $REGION -Gtopo.grd
  fi
  Z_MIN=$(gmt grdinfo topo.grd -C | awk '{{print $6}}')
  Z_MAX=$(gmt grdinfo topo.grd -C | awk '{{print $7}}')
  gmt makecpt -Cgeo -T${{Z_MIN}}/${{Z_MAX}} -Z > topo.cpt
  gmt grdimage topo.grd $REGION $PROJ -Ctopo.cpt -I+d

  # Coastline and frame (no -G fill)
  gmt coast $REGION $PROJ -W0.5p,gray40 -N1/0.5p,gray60 -A500 \\
      -Bxa2f1+l"Longitude" \\
      -Bya2f1+l"Latitude" \\
      -BWSne+t"Station Map"

  # Plot stations
  gmt plot stations.txt $REGION $PROJ -St0.25c -Gred -W0.4p,black

  gmt colorbar -DJBC+w6c/0.3c+o0/0.4c -Ctopo.cpt -Baf+l"Elevation (m)"

gmt end
"""
````

---

## Common GMT6 Command Reference

| Command | Purpose |
|---------|---------|
| `gmt basemap` | Draw base map frame and axes |
| `gmt coast` | Coastlines, national borders, terrain fill |
| `gmt plot` | Plot points, lines, polygons |
| `gmt text` | Text annotation |
| `gmt grdimage` | Raster data rendering (topography, velocity models) |
| `gmt colorbar` | Color bar |
| `gmt meca` | Focal mechanism beachball |
| `gmt coupe` | Focal mechanism cross-section |
| `gmt project` | Coordinate projection (for cross-sections) |
| `gmt surface` | Data interpolation to grid |
| `gmt contour` | Contour lines |

---

## Notes

- GMT6 modern mode: `gmt begin <name> PNG` ... `gmt end` (**recommended**)
- Script execution directory is `SAGE_OUTDIR` (temporary directory); data files must use **absolute paths**
- `run_gmt` automatically replaces the file name after `gmt begin` with `outname`
- Output `.sh` script can be re-run directly in terminal, fully reproducible
