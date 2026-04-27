---
name: gmt_plotting
category: visualization
keywords: GMT, map, seismicity, epicenter distribution, station distribution, fault, topography, terrain, terrain background, 地形底图, focal mechanism, coastline, contour, cross-section, travel time, earthquake catalog, basemap, coast, pscoast, meca, coupe, grdimage, gmt begin, gmt end, run_gmt, 地图, 震中分布, 台站分布, 地形, 地形底图, 震源机制, 海岸线
related_skills: _gen_gmt_docs_6_5
workflow: gmt_terrain_map
---

# GMT Map Plotting

## Description

Use GMT (Generic Mapping Tools) to create professional seismology maps: epicenter distribution maps, station location maps, topographic maps, focal mechanism beachballs, cross-sections, etc.

> If the user explicitly requests `地形底图` or `terrain background`, always include a terrain grid (`gmt grdimage`) before coastlines and plot layers.

---

## ⚠️ Critical Rules

- **Output a `bash` code block** — the executor runs it directly, no Python wrapper needed
- Script must `cd "${SAGE_OUTDIR}"` at the top so output files land in the right place
- Script must use `gmt begin <name> PNG` ... `gmt end` (GMT6 modern mode)
- For tasks that need Python to prepare data first, call the pre-injected `run_gmt(script_str, outname)` from within Python code
- Requires GMT >= 6.0 installed on the system

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
# --- Download relief grid (skip 01m — too slow; chain 02m→05m) ---
# Note: always check the file was actually created (grdcut may exit 0 without output)
if gmt grdcut @earth_relief_02m -R${R} -Gtopo.grd 2>/dev/null && [ -f topo.grd ]; then
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

## Core Pattern

### Pure bash output (preferred for all GMT-only tasks)

Output a ` ```bash ` code block.  The engine's `execute_bash()` runs it with
`SAGE_OUTDIR` set to the temp work directory.

```bash
#!/bin/bash
cd "${SAGE_OUTDIR}"      # all output files go here automatically

R="73/136/18/54"        # west/east/south/north
J="M15c"                # Mercator, 15 cm wide

# ... your GMT commands ...

gmt begin mymap PNG
  gmt grdimage topo.grd -J${J} -R${R} -Ctopo.cpt -I+d
  gmt coast -R${R} -J${J} -W0.6p,gray30 -N1/0.8p,gray50 -A500 -Baf
gmt end
```

Normal bash syntax throughout — `${VAR}`, `$(cmd)`, `awk '{print $6}'` — no escaping needed.

### Mixed Python + GMT (when CSV data prep is required)

Call the pre-injected `run_gmt(script_str, outname)` from Python code:

```python
import numpy as np, os

pts_file = os.path.join(os.environ.get('SAGE_OUTDIR', '/tmp'), 'points.txt')
np.savetxt(pts_file, np.column_stack([lon, lat]), fmt='%.6f')

R = f"{lon.min()-1:.2f}/{lon.max()+1:.2f}/{lat.min()-1:.2f}/{lat.max()+1:.2f}"
J = "M15c"

# Inside f-string: Python vars → {R}, bash vars → ${{Z_MIN}}, awk → {{print $6}}
script = f"""
cd "${{SAGE_OUTDIR}}"
gmt begin mymap PNG
  gmt plot {pts_file} -R{R} -J{J} -Sc0.2c -Gred -W0.3p,black -Baf
gmt end
"""
run_gmt(script, outname="mymap")
```

---

## Legend / 图例

Use `gmt legend` to draw a boxed legend after all data layers. Never use `gmt basemap` for legend placement.

### Quick rule of thumb

| Want to plot | Legend spec line |
|---|---|
| Circle point | `S 0.3c c <fill> <pen> Label` |
| Triangle (station) | `S 0.3c t <fill> <pen> Label` |
| Square | `S 0.3c s <fill> <pen> Label` |
| Line | `S 0.5c - - <pen> Label` |
| Dashed line | `S 0.5c - - <pen>,- Label` |

### Legend spec file format — all directives

```
H <fontsize> [<font>] <heading>       ← bold title inside legend box
D [gap] <pen>                          ← horizontal divider line (gap optional e.g. 0.1c)
N <ncols>                              ← number of columns for following S lines
S [dx1] <symbol> <size> <fill> <pen> [dx2] <label>
G <gap>                                ← extra vertical space (e.g. 0.1c)
T <text>                               ← plain text line
L <size> <justification> <text>        ← left/center/right-justified text (L/C/R)
```

`S` line field widths:
```
S  dx1  symbol  size  fill     pen        [dx2]  label
S  0.3c   c     0.2c   red   0.5p,black         Earthquake
```
- `dx1`: horizontal offset of symbol from left edge of box (default 0.2c is fine)
- symbol letters: `c` circle, `t` triangle, `s` square, `d` diamond, `i` inverted-triangle, `-` line, `f` fault, `v` vector
- `fill` and `pen` must match the `gmt plot` command exactly; use `-` for "none"
- `dx2` (optional): gap between symbol and label; omit to use default

### Placement `-D` options

```bash
-DjBR+w5c+o0.2c/0.2c    # bottom-right, box width 5c, 0.2c offset from edge
-DjBL+w5c+o0.2c/0.2c    # bottom-left
-DjTR+w5c+o0.2c/0.2c    # top-right
-DjBC+w8c+o0/0.5c        # bottom-center (e.g. above the colorbar)
-DjTL+w5c+o0.2c/0.2c    # top-left
```

### Box background `-F` options

Always use `-F` on terrain maps so the legend is readable:
```bash
-F+p0.8p,black+gwhite          # white background, black border
-F+p0.8p,gray30+gwhite@20      # semi-transparent white
-F+p0.5p,gray50+glightgray     # gray background
```

---

### Example 1 — Station-only legend

```bash
gmt plot stations.txt -St0.3c -Gred -W0.5p,black

cat > legend.txt << 'EOF'
H 11 Legend
D 0.1c 0.5p
S 0.2c t 0.3c red 0.5p,black 0.3c Seismic station
EOF

gmt legend legend.txt -DjBR+w4.5c+o0.2c/0.2c -F+p0.8p,black+gwhite
```

---

### Example 2 — Multi-layer legend: stations + earthquakes + faults

```bash
gmt plot faults.txt    -W1.2p,firebrick
gmt plot earthquakes.txt -Sc0.2c -Gblue   -W0.3p,black
gmt plot stations.txt    -St0.3c -Gred    -W0.5p,black

cat > legend.txt << 'EOF'
H 11 Legend
D 0.1c 0.5p
S 0.5c - - - 1.2p,firebrick 0.3c Fault
S 0.2c c 0.2c blue 0.3p,black 0.3c Earthquake
S 0.2c t 0.3c red 0.5p,black  0.3c Station
EOF

gmt legend legend.txt -DjBR+w5.5c+o0.2c/0.2c -F+p0.8p,black+gwhite
```

---

### Example 3 — Magnitude-scaled earthquake legend

When earthquake circles are scaled by magnitude (`-Sc` with variable size), use several
representative rows to communicate the scale:

```bash
# Variable-size circles: size = mag * 0.08c
awk '{print $1,$2,$3*0.08"c"}' catalog.txt | gmt plot -Sc -Gblue@40 -W0.3p,gray30

cat > legend.txt << 'EOF'
H 11 Magnitude
D 0.1c 0.5p
S 0.25c c 0.24c blue@40 0.3p,gray30 0.4c M 3
S 0.25c c 0.40c blue@40 0.3p,gray30 0.4c M 5
S 0.25c c 0.56c blue@40 0.3p,gray30 0.4c M 7
EOF

gmt legend legend.txt -DjBR+w3.8c+o0.2c/0.2c -F+p0.8p,black+gwhite
```

---

### Example 4 — Focal mechanism legend (gmt meca)

`gmt meca` symbols use type letter matching the `-S` flag: `a` = Aki-Richards, `d` = double-couple.

```bash
gmt meca focal.txt -Sa0.5c -Gred -W0.5p,black

cat > legend.txt << 'EOF'
H 11 Focal mechanism
D 0.1c 0.5p
S 0.3c a 0.5c red 0.5p,black 0.3c Focal mechanism (Mw≥4)
EOF

gmt legend legend.txt -DjTR+w5c+o0.2c/0.2c -F+p0.8p,black+gwhite
```

---

### Example 5 — Depth-colored earthquakes with separate legend + colorbar

```bash
# Color by depth using a CPT
gmt makecpt -Cjet -T0/100/10 > depth.cpt
awk '{print $1,$2,$3}' catalog.txt | gmt plot -Sc0.15c -Cdepth.cpt -W0.2p,gray30

# Colorbar for depth
gmt colorbar -DJBC+w7c/0.35c+o0/0.5c -Cdepth.cpt -Baf+l"Depth (km)"

# Separate legend for symbol type
cat > legend.txt << 'EOF'
H 11 Data
D 0.1c 0.5p
S 0.2c c 0.15c gray50 0.2p,gray30 0.3c Earthquake (color = depth)
EOF

gmt legend legend.txt -DjTR+w5c+o0.2c/0.2c -F+p0.8p,black+gwhite
```

---

### f-string safe legend patterns (Python-generated scripts)

When the GMT script is built inside a Python f-string, use `printf` + pipe or a heredoc
with `'EOF'` (single-quoted = no variable expansion). Both are safe:

```python
gmt_script = f"""
gmt begin mymap PNG

  # ... terrain, coast, plot layers ...

  gmt plot {pts_file} -R{R} -J{J} -Sc0.2c -Gblue -W0.3p,black

  printf 'H 11 Legend\\nD 0.1c 0.5p\\nS 0.2c c 0.2c blue 0.3p,black 0.3c Data point\\n' \\
    | gmt legend -DjBR+w4.5c+o0.2c/0.2c -F+p0.8p,black+gwhite

gmt end
"""
```

Or, using a single-quoted heredoc (no `${{}}` escaping needed):
```python
gmt_script = f"""
gmt begin mymap PNG

  gmt plot {pts_file} -R{R} -J{J} -St0.3c -Gred -W0.5p,black

  cat > legend.txt << 'EOF'
H 11 Legend
D 0.1c 0.5p
S 0.2c t 0.3c red 0.5p,black 0.3c Seismic station
EOF
  gmt legend legend.txt -DjBR+w4.5c+o0.2c/0.2c -F+p0.8p,black+gwhite

gmt end
"""
```

> **Rule**: inside an f-string, use `'EOF'` (single-quoted) for the legend heredoc — this prevents the shell from expanding any `$` inside the legend spec, and you don't need to escape anything.

---

### Complete working example: terrain map with stations + earthquakes + legend

```bash
#!/bin/bash
cd "${SAGE_OUTDIR}"

STATIONS="/path/to/stations.txt"   # lon lat name
CATALOG="/path/to/catalog.txt"     # lon lat depth mag

R="95/115/25/42"
J="M15c"

# ── Topography ──────────────────────────────────────────────────────────
if gmt grdcut @earth_relief_02m -R${R} -Gtopo.grd 2>/dev/null && [ -f topo.grd ]; then
  echo "02m OK"
else
  gmt grdcut @earth_relief_05m -R${R} -Gtopo.grd
fi
Z_MIN=$(gmt grdinfo topo.grd -C | awk '{print $6}')
Z_MAX=$(gmt grdinfo topo.grd -C | awk '{print $7}')
gmt makecpt -Cgeo -T${Z_MIN}/${Z_MAX} -Z > topo.cpt

gmt begin seismic_map PNG

  gmt grdimage topo.grd -J${J} -R${R} -Ctopo.cpt -I+d

  # ── Coast + frame ────────────────────────────────────────────────────
  gmt coast -R${R} -J${J} -W0.6p,gray30 -N1/0.8p,gray50 -A500 \
      -Bxaf+l"Longitude (E)" -Byaf+l"Latitude (N)" -BWSne+t"Seismicity Map"

  # ── Data layers ──────────────────────────────────────────────────────
  gmt plot ${CATALOG}   -R${R} -J${J} -Sc0.15c -Gblue@40 -W0.3p,gray20
  gmt plot ${STATIONS}  -R${R} -J${J} -St0.35c -Gred     -W0.6p,black

  # ── Elevation colorbar ───────────────────────────────────────────────
  gmt colorbar -DJBC+w7c/0.35c+o0/0.5c -Ctopo.cpt -Baf+l"Elevation (m)"

  # ── Legend ───────────────────────────────────────────────────────────
  cat > legend.txt << 'EOF'
H 12 Legend
D 0.1c 0.5p
S 0.2c c 0.15c blue@40 0.3p,gray20 0.3c Earthquake
S 0.2c t 0.35c red    0.6p,black   0.3c Station
EOF
  gmt legend legend.txt -DjBR+w5c+o0.3c/0.3c -F+p0.8p,black+gwhite

gmt end
```

---

### Common legend mistakes

- ❌ Placing `gmt legend` **before** `gmt grdimage` or before data layers — legend gets buried
- ❌ Missing `-F` — legend text invisible on top of terrain
- ❌ Symbol letter in spec doesn't match `gmt plot -S` flag (e.g. plotting `-St` but writing `c` in spec)
- ❌ Fill/pen in legend spec doesn't match the plot command — use copy-paste to avoid mismatch
- ❌ Using `<< EOF` (unquoted) inside f-string — shell expands `$` in spec lines, causing errors
- ❌ Legend box too narrow (`+w` too small) — label text gets clipped; use `+w5c` or wider as default

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

**Color and legend example:**
```bash
gmt plot stations.txt -Sc0.25c -Gblue -W0.3p,black
cat > legend.txt << EOF
S 0.25c c blue 0.3p,black Seismic station
EOF

gmt legend legend.txt -DjTR+w4c/1.5c+o0.2c/0.2c -F+p0.8p,black+gwhite
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

```bash
#!/bin/bash
cd "${SAGE_OUTDIR}"

R="70/140/15/55"
J="M15c"

if gmt grdcut @earth_relief_02m -R${R} -Gtopo.grd 2>/dev/null && [ -f topo.grd ]; then
  echo "02m OK"
else
  gmt grdcut @earth_relief_05m -R${R} -Gtopo.grd
fi
Z_MIN=$(gmt grdinfo topo.grd -C | awk '{print $6}')
Z_MAX=$(gmt grdinfo topo.grd -C | awk '{print $7}')
gmt makecpt -Cgeo -T${Z_MIN}/${Z_MAX} -Z > topo.cpt

gmt begin seismicity PNG
  gmt grdimage topo.grd -J${J} -R${R} -Ctopo.cpt -I+d
  gmt coast -R${R} -J${J} -W0.5p,gray40 -N1/0.8p,gray60 -A500 \
      -Baf -BWSne+t"Seismicity Map"
  # gmt plot catalog.txt -i0,1 -Sc0.2c -Gred -W0.3p,black -t30
gmt end
```

---

### 2. Station Location Map

```bash
#!/bin/bash
cd "${SAGE_OUTDIR}"

gmt begin station_map PNG
  gmt basemap -R100/115/25/42 -JM14c -Baf -BWSne+t"Station Map"
  gmt coast -Gwheat -Slightblue -W0.5p -N1/0.8p,gray
  # Plot stations (triangles): data file lon lat sta_name
  gmt plot stations.txt -i0,1 -St0.5c -Gred -W1p,black
  gmt text stations.txt -i0,1,2 -F+f7p,Helvetica,black+jBL -D0.1c/0.1c
gmt end
```

---

### 3. Focal Mechanism Beachballs

```bash
#!/bin/bash
cd "${SAGE_OUTDIR}"

gmt begin focal_map PNG
  gmt basemap -R95/105/28/38 -JM12c -Baf -BWSne+t"Focal Mechanisms"
  gmt coast -Gtan -Slightblue -W0.5p -N1/0.5p
  cat << 'EOF' | gmt meca -Sa1c -Gred -W0.5p,black
  100.5 33.2 15 120 60 -90 5.8 1 1
  102.1 30.5 20  45 75  30 4.9 1 1
  103.4 31.8 10  90 45 180 5.2 1 1
EOF
gmt end
```

---

### 4. Topographic Map (ETOPO / SRTM) — Robust Version

```bash
#!/bin/bash
cd "${SAGE_OUTDIR}"

R="100/115/25/40"
J="M14c"

if gmt grdcut @earth_relief_02m -R${R} -Gtopo.grd 2>/dev/null && [ -f topo.grd ]; then
  echo "02m OK"
else
  gmt grdcut @earth_relief_05m -R${R} -Gtopo.grd
fi
Z_MIN=$(gmt grdinfo topo.grd -C | awk '{print $6}')
Z_MAX=$(gmt grdinfo topo.grd -C | awk '{print $7}')
gmt makecpt -Cgeo -T${Z_MIN}/${Z_MAX} -Z > topo.cpt

gmt begin topo_map PNG
  gmt grdimage topo.grd -J${J} -R${R} -Ctopo.cpt -I+d
  gmt coast -R${R} -J${J} -W0.6p,gray30 -N1/0.8p,gray50 -A500 \
      -Bxaf+l"Longitude" -Byaf+l"Latitude" -BWSne+t"Topographic Map"
  gmt colorbar -DJBC+w8c/0.4c+e -Ctopo.cpt -Baf+l"Elevation (m)"
gmt end
```

---

### 5. Cross-Section

```bash
#!/bin/bash
cd "${SAGE_OUTDIR}"

# gmt project catalog.csv -C100/30 -E110/35 -W-50/50 -Fxyzpqrs > proj.txt

gmt begin cross_section PNG
  gmt basemap -R0/1200/0/100 -JX15c/-8c \
      -Bxaf+l"Distance (km)" -Byaf+l"Depth (km)" -BWSne+t"Cross Section A-B"
  # gmt plot proj.txt -i4,2 -Sc0.15c -Cjet -W0.2p
  printf '0 35\n1200 35\n' | gmt plot -W1p,red,-
gmt end
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
