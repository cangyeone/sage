---
name: gmt_plotting
category: visualization
keywords: GMT, map, seismicity, epicenter distribution, station distribution, fault, topography, focal mechanism, coastline, contour, cross-section, travel time, earthquake catalog, basemap, coast, pscoast, meca, coupe, grdimage, gmt begin, gmt end, run_gmt, 地图, 震中分布, 台站分布, 地形, 震源机制, 海岸线
---

# GMT Map Plotting

## Description

Use GMT (Generic Mapping Tools) to create professional seismology maps: epicenter distribution maps, station location maps, topographic maps, focal mechanism beachballs, cross-sections, etc.

---

## ⚠️ Critical Note

- `run_gmt(script, outname, title)` is pre-injected; **call directly, no import needed**
- `script` parameter is a **complete GMT6 bash script string** (multi-line string)
- Script must use `gmt begin <name> PNG` ... `gmt end` structure (GMT6 modern mode)
- **Write Chinese/non-ASCII titles and labels directly in the script** (e.g., `-BWSne+t"中国地形图"`); `run_gmt` auto-handles CJK characters without conversion needed
- After execution, automatically generates **PNG image** and **.sh script file**, both downloadable from interface
- Requires GMT >= 6.0 installed on system

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

### 1. Epicenter Distribution Map (Most Common)

```python
gmt_script = """
gmt begin seismicity PNG
  # Base map: China and surrounding regions
  gmt basemap -R70/140/15/55 -JM15c -Baf -BWSne+t"Seismicity Map"

  # Coastlines + national boundaries
  gmt coast -Gtan -Slightblue -W0.5p,gray40 -N1/0.8p,gray60

  # Read earthquake catalog from CSV and plot epicenters (lon, lat, depth, mag)
  # CSV format: lon lat depth mag
  gmt plot catalog.csv -i0,1 -Sc0.2c -Cjet -W0.3p,black -t30

  # Legend
  gmt legend -DjBL+w4c+o0.2c -F+g255/255/255@30 << 'EOF'
G 0.1c
H 9p,Helvetica-Bold Epicenter Distribution
D 0.1c 1p
S 0.2c c 0.3c red 0.5p,black 0.4c M<=4
S 0.2c c 0.4c orange 0.5p,black 0.4c 4<M<=5
S 0.2c c 0.5c blue 0.5p,black 0.4c M>5
EOF

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

### 4. Topographic Map (ETOPO / SRTM)

```python
gmt_script = """
gmt begin topo_map PNG
  # Download and use ETOPO topographic data (requires internet)
  gmt grdcut @earth_relief_01m -R100/115/25/40 -Gtopo.grd

  # Topography rendering
  gmt grdimage topo.grd -JM14c -Cetopo1 -I+d

  # Overlay coastlines and boundaries
  gmt coast -W0.5p,gray40 -N1/0.8p,gray60 -Baf -BWSne+t"Topography"

  # Color bar
  gmt colorbar -DJBC+w8c/0.4c+e -Baf+l"Elevation (m)"

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

## Complete Chained Example: Read CSV Data and Plot Locations with GMT

```python
import pandas as pd
import os

# Read CSV data file
data_file = "/Users/yuziye/Documents/GitHub/sage/data/seismic/waveform/data.csv"
df = pd.read_csv(data_file)

# Extract unique station locations (lon1,lat1 and lon2,lat2 pairs)
stations = []
for _, row in df.iterrows():
    stations.append((row['lon1'], row['lat1']))
    stations.append((row['lon2'], row['lat2']))

# Remove duplicates
unique_stations = list(set(stations))

# Create temporary file for GMT plotting
temp_file = "/tmp/stations.txt"
with open(temp_file, 'w') as f:
    for lon, lat in unique_stations:
        f.write(f"{lon} {lat}\n")

# Create GMT script for station map
gmt_script = """
gmt begin station_locations PNG
  # Set map region to cover China and surrounding areas
  gmt basemap -R100/125/35/45 -JM15c -Baf -BWSne+t"Station Locations"

  # Add topography
  gmt grdcut @earth_relief_01m -R100/125/35/45 -Gtopo.grd
  gmt grdimage topo.grd -Cetopo1 -I+d

  # Coastlines and political boundaries
  gmt coast -W0.5p,gray40 -N1/0.8p,gray60 -A500

  # Plot station locations as red triangles
  gmt plot """ + temp_file + """ -St0.3c -Gred -W0.5p,black

  # Add station labels (optional)
  awk '{print $1,$2,NR}' """ + temp_file + """ | gmt text -F+f6p,Helvetica-Bold,white+jBL -Gblack@50 -D0.1c/0.1c

gmt end
"""

# Execute GMT script
run_gmt(gmt_script, outname="station_locations", title="Station location map")

# Clean up temporary file
os.remove(temp_file)
```

---

## Complete Chained Example: Ray Path Visualization

```python
import pandas as pd
import os

# Read CSV data file
data_file = "/Users/yuziye/Documents/GitHub/sage/data/seismic/waveform/data.csv"
df = pd.read_csv(data_file)

# Create temporary file for ray paths
temp_file = "/tmp/ray_paths.txt"
with open(temp_file, 'w') as f:
    for _, row in df.iterrows():
        # Write great circle path points (simplified as straight lines for visualization)
        f.write(f">{row['ray_id']}\\n")
        f.write(f"{row['lon1']} {row['lat1']}\\n")
        f.write(f"{row['lon2']} {row['lat2']}\\n")

# Create GMT script for ray path visualization
gmt_script = """
gmt begin ray_paths PNG
  # Set map region
  gmt basemap -R100/125/35/45 -JM15c -Baf -BWSne+t"Ray Paths"

  # Add topography
  gmt grdcut @earth_relief_01m -R100/125/35/45 -Gtopo.grd
  gmt grdimage topo.grd -Cetopo1 -I+d

  # Coastlines
  gmt coast -W0.5p,gray40 -N1/0.8p,gray60 -A500

  # Plot ray paths as blue lines
  gmt plot """ + temp_file + """ -W1p,blue -Gblue@20

  # Plot station locations
  awk 'NR>1 && NF==2 {print $1,$2}' """ + temp_file + """ | gmt plot -Sc0.15c -Gred -W0.5p,black

gmt end
"""

# Execute GMT script
run_gmt(gmt_script, outname="ray_paths", title="Ray path visualization")

# Clean up temporary file
os.remove(temp_file)
```

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
