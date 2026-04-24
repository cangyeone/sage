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
