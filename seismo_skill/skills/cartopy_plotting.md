---
name: cartopy_plotting
category: visualization
keywords: cartopy, matplotlib, map, seismicity map, epicenter, scatter map, location map, geographic plot, station map, cross-section, depth section, magnitude, colormap, 地图, 震中分布, 位置图, 台站分布, 剖面图, 散点图, 经纬度绘图, 地震分布, 绘图, plot map, scatter plot
---

# Cartopy / Matplotlib Plotting Skills

**Default choice for all map/geographic plots.**  
Use cartopy unless the user explicitly asks for GMT.

---

## 1. Basic earthquake location map (scatter)

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

# --- lon, lat, depth (and optionally magnitude) are numpy arrays ---
pad = max((lon.max()-lon.min())*0.1, (lat.max()-lat.min())*0.1, 0.5)
extent = [lon.min()-pad, lon.max()+pad, lat.min()-pad, lat.max()+pad]

fig, ax = plt.subplots(figsize=(10, 8),
                        subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_extent(extent, crs=ccrs.PlateCarree())

# Background features
ax.add_feature(cfeature.LAND,      facecolor='#f0ede5', zorder=0)
ax.add_feature(cfeature.OCEAN,     facecolor='#d6eaf8', zorder=0)
ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='#444', zorder=2)
ax.add_feature(cfeature.BORDERS,   linewidth=0.5, linestyle=':', color='#777', zorder=2)
ax.add_feature(cfeature.RIVERS,    linewidth=0.3, color='#90caf9', zorder=1)

# Grid labels
gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False

# Plot earthquakes coloured by depth
sc = ax.scatter(lon, lat, c=depth, cmap='plasma_r', s=20,
                transform=ccrs.PlateCarree(), zorder=5,
                alpha=0.85, edgecolors='none',
                vmin=0, vmax=depth.max() if depth.max() > 0 else 1)
plt.colorbar(sc, ax=ax, label='Depth (km)', shrink=0.65, pad=0.02)
ax.set_title('Seismicity Map', fontsize=13, pad=10)

savefig('seismicity_map.png')
plt.close()
```

---

## 2. Magnitude-scaled scatter (size ∝ magnitude)

```python
# mag is a numpy array
sizes = (2 ** mag) * 3          # exponential scaling — looks natural
sc = ax.scatter(lon, lat, c=depth, cmap='plasma_r',
                s=sizes, transform=ccrs.PlateCarree(),
                alpha=0.7, edgecolors='black', linewidths=0.3, zorder=5)
```

---

## 3. Station + event map (two symbol types)

```python
# Events
ax.scatter(ev_lon, ev_lat, c='red', s=20, marker='o',
           transform=ccrs.PlateCarree(), zorder=5, label='Events')
# Stations
ax.scatter(st_lon, st_lat, c='blue', s=60, marker='^',
           transform=ccrs.PlateCarree(), zorder=6, label='Stations')
ax.legend(loc='lower right', fontsize=9)
```

---

## 4. Coloured by time (seismicity evolution)

```python
import matplotlib.dates as mdates
import pandas as pd

# times is a list of datetime or timestamp strings → convert to float for colourmap
t_num = mdates.date2num(pd.to_datetime(times))
sc = ax.scatter(lon, lat, c=t_num, cmap='viridis',
                s=15, transform=ccrs.PlateCarree(), zorder=5, alpha=0.8)
cb = plt.colorbar(sc, ax=ax, shrink=0.65, pad=0.02, label='Time')
cb.ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
```

---

## 5. SRTM-like hill shading with cartopy (no internet required)

```python
# Uses NaturalEarth built-in shaded relief at ~10m resolution
ax.stock_img()                  # NASA Blue Marble background (requires cartopy data)
# OR use a simple terrain colour from OSM tiles (requires internet):
# import cartopy.io.img_tiles as cimgt
# ax.add_image(cimgt.Stamen('terrain-background'), 8)  # zoom level 8
```

---

## 6. Add fault lines from a file (lon/lat pairs separated by >)

```python
import numpy as np

# fault_file has lon lat rows; segments separated by blank lines or '>'
faults = np.loadtxt(fault_file, comments='>')
# OR read segment by segment:
with open(fault_file) as f:
    seg, segs = [], []
    for line in f:
        line = line.strip()
        if not line or line.startswith('>'):
            if seg:
                segs.append(np.array(seg))
                seg = []
        else:
            seg.append([float(x) for x in line.split()])
    if seg:
        segs.append(np.array(seg))

for s in segs:
    ax.plot(s[:, 0], s[:, 1], 'k-', linewidth=0.5,
            transform=ccrs.PlateCarree(), zorder=3)
```

---

## 7. Cross-section: depth vs epicentral distance

```python
from obspy.geodetics import gps2dist_azimuth

# reference point
ref_lon, ref_lat = 110.0, 35.0
dist_km = np.array([gps2dist_azimuth(ref_lat, ref_lon, la, lo)[0]/1000
                    for lo, la in zip(lon, lat)])

fig, ax = plt.subplots(figsize=(10, 5))
sc = ax.scatter(dist_km, depth, c=mag, cmap='hot_r', s=20, alpha=0.7,
                edgecolors='none', vmin=mag.min(), vmax=mag.max())
plt.colorbar(sc, ax=ax, label='Magnitude', shrink=0.8)
ax.set_xlabel('Distance (km)')
ax.set_ylabel('Depth (km)')
ax.invert_yaxis()
ax.set_title('Vertical Cross-Section')
ax.grid(True, linewidth=0.3, alpha=0.5)
savefig('cross_section.png')
plt.close()
```

---

## 8. Multi-panel: map + cross-section side by side

```python
fig = plt.figure(figsize=(14, 6))
ax_map = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
ax_xsec = fig.add_subplot(1, 2, 2)

# --- Map panel ---
ax_map.set_extent(extent, crs=ccrs.PlateCarree())
ax_map.add_feature(cfeature.LAND,      facecolor='#f0ede5')
ax_map.add_feature(cfeature.OCEAN,     facecolor='#d6eaf8')
ax_map.add_feature(cfeature.COASTLINE, linewidth=0.7)
sc1 = ax_map.scatter(lon, lat, c=depth, cmap='plasma_r', s=15,
                      transform=ccrs.PlateCarree(), alpha=0.8)
fig.colorbar(sc1, ax=ax_map, label='Depth (km)', shrink=0.6)
ax_map.set_title('Epicentre Map', fontsize=11)

# --- Cross-section panel ---
ax_xsec.scatter(dist_km, depth, c=mag, cmap='hot_r', s=15, alpha=0.7)
ax_xsec.invert_yaxis()
ax_xsec.set_xlabel('Distance (km)')
ax_xsec.set_ylabel('Depth (km)')
ax_xsec.set_title('Cross-Section', fontsize=11)
ax_xsec.grid(True, linewidth=0.3, alpha=0.4)

plt.tight_layout()
savefig('map_and_xsec.png')
plt.close()
```

---

## 9. Projection variants

```python
# Lambert Conformal — good for mid-latitude regions
ccrs.LambertConformal(central_longitude=lon.mean(), central_latitude=lat.mean())

# Albers Equal Area — good for large countries
ccrs.AlbersEqualArea(central_longitude=lon.mean(), standard_parallels=(25, 47))

# PlateCarree — simple, global or regional
ccrs.PlateCarree()

# Mercator — avoid for high latitudes
ccrs.Mercator()
```

---

## Common mistakes to avoid

| Wrong | Correct |
|-------|---------|
| `plt.show()` | `savefig('out.png')` — NEVER call show() |
| `import matplotlib.use('Agg')` | `import matplotlib; matplotlib.use('Agg')` |
| Forget `transform=ccrs.PlateCarree()` in scatter/plot | Always pass `transform=ccrs.PlateCarree()` |
| `ax.set_xlim(...)` on a GeoAxes | `ax.set_extent([w,e,s,n])` instead |
| `ax.xlabel(...)` on a GeoAxes | `ax.set_xlabel(...)` or use `fig.text()` |

---

## Tip: saving multiple figures

```python
for i, (title, data) in enumerate(panels):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(data['x'], data['y'])
    ax.set_title(title)
    savefig(f'panel_{i:02d}.png')   # each call registers one [FIGURE]
    plt.close()
```
