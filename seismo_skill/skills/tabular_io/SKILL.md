---
name: tabular_io
category: data
keywords: csv, txt, pandas, read_csv, read_table, load_catalog_file, 读取CSV, 读取TXT, 文本数据, 数据读取, 站点列表, 目录
---

# Tabular Data Reading (CSV / TXT)

## Description

Read general text-table data files, station lists, and earthquake catalogs.
This skill covers robust CSV/TXT parsing and common patterns for data inspection.

---

## Recommended Functions

### `load_catalog_file(path)`

Use this repository helper when the file contains an earthquake catalog or SAGE pick format.
Supports `.csv`, `.json`, and `.txt` catalog formats.

```python
from seismo_stats.catalog_loader import load_catalog_file
catalog = load_catalog_file("/data/seismic/waveform/data.csv")
print(catalog.df.head())
```

### `pandas.read_csv(path, sep=None, engine='python')`

For general CSV or whitespace-delimited `.txt` tables, use pandas with automatic delimiter detection:

```python
import pandas as pd
path = "/data/seismic/waveform/data.csv"
try:
    df = pd.read_csv(path, sep=None, engine='python')
except Exception:
    df = pd.read_csv(path, delim_whitespace=True, header=None)
print(df.columns)
print(df.head())

# If the file schema is unclear, inspect the file and infer coordinates
if 'latitude' not in df.columns and 'longitude' not in df.columns:
    print('Sample rows:')
    print(df.head(10))
    print('Column names:', df.columns.tolist())
    # Try common coordinate fields
    candidates = [c for c in df.columns if c.lower() in ('lon','lat','lon1','lat1','lon2','lat2','longitude','latitude','x','y')]
    print('Possible coordinate fields:', candidates)
```

### `pd.read_table(path, sep='\t')`

Use when the file is tab-delimited:

```python
import pandas as pd
path = "/data/seismic/waveform/stations.txt"
df = pd.read_table(path)
print(df.head())
```

### `numpy.loadtxt(path, delimiter=',')`

For simple numeric tables without headers:

```python
import numpy as np
path = "/data/seismic/waveform/data.txt"
arr = np.loadtxt(path, delimiter=',')
print(arr.shape)
print(arr[:5])
```

---

## Tips for robust reading

- If the file has no header row, use `header=None` and assign `names=[...]`.
- If delimiter is unknown, try `sep=None, engine='python'` or `delim_whitespace=True`.
- For `.txt` files, whitespace-delimited data is common.
- If parsing fails, inspect the first few lines with `open(path).read().splitlines()[:10]`.
- Print available column names and sample rows to verify the loaded table.
- When column names are uncertain, search for common alternatives such as `lon`, `lat`, `lon1`, `lat1`, `longitude`, `latitude`, `x`, `y`.

---

## Common file patterns

- `station_longitude`, `station_latitude`, `latitude`, `longitude`
- `time`, `magnitude`, `depth`
- `phase`, `pick_time`, `confidence`
- Data may appear in `.csv` or `.txt` files under `/data/` or `/data/seismic/`
