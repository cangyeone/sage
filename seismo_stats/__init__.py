"""
seismo_stats — Seismological statistics toolkit for SAGE

Modules
-------
catalog_loader  Load earthquake catalogs from picks .txt or CSV/JSON files
bvalue          Gutenberg-Richter b-value and completeness magnitude Mc
plotting        G-R plots, temporal and spatial distribution figures
"""

from .catalog_loader import load_picks_txt, load_catalog_file, CatalogData
from .bvalue import calc_mc_maxcurvature, calc_bvalue_mle, calc_bvalue_lsq, BvalueResult
from .plotting import plot_gr, plot_temporal, plot_spatial, plot_all

__all__ = [
    'load_picks_txt', 'load_catalog_file', 'CatalogData',
    'calc_mc_maxcurvature', 'calc_bvalue_mle', 'calc_bvalue_lsq', 'BvalueResult',
    'plot_gr', 'plot_temporal', 'plot_spatial', 'plot_all',
]
