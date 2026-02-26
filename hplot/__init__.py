"""
h-plot: A spatial heterogeneity plot for tumor border analysis

This package provides:
- Core API for fitting and plotting spatial layer-based profiles
- Batch utilities for handling grouped plotting by tumor types
- CI computation adapted for varying region counts
"""

__version__ = "0.1.0"

from .core import HPlot
from .runners import run_hplot_batch