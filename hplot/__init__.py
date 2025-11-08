"""h-plot: tools for spatial heterogeneity visualisation."""

from __future__ import annotations

from .core import HPlot
from .runners import run_hplot_batch

__all__ = ["HPlot", "run_hplot_batch"]
__version__ = "0.1.0"
