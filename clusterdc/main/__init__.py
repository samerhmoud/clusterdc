"""
ClusterDC: Density-Contour Clustering with Advanced Kernel Density Estimation.

This package provides tools for density-based clustering using kernel density 
estimation and contour analysis to identify clusters in 2D data.
"""

__version__ = "0.0.0.2"

from .data import Data
from .kde import KDE
from .clusterdc import ClusterDC

__all__ = ["Data", "KDE", "ClusterDC"]