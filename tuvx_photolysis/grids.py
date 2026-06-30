# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Grids (edges / midpoints / deltas) and interpolators.

Reference: ``src/grid.F90``, ``src/interpolate.F90``.

A :class:`Grid` matches TUV-x's edge/midpoint convention: a cell ``i`` is bounded by
``edges[i]`` and ``edges[i+1]``; its midpoint is the average of those edges and its delta their
difference. Two interpolators are ported:

* :func:`interp_linear` -- standard point-to-point linear interpolation; target points outside the
  source range are set to zero (no extrapolation), matching ``interpolate_linear``.
* :func:`interp_conserving` -- area-conserving rebinning of point data onto target *bins*; the
  average value in each bin is the trapezoidal integral of the piecewise-linear source curve over
  the bin divided by the bin width. This is what TUV-x uses to put cross sections, quantum yields,
  and the extraterrestrial flux onto the model wavelength grid (``interpolate_conserving``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "Grid",
    "interp_linear",
    "interp_conserving",
    "interp_fractional_source",
    "interp_fractional_target",
]


@dataclass
class Grid:
    """A 1-D grid defined by its cell edges (``n_cells + 1`` values)."""

    edges: np.ndarray
    units: str = ""

    def __post_init__(self):
        self.edges = np.asarray(self.edges, dtype=float)
        if self.edges.ndim != 1 or self.edges.size < 2:
            raise ValueError("edges must be a 1-D array with at least two values")
        if np.any(np.diff(self.edges) <= 0):
            raise ValueError("edges must be strictly increasing")

    @property
    def n_cells(self) -> int:
        return self.edges.size - 1

    @property
    def midpoints(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    @property
    def deltas(self) -> np.ndarray:
        return np.diff(self.edges)


def interp_linear(x_target: np.ndarray, x_source: np.ndarray, y_source: np.ndarray) -> np.ndarray:
    """Linear interpolation of point data onto target points; 0 outside the source range.

    Mirrors ``interpolate_linear`` in ``src/interpolate.F90`` (no extrapolation).
    """
    x_target = np.asarray(x_target, dtype=float)
    x_source = np.asarray(x_source, dtype=float)
    y_source = np.asarray(y_source, dtype=float)
    return np.interp(x_target, x_source, y_source, left=0.0, right=0.0)


def interp_conserving(
    x_target_edges: np.ndarray, x_source: np.ndarray, y_source: np.ndarray
) -> np.ndarray:
    """Area-conserving rebinning of point data onto target bins.

    Parameters
    ----------
    x_target_edges : (n_edges,) increasing
        Edges of the target bins. The result has ``n_edges - 1`` values (one per bin).
    x_source, y_source : (n,) increasing x
        Piecewise-linear source curve.

    Returns
    -------
    (n_edges - 1,) array
        The average of the source curve over each target bin. The source must fully span the
        target range (no extrapolation), matching ``interpolate_conserving``.
    """
    x_target_edges = np.asarray(x_target_edges, dtype=float)
    x_source = np.asarray(x_source, dtype=float)
    y_source = np.asarray(y_source, dtype=float)

    if np.any(np.diff(x_source) <= 0):
        raise ValueError("source grid must be strictly increasing")
    if np.any(np.diff(x_target_edges) <= 0):
        raise ValueError("target grid must be strictly increasing")
    if x_source[0] > x_target_edges[0] or x_source[-1] < x_target_edges[-1]:
        raise ValueError("source and target grids do not overlap (extrapolation not permitted)")

    xl = x_source[:-1]
    xr = x_source[1:]
    yl = y_source[:-1]
    yr = y_source[1:]
    width = xr - xl
    # slope per source segment; coincident points contribute zero area (guard divide-by-zero)
    slope = np.where(width != 0.0, (yr - yl) / np.where(width != 0.0, width, 1.0), 0.0)

    n_bins = x_target_edges.size - 1
    out = np.zeros(n_bins)
    for i in range(n_bins):
        xgl = x_target_edges[i]
        xgu = x_target_edges[i + 1]
        a1 = np.maximum(xl, xgl)
        a2 = np.minimum(xr, xgu)
        overlap = a2 > a1
        b1 = yl + slope * (a1 - xl)
        b2 = yl + slope * (a2 - xl)
        darea = 0.5 * (a2 - a1) * (b1 + b2)
        out[i] = np.sum(np.where(overlap, darea, 0.0)) / (xgu - xgl)
    return out


def interp_fractional_source(
    x_target_edges: np.ndarray,
    x_source_edges: np.ndarray,
    y_source: np.ndarray,
    fold_in: bool = False,
) -> np.ndarray:
    """Rebin binned data onto target bins by fractional overlap (source-width normalized).

    ``y_source`` is one value per source bin (``len(x_source_edges) - 1``); the result has one
    value per target bin (``len(x_target_edges) - 1``). Each target bin sums the source values
    weighted by the fraction of each source bin it overlaps. Bins with no source coverage are zero.
    With ``fold_in=True``, source "overhang" beyond the last target edge is integrated and folded
    into the last target bin (used for vertical optical-depth profiles). Ports
    ``interpolate_fractional_source`` in ``src/interpolate.F90``.
    """
    xto = np.asarray(x_target_edges, dtype=float)
    xfrom = np.asarray(x_source_edges, dtype=float)
    yfrom = np.asarray(y_source, dtype=float)
    nto = xto.size
    nfrom = xfrom.size
    if yfrom.size < nfrom - 1:
        raise ValueError("y_source must have at least one value per source bin")
    ntobins = nto - 1
    yto = np.zeros(ntobins)
    jstart = 0  # 0-based source-bin index
    j = jstart
    for i in range(ntobins):
        s = 0.0
        j = jstart
        if j < nfrom - 1:
            while xfrom[j + 1] < xto[i]:
                jstart = j
                j += 1
                if j >= nfrom - 1:
                    break
            while j < nfrom - 1 and xfrom[j] <= xto[i + 1]:
                a1 = max(xfrom[j], xto[i])
                a2 = min(xfrom[j + 1], xto[i + 1])
                s += yfrom[j] * (a2 - a1) / (xfrom[j + 1] - xfrom[j])
                j += 1
            yto[i] = s

    if fold_in:
        j -= 1
        a1 = xto[nto - 1]
        a2 = xfrom[j + 1]
        if a2 > a1 or (j + 1) < (nfrom - 1):
            tail = yfrom[j] * (a2 - a1) / (xfrom[j + 1] - xfrom[j])
            for k in range(j + 1, nfrom - 1):
                tail += yfrom[k] * (xfrom[k + 1] - xfrom[k])
            yto[ntobins - 1] += tail
    return yto


def interp_fractional_target(
    x_target_edges: np.ndarray,
    x_source_edges: np.ndarray,
    y_source: np.ndarray,
    fold_in: bool = False,
) -> np.ndarray:
    """Rebin binned data onto target bins by fractional overlap (target-width normalized).

    Like :func:`interp_fractional_source` but each target bin sums ``y_source * overlap_width`` and
    divides by the *target* bin width. Ports ``interpolate_fractional_target`` in
    ``src/interpolate.F90``. With ``fold_in=True`` the source overhang beyond the last target edge
    is folded into the last target bin. ``y_source`` has at least one value per source bin.
    """
    xto = np.asarray(x_target_edges, dtype=float)
    xfrom = np.asarray(x_source_edges, dtype=float)
    yfrom = np.asarray(y_source, dtype=float)
    nfrom = xfrom.size
    nto = xto.size
    if yfrom.size < nfrom - 1:
        raise ValueError("y_source must have at least one value per source bin")

    ntobins = nto - 1
    out = np.zeros(ntobins)
    jstart = 0
    j = jstart
    for i in range(ntobins):
        s = 0.0
        j = jstart
        if j < nfrom - 1:
            while xfrom[j + 1] < xto[i]:
                jstart = j
                j += 1
                if j >= nfrom - 1:
                    break
            while j < nfrom - 1 and xfrom[j] <= xto[i + 1]:
                a1 = max(xfrom[j], xto[i])
                a2 = min(xfrom[j + 1], xto[i + 1])
                s += yfrom[j] * (a2 - a1)
                j += 1
            out[i] = s / (xto[i + 1] - xto[i])

    if fold_in:
        j -= 1
        a1 = xto[nto - 1]
        a2 = xfrom[j + 1]
        if a2 > a1 or (j + 1) < (nfrom - 1):
            tail = yfrom[j] * (a2 - a1) / (xfrom[j + 1] - xfrom[j])
            for k in range(j + 1, nfrom - 1):
                tail += yfrom[k] * (xfrom[k + 1] - xfrom[k])
            out[ntobins - 1] += tail  # Fortran adds the tail without target-width division
    return out
