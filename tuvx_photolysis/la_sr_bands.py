# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Lyman-alpha and Schumann-Runge band parameterization for O2.

Reference: ``src/la_sr_bands.F90``.

Below ~206 nm the O2 absorption is dominated by the Schumann-Runge bands (175.4-206.2 nm) and the
Lyman-alpha line (121.4-121.9 nm), where the cross section is far too finely structured to put on the
model wavelength grid and where O2 *self-shielding* makes the effective absorption depend on the O2
slant column. This module replaces the plain O2 cross section / optical depth in those wavelength
bins with column-dependent *effective* values:

* Lyman-alpha: Chabrillat & Kockarts (1997) reduction-factor parameterization.
* Schumann-Runge bands: Koppers & Murtagh (1996) — ``ln(xs) = A(X)(T-220) + B(X)`` with A, B from
  20-term Chebyshev polynomials in ``X = ln(O2 slant column)`` over 38 <= X <= 56, per the 17 SR
  wavelength intervals.

Both an optical-depth form (used in the radiation-field solve, per layer) and a cross-section form
(used in the photolysis-rate integral, per level) are provided, matching ``la_srb_OD``/``la_srb_xs``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["LaSrBands"]

# Lyman-alpha band: 1 bin
_WLLA = np.array([121.4, 121.9])
# Schumann-Runge bands: 17 bins (18 edges)
_WLSRB = np.array([
    175.4, 177.0, 178.6, 180.2, 181.8, 183.5, 185.2, 186.9, 188.7, 190.5,
    192.3, 194.2, 196.1, 198.0, 200.0, 202.0, 204.1, 206.2,
])
_NLA = 1
_NSRB = 17
_NPOLY = 20
_KLOWER = 38.0
_KUPPER = 56.0
_PRECIS = 1.0e-7
_LARGEST = 1.0e36
_COLMIN = np.exp(38.0)

# SR cross sections at large optical depth (Koppers & Murtagh), one per SR bin
_XSLOD = np.array([
    6.2180730e-21, 5.8473627e-22, 5.6996334e-22, 4.5627094e-22, 1.7668250e-22,
    1.1178808e-22, 1.2040544e-22, 4.0994668e-23, 1.8450616e-23, 1.5639540e-23,
    8.7961075e-24, 7.6475608e-24, 7.6260556e-24, 7.5565696e-24, 7.6334338e-24,
    7.4371992e-24, 7.3642966e-24,
])

# Chabrillat & Kockarts (1997) Lyman-alpha coefficients
_LA_B = np.array([6.8431e-01, 2.29841e-01, 8.65412e-02])
_LA_C = np.array([8.22114e-21, 1.77556e-20, 8.22112e-21])
_LA_D = np.array([6.0073e-21, 4.28569e-21, 1.28059e-20])
_LA_E = np.array([8.21666e-21, 1.63296e-20, 4.85121e-17])
_EXP_LIM = 100.0e8
_TINY = 1.0e-100
_XSMIN = 1.0e-20
_LARGE_OD = 1000.0


def _find_edge(wl_edges, value):
    """Index of the grid edge equal to ``value`` (within tolerance), or None."""
    idx = np.where(np.abs(wl_edges - value) < 10.0 * _PRECIS)[0]
    return int(idx[0]) if idx.size else None


@dataclass
class LaSrBands:
    """LA/SR O2 parameterization tied to a model wavelength grid."""

    ac: np.ndarray  # (nPoly, nsrb) Chebyshev coefficients for A
    bc: np.ndarray  # (nPoly, nsrb) Chebyshev coefficients for B
    ila: int | None  # wavelength-cell index of the Lyman-alpha bin (or None)
    isrb: int | None  # wavelength-cell index of the first SR band bin (or None)

    @property
    def has_la(self) -> bool:
        return self.ila is not None

    @property
    def has_srb(self) -> bool:
        return self.isrb is not None

    @property
    def has_la_srb(self) -> bool:
        return self.has_la or self.has_srb

    @classmethod
    def from_file(cls, wl_edges, o2_parameters_path) -> "LaSrBands":
        """Build for a wavelength grid, loading the Chebyshev coefficients from O2_parameters.txt."""
        wl_edges = np.asarray(wl_edges, dtype=float)
        ac, bc = _read_chebyshev(o2_parameters_path)
        # the LA/SR bins must align exactly with the model grid (as the Fortran asserts)
        ila = _find_edge(wl_edges, _WLLA[0])
        if ila is not None:
            for i in range(1, _NLA + 1):
                if _find_edge(wl_edges, _WLLA[i]) is None:
                    ila = None
                    break
        isrb = _find_edge(wl_edges, _WLSRB[0])
        if isrb is not None:
            for i in range(1, _NSRB + 1):
                if _find_edge(wl_edges, _WLSRB[i]) is None:
                    isrb = None
                    break
        return cls(ac=ac, bc=bc, ila=ila, isrb=isrb)

    # -- public entry points ---------------------------------------------------------------------
    def optical_depth(self, o2_optical_depth, o2_slant_col, air_vcol, air_scol, temperature_edge):
        """Replace the O2 layer optical depth in the LA/SR bins (ports ``la_srb_OD``).

        ``o2_optical_depth`` is ``(n_layers, n_wl)``; the others are per level (length n_levels).
        Returns a copy with the LA bin and SR-band columns overwritten by the effective OD.
        """
        if not self.has_la_srb:
            return o2_optical_depth
        od = np.array(o2_optical_depth, dtype=float)
        secchi = self._secchi(air_vcol, air_scol)
        if self.has_la:
            od[:, self.ila] = _lymana_od(o2_slant_col, secchi)
        if self.has_srb:
            od[:, self.isrb:self.isrb + _NSRB] = self._schum_od(
                o2_slant_col, temperature_edge, secchi
            )
        return od

    def cross_section(self, o2_cross_section, o2_slant_col, air_vcol, air_scol, temperature_edge):
        """Replace the O2 photolysis cross section in the LA/SR bins (ports ``la_srb_xs``).

        ``o2_cross_section`` is ``(n_levels, n_wl)``. Returns a copy with the LA/SR bins overwritten.
        """
        if not self.has_la_srb:
            return o2_cross_section
        xs = np.array(o2_cross_section, dtype=float)
        secchi = self._secchi(air_vcol, air_scol)
        if self.has_la:
            xs[:, self.ila] = _lymana_xs(o2_slant_col)
        if self.has_srb:
            xs[:, self.isrb:self.isrb + _NSRB] = self._schum_xs(o2_slant_col, temperature_edge)
        return xs

    # -- internals -------------------------------------------------------------------------------
    @staticmethod
    def _secchi(air_vcol, air_scol):
        """Effective sec(SZA) = slant/vertical air column; 2.0 where there is no direct sun."""
        air_vcol = np.asarray(air_vcol, dtype=float)
        air_scol = np.asarray(air_scol, dtype=float)
        nz = air_scol.size
        secchi = np.empty(nz)
        lit = air_scol[: nz - 1] <= 0.1 * _LARGEST
        secchi[: nz - 1] = np.where(lit, air_scol[: nz - 1] / air_vcol[: nz - 1], 2.0)
        secchi[nz - 1] = secchi[nz - 2]
        return secchi

    def _effxs(self, x, temperature):
        """Effective SR cross sections at log-column ``x`` and temperature ``T`` (17 bins)."""
        a = np.array([_chebyshev(self.ac[:, i], x) for i in range(_NSRB)])
        b = np.array([_chebyshev(self.bc[:, i], x) for i in range(_NSRB)])
        return np.exp(a * (temperature - 220.0) + b)

    def _schum_cross_section(self, o2col, tlev):
        """SR effective cross sections per level (n_levels, nsrb). Ports the shared part of schum."""
        nz = o2col.size
        xs = np.tile(_XSLOD, (nz, 1))
        ktop = nz  # 0-based: first level (from top) below colmin
        kbot = -1  # 0-based: last level with x > 56
        o2col1 = np.maximum(o2col, _COLMIN)
        for k in range(nz):
            if o2col[k] < _COLMIN:
                ktop = min(k - 1, ktop)
            else:
                x = np.log(o2col1[k])
                if x > 56.0:
                    kbot = k
                else:
                    xs[k, :] = self._effxs(x, tlev[k])
        if kbot == nz - 1:
            kbot = nz - 2
        # fill out-of-range levels by repeating the edge table values
        if kbot >= 0:
            xs[: kbot + 1, :] = xs[kbot + 1, :]
        if ktop + 1 <= nz - 1:
            xs[ktop + 1:, :] = xs[ktop, :]
        return xs, o2col1

    def _schum_xs(self, o2col, tlev):
        return self._schum_cross_section(np.asarray(o2col), np.asarray(tlev))[0]

    def _schum_od(self, o2col, tlev, secchi):
        """SR effective optical depth per layer (n_layers, nsrb). Ports ``schum_OD``."""
        o2col = np.asarray(o2col, dtype=float)
        nz = o2col.size
        xs, o2col1 = self._schum_cross_section(o2col, np.asarray(tlev))
        norm = 1.0 / (nz - 1)
        od = np.zeros((nz - 1, _NSRB))
        for k in range(nz - 1):
            kp1 = k + 1
            if abs(1.0 - o2col1[kp1] / o2col1[k]) <= 2.0 * _PRECIS:
                od[k, :] = xs[kp1, :] * o2col1[kp1] * norm
            else:
                num = xs[kp1, :] * o2col1[kp1] - xs[k, :] * o2col1[k]
                den = 1.0 + np.log(xs[kp1, :] / xs[k, :]) / np.log(o2col1[kp1] / o2col1[k])
                od[k, :] = np.abs(num / den)
                od[k, :] = 2.0 * od[k, :] / (secchi[k] + secchi[kp1])
        return od


def _read_chebyshev(path):
    """Read the AC, BC (nPoly x nsrb) Chebyshev coefficient blocks from O2_parameters.txt."""
    lines = Path(path).read_text().splitlines()

    def block(start):
        rows = []
        for ln in lines[start:start + _NPOLY]:
            vals = [float(v) for v in ln.replace(",", " ").split()]
            rows.append(vals[:_NSRB])
        return np.array(rows)  # (nPoly, nsrb)

    # line 0: "ChebcoefA", line 1: Region header, lines 2..21: AC; then ChebcoefB header + BC
    ac = block(2)
    bc = block(2 + _NPOLY + 2)
    return ac, bc


def _chebyshev(coefficients, x):
    """Clenshaw evaluation of an nPoly-term Chebyshev series on [kLower, kUpper] at ``x``."""
    di = 0.0
    di1 = 0.0
    y = (2.0 * x - (_KLOWER + _KUPPER)) / (_KUPPER - _KLOWER)
    y2 = 2.0 * y
    for i in range(_NPOLY - 1, 0, -1):  # nPoly-1 .. 1 (0-based) == Fortran nPoly..2
        dtemp = di
        di = y2 * di - di1 + coefficients[i]
        di1 = dtemp
    return y * di - di1 + 0.5 * coefficients[0]


def _lymana_reduction(o2col, coeff):
    """Reduction factor sum_j weight_j * exp(-coeff_j * o2col) at each level."""
    sigma = np.outer(o2col, coeff)  # (nz, 3)
    tau = np.where(sigma < _EXP_LIM, np.exp(-np.where(sigma < _EXP_LIM, sigma, 0.0)), 0.0)
    return tau


def _lymana_od(o2col, secchi):
    """Effective Lyman-alpha O2 optical depth per layer (length n_layers). Ports ``lymana_OD``."""
    o2col = np.asarray(o2col, dtype=float)
    nz = o2col.size
    rm = _lymana_reduction(o2col, _LA_C) @ _LA_B  # (nz,)
    od = np.full(nz - 1, _LARGE_OD)
    for iz in range(nz - 1):
        if rm[iz] > _TINY and rm[iz + 1] > 0.0:
            od[iz] = np.log(rm[iz + 1]) / secchi[iz + 1] - np.log(rm[iz]) / secchi[iz]
    return od


def _lymana_xs(o2col):
    """Effective Lyman-alpha O2 cross section per level (length n_levels). Ports ``lymana_xs``."""
    o2col = np.asarray(o2col, dtype=float)
    rm = _lymana_reduction(o2col, _LA_C) @ _LA_B
    ro2 = _lymana_reduction(o2col, _LA_E) @ _LA_D
    xs = np.where((rm > _TINY) & (ro2 > _TINY), ro2 / np.where(rm > 0, rm, 1.0), _XSMIN)
    return xs
