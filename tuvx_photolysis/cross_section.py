# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Cross-section evaluation: rebin tabulated data onto the wavelength grid, apply T-dependence.

Reference: ``src/cross_section.F90`` (base type: ``process_file``, ``add_points``, ``calculate``) and
``src/cross_sections/o3_tint.F90`` (temperature-interpolated O3).

Two evaluators are provided:

* :class:`BaseCrossSection` -- the ``base`` type. Each input file's data is padded with endpoint
  points (:func:`add_points`) and area-conservingly rebinned onto the model wavelength grid; multiple
  files are summed. Evaluation replicates the (temperature-independent) spectrum across all altitudes.
* :class:`O3TintCrossSection` -- the ``O3`` type. Four files cover different wavelength bands; the
  cross section is linearly interpolated in temperature per altitude, with a vacuum->air refraction
  correction applied to the data wavelengths and band switching at 185/195/345 nm.

The temperature interpolation is exactly ``numpy.interp`` (clamp to the tabulated temperature range,
then linear), matching the Fortran ``Tadj``/``Tstar`` arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import TabulatedData
from .grids import interp_conserving, interp_fractional_source, interp_fractional_target

__all__ = [
    "refraction",
    "add_points",
    "BaseCrossSection",
    "O3TintCrossSection",
]

_DELTAX = 1.0e-5
_REFRAC_DENSITY = 2.45e19  # molecule cm-3 (o3_tint refracDensity)


def refraction(wavelength_nm: np.ndarray, atm_density: float = _REFRAC_DENSITY) -> np.ndarray:
    """Refractive index of standard air vs vacuum wavelength [nm] (Edlen 1966).

    Ports ``refraction`` in ``o3_tint.F90``. Valid 200-2000 nm; clamped outside that range.
    """
    wl = np.clip(np.asarray(wavelength_nm, dtype=float), 200.0, 2000.0)
    sig = 1.0e3 / wl
    sigsq = sig * sig
    wrk = 8342.13 + 2406030.0 / (130.0 - sigsq) + 15997.0 / (38.9 - sigsq)
    divisor = 2.69e19 * 273.15 / 288.15
    wrk = wrk * atm_density / divisor
    return 1.0 + 1.0e-8 * wrk


def _add_point(x: np.ndarray, y: np.ndarray, xnew: float, ynew: float):
    """Insert (xnew, ynew) keeping x sorted ascending (ports util add_point for distinct points)."""
    idx = int(np.searchsorted(x, xnew))
    if idx < x.size and x[idx] == xnew:
        xnew = xnew * (1.0 + _DELTAX) if xnew != 0.0 else _DELTAX
        idx = int(np.searchsorted(x, xnew))
    return np.insert(x, idx, xnew), np.insert(y, idx, ynew)


def add_points(
    data_lambda: np.ndarray,
    data_parameter: np.ndarray,
    lower_extrapolation: str | None = None,
    upper_extrapolation: str | None = None,
    lower_value: float = 0.0,
    upper_value: float = 0.0,
):
    """Pad tabulated data with endpoints so it spans the full wavelength range.

    Ports ``add_points`` in ``src/cross_section.F90``: adds points just inside/outside the data range
    so the conserving interpolator always covers the target grid. With no extrapolation the padded
    value is 0 (cross section falls to zero outside the data); ``"boundary"`` holds the edge value,
    ``"constant"`` uses ``lower_value``/``upper_value``.
    """
    x = np.asarray(data_lambda, dtype=float).copy()
    y = np.asarray(data_parameter, dtype=float).copy()
    lower_lambda = x[0]
    upper_lambda = x[-1]

    val_lower = 0.0
    if lower_extrapolation == "boundary":
        val_lower = y[0]
    elif lower_extrapolation == "constant":
        val_lower = lower_value

    val_upper = 0.0
    if upper_extrapolation == "boundary":
        val_upper = y[-1]
    elif upper_extrapolation == "constant":
        val_upper = upper_value

    x, y = _add_point(x, y, (1.0 - _DELTAX) * lower_lambda, val_lower)
    x, y = _add_point(x, y, 0.0, val_lower)
    x, y = _add_point(x, y, (1.0 + _DELTAX) * upper_lambda, val_upper)
    x, y = _add_point(x, y, 1.0e38, val_upper)
    return x, y


def _rebin_param(
    data_lambda, data_param, wl_edges, lower_extrap, upper_extrap,
    interpolator="conserving", fold_in=False,
):
    """add_points then rebin one parameter column onto the wavelength grid.

    The interpolator defaults to ``"conserving"`` but a file may request ``"fractional source"`` or
    ``"fractional target"`` (with optional ``fold_in``), matching the per-file ``interpolator``
    option in the TUV-x config.
    """
    x, y = add_points(data_lambda, data_param, lower_extrap, upper_extrap)
    if interpolator == "conserving":
        return interp_conserving(wl_edges, x, y)
    if interpolator == "fractional source":
        return interp_fractional_source(wl_edges, x, y, fold_in=fold_in)
    if interpolator == "fractional target":
        return interp_fractional_target(wl_edges, x, y, fold_in=fold_in)
    raise ValueError(f"unsupported cross-section interpolator: {interpolator}")


@dataclass
class BaseCrossSection:
    """``base``-type cross section: a temperature-independent spectrum on the model wavelength grid.

    Build with :meth:`from_files`. ``array`` has shape ``(n_wavelength_cells,)`` [cm^2].
    """

    array: np.ndarray  # (n_wl_cells,)

    @classmethod
    def from_files(cls, files, wl_edges) -> "BaseCrossSection":
        """Build from a list of per-file specs (summed).

        Each spec is ``(TabulatedData, lower_extrap, upper_extrap)`` or
        ``(TabulatedData, lower_extrap, upper_extrap, interpolator, fold_in)``. The first parameter
        is padded and rebinned onto ``wl_edges`` with the file's interpolator (default conserving);
        contributions are summed (matching the multi-file base cross section).
        """
        wl_edges = np.asarray(wl_edges, dtype=float)
        total = np.zeros(wl_edges.size - 1)
        for spec in files:
            td, lo, up = spec[0], spec[1], spec[2]
            interp = spec[3] if len(spec) > 3 else "conserving"
            fold_in = spec[4] if len(spec) > 4 else False
            total = total + _rebin_param(
                td.wavelength, td.parameters[:, 0], wl_edges, lo, up, interp, fold_in
            )
        return cls(array=total)

    def evaluate(self, n_levels: int) -> np.ndarray:
        """Return the cross section at every level, shape ``(n_levels, n_wavelength_cells)`` [cm^2]."""
        return np.repeat(self.array[None, :], n_levels, axis=0)


@dataclass
class _TintFile:
    array: np.ndarray  # (n_wl_cells, n_params), temperature-ascending columns
    temperature: np.ndarray  # (n_params,) ascending [K]


@dataclass
class O3TintCrossSection:
    """``O3``-type temperature-interpolated cross section (ports ``o3_tint.F90``).

    Build with :meth:`from_files` from the four O3 NetCDF files (in O3_1..O3_4 order). Evaluate at a
    per-level temperature with :meth:`evaluate`.
    """

    files: list  # list[_TintFile]
    base: np.ndarray  # (n_wl_cells,) file-1 base spectrum, used for lambda >= 345 nm
    wl_lower_edges: np.ndarray  # (n_wl_cells,) lower edge of each wavelength cell [nm]
    v185: float
    v195: float
    v345: float

    @classmethod
    def from_files(cls, tabulated_files, wl_edges) -> "O3TintCrossSection":
        """``tabulated_files`` is [O3_1, O3_2, O3_3, O3_4] as :class:`TabulatedData` (with temps)."""
        wl_edges = np.asarray(wl_edges, dtype=float)
        files = []
        for td in tabulated_files:
            # vacuum -> air refraction correction on the data wavelengths
            wl = np.asarray(td.wavelength, dtype=float)
            wl_corr = refraction(wl, _REFRAC_DENSITY) * wl
            temps = np.asarray(td.temperature, dtype=float).copy()
            params = td.parameters.copy()  # (n_data, n_params)
            n_params = params.shape[1]
            # reverse temperature order if descending (matches the o3_tint constructor)
            if temps.size > 1:
                dT = np.diff(temps)
                if not np.all(dT > 0):
                    if np.any(dT > 0):
                        raise ValueError("o3_tint temperature array not monotonic")
                    temps = temps[::-1].copy()
                    params = params[:, ::-1].copy()
            # rebin each parameter column onto the model grid (no extrapolation -> 0 outside)
            cols = [
                _rebin_param(wl_corr, params[:, p], wl_edges, None, None) for p in range(n_params)
            ]
            files.append(_TintFile(array=np.stack(cols, axis=1), temperature=temps))

        base = files[0].array[:, 0]
        # file-2's highest-temperature (295 K) column is replaced by the authoritative file-1 base
        files[1].array[:, -1] = base

        v185 = float(refraction(np.array([185.0]))[0] * 185.0)
        v195 = float(refraction(np.array([195.0]))[0] * 195.0)
        v345 = float(refraction(np.array([345.0]))[0] * 345.0)
        return cls(
            files=files,
            base=base,
            wl_lower_edges=wl_edges[:-1],
            v185=v185,
            v195=v195,
            v345=v345,
        )

    def evaluate(self, temperature: np.ndarray) -> np.ndarray:
        """Cross section at each level temperature, shape ``(n_levels, n_wavelength_cells)`` [cm^2].

        ``temperature`` is the model temperature per level [K] (interfaces or midpoints, matching the
        levels you want). Band switching uses each cell's lower edge, exactly as ``o3_tint`` does.
        """
        temperature = np.asarray(temperature, dtype=float)
        n_levels = temperature.size
        n_wl = self.wl_lower_edges.size
        out = np.zeros((n_levels, n_wl))

        lam = self.wl_lower_edges
        # band -> file index (0-based): lambda<185 -> file3; 185<=l<195 -> file4; 195<=l<345 -> file2
        for w in range(n_wl):
            l_ = lam[w]
            if l_ < self.v185:
                f = self.files[2]
            elif self.v185 <= l_ < self.v195:
                f = self.files[3]
            elif self.v195 <= l_ < self.v345:
                f = self.files[1]
            else:
                out[:, w] = self.base[w]  # temperature-independent above 345 nm
                continue
            # linear-in-T with clamping == np.interp over the (ascending) temperature nodes
            out[:, w] = np.interp(temperature, f.temperature, f.array[w, :])
        return out
