# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Quantum-yield evaluation.

Reference: ``src/quantum_yield.F90``. The ``base`` quantum yield is either a constant value held
across wavelength, or a tabulated spectrum read from NetCDF and area-conservingly rebinned onto the
model wavelength grid (then replicated across altitude). Reaction-specific quantum-yield recipes
(e.g. the O3 photolysis branching) are separate evaluators to be added as needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import TabulatedData
from .grids import interp_conserving

__all__ = [
    "ConstantQuantumYield",
    "TabulatedQuantumYield",
    "TintQuantumYield",
    "clono2_quantum_yield",
    "hno4_branching_quantum_yield",
    "clooocl_branching_quantum_yield",
]


def hno4_branching_quantum_yield(wl_mid, n_levels, channel) -> np.ndarray:
    """HO2NO2 (HNO4) photolysis branching quantum yield (JPL 19-5, Table 4C-9-2).

    Φ(HO2+NO2) = 0.8 for λ > 200 nm, 0.7 for λ ≤ 200 nm; Φ(OH+NO3) = 1 − Φ(HO2+NO2). Both channels
    share the HNO4 absorption cross section and partition unity. ``channel`` is ``"HO2+NO2"`` or
    ``"OH+NO3"``.
    """
    wl = np.asarray(wl_mid, dtype=float)
    ho2no2 = np.where(wl > 200.0, 0.8, 0.7)
    if channel == "HO2+NO2":
        phi = ho2no2
    elif channel == "OH+NO3":
        phi = 1.0 - ho2no2
    else:
        raise ValueError(f"unknown HNO4 branch: {channel}")
    return np.repeat(phi[None, :], n_levels, axis=0)


def clooocl_branching_quantum_yield(wl_mid, n_levels, channel) -> np.ndarray:
    """ClOOCl (Cl2O2) photolysis branching quantum yield (JPL 19-5, Section F7).

    Φ(Cl+ClOO) = 0.8 at all wavelengths; the remaining 0.2 goes to the 2ClO (+ClO+Cl+O) channel.
    Both channels share the ClOOCl absorption cross section. ``channel`` is ``"Cl+ClOO"`` or
    ``"2ClO"``.
    """
    wl = np.asarray(wl_mid, dtype=float)
    val = 0.8 if channel == "Cl+ClOO" else (0.2 if channel == "2ClO" else None)
    if val is None:
        raise ValueError(f"unknown ClOOCl branch: {channel}")
    return np.full((n_levels, wl.size), val)


def clono2_quantum_yield(wl_mid, n_levels, branch) -> np.ndarray:
    """ClONO2 photolysis branching quantum yield (analytic; ports clono2-cl_no3 / clono2-clo_no2).

    The Cl+NO3 branch is 0.6 below 308 nm, ``7.143e-3*lambda - 1.6`` over 308-364 nm, and 1.0 above;
    the ClO+NO2 branch is ``1 - (Cl+NO3)``. ``branch`` is ``"Cl+NO3"`` or ``"ClO+NO2"``.
    """
    wl = np.asarray(wl_mid, dtype=float)
    phi = np.where(wl < 308.0, 0.6, np.where(wl <= 364.0, 7.143e-3 * wl - 1.6, 1.0))
    if branch == "ClO+NO2":
        phi = 1.0 - phi
    elif branch != "Cl+NO3":
        raise ValueError(f"unknown ClONO2 branch: {branch}")
    return np.repeat(phi[None, :], n_levels, axis=0)


@dataclass
class ConstantQuantumYield:
    """A wavelength- and temperature-independent quantum yield (``base`` with ``constant value``)."""

    value: float

    def evaluate(self, n_levels: int, n_wavelengths: int) -> np.ndarray:
        return np.full((n_levels, n_wavelengths), float(self.value))


@dataclass
class TabulatedQuantumYield:
    """A tabulated quantum-yield spectrum on the model wavelength grid (replicated across altitude)."""

    array: np.ndarray  # (n_wavelength_cells,)

    @classmethod
    def from_netcdf(
        cls, td: TabulatedData, wl_edges, lower_extrapolation=None, upper_extrapolation=None,
        lower_value=0.0, upper_value=0.0,
    ) -> "TabulatedQuantumYield":
        """Conservingly rebin a single-parameter quantum-yield NetCDF onto the wavelength grid.

        Endpoint extrapolation (e.g. holding the yield at a constant value below the data range,
        as some reactions specify) is applied via :func:`add_points`.
        """
        wl_edges = np.asarray(wl_edges, dtype=float)
        from .cross_section import add_points

        x, y = add_points(
            td.wavelength, td.parameters[:, 0],
            lower_extrapolation, upper_extrapolation, lower_value, upper_value,
        )
        return cls(array=interp_conserving(wl_edges, x, y))

    def evaluate(self, n_levels: int, n_wavelengths: int) -> np.ndarray:
        if self.array.size != n_wavelengths:
            raise ValueError("quantum yield wavelength size mismatch")
        return np.repeat(self.array[None, :], n_levels, axis=0)


def _tint_extrapolate(temps, arr, T):
    """Linear-in-T evaluation that extrapolates beyond the tabulated range (ports no2_tint.F90).

    ``arr`` is ``(n_wl, n_temps)`` with ascending ``temps``. Returns ``(n_levels, n_wl)``. Unlike
    ``numpy.interp`` (which clamps), this continues the nearest interval's slope past the endpoints.
    """
    T = np.asarray(T, dtype=float)
    n_temp = temps.size
    lo = np.clip(np.searchsorted(temps, T, side="left") - 1, 0, n_temp - 2)
    t0 = temps[lo]
    frac = (T - t0) / (temps[lo + 1] - t0)  # (n_levels,), <0 or >1 when extrapolating
    a0 = arr[:, lo]  # (n_wl, n_levels)
    a1 = arr[:, lo + 1]
    return (a0 + frac[None, :] * (a1 - a0)).T


@dataclass
class TintQuantumYield:
    """Temperature-interpolated quantum yield (ports quantum_yields/tint.F90 and no2_tint.F90).

    Each file has a temperature axis; the yield is interpolated in temperature and summed across
    files. ``extrapolate=False`` clamps to the tabulated range (generic ``tint``, used by NO3);
    ``extrapolate=True`` continues the end-interval slope and clamps the result to >= 0 (``NO2 tint``).
    """

    files: list  # list of (array (n_wl, n_temps), temps (n_temps,))
    extrapolate: bool = False

    @classmethod
    def from_netcdf(cls, tds, wl_edges, lower_extrapolation=None, upper_extrapolation=None,
                    lower_value=0.0, upper_value=0.0, extrapolate=False):
        from .cross_section import add_points
        from .special import _reverse_if_descending

        files = []
        for td in tds:
            cols = []
            for p in range(td.parameters.shape[1]):
                x, y = add_points(
                    td.wavelength, td.parameters[:, p],
                    lower_extrapolation, upper_extrapolation, lower_value, upper_value,
                )
                cols.append(interp_conserving(wl_edges, x, y))
            temps, arr = _reverse_if_descending(td.temperature, np.stack(cols, axis=1))
            files.append((arr, temps))
        return cls(files=files, extrapolate=extrapolate)

    def evaluate(self, temperature) -> np.ndarray:
        T = np.asarray(temperature, dtype=float)
        n_wl = self.files[0][0].shape[0]
        out = np.zeros((T.size, n_wl))
        for arr, temps in self.files:
            if self.extrapolate:
                out += _tint_extrapolate(temps, arr, T)
            else:
                for w in range(n_wl):
                    out[:, w] += np.interp(T, temps, arr[w, :])
        return np.maximum(out, 0.0) if self.extrapolate else out
