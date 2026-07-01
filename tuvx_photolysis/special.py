# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Reaction-specific cross-section evaluators (the temperature recipes JPL/IUPAC prescribe).

These port the TUV-x ``src/cross_sections`` modules that frank-model's stratospheric
chlorine/bromine/nitrogen reactions use. Each returns ``sigma(level, wavelength)`` [cm^2] given the
per-level temperature [K]:

* :func:`cl2_cross_section`  -- Cl2: analytic, temperature-dependent (cl2-cl_cl.F90).
* :func:`hobr_cross_section` -- HOBr: analytic, temperature-independent (hobr-oh_br.F90).
* :class:`HNO3CrossSection`  -- HNO3: ``sigma0 * exp(B * (T - 298))`` (hno3-oh_no2.F90).
* :class:`N2O5CrossSection`  -- N2O5: ``sigma0 * 10**(1000 * B * (Tc - Tadj)/(Tc * Tadj))``
  with ``Tadj`` clamped to [233, 300] (n2o5-no2_no3.F90).
* :class:`ClONO2CrossSection` -- ClONO2: Taylor series ``c1*(1 + dT*(c2 + dT*c3))``, ``dT = T-296``
  (clono2.F90).
* :class:`TintCrossSection`  -- generic temperature interpolation, one or more files each with a
  temperature axis (no2_tint.F90); linear-in-T == ``numpy.interp``.
* :class:`OCloCrossSection`  -- OClO: each file is one temperature node, interpolated across files.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cross_section import _rebin_param

__all__ = [
    "cl2_cross_section",
    "hobr_cross_section",
    "HNO3CrossSection",
    "N2O5CrossSection",
    "ClONO2CrossSection",
    "TintCrossSection",
    "OCloCrossSection",
]


def cl2_cross_section(wl_mid, temperature) -> np.ndarray:
    """Cl2 -> 2 Cl analytic cross section [cm^2], shape ``(n_levels, n_wl)`` (ports cl2-cl_cl.F90)."""
    wc = np.asarray(wl_mid, dtype=float)
    T = np.asarray(temperature, dtype=float)
    bb = np.exp(402.7 / T)
    bbsq = bb * bb
    alpha = (bbsq - 1.0) / (bbsq + 1.0)  # (n_levels,)
    ex1 = 27.3 * np.exp(-99.0 * alpha[:, None] * (np.log(329.5 / wc)[None, :]) ** 2)
    ex2 = 0.932 * np.exp(-91.5 * alpha[:, None] * (np.log(406.5 / wc)[None, :]) ** 2)
    return 1.0e-20 * np.sqrt(alpha)[:, None] * (ex1 + ex2)


def hobr_cross_section(wl_mid, n_levels) -> np.ndarray:
    """HOBr -> OH + Br analytic cross section [cm^2] (temperature-independent; ports hobr-oh_br.F90)."""
    wc = np.asarray(wl_mid, dtype=float)
    xs = np.zeros_like(wc)
    band = (wc >= 250.0) & (wc <= 550.0)
    wb = wc[band]
    xs[band] = 1.0e-20 * (
        24.77 * np.exp(-109.80 * (np.log(284.01 / wb)) ** 2)
        + 12.22 * np.exp(-93.63 * (np.log(350.57 / wb)) ** 2)
        + 2.283 * np.exp(-242.40 * (np.log(457.38 / wb)) ** 2)
    )
    return np.repeat(xs[None, :], n_levels, axis=0)


@dataclass
class HNO3CrossSection:
    """HNO3 -> OH + NO2: ``sigma0(lambda) * exp(B(lambda) * (T - 298 K))`` (hno3-oh_no2.F90)."""

    sigma0: np.ndarray  # (n_wl,)
    b: np.ndarray  # (n_wl,)

    @classmethod
    def from_file(cls, td, wl_edges, lower_extrap=None, upper_extrap=None):
        s0 = _rebin_param(td.wavelength, td.parameters[:, 0], wl_edges, lower_extrap, upper_extrap)
        # the temperature coefficient B is boundary-extrapolated at both ends (hno3-oh_no2.F90
        # forces this), which affects its conserving rebin in the edge wavelength cells
        b = _rebin_param(td.wavelength, td.parameters[:, 1], wl_edges, "boundary", "boundary")
        return cls(sigma0=s0, b=b)

    def evaluate(self, temperature) -> np.ndarray:
        T = np.asarray(temperature, dtype=float)
        return self.sigma0[None, :] * np.exp(self.b[None, :] * (T[:, None] - 298.0))


@dataclass
class N2O5CrossSection:
    """N2O5 -> NO2 + NO3: ``sigma0 * 10**(1000 * B * (Tc - Tadj)/(Tc*Tadj))`` (n2o5-no2_no3.F90)."""

    sigma0: np.ndarray  # (n_wl,)  base cross section (file 1)
    b: np.ndarray  # (n_wl,)  temperature coefficient (file 2)
    t_floor: float = 233.0
    t_ceil: float = 300.0
    t_scale: float = 1000.0

    @classmethod
    def from_files(cls, td0, td1, wl_edges):
        s0 = _rebin_param(td0.wavelength, td0.parameters[:, 0], wl_edges, None, None)
        b = _rebin_param(td1.wavelength, td1.parameters[:, 0], wl_edges, None, None)
        return cls(sigma0=s0, b=b)

    def evaluate(self, temperature) -> np.ndarray:
        T = np.clip(np.asarray(temperature, dtype=float), self.t_floor, self.t_ceil)
        tfac = self.t_scale * self.b[None, :] * (self.t_ceil - T[:, None]) / (self.t_ceil * T[:, None])
        return self.sigma0[None, :] * 10.0**tfac


@dataclass
class ClONO2CrossSection:
    """ClONO2: Taylor series ``c1*(1 + dT*(c2 + dT*c3))``, ``dT = T - 296 K`` (clono2.F90)."""

    c: np.ndarray  # (n_wl, 3)

    @classmethod
    def from_file(cls, td, wl_edges, lower_extrap=None, upper_extrap=None):
        cols = [
            _rebin_param(td.wavelength, td.parameters[:, p], wl_edges, lower_extrap, upper_extrap)
            for p in range(3)
        ]
        return cls(c=np.stack(cols, axis=1))

    def evaluate(self, temperature) -> np.ndarray:
        dT = np.asarray(temperature, dtype=float)[:, None] - 296.0
        c1, c2, c3 = self.c[None, :, 0], self.c[None, :, 1], self.c[None, :, 2]
        return c1 * (1.0 + dT * (c2 + dT * c3))


def _reverse_if_descending(temps, cols):
    temps = np.asarray(temps, dtype=float)
    if temps.size > 1 and np.all(np.diff(temps) < 0):
        return temps[::-1].copy(), cols[:, ::-1].copy()
    return temps, cols


@dataclass
class _TintFile:
    array: np.ndarray  # (n_wl, n_temps)
    temperature: np.ndarray  # (n_temps,) ascending


@dataclass
class TintCrossSection:
    """Generic temperature-interpolated cross section, summed over files (ports no2_tint.F90).

    Each file carries a temperature axis; the cross section is linearly interpolated in temperature
    (== ``numpy.interp``: clamp to the tabulated range, then linear) and summed across files (files
    cover disjoint wavelength bands, zero elsewhere).
    """

    files: list  # list[_TintFile]

    @classmethod
    def from_files(cls, tds, wl_edges, lower_extrap=None, upper_extrap=None):
        files = []
        for td in tds:
            cols = np.stack(
                [
                    _rebin_param(td.wavelength, td.parameters[:, p], wl_edges, lower_extrap, upper_extrap)
                    for p in range(td.parameters.shape[1])
                ],
                axis=1,
            )
            temps, cols = _reverse_if_descending(td.temperature, cols)
            files.append(_TintFile(array=cols, temperature=temps))
        return cls(files=files)

    def evaluate(self, temperature) -> np.ndarray:
        T = np.asarray(temperature, dtype=float)
        n_wl = self.files[0].array.shape[0]
        out = np.zeros((T.size, n_wl))
        for f in self.files:
            for w in range(n_wl):
                out[:, w] += np.interp(T, f.temperature, f.array[w, :])
        return out


@dataclass
class OCloCrossSection:
    """OClO: each file is a single temperature node; interpolate the spectrum across files."""

    temps: np.ndarray  # (n_files,) ascending
    arrays: np.ndarray  # (n_files, n_wl)

    @classmethod
    def from_files(cls, tds, wl_edges):
        temps = np.array([float(td.temperature[0]) for td in tds])
        arrays = np.stack(
            [_rebin_param(td.wavelength, td.parameters[:, 0], wl_edges, None, None) for td in tds]
        )
        order = np.argsort(temps)
        return cls(temps=temps[order], arrays=arrays[order])

    def evaluate(self, temperature) -> np.ndarray:
        T = np.asarray(temperature, dtype=float)
        n_wl = self.arrays.shape[1]
        out = np.zeros((T.size, n_wl))
        for w in range(n_wl):
            out[:, w] = np.interp(T, self.temps, self.arrays[:, w])
        return out
