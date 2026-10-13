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
* :class:`H2O2CrossSection` -- H2O2: JPL94 temperature-dependent form over [260, 350) nm, base
  tabulated data elsewhere (h2o2-oh_oh.F90).
* :class:`TintCrossSection`  -- generic temperature interpolation, one or more files each with a
  temperature axis (no2_tint.F90); linear-in-T == ``numpy.interp``.
* :class:`OCloCrossSection`  -- OClO: each file is one temperature node, interpolated across files.

Tropospheric (MCM) recipes:

* :class:`CH2OCrossSection` -- CH2O: ``p1 + p2 * (T - 298)``, no temperature clamp (ch2o.F90).
* :class:`HNO3CrossSection` also serves the alkyl nitrates: type ``RONO2`` (rono2.F90, e.g.
  i-C3H7ONO2) and type ``CH3ONO2+hv->CH3O+NO2`` (ch3ono2-ch3o_no2.F90; also C2H5ONO2, PAN, PPN).
* :func:`nitroxy_acetone_cross_section` -- CH3COCH2ONO2: analytic, T-independent
  (nitroxy_acetone.F90).
* :class:`AcetoneCrossSection` -- CH3COCH3: cubic in T clamped to [235, 298] K
  (acetone-ch3co_ch3.F90).
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
    "H2O2CrossSection",
    "TintCrossSection",
    "OCloCrossSection",
    "CH2OCrossSection",
    "nitroxy_acetone_cross_section",
    "AcetoneCrossSection",
]

# JPL94 H2O2 cross-section polynomial coefficients (h2o2-oh_oh.F90), highest power first for np.polyval.
_H2O2_A = np.array([1.5534675e-13, -2.652014e-10, 1.6878206e-07, -4.035101e-05,
                    -4.4589016e-03, 4.535649, -9.2170972e02, 6.4761e04])
_H2O2_B = np.array([-1.0924e-07, -3.0493e-05, 1.1522e-01, -5.1351e01, 6.8123e03])


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
    """``sigma0(lambda) * exp(B(lambda) * (T - 298 K))``, no temperature clamp.

    Three Fortran modules share this form and differ only in how B is padded before rebinning:

    * HNO3 -> OH + NO2 (``hno3-oh_no2.F90``) and alkyl nitrates, type ``RONO2`` (``rono2.F90``):
      B is boundary-extrapolated at both ends (``b_extrap="boundary"``, the default);
    * CH3ONO2 -> CH3O + NO2 and the other users of that type (``ch3ono2-ch3o_no2.F90``): B is
      padded like sigma0 by the base ``process_file`` (``b_extrap=None`` -> zero outside the data).
    """

    sigma0: np.ndarray  # (n_wl,)
    b: np.ndarray  # (n_wl,)

    @classmethod
    def from_file(cls, td, wl_edges, lower_extrap=None, upper_extrap=None, b_extrap="boundary"):
        s0 = _rebin_param(td.wavelength, td.parameters[:, 0], wl_edges, lower_extrap, upper_extrap)
        # the padding of the temperature coefficient B affects its conserving rebin in the edge
        # wavelength cells (hno3-oh_no2.F90 / rono2.F90 force "boundary" at both ends)
        b = _rebin_param(td.wavelength, td.parameters[:, 1], wl_edges, b_extrap, b_extrap)
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


@dataclass
class H2O2CrossSection:
    """H2O2 -> OH + OH cross section [cm^2] (ports h2o2-oh_oh.F90).

    JPL94 parameterization over [260, 350) nm; the base tabulated cross section elsewhere. Matching
    the Fortran exactly: the band is selected by each wavelength cell's LOWER edge, the polynomials
    ``sumA``/``sumB`` are evaluated at the cell MIDPOINT, and the temperature is clamped to [200, 400] K:

        chi = 1 / (1 + exp(-1265 / T))
        sigma = (chi * sumA + (1 - chi) * sumB) * 1e-21   for 260 <= lower_edge < 350 nm
        sigma = base(lambda)                              otherwise
    """

    base: np.ndarray  # (n_wl,) base tabulated cross section
    wl_mid: np.ndarray  # (n_wl,) wavelength-cell midpoints [nm]
    band: np.ndarray  # (n_wl,) bool: cells with 260 <= lower_edge < 350 nm

    @classmethod
    def from_file(cls, td, wl_edges, lower_extrap=None, upper_extrap=None):
        base = _rebin_param(td.wavelength, td.parameters[:, 0], wl_edges, lower_extrap, upper_extrap)
        wl_edges = np.asarray(wl_edges, dtype=float)
        wl_mid = 0.5 * (wl_edges[:-1] + wl_edges[1:])
        band = (wl_edges[:-1] >= 260.0) & (wl_edges[:-1] < 350.0)  # branch on the cell's lower edge
        return cls(base=base, wl_mid=wl_mid, band=band)

    def evaluate(self, temperature) -> np.ndarray:
        T = np.clip(np.asarray(temperature, dtype=float), 200.0, 400.0)  # (n_levels,)
        sumA = np.polyval(_H2O2_A, self.wl_mid)  # (n_wl,)
        sumB = np.polyval(_H2O2_B, self.wl_mid)
        chi = 1.0 / (1.0 + np.exp(-1265.0 / T))  # (n_levels,)
        param = (chi[:, None] * sumA[None, :] + (1.0 - chi)[:, None] * sumB[None, :]) * 1.0e-21
        out = np.repeat(self.base[None, :], T.size, axis=0)  # (n_levels, n_wl)
        out[:, self.band] = param[:, self.band]
        return out


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


def _rebin_params(td, wl_edges, lower_extrap=None, upper_extrap=None) -> np.ndarray:
    """Every parameter column of ``td`` padded and rebinned, shape ``(n_wl, n_params)``.

    Matches the base ``process_file`` in ``src/cross_section.F90``: the same extrapolation is
    applied to each parameter.
    """
    return np.stack(
        [_rebin_param(td.wavelength, td.parameters[:, p], wl_edges, lower_extrap, upper_extrap)
         for p in range(td.n_params)],
        axis=1,
    )


@dataclass
class CH2OCrossSection:
    """CH2O (formaldehyde) cross section [cm^2] (ports ``src/cross_sections/ch2o.F90``).

    ``sigma(T) = p1(lambda) + p2(lambda) * (T - 298 K)`` per level, with no temperature clamp. ``p1``
    and ``p2`` are the two parameters of the first file (``CH2O_1.nc``), each rebinned like the base
    cross section.
    """

    p: np.ndarray  # (n_wl, 2)

    @classmethod
    def from_file(cls, td, wl_edges, lower_extrap=None, upper_extrap=None):
        return cls(p=_rebin_params(td, wl_edges, lower_extrap, upper_extrap))

    def evaluate(self, temperature) -> np.ndarray:
        T = np.asarray(temperature, dtype=float)
        return self.p[None, :, 0] + self.p[None, :, 1] * (T[:, None] - 298.0)


@dataclass
class AcetoneCrossSection:
    """CH3COCH3 (acetone) -> CH3CO + CH3 cross section [cm^2] (ports
    ``src/cross_sections/acetone-ch3co_ch3.F90``).

    ``sigma = c1 * (1 + T * (c2 + T * (c3 + T * c4)))`` with ``T`` clamped to [235, 298] K; ``c1..c4``
    are the four parameters of the first file (``CH3COCH3_1.nc``), each rebinned like the base cross
    section.
    """

    c: np.ndarray  # (n_wl, 4)

    @classmethod
    def from_file(cls, td, wl_edges, lower_extrap=None, upper_extrap=None):
        if td.n_params != 4:
            raise ValueError("acetone cross section needs 4 parameters")  # as the Fortran asserts
        return cls(c=_rebin_params(td, wl_edges, lower_extrap, upper_extrap))

    def evaluate(self, temperature) -> np.ndarray:
        T = np.clip(np.asarray(temperature, dtype=float), 235.0, 298.0)[:, None]
        c1, c2, c3, c4 = (self.c[None, :, k] for k in range(4))
        return c1 * (1.0 + T * (c2 + T * (c3 + T * c4)))


def nitroxy_acetone_cross_section(wl_mid, n_levels) -> np.ndarray:
    """CH3COCH2(ONO2) (nitroxy-acetone) -> CH3COCH2(O.) + NO2 cross section [cm^2] (ports
    ``src/cross_sections/nitroxy_acetone.F90``).

    ``exp(c + lambda * (b + a * lambda))`` with ``a = -1.365e-3``, ``b = 0.7834``, ``c = -156.8`` at
    the cell midpoint for 284 <= lambda <= 335 nm, zero elsewhere; temperature-independent (no data
    files). Shape ``(n_levels, n_wl)``.
    """
    wc = np.asarray(wl_mid, dtype=float)
    a, b, c = -1.365e-3, 0.7834, -156.8
    band = (wc >= 284.0) & (wc <= 335.0)
    xs = np.zeros_like(wc)
    xs[band] = np.exp(c + wc[band] * (b + a * wc[band]))
    return np.repeat(xs[None, :], n_levels, axis=0)
