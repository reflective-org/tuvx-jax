# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Quantum-yield evaluation.

Reference: ``src/quantum_yield.F90``. The ``base`` quantum yield is either a constant value held
across wavelength, or a tabulated spectrum read from NetCDF and area-conservingly rebinned onto the
model wavelength grid (then replicated across altitude). As in the Fortran ``run``, only the first
parameter of the first file is used (e.g. ``CH2O_1.nc`` carries two channels; the base type takes
the H + HCO one). Reaction-specific quantum-yield recipes (e.g. the O3 photolysis branching) are
separate evaluators, each citing the ``src/quantum_yields`` module it ports.

The pressure-dependent recipes take the air number density at the height-grid interfaces
(``air_density_edge`` [molecule cm-3], the Fortran ``air`` profile ``edge_val_``) and evaluate
wavelength-dependent terms at the wavelength-cell midpoints (``wl_mid``), as the Fortran does.
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
    "o3_o1d_quantum_yield",
    "o3_o3p_quantum_yield",
    "rebin_quantum_yield_params",
    "ch2o_h2_co_quantum_yield",
    "ch3coch2ch3_quantum_yield",
    "ch3cocho_quantum_yield",
    "mvk_quantum_yield",
    "acetone_quantum_yield",
]

# Matsumi et al. (2002) O(1D) quantum-yield parameters (JGR 107, 10.1029/2001JD000510),
# shared by both O3 photolysis channels; see src/quantum_yields/o3-o2_o1d.F90.
_O3_A = np.array([0.8036, 8.9061, 0.1192])
_O3_X = np.array([304.225, 314.957, 310.737])
_O3_OM = np.array([5.576, 6.601, 2.187])


def o3_o1d_quantum_yield(wl_mid, temperature_edge) -> np.ndarray:
    """Φ for O3 + hv -> O2 + O(1D), Matsumi et al. (2002).

    Port of ``src/quantum_yields/o3-o2_o1d.F90``. Temperature-dependent, evaluated per interface
    (``temperature_edge`` = temperature at the height-grid edges). Returns ``(n_levels, n_wl)``.

    Wavelength regimes (λ in nm): λ ≤ 305 → 0.90; 305 < λ ≤ 328 → the Matsumi analytic form;
    328 < λ ≤ 340 → 0.08; λ > 340 → 0 (matches the Fortran ``where``/loop structure exactly).
    """
    w = np.asarray(wl_mid, dtype=float)
    T = np.asarray(temperature_edge, dtype=float)
    a, x, om = _O3_A, _O3_X, _O3_OM

    qy = np.zeros((T.size, w.size))
    lo = w <= 305.0
    mid = (w > 305.0) & (w <= 328.0)
    hi = (w > 328.0) & (w <= 340.0)
    lam = w[mid]
    for i, Ti in enumerate(T):
        kt = 0.695 * Ti
        q1, q2 = 1.0, np.exp(-825.518 / kt)
        qfac1, qfac2 = q1 / (q1 + q2), q2 / (q1 + q2)
        t300 = Ti / 300.0
        row = qy[i]
        row[lo] = 0.90
        row[hi] = 0.08
        row[mid] = (0.0765
                    + a[0] * qfac1 * np.exp(-((x[0] - lam) / om[0]) ** 4)
                    + a[1] * t300 * t300 * qfac2 * np.exp(-((x[1] - lam) / om[1]) ** 2)
                    + a[2] * t300 ** 1.5 * np.exp(-((x[2] - lam) / om[2]) ** 2))
    return qy


def o3_o3p_quantum_yield(wl_mid, temperature_edge) -> np.ndarray:
    """Φ for O3 + hv -> O2 + O(3P) = 1 − Φ(O(1D)); see src/quantum_yields/o3-o2_o3p.F90."""
    return 1.0 - o3_o1d_quantum_yield(wl_mid, temperature_edge)


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

    NOTE (faithfulness): this is a deliberate extension, opt-in via ``rate_constants(..., branching=True)``.
    Fortran TUV-x carries a single ClOOCl channel with unit quantum yield, so branching output is NOT
    Fortran-comparable. The 2ClO channel is also new relative to the reference MATLAB box model (which
    had only *thermal* ClOOCl->ClO+ClO plus photolytic ClOOCl->2Cl); enabling it therefore changes
    ClOOCl chemistry versus the reference run. The 0.8/0.2 split is wavelength-independent per JPL §F7;
    confirm the value against your JPL 19-5 copy before relying on it for ozone-loss magnitudes.
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


def rebin_quantum_yield_params(
    td: TabulatedData, wl_edges, lower_extrapolation=None, upper_extrapolation=None,
    lower_value=0.0, upper_value=0.0,
) -> np.ndarray:
    """Every parameter column of a quantum-yield file rebinned onto the grid, ``(n_wl, n_params)``.

    Ports the file loop of ``base_constructor`` in ``src/quantum_yield.F90``: each column is padded
    with the quantum yield's ``lower``/``upper extrapolation`` (:func:`add_points`; ``boundary`` holds
    that column's own edge value) and conservingly rebinned.
    """
    from .cross_section import add_points

    cols = []
    for p in range(td.n_params):
        x, y = add_points(
            td.wavelength, td.parameters[:, p],
            lower_extrapolation, upper_extrapolation, lower_value, upper_value,
        )
        cols.append(interp_conserving(np.asarray(wl_edges, dtype=float), x, y))
    return np.stack(cols, axis=1)


def ch2o_h2_co_quantum_yield(params, wl_mid, temperature_edge, air_density_edge) -> np.ndarray:
    """Φ for CH2O + hv -> H2 + CO, pressure- and temperature-dependent (ports
    ``src/quantum_yields/ch2o.F90``, ``quantum_yield_ch2o_h2_co_t``).

    ``params`` is ``(n_wl, 2)`` from :func:`rebin_quantum_yield_params` on ``CH2O_1.nc``: column 0 is
    Φ1 (H + HCO), column 1 is Φ2 (H2 + CO, zero-pressure). With ``M`` the air density and
    ``Tf = (300 - T) / 80``, for cells whose midpoint is in [330, 360) nm and where Φ2 > 0::

        k   = (1 - (Φ1 + Φ2)) / (2.45e19 * Φ2 * (1 - Φ1)) * (1 + 0.05 * (λ - 329) * Tf)
        Φ   = 1 / (1 / (1 - Φ1) + k * M)

    and Φ = Φ2 elsewhere. Returns ``(n_levels, n_wl)``.
    """
    wl = np.asarray(wl_mid, dtype=float)
    T = np.asarray(temperature_edge, dtype=float)
    M = np.asarray(air_density_edge, dtype=float)
    q1, q2 = params[:, 0], params[:, 1]
    out = np.repeat(q2[None, :], T.size, axis=0)
    band = (wl >= 330.0) & (wl < 360.0) & (q2 > 0.0)
    tmp = 1.0 - q1[band]
    wrk = (1.0 - (q1[band] + q2[band])) / (2.45e19 * q2[band] * tmp)  # (n_band,)
    t_factor = (300.0 - T) / 80.0  # (n_levels,)
    wrk = wrk[None, :] * (1.0 + 0.05 * (wl[band][None, :] - 329.0) * t_factor[:, None])
    out[:, band] = 1.0 / (1.0 / tmp[None, :] + wrk * M[:, None])
    return out


def ch3coch2ch3_quantum_yield(air_density_edge, n_wl) -> np.ndarray:
    """Φ for CH3COCH2CH3 (MEK) + hv -> CH3CO + CH2CH3 (ports
    ``src/quantum_yields/ch3coch2ch3-ch3co_ch2ch3.F90``).

    Wavelength-independent Stern-Volmer form with ``p = 760 * M / 2.69e19`` [torr]::

        Φ = min(1, 1 / (0.96 + 2.22e-3 * p))

    Returns ``(n_levels, n_wl)``.
    """
    M = np.asarray(air_density_edge, dtype=float)
    ptorr = 760.0 * M / 2.69e19
    phi = np.minimum(1.0 / (0.96 + 2.22e-3 * ptorr), 1.0)
    return np.repeat(phi[:, None], n_wl, axis=1)


def ch3cocho_quantum_yield(wl_mid, air_density_edge) -> np.ndarray:
    """Φ for CH3COCHO (methylglyoxal) + hv -> CH3CO + HCO (ports
    ``src/quantum_yields/ch3cocho.F90``; quenching from Koch and Moortgat).

    ``phi0 = clip(1 - (λ - 380) / 60, 0, 1)`` (1 below 380 nm, 0 above 440 nm). For 380 <= λ <= 440 nm
    it is quenched by air, with ``kq = 1.36e8 * exp(-8793 / λ)`` [torr-1] and
    ``p = M * 760 / 2.456e19`` [torr]::

        Φ = phi0 / (phi0 + kq * p)

    and Φ = phi0 elsewhere. Returns ``(n_levels, n_wl)``. Faithful to the Fortran: although its
    comments call ``phi0`` the zero-pressure yield, this form tends to 1 (not ``phi0``) as p -> 0
    inside the band.
    """
    w = np.asarray(wl_mid, dtype=float)
    M = np.asarray(air_density_edge, dtype=float)
    airfac = M * 760.0 / 2.456e19  # (n_levels,)
    phi0 = np.clip(1.0 - (w - 380.0) / 60.0, 0.0, 1.0)
    kq = 1.36e8 * np.exp(-8793.0 / w)
    out = np.repeat(phi0[None, :], M.size, axis=0)
    quench = (w >= 380.0) & (w <= 440.0) & (phi0 > 0.0)
    out[:, quench] = phi0[quench] / (phi0[quench] + kq[quench] * airfac[:, None])
    return out


def mvk_quantum_yield(wl_mid, air_density_edge) -> np.ndarray:
    """Φ for MVK (methyl vinyl ketone) + hv -> Products (ports ``src/quantum_yields/mvk.F90``).

    TUV-x carries one total (all-channel) yield::

        Φ = min(1, exp(-0.055 * (λ - 308)) / (5.5 + 9.2e-19 * M))

    Returns ``(n_levels, n_wl)``.
    """
    w = np.asarray(wl_mid, dtype=float)
    M = np.asarray(air_density_edge, dtype=float)
    divisor = 5.5 + 9.2e-19 * M  # (n_levels,)
    return np.minimum(np.exp(-0.055 * (w - 308.0))[None, :] / divisor[:, None], 1.0)


_ACETONE_BRANCHES = ("CO", "CH3CO", "CO+CH3CO")


def acetone_quantum_yield(
    wl_mid, temperature_edge, air_density_edge,
    low_wavelength_value=0.95, high_wavelength_value=0.0,
    minimum_temperature=0.0, maximum_temperature=np.inf, branch="CH3CO",
) -> np.ndarray:
    """Φ for CH3COCH3 (acetone) + hv -> CH3CO + CH3, Blitz et al. (2004, GRL 31, L06111) (ports
    ``src/quantum_yields/acetone-ch3co_ch3.F90``).

    The keyword arguments are the Fortran config options and defaults (``maximum temperature``
    defaults to ``huge``). With ``T' = clip(T, minimum_temperature, maximum_temperature) / 295``,
    per cell midpoint λ [nm]:

    * λ < 279 nm: ``low_wavelength_value``; λ > 327 nm: ``high_wavelength_value``;
    * otherwise the CO yield ``fco = 1 / (1 + cA0)`` and the CH3CO yield ``fac`` (separate forms for
      279 <= λ < 302 nm and 302 <= λ <= 327 nm, pressure-dependent through the air density M), and
      Φ is ``fac`` (``branch="CH3CO"``), ``fco`` (``"CO"``) or their sum (``"CO+CH3CO"``).

    ``cA0`` is set to 5e34 when its exponent exceeds 80, as in the Fortran. Returns
    ``(n_levels, n_wl)``.
    """
    if branch not in _ACETONE_BRANCHES:
        raise ValueError(f"invalid acetone quantum yield branch: {branch!r}")
    w = np.asarray(wl_mid, dtype=float)
    T = np.asarray(temperature_edge, dtype=float)
    M = np.asarray(air_density_edge, dtype=float)
    out = np.zeros((T.size, w.size))
    out[:, w < 279.0] = low_wavelength_value
    out[:, w > 327.0] = high_wavelength_value
    mid = (w >= 279.0) & (w <= 327.0)
    wm = w[mid]
    wadj = 1.0e7 / wm
    b1_band = wm < 302.0  # 279 <= λ < 302; the rest of `mid` is 302 <= λ <= 327
    for i in range(T.size):
        t = max(minimum_temperature, min(maximum_temperature, T[i])) / 295.0
        m = M[i]
        # CO (carbon monoxide) quantum yield
        a0 = 0.350 * t ** (-1.28)
        b0 = 0.068 * t ** (-2.65)
        dumexp = b0 * (wm - 248.0)
        cA0 = np.where(dumexp > 80.0, 5.0e34, np.exp(np.minimum(dumexp, 80.0)) * a0 / (1.0 - a0))
        fco = 1.0 / (1.0 + cA0)
        # CH3CO (acetyl radical) quantum yield
        fac = np.empty_like(wm)
        x, f = wadj[b1_band], fco[b1_band]
        a1 = 1.600e-19 * t ** (-2.38)
        b1 = 0.55e-3 * t ** (-3.19)
        cA1 = a1 * np.exp(-b1 * (x - 33113.0))
        fac[b1_band] = (1.0 - f) / (1.0 + cA1 * m)
        x, f = wadj[~b1_band], fco[~b1_band]
        a2 = 1.62e-17 * t ** (-10.03)
        b2 = 1.79e-3 * t ** (-1.364)
        cA2 = a2 * np.exp(-b2 * (x - 30488.0))
        a3 = 26.29 * t ** (-6.59)
        b3 = 5.72e-7 * t ** (-2.93)
        c3 = 30006.0 * t ** (-0.064)
        cA3 = a3 * np.exp(-b3 * (x - c3) ** 2)
        a4 = 1.67e-15 * t ** (-7.25)
        b4 = 2.08e-3 * t ** (-1.16)
        cA4 = a4 * np.exp(-b4 * (x - 30488.0))
        fac[~b1_band] = (1.0 - f) * (1.0 + cA3 + cA4 * m) / ((1.0 + cA3 + cA2 * m) * (1.0 + cA4 * m))
        qy = np.zeros_like(wm)
        if branch in ("CO", "CO+CH3CO"):
            qy = qy + fco
        if branch in ("CH3CO", "CO+CH3CO"):
            qy = qy + fac
        out[i, mid] = qy
    return out


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
