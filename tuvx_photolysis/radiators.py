# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Radiator optical properties and accumulation.

Reference: ``src/radiative_transfer/radiator.F90`` (``update_state`` and ``accumulate``),
``src/cross_sections/rayliegh.F90``, ``src/radiative_transfer/radiators/aerosol.F90``.

Each radiator contributes a per-layer, per-wavelength optical depth (OD), single-scattering albedo
(SSA), and asymmetry factor (G). They are combined by :func:`accumulate` into the total
``(OD, SSA, G)`` the delta-Eddington solver consumes. This targets the **delta-Eddington** path,
which uses a single asymmetry stream (``layer_G_(:, :, 1)`` in the Fortran); accordingly Rayleigh
(air) scattering has asymmetry 0 and the multi-stream ``precis`` clamps do not apply.

Optical-property arrays are shaped ``(n_layers, n_wavelengths)`` to match the Fortran
``layer_OD_`` etc.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "RadiatorOpticalProps",
    "rayleigh_cross_section",
    "absorber_radiator",
    "air_radiator",
    "aerosol_radiator",
    "accumulate",
]

_LARGEST = 1.0e36
_KFLOOR = 1.0 / _LARGEST  # smallest radiative property (src/radiative_transfer/radiator.F90)


@dataclass
class RadiatorOpticalProps:
    """Per-layer, per-wavelength optical properties of a single radiator (or the accumulation).

    ``optical_depth``, ``single_scattering_albedo``, ``asymmetry_factor`` all have shape
    ``(n_layers, n_wavelengths)``. ``is_air`` flags Rayleigh scattering for the accumulation.
    """

    optical_depth: np.ndarray
    single_scattering_albedo: np.ndarray
    asymmetry_factor: np.ndarray
    is_air: bool = False

    def __post_init__(self):
        self.optical_depth = np.asarray(self.optical_depth, dtype=float)
        self.single_scattering_albedo = np.broadcast_to(
            np.asarray(self.single_scattering_albedo, dtype=float), self.optical_depth.shape
        ).copy()
        self.asymmetry_factor = np.broadcast_to(
            np.asarray(self.asymmetry_factor, dtype=float), self.optical_depth.shape
        ).copy()


def rayleigh_cross_section(wavelength_nm: np.ndarray) -> np.ndarray:
    """Rayleigh scattering cross section [cm^2] vs wavelength [nm].

    WMO (1985) / Nicolet empirical formula, ported from ``src/cross_sections/rayliegh.F90``:
    with ``w = lambda[micron]``, ``sigma = 4.02e-28 / w**p`` where
    ``p = 3.6772 + 0.389 w + 0.09426 / w`` for ``w <= 0.55`` else ``p = 4.04``.
    """
    w = 1.0e-3 * np.asarray(wavelength_nm, dtype=float)  # nm -> micron
    pwr = np.where(w <= 0.55, 3.6772 + 0.389 * w + 0.09426 / np.where(w == 0, np.nan, w), 4.04)
    return 4.02e-28 / w**pwr


def absorber_radiator(layer_density: np.ndarray, cross_section: np.ndarray) -> RadiatorOpticalProps:
    """A pure-absorber radiator (e.g. O2, O3).

    ``layer_density`` is the per-layer column density [molecule cm-2], shape ``(n_layers,)``.
    ``cross_section`` is [cm^2], shape ``(n_layers, n_wavelengths)``. OD = density * sigma,
    SSA = 0, G = 0 (ports the absorber branch of ``update_state``).
    """
    layer_density = np.asarray(layer_density, dtype=float)
    cross_section = np.asarray(cross_section, dtype=float)
    od = layer_density[:, None] * cross_section
    return RadiatorOpticalProps(od, 0.0, 0.0, is_air=False)


def air_radiator(layer_density: np.ndarray, rayleigh_xs: np.ndarray) -> RadiatorOpticalProps:
    """The air (Rayleigh) radiator: OD = density * sigma_rayleigh(lambda), SSA = 1, G = 0.

    ``layer_density`` shape ``(n_layers,)`` [molecule cm-2]; ``rayleigh_xs`` shape
    ``(n_wavelengths,)`` [cm^2]. Ports the ``is_air`` branch of ``update_state``.
    """
    layer_density = np.asarray(layer_density, dtype=float)
    rayleigh_xs = np.asarray(rayleigh_xs, dtype=float)
    od = layer_density[:, None] * rayleigh_xs[None, :]
    return RadiatorOpticalProps(od, 1.0, 0.0, is_air=True)


def aerosol_radiator(
    optical_depth: np.ndarray,
    single_scattering_albedo,
    asymmetry_factor,
    n_wavelengths: int,
) -> RadiatorOpticalProps:
    """A generic aerosol radiator from explicit per-layer OD and (scalar or per-layer) SSA/G.

    ``optical_depth`` is per layer [unitless], shape ``(n_layers,)``, assumed wavelength-independent
    (as in ``aerosol.F90``'s default path); it is broadcast across ``n_wavelengths``. SSA and G may
    be scalars or per-layer arrays.

    Note: this is the generic form. Reproducing ``aerosol.F90``'s exact config parsing (vertical
    fractional-source interpolation of the input OD profile, its OD-weighted SSA handling) is left
    for a follow-up; see :func:`tuvx_photolysis.grids.interp_fractional_source`.
    """
    optical_depth = np.asarray(optical_depth, dtype=float)
    od = np.repeat(optical_depth[:, None], n_wavelengths, axis=1)
    return RadiatorOpticalProps(od, single_scattering_albedo, asymmetry_factor, is_air=False)


def accumulate(radiators: list[RadiatorOpticalProps]) -> RadiatorOpticalProps:
    """Combine radiators into the total optical properties for the delta-Eddington solver.

    Ports ``accumulate`` in ``src/radiative_transfer/radiator.F90`` for the single-stream
    (delta-Eddington) case:
      dscat = OD * SSA;  dabs = OD * (1 - SSA)
      OD_total = sum(dscat) + sum(dabs)        (each sum floored at kfloor = 1/largest)
      SSA_total = sum(dscat) / OD_total        (kfloor where there is no scattering)
      G_total = sum(g * dscat) / sum(dscat)    (air contributes g = 0)
    """
    if not radiators:
        raise ValueError("need at least one radiator")
    shape = radiators[0].optical_depth.shape
    dscat_accum = np.zeros(shape)
    dabs_accum = np.zeros(shape)
    asym_accum = np.zeros(shape)

    for r in radiators:
        dscat = r.optical_depth * r.single_scattering_albedo
        dscat_accum += dscat
        dabs_accum += r.optical_depth * (1.0 - r.single_scattering_albedo)
        if not r.is_air:
            asym_accum += r.asymmetry_factor * dscat
        # air (Rayleigh) contributes asymmetry 0 in the single-stream delta-Eddington path

    dscat_accum = np.maximum(dscat_accum, _KFLOOR)
    dabs_accum = np.maximum(dabs_accum, _KFLOOR)
    od_total = dscat_accum + dabs_accum
    ssa_total = np.where(dscat_accum == _KFLOOR, _KFLOOR, dscat_accum / od_total)
    g_total = asym_accum / dscat_accum

    return RadiatorOpticalProps(od_total, ssa_total, g_total, is_air=False)
