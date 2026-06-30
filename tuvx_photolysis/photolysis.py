# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Photolysis-rate-constant integration.

Reference: ``src/photolysis_rates.F90`` (``get``, lines 321-381).

The actinic flux is assembled from the radiation field and the extraterrestrial flux,

    actinicFlux(lambda, z) = (fdr + fup + fdn)(z, lambda) * etfl(lambda)    (clamped at >= 0)

and each reaction's rate constant is the wavelength sum (rectangular quadrature; the bin width is
folded into ``etfl``)

    J(z) = sum_lambda actinicFlux(z, lambda) * sigma(z, lambda) * phi(z, lambda).
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

__all__ = ["actinic_flux", "rate_constant"]


def actinic_flux(fdr, fdn, fup, etfl):
    """Total actinic flux ``(fdr + fup + fdn) * etfl`` [photon cm-2 s-1], clamped at >= 0.

    ``fdr``/``fdn``/``fup`` are ``(n_levels, n_wavelengths)`` (already scaled by any Earth-Sun
    distance factor); ``etfl`` is per wavelength bin ``(n_wavelengths,)``.
    """
    field = np.asarray(fdr) + np.asarray(fdn) + np.asarray(fup)
    flux = field * np.asarray(etfl)[None, :]
    return np.where(flux < 0.0, 0.0, flux)


def rate_constant(actinic, cross_section, quantum_yield) -> np.ndarray:
    """Photolysis rate constant per level [s-1] = sum over wavelength of flux * sigma * phi.

    All inputs are ``(n_levels, n_wavelengths)``. Returns ``(n_levels,)``.
    """
    return np.sum(np.asarray(actinic) * np.asarray(cross_section) * np.asarray(quantum_yield), axis=1)
