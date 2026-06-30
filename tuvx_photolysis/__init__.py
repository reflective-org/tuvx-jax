# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""JAX port of the TUV-x actinic-flux and photolysis-rate-constant calculation.

This package reimplements the "Option A" pipeline of TUV-x (NCAR) in Python/JAX:
given altitude / latitude / longitude / time-of-year and per-reaction cross sections
and quantum yields, it solves the radiation field with a delta-Eddington two-stream
solver and integrates ``J = sum_lambda F(lambda, z) * sigma(lambda, T) * phi(lambda)``
for each photolysis reaction.

The Fortran sources under ``../src`` are the authoritative reference for all physics.

Public API (built up across the staged port):
    PhotolysisCalculator -- top-level orchestrator (see ``api.py``)
"""

__version__ = "0.1.0"

# Match the Fortran reference, which is double precision. Must run before any JAX array is created.
import jax as _jax

_jax.config.update("jax_enable_x64", True)

from .api import PhotolysisCalculator  # noqa: E402,F401

__all__ = ["__version__", "PhotolysisCalculator"]
