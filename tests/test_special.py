# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Stage 9a: reaction-specific cross-section/QY modules validated against the Fortran reference.

These are the special evaluators frank-model's stratospheric reactions need (Cl2, ClONO2, HNO3,
N2O5, NO2-tint, OClO, HOBr cross sections; NO3 tint and ClONO2 branching quantum yields). Each
reaction whose photolysis is dominated by lambda >= ~206 nm must match the Fortran J to machine
precision.
"""

from pathlib import Path

import numpy as np
import pytest

from tuvx_photolysis import PhotolysisCalculator, photolysis
from tuvx_photolysis.quantum_yield import clono2_quantum_yield

REPO = Path(__file__).resolve().parents[1]
FIX = Path(__file__).resolve().parent / "fixtures"
CFG = FIX / "tuv_5_4_no_aerosol.json"
REF = FIX / "tuv_5_4_no_aerosol_reference.nc"

pytestmark = pytest.mark.skipif(
    not REF.exists(), reason="Fortran reference missing; run fixtures/regenerate_reference.sh"
)

# special-module reactions dominated by lambda >= ~206 nm (negligible deferred-LA/SR contribution)
MACHINE_PRECISION_REACTIONS = [
    "Cl2+hv->Cl+Cl",            # cl2 analytic T-dependent xs
    "NO2+hv->NO+O(3P)",         # NO2-tint xs + NO2-tint (extrapolating) QY
    "NO3+hv->NO2+O(3P)",        # base xs + tint QY
    "NO3+hv->NO+O2",            # base xs + tint QY
    "OClO+hv->Products",        # OClO per-file temperature interpolation
    "HOBr+hv->OH+Br",           # hobr analytic xs
    "BrCl+hv->Br+Cl",
]


@pytest.fixture(scope="module")
def calc():
    return PhotolysisCalculator.from_tuvx_json(CFG, data_root=REPO)


def test_special_reactions_machine_precision(calc):
    import netCDF4

    with netCDF4.Dataset(REF) as ds:
        sza = np.array(ds.variables["solar zenith angle"][:])
        esd = np.array(ds.variables["Earth-Sun distance"][:])
        ref = {r: np.array(ds.variables[r][:]) for r in MACHINE_PRECISION_REACTIONS}

    for ti in range(sza.size):
        J = calc.rate_constants_profile(sza[ti], esd[ti])
        for name in MACHINE_PRECISION_REACTIONS:
            assert name in calc.xsqy, f"{name} not covered"
            r = ref[name][:, ti]
            m = r > r.max() * 1e-3
            rel = np.abs(J[name][m] - r[m]) / r[m]
            assert rel.max() < 1e-5, f"t{ti} {name} max rel {rel.max():.2e}"


def test_clono2_branches_sum_to_one():
    # the two ClONO2 branches partition unity at every wavelength
    wl = np.linspace(200.0, 450.0, 60)
    a = clono2_quantum_yield(wl, 1, "Cl+NO3")
    b = clono2_quantum_yield(wl, 1, "ClO+NO2")
    np.testing.assert_allclose(a + b, 1.0)
    # below 308 nm the Cl+NO3 branch is 0.6
    assert a[0, wl < 308][0] == pytest.approx(0.6)


def test_no2_tint_qy_extrapolates_below_range(calc):
    # NO2 J should match the Fortran even where the atmospheric T is below the QY's tabulated
    # range (this is the temperature-extrapolation path, which previously caused a ~1% error).
    import netCDF4

    with netCDF4.Dataset(REF) as ds:
        sza = np.array(ds.variables["solar zenith angle"][:])
        esd = np.array(ds.variables["Earth-Sun distance"][:])
        ref = np.array(ds.variables["NO2+hv->NO+O(3P)"][:])[:, 0]
        temperature = np.array(ds.variables["temperature"][:])[:, 0]
    J = calc.rate_constants_profile(sza[0], esd[0])["NO2+hv->NO+O(3P)"]
    cold = temperature < 248.0  # below the NO2 quantum-yield temperature range
    assert cold.sum() > 10
    m = cold & (ref > ref.max() * 1e-3)
    assert np.max(np.abs(J[m] - ref[m]) / ref[m]) < 1e-5
