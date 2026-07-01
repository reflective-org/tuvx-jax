# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""JPL product-branching quantum yields for HNO4 and ClOOCl.

These channels are not Fortran-comparable (TUV-x carries only one channel each), so they are
validated by invariants instead: with ``branching=True`` the channel J-values must (a) partition the
faithful single-channel total (Σ channels = total, since the branching quantum yields sum to 1), and
(b) the primary channel must be scaled below the total. The branching quantum-yield helpers are also
checked against the JPL-recommended values (Table 4C-9-2; Section F7).
"""

from pathlib import Path

import numpy as np
import pytest

from tuvx_photolysis import PhotolysisCalculator
from tuvx_photolysis.quantum_yield import (
    hno4_branching_quantum_yield,
    clooocl_branching_quantum_yield,
)

REPO = Path(__file__).resolve().parents[1]
FIX = Path(__file__).resolve().parent / "fixtures"
CFG = FIX / "tuv_5_4_no_aerosol.json"


def test_hno4_branching_values():
    wl = np.array([180.0, 210.0, 300.0])  # below and above 200 nm
    a = hno4_branching_quantum_yield(wl, 1, "HO2+NO2")[0]
    b = hno4_branching_quantum_yield(wl, 1, "OH+NO3")[0]
    np.testing.assert_allclose(a, [0.7, 0.8, 0.8])  # Table 4C-9-2
    np.testing.assert_allclose(a + b, 1.0)


def test_clooocl_branching_values():
    wl = np.array([250.0, 300.0, 400.0])
    a = clooocl_branching_quantum_yield(wl, 1, "Cl+ClOO")[0]
    b = clooocl_branching_quantum_yield(wl, 1, "2ClO")[0]
    np.testing.assert_allclose(a, 0.8)  # Section F7
    np.testing.assert_allclose(a + b, 1.0)


@pytest.fixture(scope="module")
def calc():
    return PhotolysisCalculator.from_tuvx_json(CFG, data_root=REPO)


def test_branches_partition_the_total(calc):
    # Σ over channels equals the faithful single-channel (unit-QY) total, at every level.
    faithful = calc.rate_constants_profile(28.2, 0.996, branching=False)
    branched = calc.rate_constants_profile(28.2, 0.996, branching=True)
    for total_name, (c1, c2) in {
        "HNO4+hv->HO2+NO2": ("HNO4+hv->HO2+NO2", "HNO4+hv->OH+NO3"),
        "ClOOCl+hv->Cl+ClOO": ("ClOOCl+hv->Cl+ClOO", "ClOOCl+hv->ClO+ClO"),
    }.items():
        np.testing.assert_allclose(branched[c1] + branched[c2], faithful[total_name], rtol=1e-12)
        # the primary channel is reduced (it no longer carries the whole absorption)
        assert np.all(branched[c1] <= faithful[total_name] + 1e-30)
        assert np.any(branched[c1] < faithful[total_name])


def test_clooocl_primary_is_constant_fraction(calc):
    # ClOOCl uses a wavelength-independent 0.8/0.2 split -> exact scalar fractions of the total
    faithful = calc.rate_constants_profile(28.2, 0.996, branching=False)["ClOOCl+hv->Cl+ClOO"]
    branched = calc.rate_constants_profile(28.2, 0.996, branching=True)
    m = faithful > faithful.max() * 1e-3
    np.testing.assert_allclose(branched["ClOOCl+hv->Cl+ClOO"][m] / faithful[m], 0.8, rtol=1e-12)
    np.testing.assert_allclose(branched["ClOOCl+hv->ClO+ClO"][m] / faithful[m], 0.2, rtol=1e-12)


def test_branching_does_not_change_faithful_default(calc):
    # default (branching=False) is unchanged -> Fortran validation elsewhere still holds
    prof = calc.rate_constants_profile(28.2, 0.996)
    assert "HNO4+hv->OH+NO3" not in prof and "ClOOCl+hv->ClO+ClO" not in prof
