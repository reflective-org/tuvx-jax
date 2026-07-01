# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Lyman-alpha / Schumann-Runge band parameterization, validated against the Fortran.

With LA/SR enabled, the radiation field should match the Fortran reference across the *entire*
spectrum (including < 206 nm), and the O2 photolysis rate constant should match too.
"""

from pathlib import Path

import numpy as np
import pytest

from tuvx_photolysis import PhotolysisCalculator, data
from tuvx_photolysis.la_sr_bands import LaSrBands, _chebyshev

REPO = Path(__file__).resolve().parents[1]
FIX = Path(__file__).resolve().parent / "fixtures"
CFG = FIX / "tuv_5_4_no_aerosol.json"
REF = FIX / "tuv_5_4_no_aerosol_reference.nc"

pytestmark = pytest.mark.skipif(
    not REF.exists(), reason="Fortran reference missing; run fixtures/regenerate_reference.sh"
)


def test_bands_located_on_grid():
    wl = data.load_wavelength_grid(REPO / "data" / "grids" / "wavelength" / "combined.grid")
    la = LaSrBands.from_file(wl, REPO / "data" / "cross_sections" / "O2_parameters.txt")
    assert la.has_la and la.has_srb
    assert wl[la.ila] == pytest.approx(121.4)
    assert wl[la.isrb] == pytest.approx(175.4)
    assert la.ac.shape == (20, 17) and la.bc.shape == (20, 17)


def test_chebyshev_matches_numpy():
    # our Clenshaw evaluation must equal numpy's Chebyshev on the mapped interval
    rng = np.random.default_rng(0)
    coeffs = rng.normal(size=20)
    for x in (38.0, 44.0, 50.0, 56.0):
        y = (2 * x - (38.0 + 56.0)) / (56.0 - 38.0)
        c = coeffs.copy()
        c[0] *= 0.5  # numpy chebval has no 0.5 on c0
        expected = np.polynomial.chebyshev.chebval(y, c)
        assert _chebyshev(coeffs, x) == pytest.approx(expected, rel=1e-12)


@pytest.fixture(scope="module")
def calc():
    return PhotolysisCalculator.from_tuvx_json(CFG, data_root=REPO)


def test_radiation_field_matches_fortran_including_deep_uv(calc):
    import netCDF4

    with netCDF4.Dataset(REF) as ds:
        sza = np.array(ds.variables["solar zenith angle"][:])
        esd = np.array(ds.variables["Earth-Sun distance"][:])
        ref = {k: np.array(ds.variables[v][:]) for k, v in
               [("dir", "direct radiation"), ("dn", "downward radiation"), ("up", "upward radiation")]}

    for ti in range(sza.size):
        rf = calc.radiation_field(sza[ti])
        fields = {"dir": rf.fdr, "dn": rf.fdn, "up": rf.fup}
        for key, tol in [("dir", 1e-9), ("up", 1e-7), ("dn", 1e-4)]:
            mine = np.asarray(fields[key]) * esd[ti]
            r = ref[key][:, :, ti].T
            m = r > r.max() * 1e-8  # NB: no wavelength mask -- the deep UV must match now
            rel = np.abs(mine[m] - r[m]) / r[m]
            assert np.median(rel) < 1e-6, f"t{ti} {key} median {np.median(rel):.2e}"
            assert rel.max() < tol, f"t{ti} {key} max {rel.max():.2e}"


def test_o2_photolysis_matches_fortran(calc):
    import netCDF4

    assert calc.o2_reaction is not None
    with netCDF4.Dataset(REF) as ds:
        sza = np.array(ds.variables["solar zenith angle"][:])
        esd = np.array(ds.variables["Earth-Sun distance"][:])
        for ti in range(sza.size):
            J = calc.rate_constants_profile(sza[ti], esd[ti])["O2+hv->O+O"]
            ref = np.array(ds.variables["O2+hv->O+O"][:])[:, ti]
            m = ref > ref.max() * 1e-3
            rel = np.abs(J[m] - ref[m]) / ref[m]
            assert rel.max() < 1e-5, f"t{ti} O2 max rel {rel.max():.2e}"
