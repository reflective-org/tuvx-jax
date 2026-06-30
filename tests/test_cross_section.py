# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a: cross-section evaluation, with the O3 cross section validated against the Fortran."""

from pathlib import Path

import numpy as np
import pytest

from tuvx_photolysis import data, cross_section
from tuvx_photolysis.grids import interp_conserving

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
FIX = Path(__file__).resolve().parent / "fixtures"
REF = FIX / "tuv_5_4_no_aerosol_reference.nc"

pytestmark = pytest.mark.skipif(
    not REF.exists(),
    reason="Fortran reference missing; run tests/fixtures/regenerate_reference.sh",
)


def _wl_edges():
    return data.load_wavelength_grid(DATA / "grids" / "wavelength" / "combined.grid")


def test_refraction_near_one():
    # refractive index of air is ~1.0003 in the UV/visible
    n = cross_section.refraction(np.array([200.0, 300.0, 500.0]))
    assert np.all(n > 1.0)
    assert np.all(n < 1.001)


def test_add_points_pads_to_full_range_with_zero():
    x = np.array([300.0, 310.0, 320.0])
    y = np.array([1.0, 2.0, 3.0])
    xp, yp = cross_section.add_points(x, y)
    assert xp[0] == 0.0  # padded down to 0
    assert xp[-1] == 1.0e38  # and up to a very large wavelength
    assert yp[0] == 0.0 and yp[-1] == 0.0  # zero outside the data (no extrapolation)
    assert np.all(np.diff(xp) > 0)


def test_add_points_boundary_holds_edge_value():
    x = np.array([300.0, 310.0])
    y = np.array([5.0, 7.0])
    xp, yp = cross_section.add_points(x, y, lower_extrapolation="boundary")
    # the point just below the data range holds the boundary value
    below = yp[xp < 300.0]
    assert np.all(below == 5.0)


def test_base_cross_section_rebins_and_replicates():
    td = data.load_cross_section(DATA / "cross_sections" / "HNO4_1.nc")
    wl_edges = _wl_edges()
    xs = cross_section.BaseCrossSection.from_files([(td, None, None)], wl_edges)
    field = xs.evaluate(n_levels=5)
    assert field.shape == (5, wl_edges.size - 1)
    # every level identical (temperature-independent base type)
    np.testing.assert_allclose(field[0], field[4])
    # matches a direct conserving rebin of the padded data
    xp, yp = cross_section.add_points(td.wavelength, td.parameters[:, 0])
    np.testing.assert_allclose(field[0], interp_conserving(wl_edges, xp, yp))


def test_o3_cross_section_matches_fortran():
    import netCDF4

    wl_edges = _wl_edges()
    o3_files = [
        data.load_cross_section(DATA / "cross_sections" / f"O3_{i}.nc") for i in range(1, 5)
    ]
    o3 = cross_section.O3TintCrossSection.from_files(o3_files, wl_edges)

    with netCDF4.Dataset(REF) as ds:
        # reference O3 cross section: (wavelength, vertical_level, time); take time 0
        ref_xs = np.array(ds.variables["cross section O3+hv->O2+O(1D)"][:])[:, :, 0]  # (nwl, nlev)
        temperature = np.array(ds.variables["temperature"][:])[:, 0]  # (nlev,) at interfaces

    got = o3.evaluate(temperature)  # (nlev, nwl)
    ref = ref_xs.T  # (nlev, nwl)

    # compare where the reference is non-zero (O3 absorbs in the UV)
    mask = ref > 0
    assert mask.sum() > 1000
    rel = np.abs(got[mask] - ref[mask]) / ref[mask]
    # the o3_tint port reproduces the Fortran to floating-point round-off
    assert np.median(rel) < 1e-12
    assert np.percentile(rel, 99) < 1e-10
    assert rel.max() < 1e-8
    # zeros stay zero (no spurious absorption where the Fortran has none)
    np.testing.assert_allclose(got[~mask], 0.0, atol=1e-30)
