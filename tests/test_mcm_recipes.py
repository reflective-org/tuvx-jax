# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Tropospheric (MCM) photolysis recipes validated against the Fortran TUV-x.

Reference: ``fixtures/tuv_5_4_no_aerosol_sza_reference.nc``, written by the Fortran ``tuv-x`` for
``fixtures/tuv_5_4_no_aerosol_sza.json`` (the validated no-aerosol config at fixed SZA 0/30/60/80
deg, Earth-Sun distance 1 AU, with sigma/phi diagnostics); see
``fixtures/regenerate_sza_reference.py``.

For every reaction in :data:`REACTIONS`:

* the cross section and quantum yield arrays (all 121 levels x 156 wavelength cells) match the
  Fortran diagnostics to round-off (``rtol`` :data:`ARRAY_RTOL`);
* J at 0-15 km (and so at the surface) matches the Fortran at each SZA to :data:`J_RTOL`. The
  residual is the ~1.75e-8 relative offset every reaction (also the long-validated ones) shows
  against the Fortran, independent of SZA and reaction.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from tuvx_photolysis import PhotolysisCalculator, api, data, profiles

REPO = Path(__file__).resolve().parents[1]
FIX = Path(__file__).resolve().parent / "fixtures"
CFG = FIX / "tuv_5_4_no_aerosol.json"
REF = FIX / "tuv_5_4_no_aerosol_sza_reference.nc"

pytestmark = pytest.mark.skipif(
    not REF.exists(), reason="Fortran reference missing; run fixtures/regenerate_sza_reference.py"
)

ARRAY_RTOL = 1e-12
J_RTOL = 1e-7
TROPOSPHERE = slice(0, 16)  # 0..15 km on the 1 km height grid

#: Reactions ported for the MCM mechanism (and reactions that share their recipes).
REACTIONS = [
    "CH2O+hv->H+HCO",                    # ch2o xs + base QY (parameter 1 of CH2O_1.nc)
    "CH2O+hv->H2+CO",                    # ch2o xs + ch2o (H2+CO) QY
]


@pytest.fixture(scope="module")
def ref():
    import netCDF4

    with netCDF4.Dataset(REF) as ds:
        return {name: np.array(ds.variables[name][:]) for name in ds.variables}


@pytest.fixture(scope="module")
def calc():
    return PhotolysisCalculator.from_tuvx_json(CFG, data_root=REPO)


@pytest.fixture(scope="module")
def env():
    """Inputs of the per-reaction dispatch (as built by ``from_tuvx_json``)."""
    cfg = json.loads(CFG.read_text())
    wl_edges = data.load_wavelength_grid(REPO / "data/grids/wavelength/combined.grid")
    z = np.arange(0.0, 121.0)
    atm = REPO / "data/profiles/atmosphere"
    return dict(
        reactions={r["name"]: r for r in cfg["photolysis"]["reactions"]},
        rel=lambda p: REPO / p,
        wl_edges=wl_edges,
        wl_mid=0.5 * (wl_edges[:-1] + wl_edges[1:]),
        temperature=profiles.temperature_profile(z, data.load_profile_csv(atm / "ussa.temp")),
        air=profiles.air_profile(z, data.load_profile_csv(atm / "ussa.dens")),
        n_lev=z.size,
    )


def _assert_matches_fortran(got, fortran, what):
    # Fortran diagnostics are (wavelength, level); ours are (level, wavelength)
    np.testing.assert_allclose(got, fortran.T, rtol=ARRAY_RTOL, atol=0.0, err_msg=what)


@pytest.mark.parametrize("name", REACTIONS)
def test_supported(calc, name):
    assert name in calc.xsqy, calc.skipped_reactions.get(name)
    assert name not in calc.skipped_reactions


@pytest.mark.parametrize("name", REACTIONS)
def test_cross_section_matches_fortran(env, ref, name):
    e = env
    sigma = api._eval_reaction_xs(
        e["reactions"][name]["cross section"], e["rel"], e["wl_edges"], e["wl_mid"],
        e["temperature"], e["n_lev"],
    )
    _assert_matches_fortran(sigma, ref[f"cross section {name}"], f"sigma {name}")


@pytest.mark.parametrize("name", REACTIONS)
def test_quantum_yield_matches_fortran(env, ref, name):
    e = env
    phi = api._eval_reaction_qy(
        e["reactions"][name]["quantum yield"], e["rel"], e["wl_edges"], e["wl_mid"],
        e["temperature"], e["air"].edge_val, e["n_lev"], e["wl_mid"].size,
    )
    _assert_matches_fortran(phi, ref[f"quantum yield {name}"], f"phi {name}")


def test_reference_temperature_matches(env, ref):
    # sanity: the dispatch inputs above are the Fortran's
    np.testing.assert_array_equal(env["temperature"].edge_val, ref["temperature"][:, 0])


def test_tropospheric_j_matches_fortran(calc, ref):
    sza = ref["solar zenith angle"]
    np.testing.assert_array_equal(sza, [0.0, 30.0, 60.0, 80.0])
    for ti, s in enumerate(sza):
        J = calc.rate_constants_profile(float(s), float(ref["Earth-Sun distance"][ti]))
        for name in REACTIONS:
            r = ref[name][TROPOSPHERE, ti]
            assert np.all(r > 0.0), name
            np.testing.assert_allclose(
                J[name][TROPOSPHERE], r, rtol=J_RTOL, atol=0.0, err_msg=f"{name} SZA {s}"
            )


def test_rate_constants_surface_api(calc):
    # rate_constants(..., altitude_km=0.0) returns the surface value of the profile
    J0 = calc.rate_constants(0.0, 0.0, 2002, 3, 21, 12.0, altitude_km=0.0)
    for name in REACTIONS:
        assert J0[name] > 0.0


def test_ch2o_h2_co_limits(env):
    # Stern-Volmer limits: at 2.45e19 molecule cm-3 and 300 K the tabulated H2+CO yield is
    # recovered; at zero pressure the 330-360 nm band rises to 1 - phi(H+HCO); the two channels
    # never sum above 1.
    from tuvx_photolysis.quantum_yield import ch2o_h2_co_quantum_yield, rebin_quantum_yield_params

    wl = env["wl_mid"]
    td = data.load_quantum_yield(REPO / "data/quantum_yields/CH2O_1.nc")
    p = rebin_quantum_yield_params(td, env["wl_edges"], lower_extrapolation="boundary")
    band = (wl >= 330.0) & (wl < 360.0) & (p[:, 1] > 0.0)
    assert band.sum() > 5
    at_1atm = ch2o_h2_co_quantum_yield(p, wl, np.array([300.0]), np.array([2.45e19]))[0]
    np.testing.assert_allclose(at_1atm, p[:, 1], rtol=1e-12)
    at_0 = ch2o_h2_co_quantum_yield(p, wl, np.array([300.0]), np.array([0.0]))[0]
    np.testing.assert_allclose(at_0[band], 1.0 - p[band, 0], rtol=1e-12)
    np.testing.assert_array_equal(at_0[~band], p[~band, 1])
    phi = ch2o_h2_co_quantum_yield(p, wl, env["temperature"].edge_val, env["air"].edge_val)
    assert np.all(phi + p[None, :, 0] <= 1.0 + 1e-12)
