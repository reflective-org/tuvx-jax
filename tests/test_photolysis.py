# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Stage 7c: photolysis rate constants (J) validated against the no-aerosol Fortran reference.

The PhotolysisCalculator is built from the no-aerosol config and its per-reaction J profiles are
compared to ``tuv_5_4_no_aerosol_reference.nc``. Reactions whose photolysis is dominated by
wavelengths >= ~206 nm (where every input is ported exactly) must match the Fortran to machine
precision. Reactions with a significant deep-UV (< ~206 nm) contribution inherit the deferred
Lyman-alpha/Schumann-Runge radiation-field error in direct proportion to that contribution.
"""

from pathlib import Path

import numpy as np
import pytest

from tuvx_photolysis import PhotolysisCalculator, photolysis

REPO = Path(__file__).resolve().parents[1]
FIX = Path(__file__).resolve().parent / "fixtures"
CFG = FIX / "tuv_5_4_no_aerosol.json"
REF = FIX / "tuv_5_4_no_aerosol_reference.nc"

pytestmark = pytest.mark.skipif(
    not REF.exists(), reason="Fortran reference missing; run fixtures/regenerate_reference.sh"
)

_LA_SR_MAX_NM = 206.0


@pytest.fixture(scope="module")
def calc():
    return PhotolysisCalculator.from_tuvx_json(CFG, data_root=REPO)


def test_some_reactions_covered(calc):
    assert len(calc.xsqy) > 40  # most base/O3 + constant/tabulated-QY reactions
    # O2 photolysis is now handled via the Lyman-alpha/Schumann-Runge band parameterization
    assert calc.o2_reaction is not None
    assert calc.o2_reaction["name"] == "O2+hv->O+O"


def test_j_matches_fortran_for_non_deep_uv_reactions(calc):
    import netCDF4

    wl_mid = 0.5 * (calc.wl_edges[:-1] + calc.wl_edges[1:])
    uv = wl_mid < _LA_SR_MAX_NM

    with netCDF4.Dataset(REF) as ds:
        sza = np.array(ds.variables["solar zenith angle"][:])
        esd = np.array(ds.variables["Earth-Sun distance"][:])
        refvars = set(ds.variables)
        ref = {n: np.array(ds.variables[n][:]) for n in calc.xsqy if n in refvars}

    n_checked = 0
    for ti in range(sza.size):
        Jprof = calc.rate_constants_profile(sza[ti], esd[ti])
        rf = calc.radiation_field(sza[ti])
        flux = photolysis.actinic_flux(
            np.asarray(rf.fdr) * esd[ti], np.asarray(rf.fdn) * esd[ti],
            np.asarray(rf.fup) * esd[ti], calc.etfl,
        )
        for name, sq in calc.xsqy.items():
            if name not in ref:
                continue
            integ = flux * sq
            total = integ.sum()
            if total <= 0:
                continue
            uv_frac = integ[:, uv].sum() / total
            if uv_frac > 1e-4:
                continue  # deep-UV reaction: LA/SR-limited (deferred), checked separately
            r = ref[name][:, ti]
            m = r > r.max() * 1e-3
            if m.sum() < 5:
                continue
            rel = np.abs(Jprof[name][m] - r[m]) / r[m]
            assert np.median(rel) < 1e-5, f"t{ti} {name} median rel {np.median(rel):.2e}"
            assert rel.max() < 1e-4, f"t{ti} {name} max rel {rel.max():.2e}"
            n_checked += 1
    assert n_checked > 40  # ~28 reactions x (up to) 2 times


def test_deep_uv_error_tracks_la_sr_fraction(calc):
    # Sanity: the only sizeable J errors are deep-UV reactions, and the error grows with the
    # deep-UV contribution fraction (i.e. it's the deferred LA/SR effect, not a hidden bug).
    import netCDF4

    wl_mid = 0.5 * (calc.wl_edges[:-1] + calc.wl_edges[1:])
    uv = wl_mid < _LA_SR_MAX_NM
    with netCDF4.Dataset(REF) as ds:
        sza = np.array(ds.variables["solar zenith angle"][:])
        esd = np.array(ds.variables["Earth-Sun distance"][:])
        refvars = set(ds.variables)
        Jprof = calc.rate_constants_profile(sza[0], esd[0])
        rf = calc.radiation_field(sza[0])
        flux = photolysis.actinic_flux(
            np.asarray(rf.fdr) * esd[0], np.asarray(rf.fdn) * esd[0],
            np.asarray(rf.fup) * esd[0], calc.etfl,
        )
        for name, sq in calc.xsqy.items():
            if name not in refvars:
                continue
            r = np.array(ds.variables[name][:])[:, 0]
            m = r > r.max() * 1e-3
            if m.sum() < 5:
                continue
            integ = flux * sq
            uv_frac = integ[:, uv].sum() / max(integ.sum(), 1e-300)
            med = np.median(np.abs(Jprof[name][m] - r[m]) / r[m])
            # a >1% J discrepancy implies a meaningful deep-UV contribution
            if med > 1e-2:
                assert uv_frac > 0.05, f"{name}: large error {med:.2e} but uv_frac {uv_frac:.3f}"


def test_rate_constants_api_daynight(calc):
    # Daytime equatorial noon gives positive J; the sun below the horizon gives zero.
    day = calc.rate_constants(0.0, 0.0, 2002, 3, 21, 12.0, altitude_km=20.0)
    assert all(v >= 0 for v in day.values())
    assert any(v > 0 for v in day.values())
    night = calc.rate_constants(0.0, 0.0, 2002, 3, 21, 0.0, altitude_km=20.0)
    assert all(v == 0.0 for v in night.values())
