# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Stage 7b: full radiation field validated against the no-aerosol Fortran reference.

Builds the air(Rayleigh)/O2/O3 radiators from the bundled profiles and cross sections, solves the
delta-Eddington radiation field, and compares the direct/diffuse-up/diffuse-down actinic-flux
components to ``tuv_5_4_no_aerosol_reference.nc``.

Agreement is essentially exact (machine precision) at all wavelengths >= ~206 nm. The Lyman-alpha /
Schumann-Runge band region (< ~206 nm) is excluded here because the Fortran applies its LA/SR O2
parameterization to the radiation field there, which is a deferred follow-up in this port.
"""

from pathlib import Path

import numpy as np
import pytest

from tuvx_photolysis import data, cross_section, profiles, radiators, solver, geometry

REPO = Path(__file__).resolve().parents[1]
D = REPO / "data"
FIX = Path(__file__).resolve().parent / "fixtures"
REF = FIX / "tuv_5_4_no_aerosol_reference.nc"

pytestmark = pytest.mark.skipif(
    not REF.exists(), reason="Fortran reference missing; run fixtures/regenerate_reference.sh"
)

# below this wavelength the Fortran radiation field uses the (deferred) LA/SR O2 parameterization
_LA_SR_MAX_NM = 205.8


def _build_total_optics():
    wl_edges = data.load_wavelength_grid(D / "grids" / "wavelength" / "combined.grid")
    wl_mid = 0.5 * (wl_edges[:-1] + wl_edges[1:])
    h_edges = np.linspace(0.0, 120.0, 121)

    dens = data.load_profile_csv(D / "profiles" / "atmosphere" / "ussa.dens")
    o3d = data.load_profile_csv(D / "profiles" / "atmosphere" / "ussa.ozone")
    temp = data.load_profile_csv(D / "profiles" / "atmosphere" / "ussa.temp")

    air = profiles.air_profile(h_edges, dens)
    o2 = profiles.o2_profile(h_edges, dens)
    o3 = profiles.o3_profile(h_edges, o3d)
    T = profiles.temperature_profile(h_edges, temp)

    rxs = radiators.rayleigh_cross_section(wl_mid)
    o2xs = cross_section.BaseCrossSection.from_files(
        [(data.load_cross_section(D / "cross_sections" / "O2_1.nc"), "boundary", None)], wl_edges
    ).evaluate(h_edges.size - 1)
    o3obj = cross_section.O3TintCrossSection.from_files(
        [data.load_cross_section(D / "cross_sections" / f"O3_{i}.nc") for i in range(1, 5)], wl_edges
    )
    o3xs = o3obj.evaluate(T.mid_val)

    total = radiators.accumulate(
        [
            radiators.air_radiator(air.layer_dens, rxs),
            radiators.absorber_radiator(o2.layer_dens, o2xs),
            radiators.absorber_radiator(o3.layer_dens, o3xs),
        ]
    )
    return total, wl_mid, h_edges


def test_radiation_field_matches_fortran_outside_la_sr():
    import netCDF4

    total, wl_mid, h_edges = _build_total_optics()
    uv_ok = wl_mid >= _LA_SR_MAX_NM

    with netCDF4.Dataset(REF) as ds:
        sza = np.array(ds.variables["solar zenith angle"][:])
        esd = np.array(ds.variables["Earth-Sun distance"][:])
        ref = {
            "dir": np.array(ds.variables["direct radiation"][:]),
            "up": np.array(ds.variables["upward radiation"][:]),
            "dn": np.array(ds.variables["downward radiation"][:]),
        }

    for ti in range(sza.size):
        sg = geometry.SphericalGeometry().set_parameters(sza[ti], h_edges)
        S, night, valid = solver.build_slant_operator(sg.nid, sg.dsdh)
        rf = solver.solve(total, sza[ti], 0.10, S, night, valid)
        # the Fortran scales the output field by the Earth-Sun distance (tuvx.F90 -> run)
        fields = {"dir": rf.fdr, "up": rf.fup, "dn": rf.fdn}
        for key, tol in [("dir", 1e-9), ("up", 1e-7), ("dn", 1e-5)]:
            mine = np.asarray(fields[key]) * esd[ti]
            r = ref[key][:, :, ti].T  # (n_levels, n_wl)
            mask = (r > r.max() * 1e-8) & uv_ok[None, :]
            rel = np.abs(mine[mask] - r[mask]) / r[mask]
            assert np.median(rel) < 1e-6, f"t{ti} {key} median {np.median(rel):.2e}"
            assert rel.max() < tol, f"t{ti} {key} max {rel.max():.2e}"


def test_la_sr_region_is_the_only_large_discrepancy():
    # Sanity: the discrepancies really are confined to the LA/SR band (< ~206 nm).
    import netCDF4

    total, wl_mid, h_edges = _build_total_optics()
    with netCDF4.Dataset(REF) as ds:
        sza = np.array(ds.variables["solar zenith angle"][:])
        esd = np.array(ds.variables["Earth-Sun distance"][:])
        ref_dir = np.array(ds.variables["direct radiation"][:])

    sg = geometry.SphericalGeometry().set_parameters(sza[0], h_edges)
    S, night, valid = solver.build_slant_operator(sg.nid, sg.dsdh)
    rf = solver.solve(total, sza[0], 0.10, S, night, valid)
    mine = np.asarray(rf.fdr) * esd[0]
    r = ref_dir[:, :, 0].T
    mask = r > r.max() * 1e-6
    rel = np.where(mask, np.abs(mine - r) / np.where(mask, r, 1.0), 0.0)
    # every wavelength with a >1% direct-beam discrepancy lies in the LA/SR band
    bad_wl = wl_mid[(rel > 1e-2).any(axis=0)]
    assert np.all(bad_wl < _LA_SR_MAX_NM)
