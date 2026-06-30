# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Stage 4: solar + spherical geometry, validated against the Fortran reference."""

import math
from pathlib import Path

import numpy as np
import pytest

from tuvx_photolysis import data, geometry

DATA = Path(__file__).resolve().parents[1] / "data"

# Reference values read from build/photolysis_rate_constants.nc (config: 2002-03-21, lat0, lon0,
# time grid [12, 14] h). These were produced by the Fortran tuv-x for tuv_5_4.json.
REF_SZA = {12.0: 1.8294233, 14.0: 28.19972549}
REF_ESD = {12.0: 0.99618238, 14.0: 0.99620580}


def test_julian_day():
    assert geometry.julian_day_of_year(2002, 3, 21) == 80
    assert geometry.julian_day_of_year(2002, 1, 1) == 1
    assert geometry.julian_day_of_year(2000, 3, 1) == 61  # leap year


@pytest.mark.parametrize("hour", [12.0, 14.0])
def test_solar_zenith_angle_matches_fortran(hour):
    sza = geometry.solar_zenith_angle(2002, 3, 21, hour, latitude=0.0, longitude=0.0)
    assert sza == pytest.approx(REF_SZA[hour], abs=1e-4)


@pytest.mark.parametrize("hour", [12.0, 14.0])
def test_earth_sun_distance_matches_fortran(hour):
    esd = geometry.earth_sun_distance(2002, 3, 21, hour)
    assert esd == pytest.approx(REF_ESD[hour], abs=1e-6)


def test_pressure_to_altitude_68mbar():
    dens = data.load_profile_csv(DATA / "profiles" / "atmosphere" / "ussa.dens")
    temp = data.load_profile_csv(DATA / "profiles" / "atmosphere" / "ussa.temp")
    z = geometry.pressure_to_altitude_km(68.0, dens, temp)
    assert 17.0 < z < 21.0  # ~19 km in the US Standard Atmosphere
    # surface pressure maps to ~0 km
    z0 = geometry.pressure_to_altitude_km(1013.0, dens, temp)
    assert z0 < 1.0


def test_spherical_geometry_overhead_sun():
    # Overhead sun: slant path per layer ~ 1 (dsdh ~ 1), nid increases with depth.
    edges = np.linspace(0.0, 120.0, 121)
    sg = geometry.SphericalGeometry().set_parameters(0.0, edges)
    assert sg.nid[0] == 0
    assert sg.nid[-1] == 120
    # for the sun at zenith, ds/dh along the vertical is ~1 for every crossed layer
    last = sg.dsdh[-1, :]
    np.testing.assert_allclose(last, 1.0, atol=1e-6)


def test_air_mass_slant_is_vertical_over_cos_for_small_sza():
    # For a modest SZA and overhead geometry, slant column ~ vertical column / cos(sza).
    edges = np.linspace(0.0, 120.0, 121)
    sza = 30.0
    sg = geometry.SphericalGeometry().set_parameters(sza, edges)
    aircol = np.full(edges.size, 1.0e20)  # arbitrary positive per-layer column
    vcol, scol = sg.air_mass(aircol)
    # compare in the middle of the column, away from boundaries
    ratio = scol[60] / vcol[60]
    assert ratio == pytest.approx(1.0 / math.cos(math.radians(sza)), rel=0.05)


def test_air_mass_night_is_capped():
    edges = np.linspace(0.0, 120.0, 121)
    sg = geometry.SphericalGeometry().set_parameters(120.0, edges)  # sun below horizon
    aircol = np.full(edges.size, 1.0e20)
    _, scol = sg.air_mass(aircol)
    assert np.any(scol >= 1.0e36)  # capped where the beam never reaches
