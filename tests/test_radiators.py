# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Stage 5: radiator optical properties, Rayleigh cross section, and accumulation."""

import numpy as np
import pytest

from tuvx_photolysis import radiators
from tuvx_photolysis.grids import interp_fractional_source


def test_rayleigh_cross_section_formula():
    # spot-check the WMO/Nicolet formula at 300 nm against a direct evaluation
    w = 0.3  # micron
    p = 3.6772 + 0.389 * w + 0.09426 / w
    expected = 4.02e-28 / w**p
    got = radiators.rayleigh_cross_section(np.array([300.0]))[0]
    assert got == pytest.approx(expected, rel=1e-12)
    # Rayleigh cross section falls steeply with wavelength (~lambda^-4)
    xs = radiators.rayleigh_cross_section(np.array([300.0, 600.0]))
    assert xs[0] / xs[1] > 10.0


def test_absorber_radiator():
    dens = np.array([1e18, 5e17])
    xs = np.array([[1e-20, 2e-20, 3e-20], [1e-20, 2e-20, 3e-20]])
    r = radiators.absorber_radiator(dens, xs)
    np.testing.assert_allclose(r.optical_depth, dens[:, None] * xs)
    assert np.all(r.single_scattering_albedo == 0.0)
    assert np.all(r.asymmetry_factor == 0.0)
    assert r.is_air is False


def test_air_radiator():
    dens = np.array([1e24, 5e23])
    rxs = radiators.rayleigh_cross_section(np.array([300.0, 400.0, 500.0]))
    r = radiators.air_radiator(dens, rxs)
    assert r.is_air is True
    assert np.all(r.single_scattering_albedo == 1.0)
    np.testing.assert_allclose(r.optical_depth, dens[:, None] * rxs[None, :])


def test_accumulate_absorber_plus_air():
    # one absorber (SSA 0) + air (SSA 1, g 0): check totals match the hand formulas
    od_abs = np.array([[2.0]])
    od_air = np.array([[1.0]])
    absorber = radiators.RadiatorOpticalProps(od_abs, 0.0, 0.0, is_air=False)
    air = radiators.RadiatorOpticalProps(od_air, 1.0, 0.0, is_air=True)
    tot = radiators.accumulate([absorber, air])
    # scattering = 1.0 (air), absorption = 2.0 (absorber); total OD = 3.0
    np.testing.assert_allclose(tot.optical_depth, 3.0)
    np.testing.assert_allclose(tot.single_scattering_albedo, 1.0 / 3.0)
    np.testing.assert_allclose(tot.asymmetry_factor, 0.0)  # air g = 0


def test_accumulate_asymmetry_is_scatter_weighted():
    # two scatterers with different g: accumulated g is scattering-OD weighted
    a = radiators.RadiatorOpticalProps(np.array([[1.0]]), 1.0, 0.6, is_air=False)
    b = radiators.RadiatorOpticalProps(np.array([[3.0]]), 1.0, 0.2, is_air=False)
    tot = radiators.accumulate([a, b])
    expected_g = (0.6 * 1.0 + 0.2 * 3.0) / (1.0 + 3.0)
    np.testing.assert_allclose(tot.asymmetry_factor, expected_g)
    np.testing.assert_allclose(tot.single_scattering_albedo, 1.0)


def test_accumulate_pure_absorber_floor():
    # no scattering anywhere -> SSA floored at kfloor, not div-by-zero
    absorber = radiators.RadiatorOpticalProps(np.array([[5.0]]), 0.0, 0.0, is_air=False)
    tot = radiators.accumulate([absorber])
    assert tot.single_scattering_albedo[0, 0] == pytest.approx(1.0 / 1.0e36)
    assert tot.optical_depth[0, 0] == pytest.approx(5.0, rel=1e-6)


def test_fractional_source_aligned_grids():
    # aligned unit grids: each target bin gets exactly its source bin's value
    src_edges = np.array([0.0, 1.0, 2.0, 3.0])
    yk = np.array([10.0, 20.0, 30.0])
    tgt_edges = np.array([0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(interp_fractional_source(tgt_edges, src_edges, yk), yk)


def test_fractional_source_partial_coverage_is_zero_above():
    # source only spans 0-2, target spans 0-4: bins above 2 are zero (no fold-in)
    src_edges = np.array([0.0, 1.0, 2.0])
    yk = np.array([10.0, 20.0])
    tgt_edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    out = interp_fractional_source(tgt_edges, src_edges, yk)
    np.testing.assert_allclose(out, [10.0, 20.0, 0.0, 0.0])
