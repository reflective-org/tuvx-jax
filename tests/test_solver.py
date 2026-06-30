# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Stage 6: delta-Eddington JAX solver -- Thomas solve, analytic limits, jit."""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from tuvx_photolysis import solver, radiators, geometry


def test_thomas_solve_matches_dense():
    rng = np.random.default_rng(1)
    n = 12
    a = rng.uniform(-1, 1, n)  # sub-diagonal (a[0] unused)
    b = rng.uniform(5, 8, n)  # diagonally dominant main diagonal
    c = rng.uniform(-1, 1, n)  # super-diagonal (c[-1] unused)
    d = rng.uniform(-1, 1, n)
    M = np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)
    expected = np.linalg.solve(M, d)
    got = np.asarray(solver._thomas_solve(jnp.array(a), jnp.array(b), jnp.array(c), jnp.array(d)))
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)


def _build_geometry(n_layers, sza):
    edges = np.linspace(0.0, 120.0, n_layers + 1)
    sg = geometry.SphericalGeometry().set_parameters(sza, edges)
    return solver.build_slant_operator(sg.nid, sg.dsdh)


def test_pure_absorber_is_beer_lambert():
    # No scattering -> diffuse field vanishes and the direct beam is exp(-slant tau).
    n_layers = 20
    sza = 0.0
    S, night, valid = _build_geometry(n_layers, sza)
    n_wl = 3
    od = np.tile(np.linspace(0.05, 0.2, n_layers)[:, None], (1, n_wl))
    props = radiators.RadiatorOpticalProps(od, 0.0, 0.0)  # SSA=0 absorber
    rf = solver.solve(props, sza, surface_albedo=0.0, slant_operator=S,
                      night_level=night, layer_nid_valid=valid)
    # tausla = S @ taun (taun == od here since SSA=0 -> no delta scaling change), top-down
    taun_td = od[::-1, 0]
    tausla = S @ taun_td
    fdr_expected_td = np.exp(-tausla)
    fdr_expected_bottomup = fdr_expected_td[::-1]
    np.testing.assert_allclose(np.asarray(rf.fdr)[:, 0], fdr_expected_bottomup, rtol=1e-6)
    # diffuse components are zero for a pure absorber with a black surface
    np.testing.assert_allclose(np.asarray(rf.fdn), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(rf.fup), 0.0, atol=1e-12)


def test_direct_beam_attenuates_more_at_larger_sza():
    n_layers = 30
    od = np.full((n_layers, 1), 0.1)
    props = radiators.RadiatorOpticalProps(od, 0.0, 0.0)
    S0, n0, v0 = _build_geometry(n_layers, 0.0)
    S60, n60, v60 = _build_geometry(n_layers, 60.0)
    rf0 = solver.solve(props, 0.0, 0.0, S0, n0, v0)
    rf60 = solver.solve(props, 60.0, 0.0, S60, n60, v60)
    # surface direct beam (bottom level index 0) is smaller at the larger zenith angle
    assert float(rf60.fdr[0, 0]) < float(rf0.fdr[0, 0])
    # ratio ~ exp(-tau_total*(sec60 - 1)); sec(60)=2, so noticeably smaller
    assert float(rf60.fdr[0, 0]) < float(rf0.fdr[0, 0]) * 0.9


def test_conservative_scattering_reflects_more_with_bright_surface():
    n_layers = 20
    od = np.full((n_layers, 1), 0.3)
    props = radiators.RadiatorOpticalProps(od, 1.0, 0.0)  # conservative scattering, isotropic
    S, night, valid = _build_geometry(n_layers, 30.0)
    rf_dark = solver.solve(props, 30.0, 0.0, S, night, valid)
    rf_bright = solver.solve(props, 30.0, 1.0, S, night, valid)
    # upwelling diffuse flux at the top of atmosphere is larger over a bright surface
    assert float(rf_bright.fup[-1, 0]) > float(rf_dark.fup[-1, 0])
    # all actinic-flux components finite and non-negative-ish
    for arr in (rf_bright.fdr, rf_bright.fdn, rf_bright.fup):
        assert np.all(np.isfinite(np.asarray(arr)))


def test_solve_arrays_is_jittable_and_differentiable():
    n_layers, n_wl = 16, 4
    od = jnp.asarray(np.tile(np.linspace(0.05, 0.2, n_layers)[:, None], (1, n_wl)))
    ssa = jnp.full((n_layers, n_wl), 0.5)
    g = jnp.full((n_layers, n_wl), 0.3)
    S, night, valid = _build_geometry(n_layers, 20.0)
    S, night, valid = jnp.asarray(S), jnp.asarray(night), jnp.asarray(valid)
    mu = float(np.cos(np.radians(20.0)))
    rsfc = jnp.full(n_wl, 0.1)

    def total_surface_flux(od_):
        rf = solver.solve_arrays(od_, ssa, g, mu, rsfc, S, night, valid)
        return jnp.sum(rf.fdr + rf.fdn + rf.fup)

    ref = total_surface_flux(od)
    jit_val = jax.jit(total_surface_flux)(od)
    np.testing.assert_allclose(float(jit_val), float(ref), rtol=1e-9)
    # differentiable end-to-end (JAX-native goal): gradient is finite
    grad = jax.grad(total_surface_flux)(od)
    assert np.all(np.isfinite(np.asarray(grad)))
