# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Stage 3: Grid container and the linear / area-conserving interpolators."""

import numpy as np
import pytest

from tuvx_photolysis.grids import Grid, interp_linear, interp_conserving


def test_grid_convention():
    g = Grid(edges=np.array([0.0, 1.0, 3.0, 6.0]))
    assert g.n_cells == 3
    np.testing.assert_allclose(g.midpoints, [0.5, 2.0, 4.5])
    np.testing.assert_allclose(g.deltas, [1.0, 2.0, 3.0])


def test_grid_rejects_nonmonotonic():
    with pytest.raises(ValueError):
        Grid(edges=np.array([0.0, 2.0, 1.0]))


def test_interp_linear_recovers_line():
    xs = np.linspace(0.0, 10.0, 11)
    ys = 3.0 * xs + 2.0
    xt = np.array([0.0, 2.5, 7.3, 10.0])
    np.testing.assert_allclose(interp_linear(xt, xs, ys), 3.0 * xt + 2.0)


def test_interp_linear_zero_outside():
    xs = np.array([1.0, 2.0, 3.0])
    ys = np.array([10.0, 20.0, 30.0])
    out = interp_linear(np.array([-1.0, 0.5, 2.0, 5.0]), xs, ys)
    assert out[0] == 0.0 and out[1] == 0.0  # below range
    assert out[2] == pytest.approx(20.0)
    assert out[3] == 0.0  # above range


def test_conserving_constant():
    # Average of a constant over any bins is the constant.
    xs = np.linspace(0.0, 10.0, 101)
    ys = np.full_like(xs, 4.2)
    edges = np.array([0.0, 1.0, 5.0, 10.0])
    np.testing.assert_allclose(interp_conserving(edges, xs, ys), [4.2, 4.2, 4.2])


def test_conserving_linear_gives_midpoint_average():
    # For a linear source, the bin average equals the value at the bin midpoint.
    xs = np.linspace(0.0, 10.0, 2)  # exact line needs only endpoints
    ys = 2.0 * xs + 1.0
    edges = np.array([0.0, 2.0, 7.0, 10.0])
    mids = 0.5 * (edges[:-1] + edges[1:])
    np.testing.assert_allclose(interp_conserving(edges, xs, ys), 2.0 * mids + 1.0)


def test_conserving_is_area_conserving():
    # sum(avg_i * width_i) over all bins equals the total integral of the source curve.
    rng = np.random.default_rng(0)
    xs = np.sort(rng.uniform(0.0, 50.0, 200))
    xs[0], xs[-1] = 0.0, 50.0
    ys = np.abs(np.sin(xs / 5.0)) + 0.1
    edges = np.linspace(0.0, 50.0, 12)
    avg = interp_conserving(edges, xs, ys)
    rebinned_integral = np.sum(avg * np.diff(edges))
    total_integral = np.trapezoid(ys, xs)
    np.testing.assert_allclose(rebinned_integral, total_integral, rtol=1e-12)


def test_conserving_requires_overlap():
    xs = np.array([2.0, 3.0, 4.0])
    ys = np.array([1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        interp_conserving(np.array([0.0, 1.0, 5.0]), xs, ys)  # target extends beyond source
