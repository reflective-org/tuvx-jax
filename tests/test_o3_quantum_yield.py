# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""O3 photolysis quantum yields (Matsumi et al. 2002), port of o3-o2_o1d.F90 / o3-o2_o3p.F90.

Validated against the analytic recommendation directly (the module *is* the parameterization):
the four wavelength regimes, the O(1D)+O(3P) partition, temperature dependence, and one
hand-evaluated point in the 305-328 nm analytic band.
"""

import numpy as np

from tuvx_photolysis.quantum_yield import o3_o1d_quantum_yield, o3_o3p_quantum_yield


def test_regimes_and_partition():
    wl = np.array([300.0, 315.0, 335.0, 350.0])  # <=305, analytic, (328,340], >340
    T = np.array([200.0, 298.0])
    o1d = o3_o1d_quantum_yield(wl, T)
    o3p = o3_o3p_quantum_yield(wl, T)
    assert o1d.shape == (2, 4)
    # fixed regimes (temperature-independent)
    np.testing.assert_allclose(o1d[:, 0], 0.90)   # lambda <= 305
    np.testing.assert_allclose(o1d[:, 2], 0.08)   # 328 < lambda <= 340
    np.testing.assert_allclose(o1d[:, 3], 0.0)    # lambda > 340
    # the two channels always partition unity
    np.testing.assert_allclose(o1d + o3p, 1.0)


def test_analytic_band_point():
    # hand-evaluate the Matsumi form at lambda=315 nm, T=298 K
    a = np.array([0.8036, 8.9061, 0.1192])
    x = np.array([304.225, 314.957, 310.737])
    om = np.array([5.576, 6.601, 2.187])
    lam, T = 315.0, 298.0
    kt = 0.695 * T
    q2 = np.exp(-825.518 / kt)
    qfac1, qfac2 = 1.0 / (1.0 + q2), q2 / (1.0 + q2)
    t300 = T / 300.0
    expect = (0.0765
              + a[0] * qfac1 * np.exp(-((x[0] - lam) / om[0]) ** 4)
              + a[1] * t300 * t300 * qfac2 * np.exp(-((x[1] - lam) / om[1]) ** 2)
              + a[2] * t300 ** 1.5 * np.exp(-((x[2] - lam) / om[2]) ** 2))
    got = o3_o1d_quantum_yield(np.array([lam]), np.array([T]))[0, 0]
    np.testing.assert_allclose(got, expect, rtol=1e-12)


def test_temperature_dependence():
    # in the analytic band the yield varies with temperature; outside it does not
    wl = np.array([315.0])
    warm = o3_o1d_quantum_yield(wl, np.array([298.0]))[0, 0]
    cold = o3_o1d_quantum_yield(wl, np.array([200.0]))[0, 0]
    assert warm != cold
