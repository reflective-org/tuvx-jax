# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Delta-Eddington two-stream radiative-transfer solver (JAX).

Reference: ``src/radiative_transfer/solvers/delta_eddington.F90`` (``update_radiation_field`` and
``slant_optical_depth``) and the tridiagonal solve in ``src/linear_algebras/linpack.F90``.

The algorithm (Toon et al. 1989; Joseph & Wiscombe 1976 delta-Eddington): delta-scale each layer,
form the Eddington two-stream coefficients, build a ``2N x 2N`` tridiagonal system for the diffuse
field, solve it, and reconstruct the direct / diffuse-down / diffuse-up actinic-flux components at
each level. Layers are processed top-down (the Fortran reverses the optical-property arrays); the
results are flipped back to bottom-up at the end.

Implementation notes:
* The solve is written for a single wavelength and ``vmap``-ed over wavelength.
* The slant-path optical depth ``tausla`` is ``S @ taun`` where ``S`` is a constant lower-triangular
  operator built once per solar position from ``nid``/``dsdh`` (it does not depend on wavelength).
  This is the daytime (``sza < 90``) regime that matters for photolysis; night levels (``nid < 0``)
  are flagged and given an effectively infinite slant path (direct beam -> 0).
* The tridiagonal system is solved with a ``lax.scan`` Thomas algorithm (jit-friendly).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from .radiators import RadiatorOpticalProps

__all__ = ["RadiationField", "build_slant_operator", "solve", "solve_arrays"]

_LARGEST = 1.0e36
_PRECIS = 1.0e-7
_EPS = 1.0e-3
_D2R = np.pi / 180.0


class RadiationField(NamedTuple):
    """Radiation field components, each shaped ``(n_levels, n_wavelengths)``, bottom-up in altitude.

    ``fdr``/``fdn``/``fup`` are the direct, diffuse-down, and diffuse-up actinic-flux components
    (normalized to incident pi*F = 1); ``edr``/``edn``/``eup`` are the matching irradiances.
    The total actinic flux is ``fdr + fdn + fup`` (times the extraterrestrial flux, applied later).
    """

    fdr: jnp.ndarray
    fdn: jnp.ndarray
    fup: jnp.ndarray
    edr: jnp.ndarray
    edn: jnp.ndarray
    eup: jnp.ndarray


def build_slant_operator(nid: np.ndarray, dsdh: np.ndarray):
    """Build the constant slant-path operator ``S`` and the night masks from the geometry.

    ``tausla[level] = sum_j weight[level, j] * dsdh[level, j] * taun[j]`` where ``weight`` is 1 for
    single-pass layers (``j <= min(nid, level)``) and 2 for double-pass layers, matching
    ``slant_optical_depth``. Returns ``(S, night_level, layer_nid_valid)``:

    * ``S`` -- ``(n_levels, n_layers)`` operator (top-down layer index).
    * ``night_level`` -- ``(n_levels,)`` bool, levels the direct beam never reaches (``nid < 0``).
    * ``layer_nid_valid`` -- ``(n_layers,)`` bool, ``nid[level] >= 0`` for levels ``1..n_layers``
      (used to gate the ``mu2`` slant/vertical ratio).
    """
    nid = np.asarray(nid)
    dsdh = np.asarray(dsdh, dtype=float)
    n_levels = nid.size
    n_layers = n_levels - 1
    S = np.zeros((n_levels, n_layers))
    night_level = np.zeros(n_levels, dtype=bool)
    for i in range(n_levels):
        ncross = nid[i]
        if ncross < 0:
            night_level[i] = True
            continue
        single = min(ncross, i)
        for j in range(1, single + 1):
            S[i, j - 1] = dsdh[i, j - 1]
        for j in range(single + 1, ncross + 1):
            S[i, j - 1] = 2.0 * dsdh[i, j - 1]
    layer_nid_valid = nid[1:] >= 0
    return S, night_level, layer_nid_valid


def _thomas_solve(a, b, c, d):
    """Solve a tridiagonal system (a=sub, b=diag, c=super, d=rhs) via the Thomas algorithm.

    Mirrors ``tridiag`` in ``src/linear_algebras/linpack.F90``. All inputs length ``n``; ``a[0]`` and
    ``c[-1]`` are unused. Implemented with ``lax.scan`` so it is jit/vmap-friendly.
    """
    n = b.shape[0]

    cp0 = c[0] / b[0]
    dp0 = d[0] / b[0]

    def fwd(carry, inp):
        cp_prev, dp_prev = carry
        ai, bi, ci, di = inp
        m = bi - ai * cp_prev
        cp = ci / m
        dp = (di - ai * dp_prev) / m
        return (cp, dp), (cp, dp)

    _, (cp_rest, dp_rest) = jax.lax.scan(
        fwd, (cp0, dp0), (a[1:], b[1:], c[1:], d[1:])
    )
    cp = jnp.concatenate([cp0[None], cp_rest])
    dp = jnp.concatenate([dp0[None], dp_rest])

    def back(x_next, dp_cp):
        dp_i, cp_i = dp_cp
        x_i = dp_i - cp_i * x_next
        return x_i, x_i

    _, x_rev = jax.lax.scan(back, dp[-1], (dp[:-1], cp[:-1]), reverse=True)
    return jnp.concatenate([x_rev, dp[-1][None]])


def _solve_one_wavelength(tauu, omu, gu, rsfc, S, mu, night_level, layer_nid_valid):
    """Delta-Eddington solve for one wavelength. Arrays are top-down, length ``n_layers``.

    Returns the six radiation-field components for this wavelength (length ``n_levels``, bottom-up).
    """
    n_layers = tauu.shape[0]
    pifs = 1.0
    fdn0 = 0.0
    surfem = 0.0

    # --- delta scaling (Joseph & Wiscombe) ---
    f = gu * gu
    gi = (gu - f) / (1.0 - f)
    omi = (1.0 - f) * omu / (1.0 - omu * f)
    taun = (1.0 - omu * f) * tauu

    # --- slant + vertical optical depth ---
    tausla = jnp.where(night_level, _LARGEST, S @ taun)  # (n_levels,)
    tauc = jnp.concatenate([jnp.zeros(1), jnp.cumsum(taun)])  # (n_levels,)

    # keep g, omega away from the boundaries by precision
    g = jnp.sign(gi) * jnp.minimum(jnp.abs(gi), 1.0 - _PRECIS)
    om = jnp.minimum(omi, 1.0 - _PRECIS)

    # mu2 = d(vertical tau) / d(slant tau) per layer (gated where nid >= 0)
    dtauc = taun  # tauc[i] - tauc[i-1]
    dsla = tausla[1:] - tausla[:-1]
    inv_sqrt_largest = 1.0 / np.sqrt(_LARGEST)
    ratio = dtauc / jnp.where(dsla == 0.0, 1.0, dsla)
    mu2_calc = jnp.where(
        dsla == 0.0,
        np.sqrt(_LARGEST),
        jnp.sign(ratio) * jnp.maximum(jnp.abs(ratio), inv_sqrt_largest),
    )
    mu2 = jnp.where(layer_nid_valid, mu2_calc, inv_sqrt_largest)

    # --- Eddington two-stream coefficients (Toon 1989, Table 1) ---
    gam1 = (7.0 - om * (4.0 + 3.0 * g)) / 4.0
    gam2 = -(1.0 - om * (4.0 - 3.0 * g)) / 4.0
    gam3 = (2.0 - 3.0 * g * mu) / 4.0
    gam4 = 1.0 - gam3
    mu1 = 0.5

    lam = jnp.sqrt(gam1 * gam1 - gam2 * gam2)
    bgam = jnp.where(gam2 != 0.0, (gam1 - lam) / jnp.where(gam2 == 0.0, 1.0, gam2), 0.0)
    expon = jnp.exp(-lam * taun)

    e1 = 1.0 + bgam * expon
    e2 = 1.0 - bgam * expon
    e3 = bgam + expon
    e4 = bgam - expon

    expon0 = jnp.exp(-tausla[:-1])  # tausla(i-1), i=1..n_layers
    expon1 = jnp.exp(-tausla[1:])  # tausla(i)

    divisr = lam * lam - 1.0 / (mu2 * mu2)
    divisr = jnp.sign(divisr) * jnp.maximum(_EPS, jnp.abs(divisr))
    up = om * pifs * ((gam1 - 1.0 / mu2) * gam3 + gam4 * gam2) / divisr
    dn = om * pifs * ((gam1 + 1.0 / mu2) * gam4 + gam2 * gam3) / divisr

    cup = up * expon0
    cdn = dn * expon0
    cuptn = up * expon1
    cdntn = dn * expon1

    # --- assemble the 2N tridiagonal system ---
    mrows = 2 * n_layers
    a_ = jnp.zeros(mrows)
    b_ = jnp.zeros(mrows)
    d_ = jnp.zeros(mrows)
    e_ = jnp.zeros(mrows)

    # first row
    a_ = a_.at[0].set(0.0)
    b_ = b_.at[0].set(e1[0])
    d_ = d_.at[0].set(-e2[0])
    e_ = e_.at[0].set(fdn0 - cdn[0])

    k = jnp.arange(n_layers - 1)  # layer pairs (0-based) i, i+1
    # odd Fortran rows 3..2N-1  -> 0-based rows 2,4,...,2N-2
    odd = 2 + 2 * k
    a_ = a_.at[odd].set(e2[:-1] * e3[:-1] - e4[:-1] * e1[:-1])
    b_ = b_.at[odd].set(e1[:-1] * e1[1:] - e3[:-1] * e3[1:])
    d_ = d_.at[odd].set(e3[:-1] * e4[1:] - e1[:-1] * e2[1:])
    e_ = e_.at[odd].set(e3[:-1] * (cup[1:] - cuptn[:-1]) + e1[:-1] * (cdntn[:-1] - cdn[1:]))
    # even Fortran rows 2..2N-2 -> 0-based rows 1,3,...,2N-3
    even = 1 + 2 * k
    a_ = a_.at[even].set(e2[1:] * e1[:-1] - e3[:-1] * e4[1:])
    b_ = b_.at[even].set(e2[:-1] * e2[1:] - e4[:-1] * e4[1:])
    d_ = d_.at[even].set(e1[1:] * e4[1:] - e2[1:] * e3[1:])
    e_ = e_.at[even].set((cup[1:] - cuptn[:-1]) * e2[1:] - (cdn[1:] - cdntn[:-1]) * e4[1:])

    # last row
    ssfc = rsfc * mu * jnp.exp(-tausla[n_layers]) * pifs + surfem
    a_ = a_.at[mrows - 1].set(e1[-1] - rsfc * e3[-1])
    b_ = b_.at[mrows - 1].set(e2[-1] - rsfc * e4[-1])
    d_ = d_.at[mrows - 1].set(0.0)
    e_ = e_.at[mrows - 1].set(ssfc - cuptn[-1] + rsfc * cdntn[-1])

    y = _thomas_solve(a_, b_, d_, e_)

    # --- reconstruct fluxes (top-down) ---
    fdr = pifs * jnp.exp(-tausla)  # level k uses tausla[k], k = 0..n_layers
    edr = mu * fdr

    edn0 = fdn0
    eup0 = y[0] * e3[0] - y[1] * e4[0] + cup[0]

    yr = y[0::2]  # y(row),   row = 1,3,... (0-based 0,2,...)  -> per layer j
    yr1 = y[1::2]  # y(row+1)
    edn_rest = yr * e3 + yr1 * e4 + cdntn
    eup_rest = yr * e1 + yr1 * e2 + cuptn

    edn = jnp.concatenate([jnp.array([edn0]), edn_rest])
    eup = jnp.concatenate([jnp.array([eup0]), eup_rest])
    fdn = edn / mu1
    fup = eup / mu1

    # transform from top-down to bottom-up
    return (fdr[::-1], fdn[::-1], fup[::-1], edr[::-1], eup[::-1], edn[::-1])


# Built ONCE at module level and jit-compiled: a per-call `jax.vmap(...)` closure defeats the
# compile cache (the eager `lax.scan`s inside recompile on every call -- ~90 ms/solve vs <1 ms
# compiled). `mu` is a traced scalar, so different solar positions reuse the same compilation.
_SOLVE_VMAP = jax.jit(
    jax.vmap(_solve_one_wavelength, in_axes=(0, 0, 0, 0, None, None, None, None))
)


def solve_arrays(od, ssa, g, mu, rsfc, S, night_level, layer_nid_valid) -> RadiationField:
    """Pure-array, jit/grad-friendly core. All arrays are JAX arrays.

    ``od``/``ssa``/``g`` are ``(n_layers, n_wavelengths)`` bottom-up; ``mu = cos(sza)`` scalar;
    ``rsfc`` is ``(n_wavelengths,)``; ``S``/``night_level``/``layer_nid_valid`` come from
    :func:`build_slant_operator`. Returns a :class:`RadiationField` (``(n_levels, n_wavelengths)``).
    """
    # Fortran processes layers top-down: reverse the layer axis, move wavelength to axis 0.
    tauu = od[::-1, :].T  # (n_wl, n_layers)
    omu = ssa[::-1, :].T
    gu = g[::-1, :].T

    fdr, fdn, fup, edr, eup, edn = _SOLVE_VMAP(
        tauu, omu, gu, rsfc, S, mu, night_level, layer_nid_valid
    )
    # vmap stacks along axis 0 (wavelength); transpose to (n_levels, n_wavelengths)
    return RadiationField(
        fdr=fdr.T, fdn=fdn.T, fup=fup.T, edr=edr.T, edn=edn.T, eup=eup.T
    )


def solve(
    optical_props: RadiatorOpticalProps,
    solar_zenith_angle_deg: float,
    surface_albedo,
    slant_operator,
    night_level=None,
    layer_nid_valid=None,
) -> RadiationField:
    """Solve the radiation field for all wavelengths (host-side convenience wrapper).

    Parameters
    ----------
    optical_props : RadiatorOpticalProps
        Accumulated total ``(OD, SSA, G)``, each ``(n_layers, n_wavelengths)`` (bottom-up layers).
    solar_zenith_angle_deg : float
    surface_albedo : scalar or (n_wavelengths,) array
    slant_operator, night_level, layer_nid_valid :
        Output of :func:`build_slant_operator` for this solar position.

    For jit/vmap/grad over the optical state, call :func:`solve_arrays` directly with JAX arrays.
    """
    od = jnp.asarray(optical_props.optical_depth)
    ssa = jnp.asarray(optical_props.single_scattering_albedo)
    g = jnp.asarray(optical_props.asymmetry_factor)
    n_layers, n_wl = od.shape

    if night_level is None:
        night_level = np.zeros(n_layers + 1, dtype=bool)
    if layer_nid_valid is None:
        layer_nid_valid = np.ones(n_layers, dtype=bool)

    rsfc = jnp.broadcast_to(jnp.asarray(surface_albedo, dtype=float), (n_wl,))
    mu = float(np.cos(solar_zenith_angle_deg * _D2R))
    return solve_arrays(
        od, ssa, g, mu, rsfc,
        jnp.asarray(slant_operator), jnp.asarray(night_level), jnp.asarray(layer_nid_valid),
    )
