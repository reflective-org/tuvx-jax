# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Vertical profiles on the model height grid: edge/midpoint values and per-layer column densities.

Reference: ``src/profiles/air.F90``, ``o2.F90``, ``o3.F90``, ``from_csv_file.F90``.

Each profile is placed on the height-grid edges and yields:

* ``edge_val`` -- value at every grid interface,
* ``mid_val``  -- midpoint value, ``0.5 * (edge[i] + edge[i+1])``,
* ``layer_dens`` -- per-layer column density [molecule cm-2], including the exospheric top layer.

Faithful details from the Fortran:

* **air / O2** -- log-linear interpolation of number density; per-layer column uses the *geometric*
  mean of the bounding edge densities; an exospheric layer ``edge_top * scale_height`` is added to
  the top layer (scale height 8.01 km). O2 multiplies air by the 0.2095 volume mixing ratio.
* **O3** -- the data are first extended upward in 1 km steps decaying by ``exp(-1/H)`` (H = 4.5 km)
  to the model top, then *linearly* interpolated; per-layer column uses the *arithmetic* midpoint.
  The whole profile is then rescaled so the total column equals a reference value (default 300 DU).
* **temperature (csv)** -- linear interpolation to edges; midpoints are the edge average.

``km2cm = 1e5`` converts the layer thickness (km) to cm for the column density.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import AtmosphereProfile, SolarFlux
from .grids import interp_linear, interp_conserving, interp_fractional_target

__all__ = [
    "Profile",
    "air_profile",
    "o2_profile",
    "o3_profile",
    "temperature_profile",
    "extraterrestrial_flux",
]

_KM2CM = 1.0e5
_DU = 2.687e16  # molecule cm-2 per Dobson Unit
_O2_VMR = 0.2095
_HC = 6.626068e-34 * 2.99792458e8  # Planck constant x speed of light [J m]
_DELTAX = 1.0e-5
# wavelength ranges each solar-flux file covers (extraterrestrial_flux.F90 bin_edge)
_ETFL_BIN_EDGES = (0.0, 150.01, 200.07, 1000.99, np.inf)


@dataclass
class Profile:
    """A vertical profile on the height grid.

    ``layer_dens`` (n_layers) is the per-layer column density with the exospheric top layer folded
    into the last cell. ``exo_layer_dens`` (n_layers + 1) keeps the per-layer columns *unfolded*
    plus the exospheric layer as its final element — the form the slant-path ``air_mass`` expects.
    """

    edge_val: np.ndarray  # (n_levels,)
    mid_val: np.ndarray  # (n_layers,)
    layer_dens: np.ndarray | None = None  # (n_layers,) [molecule cm-2]
    exo_layer_dens: np.ndarray | None = None  # (n_layers + 1,) [molecule cm-2]


def _log_interp_density(height_edges_km, prof: AtmosphereProfile):
    """Log-linear interpolation of a number-density profile onto the grid edges (air/O2)."""
    zdata = prof.altitude_km.copy()
    zdata[-1] = zdata[-1] + 0.001  # matches the Fortran's tiny top bump
    return np.exp(interp_linear(height_edges_km, zdata, np.log(prof.values)))


def _geometric_layer_density(edge_val, delta_km, edge_top, scale_height):
    """Per-layer column density (geometric mean of edge densities) + exospheric top layer.

    Returns ``(layer_dens, exo_layer_dens)``: ``layer_dens`` has the exospheric layer folded into
    its last cell; ``exo_layer_dens`` is ``[per-layer columns, exospheric layer]`` (length n+1).
    """
    layer = delta_km * np.sqrt(edge_val[:-1]) * np.sqrt(edge_val[1:]) * _KM2CM
    exo = edge_top * scale_height * _KM2CM
    exo_layer = np.concatenate([layer, [exo]])
    folded = layer.copy()
    folded[-1] = folded[-1] + exo
    return folded, exo_layer


def air_profile(height_edges_km, density: AtmosphereProfile, scale_height: float = 8.01) -> Profile:
    """Air number-density profile (ports ``air.F90``)."""
    edges = np.asarray(height_edges_km, dtype=float)
    delta = np.diff(edges)
    edge_val = _log_interp_density(edges, density)
    mid_val = 0.5 * (edge_val[:-1] + edge_val[1:])
    layer_dens, exo = _geometric_layer_density(edge_val, delta, edge_val[-1], scale_height)
    return Profile(edge_val=edge_val, mid_val=mid_val, layer_dens=layer_dens, exo_layer_dens=exo)


def o2_profile(height_edges_km, density: AtmosphereProfile, scale_height: float = 8.01) -> Profile:
    """O2 number-density profile: air density times the 0.2095 VMR (ports ``o2.F90``)."""
    edges = np.asarray(height_edges_km, dtype=float)
    delta = np.diff(edges)
    edge_val = _O2_VMR * _log_interp_density(edges, density)
    mid_val = 0.5 * (edge_val[:-1] + edge_val[1:])
    layer_dens, exo = _geometric_layer_density(edge_val, delta, edge_val[-1], scale_height)
    return Profile(edge_val=edge_val, mid_val=mid_val, layer_dens=layer_dens, exo_layer_dens=exo)


def o3_profile(
    height_edges_km,
    ozone: AtmosphereProfile,
    scale_height: float = 4.5,
    reference_column_du: float = 300.0,
) -> Profile:
    """O3 number-density profile (ports ``o3.F90``).

    Extends the data to the model top with an ``exp(-1/H)`` per-km decay, linearly interpolates,
    forms arithmetic-mean layer columns, then rescales to ``reference_column_du`` (default 300 DU).
    """
    edges = np.asarray(height_edges_km, dtype=float)
    delta = np.diff(edges)
    ztop = edges[-1]

    zdata = list(ozone.altitude_km)
    prof = list(ozone.values)
    rfact = np.exp(-1.0 / scale_height) if scale_height != 0.0 else 0.0
    while zdata[-1] <= ztop:
        zdata.append(zdata[-1] + 1.0)
        prof.append(prof[-1] * rfact)
    zdata = np.array(zdata)
    prof = np.array(prof)

    def _layers(ev):
        base = delta * (0.5 * (ev[:-1] + ev[1:])) * _KM2CM
        exo = ev[-1] * scale_height * _KM2CM
        folded = base.copy()
        folded[-1] = folded[-1] + exo
        return folded, np.concatenate([base, [exo]])

    edge_val = interp_linear(edges, zdata, prof)
    layer_dens, exo_layer = _layers(edge_val)

    if reference_column_du != 1.0:
        input_du = np.sum(layer_dens) / _DU
        scale = reference_column_du / input_du
        if scale != 1.0:
            edge_val = scale * edge_val
            layer_dens, exo_layer = _layers(edge_val)

    mid_val = 0.5 * (edge_val[:-1] + edge_val[1:])
    return Profile(edge_val=edge_val, mid_val=mid_val, layer_dens=layer_dens, exo_layer_dens=exo_layer)


def _pad_flux(grid, datav):
    """Pad a flux dataset with zero endpoints so it spans the grid (extraterrestrial_flux.F90)."""
    x = np.asarray(grid, dtype=float).copy()
    y = np.asarray(datav, dtype=float).copy()
    pts = [
        ((1.0 - _DELTAX) * x[0], 0.0),
        (0.0, 0.0),
        ((1.0 + _DELTAX) * x[-1], 0.0),
        (1.0e38, 0.0),
    ]
    for xn, yn in pts:
        idx = int(np.searchsorted(x, xn))
        x = np.insert(x, idx, xn)
        y = np.insert(y, idx, yn)
    return x, y


def extraterrestrial_flux(wl_edges, files) -> np.ndarray:
    """Build the extraterrestrial flux per wavelength bin [photon cm-2 s-1] (ports the F90 profile).

    ``files`` is a list of ``(SolarFlux, interpolator)`` in the standard order
    [susim, atlas3, sao2010, neckel]; each file supplies the flux over its wavelength band
    (band edges ``[0, 150.01, 200.07, 1000.99, inf]``). Non-neckel files are zero-padded and
    interpolated (default ``"conserving"``); neckel is converted from photon units back to
    W m-2 nm-1 with a per-range grid shift, then interpolated (``"fractional target"``). The combined
    W m-2 nm-1 spectrum is converted to per-bin photon flux via ``1e-13 * etfl * lambda * dlambda /
    hc``.
    """
    wl_edges = np.asarray(wl_edges, dtype=float)
    wl_lower = wl_edges[:-1]
    wl_mid = 0.5 * (wl_edges[:-1] + wl_edges[1:])
    wl_delta = np.diff(wl_edges)
    etfl = np.zeros(wl_edges.size - 1)

    for i, (flux, interp) in enumerate(files):
        if interp in (None, ""):
            interp = "conserving"
        grid = np.asarray(flux.wavelength, dtype=float)
        datav = np.asarray(flux.flux, dtype=float)

        is_neckel = i == 3  # 4th file in the standard ordering
        if is_neckel:
            shifted = grid.copy()
            shifted[grid < 630.0] = grid[grid < 630.0] - 0.5
            mid = (grid >= 630.0) & (grid < 870.0)
            shifted[mid] = grid[mid] - 1.0
            shifted[grid >= 870.0] = grid[grid >= 870.0] - 2.5
            datav = 1.0e13 * _HC * datav / grid  # photons cm-2 s-1 nm-1 -> W m-2 nm-1
            grid = np.append(shifted, shifted[-1] + 2.5)
            datav = np.append(datav, 0.0)
        else:
            grid, datav = _pad_flux(grid, datav)

        if interp == "conserving":
            interpolated = interp_conserving(wl_edges, grid, datav)
        elif interp == "fractional target":
            interpolated = interp_fractional_target(wl_edges, grid, datav)
        else:
            raise ValueError(f"unsupported etfl interpolator: {interp}")

        in_band = (_ETFL_BIN_EDGES[i] <= wl_lower) & (wl_lower < _ETFL_BIN_EDGES[i + 1])
        etfl = np.where(in_band, interpolated, etfl)

    # W m-2 nm-1 -> per-bin photon flux [photon cm-2 s-1]
    return 1.0e-13 * etfl * wl_mid * wl_delta / _HC


def temperature_profile(height_edges_km, temperature: AtmosphereProfile) -> Profile:
    """Temperature profile: linear interpolation to edges, midpoints by averaging (ports csv type).

    The ``ussa.temp`` exospheric sentinel row (altitude 1e10) makes the source span the grid so the
    top edge does not extrapolate to zero.
    """
    edges = np.asarray(height_edges_km, dtype=float)
    edge_val = interp_linear(edges, temperature.altitude_km, temperature.values)
    mid_val = 0.5 * (edge_val[:-1] + edge_val[1:])
    return Profile(edge_val=edge_val, mid_val=mid_val)
