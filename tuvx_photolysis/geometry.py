# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Solar and spherical geometry.

Reference: ``src/profiles/profile_utils.F90`` (solar zenith angle, Earth-Sun distance, Julian day),
``src/profiles/earth_sun_distance.F90``, ``src/spherical_geometry.F90`` (Dahlback & Stamnes 1991),
and ``src/constants.F90`` (Earth radius).

The solar-position routines are a direct port of TUV-x's Michalsky/Astronomical-Almanac chain, so
this package reproduces the Fortran solar zenith angle and Earth-Sun distance exactly (important for
1:1 validation). ``frank-model`` already has its own ``solar.py``; the public API accepts an
explicit ``solar_zenith_angle`` so the consumer may substitute that instead -- ``tuvx_photolysis``
does not depend on ``frank-model``.

All routines here are plain NumPy/``math`` (computed once per solar position, not in the JAX hot
path).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .data import AtmosphereProfile

__all__ = [
    "EARTH_RADIUS_KM",
    "julian_day_of_year",
    "solar_zenith_angle",
    "earth_sun_distance",
    "pressure_to_altitude_km",
    "SphericalGeometry",
]

EARTH_RADIUS_KM = 6.371e3  # src/constants.F90
_BOLTZMANN = 1.380649e-23  # J / K
_D2R = math.pi / 180.0

_DAYS_IN_MONTH = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def julian_day_of_year(year: int, month: int, day: int) -> int:
    """Day-of-year for ``year/month/day`` (1 = Jan 1). Ports ``julian_day_of_year``."""
    if not (1 <= month <= 12):
        raise ValueError("1 <= month <= 12")
    days = list(_DAYS_IN_MONTH)
    if _is_leap(year):
        days[1] = 29
    if day > days[month - 1]:
        raise ValueError("day exceeds days in month")
    return sum(days[: month - 1]) + day


def _calculate_time(year: int, day_of_year: int, hour: float) -> float:
    """Modified Julian time relative to J2000 noon. Ports ``calculate_time``."""
    delta = year - 1949
    leap = delta // 4
    jd = 32916.5 + (delta * 365 + leap + day_of_year) + hour / 24.0
    if year % 100 == 0 and year % 400 != 0:
        jd -= 1.0
    return jd - 51545.0


def _mean_anomaly(time: float) -> float:
    m = (357.528 + 0.9856003 * time) % 360.0
    if m < 0:
        m += 360.0
    return m * _D2R


def _mean_longitude(time: float) -> float:
    lon = (280.460 + 0.9856474 * time) % 360.0
    if lon < 0:
        lon += 360.0
    return lon  # degrees


def _obliquity(time: float) -> float:
    return (23.439 - 0.0000004 * time) * _D2R


def _ecliptic_longitude(mean_anomaly: float, mean_long_deg: float) -> float:
    eclong = mean_long_deg + 1.915 * math.sin(mean_anomaly) + 0.020 * math.sin(2.0 * mean_anomaly)
    eclong %= 360.0
    if eclong < 0:
        eclong += 360.0
    return eclong * _D2R


def _right_ascension(eclong: float, obliquity: float) -> float:
    num = math.cos(obliquity) * math.sin(eclong)
    den = math.cos(eclong)
    ra = math.atan(num / den)
    if den < 0:
        ra += math.pi
    elif num < 0:
        ra += 2.0 * math.pi
    return ra


def _declination(eclong: float, obliquity: float) -> float:
    return math.asin(math.sin(obliquity) * math.sin(eclong))


def _hour_angle(time: float, hour: float, longitude: float, right_ascension: float) -> float:
    gmst = (6.697375 + 0.0657098242 * time + hour) % 24.0
    if gmst < 0:
        gmst += 24.0
    lmst = (gmst + longitude / 15.0) % 24.0
    if lmst < 0:
        lmst += 24.0
    lmst = lmst * 15.0 * _D2R
    ha = lmst - right_ascension
    if ha < -math.pi:
        ha += 2.0 * math.pi
    elif ha > math.pi:
        ha -= 2.0 * math.pi
    return ha


def solar_zenith_angle(
    year: int, month: int, day: int, hour: float, latitude: float, longitude: float
) -> float:
    """Solar zenith angle [degrees]. Latitude N-positive, longitude E-positive, ``hour`` in UT.

    Direct port of TUV-x ``solar_zenith_angle`` (returns 90 - elevation).
    """
    doy = julian_day_of_year(year, month, day)
    time = _calculate_time(year, doy, hour)
    mean_anom = _mean_anomaly(time)
    mean_long = _mean_longitude(time)
    obl = _obliquity(time)
    eclong = _ecliptic_longitude(mean_anom, mean_long)
    ra = _right_ascension(eclong, obl)
    dec = _declination(eclong, obl)
    ha = _hour_angle(time, hour, longitude, ra)
    elevation = math.asin(
        math.sin(dec) * math.sin(latitude * _D2R)
        + math.cos(dec) * math.cos(latitude * _D2R) * math.cos(ha)
    ) / _D2R
    return 90.0 - elevation


def earth_sun_distance(year: int, month: int, day: int, hour: float) -> float:
    """Earth-Sun distance [AU]. Direct port of TUV-x ``earth_sun_distance``."""
    doy = julian_day_of_year(year, month, day)
    time = _calculate_time(year, doy, hour)
    m = _mean_anomaly(time)
    return 1.00014 - 0.01671 * math.cos(m) - 0.00014 * math.cos(2.0 * m)


def pressure_to_altitude_km(
    pressure_mbar: float, density: AtmosphereProfile, temperature: AtmosphereProfile
) -> float:
    """Altitude [km] of a given pressure, via the US Standard Atmosphere ``p = n k_B T``.

    ``density`` is number density [molecule cm-3] and ``temperature`` [K] on (overlapping) altitude
    grids (e.g. ``ussa.dens`` and ``ussa.temp``). Interpolation is linear in ``log(p)`` vs altitude.
    """
    z = density.altitude_km
    n_cm3 = density.values
    # temperature may carry the 1e10 exospheric sentinel; restrict to the density grid by interp.
    t = np.interp(z, temperature.altitude_km, temperature.values)
    p_pa = n_cm3 * 1.0e6 * _BOLTZMANN * t  # n[m-3] * kB * T
    p_mbar = p_pa / 100.0
    if not (p_mbar.min() <= pressure_mbar <= p_mbar.max()):
        raise ValueError(
            f"pressure {pressure_mbar} mbar outside profile range "
            f"[{p_mbar.min():.3g}, {p_mbar.max():.3g}]"
        )
    # pressure decreases with altitude -> interpolate altitude against decreasing log-p (flip).
    logp = np.log(p_mbar)
    order = np.argsort(logp)
    return float(np.interp(math.log(pressure_mbar), logp[order], z[order]))


@dataclass
class SphericalGeometry:
    """Dahlback & Stamnes (1991) slant-path geometry over a height grid.

    Call :meth:`set_parameters` with a solar zenith angle and the height-grid edges (km, surface
    first) to populate ``nid`` (layers crossed by the direct beam to each level) and ``dsdh``
    (slant path per vertical depth). Ports ``src/spherical_geometry.F90``.

    Indexing follows the Fortran: ``nid`` has length ``nlayer + 1`` (levels 0..nlayer measured from
    the top of the atmosphere downward); ``dsdh`` has shape ``(nlayer + 1, nlayer)`` where
    ``dsdh[i, j - 1]`` is the Fortran ``dsdh_(i, j)`` for ``j = 1..nlayer``.
    """

    solar_zenith_angle: float = 0.0
    nid: np.ndarray | None = None
    dsdh: np.ndarray | None = None

    def set_parameters(self, zen_deg: float, height_edges_km: np.ndarray) -> "SphericalGeometry":
        edges = np.asarray(height_edges_km, dtype=float)
        nz = edges.size
        nlayer = nz - 1
        zenrad = zen_deg * _D2R

        re = EARTH_RADIUS_KM + edges[0]
        ze = edges - edges[0]
        zd = ze[::-1].copy()  # inverse coordinate: zd[0] = top, zd[nlayer] = surface

        nid = np.zeros(nlayer + 1, dtype=int)
        dsdh = np.zeros((nlayer + 1, nlayer))
        sinrad = math.sin(zenrad)

        for i in range(nlayer + 1):
            rpsinz = (re + zd[i]) * sinrad
            if zen_deg > 90.0 and rpsinz < re:
                idx = -1
            else:
                idx = i
                if zen_deg > 90.0:
                    idx = -1
                    for j in range(1, nlayer + 1):
                        if (zd[j - 1] + re) > rpsinz >= (zd[j] + re):
                            idx = j
                for j in range(1, idx + 1):
                    sm = 1.0
                    if j == idx and idx == i and zen_deg > 90.0:
                        sm = -1.0
                    rj = re + zd[j - 1]
                    rjp1 = re + zd[j]
                    dhj = zd[j - 1] - zd[j]
                    ga = max(0.0, rj * rj - rpsinz * rpsinz)
                    gb = max(0.0, rjp1 * rjp1 - rpsinz * rpsinz)
                    if idx > i and j == idx:
                        dsj = math.sqrt(ga)
                    else:
                        dsj = math.sqrt(ga) - sm * math.sqrt(gb)
                    dsdh[i, j - 1] = dsj / dhj
            nid[i] = idx

        self.solar_zenith_angle = zen_deg
        self.nid = nid
        self.dsdh = dsdh
        return self

    def air_mass(self, aircol: np.ndarray, largest: float = 1.0e36):
        """Vertical and slant air columns above each level. Ports ``air_mass``.

        ``aircol`` is the per-layer air column [molec cm-2], length ``nz`` (= nlayer + 1).
        Returns ``(vcol, scol)``, each length ``nz``.
        """
        if self.nid is None or self.dsdh is None:
            raise RuntimeError("call set_parameters() before air_mass()")
        aircol = np.asarray(aircol, dtype=float)
        nz = aircol.size
        nlayer = nz - 1
        vcol = np.zeros(nz)
        scol = np.zeros(nz)

        # vertical column (work downward); Fortran 1-based id -> here id maps to index id.
        accum = aircol[nz - 1]
        for idx in range(nlayer, 0, -1):
            accum += aircol[idx - 1]
            vcol[idx - 1] = accum

        scol_top = self.dsdh[1, 0] * aircol[nz - 1]
        scol[nz - 1] = scol_top
        for idx in range(1, nlayer + 1):
            if self.nid[idx] < 0:
                accum = largest
            else:
                accum = scol_top
                single = min(self.nid[idx], idx)
                for j in range(1, single + 1):
                    accum += aircol[nz - 1 - j] * self.dsdh[idx, j - 1]
                for j in range(single + 1, self.nid[idx] + 1):
                    accum += 2.0 * aircol[nz - 1 - j] * self.dsdh[idx, j - 1]
            scol[nz - 1 - idx] = accum
        return vcol, scol
