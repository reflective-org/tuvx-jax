# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Data loaders (NumPy / netCDF4) for cross sections, quantum yields, profiles, grids, flux.

Reference: ``src/cross_section.F90``, ``src/quantum_yield.F90``, ``src/netcdf.F90``,
``src/profiles/*``, ``src/grid.F90``.

Everything is loaded into small NumPy-backed dataclasses. The cross-section / quantum-yield
container (:class:`TabulatedData`) can be built either from a TUV-x NetCDF file or directly from
arrays, so the user's own JPL data drops in behind the same interface without touching the rest of
the pipeline.

NetCDF layout (as written by TUV-x, see ``src/netcdf.F90``):
    dim   bins, parameters[, temperatures]
    var   wavelength(bins)              [nm]
    var   temperature(temperatures)     [K]            (optional)
    var   <prefix>parameters(parameters, bins)         e.g. cross_section_parameters [cm^2]
Fortran reads ``parameters`` column-major, ending up shaped ``(bins, parameters)``; we match that
convention here (``parameters`` has shape ``(n_bins, n_params)``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import netCDF4

__all__ = [
    "TabulatedData",
    "AtmosphereProfile",
    "SolarFlux",
    "load_cross_section",
    "load_quantum_yield",
    "load_profile_csv",
    "load_wavelength_grid",
    "load_solar_flux",
]


@dataclass
class TabulatedData:
    """Tabulated cross section or quantum yield on its native wavelength grid.

    Attributes
    ----------
    wavelength : (n_bins,) float array [nm]
    parameters : (n_bins, n_params) float array
        For a ``base`` dataset ``n_params == 1`` and ``parameters[:, 0]`` is the quantity itself
        (cross section in cm^2, quantum yield as a fraction). ``n_params > 1`` holds the
        coefficients of a temperature parameterization (e.g. O3, Burkholder).
    temperature : (n_temps,) float array or None [K]
        Temperature nodes, when the dataset is tabulated at discrete temperatures.
    units : str
    """

    wavelength: np.ndarray
    parameters: np.ndarray
    temperature: np.ndarray | None = None
    units: str = ""

    def __post_init__(self):
        self.wavelength = np.asarray(self.wavelength, dtype=float)
        params = np.asarray(self.parameters, dtype=float)
        if params.ndim == 1:
            params = params[:, np.newaxis]  # a bare (n_bins,) quantity -> (n_bins, 1)
        self.parameters = params
        if self.parameters.shape[0] != self.wavelength.shape[0]:
            raise ValueError(
                f"parameters first axis ({self.parameters.shape[0]}) must match wavelength "
                f"length ({self.wavelength.shape[0]}); expected shape (n_bins, n_params)"
            )
        if self.temperature is not None:
            self.temperature = np.asarray(self.temperature, dtype=float)

    @property
    def n_params(self) -> int:
        return self.parameters.shape[1]

    @property
    def values(self) -> np.ndarray:
        """The single-parameter quantity (cross section / quantum yield), shape ``(n_bins,)``.

        Only valid for ``base`` (single-parameter) datasets.
        """
        if self.n_params != 1:
            raise ValueError(
                f"values is only defined for single-parameter data (n_params={self.n_params}); "
                "use parameters and a temperature parameterization instead"
            )
        return self.parameters[:, 0]

    @classmethod
    def from_netcdf(cls, path: str | Path, prefix: str) -> "TabulatedData":
        """Load from a TUV-x NetCDF file. ``prefix`` is ``cross_section_`` or ``quantum_yield_``."""
        with netCDF4.Dataset(str(path)) as ds:
            var = ds.variables[f"{prefix}parameters"]
            # File order is (parameters, bins); transpose to (bins, parameters) to match Fortran.
            params = np.array(var[:]).T
            wavelength = np.array(ds.variables["wavelength"][:])
            temperature = (
                np.array(ds.variables["temperature"][:]) if "temperature" in ds.variables else None
            )
            units = getattr(var, "units", "")
        return cls(wavelength=wavelength, parameters=params, temperature=temperature, units=units)


def load_cross_section(path: str | Path) -> TabulatedData:
    """Load a cross-section NetCDF file (``cross_section_parameters``)."""
    return TabulatedData.from_netcdf(path, "cross_section_")


def load_quantum_yield(path: str | Path) -> TabulatedData:
    """Load a quantum-yield NetCDF file (``quantum_yield_parameters``)."""
    return TabulatedData.from_netcdf(path, "quantum_yield_")


@dataclass
class AtmosphereProfile:
    """A two-column ASCII atmospheric profile: altitude [km] vs value.

    Reference: ``data/profiles/atmosphere/ussa.{dens,temp,ozone}`` and ``src/profiles/*``.
    """

    altitude_km: np.ndarray
    values: np.ndarray
    units: str = ""

    def __post_init__(self):
        self.altitude_km = np.asarray(self.altitude_km, dtype=float)
        self.values = np.asarray(self.values, dtype=float)
        if self.altitude_km.shape != self.values.shape:
            raise ValueError("altitude and value columns must have the same length")


def load_profile_csv(path: str | Path, units: str = "") -> AtmosphereProfile:
    """Load a TUV-x atmosphere profile (``# comment`` lines, then ``altitude_km  value``)."""
    data = np.loadtxt(str(path), comments="#")
    return AtmosphereProfile(altitude_km=data[:, 0], values=data[:, 1], units=units)


def load_wavelength_grid(path: str | Path) -> np.ndarray:
    """Load a wavelength ``.grid`` file: a leading integer count then that many edge values [nm].

    Returns the array of grid edges (length ``count``; ``count - 1`` cells).
    """
    with open(path) as fh:
        tokens = fh.read().split()
    count = int(tokens[0])
    edges = np.array([float(t) for t in tokens[1:]], dtype=float)
    if edges.size != count:
        raise ValueError(
            f"{path}: header says {count} values but found {edges.size}"
        )
    return edges


@dataclass
class SolarFlux:
    """Extraterrestrial solar flux spectrum.

    Reference: ``data/profiles/solar/*.flx`` — ``wavelength [nm]  flux [photon cm-2 s-1 nm-1]``.
    """

    wavelength: np.ndarray
    flux: np.ndarray

    def __post_init__(self):
        self.wavelength = np.asarray(self.wavelength, dtype=float)
        self.flux = np.asarray(self.flux, dtype=float)


def load_solar_flux(path: str | Path) -> SolarFlux:
    """Load a solar flux file (``# comment`` lines, then ``wavelength  flux``)."""
    data = np.loadtxt(str(path), comments="#")
    return SolarFlux(wavelength=data[:, 0], flux=data[:, 1])
