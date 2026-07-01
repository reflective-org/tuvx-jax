# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Top-level orchestrator: PhotolysisCalculator.

Reference: ``src/core.F90`` (``run``) and ``src/tuvx.F90``. Ties the pipeline together:
geometry -> radiator optical properties -> delta-Eddington radiation field -> actinic flux ->
per-reaction wavelength integration -> J profile (and J interpolated to an altitude).

:meth:`PhotolysisCalculator.from_tuvx_json` parses a TUV-x JSON config (the air/O2/O3 radiators,
base/O3 cross sections, and constant/tabulated quantum yields). Reactions whose recipes are not yet
ported (special cross-section/quantum-yield modules, or O2 ``apply O2 bands`` / LA-SR) are recorded
in ``skipped_reactions`` rather than silently dropped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import data, cross_section, profiles, radiators, solver, geometry, photolysis, special
from .la_sr_bands import LaSrBands
from .quantum_yield import (
    ConstantQuantumYield,
    TabulatedQuantumYield,
    TintQuantumYield,
    clono2_quantum_yield,
)

__all__ = ["PhotolysisCalculator"]


@dataclass
class PhotolysisCalculator:
    """Computes photolysis rate constants on the model column for a given solar position."""

    wl_edges: np.ndarray
    height_edges_km: np.ndarray
    radiator_props: list  # per-radiator RadiatorOpticalProps (air, O2, O3); accumulated per solve
    etfl: np.ndarray  # (n_wl,) per-bin extraterrestrial flux [photon cm-2 s-1]
    surface_albedo: float
    xsqy: dict  # reaction name -> sigma*phi (n_levels, n_wl), sza-independent
    temperature_edge: np.ndarray  # (n_levels,) [K]
    skipped_reactions: dict = field(default_factory=dict)
    # Lyman-alpha / Schumann-Runge band handling (SZA-dependent, applied per solve)
    la_sr: object = None  # LaSrBands or None
    o2_index: int | None = None  # index of the O2 radiator in radiator_props
    air_exo: np.ndarray | None = None  # air exospheric layer densities (for slant columns)
    o2_exo: np.ndarray | None = None  # O2 exospheric layer densities
    o2_reaction: dict | None = None  # O2 photolysis reaction: {name, sigma_base (n_lev,n_wl), phi}

    def reaction_names(self):
        names = list(self.xsqy)
        if self.o2_reaction is not None:
            names.append(self.o2_reaction["name"])
        return names

    # ---- solving -------------------------------------------------------------------------------
    def _solve(self, solar_zenith_angle_deg: float):
        """Return (radiation_field, columns). ``columns`` is (air_vcol, air_scol, o2_scol) or None.

        When the grid includes the LA/SR bands, the O2 radiator optical depth in those bins is
        replaced by the column-dependent effective values before accumulating and solving.
        """
        sg = geometry.SphericalGeometry().set_parameters(solar_zenith_angle_deg, self.height_edges_km)
        rads = list(self.radiator_props)
        columns = None
        if self.la_sr is not None and self.la_sr.has_la_srb and self.o2_index is not None:
            air_vcol, air_scol = sg.air_mass(self.air_exo)
            _, o2_scol = sg.air_mass(self.o2_exo)
            o2 = rads[self.o2_index]
            o2_od = self.la_sr.optical_depth(
                o2.optical_depth, o2_scol, air_vcol, air_scol, self.temperature_edge
            )
            rads[self.o2_index] = radiators.RadiatorOpticalProps(o2_od, 0.0, 0.0, is_air=False)
            columns = (air_vcol, air_scol, o2_scol)
        total = radiators.accumulate(rads)
        S, night, valid = solver.build_slant_operator(sg.nid, sg.dsdh)
        rf = solver.solve(total, solar_zenith_angle_deg, self.surface_albedo, S, night, valid)
        return rf, columns

    def radiation_field(self, solar_zenith_angle_deg: float):
        return self._solve(solar_zenith_angle_deg)[0]

    def rate_constants_profile(self, solar_zenith_angle_deg: float, earth_sun_distance: float = 1.0):
        """Return ``{reaction: J[n_levels]}`` (s-1) for the whole column at this solar position."""
        rf, columns = self._solve(solar_zenith_angle_deg)
        # the Fortran scales the radiation field by the Earth-Sun distance before integrating
        flux = photolysis.actinic_flux(
            np.asarray(rf.fdr) * earth_sun_distance,
            np.asarray(rf.fdn) * earth_sun_distance,
            np.asarray(rf.fup) * earth_sun_distance,
            self.etfl,
        )
        out = {name: np.sum(flux * sq, axis=1) for name, sq in self.xsqy.items()}
        if self.o2_reaction is not None:
            # O2 photolysis: the LA/SR effective cross section replaces the base in those bins
            sigma = self.o2_reaction["sigma_base"]
            if columns is not None and self.la_sr is not None:
                air_vcol, air_scol, o2_scol = columns
                sigma = self.la_sr.cross_section(
                    sigma, o2_scol, air_vcol, air_scol, self.temperature_edge
                )
            out[self.o2_reaction["name"]] = np.sum(flux * sigma * self.o2_reaction["phi"], axis=1)
        return out

    def rate_constants(
        self,
        latitude: float,
        longitude: float,
        year: int,
        month: int,
        day: int,
        utc_hour: float,
        altitude_km: float | None = None,
    ):
        """J-values (s-1) at a date/lat/lon/time, optionally interpolated to ``altitude_km``.

        Returns ``{reaction: J}`` at the requested altitude, or ``{reaction: J[n_levels]}`` for the
        full column if ``altitude_km`` is None. Solar zenith angle and Earth-Sun distance use the
        ported TUV-x astronomy (callers may instead drive :meth:`rate_constants_profile` with their
        own SZA, e.g. frank-model's solar.py).
        """
        sza = geometry.solar_zenith_angle(year, month, day, utc_hour, latitude, longitude)
        if sza >= 90.0:
            zero = np.zeros(self.height_edges_km.size)
            prof = {name: zero.copy() for name in self.reaction_names()}
        else:
            esd = geometry.earth_sun_distance(year, month, day, utc_hour)
            prof = self.rate_constants_profile(sza, esd)
        if altitude_km is None:
            return prof
        return {name: float(np.interp(altitude_km, self.height_edges_km, J)) for name, J in prof.items()}

    # ---- construction --------------------------------------------------------------------------
    @classmethod
    def from_tuvx_json(cls, config_path, data_root=None) -> "PhotolysisCalculator":
        config_path = Path(config_path)
        cfg = json.loads(config_path.read_text())
        root = Path(data_root) if data_root is not None else config_path.resolve().parents[2]

        def rel(p):
            return root / p

        # --- grids ---
        wl_edges = None
        height_edges = None
        for g in cfg["grids"]:
            if g["name"] == "wavelength":
                wl_edges = data.load_wavelength_grid(rel(g["file path"]))
            elif g["name"] == "height" and g["type"] == "equal interval":
                height_edges = np.arange(g["begins at"], g["ends at"] + 0.5 * g["cell delta"], g["cell delta"])
        n_lev = height_edges.size
        n_lay = n_lev - 1
        wl_mid = 0.5 * (wl_edges[:-1] + wl_edges[1:])

        # --- profiles ---
        prof_cfg = {p["name"]: p for p in cfg["profiles"]}
        dens = data.load_profile_csv(rel(prof_cfg["air"]["file path"]))
        o3d = data.load_profile_csv(rel(prof_cfg["O3"]["file path"]))
        temp_raw = data.load_profile_csv(rel(prof_cfg["temperature"]["file path"]))
        air = profiles.air_profile(height_edges, dens)
        o2 = profiles.o2_profile(height_edges, dens)
        o3 = profiles.o3_profile(height_edges, o3d)
        temperature = profiles.temperature_profile(height_edges, temp_raw)
        surface_albedo = float(prof_cfg["surface albedo"].get("uniform value", 0.0))

        # extraterrestrial flux
        etfl_cfg = prof_cfg["extraterrestrial flux"]
        interps = etfl_cfg.get("interpolator", [""] * len(etfl_cfg["file path"]))
        flux_files = [
            (data.load_solar_flux(rel(fp)), interps[i] if i < len(interps) else "")
            for i, fp in enumerate(etfl_cfg["file path"])
        ]
        etfl = profiles.extraterrestrial_flux(wl_edges, flux_files)

        # --- Lyman-alpha / Schumann-Runge band parameterization (O2), if configured ---
        o2_params = (cfg.get("O2 absorption") or {}).get("cross section parameters file")
        la_sr = LaSrBands.from_file(wl_edges, rel(o2_params)) if o2_params else None

        # --- radiators (radiative transfer block), kept separate for per-solve LA/SR ---
        rt = cfg["radiative transfer"]
        xs_by_name = {x["name"]: x for x in rt["cross sections"]}
        rad_list = []
        o2_index = None
        for r in rt["radiators"]:
            xs = xs_by_name[r["cross section"]]
            if r.get("treat as air") or xs.get("type") == "air":
                rad_list.append(radiators.air_radiator(air.layer_dens, radiators.rayleigh_cross_section(wl_mid)))
            elif xs.get("type") == "O3":
                o3xs = _build_o3_tint(xs, rel, wl_edges).evaluate(temperature.mid_val)
                rad_list.append(radiators.absorber_radiator(o3.layer_dens, o3xs))
            elif xs.get("type") == "base":
                base = _build_base_xs(xs, rel, wl_edges).evaluate(n_lay)
                dens_prof = o2.layer_dens if r["name"] == "O2" else air.layer_dens
                if r["name"] == "O2":
                    o2_index = len(rad_list)
                rad_list.append(radiators.absorber_radiator(dens_prof, base))
            else:
                raise ValueError(f"unsupported radiator cross section type: {xs.get('type')}")

        # --- reactions: precompute sigma * phi at interfaces (sza-independent) ---
        xsqy = {}
        skipped = {}
        o2_reaction = None
        for rxn in cfg.get("photolysis", {}).get("reactions", []):
            name = rxn["name"]
            xs = rxn["cross section"]
            qy = rxn["quantum yield"]
            try:
                phi = _eval_reaction_qy(qy, rel, wl_edges, wl_mid, temperature, n_lev, wl_mid.size)
                if xs.get("apply O2 bands"):
                    # O2 photolysis: LA/SR effective cross section is applied per solve, so keep the
                    # base cross section and quantum yield separately rather than a fixed sigma*phi.
                    if la_sr is None or not la_sr.has_la_srb:
                        skipped[name] = "needs Lyman-alpha/Schumann-Runge bands (not configured)"
                        continue
                    sigma_base = _eval_reaction_xs(xs, rel, wl_edges, wl_mid, temperature, n_lev)
                    o2_reaction = {"name": name, "sigma_base": sigma_base, "phi": phi}
                    continue
                sigma = _eval_reaction_xs(xs, rel, wl_edges, wl_mid, temperature, n_lev)
            except _Unsupported as exc:
                skipped[name] = str(exc)
                continue
            xsqy[name] = sigma * phi

        return cls(
            wl_edges=wl_edges,
            height_edges_km=height_edges,
            radiator_props=rad_list,
            etfl=etfl,
            surface_albedo=surface_albedo,
            xsqy=xsqy,
            temperature_edge=temperature.edge_val,
            skipped_reactions=skipped,
            la_sr=la_sr,
            o2_index=o2_index,
            air_exo=air.exo_layer_dens,
            o2_exo=o2.exo_layer_dens,
            o2_reaction=o2_reaction,
        )


class _Unsupported(Exception):
    pass


def _build_base_xs(xs_cfg, rel, wl_edges):
    files = []
    for nf in xs_cfg["netcdf files"]:
        td = data.load_cross_section(rel(nf["file path"]))
        lo = (nf.get("lower extrapolation") or {}).get("type")
        up = (nf.get("upper extrapolation") or {}).get("type")
        interp_cfg = nf.get("interpolator") or {}
        interp = interp_cfg.get("type", "conserving")
        fold_in = interp_cfg.get("fold in", False)
        files.append((td, lo, up, interp, fold_in))
    return cross_section.BaseCrossSection.from_files(files, wl_edges)


def _build_o3_tint(xs_cfg, rel, wl_edges):
    tds = [data.load_cross_section(rel(nf["file path"])) for nf in xs_cfg["netcdf files"]]
    return cross_section.O3TintCrossSection.from_files(tds, wl_edges)


def _tds(xs_cfg, rel):
    return [data.load_cross_section(rel(nf["file path"])) for nf in xs_cfg["netcdf files"]]


def _eval_reaction_xs(xs_cfg, rel, wl_edges, wl_mid, temperature, n_lev):
    t = xs_cfg.get("type")
    Te = temperature.edge_val  # photolysis cross sections are evaluated at interfaces
    if t == "base":
        return _build_base_xs(xs_cfg, rel, wl_edges).evaluate(n_lev)
    if t == "O3":
        return _build_o3_tint(xs_cfg, rel, wl_edges).evaluate(Te)
    if t == "Cl2+hv->Cl+Cl":
        return special.cl2_cross_section(wl_mid, Te)
    if t == "HOBr+hv->OH+Br":
        return special.hobr_cross_section(wl_mid, n_lev)
    if t == "HNO3+hv->OH+NO2":
        return special.HNO3CrossSection.from_file(_tds(xs_cfg, rel)[0], wl_edges).evaluate(Te)
    if t == "N2O5+hv->NO2+NO3":
        tds = _tds(xs_cfg, rel)
        return special.N2O5CrossSection.from_files(tds[0], tds[1], wl_edges).evaluate(Te)
    if t == "ClONO2":
        return special.ClONO2CrossSection.from_file(_tds(xs_cfg, rel)[0], wl_edges).evaluate(Te)
    if t == "NO2 tint":
        return special.TintCrossSection.from_files(_tds(xs_cfg, rel), wl_edges).evaluate(Te)
    if t == "OClO+hv->Products":
        return special.OCloCrossSection.from_files(_tds(xs_cfg, rel), wl_edges).evaluate(Te)
    raise _Unsupported(f"cross section type '{t}' not ported")


def _eval_reaction_qy(qy_cfg, rel, wl_edges, wl_mid, temperature, n_lev, n_wl):
    t = qy_cfg.get("type")
    if t in ("ClONO2+hv->Cl+NO3", "ClONO2+hv->ClO+NO2"):
        branch = "Cl+NO3" if t.endswith("Cl+NO3") else "ClO+NO2"
        return clono2_quantum_yield(wl_mid, n_lev, branch)
    if t in ("tint", "NO2 tint"):
        paths = [f["file path"] if isinstance(f, dict) else f for f in qy_cfg["netcdf files"]]
        tds = [data.load_quantum_yield(rel(p)) for p in paths]
        lo = qy_cfg.get("lower extrapolation") or {}
        up = qy_cfg.get("upper extrapolation") or {}
        return TintQuantumYield.from_netcdf(
            tds, wl_edges,
            lower_extrapolation=lo.get("type"), upper_extrapolation=up.get("type"),
            lower_value=lo.get("value", 0.0), upper_value=up.get("value", 0.0),
            extrapolate=(t == "NO2 tint"),  # NO2 tint extrapolates in T; generic tint clamps
        ).evaluate(temperature.edge_val)
    if t != "base":
        raise _Unsupported(f"quantum yield type '{t}' not ported")
    if "constant value" in qy_cfg:
        return ConstantQuantumYield(qy_cfg["constant value"]).evaluate(n_lev, n_wl)
    if "netcdf files" in qy_cfg:
        files = qy_cfg["netcdf files"]
        if len(files) != 1:
            raise _Unsupported("multi-file quantum yield not ported")
        # quantum-yield netcdf files are plain path strings (cross sections use {"file path": ...})
        path = files[0]["file path"] if isinstance(files[0], dict) else files[0]
        td = data.load_quantum_yield(rel(path))
        if td.n_params != 1:
            raise _Unsupported("multi-parameter quantum yield not ported")
        lo = qy_cfg.get("lower extrapolation") or {}
        up = qy_cfg.get("upper extrapolation") or {}
        return TabulatedQuantumYield.from_netcdf(
            td, wl_edges,
            lower_extrapolation=lo.get("type"), upper_extrapolation=up.get("type"),
            lower_value=lo.get("value", 0.0), upper_value=up.get("value", 0.0),
        ).evaluate(n_lev, n_wl)
    raise _Unsupported("quantum yield has neither constant value nor netcdf files")
