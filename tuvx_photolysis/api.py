# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Top-level orchestrator: PhotolysisCalculator.

Reference: ``src/core.F90`` (``run``) and ``src/tuvx.F90``. Ties the pipeline together:
geometry -> radiator optical properties -> delta-Eddington radiation field -> actinic flux ->
per-reaction wavelength integration -> J profile (and J interpolated to an altitude).

:meth:`PhotolysisCalculator.from_tuvx_json` parses a TUV-x JSON config (the air/O2/O3 radiators,
base/O3 cross sections, and constant/tabulated quantum yields). The O2 ``apply O2 bands`` /
Lyman-alpha-Schumann-Runge parameterization is ported (see ``la_sr_bands.py``); it is only skipped
when the grid does not carry the LA/SR band edges. Reactions whose recipes are still unported
(unsupported special cross-section/quantum-yield modules) are recorded in ``skipped_reactions``
rather than silently dropped.
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
    hno4_branching_quantum_yield,
    clooocl_branching_quantum_yield,
    o3_o1d_quantum_yield,
    o3_o3p_quantum_yield,
)

__all__ = ["PhotolysisCalculator"]

#: Planck constant x speed of light [J m] -- matches tuv-x src/constants.F90 hc (and profiles._HC).
_HC_J_M = 6.626068e-34 * 2.99792458e8

#: Photochemical-heating energy terms (threshold wavelength [nm]) per reaction, from ts1_tsmlt.json's
#: heating config. O3 both channels (Hartley/Huggins O1D + Chappuis/Huggins O3P). O2 (jo2) deferred --
#: it needs the LA/SR-corrected cross section and is minor at ~19 km (AD-5.1).
_HEATING_ENERGY_TERMS_NM = {
    "O3+hv->O2+O(1D)": 310.32,
    "O3+hv->O2+O(3P)": 1179.87,
}


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
    # JPL product-branching channels: name -> sigma*phi_channel (opt-in via branching=True). These
    # split HNO4 / ClOOCl photolysis into their product channels using the JPL branching quantum
    # yields; they are NOT Fortran-comparable (TUV-x carries only one channel each).
    branching_specs: dict = field(default_factory=dict)
    #: Optional dynamic aerosol radiator (RadiatorOpticalProps, shape (n_layers, n_wl)), injected per
    #: solve. Set by the coupled driver from the TOMAS aerosol (Phase 4, aerosol->photolysis); None ->
    #: no aerosol (identical to the pre-Phase-4 solve). Kept as mutable state so the coupled loop can
    #: update it each outer step without rebuilding the (cached) calculator.
    aerosol_props: object = None

    def reaction_names(self, branching: bool = False):
        names = list(self.xsqy)
        if self.o2_reaction is not None:
            names.append(self.o2_reaction["name"])
        if branching:
            names = [n for n in names if n not in self.branching_specs]
            names += list(self.branching_specs)
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
        if self.aerosol_props is not None:      # Phase 4: dynamic aerosol -> photolysis feedback
            rads.append(self.aerosol_props)
        total = radiators.accumulate(rads)
        S, night, valid = solver.build_slant_operator(sg.nid, sg.dsdh)
        rf = solver.solve(total, solar_zenith_angle_deg, self.surface_albedo, S, night, valid)
        return rf, columns

    def radiation_field(self, solar_zenith_angle_deg: float):
        return self._solve(solar_zenith_angle_deg)[0]

    def rate_constants_profile(
        self, solar_zenith_angle_deg: float, earth_sun_distance: float = 1.0, branching: bool = False
    ):
        """Return ``{reaction: J[n_levels]}`` (s-1) for the whole column at this solar position.

        With ``branching=True`` the JPL product-branching quantum yields are applied to HNO4 and
        ClOOCl: their primary channels are corrected (e.g. HNO4->HO2+NO2 scaled by ~0.8) and the
        secondary channels (HNO4->OH+NO3, ClOOCl->ClO+ClO) are added. Default ``False`` keeps the
        output faithful to the Fortran TUV-x (single channel with unit quantum yield).
        """
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
        if branching:
            for name, sq in self.branching_specs.items():
                out[name] = np.sum(flux * sq, axis=1)  # overrides primary, adds secondary channels
        return out

    def heating_and_actinic_flux(self, solar_zenith_angle_deg: float, earth_sun_distance: float = 1.0):
        """One radiation solve -> ``(heating, actinic_flux)``.

        ``heating`` is ``{reaction: heating[n_levels]}`` [J s-1 per absorber molecule] for the reactions
        with a photochemical-heating energy term (O3 channels; O2 deferred, AD-5.1). ``actinic_flux`` is
        the ``(n_levels, n_wl)`` flux [photon cm-2 s-1] (fdr+fdn+fup)*etfl, so callers can also form the
        aerosol shortwave-absorption heating (Σ_λ flux·b_abs·E_photon) from the same solve. Ports
        ``heating_rates.F90``: ``energy(λ)=max(0, hc(1/λ − 1/λ_threshold))``,
        ``heating(z)=Σ_λ actinic(λ,z)·energy(λ)·σφ(λ,z)``, reusing the SAME actinic flux + channel σφ as
        ``rate_constants_profile`` (photolysis and heating consistent). Multiply by [absorber] for a
        volumetric rate.
        """
        rf, _columns = self._solve(solar_zenith_angle_deg)
        flux = photolysis.actinic_flux(
            np.asarray(rf.fdr) * earth_sun_distance,
            np.asarray(rf.fdn) * earth_sun_distance,
            np.asarray(rf.fup) * earth_sun_distance,
            self.etfl,
        )
        wl_mid = 0.5 * (self.wl_edges[:-1] + self.wl_edges[1:])
        heating = {}
        for name, e_thr in _HEATING_ENERGY_TERMS_NM.items():
            if name not in self.xsqy:
                continue
            energy = np.maximum(0.0, _HC_J_M * 1.0e9 * (e_thr - wl_mid) / (e_thr * wl_mid))  # (n_wl,) [J]
            heating[name] = np.sum(flux * (energy[None, :] * self.xsqy[name]), axis=1)
        return heating, flux

    def heating_rate_profile(self, solar_zenith_angle_deg: float, earth_sun_distance: float = 1.0):
        """``{reaction: heating[n_levels]}`` [J s-1 per absorber molecule] (see
        :meth:`heating_and_actinic_flux`, which this wraps)."""
        return self.heating_and_actinic_flux(solar_zenith_angle_deg, earth_sun_distance)[0]

    def rate_constants(
        self,
        latitude: float,
        longitude: float,
        year: int,
        month: int,
        day: int,
        utc_hour: float,
        altitude_km: float | None = None,
        branching: bool = False,
    ):
        """J-values (s-1) at a date/lat/lon/time, optionally interpolated to ``altitude_km``.

        Returns ``{reaction: J}`` at the requested altitude, or ``{reaction: J[n_levels]}`` for the
        full column if ``altitude_km`` is None. Solar zenith angle and Earth-Sun distance use the
        ported TUV-x astronomy (callers may instead drive :meth:`rate_constants_profile` with their
        own SZA, e.g. frank-model's solar.py). ``branching=True`` applies the JPL product-branching
        quantum yields for HNO4 and ClOOCl (see :meth:`rate_constants_profile`).
        """
        sza = geometry.solar_zenith_angle(year, month, day, utc_hour, latitude, longitude)
        if sza >= 90.0:
            zero = np.zeros(self.height_edges_km.size)
            prof = {name: zero.copy() for name in self.reaction_names(branching)}
        else:
            esd = geometry.earth_sun_distance(year, month, day, utc_hour)
            prof = self.rate_constants_profile(sza, esd, branching=branching)
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
                # DEFERRED: the aerosol radiator (config ``"type": "aerosol"`` with explicit optical
                # depths / SSA / g, e.g. examples/tuv_5_4.json) is not yet wired here, so the aerosol
                # config currently cannot be loaded. ``radiators.aerosol_radiator`` exists but is
                # unused, and tests/fixtures/tuv_5_4_reference.nc is not compared. See REVIEW_FINDINGS.md.
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

        # --- JPL product-branching channels for HNO4 and ClOOCl ---
        # Both reactions use a base cross section with unit quantum yield in the config, so their
        # xsqy entry IS the shared absorption cross section sigma. The JPL branching quantum yields
        # split that absorption into product channels (Table 4C-9-2 for HNO4; Section F7 for ClOOCl).
        branching_specs = {}
        n_wl = wl_mid.size
        if "HNO4+hv->HO2+NO2" in xsqy:
            sigma_hno4 = xsqy["HNO4+hv->HO2+NO2"]  # phi == 1 in the config
            branching_specs["HNO4+hv->HO2+NO2"] = sigma_hno4 * hno4_branching_quantum_yield(
                wl_mid, n_lev, "HO2+NO2")
            branching_specs["HNO4+hv->OH+NO3"] = sigma_hno4 * hno4_branching_quantum_yield(
                wl_mid, n_lev, "OH+NO3")
        if "ClOOCl+hv->Cl+ClOO" in xsqy:
            sigma_clooocl = xsqy["ClOOCl+hv->Cl+ClOO"]
            branching_specs["ClOOCl+hv->Cl+ClOO"] = sigma_clooocl * clooocl_branching_quantum_yield(
                wl_mid, n_lev, "Cl+ClOO")
            branching_specs["ClOOCl+hv->ClO+ClO"] = sigma_clooocl * clooocl_branching_quantum_yield(
                wl_mid, n_lev, "2ClO")

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
            branching_specs=branching_specs,
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
    if t == "O3+hv->O2+O(1D)":
        return o3_o1d_quantum_yield(wl_mid, temperature.edge_val)
    if t == "O3+hv->O2+O(3P)":
        return o3_o3p_quantum_yield(wl_mid, temperature.edge_val)
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
