# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Generate Python-vs-Fortran consistency plots for the tuvx_photolysis port.

Run from anywhere:  python validation/validate_plots.py
Produces PNGs in this directory comparing the JAX port to the Fortran reference
(tuv_5_4_no_aerosol_reference.nc):

  1. radiation_field.png    -- total actinic flux spectrum, Python vs Fortran, at several altitudes
  2. j_profiles.png         -- J(altitude) overlaid for representative reactions, with % difference
  3. j_scatter.png          -- Python J vs Fortran J (1:1) for all fully-ported reactions
  4. j_error_vs_uv.png      -- median J error vs deep-UV (<206 nm) contribution (the LA/SR story)

These double as committed regression artifacts.
"""

from pathlib import Path

import numpy as np
import netCDF4
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tuvx_photolysis import PhotolysisCalculator, photolysis

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIX = REPO / "tests" / "fixtures"
REF = FIX / "tuv_5_4_no_aerosol_reference.nc"
LA_SR_MAX_NM = 206.0
TIME_INDEX = 1  # the 28.2-degree SZA case


def main():
    calc = PhotolysisCalculator.from_tuvx_json(FIX / "tuv_5_4_no_aerosol.json", data_root=REPO)
    wl_mid = 0.5 * (calc.wl_edges[:-1] + calc.wl_edges[1:])
    altitude = calc.height_edges_km

    with netCDF4.Dataset(REF) as ds:
        sza = np.array(ds.variables["solar zenith angle"][:])
        esd = np.array(ds.variables["Earth-Sun distance"][:])
        ref_dir = np.array(ds.variables["direct radiation"][:])
        ref_up = np.array(ds.variables["upward radiation"][:])
        ref_dn = np.array(ds.variables["downward radiation"][:])
        ref_J = {n: np.array(ds.variables[n][:]) for n in calc.xsqy if n in ds.variables}

    ti = TIME_INDEX
    rf = calc.radiation_field(sza[ti])
    fdr, fdn, fup = (np.asarray(x) * esd[ti] for x in (rf.fdr, rf.fdn, rf.fup))
    Jprof = calc.rate_constants_profile(sza[ti], esd[ti])
    flux = photolysis.actinic_flux(fdr, fdn, fup, calc.etfl)

    _plot_radiation_field(wl_mid, altitude, fdr + fdn + fup,
                          (ref_dir + ref_up + ref_dn)[:, :, ti], sza[ti])
    _plot_j_profiles(altitude, Jprof, ref_J, ti)
    _plot_j_scatter(calc, flux, wl_mid, Jprof, ref_J, ti)
    _plot_error_vs_uv(calc, flux, wl_mid, Jprof, ref_J, ti)
    print(f"Wrote plots to {HERE}")


def _plot_radiation_field(wl_mid, altitude, mine, ref, sza):
    # mine: (n_levels, n_wl); ref: (n_wl, n_levels)
    ref = ref.T
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for z_km in (0.0, 20.0, 50.0):
        k = int(np.argmin(np.abs(altitude - z_km)))
        ax1.plot(wl_mid, ref[k], lw=2.5, alpha=0.4, color="k")
        ax1.plot(wl_mid, mine[k], lw=1.0, label=f"{altitude[k]:.0f} km")
    ax1.set_yscale("log")
    ax1.set_xlabel("wavelength (nm)")
    ax1.set_ylabel("total actinic flux (norm.)")
    ax1.set_title(f"Radiation field (thick grey = Fortran, thin = Python)\nSZA = {sza:.1f}°")
    ax1.legend()
    ax1.axvspan(wl_mid[0], LA_SR_MAX_NM, color="red", alpha=0.08)

    k = int(np.argmin(np.abs(altitude - 20.0)))
    ratio = np.where(ref[k] > 0, mine[k] / np.where(ref[k] > 0, ref[k], 1), np.nan)
    ax2.plot(wl_mid, ratio, lw=1.0)
    ax2.axhline(1.0, color="k", lw=0.6)
    ax2.axvspan(wl_mid[0], LA_SR_MAX_NM, color="red", alpha=0.12, label="LA/SR (deferred)")
    ax2.set_ylim(0.9, 1.1)
    ax2.set_xlabel("wavelength (nm)")
    ax2.set_ylabel("Python / Fortran")
    ax2.set_title("Ratio at 20 km (≈1 outside LA/SR band)")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(HERE / "radiation_field.png", dpi=120)
    plt.close(fig)


def _plot_j_profiles(altitude, Jprof, ref_J, ti):
    reactions = [r for r in ["O3+hv->O2+O(1D)", "NO2+hv->NO+O(3P)", "CH2O+hv->H+HCO",
                             "HNO4+hv->HO2+NO2"] if r in Jprof and r in ref_J]
    fig, axes = plt.subplots(1, len(reactions), figsize=(3.6 * len(reactions), 5), sharey=True)
    if len(reactions) == 1:
        axes = [axes]
    for ax, name in zip(axes, reactions):
        mine = Jprof[name]
        ref = ref_J[name][:, ti]
        ax.plot(ref, altitude, lw=3, alpha=0.4, color="k", label="Fortran")
        ax.plot(mine, altitude, lw=1.2, color="C0", label="Python")
        ax.set_xscale("log")
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("J (s$^{-1}$)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("altitude (km)")
    axes[0].legend()
    fig.suptitle("Photolysis rate constants vs altitude (Python vs Fortran)")
    fig.tight_layout()
    fig.savefig(HERE / "j_profiles.png", dpi=120)
    plt.close(fig)


def _uv_fraction(calc, flux, wl_mid, name):
    integ = flux * calc.xsqy[name]
    tot = integ.sum()
    return integ[:, wl_mid < LA_SR_MAX_NM].sum() / tot if tot > 0 else 0.0


def _plot_j_scatter(calc, flux, wl_mid, Jprof, ref_J, ti):
    mine_all, ref_all = [], []
    for name in calc.xsqy:
        if name not in ref_J:
            continue
        if _uv_fraction(calc, flux, wl_mid, name) > 1e-4:
            continue  # exclude LA/SR-limited reactions
        r = ref_J[name][:, ti]
        m = r > r.max() * 1e-3
        mine_all.append(Jprof[name][m])
        ref_all.append(r[m])
    mine_all = np.concatenate(mine_all)
    ref_all = np.concatenate(ref_all)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.loglog(ref_all, mine_all, ".", ms=3, alpha=0.4)
    lim = [min(ref_all.min(), mine_all.min()), max(ref_all.max(), mine_all.max())]
    ax.plot(lim, lim, "k-", lw=1, label="1:1")
    ax.set_xlabel("Fortran J (s$^{-1}$)")
    ax.set_ylabel("Python J (s$^{-1}$)")
    ax.set_title(f"J: Python vs Fortran, {len(mine_all)} points\n(reactions dominated by λ≥206 nm)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "j_scatter.png", dpi=120)
    plt.close(fig)


def _plot_error_vs_uv(calc, flux, wl_mid, Jprof, ref_J, ti):
    fracs, errs = [], []
    for name in calc.xsqy:
        if name not in ref_J:
            continue
        r = ref_J[name][:, ti]
        m = r > r.max() * 1e-3
        if m.sum() < 5:
            continue
        fracs.append(_uv_fraction(calc, flux, wl_mid, name))
        errs.append(np.median(np.abs(Jprof[name][m] - r[m]) / r[m]))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(np.array(fracs) + 1e-8, np.clip(errs, 1e-9, None), s=18, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("deep-UV (<206 nm) fraction of J")
    ax.set_ylabel("median |Python−Fortran| / Fortran")
    ax.set_title("J error is set by the deep-UV (deferred LA/SR) contribution")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "j_error_vs_uv.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
