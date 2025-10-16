#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INSIDE eFEDS selection with randomized picks, strict common prioritization in Set(1),
rebalanced Set(3), and per-catalog breakdown.

Sets per field (radius = 1.128 deg):
- Set (1) 50%: QSO master (r<23.8), prioritize "common", ensure ≥50% of Set(1) with 1<z<2
- Set (2) 20%: GALAXY master
- Set (3) 30%: nospectra + unWISE (random; rebalance if one catalog is short)

Inputs (current folder):
  field_centers_outside.csv                         (RA_center_deg, DEC_center_deg)
  masterSDSS_V_SDSS_IV_DESI_unique_sources_QSO_ZWARN_RADEC_filters.csv
  masterSDSS_V_SDSS_IV_DESI_unique_sources_GALAXY_ZWARN_RADEC_filters.csv
  common_eFEDS_filtered_extragalactic.fits         (SPECZ_RA, SPECZ_DEC)
  nospectra_eRASS1_Main.v1.1.fits                  (HDU[1]: RA, DEC)
  unWISE_W1SNR5_W2SN3_W2-W1.0.65_RA0p360_DECm6.fits (HDU[1]: ra, dec)

Outputs per field:
  First_selected_sources_<RAcen>_<DECcen>_outsideeFEDs.csv
  Second_selected_sources_<RAcen>_<DECcen>_outsideeFEDs.csv
  Third_selected_sources_<RAcen>_<DECcen>_outsideeFEDs.csv
Summaries:
  summary_outside_fields.csv
  catalog_breakdown_outside_fields.csv
Plots:
  ./plots_outside/plot_<RAcen>_<DECcen>.png
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits

# ------------------- Config -------------------
RADIUS_DEG = 1.128
N_TARGET_PER_FIELD = 1600
COMMON_MATCH_TOL_ARCSEC = 1.0   # 1 arcsec
PLOTS_DIR = "plots_outside"

# Filenames (must be in current folder)
CENTERS_INSIDE = "field_centers_outside.csv"

# Set (1): QSO master + common FITS
QSO_MASTER_CSV = "../masterSDSS_V_SDSS_IV_DESI_unique_sources_QSO_ZWARN_RADEC_filters.csv"   # RA, DEC, redshift, r-mag, g-mag
COMMON_FITS    = "../common_eFEDS_filtered_extragalactic.fits"                                # SPECZ_RA, SPECZ_DEC

# Set (2): GALAXY master
GALAXY_MASTER_CSV = "../masterSDSS_V_SDSS_IV_DESI_unique_sources_GALAXY_ZWARN_RADEC_filters.csv"  # RA, DEC, redshift, r-mag, g-mag

# Set (3): nospectra + unWISE (read via HDU[1])
NOSPEC_FITS = "../nospectra_eRASS1_Main.v1.1.fits"                                            # expects RA, DEC in HDU[1]
UNWISE_FITS = "../unWISE_W1SNR5_W2SN3_W2-W1.0.65_RA0p360_DECm6.fits"                          # expects ra, dec in HDU[1]

# ------------------- Utilities -------------------
def try_read_fits_table(path, required=None):
    """Read first TABLE HDU in 'path' having all 'required' columns (case-insensitive)."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with fits.open(path, memmap=True) as hdul:
        for hdu in hdul:
            if not hasattr(hdu, "data") or hdu.data is None:
                continue
            cols = []
            if hasattr(hdu, "columns"):
                cols = [c.name.upper() for c in hdu.columns]
            if required is not None:
                req = [r.upper() for r in required]
                if not all(r in cols for r in req):
                    continue
            arr = np.array(hdu.data).byteswap().newbyteorder()
            df = pd.DataFrame(arr)
            df.columns = [str(c).upper() for c in df.columns]
            return df
    need = "" if not required else f" with columns {required}"
    raise RuntimeError(f"No table HDU found in {path}{need}.")

def as_float_series(s):
    return pd.to_numeric(s, errors="coerce")

def angsep_deg(ra1_deg, dec1_deg, ra2_deg, dec2_deg):
    """Great-circle separation (deg). Works with broadcastable arrays."""
    r1 = np.deg2rad(ra1_deg); d1 = np.deg2rad(dec1_deg)
    r2 = np.deg2rad(ra2_deg); d2 = np.deg2rad(dec2_deg)
    dra = r2 - r1
    dra = (dra + np.pi) % (2*np.pi) - np.pi
    ddec = d2 - d1
    a = np.sin(ddec/2.0)**2 + np.cos(d1)*np.cos(d2)*np.sin(dra/2.0)**2
    c = 2.0*np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return np.rad2deg(c)

def in_circle(ra_c, dec_c, ra_arr, dec_arr, rdeg):
    return angsep_deg(ra_c, dec_c, ra_arr, dec_arr) <= rdeg

def small_circle_poly(ra_c_deg, dec_c_deg, r_deg=RADIUS_DEG, n=256):
    ra_c = np.deg2rad(ra_c_deg); dec_c = np.deg2rad(dec_c_deg); r = np.deg2rad(r_deg)
    cx = np.cos(dec_c)*np.cos(ra_c); cy = np.cos(dec_c)*np.sin(ra_c); cz = np.sin(dec_c)
    cvec = np.array([cx, cy, cz])
    a = np.array([0.0, 0.0, 1.0])
    if np.allclose(cvec, a, atol=1e-8):
        a = np.array([1.0, 0.0, 0.0])
    u = np.cross(a, cvec); u /= np.linalg.norm(u); v = np.cross(cvec, u)
    ph = np.linspace(0, 2*np.pi, n, endpoint=True)
    pts = (np.cos(r)*cvec.reshape(3,1) + np.sin(r)*(u.reshape(3,1)*np.cos(ph) + v.reshape(3,1)*np.sin(ph)))
    xs, ys, zs = pts; decs = np.arcsin(zs); ras = np.arctan2(ys, xs) % (2*np.pi)
    return np.rad2deg(ras), np.rad2deg(decs)

def wrap_ra360(x):
    return np.mod(np.asarray(x), 360.0)

def plot_circle_wrapped(ax, ra_arr_deg, dec_arr_deg, **kwargs):
    ra_wrapped = wrap_ra360(ra_arr_deg)
    dec_vals = np.asarray(dec_arr_deg)
    dr = np.abs(np.diff(ra_wrapped))
    breaks = np.where(dr > 180.0)[0] + 1
    segments = np.split(np.arange(len(ra_wrapped)), breaks)
    for seg in segments:
        ax.plot(ra_wrapped[seg], dec_vals[seg], **kwargs)

def match_positions_arcsec(ra1, dec1, ra2, dec2, tol_arcsec=1.0, chunk=10000):
    """Boolean mask for ra1/dec1 entries that have ≥1 match in ra2/dec2 within tol_arcsec."""
    tol_deg = tol_arcsec / 3600.0
    matched1 = np.zeros(len(ra1), dtype=bool)
    N = len(ra1)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        d = angsep_deg(ra1[s:e][:, None], dec1[s:e][:, None], ra2[None, :], dec2[None, :])
        matched1[s:e] = (np.nanmin(d, axis=1) <= tol_deg)
    return matched1

def safe_filename_from_center(ra_c, dec_c):
    def enc(x): return f"{x:.5f}".replace("-", "m").replace("+", "p")
    return f"{enc(ra_c)}_{enc(dec_c)}"

def field_rng(base_tag: str):
    """Deterministic RNG per field (same field -> same random picks across runs)."""
    seed = np.uint32(abs(hash(base_tag)) & 0xFFFFFFFF)
    return np.random.RandomState(int(seed))

# ------------------- Main -------------------
def main():
    required_files = [
        CENTERS_INSIDE, QSO_MASTER_CSV, COMMON_FITS,
        GALAXY_MASTER_CSV, NOSPEC_FITS, UNWISE_FITS
    ]
    missing = [p for p in required_files if not os.path.exists(p)]
    if missing:
        print("ERROR: Missing required file(s):")
        for p in missing:
            print(" -", p)
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Load field centres (outside eFEDS)
    centers = pd.read_csv(CENTERS_INSIDE)
    if not {"RA_center_deg", "DEC_center_deg"}.issubset(centers.columns):
        raise ValueError(f"{CENTERS_INSIDE} must contain RA_center_deg and DEC_center_deg.")
    centers = centers.rename(columns={"RA_center_deg": "RA_CEN", "DEC_center_deg": "DEC_CEN"})

    # -------------------- Set (1) inputs --------------------
    # QSO master
    qso = pd.read_csv(QSO_MASTER_CSV)
    need_qso = {"RA", "DEC", "redshift", "r-mag"}
    if not need_qso.issubset(qso.columns):
        raise ValueError(f"{QSO_MASTER_CSV} missing columns {need_qso - set(qso.columns)}")
    qso["RA"] = as_float_series(qso["RA"]) % 360.0
    qso["DEC"] = as_float_series(qso["DEC"])
    qso["redshift"] = as_float_series(qso["redshift"])
    qso["r-mag"] = as_float_series(qso["r-mag"])
    qso = qso.dropna(subset=["RA", "DEC"]).reset_index(drop=True)

    # "common" FITS (for prioritization in Set 1)
    common_df = try_read_fits_table(COMMON_FITS, required=["SPECZ_RA", "SPECZ_DEC"])
    common_df["SPECZ_RA"] = as_float_series(common_df["SPECZ_RA"]) % 360.0
    common_df["SPECZ_DEC"] = as_float_series(common_df["SPECZ_DEC"])
    common_df = common_df.dropna(subset=["SPECZ_RA", "SPECZ_DEC"]).reset_index(drop=True)

    # -------------------- Set (2) inputs: GALAXY master --------------------
    gal = pd.read_csv(GALAXY_MASTER_CSV)
    need_gal = {"RA", "DEC", "redshift", "r-mag"}
    if not need_gal.issubset(gal.columns):
        raise ValueError(f"{GALAXY_MASTER_CSV} missing columns {need_gal - set(gal.columns)}")
    gal["RA"] = as_float_series(gal["RA"]) % 360.0
    gal["DEC"] = as_float_series(gal["DEC"])
    gal["redshift"] = as_float_series(gal["redshift"])
    gal["r-mag"] = as_float_series(gal["r-mag"])
    gal = gal.dropna(subset=["RA", "DEC"]).reset_index(drop=True)

    # -------------------- Set (3) inputs: nospec + unWISE via HDU[1] --------------------
    # nospec (expects RA, DEC)
    with fits.open(NOSPEC_FITS, memmap=True) as hdul:
        if len(hdul) < 2 or hdul[1].data is None:
            raise RuntimeError(f"{NOSPEC_FITS}: HDU[1] has no table data.")
        arr = np.array(hdul[1].data).byteswap().newbyteorder()
    nospec_df = pd.DataFrame(arr)
    cols_lower = {c.lower(): c for c in nospec_df.columns}
    ra_key = cols_lower.get("ra")
    dec_key = cols_lower.get("dec")
    if ra_key is None or dec_key is None:
        raise ValueError(f"{NOSPEC_FITS} must contain RA and DEC columns in HDU[1].")
    nospec_df["RA_ANY"]  = as_float_series(nospec_df[ra_key]) % 360.0
    nospec_df["DEC_ANY"] = as_float_series(nospec_df[dec_key])

    # unWISE (expects ra, dec)
    with fits.open(UNWISE_FITS, memmap=True) as hdul:
        if len(hdul) < 2 or hdul[1].data is None:
            raise RuntimeError(f"{UNWISE_FITS}: HDU[1] has no table data.")
        arr = np.array(hdul[1].data).byteswap().newbyteorder()
    unwise_df = pd.DataFrame(arr)
    cols_lower = {c.lower(): c for c in unwise_df.columns}
    uw_ra_key = cols_lower.get("ra")
    uw_dec_key = cols_lower.get("dec")
    if uw_ra_key is None or uw_dec_key is None:
        raise ValueError(f"{UNWISE_FITS} must contain ra and dec columns in HDU[1].")
    unwise_df["RA_UW"]  = as_float_series(unwise_df[uw_ra_key]) % 360.0
    unwise_df["DEC_UW"] = as_float_series(unwise_df[uw_dec_key])

    # -------------------- Iterate fields --------------------
    summary_rows = []
    breakdown_rows = []   # per-catalog breakdown

    for _, crow in centers.iterrows():
        ra_c = float(crow["RA_CEN"])
        dec_c = float(crow["DEC_CEN"])
        base_tag = safe_filename_from_center(ra_c, dec_c)
        rs = field_rng(base_tag)  # Deterministic RNG for this field

        # Target counts
        N1 = int(round(0.50 * N_TARGET_PER_FIELD))  # Set 1 (QSO)
        N2 = int(round(0.20 * N_TARGET_PER_FIELD))  # Set 2 (GALAXY)
        N3 = N_TARGET_PER_FIELD - N1 - N2           # Set 3 (nospec+unWISE)
        N3a = N3 // 2
        N3b = N3 - N3a

        # ---------- Set (1): QSO master r<23.8, prioritize common, ensure ≥50% with 1<z<2 ----------
        m_in = qso[in_circle(ra_c, dec_c, qso["RA"].values, qso["DEC"].values, RADIUS_DEG)].copy()
        m_in = m_in[m_in["r-mag"] < 23.8]

        if len(m_in):
            common_mask = match_positions_arcsec(
                m_in["RA"].values, m_in["DEC"].values,
                common_df["SPECZ_RA"].values, common_df["SPECZ_DEC"].values,
                tol_arcsec=COMMON_MATCH_TOL_ARCSEC
            )
            m_in["IS_COMMON"] = common_mask
        else:
            m_in["IS_COMMON"] = []

        # Build disjoint pools
        z12 = (m_in["redshift"] > 1.0) & (m_in["redshift"] < 2.0)
        common = m_in["IS_COMMON"].astype(bool)

        common_z12      = m_in[ common &  z12]
        common_rest     = m_in[ common & ~z12]
        noncommon_z12   = m_in[~common &  z12]
        noncommon_rest  = m_in[~common & ~z12]

        # Targets
        need_total = N1
        need_z12   = int(math.floor(0.5 * need_total))  # at least half in 1<z<2

        sel1_parts = []

        # 1) Satisfy z∈(1,2) requirement with common first, then non-common
        take = min(len(common_z12), need_z12)
        if take > 0:
            sel1_parts.append(common_z12.sample(n=take, random_state=rs))
        need_z12 -= take

        if need_z12 > 0:
            take = min(len(noncommon_z12), need_z12)
            if take > 0:
                sel1_parts.append(noncommon_z12.sample(n=take, random_state=rs))
            need_z12 -= take

        # Compute how many more to reach total N1
        already_idx = pd.concat(sel1_parts).index if sel1_parts else pd.Index([])
        remaining   = max(0, need_total - len(already_idx))

        # 2) Fill remaining from COMMON pool (any z) first
        if remaining > 0:
            c_rest = pd.concat([common_z12, common_rest]).drop(index=already_idx, errors="ignore")
            take = min(len(c_rest), remaining)
            if take > 0:
                sel1_parts.append(c_rest.sample(n=take, random_state=rs))
            remaining -= take

        # 3) Then fill from NON-COMMON pool (any z)
        if remaining > 0:
            nc_rest = pd.concat([noncommon_z12, noncommon_rest]).drop(index=already_idx, errors="ignore")
            if sel1_parts:
                chosen_idx = pd.concat(sel1_parts).index
                nc_rest = nc_rest.drop(index=chosen_idx, errors="ignore")
            take = min(len(nc_rest), remaining)
            if take > 0:
                sel1_parts.append(nc_rest.sample(n=take, random_state=rs))
            remaining -= take

        sel1_full = pd.concat(sel1_parts, ignore_index=True) if sel1_parts else pd.DataFrame(columns=m_in.columns)

        # counts for breakdown BEFORE dropping IS_COMMON
        sel1_total     = len(sel1_full)
        sel1_common    = int(sel1_full["IS_COMMON"].sum()) if "IS_COMMON" in sel1_full.columns and len(sel1_full) else 0
        sel1_noncommon = sel1_total - sel1_common

        sel1 = sel1_full.rename(columns={"RA": "RA_sel", "DEC": "DEC_sel", "redshift": "redshift_sel", "r-mag": "rmag_sel"})
        sel1 = sel1[["RA_sel", "DEC_sel", "redshift_sel", "rmag_sel"]].copy()

        # ---------- Set (2): GALAXY master (RANDOM) ----------
        g_in = gal[in_circle(ra_c, dec_c, gal["RA"].values, gal["DEC"].values, RADIUS_DEG)].copy()
        g_in = g_in.rename(columns={"RA": "RA_sel", "DEC": "DEC_sel", "redshift": "redshift_sel", "r-mag": "rmag_sel"})
        if len(g_in) > 0 and N2 > 0:
            sel2 = g_in.sample(n=min(N2, len(g_in)), random_state=rs)[["RA_sel", "DEC_sel", "redshift_sel", "rmag_sel"]].copy()
        else:
            sel2 = pd.DataFrame(columns=["RA_sel", "DEC_sel", "redshift_sel", "rmag_sel"])
        sel2_total = len(sel2)

        # ---------- Set (3): nospectra + unWISE (RANDOM, with rebalancing) ----------
        # Pools outside the field
        ns_mask = in_circle(ra_c, dec_c, nospec_df["RA_ANY"].values, nospec_df["DEC_ANY"].values, RADIUS_DEG)
        uw_mask = in_circle(ra_c, dec_c, unwise_df["RA_UW"].values, unwise_df["DEC_UW"].values, RADIUS_DEG)

        ns_in = nospec_df[ns_mask]
        uw_in = unwise_df[uw_mask]

        ns_avail = len(ns_in)
        uw_avail = len(uw_in)

        # Start with half-half targets
        ns_take = min(ns_avail, N3a)
        uw_take = min(uw_avail, N3b)

        total_take = ns_take + uw_take
        if total_take < N3:
            remaining = N3 - total_take
            rem_ns = ns_avail - ns_take
            rem_uw = uw_avail - uw_take
            # Fill from the side with more remaining availability first (ties -> favor unWISE)
            fill_order = ["uw", "ns"] if rem_uw >= rem_ns else ["ns", "uw"]
            for side in fill_order:
                if remaining <= 0:
                    break
                if side == "uw" and rem_uw > 0:
                    add = min(remaining, rem_uw)
                    uw_take += add
                    remaining -= add
                    rem_uw -= add
                elif side == "ns" and rem_ns > 0:
                    add = min(remaining, rem_ns)
                    ns_take += add
                    remaining -= add
                    rem_ns -= add

        # Sample rows (without replacement) with per-field RNG
        if ns_take > 0:
            ns_pick = ns_in.sample(n=ns_take, random_state=rs)
            sel3a = pd.DataFrame({
                "RA_sel": ns_pick["RA_ANY"].values,
                "DEC_sel": ns_pick["DEC_ANY"].values,
                "redshift_sel": np.full(ns_take, np.nan),
                "rmag_sel": np.full(ns_take, np.nan),
            })
        else:
            sel3a = pd.DataFrame(columns=["RA_sel","DEC_sel","redshift_sel","rmag_sel"])

        if uw_take > 0:
            uw_pick = uw_in.sample(n=uw_take, random_state=rs)
            sel3b = pd.DataFrame({
                "RA_sel": uw_pick["RA_UW"].values,
                "DEC_sel": uw_pick["DEC_UW"].values,
                "redshift_sel": np.full(uw_take, np.nan),
                "rmag_sel": np.full(uw_take, np.nan),
            })
        else:
            sel3b = pd.DataFrame(columns=["RA_sel","DEC_sel","redshift_sel","rmag_sel"])

        sel3 = pd.concat([sel3a, sel3b], ignore_index=True)
        sel3_nospec = len(sel3a)
        sel3_unwise = len(sel3b)
        sel3_total = len(sel3)

        # ---------- Write per-field CSVs ----------
        ftag = safe_filename_from_center(ra_c, dec_c)
        f1 = f"First_selected_sources_{ftag}_outsideeFEDs.csv"
        f2 = f"Second_selected_sources_{ftag}_outsideeFEDs.csv"
        f3 = f"Third_selected_sources_{ftag}_outsideeFEDs.csv"

        sel1.rename(columns={"RA_sel": "RA", "DEC_sel": "DEC", "redshift_sel": "redshift", "rmag_sel": "rmag"}).to_csv(f1, index=False)
        sel2.rename(columns={"RA_sel": "RA", "DEC_sel": "DEC", "redshift_sel": "redshift", "rmag_sel": "rmag"}).to_csv(f2, index=False)
        sel3.rename(columns={"RA_sel": "RA", "DEC_sel": "DEC", "redshift_sel": "redshift", "rmag_sel": "rmag"}).to_csv(f3, index=False)

        # ---------- Plot per-field ----------
        fig, ax = plt.subplots(figsize=(9, 6))
        if len(sel1):
            ax.plot(wrap_ra360(sel1["RA_sel"]), sel1["DEC_sel"], linestyle="none", marker="o", markersize=3, alpha=0.9, label=f"Set(1) QSO [{len(sel1)}]")
        if len(sel2):
            ax.plot(wrap_ra360(sel2["RA_sel"]), sel2["DEC_sel"], linestyle="none", marker="^", markersize=3, alpha=0.9, label=f"Set(2) GALAXY [{len(sel2)}]")
        if len(sel3):
            ax.plot(wrap_ra360(sel3["RA_sel"]), sel3["DEC_sel"], linestyle="none", marker="s", markersize=3, alpha=0.9, label=f"Set(3) nospec+unWISE [{len(sel3)}]")
        cr_ra, cr_dec = small_circle_poly(ra_c, dec_c, RADIUS_DEG, n=256)
        plot_circle_wrapped(ax, cr_ra, cr_dec, linewidth=1.3)
        ax.set_xlim(0, 360)
        ax.set_ylim(-30, 15)
        ax.set_xlabel("Right Ascension [deg]")
        ax.set_ylabel("Declination [deg]")
        ax.set_title(f"Inside eFEDS field @ RA={ra_c:.4f}, Dec={dec_c:.4f}\n"
                     f"S1={len(sel1)} (common={sel1_common})  S2={len(sel2)}  "
                     f"S3={len(sel3)} (ns={sel3_nospec}, uw={sel3_unwise})")
        ax.legend(loc="best", frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"plot_{ftag}.png"), dpi=180)
        plt.close(fig)

        # ---------- Original summary ----------
        c1, c2, c3 = len(sel1), len(sel2), len(sel3)
        tot = c1 + c2 + c3
        summary_rows.append({
            "RA_center_deg": ra_c,
            "DEC_center_deg": dec_c,
            "set1_count": c1, "set1_pct": 100.0 * c1 / max(1, tot),
            "set2_count": c2, "set2_pct": 100.0 * c2 / max(1, tot),
            "set3_count": c3, "set3_pct": 100.0 * c3 / max(1, tot),
            "total_selected": tot, "target_total": N_TARGET_PER_FIELD
        })

        # ---------- Per-catalog breakdown ----------
        breakdown_rows.append({
            "RA_center_deg": ra_c, "DEC_center_deg": dec_c,
            "sel1_total": sel1_total,
            "sel1_common": sel1_common,
            "sel1_noncommon": sel1_noncommon,
            "sel2_total": sel2_total,
            "sel3_total": sel3_total,
            "sel3_nospec": sel3_nospec,
            "sel3_unwise": sel3_unwise,
            "grand_total": sel1_total + sel2_total + sel3_total
        })

    # Write summaries
    summary = pd.DataFrame(summary_rows)
    summary.to_csv("summary_outside_fields.csv", index=False)

    breakdown = pd.DataFrame(breakdown_rows)
    breakdown.to_csv("catalog_breakdown_outside_fields.csv", index=False)

    # Print breakdown (compact)
    print("\n=== Catalog breakdown per field ===")
    cols_show = [
        "RA_center_deg","DEC_center_deg",
        "sel1_total","sel1_common","sel1_noncommon",
        "sel2_total",
        "sel3_total","sel3_nospec","sel3_unwise",
        "grand_total"
    ]
    print(breakdown[cols_show].to_string(index=False))

    print("\n✅ Done.")
    print("  - Per-field CSVs: First_*, Second_*, Third_*")
    print("  - Plots in:", PLOTS_DIR)
    print("  - Summary: summary_outside_fields.csv")
    print("  - Catalog breakdown: catalog_breakdown_outside_fields.csv")

if __name__ == "__main__":
    main()

