#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["axes.linewidth"] = 2
mpl.rcParams["xtick.major.width"] = 2
mpl.rcParams["ytick.major.width"] = 2
mpl.rcParams["xtick.labelsize"] = 16
mpl.rcParams["ytick.labelsize"] = 16
mpl.rcParams["axes.titlesize"] = 24
mpl.rcParams["axes.labelsize"] = 16

# Optional extras (footprint adornments)
try:
    import astropy.io.fits as fits
    HAVE_FITS = True
except Exception:
    HAVE_FITS = False
try:
    from scipy.spatial import ConvexHull
    HAVE_HULL = True
except Exception:
    HAVE_HULL = False
try:
    from sklearn.cluster import KMeans
    HAVE_KMEANS = True
except Exception:
    HAVE_KMEANS = False
try:
    import alphashape, shapely.geometry as sgeom
    HAVE_ALPHA = True
except Exception:
    HAVE_ALPHA = False

# ----------------- Config -----------------
PLOTS_DIR = "plots"
RADIUS_DEG = 1.128

# Histogram settings
Z_BINS = 30
Z_RANGE = (0.0, 5.0)
RMAG_BINS = 30
RMAG_RANGE = (14.0, 26.0)

# Colors for set_id (zoomed panel)
SET_COLORS = {1: "C0", 2: "C2", 3: "C3", 4: "C4"}

# Full-footprint appearance
COLOR_INSIDE = "red"
COLOR_OUTSIDE = "darkgreen"
LABEL_DRA = 0.25
LABEL_DDEC = 0.15
LABEL_FONTSIZE = 10
HILITE_COLOR = "blue"
HILITE_FONTSIZE = 12

# Data paths for the footprint panel (load once)
MASTER_PATH = "../masterSDSS_V_SDSS_IV_DESI_unique_sources_QSO_ZWARN_RADEC_filters.csv"
INSIDE_PATH = "field_centers_inside.csv"
OUTSIDE_PATH = "field_centers_outside.csv"
HIGHZ_PATH = "redshift_gt_5_sources_Yang_master.csv"
PAQS_FITS = "PAQS_20250401_lsm.fits"                # optional
DEEP_FIELDS_CSV = "deep_fields_boundaries.csv"      # optional
LSM_VALUE = 0.00070126
N_REGIONS = 3
ALPHA_F = 0.05  # alpha-shape tightness

# ----------------- Helpers -----------------
def token_to_float(tok: str) -> float:
    """Decode RA/Dec token: supports plain float, 'm' for negative, 'p' for positive."""
    try:
        return float(tok)
    except ValueError:
        pass
    s = tok.replace("p", "+").replace("m", "-").replace("+", "")
    return float(s)

def parse_center_from_filename(fname: str):
    """
    Extract (RA_center, DEC_center) from:
      All_selected_sources_<RA>_<DEC>_insideeFEDs.csv
      All_selected_sources_<RA>_<DEC>_outsideeFEDs.csv
    """
    base = os.path.basename(fname)
    m = re.match(
        r"^All_selected_sources_([mp0-9\.\-]+)_([mp0-9\.\-]+)_(inside|outside)eFEDs\.csv$",
        base
    )
    if not m:
        raise ValueError(f"Filename does not match pattern: {base}")
    ra_tok, dec_tok, region = m.group(1), m.group(2), m.group(3)
    ra_c = token_to_float(ra_tok) % 360.0
    dec_c = token_to_float(dec_tok)
    return ra_c, dec_c, region

def as_num(s):
    return pd.to_numeric(s, errors="coerce")

def angsep_deg(ra1_deg, dec1_deg, ra2_deg, dec2_deg):
    """Great-circle separation (deg). Accepts vectorized inputs."""
    r1 = np.deg2rad(ra1_deg); d1 = np.deg2rad(dec1_deg)
    r2 = np.deg2rad(ra2_deg); d2 = np.deg2rad(dec2_deg)
    dra = (r2 - r1 + np.pi) % (2*np.pi) - np.pi
    ddec = d2 - d1
    a = np.sin(ddec/2.0)**2 + np.cos(d1)*np.cos(d2)*np.sin(dra/2.0)**2
    c = 2.0*np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return np.rad2deg(c)

def small_circle_poly(ra_c_deg, dec_c_deg, r_deg=RADIUS_DEG, n=512):
    """Return arrays of (RA, Dec) points on a small circle on the sphere."""
    ra_c = np.deg2rad(ra_c_deg); dec_c = np.deg2rad(dec_c_deg); r = np.deg2rad(r_deg)
    cx = np.cos(dec_c)*np.cos(ra_c); cy = np.cos(dec_c)*np.sin(ra_c); cz = np.sin(dec_c)
    cvec = np.array([cx, cy, cz])
    a = np.array([0.0, 0.0, 1.0])
    if np.allclose(cvec, a, atol=1e-8):
        a = np.array([1.0, 0.0, 0.0])
    u = np.cross(a, cvec); u /= np.linalg.norm(u)
    v = np.cross(cvec, u)
    ph = np.linspace(0, 2*np.pi, n, endpoint=True)
    pts = (np.cos(r)*cvec.reshape(3,1) +
           np.sin(r)*(u.reshape(3,1)*np.cos(ph) + v.reshape(3,1)*np.sin(ph)))
    xs, ys, zs = pts
    decs = np.arcsin(zs)
    ras = np.arctan2(ys, xs) % (2*np.pi)
    return np.rad2deg(ras), np.rad2deg(decs)

def ensure_plots_dir(path=PLOTS_DIR):
    os.makedirs(path, exist_ok=True)
    return path

def normalize_centers_df(df):
    """Standardize center files: ensure RA/DEC numeric; field ID column unified to 'FIELD_ID'."""
    df = df.copy()
    df["RA_center_deg"] = as_num(df["RA_center_deg"]) % 360.0
    df["DEC_center_deg"] = as_num(df["DEC_center_deg"])
    # accept either Field_ID or field_id
    fid_col = "Field_ID" if "Field_ID" in df.columns else ("field_id" if "field_id" in df.columns else None)
    if fid_col is None:
        raise ValueError("Center file is missing Field_ID/field_id column.")
    df["FIELD_ID"] = df[fid_col].astype(str)
    return df[["RA_center_deg","DEC_center_deg","FIELD_ID"]]

def nearest_field_id(ra_c, dec_c, centers_df, tol_deg=1e-3):
    """Find FIELD_ID in centers_df matching (ra_c, dec_c) within tolerance."""
    sep = angsep_deg(ra_c, dec_c, centers_df["RA_center_deg"].values, centers_df["DEC_center_deg"].values)
    idx = np.argmin(sep)
    if sep[idx] <= tol_deg:
        return centers_df.iloc[idx]["FIELD_ID"]
    return None

# ---------- One-time loads for footprint panel ----------
def load_master_density():
    if not os.path.exists(MASTER_PATH):
        warnings.warn(f"Missing master density file: {MASTER_PATH}")
        return None, None
    master = pd.read_csv(MASTER_PATH)
    if not {"RA","DEC"} <= set(master.columns):
        warnings.warn("Master file missing RA/DEC; skipping density.")
        return None, None
    ra = as_num(master["RA"]) % 360.0
    dec = as_num(master["DEC"])
    m = np.isfinite(ra) & np.isfinite(dec)
    return ra[m].values, dec[m].values

def load_highz():
    if not os.path.exists(HIGHZ_PATH):
        warnings.warn(f"Missing Yang high-z file: {HIGHZ_PATH}")
        return None, None
    hz = pd.read_csv(HIGHZ_PATH)
    if not {"ra","dec"} <= set(hz.columns):
        warnings.warn("Yang file missing ra/dec; skipping high-z points.")
        return None, None
    ra = as_num(hz["ra"]) % 360.0
    dec = as_num(hz["dec"])
    m = np.isfinite(ra) & np.isfinite(dec)
    return ra[m].values, dec[m].values

def load_centers():
    if not os.path.exists(INSIDE_PATH) or not os.path.exists(OUTSIDE_PATH):
        raise FileNotFoundError("Missing field_centers_inside.csv or field_centers_outside.csv")
    fin = normalize_centers_df(pd.read_csv(INSIDE_PATH))
    fin["region"] = "inside"
    fout = normalize_centers_df(pd.read_csv(OUTSIDE_PATH))
    fout["region"] = "outside"
    return fin.reset_index(drop=True), fout.reset_index(drop=True)

def compute_paqs_footprint():
    """Return list of polygons (Nx2 arrays) representing 4GPAQS islands (optional)."""
    polys = []
    if not (HAVE_FITS and HAVE_HULL and HAVE_KMEANS and os.path.exists(PAQS_FITS)):
        return polys
    try:
        lsm = fits.open(PAQS_FITS)[1].data
        mask = np.isclose(lsm["LSM"], LSM_VALUE)
        ra = (np.array(lsm["RA"][mask], dtype=float) % 360.0)
        dec = np.array(lsm["DEC"][mask], dtype=float)
        X = np.column_stack((ra, dec))
        labels = KMeans(n_clusters=N_REGIONS, random_state=42).fit_predict(X)
        for lab in range(N_REGIONS):
            pts = X[labels == lab]
            if len(pts) < 3:
                continue
            if HAVE_ALPHA:
                diag = np.hypot(pts[:,0].ptp(), pts[:,1].ptp())
                alpha = ALPHA_F * max(diag, 1.0)
                poly = alphashape.alphashape(pts, alpha)
                if hasattr(poly, "exterior"):
                    polys.append(np.asarray(poly.exterior.coords))
                    continue
            hull = ConvexHull(pts)
            polys.append(pts[hull.vertices])
    except Exception as e:
        warnings.warn(f"Footprint build failed: {e}")
    return polys

def load_deep_fields():
    if not os.path.exists(DEEP_FIELDS_CSV):
        return None
    try:
        df = pd.read_csv(DEEP_FIELDS_CSV)
        for c in ["Field","RA Lower (°)","RA Upper (°)","DEC Lower (°)","DEC Upper (°)"]:
            if c not in df.columns:
                return None
        return df
    except Exception:
        return None

# ---------- Footprint renderer ----------
def render_full_footprint(ax, ra_master, dec_master, hz_ra, hz_dec,
                          fin_df, fout_df, footprint_polys, deep_df,
                          highlight_ra, highlight_dec, highlight_id, highlight_region,sky_title):
    """Draw full footprint into ax; highlight current field."""
    # 1) Inverted greyscale density
    if ra_master is not None and dec_master is not None:
        hb = ax.hist2d(ra_master, dec_master, bins=[360, 180], cmap="gray_r")
        cbar = plt.colorbar(hb[3], ax=ax, fraction=0.047, pad=0.02)
        cbar.set_label("Master density (per bin)")

    # 2) high-z points
    if hz_ra is not None and hz_dec is not None:
        ax.plot(hz_ra, hz_dec, linestyle='none', marker='x', color='black',
                markersize=4, alpha=0.8, label="z>5 (Yang)")

    # 3) all field centers + circles + IDs
    def draw_set(df, color):
        for _, row in df.iterrows():
            ra_c = float(row["RA_center_deg"]); dec_c = float(row["DEC_center_deg"])
            ax.plot([ra_c], [dec_c], marker='o', color=color, markersize=3)
            cr_ra, cr_dec = small_circle_poly(ra_c, dec_c, RADIUS_DEG, n=256)
            ax.plot(cr_ra, cr_dec, color=color, linewidth=1.0, alpha=0.49)
            #ax.text(ra_c + LABEL_DRA, dec_c + LABEL_DDEC, row["FIELD_ID"],
            #        fontsize=LABEL_FONTSIZE, color=color)
    draw_set(fin_df, COLOR_INSIDE)
    draw_set(fout_df, COLOR_OUTSIDE)

    # 4) highlight current field
    if highlight_id is not None:
        ax.plot([highlight_ra], [highlight_dec], marker='o', color=HILITE_COLOR, markersize=5, zorder=5)
        hi_ra, hi_dec = small_circle_poly(highlight_ra, highlight_dec, RADIUS_DEG, n=512)
        ax.plot(hi_ra, hi_dec, color=HILITE_COLOR, linewidth=2.2, zorder=4)
        ax.text(highlight_ra + LABEL_DRA, highlight_dec + LABEL_DDEC,
                f"{highlight_id}", fontsize=HILITE_FONTSIZE, color=HILITE_COLOR,
                fontweight="bold", zorder=6)

    # 5) optional 4GPAQS polygons
    for i, poly in enumerate(footprint_polys):
        ax.plot(poly[:,0], poly[:,1], ls=":", lw=2, color="green" if i == 0 else "green",
                label="4GPAQS" if i == 0 else None)

    # 6) optional deep field rectangles
    if deep_df is not None and len(deep_df):
        color_cycle = plt.cm.tab10.colors
        for j, row in deep_df.iterrows():
            lo, hi = row['RA Lower (°)'], row['RA Upper (°)']
            dL, dH = row['DEC Lower (°)'], row['DEC Upper (°)']
            c = color_cycle[j % len(color_cycle)]
            ax.plot([lo, hi, hi, lo, lo], [dL, dL, dH, dH, dL], ls="--", lw=1.5, color=c)

    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.set_xlim(0, 360)
    ax.set_ylim(-30, 15)
    ax.set_title(sky_title)

# ----------------- Main plotting per file -----------------
def process_all_files():
    ensure_plots_dir()

    # Pre-load the shared datasets for the footprint panel once
    ra_master, dec_master = load_master_density()
    hz_ra, hz_dec = load_highz()
    fin, fout = load_centers()
    footprint_polys = compute_paqs_footprint()
    deep_df = load_deep_fields()

    files = sorted(glob.glob("All_selected_sources_*_insideeFEDs.csv") +
                   glob.glob("All_selected_sources_*_outsideeFEDs.csv"))
    if not files:
        print("No All_selected_sources_* files found.")
        return

    for csv_path in files:
        try:
            ra_c, dec_c, region = parse_center_from_filename(csv_path)
        except Exception as e:
            print(f"[SKIP] {csv_path}: {e}")
            continue

        # Read and coerce numeric
        df = pd.read_csv(csv_path)
        for col in ["RA", "Dec", "redshift", "rmag", "set_id"]:
            if col not in df.columns:
                raise ValueError(f"{csv_path} missing column '{col}'. Found: {list(df.columns)}")
        df["RA"] = as_num(df["RA"]) % 360.0
        df["Dec"] = as_num(df["Dec"])
        df["redshift"] = as_num(df["redshift"])
        df["rmag"] = as_num(df["rmag"])
        df["set_id"] = pd.to_numeric(df["set_id"], errors="coerce").astype("Int64")

        N_total = len(df)
        valid_z_mask = df["redshift"].notna() & np.isfinite(df["redshift"])
        N_valid_z = int(valid_z_mask.sum())
        N_invalid_z = int(N_total - N_valid_z)
        z_between_mask = valid_z_mask & (df["redshift"] > 1.0) & (df["redshift"] < 2.0)
        N_between = int(z_between_mask.sum())

        # Determine FIELD_ID from centers (match nearest within tol)
        if region == "inside":
            field_id = nearest_field_id(ra_c, dec_c, fin, tol_deg=1e-3)
        else:
            field_id = nearest_field_id(ra_c, dec_c, fout, tol_deg=1e-3)

        # ---------- Figure layout: 2 rows (top: footprint; bottom: 3 columns) ----------
        fig = plt.figure(figsize=(20, 14))
        # 2 rows x 3 cols; top row spans all 3 cols
        gs = fig.add_gridspec(nrows=2, ncols=3, height_ratios=[1.15, 1.0], hspace=0.30, wspace=0.28)

        ax_fp  = fig.add_subplot(gs[0, :])   # top panel: full footprint (spans all columns)
        ax_z   = fig.add_subplot(gs[1, 0])   # bottom-left  : redshift hist
        ax_r   = fig.add_subplot(gs[1, 1])   # bottom-mid   : rmag hist
        ax_sky = fig.add_subplot(gs[1, 2])   # bottom-right : zoomed sky
        sky_title = f"ID: {field_id}"
        sky_title += f"  | (center=({ra_c:.5f}, {dec_c:.5f}),radius = {RADIUS_DEG}°),"
        # === Full footprint (top) ===
        render_full_footprint(
            ax_fp,
            ra_master, dec_master, hz_ra, hz_dec,
            fin, fout,
            footprint_polys, deep_df,
            highlight_ra=ra_c, highlight_dec=dec_c,
            highlight_id=field_id, highlight_region=region,sky_title=sky_title
        )

        # === Histogram: redshift (bottom-left) ===
        zvals = df["redshift"].dropna().values
        ax_z.hist(zvals, bins=Z_BINS, range=Z_RANGE)
        ax_z.axvspan(1.0, 2.0, alpha=0.2)
        ax_z.set_xlabel("redshift (z)")
        ax_z.set_ylabel("Count")
        ax_z.set_title("Redshift distribution")
        txt = (f"Total: {N_total}\n"
               f"Valid z: {N_valid_z}\n"
               f"Invalid z: {N_invalid_z}\n"
               f"1 < z < 2: {N_between}")
        ax_z.text(0.98, 0.98, txt, ha="right", va="top",fontsize=16, transform=ax_z.transAxes)

        # === Histogram: rmag (bottom-middle) ===
        rvals = df["rmag"].dropna().values
        ax_r.hist(rvals, bins=RMAG_BINS, range=RMAG_RANGE)
        ax_r.set_xlabel("rmag")
        ax_r.set_ylabel("Count")
        ax_r.set_title("r-mag distribution")

        # === Zoomed sky panel (bottom-right) ===
        cr_ra, cr_dec = small_circle_poly(ra_c, dec_c, RADIUS_DEG, n=720)
        ax_sky.plot(cr_ra, cr_dec, lw=1.5)
        for sid, sub in df.groupby("set_id", dropna=True):
            color = SET_COLORS.get(int(sid), "k")
            ax_sky.plot(sub["RA"].values, sub["Dec"].values, linestyle="none",
                        marker="o", markersize=3.5, alpha=0.9, label=f"set_id={int(sid)}", color=color)
        ax_sky.plot([ra_c], [dec_c], marker="+", ms=10, mew=2)
        ra_min, ra_max = cr_ra.min(), cr_ra.max()
        dec_min, dec_max = cr_dec.min(), cr_dec.max()
        if (ra_max - ra_min) > 180:
            pad = 0.3
            ax_sky.set_xlim(ra_c - (RADIUS_DEG + pad), ra_c + (RADIUS_DEG + pad))
        else:
            pad = 0.3
            ax_sky.set_xlim(ra_min - pad, ra_max + pad)
        ax_sky.set_ylim(dec_min - 0.3, dec_max + 0.3)
        ax_sky.set_xlabel("RA [deg]")
        ax_sky.set_ylabel("Dec [deg]")
        #sky_title = f" (rad = {RADIUS_DEG}°), center=({ra_c:.5f}, {dec_c:.5f})"
        #if field_id is not None:
        #    sky_title += f"  |  ID: {field_id}"
        ax_sky.set_title("Source Distribution")
        ax_sky.legend(loc="best", frameon=True)

        # Overall title from filename
        fig.suptitle(os.path.basename(csv_path), y=0.995, fontsize=12)
        #plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.tight_layout()

        # Save
        ensure_plots_dir()
        outname = os.path.splitext(os.path.basename(csv_path))[0] + ".png"
        outpath = os.path.join(PLOTS_DIR, outname)
        fig.savefig(outpath, dpi=180)
        plt.close(fig)
        print(f"Saved {outpath}")

if __name__ == "__main__":
    process_all_files()
