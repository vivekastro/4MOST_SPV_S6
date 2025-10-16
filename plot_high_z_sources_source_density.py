# Update: invert the greyscale colormap and set Dec limits to [-30, 15].
# Re-generates the plot with the requested changes.
#
# Inputs (current folder):
#   - masterSDSS_V_SDSS_IV_DESI_unique_sources_QSO_ZWARN_RADEC_filters.csv
#   - field_centers_inside.csv
#   - field_centers_outside.csv
#   - redshift_gt_5_sources_Yang_master.csv
#
# Output:
#   - field_centers_highz_with_density_gray_inverted.png
#
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RADIUS_DEG = 1.128
MASTER_PATH = "masterSDSS_V_SDSS_IV_DESI_unique_sources_QSO_ZWARN_RADEC_filters.csv"
INSIDE_PATH = "field_centers_inside.csv"
OUTSIDE_PATH = "field_centers_outside.csv"
HIGHZ_PATH = "redshift_gt_5_sources_Yang_master.csv"

# eFEDS rectangle
EFEDS_RA_MIN, EFEDS_RA_MAX = 126.0, 146.0
EFEDS_DEC_MIN, EFEDS_DEC_MAX = -3.0, 6.0

missing = [p for p in [MASTER_PATH, INSIDE_PATH, OUTSIDE_PATH, HIGHZ_PATH] if not os.path.exists(p)]
if missing:
    print("Missing required file(s):")
    for p in missing:
        print(" -", p)
    print("\n➡️ Place the missing file(s) in the current folder and re-run this cell.")
else:
    # --- Load master for density ---
    master = pd.read_csv(MASTER_PATH)
    req_master = ["RA","DEC","redshift","r-mag","g-mag"]
    miss_m = [c for c in req_master if c not in master.columns]
    if miss_m:
        raise ValueError(f"{MASTER_PATH} missing columns: {miss_m}. Found: {master.columns.tolist()}")
    master["RA"]  = pd.to_numeric(master["RA"], errors="coerce")
    master["DEC"] = pd.to_numeric(master["DEC"], errors="coerce")
    master = master.dropna(subset=["RA","DEC"]).reset_index(drop=True)
    m_ra  = (master["RA"].values % 360.0)
    m_dec = master["DEC"].values

    # --- Load field centres ---
    fin = pd.read_csv(INSIDE_PATH)
    fout = pd.read_csv(OUTSIDE_PATH)
    for pth, df in [(INSIDE_PATH, fin), (OUTSIDE_PATH, fout)]:
        miss = [c for c in ["RA_center_deg","DEC_center_deg"] if c not in df.columns]
        if miss:
            raise ValueError(f"{pth} missing columns: {miss}. Found: {df.columns.tolist()}")
    centers = pd.concat([fin[["RA_center_deg","DEC_center_deg"]],
                         fout[["RA_center_deg","DEC_center_deg"]]], ignore_index=True)

    # --- Load high-z ---
    hz = pd.read_csv(HIGHZ_PATH)
    req_hz = ["desi","name","ra","dec","z","zmag","zmagerr"]
    miss_h = [c for c in req_hz if c not in hz.columns]
    if miss_h:
        raise ValueError(f"{HIGHZ_PATH} missing columns: {miss_h}. Found: {hz.columns.tolist()}")
    hz["ra"]  = pd.to_numeric(hz["ra"], errors="coerce")
    hz["dec"] = pd.to_numeric(hz["dec"], errors="coerce")
    hz = hz.dropna(subset=["ra","dec"]).reset_index(drop=True)
    hz_ra  = (hz["ra"].values % 360.0)
    hz_dec = hz["dec"].values

    # --- helpers ---
    def small_circle_points(ra_c_deg, dec_c_deg, radius_deg, n=256):
        ra_c = np.deg2rad(ra_c_deg); dec_c = np.deg2rad(dec_c_deg); r = np.deg2rad(radius_deg)
        cx = np.cos(dec_c)*np.cos(ra_c); cy = np.cos(dec_c)*np.sin(ra_c); cz = np.sin(dec_c)
        cvec = np.array([cx, cy, cz])
        a = np.array([0.0, 0.0, 1.0])
        if np.allclose(cvec, a, atol=1e-8):
            a = np.array([1.0, 0.0, 0.0])
        u = np.cross(a, cvec); u /= np.linalg.norm(u); v = np.cross(cvec, u)
        phis = np.linspace(0, 2*np.pi, n, endpoint=True)
        pts = (np.cos(r)*cvec.reshape(3,1) + np.sin(r)*(u.reshape(3,1)*np.cos(phis) + v.reshape(3,1)*np.sin(phis)))
        xs, ys, zs = pts; decs = np.arcsin(zs); ras = np.arctan2(ys, xs) % (2*np.pi)
        return np.rad2deg(ras), np.rad2deg(decs)

    def wrap_ra360(x): return np.mod(np.asarray(x), 360.0)

    def plot_circle_wrapped(ax, ra_arr_deg, dec_arr_deg, **kwargs):
        ra_wrapped = wrap_ra360(ra_arr_deg)
        dec_vals = np.asarray(dec_arr_deg)
        dr = np.abs(np.diff(ra_wrapped))
        breaks = np.where(dr > 180.0)[0] + 1
        segments = np.split(np.arange(len(ra_wrapped)), breaks)
        for seg in segments:
            ax.plot(ra_wrapped[seg], dec_vals[seg], **kwargs)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 8))

    # 1) Inverted greyscale density
    hb = ax.hist2d(m_ra, m_dec, bins=[360, 180], cmap="gray_r")
    cbar = plt.colorbar(hb[3], ax=ax)
    cbar.set_label("Master source density (per bin)")

    # 2) High‑z points (default color)
    ax.plot(hz_ra, hz_dec, linestyle='none', marker='.', markersize=3, alpha=0.9)

    # 3) Field centres + circles (default colors)
    for _, row in centers.iterrows():
        ra_c = float(row["RA_center_deg"]); dec_c = float(row["DEC_center_deg"])
        ax.plot([wrap_ra360([ra_c])[0]], [dec_c], marker='o', markersize=3)
        cr_ra, cr_dec = small_circle_points(ra_c, dec_c, RADIUS_DEG, n=256)
        plot_circle_wrapped(ax, cr_ra, cr_dec, linewidth=1.2)

    # 4) eFEDS rectangle
    rect_ra = np.array([EFEDS_RA_MIN, EFEDS_RA_MAX, EFEDS_RA_MAX, EFEDS_RA_MIN, EFEDS_RA_MIN])
    rect_dec = np.array([EFEDS_DEC_MIN, EFEDS_DEC_MIN, EFEDS_DEC_MAX, EFEDS_DEC_MAX, EFEDS_DEC_MIN])
    ax.plot(wrap_ra360(rect_ra), rect_dec, linestyle='--', linewidth=1.5)

    # Axes
    ax.set_xlabel("Right Ascension [deg]")
    ax.set_ylabel("Declination [deg]")
    ax.set_title("Field centres (r = 1.128°), high‑z sources, eFEDS, and inverted greyscale density")
    ax.set_xlim(0.0, 360.0)
    ax.set_ylim(-30.0, 15.0)   # requested Dec limits

    out_plot = "field_centers_highz_with_density_gray_inverted.png"
    plt.tight_layout()
    plt.savefig(out_plot, dpi=200)
    plt.show()

    print(f"✅ Plot saved: {out_plot}")

