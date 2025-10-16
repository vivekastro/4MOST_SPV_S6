#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import  os
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# =============== CONFIG ===============
INSIDE_PATH = "field_centers_inside.csv"
OUTSIDE_PATH = "field_centers_outside.csv"

SITE_NAME   = "Paranal, Chile"
LAT_DEG     = -24.6270
LON_DEG     = -70.4040
ELEV_M      = 2635.0

ALT_MIN_DEG = 30.0       # minimum altitude considered "visible"
NSTEP_MIN   = 30         # sampling step (minutes) during night (Astropy path)
YEAR        = datetime.now().year  # e.g., 2025

OUT_PNG     = "paranal_year_visibility.png"

# Appearance (global)
mpl.rcParams["axes.linewidth"]   = 3
mpl.rcParams["xtick.major.width"] = 3
mpl.rcParams["ytick.major.width"] = 3
mpl.rcParams["xtick.labelsize"]   = 14
mpl.rcParams["ytick.labelsize"]   = 14
mpl.rcParams["axes.titlesize"]    = 16
mpl.rcParams["axes.labelsize"]    = 15

# Alternate row shading
BAND_ALPHA = 0.22
BAND_COLOR = "#f0f4ff"   # light bluish; tweak as desired

# =============== HELPERS ===============
def as_num(x):
    return pd.to_numeric(x, errors="coerce")

def normalize_centers_df(df):
    """Return DataFrame with FIELD_ID, RA_center_deg, DEC_center_deg."""
    df = df.copy()
    if "RA_center_deg" not in df or "DEC_center_deg" not in df:
        raise ValueError("Center CSV must have RA_center_deg and DEC_center_deg columns.")
    df["RA_center_deg"]  = as_num(df["RA_center_deg"]) % 360.0
    df["DEC_center_deg"] = as_num(df["DEC_center_deg"])
    fid = "Field_ID" if "Field_ID" in df.columns else ("field_id" if "field_id" in df.columns else None)
    if fid is None:
        df["FIELD_ID"] = [f"ID_{i+1}" for i in range(len(df))]
    else:
        df["FIELD_ID"] = df[fid].astype(str)
    return df[["FIELD_ID", "RA_center_deg", "DEC_center_deg"]].dropna()

def build_date_grid(year):
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    end   = datetime(year+1, 1, 1, tzinfo=timezone.utc)
    dates = []
    t = start
    while t < end:
        dates.append(t)
        t += timedelta(days=1)
    return np.array(dates, dtype="datetime64[ns]")

def compute_visible_matrix_astropy(fields, dates):
    """Accurate path: visible if target reaches ALT_MIN_DEG during astronomical night (Sun alt < -18°)."""
    from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun
    from astropy.time import Time
    import astropy.units as u

    loc = EarthLocation.from_geodetic(lon=LON_DEG*u.deg, lat=LAT_DEG*u.deg, height=ELEV_M*u.m)
    nF, nD = len(fields), len(dates)
    vis = np.zeros((nF, nD), dtype=bool)
    sc = SkyCoord(ra=fields["RA_center_deg"].values*u.deg,
                  dec=fields["DEC_center_deg"].values*u.deg)
    step = timedelta(minutes=NSTEP_MIN)

    for j, d0 in enumerate(dates):
        dpy = pd.Timestamp(d0).to_pydatetime()
        # Crude local-midnight UTC; then sample a wide window and filter to astronomical night
        utc_offset_hours = LON_DEG / 15.0  # west longitudes negative
        midnight_local_utc = dpy + timedelta(hours=-utc_offset_hours)
        t0 = midnight_local_utc - timedelta(hours=8)
        times = [t0 + i*step for i in range(int(16*60/NSTEP_MIN) + 1)]
        at = Time(times)

        sun_alt = get_sun(at).transform_to(AltAz(obstime=at, location=loc)).alt.deg
        night_mask = sun_alt < -18.0
        if not np.any(night_mask):
            continue

        at_night = at[night_mask]
        aa = sc[:, None].transform_to(AltAz(obstime=at_night[None, :], location=loc))
        alt = aa.alt.deg  # (nF, ntimes)
        vis[:, j] = np.any(alt >= ALT_MIN_DEG, axis=1)
    return vis

def compute_visible_matrix_approx(fields, dates):
    """
    Fallback: seasonal RA window (±~60 days around opposition).
    Ignores altitude/dec constraints but useful when Astropy isn't available.
    """
    nF, nD = len(fields), len(dates)
    vis = np.zeros((nF, nD), dtype=bool)
    days = np.array([(pd.Timestamp(d).dayofyear) for d in dates])

    for i, ra in enumerate(fields["RA_center_deg"].values):
        ra_h = (ra / 15.0) % 24.0
        opp_day = (365.0 * ((ra_h - 12.0) % 24.0) / 24.0)  # opposition day
        dmin = (opp_day - 60.0) % 365.0
        dmax = (opp_day + 60.0) % 365.0
        if dmin < dmax:
            mask = (days >= dmin) & (days <= dmax)
        else:
            mask = (days >= dmin) | (days <= dmax)
        vis[i, :] = mask
    return vis

# =============== MAIN ===============
def main():
    # Read centers
    if not os.path.exists(INSIDE_PATH) or not os.path.exists(OUTSIDE_PATH):
        raise FileNotFoundError("Missing field_centers_inside.csv or field_centers_outside.csv")

    fin  = normalize_centers_df(pd.read_csv(INSIDE_PATH))
    fout = normalize_centers_df(pd.read_csv(OUTSIDE_PATH))
    fin["region"]  = "inside"
    fout["region"] = "outside"
    fields = pd.concat([fin, fout], ignore_index=True).reset_index(drop=True)

    # Sort for tidy plotting
    fields["region_order"] = fields["region"].map({"inside": 0, "outside": 1})
    fields = fields.sort_values(["region_order", "FIELD_ID"]).reset_index(drop=True)
    fields.drop(columns=["region_order"], inplace=True)

    # Date grid
    dates = build_date_grid(YEAR)

    # Visibility
    try:
        import astropy  # probe availability
        vis = compute_visible_matrix_astropy(fields, dates)
        method = "Astropy (astronomical night, alt≥{}°)".format(int(ALT_MIN_DEG))
    except Exception:
        vis = compute_visible_matrix_approx(fields, dates)
        method = "Approximate seasonal window (no altitude check)"

    # Plot
    fig, ax = plt.subplots(figsize=(15, max(6, 0.28*len(fields)+3)), constrained_layout=True)

    im = ax.imshow(vis, aspect="auto", interpolation="nearest",
                   origin="lower", cmap="Greens")

    # Alternate row shading (draw above heatmap so it remains visible)
    nrows = vis.shape[0]
    for y in range(nrows):
        if y % 2 == 1:  # shade every other row
            ax.axhspan(y - 0.5, y + 0.5, facecolor=BAND_COLOR, alpha=BAND_ALPHA, zorder=0.5)

    # Y ticks: Field IDs with region suffix
    ylabels = [f"{row.FIELD_ID} ({row.region[0]})" for row in fields.itertuples()]
    ax.set_yticks(np.arange(len(fields)))
    ax.set_yticklabels(ylabels)

    # X ticks: months
    month_starts = [datetime(YEAR, m, 1, tzinfo=timezone.utc) for m in range(1, 13)]
    month_pos = [np.searchsorted(dates, np.datetime64(ms)) for ms in month_starts]
    ax.set_xticks(month_pos)
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])

    ax.set_xlabel(f"UTC date in {YEAR}")
    ax.set_ylabel("Field ID  (i=inside, o=outside)")
    ax.set_title(f"{SITE_NAME}: Year Visibility — {method}")

    # Thicker spines + tick marks already set via rcParams, but ensure here too
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.tick_params(width=3)

    # Save
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Saved {OUT_PNG}")
    print(f"Fields: {len(fields)} | Engine: {method}")

if __name__ == "__main__":
    main()

