#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import numpy as np
import pandas as pd

RADIUS_DEG = 1.128
INPUT_YANG = "redshift_gt_5_sources_Yang_master.csv"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def token_to_float(tok: str) -> float:
    """
    Convert RA/Dec token from filename to float.
    Supports 'm' for minus in Dec tokens (e.g. 'm2.03856' -> -2.03856).
    """
    # Try direct float first
    try:
        return float(tok)
    except ValueError:
        pass
    s = tok
    # replace 'm' with '-' *only* if it is the sign prefix
    if s.startswith("m"):
        s = "-" + s[1:]
    # strip any leading 'p'
    if s.startswith("p"):
        s = s[1:]
    return float(s)

def parse_center_from_filename(fname: str):
    """
    Extract (RA_center, DEC_center) from:
      All_selected_sources_<RA>_<DEC>_insideeFEDs.csv
      All_selected_sources_<RA>_<DEC>_outsideeFEDs.csv
    Accepts digits, '.', optional leading 'm' for negative Dec.
    """
    base = os.path.basename(fname)
    m = re.match(r"^All_selected_sources_([0-9\.]+)_([mp]?[0-9\.]+)_(?:inside|outside)eFEDs\.csv$", base)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {base}")
    ra_tok, dec_tok = m.group(1), m.group(2)
    ra_c = token_to_float(ra_tok) % 360.0
    dec_c = token_to_float(dec_tok)
    return ra_c, dec_c

def angsep_deg(ra1_deg, dec1_deg, ra2_deg, dec2_deg):
    """Great-circle separation (deg). Works with arrays (broadcasting)."""
    r1 = np.deg2rad(ra1_deg); d1 = np.deg2rad(dec1_deg)
    r2 = np.deg2rad(ra2_deg); d2 = np.deg2rad(dec2_deg)
    dra = (r2 - r1 + np.pi) % (2*np.pi) - np.pi
    ddec = d2 - d1
    a = np.sin(ddec/2.0)**2 + np.cos(d1)*np.cos(d2)*np.sin(dra/2.0)**2
    c = 2.0*np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return np.rad2deg(c)

def load_yang(path=INPUT_YANG):
    df = pd.read_csv(path)
    expected = {"desi","name","ra","dec","z","zmag","zmagerr"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    # keep only z > 5 and finite
    df = df.copy()
    df["ra"] = pd.to_numeric(df["ra"], errors="coerce") % 360.0
    df["dec"] = pd.to_numeric(df["dec"], errors="coerce")
    df["z"] = pd.to_numeric(df["z"], errors="coerce")
    df["zmag"] = pd.to_numeric(df["zmag"], errors="coerce")
    df = df[(df["z"] > 5) & np.isfinite(df["z"]) & np.isfinite(df["ra"]) & np.isfinite(df["dec"])]
    return df.reset_index(drop=True)

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    yang = load_yang(INPUT_YANG)

    files = sorted(glob.glob("All_selected_sources_*_insideeFEDs.csv") +
                   glob.glob("All_selected_sources_*_outsideeFEDs.csv"))
    if not files:
        print("No All_selected_sources_* files found.")
        return

    for path in files:
        base = os.path.basename(path)
        try:
            ra_c, dec_c = parse_center_from_filename(path)
        except Exception as e:
            print(f"[SKIP] {base}: {e}")
            continue

        # Yang sources within the field circle (relative to center)
        dsep = angsep_deg(ra_c, dec_c, yang["ra"].values, yang["dec"].values)
        in_circle = dsep <= RADIUS_DEG
        hz_df = yang.loc[in_circle].copy()
        n_hz = len(hz_df)

        print(f"{base}: high-z (z>5) in circle = {n_hz}")

        # Load the All_selected_sources CSV
        df = pd.read_csv(path)
        needed = {"RA","Dec","redshift","rmag","set_id"}
        missing = needed - set(df.columns)
        if missing:
            print(f"  [WARN] Missing columns {missing} in {base}; skipping substitution.")
            # Still write a copy with Hz prefix unchanged, for bookkeeping
            out_copy = os.path.join(os.path.dirname(path), "Hz" + base)
            df.to_csv(out_copy, index=False)
            continue

        # Coerce types
        df["RA"] = pd.to_numeric(df["RA"], errors="coerce") % 360.0
        df["Dec"] = pd.to_numeric(df["Dec"], errors="coerce")
        df["redshift"] = pd.to_numeric(df["redshift"], errors="coerce")
        df["rmag"] = pd.to_numeric(df["rmag"], errors="coerce")
        # Ensure integer-ish set_id but keep NaNs if any
        df["set_id"] = pd.to_numeric(df["set_id"], errors="coerce").astype("Int64")

        if n_hz > 0:
            # Sort high-z by proximity to center (closest first)
            hz_df["dist"] = angsep_deg(ra_c, dec_c, hz_df["ra"].values, hz_df["dec"].values)
            hz_df = hz_df.sort_values("dist").reset_index(drop=True)

            # Candidates to replace: rows with set_id == 3
            idx_set3 = df.index[df["set_id"] == 3].tolist()
            k = min(n_hz, len(idx_set3))

            if k == 0:
                print(f"  [INFO] No set_id=3 rows to substitute; writing unchanged with Hz prefix.")
                out_path = os.path.join(os.path.dirname(path), "Hz" + base)
                df.to_csv(out_path, index=False)
                continue

            # Prepare replacement rows (take k closest)
            repl = hz_df.iloc[:k]
            new_rows = pd.DataFrame({
                "RA": repl["ra"].values,
                "Dec": repl["dec"].values,
                "redshift": repl["z"].values,
                "rmag": repl["zmag"].values,
                "set_id": np.full(k, 4, dtype=int)
            })

            # Replace: drop k of the set_id=3 and append k new rows (keeping row count constant)
            drop_indices = idx_set3[:k]
            df_after = df.drop(index=drop_indices).reset_index(drop=True)
            # Append new rows
            df_after = pd.concat([df_after, new_rows], ignore_index=True)

            print(f"  Substituted {k} set_id=3 rows with high-z (set_id=4).")

            # Write output with Hz prefix
            out_path = os.path.join(os.path.dirname(path), "Hz" + base)
            df_after.to_csv(out_path, index=False)
        else:
            # No high-z → just write a copy with Hz prefix (unchanged)
            out_path = os.path.join(os.path.dirname(path), "Hz" + base)
            df.to_csv(out_path, index=False)

        print(f"  → wrote {out_path}")

if __name__ == "__main__":
    main()

