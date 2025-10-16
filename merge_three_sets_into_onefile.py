#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, glob
import pandas as pd

SUFFIX = "outsideeFEDs.csv"   # keep as-is; change if your suffix differs
RE_TAG = re.compile(r"^(First|Second|Third)_selected_sources_(.+?)_outsideeFEDs\.csv$")

def find_tags():
    """Return sorted list of tags that have all First/Second/Third files present."""
    seen = {}
    for fn in glob.glob(f"*_{SUFFIX}"):
        m = RE_TAG.match(os.path.basename(fn))
        if not m:
            continue
        kind, tag = m.group(1), m.group(2)
        seen.setdefault(tag, set()).add(kind)
    return sorted([t for t, kinds in seen.items() if {"First","Second","Third"} <= kinds])

def read_std(path):
    """Read a CSV and standardize columns to RA, Dec, redshift, rmag (numeric)."""
    df = pd.read_csv(path)
    # normalize names for matching
    norm = {c.lower().replace("-", "").replace("_", ""): c for c in df.columns}
    def pick(*keys):
        for k in keys:
            k2 = k.lower().replace("-", "").replace("_", "")
            if k2 in norm:
                return norm[k2]
        return None
    ra   = pick("RA")
    dec  = pick("DEC", "Dec")
    z    = pick("redshift", "z")
    rmag = pick("rmag", "r-mag", "r")
    need = {"RA": ra, "Dec": dec, "redshift": z, "rmag": rmag}
    miss = [k for k,v in need.items() if v is None]
    if miss:
        raise ValueError(f"{os.path.basename(path)} missing columns: {miss}; has {list(df.columns)}")
    out = pd.DataFrame({
        "RA":      pd.to_numeric(df[ra], errors="coerce"),
        "Dec":     pd.to_numeric(df[dec], errors="coerce"),
        "redshift":pd.to_numeric(df[z],   errors="coerce"),
        "rmag":    pd.to_numeric(df[rmag],errors="coerce"),
    })
    return out

def process_tag(tag):
    files = {
        1: f"First_selected_sources_{tag}_{SUFFIX}",
        2: f"Second_selected_sources_{tag}_{SUFFIX}",
        3: f"Third_selected_sources_{tag}_{SUFFIX}",
    }
    for p in files.values():
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    d1 = read_std(files[1]); d1["set_id"] = 1
    d2 = read_std(files[2]); d2["set_id"] = 2
    d3 = read_std(files[3]); d3["set_id"] = 3

    all_df = pd.concat([d1, d2, d3], ignore_index=True)
    # (Optional) drop rows lacking any main values:
    # all_df = all_df.dropna(subset=["RA","Dec","redshift","rmag"])

    out = f"All_selected_sources_{tag}_{SUFFIX}"
    all_df.to_csv(out, index=False)
    print(f"Wrote {out} | rows={len(all_df)} (S1={len(d1)}, S2={len(d2)}, S3={len(d3)})")

def main():
    tags = find_tags()
    if not tags:
        print("No complete (First/Second/Third) tag sets found.")
        return
    print(f"Found {len(tags)} tag(s): {', '.join(tags)}")
    for tag in tags:
        try:
            process_tag(tag)
        except Exception as e:
            print(f"[SKIP] {tag}: {e}")

if __name__ == "__main__":
    main()

