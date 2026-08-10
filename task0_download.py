"""Download Muenster senseBox:bike overtaking data from openSenseMap.

Set DOWNLOAD_START / DOWNLOAD_END, then run. One CSV per channel is written to
input/, named after the date range (overtake distance or maneuver)
since rows are deduplicated (the API repeats rows).
"""
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# ======= what to download ======
DOWNLOAD_START = datetime(2024, 7, 1, tzinfo=timezone.utc)  # overtaking channels exist since 2024-07
DOWNLOAD_END = datetime(2026, 7, 1, tzinfo=timezone.utc)

BBOX = "7.45,51.82,7.80,52.08"                              # Muenster + margin
PHENOMENA = ["Overtaking Distance", "Overtaking Manoeuvre"]
COLUMNS = "lat,lon,boxName,boxId,unit,value,createdAt"
# ================================


def fetch(phenomenon, month_start, month_end):
    """One API request for one month. Retries up to 5 times, then aborts the run."""
    url = (
        "https://api.opensensemap.org/boxes/data?"
        + urllib.parse.urlencode({
            "phenomenon": phenomenon,
            "bbox": BBOX,
            "from-date": month_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to-date": month_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "columns": COLUMNS,
            "format": "csv",
        })
    )
    for attempt in range(1, 6):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                df = pd.read_csv(r)
            print(f"  [{month_start:%Y-%m-%d} .. {month_end:%Y-%m-%d}] {len(df):>7} rows")
            return df
        except Exception as e:
            print(f"  [{month_start:%Y-%m-%d} .. {month_end:%Y-%m-%d}] attempt {attempt} failed: {e}")
            time.sleep(15 * attempt)
    raise RuntimeError(
        f"could not fetch {phenomenon} [{month_start:%Y-%m-%d} .. {month_end:%Y-%m-%d}] "
        "after 5 attempts; rerun when the API is reachable")


def month_windows(start, end):
    """Yield one (month_start, month_end) pair per calendar month in start..end."""
    month_start = start
    while month_start < end:
        next_month = (month_start.replace(day=1) + timedelta(days=32)).replace(day=1)
        yield month_start, min(next_month, end)
        month_start = next_month


if __name__ == "__main__":
    for phenomenon in PHENOMENA:
        slug = phenomenon.lower().replace(" ", "_")
        out_path = Path(f"input/muenster_{slug}_{DOWNLOAD_START:%Y-%m}_{DOWNLOAD_END:%Y-%m}.csv")
        print(f"\n=== {phenomenon} ({DOWNLOAD_START:%Y-%m-%d} .. {DOWNLOAD_END:%Y-%m-%d}) ===")

        monthly = [fetch(phenomenon, month_start, month_end)
                   for month_start, month_end in month_windows(DOWNLOAD_START, DOWNLOAD_END)]

        df = pd.concat([m for m in monthly if len(m)], ignore_index=True)
        n_raw = len(df)
        df = df.drop_duplicates().sort_values(["boxId", "createdAt"]).reset_index(drop=True)
        out_path.parent.mkdir(exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"{phenomenon}: {n_raw} raw -> {len(df)} rows, "
              f"{df['boxId'].nunique()} boxes -> {out_path}")
    print("\nDONE")
