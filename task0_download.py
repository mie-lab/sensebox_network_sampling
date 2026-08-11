"""Download the Muenster senseBox:bike overtaking data from openSenseMap.

Pulls two channels: the measured overtake distance, and the classifier's
overtaking-manoeuvre probability. Set DOWNLOAD_START / DOWNLOAD_END, then run.
The API repeats rows, so they are deduplicated here.

Writes one CSV per channel to input/, named after the date range.
"""
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# ======= what to download ======
DOWNLOAD_START = datetime(2024, 8, 1, tzinfo=timezone.utc)
DOWNLOAD_END = datetime(2026, 8, 1, tzinfo=timezone.utc)

BBOX = "7.45,51.82,7.80,52.08"                              # Muenster + some margin
PHENOMENA = ["Overtaking Distance", "Overtaking Manoeuvre"]
COLUMNS = "lat,lon,boxName,boxId,unit,value,createdAt"
# ================================

MAX_ATTEMPTS = 5    # the API times out, so a failed month is retried before aborting


def _fetch_month(phenomenon, month_start, month_end):
    """One API request for one month, retried MAX_ATTEMPTS times, then aborts the run."""
    month = f"{month_start:%Y-%m}"
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
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                df = pd.read_csv(r)
            print(f"  {month}  {len(df):>7} rows")
            return df
        except Exception as e:
            print(f"  {month}  attempt {attempt}/{MAX_ATTEMPTS} failed: {e}")
            time.sleep(15 * attempt)
    raise RuntimeError(f"{phenomenon} {month}: unreachable after {MAX_ATTEMPTS} attempts")


def _month_windows(start, end):
    """Yield one (month_start, month_end) pair per calendar month in start..end."""
    month_start = start
    while month_start < end:
        next_month = (month_start.replace(day=1) + timedelta(days=32)).replace(day=1)
        yield month_start, min(next_month, end)
        month_start = next_month


def main():
    for phenomenon in PHENOMENA:
        slug = phenomenon.lower().replace(" ", "_")
        out_path = Path(f"input/muenster_{slug}_{DOWNLOAD_START:%Y-%m}_{DOWNLOAD_END:%Y-%m}.csv")
        print(f"\n=== {phenomenon} {DOWNLOAD_START:%Y-%m-%d}-{DOWNLOAD_END:%Y-%m-%d} ===")

        monthly = [_fetch_month(phenomenon, month_start, month_end)
                   for month_start, month_end in _month_windows(DOWNLOAD_START, DOWNLOAD_END)]

        monthly = [m for m in monthly if len(m)]
        if not monthly:
            raise RuntimeError(f"{phenomenon}: no rows in the whole range")
        df = pd.concat(monthly, ignore_index=True)
        n_raw = len(df)
        df = df.drop_duplicates().sort_values(["boxId", "createdAt"]).reset_index(drop=True)
        out_path.parent.mkdir(exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"{n_raw} rows, {len(df)} after deduplication, {df['boxId'].nunique()} boxes -> {out_path}")
    print("\nDONE")


if __name__ == "__main__":
    main()
