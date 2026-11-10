#!/usr/bin/env python3
"""Prepare the public FDNY EMS incident data for the annotated research notebook.

Data source: NYC Open Data, EMS Incident Dispatch Data (76xm-jjuj).
The research notebook was originally built on:
  - historical incidents: [2024-05-01, 2024-11-18)
  - unseen prediction week: [2024-11-18, 2024-11-25)

This script downloads source records only. It does not clean the data, construct
model features, or download future response-time outcomes. No API token is
required, but SOCRATA_APP_TOKEN may help avoid anonymous rate limiting.

Note: The live dataset can change. The notebook's saved counts (913,796 and
30,197) are reference counts, not an assertion about today's source records.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


SOURCE_URL = "https://data.cityofnewyork.us/resource/76xm-jjuj.csv"
SOURCE_PAGE = (
    "https://data.cityofnewyork.us/Public-Safety/"
    "EMS-Incident-Dispatch-Data/76xm-jjuj"
)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "_cache"

HISTORY_START = datetime(2024, 5, 1)
FUTURE_START = datetime(2024, 11, 18)
FUTURE_END = datetime(2024, 11, 25)

PAGE_SIZE = 50_000
RETRIES = 6

HISTORY_COLUMNS = [
    "incident_id",
    "incident_datetime",
    "initial_call_type",
    "initial_severity_level_code",
    "final_call_type",
    "final_severity_level_code",
    "first_assignment_datetime",
    "valid_dispatch_rspns_time_indc",
    "dispatch_response_seconds_qy",
    "first_activation_datetime",
    "first_on_scene_datetime",
    "valid_incident_rspns_time_indc",
    "incident_response_seconds_qy",
    "incident_travel_tm_seconds_qy",
    "incident_close_datetime",
    "held_indicator",
    "incident_disposition_code",
    "borough",
    "incident_dispatch_area",
    "zipcode",
    "policeprecinct",
    "citycouncildistrict",
    "communitydistrict",
    "communityschooldistrict",
    "congressionaldistrict",
    "reopen_indicator",
    "special_event_indicator",
    "standby_indicator",
    "transfer_indicator",
    "first_to_hosp_datetime",
    "first_hosp_arrival_datetime",
]

FUTURE_COLUMNS = [
    "incident_id",
    "incident_datetime",
    "initial_call_type",
    "initial_severity_level_code",
    "borough",
    "incident_dispatch_area",
    "zipcode",
    "policeprecinct",
    "citycouncildistrict",
    "communitydistrict",
    "communityschooldistrict",
    "congressionaldistrict",
]


def source_datetime(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%S")


def weekly_windows(start: datetime, end: datetime):
    cursor = start
    while cursor < end:
        next_cursor = min(cursor + timedelta(days=7), end)
        yield cursor, next_cursor
        cursor = next_cursor


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(actual: list[str], expected: list[str], name: str):
    if actual != expected:
        raise RuntimeError(
            f"Unexpected columns in {name}.\n"
            f"Expected ({len(expected)}): {expected}\n"
            f"Actual   ({len(actual)}): {actual}"
        )


def fetch_one_page(
    session: requests.Session,
    start: datetime,
    end: datetime,
    columns: list[str],
    offset: int,
) -> pd.DataFrame:
    where = (
        f"incident_datetime >= '{source_datetime(start)}' "
        f"AND incident_datetime < '{source_datetime(end)}'"
    )
    params = {
        "$select": ",".join(columns),
        "$where": where,
        "$order": "incident_datetime ASC, incident_id ASC",
        "$limit": PAGE_SIZE,
        "$offset": offset,
    }

    for attempt in range(RETRIES):
        try:
            response = session.get(SOURCE_URL, params=params, timeout=180)
            if response.status_code == 429 or 500 <= response.status_code < 600:
                response.raise_for_status()
            if response.status_code >= 400:
                # Do not retry query/schema errors. The source may have changed.
                raise RuntimeError(
                    f"NYC Open Data returned HTTP {response.status_code}: "
                    f"{response.text[:450]}"
                )

            if not response.text.strip():
                return pd.DataFrame(columns=columns)

            page = pd.read_csv(StringIO(response.text), low_memory=False)
            require_columns(list(page.columns), columns, "downloaded page")
            return page

        # Includes connection resets *during response streaming* (raised by
        # requests as ChunkedEncodingError), not just connection establishment.
        except requests.RequestException as exc:
            if attempt == RETRIES - 1:
                raise RuntimeError(
                    f"Could not download the {start.date()}–{end.date()} "
                    f"window after {RETRIES} attempts"
                ) from exc
            pause = min(2 ** attempt * 2, 40)
            print(f"    Request interrupted ({exc}); retrying in {pause}s...")
            time.sleep(pause)

    raise AssertionError("Retry loop ended unexpectedly")


def cache_path(cohort: str, start: datetime, end: datetime) -> Path:
    return CACHE_DIR / (
        f"{cohort}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
    )


def fetch_week(
    session: requests.Session,
    cohort: str,
    start: datetime,
    end: datetime,
    columns: list[str],
    *,
    refresh: bool = False,
) -> Path:
    path = cache_path(cohort, start, end)
    if path.exists() and not refresh:
        require_columns(
            list(pd.read_csv(path, nrows=0).columns), columns, str(path)
        )
        print(f"  Using cached {path.name}")
        return path

    pages = []
    offset = 0
    while True:
        print(
            f"  Download {cohort} {start.date()} to {end.date()} "
            f"(offset {offset:,})"
        )
        page = fetch_one_page(session, start, end, columns, offset)
        if page.empty:
            break
        pages.append(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    if not pages:
        raise RuntimeError(
            f"No records returned for {cohort} {start.date()}–{end.date()}. "
            "Check the API, time window, and dataset availability."
        )

    merged = pd.concat(pages, ignore_index=True)
    incident_times = pd.to_datetime(merged["incident_datetime"], errors="coerce")
    if incident_times.isna().any():
        raise RuntimeError("Source returned unparseable incident timestamps.")
    if not ((incident_times >= start) & (incident_times < end)).all():
        raise RuntimeError("Downloaded records fall outside their requested week.")
    merged = merged.sort_values(
        ["incident_datetime", "incident_id"], kind="stable"
    )

    partial = path.with_suffix(".csv.partial")
    merged.to_csv(partial, index=False)
    os.replace(partial, path)
    print(f"    Cached {len(merged):,} source rows.")
    return path


def combine_weeks(paths: list[Path], target: Path, columns: list[str]) -> int:
    """Combine CSVs without loading the million-row history into memory."""
    partial = target.with_suffix(".csv.partial")
    total = 0
    with partial.open("w", encoding="utf-8", newline="") as dest:
        for i, path in enumerate(paths):
            with path.open("r", encoding="utf-8", newline="") as src:
                reader = csv.reader(src)
                actual_header = next(reader)
                require_columns(actual_header, columns, str(path))
                if i == 0:
                    writer = csv.writer(dest)
                    writer.writerow(actual_header)
                # Raw CSV rows are copied intact (after the verified header).
                # Reading by record handles any quoted field with a newline.
                writer = csv.writer(dest)
                for row in reader:
                    if len(row) != len(columns):
                        raise RuntimeError(f"Malformed CSV record in {path}")
                    writer.writerow(row)
                    total += 1
    os.replace(partial, target)
    return total


def prepare_cohort(
    session: requests.Session,
    cohort: str,
    start: datetime,
    end: datetime,
    columns: list[str],
    filename: str,
    *,
    refresh: bool,
) -> tuple[Path, int]:
    target = DATA_DIR / filename
    if target.exists() and not refresh:
        require_columns(
            list(pd.read_csv(target, nrows=0).columns), columns, str(target)
        )
        with target.open("r", encoding="utf-8", newline="") as existing:
            row_count = sum(1 for _ in csv.reader(existing)) - 1
        print(f"Keeping existing {filename} ({row_count:,} rows).")
        return target, row_count

    windows = [
        fetch_week(session, cohort, a, b, columns, refresh=refresh)
        for a, b in weekly_windows(start, end)
    ]
    count = combine_weeks(windows, target, columns)
    print(f"Saved {filename} ({count:,} rows, {len(columns)} columns).")
    return target, count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the FDNY EMS source cohorts for the research notebook."
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Re-download all weeks and replace the two generated CSV files.",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "EMS-Research-Acquisition/1.0"})
    if os.getenv("SOCRATA_APP_TOKEN"):
        session.headers["X-App-Token"] = os.environ["SOCRATA_APP_TOKEN"]

    history, n_history = prepare_cohort(
        session, "history", HISTORY_START, FUTURE_START,
        HISTORY_COLUMNS, "incident_history.csv", refresh=args.refresh,
    )
    future, n_future = prepare_cohort(
        session, "future", FUTURE_START, FUTURE_END,
        FUTURE_COLUMNS, "incidents_to_predict.csv", refresh=args.refresh,
    )

    manifest = {
        "source": SOURCE_PAGE,
        "api": SOURCE_URL,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "historical_window": [
            source_datetime(HISTORY_START), source_datetime(FUTURE_START)
        ],
        "future_window": [
            source_datetime(FUTURE_START), source_datetime(FUTURE_END)
        ],
        "history_rows": n_history,
        "future_rows": n_future,
        "historical_columns": HISTORY_COLUMNS,
        "prediction_time_columns": FUTURE_COLUMNS,
        "no_outcomes_downloaded_for_future": True,
        "sha256": {
            history.name: sha256(history),
            future.name: sha256(future),
        },
    }
    (DATA_DIR / "acquisition_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\nPrepared notebook inputs:")
    print(f"  {history}")
    print(f"  {future}")
    if (n_history, n_future) != (913_796, 30_197):
        print(
            "\nNote: Current row counts differ from the notebook's saved "
            "2024 extracts (913,796 / 30,197). The live source may have "
            "changed; inspect the manifest rather than forcing old counts."
        )
    print("\nStart Jupyter from the directory containing this script.")


if __name__ == "__main__":
    main()
