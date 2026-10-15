"""Paths, label order, and thresholds shared by training and the API.

Sierra Leone wall-clock times are stored as naive timestamps. The country is
on UTC, so every conversion below treats that clock time as UTC and never as
the laptop's local zone. That matters for mentees working from Kenya (UTC+3):
pandas `.timestamp()` on a naive value is easy to mix with `fromtimestamp`,
which would shift the hour-of-day features by three hours.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
CASE_DIR = PACKAGE_DIR.parent
REPO_ROOT = PACKAGE_DIR.parents[2]

# On Kaggle the code dataset is read-only. Set these before importing this
# module so caches and models land in /kaggle/working. See bootstrap.py.
WORK_DIR = Path(os.environ.get("POWER_SOURCE_WORK", CASE_DIR / "work"))
RAW_DIR = WORK_DIR / "raw"
FEATURE_DIR = WORK_DIR / "features"
ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", CASE_DIR / "artifacts"))
FIGURE_DIR = ARTIFACT_DIR / "figures"

# Bump this when the feature definition changes so cached npz files rebuild.
FEATURE_VERSION = "1"

# Counted by a full csv.reader pass: header + 5,125,181 data rows, 0 malformed.
EXPECTED_DATA_ROWS = 5_125_181

# Excel worksheets stop at 2^20 rows. Two sheets are the usual "2.1 million".
EXCEL_ROW_LIMIT = 1_048_576
TWO_EXCEL_SHEETS = EXCEL_ROW_LIMIT * 2

CLASSES = ["generator", "grid", "solar"]
LABEL_TO_ID = {name: i for i, name in enumerate(CLASSES)}

HOLDOUT_START = datetime(2024, 2, 1, tzinfo=timezone.utc)
HOLDOUT_START_S = int(HOLDOUT_START.timestamp())

# Temporal folds used on the time-ordered teaching sample (train portion only).
CV_CUTOFFS = [
    datetime(2023, 8, 1, tzinfo=timezone.utc),
    datetime(2023, 10, 1, tzinfo=timezone.utc),
    datetime(2023, 12, 1, tzinfo=timezone.utc),
]

BUCKET_SECONDS = 120
TRAIL_WINDOW = 7
CENTER_RADIUS = 3
MAX_GAP_SECONDS = 600
DURATION_CAP_SECONDS = 30 * 60
SEQUENCE_LENGTH = 16

# Nominal Sierra Leone mains.
NOMINAL_HZ = 50.0
NOMINAL_VOLT = 230.0
SOLAR_HZ = (49.8, 50.2)
GRID_DEV_HZ = 0.11
SOLAR_MAD = 0.1
GRID_OFF_DEV_HZ = 0.5
LOW_VOLTAGE = 50.0

PCMH_SITE = "princess_christian"
OLA_SITE = "ola_during"

RANDOM_STATE = 42
TABULAR_PER_CLASS = 40_000
WINDOW_TRAIN_PER_CLASS = 1_500
WINDOW_TEST_PER_CLASS = 500

CSV_COLUMNS = [
    "respondent_id",
    "time",
    "site_id",
    "room",
    "voltage",
    "frequency",
    "available_power_sources",
    "active_power_source",
]

FEATURE_COLUMNS = [
    "voltage",
    "frequency",
    "abs_freq_dev_50",
    "abs_volt_dev_230",
    "freq_diff_1",
    "volt_diff_1",
    "trail_mean_freq",
    "trail_std_freq",
    "trail_mad_freq",
    "trail_min_freq",
    "trail_max_freq",
    "trail_std_volt",
    "freq_dev_site_median",
    "n_sensors_in_bucket",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "has_grid",
    "has_solar",
    "has_generator",
    "n_sources",
    "gap_minutes",
    "low_voltage",
]


def _search_roots() -> list[Path]:
    roots = []
    for candidate in (
        Path("/kaggle/input"),
        Path("/content/drive/MyDrive"),
        Path("/content"),
    ):
        if candidate.exists():
            roots.append(candidate)
    roots.append(REPO_ROOT)
    return roots


def _find_extract(roots: list[Path]) -> Path | None:
    named: list[Path] = []
    for root in roots:
        if root.exists():
            named.extend(root.rglob("labeled_power_source_data.csv"))
    if named:
        return max(named, key=lambda path: path.stat().st_size)
    # A Kaggle dataset sometimes keeps the file under a different name.
    csvs: list[Path] = []
    for root in roots:
        if root == REPO_ROOT or not root.exists():
            continue
        csvs.extend(path for path in root.rglob("*.csv") if path.is_file())
    if not csvs:
        return None
    return max(csvs, key=lambda path: path.stat().st_size)


def resolve_csv_path() -> Path:
    override = os.environ.get("POWER_SOURCE_CSV")
    if override:
        path = Path(override)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    local = (
        REPO_ROOT
        / "Kamilimu_Case_Study_Dataset"
        / "Kamilimu_Case_Study_Dataset"
        / "Sierra Leone"
        / "labeled_power_source_data.csv"
    )
    if local.exists():
        return local
    found = _find_extract(_search_roots())
    if found is not None and found.exists():
        return found
    expected = (
        REPO_ROOT
        / "Kamilimu_Case_Study_Dataset"
        / "Kamilimu_Case_Study_Dataset"
        / "Sierra Leone"
        / "labeled_power_source_data.csv"
    )
    raise FileNotFoundError(
        "Could not find labeled_power_source_data.csv. "
        f"Looked for {expected}, under /kaggle/input, and under /content. "
        "Set POWER_SOURCE_CSV to the file."
    )
