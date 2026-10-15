"""Stream the 5.1 million-row extract without ever holding it as one DataFrame.

Two things produce the "we only got 2.1 million rows" result:

1. Excel's worksheet limit is 1,048,576 rows. Saving two sheets and
   concatenating them yields 2,097,152 rows, which is the figure teams report.
2. A single `pandas.read_csv` of this 789 MB file builds a wide object-dtype
   frame. On an 8 GB laptop that allocation dies part way through, and the
   rows left in memory are again on the order of a couple of million.

This loader counts every physical record, refuses to continue if that count
is not the known extract size, and writes one compact file per sensor
(there are 35). Later steps read a single sensor at a time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.config import (
    CSV_COLUMNS,
    EXPECTED_DATA_ROWS,
    RAW_DIR,
    resolve_csv_path,
)
from pipeline.log import log

RAW_SCHEMA = pa.schema(
    [
        ("time", pa.timestamp("ns")),
        ("voltage", pa.float32()),
        ("frequency", pa.float32()),
        ("active_power_source", pa.string()),
        ("site_id", pa.string()),
        ("room", pa.string()),
        ("available_power_sources", pa.string()),
    ]
)


def count_newline_rows(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> int:
    """Count newline characters. The extract is newline-delimited with no quotes."""
    total = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            total += block.count(b"\n")
    return total


def audit_csv(path: Path | None = None) -> dict:
    """Count records two ways and return a ledger. Does not build a DataFrame."""
    path = path or resolve_csv_path()
    newlines = count_newline_rows(path)
    # Header is one line and the extract ends in a newline, so data rows are
    # newlines minus the header.
    data_rows_from_newlines = newlines - 1
    size_mb = path.stat().st_size / 1e6
    ledger = {
        "path": str(path),
        "size_mb": round(size_mb, 1),
        "newline_count": newlines,
        "data_rows_from_newlines": data_rows_from_newlines,
        "expected_data_rows": EXPECTED_DATA_ROWS,
        "matches_expected": data_rows_from_newlines == EXPECTED_DATA_ROWS,
    }
    log(
        f"newline audit: {data_rows_from_newlines:,} data rows "
        f"({size_mb:.0f} MB), expected {EXPECTED_DATA_ROWS:,}"
    )
    return ledger


def _ready_marker(directory: Path) -> Path:
    return directory / "READY.json"


def partition_by_sensor(path: Path | None = None, force: bool = False) -> dict:
    """Write one parquet file per respondent_id. Returns the row ledger."""
    path = path or resolve_csv_path()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    marker = _ready_marker(RAW_DIR)
    if marker.exists() and not force:
        cached = json.loads(marker.read_text(encoding="utf-8"))
        if cached.get("data_rows") == EXPECTED_DATA_ROWS and cached.get("source") == str(path):
            log(f"reusing sensor partitions in {RAW_DIR}")
            return cached

    audit = audit_csv(path)
    if not audit["matches_expected"]:
        raise RuntimeError(
            "Row count does not match the extract. "
            f"Got {audit['data_rows_from_newlines']:,}, expected {EXPECTED_DATA_ROWS:,}. "
            "Refusing to train on a partial file."
        )

    for old in RAW_DIR.glob("*.parquet"):
        old.unlink()

    writers: dict[str, pq.ParquetWriter] = {}
    pandas_rows = 0
    try:
        reader = pd.read_csv(
            path,
            usecols=CSV_COLUMNS,
            chunksize=100_000,
            dtype={
                "voltage": np.float32,
                "frequency": np.float32,
                "respondent_id": "string",
                "site_id": "string",
                "room": "string",
                "available_power_sources": "string",
                "active_power_source": "string",
            },
            parse_dates=["time"],
            engine="c",
            on_bad_lines="error",
        )
        for chunk_id, chunk in enumerate(reader, start=1):
            if chunk["voltage"].isna().any() or chunk["frequency"].isna().any():
                raise RuntimeError(f"Chunk {chunk_id} contains non-numeric voltage or frequency.")
            pandas_rows += len(chunk)
            for respondent_id, group in chunk.groupby("respondent_id", sort=False):
                table = pa.Table.from_pandas(
                    group.drop(columns=["respondent_id"]),
                    schema=RAW_SCHEMA,
                    preserve_index=False,
                )
                key = str(respondent_id)
                if key not in writers:
                    writers[key] = pq.ParquetWriter(RAW_DIR / f"{key}.parquet", RAW_SCHEMA)
                writers[key].write_table(table)
            if chunk_id % 10 == 0:
                log(f"partitioned {pandas_rows:,} rows")
    finally:
        for writer in writers.values():
            writer.close()

    if pandas_rows != EXPECTED_DATA_ROWS:
        raise RuntimeError(
            "pandas read a different number of rows than the newline audit. "
            f"pandas={pandas_rows:,}, expected={EXPECTED_DATA_ROWS:,}. "
            "No rows were skipped on purpose; this is a parser mismatch."
        )

    ledger = {
        "source": str(path),
        "data_rows": pandas_rows,
        "sensors": sorted(writers),
        "n_sensors": len(writers),
        "newline_audit": audit,
    }
    marker.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    log(f"partitioned {pandas_rows:,} rows across {len(writers)} sensors")
    return ledger
