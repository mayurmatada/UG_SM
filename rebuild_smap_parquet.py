from pathlib import Path
import os
import shutil
import sys

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path("Data/Global")
SMAP_DIR = ROOT / "smap"
SAMPLES_FILE = ROOT / "station_date_candidates.parquet"
OUTPUT = SMAP_DIR / "smap_by_sample.parquet"
TEMP = SMAP_DIR / "smap_by_sample.rebuild.tmp.parquet"
BACKUP = SMAP_DIR / "smap_by_sample.parquet.pre_csv_rebuild"

# Canonical sample columns and SMAP CSV columns.
CSV_COLUMNS = [
    "date",
    "sm_surface",
    "smap_n",
    "station_key",
]

def stop(message):
    raise SystemExit(f"\nERROR: {message}")

if not SAMPLES_FILE.exists():
    stop(f"Canonical samples not found: {SAMPLES_FILE}")

csv_files = sorted(SMAP_DIR.glob("*.csv"))
if len(csv_files) != 6:
    stop(
        f"Expected 6 CSV files in {SMAP_DIR}, found {len(csv_files)}:\n"
        + "\n".join(str(p) for p in csv_files)
    )

print("CSV files:")
for p in csv_files:
    print(f"  {p.name} ({p.stat().st_size:,} bytes)")

# Read and combine CSV exports.
parts = []
for path in csv_files:
    print(f"\nReading {path.name} ...")
    df = pd.read_csv(path, low_memory=False)

    # Handle harmless whitespace in exported column names.
    df.columns = [str(c).strip() for c in df.columns]

    missing = set(CSV_COLUMNS) - set(df.columns)
    if missing:
        stop(f"{path.name} is missing columns: {sorted(missing)}")

    df = df[CSV_COLUMNS].copy()
    df["station_key"] = df["station_key"].astype("string").str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["date"] = df["date"].dt.tz_convert(None).dt.normalize()
    df["sm_surface"] = pd.to_numeric(df["sm_surface"], errors="coerce")
    df["smap_n"] = pd.to_numeric(df["smap_n"], errors="coerce")

    bad = df["station_key"].isna() | df["date"].isna()
    if bad.any():
        print(f"  Dropping {int(bad.sum()):,} rows with missing key/date")
        df = df.loc[~bad].copy()

    print(f"  Rows: {len(df):,}")
    parts.append(df)

smap = pd.concat(parts, ignore_index=True)
print(f"\nCombined CSV rows: {len(smap):,}")

# Remove exact duplicate records caused by overlapping exports.
before = len(smap)
smap = smap.drop_duplicates(
    subset=["station_key", "date", "sm_surface", "smap_n"]
).copy()
print(f"Exact duplicates removed: {before - len(smap):,}")

# Each station/date should have one row. Conflicting values need attention.
key = ["station_key", "date"]
conflicts = (
    smap.groupby(key, dropna=False)["sm_surface"]
    .nunique(dropna=True)
)
conflicts = conflicts[conflicts > 1]

if len(conflicts):
    print("\nConflicting SMAP values for station/date pairs.")
    print(conflicts.head(20).to_string())
    stop(
        f"Found {len(conflicts):,} station/date pairs with multiple "
        "distinct sm_surface values. Resolve these before merging."
    )

# Collapse duplicate station/date records if only their counts differ.
smap = (
    smap.groupby(key, as_index=False, dropna=False)
    .agg(
        sm_surface=("sm_surface", "mean"),
        smap_n=("smap_n", "max"),
    )
)

# Load canonical samples; normalize their join keys.
samples = pd.read_parquet(SAMPLES_FILE)
required = {"sample_id", "station_key", "sample_date"}
missing = required - set(samples.columns)
if missing:
    stop(
        f"{SAMPLES_FILE} is missing columns: {sorted(missing)}. "
        f"Available columns: {samples.columns.tolist()}"
    )

samples["station_key"] = samples["station_key"].astype("string").str.strip()
samples["sample_date"] = pd.to_datetime(
    samples["sample_date"], errors="coerce", utc=True
).dt.tz_convert(None).dt.normalize()

if samples["sample_id"].isna().any():
    stop("Canonical samples contain null sample_id values.")
if samples["sample_id"].duplicated().any():
    stop("Canonical samples contain duplicate sample_id values.")

print(f"\nCanonical samples: {len(samples):,}")
print(f"Unique SMAP station/date pairs: {len(smap):,}")

# Left join ensures the output retains every canonical sample.
merged = samples.merge(
    smap,
    how="left",
    left_on=["station_key", "sample_date"],
    right_on=["station_key", "date"],
    validate="one_to_one",
    indicator=True,
)

matched = int((merged["_merge"] == "both").sum())
unmatched = len(merged) - matched
print(f"Matched samples:   {matched:,}")
print(f"Unmatched samples: {unmatched:,}")

if matched == 0:
    stop(
        "No samples matched. Check whether station_key and date formats "
        "in the CSVs match the canonical sample index."
    )

# Keep the original date column name expected in the feature table.
merged["SMAP"] = merged["sm_surface"]
merged["SMAP_n_obs"] = merged["smap_n"]

# Preserve useful canonical metadata, plus SMAP fields.
keep = list(samples.columns) + ["SMAP", "SMAP_n_obs"]
result = merged[keep].copy()

# Write to a temporary file and validate before touching the output.
if TEMP.exists():
    TEMP.unlink()

result.to_parquet(TEMP, index=False, compression="zstd")

pf = pq.ParquetFile(TEMP)
if pf.metadata.num_rows != len(samples):
    TEMP.unlink(missing_ok=True)
    stop("Temporary Parquet row count differs from canonical sample count.")

# Validate uniqueness after persistence.
written = pd.read_parquet(TEMP, columns=["sample_id"])
if written["sample_id"].isna().any() or written["sample_id"].duplicated().any():
    TEMP.unlink(missing_ok=True)
    stop("Temporary Parquet sample_id validation failed.")

# Preserve the current damaged output if there isn't already a pre-rebuild backup.
if OUTPUT.exists() and not BACKUP.exists():
    shutil.copy2(OUTPUT, BACKUP)
    print(f"\nPreserved existing output as: {BACKUP}")

os.replace(TEMP, OUTPUT)

# Final footer/row-count check.
final = pq.ParquetFile(OUTPUT)
print("\nSUCCESS")
print(f"Output: {OUTPUT}")
print(f"Rows: {final.metadata.num_rows:,}")
print(f"Columns: {final.schema.names}")
print(f"SMAP matched: {matched:,} ({matched / len(samples):.1%})")
print(f"SMAP missing: {unmatched:,} ({unmatched / len(samples):.1%})")
print("Note: unmatched samples remain in the output with null SMAP values.")
