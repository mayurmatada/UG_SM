"""Convert the ISMN archive into a compact, satellite-joinable Parquet table.

The output contains one row per station file and UTC calendar day.  Hourly
observations are averaged only after filtering to the documented good-quality
flag, while sensor, depth, location, and source-file provenance are retained.
Both directories and ISMN ZIP archives are supported without extracting the
archive to disk.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Iterator, TextIO

import pyarrow as pa
import pyarrow.parquet as pq

from build_global_ismn_catalog import HEADER_PATTERN
from global_data_contract import GlobalDataContract, QUALITY_FLAG_GOOD


@dataclass(frozen=True)
class SourceFile:
    name: str
    stream: TextIO
    static_stream: TextIO | None


def _text_stream(stream: object) -> TextIO:
    class Decoded:
        def __iter__(self) -> Iterator[str]:
            for line in stream:  # type: ignore[union-attr]
                yield line.decode("utf-8", errors="replace")

        def close(self) -> None:
            stream.close()  # type: ignore[union-attr]

    return Decoded()  # type: ignore[return-value]


def iter_source_files(source: Path) -> Iterator[SourceFile]:
    if source.is_dir():
        for path in sorted(source.rglob("*.stm")):
            static = next(path.parent.glob("*_static_variables.csv"), None)
            yield SourceFile(
                str(path),
                path.open("r", encoding="utf-8", errors="replace"),
                static.open("r", encoding="utf-8", errors="replace") if static else None,
            )
        return
    if source.suffix.lower() != ".zip":
        raise ValueError(f"Source must be a directory or .zip archive: {source}")
    with zipfile.ZipFile(source) as archive:
        names = {info.filename for info in archive.infolist()}
        for name in sorted(n for n in names if n.endswith(".stm")):
            static_name = next(
                (n for n in names if n.endswith("_static_variables.csv")
                 and Path(n).parent == Path(name).parent),
                None,
            )
            observation = _text_stream(archive.open(name, "r"))
            static = _text_stream(archive.open(static_name, "r")) if static_name else None
            yield SourceFile(name, observation, static)
            observation.close()
            if static:
                static.close()


def _static_values(stream: TextIO | None) -> dict[str, str | None]:
    if stream is None:
        return {}
    values: dict[str, str | None] = {}
    for row in csv.DictReader(stream, delimiter=";"):
        name = (row.get("quantity_name") or "").strip().lower()
        if name in {"clay fraction", "sand fraction", "silt fraction"} and name not in values:
            values[name] = (row.get("value") or "").strip() or None
        elif name == "climate classification" and "climate_zone" not in values:
            values["climate_zone"] = (row.get("value") or "").strip() or None
    return values


def _daily_rows(source_file: SourceFile, include_deeper: bool) -> tuple[list[dict[str, object]], dict[str, int]]:
    header = next(iter(source_file.stream), "")
    match = HEADER_PATTERN.match(header.strip())
    stats = defaultdict(int)
    if match is None:
        stats["malformed_station_files"] += 1
        return [], stats

    values = match.groupdict()
    depth_from = float(values["depth_from"])
    depth_to = float(values["depth_to"])
    depth_class = GlobalDataContract().depth_class(depth_from, depth_to)
    if not include_deeper and depth_class not in {"near_surface", "fallback_deeper"}:
        stats["excluded_depth_files"] += 1
        return [], stats

    static = _static_values(source_file.static_stream)
    by_day: dict[str, list[float]] = defaultdict(list)
    for line in source_file.stream:
        fields = line.split()
        stats["raw_rows"] += 1
        if len(fields) < 4 or fields[3] != QUALITY_FLAG_GOOD:
            stats["non_good_rows"] += 1
            continue
        try:
            day = date.fromisoformat(fields[0].replace("/", "-")).isoformat()
            measurement = float(fields[2])
        except (ValueError, TypeError):
            stats["malformed_rows"] += 1
            continue
        if not math.isfinite(measurement):
            stats["malformed_rows"] += 1
            continue
        by_day[day].append(measurement)

    station_id = f"{values['network']}__{values['station']}"
    rows = []
    for day, measurements in sorted(by_day.items()):
        rows.append({
            "station_id": station_id,
            "network": values["network"],
            "network_name": values["network_name"],
            "station": values["station"],
            "latitude": float(values["latitude"]),
            "longitude": float(values["longitude"]),
            "elevation_m": float(values["elevation"]),
            "sensor": values["sensor"],
            "depth_from_m": depth_from,
            "depth_to_m": depth_to,
            "depth_class": depth_class,
            "sample_date": date.fromisoformat(day),
            "soil_moisture_m3_m3": sum(measurements) / len(measurements),
            "observation_count": len(measurements),
            "source_file": source_file.name,
            "clay_pct": _float_or_none(static.get("clay fraction")),
            "sand_pct": _float_or_none(static.get("sand fraction")),
            "silt_pct": _float_or_none(static.get("silt fraction")),
            "climate_zone": static.get("climate_zone"),
        })
        stats["daily_rows"] += 1
        stats["good_rows"] += len(measurements)
    return rows, stats


def _float_or_none(value: str | None) -> float | None:
    try:
        return float(value) if value else None
    except ValueError:
        return None


def build_dataset(source: Path, output: Path, summary_output: Path, include_deeper: bool = True) -> dict[str, int]:
    """Stream ``source`` into Parquet and return conversion counters."""
    schema = pa.schema([
        ("station_id", pa.string()), ("network", pa.string()), ("network_name", pa.string()),
        ("station", pa.string()), ("latitude", pa.float64()), ("longitude", pa.float64()),
        ("elevation_m", pa.float64()), ("sensor", pa.string()), ("depth_from_m", pa.float64()),
        ("depth_to_m", pa.float64()), ("depth_class", pa.string()), ("sample_date", pa.date32()),
        ("soil_moisture_m3_m3", pa.float32()), ("observation_count", pa.int32()),
        ("source_file", pa.string()), ("clay_pct", pa.float32()), ("sand_pct", pa.float32()),
        ("silt_pct", pa.float32()), ("climate_zone", pa.string()),
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    totals: defaultdict[str, int] = defaultdict(int)
    with pq.ParquetWriter(output, schema, compression="zstd", use_dictionary=True) as writer:
        for source_file in iter_source_files(source):
            try:
                rows, stats = _daily_rows(source_file, include_deeper)
            finally:
                source_file.stream.close()
                if source_file.static_stream:
                    source_file.static_stream.close()
            for key, value in stats.items():
                totals[key] += value
            if rows:
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    summary = {
        "format": "parquet",
        "compression": "zstd",
        "aggregation": "UTC calendar day mean of good-quality observations",
        "include_deeper": include_deeper,
        "rows": totals["daily_rows"],
        **dict(totals),
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return dict(totals)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("Data/Global/global_ismn_daily.parquet"))
    parser.add_argument("--summary", type=Path, default=Path("Data/Global/global_ismn_daily_summary.json"))
    parser.add_argument("--compatible-depths-only", action="store_true")
    args = parser.parse_args()
    stats = build_dataset(args.source, args.output, args.summary, not args.compatible_depths_only)
    print(f"Wrote {stats.get('daily_rows', 0):,} daily rows to {args.output}")


if __name__ == "__main__":
    main()
