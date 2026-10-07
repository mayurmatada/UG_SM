"""Build a station-level audit catalog from an ISMN directory or archive.

The default mode reads only station-file headers, so it is suitable for the
first global cohort-selection pass. ``--scan-observations`` additionally
counts rows and good-quality observations, but can take substantially longer
for a complete global archive.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Iterator, TextIO

from global_data_contract import (
    GlobalDataContract,
    QUALITY_FLAG_GOOD,
)


HEADER_PATTERN = re.compile(
    r"^(?P<network>\S+)\s+(?P<network_name>\S+)\s+"
    r"(?P<station>.+?)\s+(?P<latitude>-?\d+(?:\.\d+)?)\s+"
    r"(?P<longitude>-?\d+(?:\.\d+)?)\s+(?P<elevation>-?\d+(?:\.\d+)?)\s+"
    r"(?P<depth_from>\d+(?:\.\d+)?)\s+(?P<depth_to>\d+(?:\.\d+)?)\s+"
    r"'(?P<sensor>[^']*)'$"
)


@dataclass
class StationRecord:
    source_file: str
    network: str
    network_name: str
    station: str
    latitude: float
    longitude: float
    elevation_m: float
    depth_from_m: float
    depth_to_m: float
    sensor: str
    depth_class: str
    exclusion_reason: str | None
    static_variables_file: str | None
    observation_rows: int | None = None
    good_quality_rows: int | None = None
    malformed_rows: int | None = None
    first_observation_date: str | None = None
    last_observation_date: str | None = None

    @property
    def compatible_depth(self) -> bool:
        """Backward-compatible alias for consumers of the starter catalog."""
        return self.depth_class in {"near_surface", "fallback_deeper"}


def parse_header(line: str, source_file: str) -> StationRecord:
    match = HEADER_PATTERN.match(line.strip())
    if match is None:
        raise ValueError(f"Unsupported ISMN station header in {source_file!r}: {line!r}")
    values = match.groupdict()
    depth_from = float(values["depth_from"])
    depth_to = float(values["depth_to"])
    contract = GlobalDataContract()
    depth_class = contract.depth_class(depth_from, depth_to)
    exclusion_reason = None if depth_class != "excluded" else "depth_outside_0_to_10_cm"
    return StationRecord(
        source_file=source_file,
        network=values["network"],
        network_name=values["network_name"],
        station=values["station"],
        latitude=float(values["latitude"]),
        longitude=float(values["longitude"]),
        elevation_m=float(values["elevation"]),
        depth_from_m=depth_from,
        depth_to_m=depth_to,
        sensor=values["sensor"],
        depth_class=depth_class,
        exclusion_reason=exclusion_reason,
        static_variables_file=_static_file_for(source_file),
    )


def _static_file_for(source_file: str) -> str | None:
    path = Path(source_file)
    if path.name.endswith(".stm"):
        candidate = path.with_name(path.name.split("_sm_", 1)[0] + "_static_variables.csv")
        return str(candidate) if source_file.startswith("/") else str(candidate).replace("\\", "/")
    return None


def scan_rows(lines: Iterable[str]) -> tuple[int, int, int, str | None, str | None]:
    rows = good = malformed = 0
    dates: list[date] = []
    for line in lines:
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 4:
            malformed += 1
            continue
        try:
            observation_date = date.fromisoformat(fields[0].replace("/", "-"))
        except ValueError:
            malformed += 1
            continue
        rows += 1
        dates.append(observation_date)
        if fields[3] == QUALITY_FLAG_GOOD:
            good += 1
    return (
        rows,
        good,
        malformed,
        min(dates).isoformat() if dates else None,
        max(dates).isoformat() if dates else None,
    )


def iter_files(source: Path) -> Iterator[tuple[str, TextIO | _decoded_stream]]:
    if source.is_dir():
        for path in sorted(source.rglob("*.stm")):
            yield str(path), path.open("r", encoding="utf-8", errors="replace")
        return
    if source.suffix.lower() != ".zip":
        raise ValueError(f"Source must be a directory or .zip archive: {source}")
    with zipfile.ZipFile(source) as archive:
        for name in sorted(info.filename for info in archive.infolist() if info.filename.endswith(".stm")):
            stream = archive.open(name, "r")
            yield name, _decoded_stream(stream)


class _decoded_stream:
    def __init__(self, stream: object) -> None:
        self.stream = stream

    def __iter__(self) -> Iterator[str]:
        for line in self.stream:  # type: ignore[union-attr]
            yield line.decode("utf-8", errors="replace")

    def close(self) -> None:
        self.stream.close()  # type: ignore[union-attr]


def build_catalog(source: Path, scan_observations: bool) -> list[StationRecord]:
    records: list[StationRecord] = []
    for source_file, stream in iter_files(source):
        try:
            header = next(iter(stream))
            record = parse_header(header, source_file)
            if scan_observations:
                rows, good, malformed, first_date, last_date = scan_rows(stream)
                record.observation_rows = rows
                record.good_quality_rows = good
                record.malformed_rows = malformed
                record.first_observation_date = first_date
                record.last_observation_date = last_date
            records.append(record)
        finally:
            stream.close()
    return records


def write_outputs(records: list[StationRecord], output_csv: Path, output_json: Path) -> None:
    if not records:
        raise ValueError("Cannot write an empty station catalog")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0])))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)

    network_counts = Counter(record.network for record in records)
    depth_counts = Counter(record.depth_class for record in records)
    exclusion_counts = Counter(
        record.exclusion_reason for record in records if record.exclusion_reason is not None
    )
    summary = {
        "contract": GlobalDataContract().as_dict(),
        "station_files": len(records),
        "compatible_depth_files": sum(record.compatible_depth for record in records),
        "networks": len(network_counts),
        "files_by_network": dict(sorted(network_counts.items())),
        "files_by_depth_class": dict(sorted(depth_counts.items())),
        "exclusions": dict(sorted(exclusion_counts.items())),
        "good_quality_rows": sum(record.good_quality_rows or 0 for record in records),
        "observation_rows": sum(record.observation_rows or 0 for record in records),
        "non_good_quality_rows": sum(
            (record.observation_rows or 0) - (record.good_quality_rows or 0)
            for record in records
        ),
        "malformed_rows": sum(record.malformed_rows or 0 for record in records),
    }
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="ISMN directory or global .zip archive")
    parser.add_argument("--output-csv", type=Path, default=Path("global_station_catalog.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("global_station_catalog_summary.json"))
    parser.add_argument(
        "--scan-observations",
        action="store_true",
        help="Count rows and good-quality observations in every .stm file",
    )
    args = parser.parse_args()
    records = build_catalog(args.source, args.scan_observations)
    if not records:
        raise SystemExit("No .stm station files found")
    write_outputs(records, args.output_csv, args.output_json)
    print(f"Wrote {len(records)} station files to {args.output_csv}")
    print(f"Compatible depth files: {sum(record.compatible_depth for record in records)}")


if __name__ == "__main__":
    main()
