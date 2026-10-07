"""Build a cell/date manifest from good-quality ISMN observations."""

from __future__ import annotations

import argparse
import csv
import re
import zipfile
from datetime import date
from pathlib import Path

from build_global_ismn_catalog import HEADER_PATTERN
from gee_global_export import _global_cell_id
from global_data_contract import GlobalDataContract, QUALITY_FLAG_GOOD


def build_label_dates(source: Path) -> list[dict[str, str]]:
    if source.is_dir():
        streams = (
            (str(path), path.open("r", encoding="utf-8", errors="replace"))
            for path in sorted(source.rglob("*.stm"))
        )
    elif source.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(source)
        streams = (
            (name, _decoded(archive.open(name, "r")))
            for name in sorted(info.filename for info in archive.infolist() if info.filename.endswith(".stm"))
        )
    else:
        raise ValueError("source must be an ISMN directory or .zip archive")

    contract = GlobalDataContract()
    dates: set[tuple[str, str, str, str]] = set()
    for source_file, stream in streams:
        try:
            header = next(iter(stream), "")
            match = HEADER_PATTERN.match(header.strip())
            if match is None:
                continue
            values = match.groupdict()
            depth_class = contract.depth_class(float(values["depth_from"]), float(values["depth_to"]))
            if depth_class not in {"near_surface", "fallback_deeper"}:
                continue
            station_id = f"{values['network']}__{values['station']}"
            cell_id = _global_cell_id(float(values["latitude"]), float(values["longitude"]))
            for line in stream:
                fields = line.split()
                if len(fields) < 4 or fields[3] != QUALITY_FLAG_GOOD:
                    continue
                try:
                    day = date.fromisoformat(fields[0].replace("/", "-")).isoformat()
                except ValueError:
                    continue
                dates.add((cell_id, day, station_id, depth_class))
        finally:
            stream.close()
    return [
        {"cell_id": cell, "sample_date": day, "station_id": station, "depth_class": depth}
        for cell, day, station, depth in sorted(dates)
    ]


class _decoded:
    def __init__(self, stream: object) -> None:
        self.stream = stream

    def __iter__(self):
        for line in self.stream:  # type: ignore[union-attr]
            yield line.decode("utf-8", errors="replace")

    def close(self) -> None:
        self.stream.close()  # type: ignore[union-attr]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("global_label_dates.csv"))
    args = parser.parse_args()
    rows = build_label_dates(args.source)
    if not rows:
        raise SystemExit("No good-quality compatible label dates found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} cell/date label rows to {args.output}")


if __name__ == "__main__":
    main()
