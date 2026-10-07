"""Create a deterministic, network-disjoint global cohort split manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


REQUIRED_COLUMNS = {
    "source_file",
    "network",
    "station",
    "latitude",
    "longitude",
    "depth_class",
    "exclusion_reason",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"Catalog is missing required columns: {sorted(missing)}")
        return list(reader)


def _as_int(row: dict[str, str], name: str) -> int:
    value = row.get(name, "")
    if not value:
        return 0
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"Catalog column {name!r} is not an integer: {value!r}") from error


def select_cohort(
    rows: Iterable[dict[str, str]],
    *,
    minimum_good_quality_rows: int = 1,
) -> list[dict[str, str]]:
    """Keep contract-compatible station files with audited good observations."""
    selected: list[dict[str, str]] = []
    for row in rows:
        if row["depth_class"] not in {"near_surface", "fallback_deeper"}:
            continue
        if row["exclusion_reason"]:
            continue
        if _as_int(row, "good_quality_rows") < minimum_good_quality_rows:
            continue
        item = dict(row)
        item["cohort_eligible"] = "true"
        selected.append(item)
    if not selected:
        raise ValueError("No catalog rows satisfy the cohort filters")
    return selected


def _stable_network_order(networks: Iterable[str], seed: int) -> list[str]:
    return sorted(
        set(networks),
        key=lambda network: hashlib.sha256(f"{seed}:{network}".encode()).hexdigest(),
    )


def assign_network_splits(
    rows: list[dict[str, str]],
    *,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 20261003,
) -> list[dict[str, str]]:
    """Assign every network to exactly one split using stable hash ordering."""
    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must be non-negative and sum to less than one")
    networks = _stable_network_order((row["network"] for row in rows), seed)
    if len(networks) < 3:
        raise ValueError("At least three networks are required for disjoint train/validation/test splits")
    test_count = max(1, round(len(networks) * test_fraction))
    validation_count = max(1, round(len(networks) * validation_fraction))
    if test_count + validation_count >= len(networks):
        raise ValueError("Split fractions leave no network for training")
    assignments = {
        network: (
            "test"
            if index < test_count
            else "validation"
            if index < test_count + validation_count
            else "train"
        )
        for index, network in enumerate(networks)
    }
    output = []
    for row in sorted(rows, key=lambda item: (item["network"], item["station"], item["source_file"])):
        item = dict(row)
        item["split"] = assignments[item["network"]]
        item["split_seed"] = str(seed)
        output.append(item)
    return output


def write_manifest(rows: list[dict[str, str]], output_csv: Path, output_json: Path) -> None:
    if not rows:
        raise ValueError("Cannot write an empty split manifest")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "rows": len(rows),
        "networks": len({row["network"] for row in rows}),
        "stations": len({(row["network"], row["station"]) for row in rows}),
        "rows_by_split": dict(sorted(Counter(row["split"] for row in rows).items())),
        "networks_by_split": {
            split: sorted({row["network"] for row in rows if row["split"] == split})
            for split in ("train", "validation", "test")
        },
    }
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("--output-csv", type=Path, default=Path("global_split_manifest.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("global_split_summary.json"))
    parser.add_argument("--minimum-good-quality-rows", type=int, default=1)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()
    if args.minimum_good_quality_rows < 1:
        raise SystemExit("--minimum-good-quality-rows must be positive")
    rows = assign_network_splits(
        select_cohort(
            _read_csv(args.catalog),
            minimum_good_quality_rows=args.minimum_good_quality_rows,
        ),
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    write_manifest(rows, args.output_csv, args.output_json)
    print(f"Wrote {len(rows)} eligible station files to {args.output_csv}")


if __name__ == "__main__":
    main()
