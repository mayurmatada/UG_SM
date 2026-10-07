"""Interactively submit optimized global Earth Engine exports year by year.

Run this script with the Python environment that has earthengine-api installed.
Static features are submitted once; each following prompt submits only labeled
dates for one calendar year.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent
DEFAULT_EXPORTER = REPOSITORY_ROOT / "gee_global_export.py"
DEFAULT_STATIONS = REPOSITORY_ROOT / "global_station_catalog.csv"
DEFAULT_LABEL_DATES = REPOSITORY_ROOT / "global_pipeline_2015" / "global_label_dates.csv"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "global_gee_exports"


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{description} does not exist: {path}")


def _ask(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() in {"y", "yes"}
    except EOFError:
        return False


def _run_exporter(
    *,
    python: str,
    exporter: Path,
    common_args: list[str],
    extra_args: list[str],
) -> None:
    command = [python, str(exporter), *common_args, *extra_args]
    print("\nRunning:", " ".join(command))
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Exporter failed with exit code {completed.returncode}: "
            f"{' '.join(command)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ee-project", default=os.environ.get("EE_PROJECT"))
    parser.add_argument("--exporter", type=Path, default=DEFAULT_EXPORTER)
    parser.add_argument("--stations", type=Path, default=DEFAULT_STATIONS)
    parser.add_argument("--label-dates", type=Path, default=DEFAULT_LABEL_DATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--first-year", type=int, default=2015)
    parser.add_argument("--last-year", type=int, default=2026)
    parser.add_argument("--start-date", default="2015-04-01")
    parser.add_argument("--end-date", default="2026-10-03")
    parser.add_argument("--drive-folder", default="global_ismn_exports")
    parser.add_argument("--max-stations-per-tile", type=int, default=100)
    parser.add_argument("--tile-scale", type=int, default=16)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create manifests without submitting Earth Engine tasks",
    )
    args = parser.parse_args()

    _require_file(args.exporter, "Exporter")
    _require_file(args.stations, "Station manifest")
    _require_file(args.label_dates, "Label-date manifest")
    if not args.ee_project and not args.dry_run:
        raise SystemExit(
            "Earth Engine project is required. Pass --ee-project PROJECT_ID "
            "or set EE_PROJECT."
        )
    if args.first_year > args.last_year:
        raise SystemExit("--first-year must not be after --last-year")

    common_args = [
        str(args.stations),
        "--label-dates",
        str(args.label_dates),
        "--start-date",
        args.start_date,
        "--end-date",
        args.end_date,
        "--drive-folder",
        args.drive_folder,
        "--max-stations-per-tile",
        str(args.max_stations_per_tile),
        "--tile-scale",
        str(args.tile_scale),
    ]
    if args.ee_project:
        common_args.extend(["--ee-project", args.ee_project])
    if args.dry_run:
        common_args.append("--dry-run")

    static_output = args.output_dir / "static"
    static_args = ["--static-only", "--output-dir", str(static_output)]
    if args.dry_run or _ask("Submit static exports once? [y/N] "):
        try:
            _run_exporter(
                python=sys.executable,
                exporter=args.exporter,
                common_args=common_args,
                extra_args=static_args,
            )
        except RuntimeError as error:
            raise SystemExit(str(error)) from error
    else:
        print("Static exports skipped.")

    for year in range(args.first_year, args.last_year + 1):
        if not args.dry_run and not _ask(f"Submit labeled exports for {year}? [y/N] "):
            print(f"Skipping {year}.")
            continue

        year_args = [
            "--year",
            str(year),
            "--skip-static",
            "--output-dir",
            str(args.output_dir / str(year)),
        ]
        try:
            _run_exporter(
                python=sys.executable,
                exporter=args.exporter,
                common_args=common_args,
                extra_args=year_args,
            )
        except RuntimeError as error:
            print(error, file=sys.stderr)
            if not _ask("Continue with the next year? [y/N] "):
                raise SystemExit(1) from error

    print("\nFinished interactive export loop.")


if __name__ == "__main__":
    main()
