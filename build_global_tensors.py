"""Validate station-date feature manifests and pack model-ready NumPy arrays.

Each feature row references an ``.npz`` file with arrays named ``spatial``,
``temporal``, and ``static``. The target and anchor are supplied in the CSV so
the converter never derives labels from predictor values.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from global_data_contract import GlobalDataContract


REQUIRED_COLUMNS = {
    "station_id",
    "network",
    "sample_date",
    "split",
    "feature_path",
    "target",
    "anchor",
}
SPLITS = {"train", "validation", "test"}


def read_feature_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - columns
        if missing:
            catalog_columns = {"source_file", "station", "depth_class", "good_quality_rows"}
            if catalog_columns <= columns:
                raise ValueError(
                    "This file is an ISMN station catalog, not a station-date feature manifest. "
                    "Run build_global_splits.py first, then create a feature manifest with one "
                    "row per station/date and feature_path, target, and anchor columns after "
                    "the Earth Engine/patch exports are available. "
                    f"Missing feature-manifest columns: {sorted(missing)}"
                )
            raise ValueError(f"Feature manifest is missing required columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Feature manifest is empty")
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["station_id"], row["sample_date"])
        if key in seen:
            raise ValueError(f"Duplicate station/date feature row: {key}")
        seen.add(key)
        if row["split"] not in SPLITS:
            raise ValueError(f"Unsupported split {row['split']!r} for {key}")
    return rows


def _number(row: dict[str, str], name: str) -> float:
    try:
        return float(row[name])
    except (KeyError, ValueError) as error:
        raise ValueError(f"Invalid numeric value for {name!r}: {row.get(name)!r}") from error


def _load_feature(path: Path, contract: GlobalDataContract) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Feature file does not exist: {path}")
    with np.load(path) as data:
        missing = {"spatial", "temporal", "static"} - set(data.files)
        if missing:
            raise ValueError(f"Feature file {path} is missing arrays: {sorted(missing)}")
        spatial = np.asarray(data["spatial"], dtype=np.float32)
        temporal = np.asarray(data["temporal"], dtype=np.float32)
        static = np.asarray(data["static"], dtype=np.float32)
    expected_spatial = (contract.spatial_channel_count, contract.patch_size, contract.patch_size)
    if spatial.shape != expected_spatial:
        raise ValueError(f"{path}: expected spatial shape {expected_spatial}, got {spatial.shape}")
    if temporal.ndim != 2 or temporal.shape[0] < 1:
        raise ValueError(f"{path}: temporal array must have shape [sequence, features]")
    if static.shape != (contract.static_channel_count,):
        raise ValueError(
            f"{path}: expected static shape {(contract.static_channel_count,)}, got {static.shape}"
        )
    if not np.isfinite(spatial).all() or not np.isfinite(temporal).all() or not np.isfinite(static).all():
        raise ValueError(f"{path}: feature arrays contain non-finite values")
    return spatial, temporal, static


def pack_tensors(
    rows: list[dict[str, str]],
    *,
    manifest_dir: Path,
    output_dir: Path,
    contract: GlobalDataContract | None = None,
) -> dict[str, Path]:
    contract = contract or GlobalDataContract()
    output_dir.mkdir(parents=True, exist_ok=True)
    packed: dict[str, Path] = {}
    for split in ("train", "validation", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        if not split_rows:
            raise ValueError(f"Split {split!r} has no feature rows")
        spatial_items: list[np.ndarray] = []
        temporal_items: list[np.ndarray] = []
        static_items: list[np.ndarray] = []
        targets: list[float] = []
        anchors: list[float] = []
        metadata: list[dict[str, str | int]] = []
        sequence_length: int | None = None
        for index, row in enumerate(split_rows):
            feature_path = Path(row["feature_path"])
            if not feature_path.is_absolute():
                feature_path = manifest_dir / feature_path
            spatial, temporal, static = _load_feature(feature_path, contract)
            sequence_length = sequence_length or temporal.shape[0]
            if temporal.shape[0] != sequence_length:
                raise ValueError(
                    f"Mixed temporal sequence lengths in {split!r}: "
                    f"expected {sequence_length}, got {temporal.shape[0]} at {feature_path}"
                )
            spatial_items.append(spatial)
            temporal_items.append(temporal)
            static_items.append(static)
            targets.append(_number(row, "target"))
            anchors.append(_number(row, "anchor"))
            metadata.append(
                {"index": index, "station_id": row["station_id"], "network": row["network"],
                 "sample_date": row["sample_date"], "split": split}
            )
        output_path = output_dir / f"{split}.npz"
        np.savez_compressed(
            output_path,
            spatial=np.stack(spatial_items),
            temporal=np.stack(temporal_items),
            static=np.stack(static_items),
            target=np.asarray(targets, dtype=np.float32)[:, None],
            anchor=np.asarray(anchors, dtype=np.float32)[:, None],
        )
        (output_dir / f"{split}.metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        packed[split] = output_path
    return packed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feature_manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("global_tensors"))
    args = parser.parse_args()
    rows = read_feature_manifest(args.feature_manifest)
    outputs = pack_tensors(rows, manifest_dir=args.feature_manifest.parent, output_dir=args.output_dir)
    print(json.dumps({split: str(path) for split, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
