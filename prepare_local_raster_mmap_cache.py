#!/usr/bin/env python3
"""Build/verify the Global SM raw-raster NumPy memmap mirror locally for Kaggle.

The output folder is the exact directory Kaggle's local-speed training notebook mounts
as RAM_CACHE_DIR. It contains:
  raster_mmap_manifest.json
  modis/<year>/{evi,lst}.npy
  s1/<year>/{vv,vh,vv_vh_ratio}.npy

It reuses valid existing .npy arrays and only builds missing/invalid arrays. Copying is
streamed by row, so the full ~17 GiB source does not need to fit in RAM.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import zarr

try:
    import pyarrow.parquet as pq
except Exception:
    pq = None

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(it, **kwargs):
        return it

GIB = 1024 ** 3
MODIS_ARRAYS = ("evi", "lst")
S1_ARRAYS = ("vv", "vh", "vv_vh_ratio")


def open_group(path: Path):
    # Same Zarr open style as the training notebook (works with its installed Zarr API).
    return zarr.open_group(str(path), mode="r")


def years_from_manifests(run_dir: Path) -> list[int]:
    manifest_paths = [
        run_dir / "manifests" / mode / f"{split}.parquet"
        for mode in ("location_history", "environment_only")
        for split in ("train", "test")
    ]
    existing = [p for p in manifest_paths if p.is_file()]
    if existing and len(existing) != len(manifest_paths):
        missing = [str(p) for p in manifest_paths if not p.is_file()]
        raise FileNotFoundError(
            "The local train/test manifest set is incomplete; refusing to build a partial raster mirror. "
            "Missing: " + "; ".join(missing)
        )
    if existing and pq is None:
        raise RuntimeError("pyarrow is required to extract required years from the local manifests.")
    years: set[int] = set()
    for path in existing:
        table = pq.read_table(path, columns=["year"])
        for value in table.column("year").to_pylist():
            if value is not None:
                years.add(int(value))
    if years:
        return sorted(years)
    return []


def discover_years(global_root: Path, run_dir: Path) -> list[int]:
    years = years_from_manifests(run_dir)
    if years:
        print(f"Using {len(years)} years required by local train/test manifests: {years[0]}–{years[-1]}")
        return years
    modis_root = global_root / "modis" / "evi_lst_patch_dataset"
    s1_root = global_root / "sentinel1" / "nearest_valid_overview_v1"
    modis_years = {
        int(p.name) for p in modis_root.iterdir()
        if p.is_dir() and p.name.isdigit() and (p / "patches.zarr").exists()
    }
    s1_years = set()
    for p in s1_root.glob("s1_nearest_valid_*.zarr"):
        try:
            s1_years.add(int(p.name.removeprefix("s1_nearest_valid_").removesuffix(".zarr")))
        except ValueError:
            pass
    years = sorted(modis_years & s1_years)
    if not years:
        raise FileNotFoundError(
            "Could not infer years. Run the local training notebook once to create manifests, "
            "or verify the MODIS/Sentinel-1 source directories."
        )
    print(f"No local manifests found; using all paired source years: {years[0]}–{years[-1]}")
    return years


def array_specs(global_root: Path, years: list[int]) -> list[dict[str, Any]]:
    modis_root = global_root / "modis" / "evi_lst_patch_dataset"
    s1_root = global_root / "sentinel1" / "nearest_valid_overview_v1"
    specs: list[dict[str, Any]] = []
    for year in years:
        sources = {
            "modis": modis_root / str(year) / "patches.zarr",
            "s1": s1_root / f"s1_nearest_valid_{year}.zarr",
        }
        for modality, names in (("modis", MODIS_ARRAYS), ("s1", S1_ARRAYS)):
            source = sources[modality]
            if not source.exists():
                raise FileNotFoundError(f"Missing source Zarr store: {source}")
            group = open_group(source)
            for name in names:
                if name not in group:
                    raise KeyError(f"Array {name!r} missing in {source}")
                arr = group[name]
                shape = tuple(int(x) for x in arr.shape)
                dtype = np.dtype(arr.dtype)
                specs.append({
                    "year": int(year), "modality": modality, "name": name,
                    "source": str(source), "shape": list(shape), "dtype": dtype.str,
                    "nbytes": int(np.prod(shape, dtype=np.int64)) * dtype.itemsize,
                })
    return specs


def valid_npy(path: Path, spec: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        return list(arr.shape) == spec["shape"] and arr.dtype.str == spec["dtype"]
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--global-root", type=Path,
        default=Path(os.environ.get("UG_SM_GLOBAL_ROOT", str(Path.home() / "Code/UG_SM/Data/Global"))),
        help="Local Global dataset root containing modis/, sentinel1/, and model_ready_global_v1/.",
    )
    parser.add_argument(
        "--run-dir", type=Path, default=None,
        help="Local training run dir containing manifests/. Defaults to the known local-speed run.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Cache output directory. Defaults to <run-dir>/zarr_array_mmap_cache.",
    )
    parser.add_argument("--copy-rows", type=int, default=0,
                        help="Rows per Zarr read. 0 uses at least 128 and otherwise the source chunk rows (max 4096).")
    parser.add_argument("--disk-reserve-gib", type=float, default=2.0,
                        help="Minimum free disk to preserve while creating missing arrays.")
    parser.add_argument("--verify-only", action="store_true",
                        help="Do not build. Verify all expected arrays and write/update the manifest only if complete.")
    args = parser.parse_args()

    global_root = args.global_root.expanduser().resolve()
    data_dir = global_root / "model_ready_global_v1"
    run_dir = args.run_dir.expanduser().resolve() if args.run_dir else (
        data_dir / "training_runs" / "run_optuna_s1_filtered_global_r2_v1"
    )
    out_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_dir / "zarr_array_mmap_cache"

    if not global_root.is_dir():
        raise FileNotFoundError(f"Global root not found: {global_root}")
    years = discover_years(global_root, run_dir)
    specs = array_specs(global_root, years)
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    for spec in specs:
        target = out_dir / spec["modality"] / str(spec["year"]) / f"{spec['name']}.npy"
        spec["file"] = str(target.relative_to(out_dir))
        if not valid_npy(target, spec):
            missing.append((target, spec))

    total_bytes = sum(s["nbytes"] for s in specs)
    missing_bytes = sum(s["nbytes"] for _, s in missing)
    free = shutil.disk_usage(out_dir).free
    reserve = int(args.disk_reserve_gib * GIB)
    print(f"Global root: {global_root}")
    print(f"Local run:   {run_dir}")
    print(f"Output:      {out_dir}")
    print(f"Years:       {years}")
    print(f"Arrays:      {len(specs)}; expected size={total_bytes/GIB:.2f} GiB; missing={len(missing)} ({missing_bytes/GIB:.2f} GiB)")
    print(f"Disk free:   {free/GIB:.2f} GiB; reserve={args.disk_reserve_gib:.2f} GiB")

    if args.verify_only and missing:
        print("VERIFY FAILED: some arrays are missing or have mismatched shapes/dtypes:")
        for path, spec in missing:
            print(f"  {path} expected shape={spec['shape']} dtype={spec['dtype']}")
        return 2
    if missing and free < missing_bytes + reserve:
        raise RuntimeError(
            f"Insufficient local disk space: need about {missing_bytes/GIB:.2f} GiB plus "
            f"{args.disk_reserve_gib:.2f} GiB reserve, have {free/GIB:.2f} GiB."
        )
    if args.verify_only and not missing:
        print("All expected arrays are present and valid.")

    built = 0
    for target, spec in missing:
        if args.verify_only:
            break
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".building")
        source = Path(spec["source"])
        src_arr = open_group(source)[spec["name"]]
        dst = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.dtype(spec["dtype"]), shape=tuple(spec["shape"]))
        if len(spec["shape"]) == 0:
            dst[...] = src_arr[()]
        else:
            chunks = getattr(src_arr, "chunks", None)
            inferred = int(chunks[0]) if chunks and chunks[0] else 128
            rows = int(args.copy_rows) if args.copy_rows > 0 else max(128, min(inferred, 4096))
            starts = range(0, spec["shape"][0], rows)
            label = f"{spec['modality']} {spec['year']} {spec['name']}"
            for start in tqdm(starts, desc=label, unit="block", leave=False):
                end = min(spec["shape"][0], start + rows)
                # Stream one row slab at a time: peak temporary RAM is bounded by slab size.
                dst[start:end] = src_arr[start:end]
        dst.flush()
        del dst, src_arr
        os.replace(tmp, target)
        built += 1
        print(f"Built {built}/{len(missing)}: {spec['file']} ({spec['nbytes']/GIB:.3f} GiB)")

    # Final full validation before making this directory uploadable.
    invalid = [(target, spec) for target, spec in missing if not valid_npy(target, spec)]
    all_specs = []
    for spec in specs:
        target = out_dir / spec["file"]
        if not valid_npy(target, spec):
            invalid.append((target, spec))
        all_specs.append({
            "year": spec["year"], "modality": spec["modality"], "name": spec["name"],
            "file": spec["file"], "shape": spec["shape"], "dtype": spec["dtype"],
            "nbytes": spec["nbytes"],
        })
    if invalid:
        raise RuntimeError("Final validation failed for one or more arrays: " + ", ".join(str(x[0]) for x in invalid[:8]))

    manifest = {
        "schema": 1,
        "cache_type": "global_sm_raw_zarr_numpy_memmap",
        "created_unix": time.time(),
        "global_root_at_build": str(global_root),
        "years": years,
        "arrays": all_specs,
        "total_uncompressed_bytes": total_bytes,
        "notes": [
            "Raw encoded Zarr array values copied exactly; no raster normalization or scaler fitting is performed here.",
            "Open .npy files read-only with np.load(mmap_mode='r').",
            "Contains only years represented in local train/test manifests when those manifests are available.",
        ],
    }
    manifest_path = out_dir / "raster_mmap_manifest.json"
    tmp_manifest = manifest_path.with_suffix(".json.building")
    tmp_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_manifest, manifest_path)
    actual_bytes = sum((out_dir / s["file"]).stat().st_size for s in all_specs)
    print("\nREADY FOR KAGGLE UPLOAD")
    print(f"Validated arrays: {len(all_specs)}")
    print(f"Payload size:     {actual_bytes/GIB:.2f} GiB")
    print(f"Cache directory:  {out_dir}")
    print(f"Manifest:         {manifest_path}")
    print("Upload the CONTENTS of this directory as a separate Kaggle Dataset; do not zip/unzip it inside the training notebook.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
