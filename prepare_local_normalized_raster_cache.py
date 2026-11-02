#!/usr/bin/env python3
"""Build normalized, task-mode-specific Global SM raster memmaps locally for Kaggle.

Output layout (upload the directory contents as a Kaggle Dataset):
  normalized_raster_manifest.json
  location_history/modis/<year>/{evi,lst}.npy   float16 [rows, 2, H, W]
  location_history/s1/<year>/s1.npy              float16 [rows, 6, H, W]
  environment_only/modis/<year>/{evi,lst}.npy
  environment_only/s1/<year>/s1.npy

The values and validity-mask channels exactly follow the notebook's decode/normalization:
- EVI/LST fill/range filtering and physical scaling.
- Sentinel-1 finite-value masks.
- Per-task-mode training-only raster mean/std, clipped to [-10, 10].
- Channel order: EVI/LST = [normalized, mask]; S1 = [VV,VH,ratio, VV-mask,VH-mask,ratio-mask].

This moves all raster decoding, mask creation and z-score normalization off Kaggle. The
Kaggle notebook only memory-maps these arrays and performs vectorized batch indexing.
"""
from __future__ import annotations

import argparse
import hashlib
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
TASK_MODES = ("location_history", "environment_only")
S1_ARRAYS = ("vv", "vh", "vv_vh_ratio")
EVI_FILL_RAW, EVI_RAW_MIN, EVI_RAW_MAX, EVI_SCALE = -32768, -2000, 10000, 0.0001
LST_FILL_RAW, LST_RAW_MIN, LST_RAW_MAX, LST_SCALE = 0, 7500, 20000, 0.02
FORMAT_SCHEMA = 1
CACHE_TYPE = "global_sm_normalized_raster_numpy_memmap"


def open_group(path: Path):
    try:
        return zarr.open_group(str(path), mode="r")
    except Exception as exc:
        raise RuntimeError(f"Cannot open Zarr store {path}: {type(exc).__name__}: {exc}") from exc


def find_years(global_root: Path, run_dir: Path) -> list[int]:
    paths = [run_dir / "manifests" / m / f"{s}.parquet"
             for m in TASK_MODES for s in ("train", "test")]
    existing = [p for p in paths if p.is_file()]
    if existing and len(existing) != len(paths):
        missing = [str(p) for p in paths if not p.is_file()]
        raise FileNotFoundError("Manifest set is incomplete; missing: " + "; ".join(missing))
    years: set[int] = set()
    if existing:
        if pq is None:
            raise RuntimeError("pyarrow is required to read years from local manifests.")
        for p in existing:
            col = pq.read_table(p, columns=["year"]).column("year")
            years.update(int(x) for x in col.to_pylist() if x is not None)
    if years:
        return sorted(years)
    modis_root = global_root / "modis" / "evi_lst_patch_dataset"
    s1_root = global_root / "sentinel1" / "nearest_valid_overview_v1"
    modis_years = {int(p.name) for p in modis_root.iterdir()
                   if p.is_dir() and p.name.isdigit() and (p / "patches.zarr").exists()}
    s1_years = set()
    for p in s1_root.glob("s1_nearest_valid_*.zarr"):
        try:
            s1_years.add(int(p.name.removeprefix("s1_nearest_valid_").removesuffix(".zarr")))
        except ValueError:
            pass
    common = sorted(modis_years & s1_years)
    if not common:
        raise FileNotFoundError("No paired MODIS/Sentinel-1 source years found.")
    print("WARNING: local train/test manifests were not found; using every paired source year:", common)
    return common


def load_scalers(scaler_dir: Path) -> dict[str, dict[str, dict[str, float]]]:
    result = {}
    required = {"evi", "lst", *S1_ARRAYS}
    for mode in TASK_MODES:
        path = scaler_dir / f"{mode}_raster.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing locally fitted raster scaler: {path}. Use the scalers from the same training-assets bundle/run."
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not required.issubset(payload):
            raise ValueError(f"{path} is missing scaler keys: {sorted(required - set(payload))}")
        for key in required:
            mean, std = float(payload[key]["mean"]), float(payload[key]["std"])
            if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
                raise ValueError(f"Invalid scaler in {path} for {key}: mean={mean}, std={std}")
        result[mode] = payload
    return result


def scaler_hashes(scaler_dir: Path) -> dict[str, str]:
    out = {}
    for mode in TASK_MODES:
        p = scaler_dir / f"{mode}_raster.json"
        out[mode] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def source_groups(global_root: Path, year: int):
    modis_path = global_root / "modis" / "evi_lst_patch_dataset" / str(year) / "patches.zarr"
    s1_path = global_root / "sentinel1" / "nearest_valid_overview_v1" / f"s1_nearest_valid_{year}.zarr"
    mg, sg = open_group(modis_path), open_group(s1_path)
    for name in ("evi", "lst"):
        if name not in mg:
            raise KeyError(f"{name} missing from {modis_path}")
    for name in S1_ARRAYS:
        if name not in sg:
            raise KeyError(f"{name} missing from {s1_path}")
    shapes = {"evi": tuple(int(x) for x in mg["evi"].shape),
              "lst": tuple(int(x) for x in mg["lst"].shape)}
    if shapes["evi"] != shapes["lst"]:
        raise ValueError(f"MODIS EVI/LST shape mismatch in {year}: {shapes}")
    s1_shapes = {name: tuple(int(x) for x in sg[name].shape) for name in S1_ARRAYS}
    if len(set(s1_shapes.values())) != 1:
        raise ValueError(f"Sentinel-1 shape mismatch in {year}: {s1_shapes}")
    return mg, sg, shapes, s1_shapes[S1_ARRAYS[0]]


def output_specs(years: list[int], source_info: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    specs = []
    for mode in TASK_MODES:
        for year in years:
            src = source_info[year]
            for name in ("evi", "lst"):
                n, h, w = src["modis_shape"]
                shape = [n, 2, h, w]
                specs.append({"task_mode": mode, "year": year, "modality": "modis", "name": name,
                              "file": f"{mode}/modis/{year}/{name}.npy", "shape": shape,
                              "dtype": np.dtype(np.float16).str,
                              "nbytes": int(np.prod(shape, dtype=np.int64)) * 2})
            n, h, w = src["s1_shape"]
            shape = [n, 2 * len(S1_ARRAYS), h, w]
            specs.append({"task_mode": mode, "year": year, "modality": "s1", "name": "s1",
                          "file": f"{mode}/s1/{year}/s1.npy", "shape": shape,
                          "dtype": np.dtype(np.float16).str,
                          "nbytes": int(np.prod(shape, dtype=np.int64)) * 2})
    return specs


def valid_array(path: Path, spec: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        ok = list(arr.shape) == spec["shape"] and arr.dtype.str == spec["dtype"]
        del arr
        return ok
    except Exception:
        return False


def normalize_block(raw: np.ndarray, name: str, scaler: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(raw)
    if name == "evi":
        valid = np.isfinite(arr) & (arr != EVI_FILL_RAW) & (arr >= EVI_RAW_MIN) & (arr <= EVI_RAW_MAX)
        physical = arr.astype(np.float32, copy=False) * EVI_SCALE
    elif name == "lst":
        valid = np.isfinite(arr) & (arr != LST_FILL_RAW) & (arr >= LST_RAW_MIN) & (arr <= LST_RAW_MAX)
        physical = arr.astype(np.float32, copy=False) * LST_SCALE
    else:
        physical = np.asarray(arr, dtype=np.float32)
        valid = np.isfinite(physical)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(physical)
    mean, std = float(scaler["mean"]), float(scaler["std"])
    filled = np.where(valid, physical, mean).astype(np.float32, copy=False)
    norm = np.clip((filled - mean) / std, -10.0, 10.0)
    return norm.astype(np.float16, copy=False), valid.astype(np.float16, copy=False)


def atomic_memmap(path: Path, shape: tuple[int, ...]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".building")
    if tmp.exists():
        tmp.unlink()
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=shape)
    return tmp, arr


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-root", type=Path,
        default=Path(os.environ.get("UG_SM_GLOBAL_ROOT", str(Path.home() / "Code/UG_SM/Data/Global"))))
    parser.add_argument("--run-dir", type=Path, default=None,
        help="Local training run containing manifests/ and scalers/. Defaults to run_optuna_s1_filtered_global_r2_v1.")
    parser.add_argument("--scaler-dir", type=Path, default=None,
        help="Folder containing location_history_raster.json and environment_only_raster.json.")
    parser.add_argument("--output-dir", type=Path, default=None,
        help="Output directory. Defaults to <run-dir>/normalized_raster_mmap_cache_v1.")
    parser.add_argument("--rows-per-block", type=int, default=128,
        help="Zarr rows processed per block. Lower this if local RAM is constrained.")
    parser.add_argument("--disk-reserve-gib", type=float, default=3.0)
    parser.add_argument("--overwrite", action="store_true",
        help="Rebuild all arrays. Without this, matching completed arrays are reused.")
    parser.add_argument("--verify-only", action="store_true",
        help="Only validate a completed output folder; do not build.")
    args = parser.parse_args()

    global_root = args.global_root.expanduser().resolve()
    data_dir = global_root / "model_ready_global_v1"
    run_dir = args.run_dir.expanduser().resolve() if args.run_dir else (
        data_dir / "training_runs" / "run_optuna_s1_filtered_global_r2_v1")
    scaler_dir = args.scaler_dir.expanduser().resolve() if args.scaler_dir else run_dir / "scalers"
    out_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_dir / "normalized_raster_mmap_cache_v1"
    if not global_root.is_dir():
        raise FileNotFoundError(f"Global data root not found: {global_root}")
    if args.rows_per_block < 1:
        raise ValueError("--rows-per-block must be >= 1")

    years = find_years(global_root, run_dir)
    scalers = load_scalers(scaler_dir)
    hashes = scaler_hashes(scaler_dir)
    source_info = {}
    for year in years:
        mg, sg, modis_shape, s1_shape = source_groups(global_root, year)
        source_info[year] = {"modis_shape": modis_shape["evi"], "s1_shape": s1_shape,
                             "modis_chunks": getattr(mg["evi"], "chunks", None),
                             "s1_chunks": getattr(sg[S1_ARRAYS[0]], "chunks", None)}
        del mg, sg

    specs = output_specs(years, source_info)
    total_bytes = sum(s["nbytes"] for s in specs)
    build_signature = {
        "schema": FORMAT_SCHEMA, "cache_type": CACHE_TYPE, "years": years,
        "scaler_sha256": hashes,
        "decode": {"evi": [EVI_FILL_RAW, EVI_RAW_MIN, EVI_RAW_MAX, EVI_SCALE],
                   "lst": [LST_FILL_RAW, LST_RAW_MIN, LST_RAW_MAX, LST_SCALE],
                   "s1_valid": "finite_only"},
        "arrays": [{k: s[k] for k in ("task_mode", "year", "modality", "name", "shape", "dtype")} for s in specs],
    }
    signature_text = json.dumps(build_signature, sort_keys=True, separators=(",", ":"))
    signature_hash = hashlib.sha256(signature_text.encode()).hexdigest()
    out_dir.mkdir(parents=True, exist_ok=True)
    sig_path = out_dir / "normalized_raster_build_signature.json"
    manifest_path = out_dir / "normalized_raster_manifest.json"

    if sig_path.exists():
        old_sig = json.loads(sig_path.read_text(encoding="utf-8"))
        if old_sig.get("signature_sha256") != signature_hash and not args.overwrite:
            raise RuntimeError(
                "Existing normalized cache was built with different scaler/source settings. "
                "Use a new --output-dir or pass --overwrite; refusing to mix arrays from different versions."
            )
    elif any(out_dir.iterdir()) and not args.overwrite and not args.verify_only:
        # A non-empty folder without a signature could be an interrupted/old raw cache.
        # Require a new folder or an explicit overwrite to avoid accidental file reuse.
        raise RuntimeError(
            f"Output directory is non-empty but has no {sig_path.name}: {out_dir}. "
            "Use a new --output-dir or pass --overwrite after checking its contents."
        )

    missing = []
    for spec in specs:
        p = out_dir / spec["file"]
        if args.overwrite or not valid_array(p, spec):
            missing.append(spec)

    free = shutil.disk_usage(out_dir).free
    reserve = int(args.disk_reserve_gib * GIB)
    missing_bytes = sum(s["nbytes"] for s in missing)
    print(f"Global root: {global_root}\nRun directory: {run_dir}\nScalers: {scaler_dir}\nOutput: {out_dir}")
    print(f"Years: {years}; normalized arrays: {len(specs)}; expected total={total_bytes/GIB:.2f} GiB")
    print(f"Missing/rebuild arrays: {len(missing)} ({missing_bytes/GIB:.2f} GiB); free disk={free/GIB:.2f} GiB; reserve={args.disk_reserve_gib:.2f} GiB")
    print("Scaler SHA256:", hashes)

    if args.verify_only:
        invalid = [s["file"] for s in specs if not valid_array(out_dir / s["file"], s)]
        if invalid or not manifest_path.is_file():
            raise RuntimeError("VERIFY FAILED; missing/invalid arrays or manifest: " + ", ".join(invalid[:20]))
        print("VERIFY PASSED:", manifest_path)
        return 0

    if missing and free < int(missing_bytes * 1.05) + reserve:
        raise RuntimeError(f"Insufficient local disk: need {missing_bytes/GIB:.2f} GiB plus reserve; free={free/GIB:.2f} GiB")

    # Atomic signature written before builds; final .npy files are only renamed into place after flush.
    sig_tmp = sig_path.with_suffix(".json.building")
    sig_tmp.write_text(json.dumps({"signature_sha256": signature_hash, "signature": build_signature}, indent=2) + "\n")
    os.replace(sig_tmp, sig_path)

    block_rows = max(1, int(args.rows_per_block))
    built = 0
    for year in years:
        mg, sg, modis_shape, s1_shape = source_groups(global_root, year)
        n_rows = int(modis_shape["evi"][0])
        if int(s1_shape[0]) != n_rows:
            raise ValueError(f"MODIS/S1 first-axis length mismatch for {year}: {n_rows} vs {s1_shape[0]}")

        # Open/reuse outputs for every mode. Reused completed arrays are never rewritten.
        outputs: dict[str, dict[str, Any]] = {}
        for mode in TASK_MODES:
            outputs[mode] = {}
            for name in ("evi", "lst"):
                spec = next(s for s in specs if s["task_mode"] == mode and s["year"] == year and s["name"] == name)
                target = out_dir / spec["file"]
                if valid_array(target, spec) and not args.overwrite:
                    continue
                tmp, dst = atomic_memmap(target, tuple(spec["shape"]))
                outputs[mode][name] = (tmp, target, dst, spec)
            spec = next(s for s in specs if s["task_mode"] == mode and s["year"] == year and s["name"] == "s1")
            target = out_dir / spec["file"]
            if valid_array(target, spec) and not args.overwrite:
                continue
            tmp, dst = atomic_memmap(target, tuple(spec["shape"]))
            outputs[mode]["s1"] = (tmp, target, dst, spec)

        active_modes = [m for m in TASK_MODES if outputs[m]]
        if not active_modes:
            print(f"[{year}] all normalized arrays already valid; skipping.")
            del mg, sg, outputs
            continue

        starts = range(0, n_rows, block_rows)
        progress = tqdm(starts, desc=f"Normalize rasters {year}", unit="block", leave=False)
        for start in progress:
            end = min(n_rows, start + block_rows)
            raw_evi = np.asarray(mg["evi"][start:end])
            raw_lst = np.asarray(mg["lst"][start:end])
            raw_s1 = {name: np.asarray(sg[name][start:end]) for name in S1_ARRAYS}
            for mode in active_modes:
                scaler = scalers[mode]
                for name, raw in (("evi", raw_evi), ("lst", raw_lst)):
                    if name not in outputs[mode]:
                        continue
                    norm, mask = normalize_block(raw, name, scaler[name])
                    packed = np.stack((norm, mask), axis=1).astype(np.float16, copy=False)
                    outputs[mode][name][2][start:end] = packed
                if "s1" in outputs[mode]:
                    # Pack value channels first, then the matching validity-mask channels.
                    normalized, masks = [], []
                    for name in S1_ARRAYS:
                        norm, mask = normalize_block(raw_s1[name], name, scaler[name])
                        normalized.append(norm)
                        masks.append(mask)
                    packed_s1 = np.concatenate((np.stack(normalized, axis=1),
                                                 np.stack(masks, axis=1)), axis=1).astype(np.float16, copy=False)
                    outputs[mode]["s1"][2][start:end] = packed_s1
            del raw_evi, raw_lst, raw_s1

        for mode in active_modes:
            for key, (tmp, target, dst, spec) in outputs[mode].items():
                dst.flush()
                del dst
                os.replace(tmp, target)
                built += 1
                print(f"Built {built}/{len(missing)}: {spec['file']} ({spec['nbytes']/GIB:.2f} GiB)")
        del outputs, mg, sg

    # Final strict validation and manifest publication.
    entries = []
    for spec in specs:
        p = out_dir / spec["file"]
        if not valid_array(p, spec):
            raise RuntimeError(f"Final validation failed: {p}")
        entries.append({k: spec[k] for k in ("task_mode", "year", "modality", "name", "file", "shape", "dtype", "nbytes")})
    manifest = {
        "schema": FORMAT_SCHEMA, "cache_type": CACHE_TYPE,
        "created_unix": time.time(), "signature_sha256": signature_hash,
        "years": years, "task_modes": list(TASK_MODES), "scaler_sha256": hashes,
        "arrays": entries, "total_uncompressed_bytes": total_bytes,
        "notes": [
            "All arrays are normalized locally using the supplied training-only scaler JSON files.",
            "Float16 packed channels preserve model-input values and validity masks.",
            "Kaggle should open arrays read-only with np.load(mmap_mode='r').",
            "No Kaggle-side Zarr decoding, normalization, or raster preprocessing is required.",
        ],
    }
    tmp_manifest = manifest_path.with_suffix(".json.building")
    tmp_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_manifest, manifest_path)
    actual = sum((out_dir / e["file"]).stat().st_size for e in entries)
    print("\nREADY FOR KAGGLE UPLOAD")
    print(f"Validated arrays: {len(entries)}")
    print(f"Payload size: {actual/GIB:.2f} GiB")
    print(f"Cache directory: {out_dir}")
    print(f"Manifest: {manifest_path}")
    print("Upload the CONTENTS of this folder as a separate Kaggle Dataset (do not zip it for notebook runtime).")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
