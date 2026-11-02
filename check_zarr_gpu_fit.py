#!/usr/bin/env python3
"""Read-only Zarr size and GPU-fit estimator for Global MultiModalSoilNet.

Reports on-disk bytes, logical/uncompressed array bytes, and estimates for the
native-grid model input cache (normalized float32 values + validity masks).
It does not read raster pixels or modify any files.

Run: python check_zarr_gpu_fit.py
Optional: python check_zarr_gpu_fit.py --root /path/to/Global
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

try:
    import zarr
except ImportError:
    raise SystemExit("Missing dependency: install zarr in this environment (pip install zarr).")

# Active model inputs in the current training notebook. HH/HV are deliberately omitted.
MODIS_ARRAYS = ("evi", "lst")
S1_ACTIVE_ARRAYS = ("vv", "vh", "vv_vh_ratio")
S1_ALL_ARRAYS = ("vv", "vh", "hh", "hv", "vv_vh_ratio")

GiB = 1024 ** 3
MiB = 1024 ** 2


def fmt_bytes(n: int | float) -> str:
    n = float(n)
    if n >= GiB:
        return f"{n / GiB:.3f} GiB"
    if n >= MiB:
        return f"{n / MiB:.2f} MiB"
    if n >= 1024:
        return f"{n / 1024:.2f} KiB"
    return f"{n:.0f} B"


def directory_bytes(path: Path) -> int:
    """Sum file sizes without following symlinks."""
    total = 0
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    for f in path.rglob("*"):
        try:
            if f.is_file() and not f.is_symlink():
                total += f.stat().st_size
        except OSError:
            pass
    return total


def open_group(path: Path):
    # mode='r' makes the diagnostic read-only; Zarr array metadata is lazy.
    return zarr.open_group(str(path), mode="r")


def iter_arrays(group, prefix=""):
    """Recursively enumerate Zarr arrays, compatible with Zarr v2/v3 group APIs."""
    try:
        for name in group.array_keys():
            arr = group[name]
            full = f"{prefix}/{name}" if prefix else str(name)
            yield full, arr
    except Exception:
        pass
    try:
        group_names = list(group.group_keys())
    except Exception:
        group_names = []
    for name in group_names:
        try:
            child = group[name]
            child_prefix = f"{prefix}/{name}" if prefix else str(name)
            yield from iter_arrays(child, child_prefix)
        except Exception as exc:
            print(f"WARNING: couldn't inspect subgroup {prefix}/{name}: {exc}")


def dtype_bytes(arr) -> int:
    try:
        return int(arr.dtype.itemsize)
    except Exception:
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path.home() / "Code" / "UG_SM" / "Data" / "Global",
        help="Global data root (default matches the training notebook)",
    )
    parser.add_argument(
        "--reserve-gib", type=float, default=1.5,
        help="VRAM to leave free for model, activations, optimizer, CUDA context and workspace (default: 1.5 GiB)",
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    modis_root = root / "modis" / "evi_lst_patch_dataset"
    s1_root = root / "sentinel1" / "nearest_valid_overview_v1"

    if not modis_root.exists() and not s1_root.exists():
        raise SystemExit(
            f"Neither raster root exists.\nMODIS: {modis_root}\nS1: {s1_root}\n"
            "Pass --root with the correct Global directory."
        )

    modis_paths = {
        int(p.parent.name): p
        for p in modis_root.glob("*/patches.zarr")
        if p.parent.name.isdigit()
    } if modis_root.exists() else {}
    s1_paths = {}
    if s1_root.exists():
        import re
        for p in s1_root.glob("s1_nearest_valid_*.zarr"):
            m = re.search(r"(20\d{2})", p.stem)
            if m:
                s1_paths[int(m.group(1))] = p

    years = sorted(set(modis_paths) | set(s1_paths))
    if not years:
        raise SystemExit(f"No expected Zarr stores found under:\n{modis_root}\n{s1_root}")

    print("=" * 92)
    print("ZARR SIZE / GPU-FIT REPORT (metadata-only; no pixel arrays loaded)")
    print("Global root:", root)
    print("MODIS root:", modis_root)
    print("S1 root:   ", s1_root)
    print("Active Sentinel-1 inputs:", ", ".join(S1_ACTIVE_ARRAYS))
    print("HH/HV included in all-array accounting only, not in model-input estimates.")

    total_disk_all = total_logical_all = 0
    total_disk_active = total_logical_active = 0
    total_native_float32 = total_model_cache = 0
    table = []

    for year in years:
        print(f"\n--- {year} ---")
        for family, store_path, desired in (
            ("MODIS", modis_paths.get(year), MODIS_ARRAYS),
            ("S1", s1_paths.get(year), S1_ALL_ARRAYS),
        ):
            if store_path is None:
                print(f"{family}: MISSING STORE")
                continue
            on_disk = directory_bytes(store_path)
            try:
                group = open_group(store_path)
            except Exception as exc:
                print(f"{family}: could not open {store_path}: {type(exc).__name__}: {exc}")
                continue
            items = list(iter_arrays(group))
            by_name = {name.split("/")[-1]: arr for name, arr in items}
            logical_family = 0
            logical_active_family = 0
            native_f32_family = 0
            model_cache_family = 0
            print(f"{family} store: {store_path}")
            print(f"  on disk (compressed + metadata): {fmt_bytes(on_disk)}")
            for name, arr in items:
                try:
                    shape = tuple(int(v) for v in arr.shape)
                    logical = math.prod(shape) * dtype_bytes(arr)
                    chunks = tuple(int(v) for v in arr.chunks) if getattr(arr, "chunks", None) is not None else None
                    dtype = str(arr.dtype)
                except Exception as exc:
                    print(f"  {name}: metadata error: {exc}")
                    continue
                active = name.split("/")[-1] in desired
                logical_family += logical
                if active:
                    logical_active_family += logical
                    # Estimate source values converted to float32, without an added validity mask.
                    native_f32_family += math.prod(shape) * 4
                    # Model-ready normalized values + one validity-mask channel per source band.
                    # EVI/LST each supply [normalized value, mask]. S1 has 3 active bands, each [value, mask].
                    n, *spatial = shape if len(shape) >= 1 else (0,)
                    if len(shape) >= 3:
                        h, w = shape[-2], shape[-1]
                        model_cache_family += n * h * w * 4 * 2
                        if family == "S1":
                            # only the active 3 bands count in this estimate, even if HH/HV also exist
                            pass
                    print(f"  {'ACTIVE' if active else 'other':6} {name:16} shape={shape} dtype={dtype:10} chunks={chunks} logical={fmt_bytes(logical)}")
                else:
                    print(f"  {'other':6} {name:16} shape={shape} dtype={dtype:10} chunks={chunks} logical={fmt_bytes(logical)}")
            if family == "S1":
                # Remove model-ready memory assigned to HH/HV by the generic loop above; calculate only active arrays.
                model_cache_family = 0
                for name in S1_ACTIVE_ARRAYS:
                    arr = by_name.get(name)
                    if arr is None:
                        print(f"  WARNING: active model array {name!r} missing")
                        continue
                    shape = tuple(int(v) for v in arr.shape)
                    if len(shape) >= 3:
                        model_cache_family += math.prod(shape[:-2]) * shape[-2] * shape[-1] * 4 * 2
                # compute active float32/native exactly (exclude HH/HV)
                native_f32_family = sum(math.prod(tuple(map(int, by_name[n].shape))) * 4
                                        for n in S1_ACTIVE_ARRAYS if n in by_name)
                logical_active_family = sum(math.prod(tuple(map(int, by_name[n].shape))) * dtype_bytes(by_name[n])
                                            for n in S1_ACTIVE_ARRAYS if n in by_name)
            # Disk size is per store and includes all arrays; active-only on-disk size cannot safely be attributed by array.
            total_disk_all += on_disk
            total_logical_all += logical_family
            total_logical_active += logical_active_family
            total_native_float32 += native_f32_family
            total_model_cache += model_cache_family
            table.append({"year": year, "family": family, "store_disk_bytes": on_disk,
                          "all_arrays_logical_bytes": logical_family,
                          "active_arrays_logical_bytes": logical_active_family,
                          "active_float32_bytes": native_f32_family,
                          "model_value_plus_mask_bytes": model_cache_family})
            print(f"  all arrays logical total: {fmt_bytes(logical_family)}")
            print(f"  active arrays logical total: {fmt_bytes(logical_active_family)}")
            print(f"  active float32 total: {fmt_bytes(native_f32_family)}")
            print(f"  model-ready value+mask estimate: {fmt_bytes(model_cache_family)}")

    # Disk totals include inactive HH/HV arrays in the source store footprint.
    print("\n" + "=" * 92)
    print("TOTALS ACROSS FOUND STORES")
    print("Compressed/on-disk size (all arrays, includes metadata):", fmt_bytes(total_disk_all))
    print("Logical/uncompressed size (all arrays):                ", fmt_bytes(total_logical_all))
    print("Logical size of active source arrays only:             ", fmt_bytes(total_logical_active))
    print("Active source arrays if converted to float32:            ", fmt_bytes(total_native_float32))
    print("Per-unique-raster model-ready [value + mask] estimate:   ", fmt_bytes(total_model_cache))
    print("Note: model-ready estimate uses native per-store row counts, not duplicated sample-index rows.")

    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            free_b, total_b = torch.cuda.mem_get_info(device)
            reserve = max(0.0, args.reserve_gib) * GiB
            usable = max(0, free_b - reserve)
            print("\nGPU:", props.name)
            print("Total VRAM:", fmt_bytes(props.total_memory))
            print("Currently free VRAM:", fmt_bytes(free_b))
            print("Configured reserve:", fmt_bytes(reserve))
            print("Free VRAM after reserve:", fmt_bytes(usable))
            for title, needed in (
                ("active source arrays as float32", total_native_float32),
                ("normalized model-ready values + masks", total_model_cache),
            ):
                ratio = needed / usable if usable else float("inf")
                if needed <= usable:
                    verdict = "fits in remaining VRAM by estimate (still leave runtime headroom)"
                else:
                    verdict = "does NOT fit in remaining VRAM"
                print(f"{title}: {fmt_bytes(needed)}; {ratio:.1%} of post-reserve free VRAM -> {verdict}")
            if total_model_cache <= usable:
                print("\nConclusion: preloading the normalized unique-raster cache may be feasible, but test one modality/year first.")
            else:
                print("\nConclusion: do not preload the whole normalized raster cache onto this GPU; use CPU RAM/NVMe cache + pinned batches.")
        else:
            print("\nGPU: PyTorch reports no CUDA device in this environment.")
    except ImportError:
        print("\nGPU availability skipped: PyTorch is not installed in this environment.")

    try:
        import pandas as pd
        report_path = root / "zarr_gpu_fit_report.csv"
        pd.DataFrame(table).to_csv(report_path, index=False)
        print("\nPer-store CSV report:", report_path)
    except Exception as exc:
        print("Could not write CSV report:", exc)

    print("\nCaveats:")
    print("  • Zarr on-disk bytes are file sizes; logical bytes are shape × dtype itemsize, before compression.")
    print("  • Model-ready estimate is for unique stored raster rows. Dataset index rows may reference the same patch multiple times.")
    print("  • This excludes model parameters, activations, gradients, optimizer state, CUDA context, and temporary workspaces.")
    print("  • The estimate assumes 2 float32 channels (value + mask) for EVI/LST and active S1 bands; actual allocations may differ.")
    print("  • The script reads metadata only, not every pixel; it does not measure process RSS or GPU allocation after a real batch.")


if __name__ == "__main__":
    main()
