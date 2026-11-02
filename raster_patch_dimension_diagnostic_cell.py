# ======================= RASTER PATCH-DIMENSION DIAGNOSTIC =======================
# Safe by default: inspect Zarr metadata only; do not rebuild caches or read pixel data.
import json
from pathlib import Path
# Change to True only if you also want an exact, sequential audit of cached raster indices.
RASTER_DIAG_SCAN_COMPACT_INDICES = False
RASTER_DIAG_CACHE_CHUNK_ROWS = 1_000_000

import re as _rpd_re
from collections import defaultdict as _rpd_defaultdict

_RPD_HINT_WORDS = ("patch", "size", "radius", "window", "buffer", "pixel", "resol", "shape", "grid", "crop", "transform")

def _rpd_attrs(obj):
    """Read small Zarr attrs safely; show only likely patch/grid metadata."""
    try:
        attrs = obj.attrs.asdict() if hasattr(obj.attrs, "asdict") else dict(obj.attrs)
    except Exception:
        return {}
    return {str(k): v for k, v in attrs.items()
            if any(word in str(k).lower() for word in _RPD_HINT_WORDS)}

def _rpd_year_paths():
    modis, s1 = {}, {}
    if Path(MODIS_ROOT).exists():
        for p in Path(MODIS_ROOT).glob("*/patches.zarr"):
            if p.parent.name.isdigit():
                modis[int(p.parent.name)] = p
    if Path(S1_ROOT).exists():
        for p in Path(S1_ROOT).glob("s1_nearest_valid_*.zarr"):
            m = _rpd_re.search(r"(20\d{2})", p.stem)
            if m:
                s1[int(m.group(1))] = p
    return modis, s1

def diagnose_raster_patch_dimensions():
    print("=" * 96)
    print("RASTER PATCH DIMENSION DIAGNOSTIC (metadata-only; no raster pixels loaded)")
    print("MODIS root:", MODIS_ROOT)
    print("S1 root:   ", S1_ROOT)
    modis_paths, s1_paths = _rpd_year_paths()
    years = sorted(set(modis_paths) | set(s1_paths))
    if not years:
        print("No year stores found. Check MODIS_ROOT and S1_ROOT; the paths may not be mounted on this machine.")
        return

    rows = []
    shapes_by_family = _rpd_defaultdict(lambda: _rpd_defaultdict(list))
    problems = []
    source_rows = {}

    for year in years:
        print(f"\n--- YEAR {year} ---")
        mg = sg = None
        if year not in modis_paths:
            print("  MISSING MODIS store:", Path(MODIS_ROOT) / str(year) / "patches.zarr")
        else:
            try:
                mg = open_group(modis_paths[year])
                attrs = _rpd_attrs(mg)
                print("  MODIS store:", modis_paths[year])
                if attrs:
                    print("  MODIS group attrs (patch/grid-related):", attrs)
                for name in ("evi", "lst"):
                    if name not in mg:
                        problems.append(f"{year} MODIS: missing array {name}")
                        print(f"  MODIS {name}: MISSING")
                        continue
                    arr = mg[name]
                    shape = tuple(int(x) for x in arr.shape)
                    chunks = tuple(int(x) for x in arr.chunks) if getattr(arr, "chunks", None) is not None else None
                    spatial = shape[-2:] if len(shape) >= 2 else shape
                    record = {"year": year, "store": "MODIS", "array": name,
                              "shape": shape, "spatial": spatial,
                              "chunks": chunks, "dtype": str(getattr(arr, "dtype", "?"))}
                    rows.append(record)
                    shapes_by_family[f"MODIS/{name}"][spatial].append(year)
                    if len(shape) != 3:
                        problems.append(f"{year} MODIS/{name}: expected (N,H,W), got {shape}")
                    else:
                        source_rows[(year, "modis", name)] = shape[0]
                    print(f"  MODIS {name:>4}: full_shape={shape}, patch_hw={spatial}, chunks={chunks}, dtype={record['dtype']}")
                    aattrs = _rpd_attrs(arr)
                    if aattrs:
                        print(f"           array attrs: {aattrs}")
                if "evi" in mg and "lst" in mg:
                    es, ls = tuple(mg["evi"].shape), tuple(mg["lst"].shape)
                    if len(es) >= 3 and len(ls) >= 3 and es[-2:] != ls[-2:]:
                        problems.append(f"{year} MODIS EVI/LST spatial shapes differ: {es[-2:]} vs {ls[-2:]}")
                    if es and ls and es[0] != ls[0]:
                        problems.append(f"{year} MODIS EVI/LST row counts differ: {es[0]} vs {ls[0]}")
            except Exception as exc:
                problems.append(f"{year} MODIS open/inspect failed: {type(exc).__name__}: {exc}")
                print(f"  MODIS inspect FAILED: {type(exc).__name__}: {exc}")

        if year not in s1_paths:
            print("  MISSING Sentinel-1 store:", Path(S1_ROOT) / f"s1_nearest_valid_{year}.zarr")
        else:
            try:
                sg = open_group(s1_paths[year])
                attrs = _rpd_attrs(sg)
                print("  S1 store:   ", s1_paths[year])
                if attrs:
                    print("  S1 group attrs (patch/grid-related):", attrs)
                s1_names = list(globals().get("S1_ARRAYS", globals().get("S1_CACHE_ARRAYS", [])))
                for name in s1_names:
                    if name not in sg:
                        problems.append(f"{year} S1: missing array {name}")
                        print(f"  S1 {name}: MISSING")
                        continue
                    arr = sg[name]
                    shape = tuple(int(x) for x in arr.shape)
                    chunks = tuple(int(x) for x in arr.chunks) if getattr(arr, "chunks", None) is not None else None
                    spatial = shape[-2:] if len(shape) >= 2 else shape
                    record = {"year": year, "store": "S1", "array": name,
                              "shape": shape, "spatial": spatial,
                              "chunks": chunks, "dtype": str(getattr(arr, "dtype", "?"))}
                    rows.append(record)
                    shapes_by_family[f"S1/{name}"][spatial].append(year)
                    if len(shape) != 3:
                        problems.append(f"{year} S1/{name}: expected (N,H,W), got {shape}")
                    else:
                        source_rows[(year, "s1", name)] = shape[0]
                    print(f"  S1   {name:>8}: full_shape={shape}, patch_hw={spatial}, chunks={chunks}, dtype={record['dtype']}")
                    aattrs = _rpd_attrs(arr)
                    if aattrs:
                        print(f"           array attrs: {aattrs}")
                # All S1 channels should normally share the same sample count and spatial grid.
                found = [(name, tuple(sg[name].shape)) for name in s1_names if name in sg]
                row_counts = {sh[0] for _, sh in found if len(sh) >= 1}
                spatial_shapes = {sh[-2:] for _, sh in found if len(sh) >= 3}
                if len(row_counts) > 1:
                    problems.append(f"{year} S1 arrays have inconsistent row counts: {found}")
                if len(spatial_shapes) > 1:
                    problems.append(f"{year} S1 arrays have inconsistent spatial sizes: {found}")
            except Exception as exc:
                problems.append(f"{year} S1 open/inspect failed: {type(exc).__name__}: {exc}")
                print(f"  S1 inspect FAILED: {type(exc).__name__}: {exc}")

    print("\n" + "=" * 96)
    print("CROSS-YEAR SHAPE SUMMARY")
    for family in sorted(shapes_by_family):
        print(f"  {family}:")
        for shape, ys in sorted(shapes_by_family[family].items(), key=lambda kv: str(kv[0])):
            uniq_years = sorted(set(ys))
            print(f"    patch_hw={shape}: years={uniq_years}")
        if len(shapes_by_family[family]) > 1:
            print("    !! Native dimensions vary across years for this array.")

    # Compare all arrays across MODIS and S1 per year without conflating their resolutions.
    print("\nINTERPRETATION GUIDE")
    print("  • If dimensions differ between years in the source-store table, the mismatch is already present in the saved Zarrs.")
    print("  • If MODIS EVI and LST differ within one year, their source arrays were not built on the same patch grid.")
    print("  • If S1 channels differ within one year, that store is internally inconsistent or has mixed schemas.")
    print("  • If every source store reports one consistent shape but training still reports 13x13 vs 27x27, inspect which collator/class definition is active and whether a legacy/mmap path is being used.")

    if problems:
        print("\nPOTENTIAL SOURCE/SCHEMA PROBLEMS:")
        for item in problems:
            print("  -", item)
    else:
        print("\nNo missing arrays or within-year shape inconsistencies detected from metadata.")

    if RASTER_DIAG_SCAN_COMPACT_INDICES:
        print("\n" + "=" * 96)
        print("COMPACT CACHE INDEX AUDIT (sequential scan; may take time on very large caches)")
        root = Path(MODEL_READY_CACHE_DIR)
        candidates = []
        active = globals().get("MODEL_READY_CACHE_PATHS", {}).get(("environment_only", "train"))
        if active and Path(active).exists() and (Path(active) / "metadata.json").exists():
            candidates.append(Path(active) / "metadata.json")
        if root.exists():
            candidates.extend(sorted(root.glob("environment_only_train_*/metadata.json")))
        # Deduplicate candidates and use the first completed current-format cache.
        chosen = None
        seen = set()
        for mp in candidates:
            if str(mp.resolve()) in seen:
                continue
            seen.add(str(mp.resolve()))
            try:
                meta = json.loads(mp.read_text(encoding="utf-8"))
                if meta.get("cache_kind") == globals().get("_COMPACT_CACHE_KIND", "compact_metadata_parallel_rowgroup_v2") and all(
                    k in meta.get("arrays", {}) for k in ("year", "modis_idx", "s1_idx", "sample_valid")
                ):
                    chosen = (mp.parent, meta)
                    break
            except Exception:
                continue
        if chosen is None:
            print("No completed environment_only/train compact cache found; skip index audit.")
        else:
            cache_dir, meta = chosen
            n = int(meta["n_rows"])
            def _rpd_memmap(key):
                spec = meta["arrays"][key]
                return np.memmap(cache_dir / spec["file"], mode="r", dtype=np.dtype(spec["dtype"]), shape=(n,))
            years_arr = _rpd_memmap("year")
            modis_idx_arr = _rpd_memmap("modis_idx")
            s1_idx_arr = _rpd_memmap("s1_idx")
            valid_arr = _rpd_memmap("sample_valid")
            stats = {}
            step = max(1, int(RASTER_DIAG_CACHE_CHUNK_ROWS))
            for start in range(0, n, step):
                end = min(n, start + step)
                yy = np.asarray(years_arr[start:end])
                mi = np.asarray(modis_idx_arr[start:end])
                si = np.asarray(s1_idx_arr[start:end])
                vv = np.asarray(valid_arr[start:end], dtype=bool)
                for year in np.unique(yy[vv]):
                    mask = vv & (yy == year)
                    y = int(year)
                    st = stats.setdefault(y, {"valid_rows": 0, "modis_min": None, "modis_max": None,
                                               "s1_min": None, "s1_max": None, "modis_negative": 0, "s1_negative": 0})
                    if not np.any(mask):
                        continue
                    mvals, svals = mi[mask], si[mask]
                    st["valid_rows"] += int(mask.sum())
                    st["modis_min"] = int(mvals.min()) if st["modis_min"] is None else min(st["modis_min"], int(mvals.min()))
                    st["modis_max"] = int(mvals.max()) if st["modis_max"] is None else max(st["modis_max"], int(mvals.max()))
                    st["s1_min"] = int(svals.min()) if st["s1_min"] is None else min(st["s1_min"], int(svals.min()))
                    st["s1_max"] = int(svals.max()) if st["s1_max"] is None else max(st["s1_max"], int(svals.max()))
                    st["modis_negative"] += int((mvals < 0).sum())
                    st["s1_negative"] += int((svals < 0).sum())
            print("Audited cache:", cache_dir, "rows=", f"{n:,}")
            for year, st in sorted(stats.items()):
                checks = []
                for modality, names, lo_key, hi_key in (
                    ("MODIS", ("evi", "lst"), "modis_min", "modis_max"),
                    ("S1", list(globals().get("S1_ARRAYS", [])), "s1_min", "s1_max"),
                ):
                    lengths = [source_rows.get((year, modality.lower() if modality == "S1" else "modis", name)) for name in names]
                    lengths = [int(x) for x in lengths if x is not None]
                    if lengths and st[hi_key] is not None and st[hi_key] >= min(lengths):
                        checks.append(f"{modality} max index {st[hi_key]} >= source rows {min(lengths)}")
                print(f"  {year}: n_valid={st['valid_rows']:,}, MODIS idx=[{st['modis_min']},{st['modis_max']}], "
                      f"S1 idx=[{st['s1_min']},{st['s1_max']}], negative_indices=(MODIS {st['modis_negative']}, S1 {st['s1_negative']})")
                for check in checks:
                    print("    !! OUT-OF-RANGE:", check)
            print("Index audit checks max indices against loaded source row counts. It does not read raster pixel values.")

# Run only when this cell is executed.
diagnose_raster_patch_dimensions()
