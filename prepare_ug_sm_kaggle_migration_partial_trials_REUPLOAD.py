#!/usr/bin/env python3
"""Safely clean one obsolete cache and stage v18's PREBUILT caches for Kaggle.

Default: dry-run only.
  python prepare_ug_sm_kaggle_migration_cached.py

Delete only the legacy zarr_array_mmap_cache and stage the existing caches:
  python prepare_ug_sm_kaggle_migration_cached.py --execute

Upload an already staged dataset (private unless explicitly changed in Kaggle UI):
  python prepare_ug_sm_kaggle_migration_cached.py --upload \\
      --kaggle-id YOUR_USERNAME/ug-sm-global-soil-moisture-data \\
      --license-name YOUR_LICENSE_ID [--upload-temp-dir /mnt/d/kaggle_tmp]

v18 cache policy:
- KEEP and stage zarr_active_input_mmap_v10: v18 consumes these prebuilt raw arrays.
- KEEP and stage model_ready_compact_cache_parallel_v2: all four task/split caches are reused.
- DELETE CANDIDATE: zarr_array_mmap_cache only; v18 does not reference this legacy path.
- Do NOT stage source MODIS/Sentinel-1 Zarr trees in cached-input mode; the verified raw cache
  contains the active arrays the model reads. The notebook uses cache_manifest.json to validate
  those arrays and fails fast rather than rebuilding caches on Kaggle.
- Keep manifests, scalers, filtered Parquet, and schema. Snapshot whichever v18 Optuna studies exist; a missing mode is created and searched on Kaggle.

Hard links avoid another local physical copy. They do NOT reduce Kaggle upload bytes.
"""
from __future__ import annotations
import argparse, errno, json, os, shutil, sqlite3, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

PROJECT = Path.home() / "Code" / "UG_SM"
GLOBAL = PROJECT / "Data" / "Global"
DATA = GLOBAL / "model_ready_global_v1"
RUN_NAME = "run_optuna_s1_filtered_global_r2_v1"
RUN = DATA / "training_runs" / RUN_NAME
RAW_CACHE = RUN / "zarr_active_input_mmap_v10"
LEGACY_CACHE = RUN / "zarr_array_mmap_cache"
COMPACT_CACHE = RUN / "model_ready_compact_cache_parallel_v2"
OUT = PROJECT / "kaggle_ug_sm_global_data_migration_v18_cached"
OLD_STAGE = PROJECT / "kaggle_ug_sm_global_data"
STAGE_MARKER = ".ug_sm_kaggle_stage_v18_cached"
GIB = 1024 ** 3
S1_ARRAYS = ("vv", "vh", "vv_vh_ratio")


def fmt(n: int) -> str: return f"{n / GIB:.2f} GiB"
def require(p: Path):
    if not p.exists(): raise FileNotFoundError(f"Required input missing: {p}")


def allocated_size(path: Path) -> int:
    if not path.exists(): return 0
    candidates = [path] if path.is_file() else list(path.rglob("*"))
    counts, stats = {}, {}
    for p in candidates:
        try:
            st = p.lstat()
            if p.is_file() and not p.is_symlink():
                k=(st.st_dev,st.st_ino); counts[k]=counts.get(k,0)+1; stats[k]=st
        except OSError: pass
    total=0
    for k,n in counts.items():
        st=stats[k]
        if st.st_nlink <= n: total += getattr(st,"st_blocks",0)*512 or st.st_size
    return total


def logical_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink()) if path.exists() else 0


def old_stage_is_ours(path: Path) -> bool:
    marker = path / ".ug_sm_kaggle_stage_v18"
    readme = path / "README_KAGGLE.txt"
    metadata = path / "dataset-metadata.json"
    if marker.is_file(): return True
    try:
        if readme.is_file() and "UG_SM Kaggle input bundle" in readme.read_text(errors="ignore"): return True
        if metadata.is_file() and json.loads(metadata.read_text()).get("title") == "UG SM Global Soil Moisture Model Data": return True
    except Exception: pass
    return False


def discover_years() -> list[int]:
    import pyarrow.parquet as pq
    files=sorted((RUN/"manifests").rglob("*.parquet"))
    if not files: raise FileNotFoundError(f"No split manifest Parquet files under {RUN/'manifests'}")
    years=set()
    for p in files:
        t=pq.read_table(p, columns=["year"])
        years.update(int(y) for y in t["year"].combine_chunks().to_pylist() if y is not None)
    if not years: raise RuntimeError("No years found in manifests; refusing to stage all data.")
    return sorted(years)


def validate_prebuilt_raw_cache(years: list[int]) -> tuple[int,int]:
    """Validate the exact array paths/shapes/dtypes expected by the Kaggle notebook."""
    require(RAW_CACHE / "cache_manifest.json")
    payload=json.loads((RAW_CACHE/"cache_manifest.json").read_text(encoding="utf-8"))
    if payload.get("storage") != "raw_native_dtype_npy_memmap":
        raise RuntimeError(f"Unexpected cache storage in {RAW_CACHE/'cache_manifest.json'}")
    if set(payload.get("active_s1_arrays",[])) != set(S1_ARRAYS):
        raise RuntimeError(f"Cached Sentinel channels {payload.get('active_s1_arrays')} do not match v18 {S1_ARRAYS}")
    records=payload.get("arrays",[])
    lookup={(int(x["year"]),str(x["modality"]),str(x["name"])):x for x in records}
    expected={(y,m,n) for y in years for m,names in (("modis",("evi","lst")),("s1",S1_ARRAYS)) for n in names}
    missing=sorted(expected-set(lookup))
    if missing: raise RuntimeError(f"Raw cache manifest does not cover all manifest years/channels; missing {missing[:12]}")
    total=0; checked=0
    for year,modality,name in sorted(expected):
        meta=lookup[(year,modality,name)]
        p=RAW_CACHE/modality/str(year)/f"{name}.npy"
        if not p.is_file(): raise FileNotFoundError(f"Prebuilt raw cache array missing: {p}")
        try: arr=np.load(p,mmap_mode="r",allow_pickle=False)
        except Exception as exc: raise RuntimeError(f"Cannot open cached array {p}: {exc}") from exc
        if tuple(arr.shape) != tuple(int(x) for x in meta["shape"]): raise RuntimeError(f"Shape mismatch for {p}: {arr.shape} vs {meta['shape']}")
        if arr.dtype.str != np.dtype(meta["dtype"]).str: raise RuntimeError(f"Dtype mismatch for {p}: {arr.dtype.str} vs {meta['dtype']}")
        if p.stat().st_size < int(meta.get("nbytes",0)): raise RuntimeError(f"Cache file is smaller than expected: {p}")
        total += p.stat().st_size; checked += 1
    return checked,total



def discover_optuna_studies() -> dict[str, tuple[Path, int]]:
    """Find v18 Optuna DBs actually created locally; missing modes are valid.

    A completed Optuna search writes its trials to SQLite before final model
    retraining. The full-training epoch/checkpoint is not required for reuse.
    """
    found: dict[str, tuple[Path, int]] = {}
    for mode in ("location_history", "environment_only"):
        db = RUN / f"optuna_filtered_s1_{mode}_multiscale_attnpool_gatedfusion_v18.sqlite3"
        if not db.is_file() or db.stat().st_size == 0:
            continue
        try:
            with sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30) as conn:
                check = conn.execute("PRAGMA quick_check").fetchone()
                if not check or check[0] != "ok":
                    raise RuntimeError(f"SQLite integrity check failed for {db}: {check}")
                tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if "trials" not in tables:
                    raise RuntimeError(f"Not a recognizable Optuna database (no trials table): {db}")
                n_trials = int(conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0])
        except sqlite3.Error as exc:
            raise RuntimeError(f"Could not inspect Optuna database {db}: {exc}") from exc
        found[mode] = (db, n_trials)
    return found


def validate_layout(years: list[int]) -> tuple[int,int,dict[str,tuple[Path,int]]]:
    for p in [DATA/"s1_valid_global_v1"/"sample_index_s1_valid.parquet", DATA/"s1_valid_global_v1"/"daily_targets_s1_valid.parquet",
              DATA/"feature_schema.json", RUN/"manifests"/"manifest_config.json", COMPACT_CACHE, RUN/"scalers"]: require(p)
    # Require all four compact cache folders and metadata files. Never rebuild on Kaggle.
    compact_dirs=[p for p in COMPACT_CACHE.iterdir() if p.is_dir() and (p/"metadata.json").is_file()]
    found={(p.name.split("_")[0],) for p in compact_dirs}
    names=[p.name for p in compact_dirs]
    for mode in ("location_history","environment_only"):
        for split in ("train","test"):
            if not any(n.startswith(f"{mode}_{split}_") for n in names):
                raise FileNotFoundError(f"Missing prebuilt compact cache for {mode}/{split} under {COMPACT_CACHE}")
    studies = discover_optuna_studies()
    return (*validate_prebuilt_raw_cache(years), studies)


def hardlink_file(src: Path,dst: Path):
    dst.parent.mkdir(parents=True,exist_ok=True)
    try: os.link(src,dst,follow_symlinks=False)
    except OSError as exc:
        reason=errno.errorcode.get(exc.errno,str(exc.errno))
        raise RuntimeError(f"Cannot hard-link {src} -> {dst} ({reason}: {exc}); no full-copy fallback used.") from exc


def stage_dir(src: Path,dst: Path,ignore=None):
    require(src); dst.mkdir(parents=True,exist_ok=False); count=0; logical=0
    for root,dirs,files in os.walk(src):
        rootp=Path(root); rel=rootp.relative_to(src)
        if ignore:
            ignored=set(ignore(rootp,dirs+files)); dirs[:]=[d for d in dirs if d not in ignored]; files=[f for f in files if f not in ignored]
        dr=dst/rel; dr.mkdir(parents=True,exist_ok=True)
        for name in files:
            a=rootp/name
            if a.is_symlink(): continue
            if name.endswith((".building",".tmp")): continue
            hardlink_file(a,dr/name); count+=1; logical+=a.stat().st_size
    print(f"Hard-linked {count:,} files ({fmt(logical)} logical): {src.name}")
    return count,logical


def sqlite_snapshot(src:Path,dst:Path):
    dst.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(f"file:{src}?mode=ro",uri=True,timeout=60) as a:
        with sqlite3.connect(dst,timeout=60) as b: a.backup(b)
    print(f"SQLite snapshot: {dst.name} ({dst.stat().st_size/1024**2:.2f} MiB)")


def build_stage(years:list[int],kaggle_id:str,include_zarr:bool=False,studies:dict[str,tuple[Path,int]]|None=None):
    if OUT.exists(): raise FileExistsError(f"Refusing to overwrite staging folder {OUT}")
    if os.stat(PROJECT).st_dev != os.stat(OUT.parent).st_dev: raise RuntimeError("Project and staging folder are not on same filesystem; hard links unavailable.")
    OUT.mkdir(parents=True); (OUT/STAGE_MARKER).write_text("UG_SM v18 cached-input Kaggle bundle\n")
    try:
        root=OUT/"Global"; sd=root/"model_ready_global_v1"; sr=sd/"training_runs"/RUN_NAME; sr.mkdir(parents=True)
        stage_dir(DATA/"s1_valid_global_v1",sd/"s1_valid_global_v1")
        hardlink_file(DATA/"feature_schema.json",sd/"feature_schema.json")
        stage_dir(RUN/"manifests",sr/"manifests")
        stage_dir(RUN/"scalers",sr/"scalers")
        def ignore_compact(_d,names): return [n for n in names if n.startswith("raster_order_") or n.endswith((".tmp",".building"))]
        stage_dir(COMPACT_CACHE,sr/"model_ready_compact_cache_parallel_v2",ignore_compact)
        # CRITICAL: include the existing ~12.5 GiB raw active-array cache; do not rebuild it on Kaggle.
        stage_dir(RAW_CACHE,sr/"zarr_active_input_mmap_v10",lambda _d,n:[x for x in n if x.endswith((".tmp",".building"))])
        # Source Zarr is deliberately omitted in cached-only mode; optional only for archival fallback.
        if include_zarr:
            modis_dst=root/"modis"/"evi_lst_patch_dataset"; s1_dst=root/"sentinel1"/"nearest_valid_overview_v1"
            for y in years:
                stage_dir(GLOBAL/"modis"/"evi_lst_patch_dataset"/str(y)/"patches.zarr",modis_dst/str(y)/"patches.zarr")
                stage_dir(GLOBAL/"sentinel1"/"nearest_valid_overview_v1"/f"s1_nearest_valid_{y}.zarr",s1_dst/f"s1_nearest_valid_{y}.zarr")
        studies = studies or {}
        staged_studies = {}
        for mode in ("location_history", "environment_only"):
            item = studies.get(mode)
            if item is None:
                print(f"No local {mode} v18 Optuna DB found; Kaggle will create/resume this study and fill trials up to the configured cap.")
                staged_studies[mode] = {"status": "not_present_locally", "trials": 0}
                continue
            src, n_trials = item
            sqlite_snapshot(src, sr/src.name)
            staged_studies[mode] = {"status": "snapshotted", "filename": src.name, "trials": n_trials}
        (sr/"optuna_studies_status.json").write_text(json.dumps(staged_studies, indent=2)+"\n")
        (OUT/"dataset-metadata.json").write_text(json.dumps({"title":"UG SM Global Soil Moisture Model Data","id":"REPLACE_WITH_USERNAME/ug-sm-global-soil-moisture-data"},indent=2)+"\n")
        (OUT/"README_KAGGLE.txt").write_text(
            "UG_SM v18 cached-input Kaggle bundle\n\n"
            "Attach this private Dataset to UG_SM_Kaggle_T4x2_model_v18_zarr3_prebuilt_caches.ipynb.\n"
            "Includes all four prebuilt compact task/split caches and zarr_active_input_mmap_v10 raw active arrays.\n"
            "Kaggle notebook uses strict reuse mode and refuses to rebuild caches.\n"
            "Only Optuna study databases found locally are included; any missing task study is created and filled on Kaggle up to the configured trial cap.\n"
            "Source MODIS/Sentinel-1 Zarr stores are omitted in default cached-only mode.\n"
            "Large files are hard-linked to the local source; do not modify them until upload completes.\n"
            "Set a Kaggle dataset license consistent with your data rights.\n")
    except Exception:
        print(f"Staging failed; partial generated folder left for inspection: {OUT}")
        raise
    files=[p for p in OUT.rglob("*") if p.is_file() and p.name!=STAGE_MARKER]
    logical=sum(p.stat().st_size for p in files); linked=sum(p.stat().st_nlink>1 for p in files)
    print("\nSTAGING COMPLETE")
    print("Folder:",OUT); print(f"Files: {len(files):,}; logical upload size: {fmt(logical)}; hard-linked: {linked:,}")
    print("Source Zarr stores:","included (optional archive mode)" if include_zarr else "omitted; prebuilt raw mirror is the training input")
    print("No source cache files were moved or copied locally; all large payload files are hard links.")


def upload(kaggle_id:str,license_name:str,temp_dir:Path|None):
    if not (OUT/STAGE_MARKER).is_file(): raise FileNotFoundError(f"No completed cached staging folder: {OUT}")
    if not kaggle_id or "/" not in kaggle_id: raise ValueError("Use --kaggle-id owner/dataset-slug")
    if not license_name: raise ValueError("Supply --license-name matching the rights for the data")
    if shutil.which("kaggle") is None: raise RuntimeError("Kaggle CLI is missing; install with python -m pip install -U kaggle")
    if not ((Path.home()/".kaggle"/"kaggle.json").is_file() or (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))):
        raise RuntimeError("Kaggle credentials not found (~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY).")
    metadata={"title":"UG SM Global Soil Moisture Model Data","id":kaggle_id,"licenses":[{"name":license_name}]}
    (OUT/"dataset-metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    bytes_data=sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file() and p.name not in {STAGE_MARKER,"dataset-metadata.json","README_KAGGLE.txt"})
    if temp_dir:
        temp_dir=temp_dir.expanduser().resolve(); temp_dir.mkdir(parents=True,exist_ok=True)
        os.environ["TMPDIR"]=str(temp_dir); tempfile.tempdir=str(temp_dir)
    else: temp_dir=Path(tempfile.gettempdir())
    free=shutil.disk_usage(temp_dir).free
    # Kaggle CLI's tar directory mode creates archive payload(s) before upload.
    # Require full-size temp headroom rather than risk filling the drive mid-upload.
    need=int(bytes_data*1.05)
    print(f"Dataset payload: {fmt(bytes_data)} | archive temp: {temp_dir} | free: {fmt(free)} | required safety headroom: {fmt(need)}")
    if free<need:
        raise RuntimeError(f"Not enough space for Kaggle CLI tar staging. Use --upload-temp-dir on a different drive with at least {fmt(need)} free; no upload was started.")
    cmd=["kaggle","datasets","create","-p",str(OUT),"--dir-mode","tar","--keep-tabular"]
    print("Uploading as private dataset (no --public):", " ".join(cmd)); subprocess.run(cmd,check=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    g=ap.add_mutually_exclusive_group(); g.add_argument("--execute",action="store_true",help="delete legacy cache and stage preserved caches"); g.add_argument("--upload",action="store_true",help="upload previously staged cached-input bundle")
    ap.add_argument("--kaggle-id",default=""); ap.add_argument("--license-name",default="")
    ap.add_argument("--upload-temp-dir",type=Path,default=None,help="drive/folder with space for temporary tar archive, e.g. /mnt/d/kaggle_tmp")
    ap.add_argument("--include-source-zarr",action="store_true",help="also include original source Zarr stores as archival fallback (adds ~19 GiB logical)")
    args=ap.parse_args()
    if args.upload: upload(args.kaggle_id,args.license_name,args.upload_temp_dir); return
    years=discover_years(); n_arrays,nbytes,studies=validate_layout(years)
    candidates=[]
    if LEGACY_CACHE.exists(): candidates.append((LEGACY_CACHE,allocated_size(LEGACY_CACHE)))
    print("\n=== DRY-RUN / SAFE CLEANUP ===")
    if candidates:
        for p,n in candidates: print(f"DELETE CANDIDATE (not used by v18): {p} | estimated actual reclaim: {fmt(n)}")
    else: print("No legacy zarr_array_mmap_cache found; nothing to remove.")
    print("KEEP + UPLOAD:",RAW_CACHE, f"| {n_arrays} arrays validated | {fmt(nbytes)}")
    print("KEEP + UPLOAD:",COMPACT_CACHE, f"| {fmt(logical_size(COMPACT_CACHE))} logical")
    print("KEEP:",RUN/"manifests",RUN/"scalers",RUN/"checkpoints",RUN/"mlflow_tracking")
    for mode in ("location_history", "environment_only"):
        if mode in studies:
            print(f"OPTUNA STUDY FOUND: {mode} | {studies[mode][1]} stored trials | {studies[mode][0].name}")
        else:
            print(f"OPTUNA STUDY MISSING: {mode} | Kaggle will create it and run up to 12 trials")
    print("KEEP:",DATA/"s1_valid_global_v1",DATA/"feature_schema.json")
    print("Source MODIS/S1 Zarr stores: omitted by default (prebuilt raw cache is complete).")
    print("Staging output:",OUT)
    if old_stage_is_ours(OLD_STAGE): print("Recognized previous generated stage (optional separate removal):",OLD_STAGE,fmt(allocated_size(OLD_STAGE)))
    if not args.execute:
        print("\nDRY RUN ONLY. Nothing was deleted or staged. Run --execute after checking this plan.")
        return
    if OUT.exists(): raise FileExistsError(f"Staging output exists; refusing to overwrite: {OUT}")
    phrase="DELETE ONLY THE LEGACY CACHE AND STAGE THE TWO PREBUILT CACHES"
    typed=input(f"\nThis removes ONLY {LEGACY_CACHE.name}; it PRESERVES {RAW_CACHE.name}.\nType exactly:\n{phrase}\n> ").strip()
    if typed!=phrase: print("Confirmation mismatch; no changes made."); return
    if candidates:
        # The only deletion target is the exact legacy folder in the known training-run path.
        print("Deleting legacy cache:",LEGACY_CACHE); shutil.rmtree(LEGACY_CACHE)
    if old_stage_is_ours(OLD_STAGE):
        phrase2="REMOVE PREVIOUS GENERATED STAGING FOLDER"
        if input(f"\nRemove previous generated staging folder {OLD_STAGE}? Type '{phrase2}' or press Enter to keep: ").strip()==phrase2:
            shutil.rmtree(OLD_STAGE); print("Removed old generated staging folder.")
    build_stage(years,"",include_zarr=args.include_source_zarr,studies=studies)

if __name__=="__main__":
    try: main()
    except Exception as e: print(f"ERROR: {e}",file=sys.stderr); raise SystemExit(1)
