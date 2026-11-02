#!/usr/bin/env bash
set -euo pipefail

# Packages already-computed laptop artifacts for a Kaggle full-training run.
# It DOES NOT rebuild manifests/caches; it only validates and copies existing files.

GLOBAL_ROOT="${UG_SM_GLOBAL_ROOT:-$HOME/Code/UG_SM/Data/Global}"
DATA_DIR="$GLOBAL_ROOT/model_ready_global_v1"
RUN_DIR="${UG_SM_RUN_DIR:-$DATA_DIR/training_runs/run_optuna_s1_filtered_global_r2_v1}"
MANIFEST_DIR="$RUN_DIR/manifests"
SCALER_DIR="$RUN_DIR/scalers"
CACHE_DIR="$RUN_DIR/model_ready_compact_cache_parallel_v2"
OUT_DIR="${UG_SM_ASSET_OUT_DIR:-$HOME/kaggle_upload/global-sm-training-assets}"
ZIP_PATH="${UG_SM_ASSET_ZIP_PATH:-/mnt/c/Users/Mayur/Downloads/global-sm-training-assets.zip}"

for p in "$MANIFEST_DIR" "$SCALER_DIR" "$CACHE_DIR"; do
  if [[ ! -d "$p" ]]; then
    echo "ERROR: Required precomputed directory not found: $p" >&2
    echo "Set UG_SM_RUN_DIR to the prior completed run directory if it lives elsewhere." >&2
    exit 2
  fi
done

required_scalers=(environment_only_tabular.json location_history_tabular.json environment_only_raster.json location_history_raster.json)
for f in "${required_scalers[@]}"; do
  if [[ ! -s "$SCALER_DIR/$f" ]]; then
    echo "ERROR: Missing scaler JSON: $SCALER_DIR/$f" >&2
    exit 3
  fi
done

required_manifests=(base_manifest.parquet manifest_config.json base_manifest_config.json \
  environment_only/train.parquet environment_only/test.parquet \
  location_history/train.parquet location_history/test.parquet)
for f in "${required_manifests[@]}"; do
  if [[ ! -s "$MANIFEST_DIR/$f" ]]; then
    echo "ERROR: Missing precomputed manifest: $MANIFEST_DIR/$f" >&2
    exit 4
  fi
done

# Fresh staging directory: prevents a previous run's stale cache folders being re-uploaded.
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

# Validate and select exactly one complete cache per task/split. Ignores stale and .building dirs.
python - "$CACHE_DIR" "$OUT_DIR" "$MANIFEST_DIR" "$SCALER_DIR" <<'PY'
import json, os, shutil, sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

src_root, out_root, manifest_root, scaler_root = map(Path, sys.argv[1:5])
out_cache = out_root / "model_ready_compact_cache_parallel_v2"
EXPECTED_BUILDER_ID = "compact_metadata_parallel_rowgroup_v2_filtered_s1_2026_10_24"
scaler_values = {
    mode: {
        "tabular": json.loads((scaler_root / f"{mode}_tabular.json").read_text(encoding="utf-8")),
        "raster": json.loads((scaler_root / f"{mode}_raster.json").read_text(encoding="utf-8")),
    }
    for mode in ("location_history", "environment_only")
}
out_cache.mkdir(parents=True, exist_ok=True)

required_modes = ["location_history", "environment_only"]
required_splits = ["train", "test"]
selected = {}

for mode in required_modes:
    for split in required_splits:
        valid = []
        for candidate in src_root.glob(f"{mode}_{split}_*"):
            if not candidate.is_dir() or candidate.name.endswith(".building") or ".old_" in candidate.name or ".incompatible_" in candidate.name:
                continue
            meta_path = candidate / "metadata.json"
            if not meta_path.is_file():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if (int(meta.get("schema", -1)) != 4 or meta.get("task_mode") != mode
                        or meta.get("split") != split or meta.get("builder_id") != EXPECTED_BUILDER_ID):
                    continue
                # Reject cache folders from an old manifest/scaler revision before packaging.
                signature = meta.get("input_signature")
                manifest_path = manifest_root / mode / f"{split}.parquet"
                if not manifest_path.is_file():
                    continue
                manifest_pf = pq.ParquetFile(str(manifest_path))
                manifest_meta = manifest_pf.metadata
                if int(meta.get("n_rows", -1)) != int(manifest_meta.num_rows):
                    continue
                if signature is not None:
                    if (signature.get("task_mode") != mode or signature.get("split") != split
                            or int(signature.get("schema", -1)) != 4
                            or int(signature.get("manifest_size", -1)) != manifest_path.stat().st_size
                            or int(signature.get("manifest_num_rows", -1)) != int(manifest_meta.num_rows)
                            or int(signature.get("manifest_num_row_groups", -1)) != int(manifest_meta.num_row_groups)
                            or signature.get("manifest_schema") != str(manifest_pf.schema)
                            or signature.get("tabular_scaler") != scaler_values[mode]["tabular"]
                            or signature.get("raster_scaler") != scaler_values[mode]["raster"]):
                        continue
                arrays = meta.get("arrays", {})
                n = int(meta.get("n_rows", -1))
                nvalid = int(meta.get("n_valid_rows", -1))
                if n <= 0 or not (0 <= nvalid <= n) or not arrays:
                    continue
                complete = True
                for spec in arrays.values():
                    f = candidate / spec["file"]
                    expected = n
                    import numpy as np
                    expected *= int(np.prod(spec["shape"], dtype=np.int64)) * np.dtype(spec["dtype"]).itemsize
                    if not f.is_file() or f.stat().st_size != expected:
                        complete = False
                        break
                if complete:
                    size = sum(p.stat().st_size for p in candidate.rglob("*") if p.is_file())
                    valid.append((candidate.stat().st_mtime, size, candidate, meta))
            except Exception:
                continue
        if not valid:
            raise SystemExit(f"ERROR: No complete schema-v4 cache found for {mode}/{split} under {src_root}.")
        valid.sort(key=lambda item: (item[0], item[1]), reverse=True)
        _mtime, _size, chosen, meta = valid[0]
        destination = out_cache / chosen.name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(chosen, destination,
                        ignore=shutil.ignore_patterns("*:Zone.Identifier", "Zone.Identifier"))
        selected[f"{mode}/{split}"] = {
            "directory": chosen.name,
            "n_rows": meta.get("n_rows"),
            "n_valid_rows": meta.get("n_valid_rows"),
            "builder_id": meta.get("builder_id"),
            "schema": meta.get("schema"),
            "size_bytes": _size,
        }
        print(f"Selected {mode}/{split}: {chosen.name}; rows={meta.get('n_rows'):,}; size={_size/1024**3:.2f} GiB")

(out_root / "selected_cache_manifest.json").write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
PY

# Rebuild staging folder each time, but never mutate the original training run.
mkdir -p "$OUT_DIR/manifests" "$OUT_DIR/scalers"
cp -a "$MANIFEST_DIR/base_manifest.parquet" "$OUT_DIR/manifests/"
cp -a "$MANIFEST_DIR/manifest_config.json" "$OUT_DIR/manifests/"
cp -a "$MANIFEST_DIR/base_manifest_config.json" "$OUT_DIR/manifests/"
for mode in environment_only location_history; do
  mkdir -p "$OUT_DIR/manifests/$mode"
  cp -a "$MANIFEST_DIR/$mode/train.parquet" "$OUT_DIR/manifests/$mode/"
  cp -a "$MANIFEST_DIR/$mode/test.parquet" "$OUT_DIR/manifests/$mode/"
done
for f in "${required_scalers[@]}"; do cp -a "$SCALER_DIR/$f" "$OUT_DIR/scalers/"; done

# If the corrected-loss Optuna studies exist, export their best settings as a fixed training config.
# This does not run additional trials and falls back to notebook defaults if unavailable.
python - "$RUN_DIR" "$OUT_DIR/best_config.json" <<'PY'
import json, sys, glob, os
from pathlib import Path
run_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
config = {}
try:
    import optuna
    for mode in ("location_history", "environment_only"):
        candidates = sorted(run_dir.glob(f"optuna_filtered_s1_{mode}*_lossfix_r2_rmse_v3.sqlite3"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        for db in candidates:
            suffix = db.stem.removeprefix(f"optuna_filtered_s1_{mode}")
            study_name = f"global_soil_{mode}_filtered_s1{suffix}"
            try:
                study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db.resolve()}")
                best = study.best_trial
                params = dict(best.params)
                attrs = dict(best.user_attrs or {})
                config[mode] = {
                    "source_study": study_name,
                    "source_db": db.name,
                    "source_trial": int(best.number),
                    "source_best_test_r2": float(best.value),
                    "params": params,
                    "effective_batch_size": attrs.get("effective_batch_size", 2048),
                    "loader_microbatch_size": attrs.get("loader_microbatch_size", 512),
                    "accumulation_steps": attrs.get("accumulation_steps", 4),
                    "safe_microbatch_ceiling": attrs.get("safe_microbatch_ceiling", 512),
                }
                print(f"Exported prior best config for {mode}: trial {best.number}, R²={best.value:.4f}")
                break
            except Exception as exc:
                print(f"Skipping {db.name}: {type(exc).__name__}: {exc}")
except Exception as exc:
    print(f"Optuna unavailable locally; notebook will use fixed default parameters: {type(exc).__name__}: {exc}")
(out.parent / "best_config.json").write_text(json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")
PY

python - "$GLOBAL_ROOT" "$OUT_DIR" <<'PY'
import json, sys, hashlib
from pathlib import Path
root, out = Path(sys.argv[1]), Path(sys.argv[2])
inputs = {}
for rel in ["model_ready_global_v1/s1_valid_global_v1/sample_index_s1_valid.parquet",
            "model_ready_global_v1/s1_valid_global_v1/daily_targets_s1_valid.parquet"]:
    p = root / rel
    inputs[rel] = {"exists": p.is_file(), "size_bytes": p.stat().st_size if p.is_file() else None}
meta = {
    "bundle_type": "global_sm_precomputed_training_assets",
    "created_from_global_root": str(root),
    "source_inputs": inputs,
    "notes": [
        "Contains locally generated manifests, scaler JSONs, and schema-v4 compact metadata caches.",
        "Does not duplicate MODIS or Sentinel-1 Zarr raster stores.",
        "The notebook verifies cache schema, task/split, row count, scaler signature and arrays before use."
    ]
}
(out / "asset_bundle.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
PY

# Build a ZIP with no compression. This avoids wasting CPU compressing Parquet/memmap data.
ZIP_DIR="$(dirname "$ZIP_PATH")"
mkdir -p "$ZIP_DIR"
rm -f "$ZIP_PATH"
python - "$OUT_DIR" "$ZIP_PATH" <<'PY'
import sys, zipfile
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
base = src.parent
with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
    for p in src.rglob("*"):
        rel = p.relative_to(base)
        # Windows Zone.Identifier streams are filesystem metadata, not training inputs.
        if p.is_file() and "Zone.Identifier" not in p.name and ":" not in str(rel):
            zf.write(p, arcname=str(rel))
print("Created", dst, "size GiB:", round(dst.stat().st_size / 1024**3, 2))
PY

# Non-destructive archive verification.
unzip -tq "$ZIP_PATH"
echo
echo "Training assets folder: $OUT_DIR"
du -sh "$OUT_DIR"
echo "ZIP archive: $ZIP_PATH"
ls -lh "$ZIP_PATH"
echo "Upload this as a second Kaggle dataset and attach it alongside Global_SM."
