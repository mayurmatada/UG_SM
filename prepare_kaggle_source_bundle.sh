#!/usr/bin/env bash
set -Eeuo pipefail

# Build a lean Kaggle upload tree from the local Global folder.
# It intentionally excludes all generated training_runs and cache directories.
SRC="${SRC:-/home/mayur/Code/UG_SM/Data/Global}"
OUT="${OUT:-$HOME/kaggle_upload/ug-sm-global-source}"

need=(
  "$SRC/model_ready_global_v1/feature_schema.json"
  "$SRC/model_ready_global_v1/s1_valid_global_v1/sample_index_s1_valid.parquet"
  "$SRC/model_ready_global_v1/s1_valid_global_v1/daily_targets_s1_valid.parquet"
  "$SRC/modis/evi_lst_patch_dataset"
  "$SRC/sentinel1/nearest_valid_overview_v1"
)
for p in "${need[@]}"; do
  if [[ ! -e "$p" ]]; then
    echo "ERROR: required project input not found: $p" >&2
    exit 2
  fi
done

# Confirm that yearly source stores exist before copying.
if ! find "$SRC/modis/evi_lst_patch_dataset" -mindepth 2 -maxdepth 2 -type d -name 'patches.zarr' -print -quit | grep -q .; then
  echo "ERROR: no MODIS */patches.zarr stores found under $SRC/modis/evi_lst_patch_dataset" >&2
  exit 2
fi
if ! find "$SRC/sentinel1/nearest_valid_overview_v1" -maxdepth 1 -type d -name 's1_nearest_valid_*.zarr' -print -quit | grep -q .; then
  echo "ERROR: no Sentinel-1 yearly Zarr stores found under $SRC/sentinel1/nearest_valid_overview_v1" >&2
  exit 2
fi

mkdir -p "$OUT/model_ready_global_v1/s1_valid_global_v1" \
         "$OUT/modis/evi_lst_patch_dataset" \
         "$OUT/sentinel1/nearest_valid_overview_v1"

copy_tree() {
  local from="$1" to="$2"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --info=progress2 "$from/" "$to/"
  else
    cp -a "$from/." "$to/"
  fi
}

echo "Copying feature schema and filtered labels/index..."
cp -av "$SRC/model_ready_global_v1/feature_schema.json" "$OUT/model_ready_global_v1/feature_schema.json"
cp -av "$SRC/model_ready_global_v1/s1_valid_global_v1/sample_index_s1_valid.parquet" "$OUT/model_ready_global_v1/s1_valid_global_v1/"
cp -av "$SRC/model_ready_global_v1/s1_valid_global_v1/daily_targets_s1_valid.parquet" "$OUT/model_ready_global_v1/s1_valid_global_v1/"

echo "Copying MODIS EVI/LST yearly Zarr stores..."
copy_tree "$SRC/modis/evi_lst_patch_dataset" "$OUT/modis/evi_lst_patch_dataset"
echo "Copying Sentinel-1 yearly Zarr stores..."
copy_tree "$SRC/sentinel1/nearest_valid_overview_v1" "$OUT/sentinel1/nearest_valid_overview_v1"

echo
echo "Curated bundle complete: $OUT"
du -sh "$OUT" "$OUT/model_ready_global_v1" "$OUT/modis" "$OUT/sentinel1"
echo "This bundle does not include training_runs, model_ready_compact_cache_parallel_v2, zarr_array_mmap_cache, checkpoints, or Optuna/MLflow logs."
echo "Next: upload this folder as a PRIVATE Kaggle Dataset, using the licence applicable to the underlying data."
echo "Then attach that dataset to notebook2_global_kaggle_t4x2_parallel_pilot.ipynb."
