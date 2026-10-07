#!/usr/bin/env bash
set -euo pipefail

# Run the local global-data pipeline:
# archive -> audited ISMN catalog -> network split manifest -> tensor packing.
# Earth Engine exports are intentionally external. Resume with --feature-manifest
# once a station-date feature manifest and its referenced .npz files exist.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE=""
FEATURE_MANIFEST=""
OUTPUT_DIR="${ROOT_DIR}/global_pipeline_output"
MIN_GOOD_ROWS=1
SEED=20261003
VALIDATION_FRACTION=0.2
TEST_FRACTION=0.2
SKIP_TESTS=0
RUN_GEE=1
SUBMIT_GEE=0
GEE_MAX_STATIONS=100
EE_PROJECT=""
EXPORT_YEAR=""
EXPORT_START_DATE="2015-04-01"
EXPORT_END_DATE="2026-10-03"

usage() {
    cat <<'EOF'
Usage:
  ./run_global_pipeline.sh [options]

Options:
  --archive PATH              Global ISMN .zip archive. Auto-detected under Data/Global.
  --feature-manifest PATH     Station-date feature CSV; enables final tensor packing.
  --output-dir PATH           Output directory (default: ./global_pipeline_output).
  --minimum-good-rows N       Minimum good-quality observations per station file (default: 1).
  --seed N                    Deterministic network split seed (default: 20261003).
  --validation-fraction F     Validation network fraction (default: 0.2).
  --test-fraction F           Test network fraction (default: 0.2).
  --gee-max-stations N       Maximum stations per adaptive EE tile (default: 100).
  --ee-project ID             Optional Earth Engine Cloud project for submission.
  --submit-gee                Submit Earth Engine Drive TFRecord tasks (default is dry-run).
  --year YYYY                 Export only one calendar year to Drive.
  --start-date YYYY-MM-DD     Start of the export range (inclusive).
  --end-date YYYY-MM-DD       End of the export range (exclusive).
  --no-gee                    Skip Earth Engine tile/task manifest generation.
  --skip-tests                Skip the local regression test suite.
  -h, --help                  Show this help.

Without --feature-manifest, the script completes the catalog and split stages,
then exits with instructions for resuming after feature exports are available.
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

while (($#)); do
    case "$1" in
        --archive)
            (($# >= 2)) || die "--archive requires a path"
            ARCHIVE="$2"
            shift 2
            ;;
        --feature-manifest)
            (($# >= 2)) || die "--feature-manifest requires a path"
            FEATURE_MANIFEST="$2"
            shift 2
            ;;
        --output-dir)
            (($# >= 2)) || die "--output-dir requires a path"
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --minimum-good-rows)
            (($# >= 2)) || die "--minimum-good-rows requires an integer"
            MIN_GOOD_ROWS="$2"
            shift 2
            ;;
        --seed)
            (($# >= 2)) || die "--seed requires an integer"
            SEED="$2"
            shift 2
            ;;
        --validation-fraction)
            (($# >= 2)) || die "--validation-fraction requires a number"
            VALIDATION_FRACTION="$2"
            shift 2
            ;;
        --test-fraction)
            (($# >= 2)) || die "--test-fraction requires a number"
            TEST_FRACTION="$2"
            shift 2
            ;;
        --gee-max-stations)
            (($# >= 2)) || die "--gee-max-stations requires an integer"
            GEE_MAX_STATIONS="$2"
            shift 2
            ;;
        --ee-project)
            (($# >= 2)) || die "--ee-project requires a project ID"
            EE_PROJECT="$2"
            shift 2
            ;;
        --year)
            (($# >= 2)) || die "--year requires a four-digit year"
            EXPORT_YEAR="$2"
            shift 2
            ;;
        --start-date)
            (($# >= 2)) || die "--start-date requires YYYY-MM-DD"
            EXPORT_START_DATE="$2"
            shift 2
            ;;
        --end-date)
            (($# >= 2)) || die "--end-date requires YYYY-MM-DD"
            EXPORT_END_DATE="$2"
            shift 2
            ;;
        --submit-gee)
            SUBMIT_GEE=1
            shift
            ;;
        --no-gee)
            RUN_GEE=0
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1 (use --help)"
            ;;
    esac
done

if [[ -z "$ARCHIVE" ]]; then
    mapfile -t archives < <(find "${ROOT_DIR}/Data/Global" -maxdepth 1 -type f -iname '*.zip' -print | sort)
    [[ "${#archives[@]}" -eq 1 ]] || die "expected exactly one .zip archive under Data/Global; pass --archive PATH"
    ARCHIVE="${archives[0]}"
fi

if ((SUBMIT_GEE == 1)); then
    case "$EE_PROJECT" in
        YOUR_GEE_PROJECT_ID|YOUR_PROJECT_ID|PROJECT_ID|your-project-id)
            die "--ee-project is still a placeholder; pass a real Google Cloud project ID"
            ;;
    esac
fi

[[ -f "$ARCHIVE" ]] || die "archive does not exist: $ARCHIVE"
[[ "$MIN_GOOD_ROWS" =~ ^[1-9][0-9]*$ ]] || die "--minimum-good-rows must be a positive integer"

if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PYTHON="${ROOT_DIR}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
else
    die "Python 3 was not found; create .venv or install python3"
fi

mkdir -p "$OUTPUT_DIR"
CATALOG_CSV="${OUTPUT_DIR}/global_station_catalog.csv"
CATALOG_JSON="${OUTPUT_DIR}/global_station_catalog_summary.json"
SPLIT_CSV="${OUTPUT_DIR}/global_split_manifest.csv"
SPLIT_JSON="${OUTPUT_DIR}/global_split_summary.json"
LABEL_DATES_CSV="${OUTPUT_DIR}/global_label_dates.csv"
TENSOR_DIR="${OUTPUT_DIR}/global_tensors"

printf 'Using Python: %s\n' "$PYTHON"
printf 'Using archive: %s\n' "$ARCHIVE"
printf 'Output directory: %s\n' "$OUTPUT_DIR"

"$PYTHON" "${ROOT_DIR}/build_global_ismn_catalog.py" "$ARCHIVE" \
    --scan-observations \
    --output-csv "$CATALOG_CSV" \
    --output-json "$CATALOG_JSON"

"$PYTHON" "${ROOT_DIR}/build_global_splits.py" "$CATALOG_CSV" \
    --output-csv "$SPLIT_CSV" \
    --output-json "$SPLIT_JSON" \
    --minimum-good-quality-rows "$MIN_GOOD_ROWS" \
    --validation-fraction "$VALIDATION_FRACTION" \
    --test-fraction "$TEST_FRACTION" \
    --seed "$SEED"

"$PYTHON" "${ROOT_DIR}/build_global_label_dates.py" "$ARCHIVE" \
    --output "$LABEL_DATES_CSV"

if ((RUN_GEE == 1)); then
    GEE_DIR="${OUTPUT_DIR}/global_gee_exports"
    GEE_ARGS=(
        "$PYTHON" "${ROOT_DIR}/gee_global_export.py" "$SPLIT_CSV"
        --output-dir "$GEE_DIR"
        --max-stations-per-tile "$GEE_MAX_STATIONS"
        --start-date "$EXPORT_START_DATE"
        --end-date "$EXPORT_END_DATE"
        --window-years 1
        --label-dates "$LABEL_DATES_CSV"
    )
    if [[ -n "$EE_PROJECT" ]]; then
        GEE_ARGS+=(--ee-project "$EE_PROJECT")
    fi
    if [[ -n "$EXPORT_YEAR" ]]; then
        GEE_ARGS+=(--year "$EXPORT_YEAR")
    fi
    if ((SUBMIT_GEE == 0)); then
        GEE_ARGS+=(--dry-run)
    fi
    "${GEE_ARGS[@]}"
fi

if ((SKIP_TESTS == 0)); then
    "$PYTHON" -m unittest discover -s "${ROOT_DIR}/tests" -v
fi

if [[ -z "$FEATURE_MANIFEST" ]]; then
    cat <<EOF

Catalog and split stages completed.
$(if ((RUN_GEE == 1)); then echo "Earth Engine tile/task manifest: ${OUTPUT_DIR}/global_gee_exports"; fi)
No --feature-manifest was supplied, so tensor packing was not run.

Create a station-date feature manifest with these columns:
  station_id,network,sample_date,split,feature_path,target,anchor

Then resume with:
  "$0" --archive "$ARCHIVE" --output-dir "$OUTPUT_DIR" \\
    --feature-manifest PATH_TO_FEATURE_MANIFEST
EOF
    exit 0
fi

[[ -f "$FEATURE_MANIFEST" ]] || die "feature manifest does not exist: $FEATURE_MANIFEST"
"$PYTHON" "${ROOT_DIR}/build_global_tensors.py" "$FEATURE_MANIFEST" \
    --output-dir "$TENSOR_DIR"

printf '\nGlobal pipeline completed successfully.\n'
printf 'Catalog: %s\nSplit manifest: %s\nTensor directory: %s\n' \
    "$CATALOG_CSV" "$SPLIT_CSV" "$TENSOR_DIR"
