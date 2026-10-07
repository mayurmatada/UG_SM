# Global MultiModalSoilNet Extension Plan

## Problem and proposed approach

Extend the Spain-focused `MultiModalSoilNet` into a globally generalizable 1 km
soil-moisture downscaling model without treating the current global ISMN archive
as if it were already a training-ready multimodal dataset. The first milestone
is station-generalization across regions and climate zones; wall-to-wall global
inference follows after the validation protocol and feature stack are stable.

The efficient approach is to preserve the existing residual-anchored CNN +
static-MLP + temporal-LSTM design and its 13x13, 1 km tensor contract, while
rebuilding the ingestion and feature-extraction layers around a global station
catalog and global Earth Engine exports. Do not concatenate all observations
blindly: first harmonize depths, units, timestamps, quality flags, station
metadata, missingness, and feature scaling.

## Current evidence from the project

- The report identifies global generalization as requiring harmonized
  multi-region training data, climate-zone-aware validation, and strict
  out-of-region tests, while retaining residual anchoring.
- `Data/Global/` contains a current ISMN archive with station observations,
  network metadata, sensor metadata, quality descriptions, and per-station
  static-variable CSVs. It does not contain the satellite/reanalysis patch
  tensors required by `MultiModalSoilNet`.
- The current Spain model expects a 16-channel, 13x13 spatial patch, a temporal
  sequence, and an anchor residual target. Its Earth Engine feature stack is
  based on SMAP, Sentinel-1, MODIS, ERA5-Land, SoilGrids, and NASADEM.
- The global archive is heterogeneous: networks use different sensors, depths,
  sampling intervals, coverage periods, and station densities. This is the
  main scientific risk, not the neural-network architecture.

## Data to obtain and provenance

### Already available locally

1. **Ground truth and station metadata**: the global ISMN archive in
   `Data/Global/`. Use the included `Readme.txt`, quality-flag descriptions,
   network descriptions, sensor list, references, and acknowledgements as the
   schema and citation authority.
2. **Regional reference data and working code**: `Data/Spain/`, the Spain
   notebook, and the Denmark/Oklahoma/Texas notebooks. These are useful for
   cross-region parser and feature-contract comparisons, not necessarily for
   direct global concatenation.

### Obtain from Google Earth Engine

For every selected global station and date, export the same harmonized
1 km/13x13 feature contract used by the final model:

- Sentinel-1 GRD: VV, VH, polarization difference/ratio, incidence angle.
- MODIS: NDVI, LST, and albedo using the report's documented temporal windows
  and scale factors.
- SMAP L4 (`SPL4SMGP`): surface soil-moisture anchor.
- ERA5-Land: precipitation, temperature/dewpoint-derived VPD, and antecedent
  precipitation; preserve units and document aggregation windows.
- SoilGrids: 0-5 cm clay, sand, and silt, with a documented fallback or
  missingness policy.
- NASADEM: elevation, slope, and aspect.

Use Earth Engine exports (rather than station-by-station API calls) for
batched, reproducible regional tiles. Partition the globe into manageable
regions/time blocks and retain export manifests.

### Obtain or derive for global stratification and diagnostics

- **Climate zones**: Koppen-Geiger or an equivalent documented global climate
  classification, from a stable public raster such as Beck et al. / WorldClim.
- **Land cover/cropland masks**: ESA WorldCover or Dynamic World, for coverage
  diagnostics and stratified error analysis.
- **Optional independent evaluation**: a held-out network/product such as
  COSMOS, SNOTEL/SCAN, or a national network not used in training. Use these
  only after checking depth/unit compatibility.
- **Optional global baseline products**: SMAP L4 and/or GLDAS/ERA5-Land for
  sanity checks, not as substitute labels.

Record product version, access date, projection, scale, temporal coverage,
license/citation, and preprocessing for every external source.

## Step-by-step implementation plan

### 1. Freeze the global data contract

- Define the target as quality-controlled near-surface soil moisture, initially
  0-5 cm where available; record depth rather than silently mixing deeper
  measurements.
- Define canonical units, UTC daily aggregation, valid ranges, quality-flag
  policy, duplicate-sensor collapse, and minimum observations per station.
- Define canonical station, network, country/region, climate-zone, latitude,
  longitude, elevation, and land-cover metadata.
- Define the exact spatial channels, temporal channels, static channels, anchor,
  residual target, and missingness indicators.
- Add provenance fields so every sample can be traced to network, file, sensor,
  depth, product version, and export batch.

### 2. Build and audit a global ISMN catalog

- Parse all `.stm` files from `Data/Global/` using a generalized version of the
  existing Spain parser.
- Select compatible near-surface files, retain only documented good-quality
  observations, normalize timestamps to UTC days, and collapse redundant
  sensors only after checking sensor/depth compatibility.
- Join per-station static-variable CSVs and network metadata.
- Produce an audit table: station counts, samples, date ranges, missingness,
  depth distribution, units, networks, countries, climate zones, and spatial
  density.
- Exclude or flag stations with ambiguous depth, impossible ranges, insufficient
  temporal overlap with predictors, or unresolved coordinates.

### 3. Select a staged global training cohort

- Start with networks and stations that meet the contract and have overlap with
  the selected Earth Engine period.
- Balance by network/climate zone rather than allowing SNOTEL/SCAN or another
  dense network to dominate.
- Hold out entire networks/regions before any normalization or calibration fit.
- Keep a small untouched benchmark cohort from geographically distinct networks.
- Only add heterogeneous deeper-depth observations in a later experiment with
  an explicit depth-aware target or depth-conditioned model.

### 4. Generalize the Earth Engine feature extraction

- Replace the Spain bounding box with a station-driven global tiling/export
  system.
- Use one tested implementation of the report's temporal windows, scale
  factors, reprojection, masking, and fallback behavior.
- Export station-date records and 13x13 patches in batches; avoid a single
  unbounded global task.
- Add coverage/missingness bands or flags so zero-filled imagery is not confused
  with a physical zero.
- Validate sample counts and feature ranges before creating TFRecords.

### 5. Create a versioned global tensor dataset

- Convert exports to the existing PyTorch/TFRecord contract:
  spatial `[B,C,13,13]`, temporal `[B,L,F]`, residual target `[B,1]`.
- Build chronological station sequences without crossing station boundaries or
  split boundaries.
- Fit normalization statistics on training stations only; persist them with the
  dataset version.
- Store station/network/region/climate metadata beside each record.
- Add deterministic manifests, checksums, and train/validation/test membership.

### 6. Establish leakage-safe global validation

- Primary split: entire networks or geographic regions held out.
- Secondary split: climate-zone holdouts and continent/biome stress tests.
- Keep dates disjoint where temporal extrapolation is being measured.
- Ensure no station, neighboring duplicate, normalization statistic, or
  calibration coefficient crosses a split.
- Report per-network, per-climate-zone, per-continent, wet/dry regime, and
  station-level metrics, not only one global R2.
- Compare against SMAP anchor, regional model, global tabular/XGBoost baseline,
  and the global MultiModalSoilNet.

### 7. Retrain and adapt MultiModalSoilNet

- Start with the existing architecture and residual target.
- Re-tune only after the data contract and split protocol are fixed.
- Evaluate whether global heterogeneity requires climate/network embeddings,
  domain-adversarial regularization, mixture-of-experts, or fine-tuning; treat
  these as controlled ablations, not default complexity.
- Fit the quadratic calibration layer on training/validation data only and
  evaluate it on untouched held-out networks.
- Add uncertainty estimates or prediction intervals before operational use.

### 8. Validate scientifically and operationally

- Run data integrity checks, feature coverage maps, and station footprint plots.
- Investigate failure modes by sensor family, depth, climate, land cover,
  season, precipitation regime, and missingness.
- Verify that gains over the anchor are positive on unseen regions, not merely
  on dense or familiar networks.
- Test reproducibility from manifests and rerun a small regional fixture.
- Preserve ISMN citations/acknowledgements and all Earth Engine/product
  attributions.

### 9. Only then build wall-to-wall global inference

- Tile the globe at 1 km with the same feature schema and date windows.
- Run inference with the frozen global scaler, model, calibration, and versioned
  metadata.
- Add overlap/coverage masks, quality flags, uncertainty, and no-data outputs.
- Validate map continuity and conservation of the anchor-scale signal.
- Publish a product specification before generating large global archives.

## Efficiency priorities

1. Build the parser/catalog audit before downloading any global satellite data.
2. Run a small representative pilot spanning several networks and climate zones.
3. Reuse the existing Earth Engine feature functions and model branches; change
   configuration and data plumbing before changing architecture.
4. Export only station-centered patches for training; defer wall-to-wall exports.
5. Use manifests and cached intermediate tables so failed Earth Engine jobs do
   not force reprocessing.
6. Treat balance, split design, and normalization as first-class artifacts.

## First concrete deliverables

- `global_station_catalog` with audit report and exclusion reasons.
- A versioned global data-contract/configuration file.
- A pilot Earth Engine export covering representative networks.
- Global TFRecord/tensor conversion and sequence-builder validation.
- A split manifest with network/climate/region holdouts.
- Baseline comparison report: anchor, XGBoost, regional MultiModalSoilNet,
  and global MultiModalSoilNet.
