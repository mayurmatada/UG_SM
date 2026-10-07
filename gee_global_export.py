"""Create and optionally submit tiled, station-centered Earth Engine exports.

The feature stack follows the Spain notebook: dynamic spatial bands are
sampled as 13x13 neighborhoods at station locations, while temporal scalar
inputs and static cell features are exported separately. VPD is derived from
ERA5 temperature/dewpoint and antecedent precipitation from ERA5 precipitation
without additional collections. Static cell features are exported once.

Exports are partitioned by adaptive station-count tiles and yearly windows.
Earth Engine writes GZIP-compressed TFRecord tables to Google Drive.

Use ``--dry-run`` to create the deterministic task manifest without importing
the Earth Engine client or submitting tasks.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, fields
from datetime import date
from pathlib import Path
from typing import Iterable

from global_data_contract import GlobalDataContract


EARTH_ENGINE_GLOBAL_CRS = "SR-ORG:6974"
PROJECT_PLACEHOLDERS = frozenset(
    {"YOUR_GEE_PROJECT_ID", "YOUR_PROJECT_ID", "PROJECT_ID", "your-project-id"}
)


@dataclass(frozen=True)
class Station:
    station_id: str
    network: str
    station: str
    latitude: float
    longitude: float
    cell_id: str


@dataclass(frozen=True)
class ExportTask:
    task_id: str
    tile_id: str
    station_count: int
    start_date: str
    end_date: str
    description: str
    drive_folder: str
    file_prefix: str
    file_format: str = "TFRecord"
    compressed: bool = True
    export_kind: str = "daily"
    label_date_count: int = 0


def _global_cell_id(latitude: float, longitude: float) -> str:
    """Approximate the 1 km EASE-Grid cell identity without GIS dependencies."""
    # SR-ORG:6974 is Earth Engine's registered alias for the global EASE-Grid
    # projection corresponding to EPSG:6933. These formulas provide a stable
    # cell key; Earth Engine remains authoritative for reprojection.
    import math

    radius = 6378137.0
    x = radius * math.cos(math.radians(30.0)) * math.radians(longitude)
    y = radius * math.sin(math.radians(latitude)) / math.cos(math.radians(30.0))
    return f"x{math.floor(x / 1000):+07d}_y{math.floor(y / 1000):+07d}"


def read_stations(path: Path) -> list[Station]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"network", "station", "latitude", "longitude"}
    missing = required - set(rows[0] if rows else ())
    if missing:
        raise ValueError(f"Station manifest is missing columns: {sorted(missing)}")
    stations: dict[str, Station] = {}
    for row in rows:
        station_id = f"{row['network']}__{row['station']}"
        station = Station(
            station_id=station_id,
            network=row["network"],
            station=row["station"],
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            cell_id=_global_cell_id(float(row["latitude"]), float(row["longitude"])),
        )
        if not -90 <= station.latitude <= 90 or not -180 <= station.longitude <= 180:
            raise ValueError(f"Invalid coordinates for {station_id}: {station.latitude}, {station.longitude}")
        stations.setdefault(station_id, station)
    if not stations:
        raise ValueError("No stations found")
    return sorted(stations.values(), key=lambda item: item.station_id)


def collapse_resolution_cells(stations: list[Station]) -> tuple[list[Station], list[dict[str, str]]]:
    """Keep one deterministic point per 1 km cell and retain a collision map."""
    by_cell: dict[str, list[Station]] = {}
    for station in stations:
        by_cell.setdefault(station.cell_id, []).append(station)
    representatives: list[Station] = []
    mapping: list[dict[str, str]] = []
    for cell_id, items in sorted(by_cell.items()):
        representative = sorted(items, key=lambda item: item.station_id)[0]
        representatives.append(representative)
        for item in sorted(items, key=lambda item: item.station_id):
            mapping.append(
                {
                    "cell_id": cell_id,
                    "representative_station_id": representative.station_id,
                    "station_id": item.station_id,
                    "collapsed": str(item.station_id != representative.station_id).lower(),
                }
            )
    return representatives, mapping


def _split_spatial(stations: list[Station], maximum: int, prefix: str = "tile") -> list[list[Station]]:
    """Recursively bisect the wider coordinate extent until count is bounded."""
    if len(stations) <= maximum:
        return [sorted(stations, key=lambda item: item.station_id)]
    lat_span = max(item.latitude for item in stations) - min(item.latitude for item in stations)
    lon_span = max(item.longitude for item in stations) - min(item.longitude for item in stations)
    coordinate = "latitude" if lat_span >= lon_span else "longitude"
    ordered = sorted(stations, key=lambda item: (getattr(item, coordinate), item.station_id))
    midpoint = len(ordered) // 2
    return _split_spatial(ordered[:midpoint], maximum, prefix) + _split_spatial(
        ordered[midpoint:], maximum, prefix
    )


def adaptive_tiles(stations: list[Station], maximum: int) -> dict[str, list[Station]]:
    if maximum < 1:
        raise ValueError("maximum stations per tile must be positive")
    if len(stations) <= maximum:
        chunks = [sorted(stations, key=lambda item: item.station_id)]
    else:
        # Use the smallest number of coordinate-ordered chunks that satisfies
        # the quota. Recursive bisection would round the tile count up to the
        # next power of two (for example, 1,631 stations -> 32 tiles).
        tile_count, remainder = divmod(len(stations), maximum)
        tile_count += remainder > 0
        lat_span = max(item.latitude for item in stations) - min(item.latitude for item in stations)
        lon_span = max(item.longitude for item in stations) - min(item.longitude for item in stations)
        coordinate = "latitude" if lat_span >= lon_span else "longitude"
        ordered = sorted(stations, key=lambda item: (getattr(item, coordinate), item.station_id))
        base_size, extra = divmod(len(ordered), tile_count)
        chunks = []
        start = 0
        for index in range(tile_count):
            size = base_size + (index < extra)
            chunks.append(ordered[start : start + size])
            start += size
    return {f"tile_{index:04d}": chunk for index, chunk in enumerate(chunks)}


def time_windows(start: date, end: date, years: int = 1) -> list[tuple[date, date]]:
    if years < 1:
        raise ValueError("years must be positive")
    if end <= start:
        raise ValueError("end date must be after start date")
    windows: list[tuple[date, date]] = []
    year = start.year
    while date(year, 1, 1) < end:
        next_year = date(year + years, 1, 1)
        windows.append((max(start, date(year, 1, 1)), min(end, next_year)))
        year = next_year.year
    return windows


def month_windows(start: date, end: date) -> list[tuple[date, date]]:
    """Backward-compatible monthly planner for callers that need it."""
    if end <= start:
        raise ValueError("end date must be after start date")
    windows: list[tuple[date, date]] = []
    year, month = start.year, start.month
    while date(year, month, 1) < end:
        next_month = date(year + (month == 12), 1 if month == 12 else month + 1, 1)
        windows.append((max(start, date(year, month, 1)), min(end, next_month)))
        year, month = next_month.year, next_month.month
    return windows


def build_tasks(
    tiles: dict[str, list[Station]],
    start: date,
    end: date,
    drive_folder: str,
    years: int = 1,
    only_year: int | None = None,
    label_dates: dict[str, set[str]] | None = None,
    include_static: bool = True,
    static_only: bool = False,
) -> list[ExportTask]:
    tasks: list[ExportTask] = []
    for tile_id, stations in tiles.items():
        if include_static:
            tasks.append(
                ExportTask(
                    task_id=f"{tile_id}_static",
                    tile_id=tile_id,
                    station_count=len(stations),
                    start_date=start.isoformat(),
                    end_date=end.isoformat(),
                    description=f"global_ismn_{tile_id}_static",
                    drive_folder=drive_folder,
                    file_prefix=f"global_ismn_{tile_id}_static",
                    export_kind="static",
                )
            )
        if static_only:
            continue
        for window_start, window_end in time_windows(start, end, years):
            if only_year is not None and window_start.year != only_year:
                continue
            dates = (
                {
                    day
                    for station in stations
                    for day in (label_dates or {}).get(station.cell_id, set())
                    if window_start.isoformat() <= day < window_end.isoformat()
                }
                if label_dates is not None
                else set()
            )
            if label_dates is not None and not dates:
                continue
            suffix = f"{tile_id}_{window_start:%Y%m%d}_{window_end:%Y%m%d}"
            tasks.append(
                ExportTask(
                    task_id=suffix,
                    tile_id=tile_id,
                    station_count=len(stations),
                    start_date=window_start.isoformat(),
                    end_date=window_end.isoformat(),
                    description=f"global_ismn_{suffix}",
                    drive_folder=drive_folder,
                    file_prefix=f"global_ismn_{suffix}",
                    export_kind="daily",
                    label_date_count=len(dates),
                )
            )
    return tasks


def write_manifest(
    tasks: list[ExportTask],
    tiles: dict[str, list[Station]],
    output_dir: Path,
    *,
    start: date,
    end: date,
    maximum: int,
    collision_map: list[dict[str, str]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "export_tasks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[field.name for field in fields(ExportTask)])
        writer.writeheader()
        writer.writerows(asdict(task) for task in tasks)
    (output_dir / "stations.json").write_text(
        json.dumps({key: [asdict(item) for item in value] for key, value in tiles.items()}, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "resolution_cell_mapping.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["cell_id", "representative_station_id", "station_id", "collapsed"])
        writer.writeheader()
        writer.writerows(collision_map)
    summary = {
        "contract": GlobalDataContract().as_dict(),
        "station_count": sum(len(value) for value in tiles.values()),
        "unique_station_count": len({item.station_id for value in tiles.values() for item in value}),
        "resolution": "1000 m nominal",
        "earth_engine_crs": EARTH_ENGINE_GLOBAL_CRS,
        "resolution_cell_count": len({item.cell_id for value in tiles.values() for item in value}),
        "collapsed_station_count": sum(item["collapsed"] == "true" for item in collision_map),
        "tile_count": len(tiles),
        "maximum_stations_per_tile": maximum,
        "task_count": len(tasks),
        "start_date": start.isoformat(),
        "end_date_exclusive": end.isoformat(),
        "export_format": "TFRecord",
        "compression": "GZIP",
        "drive_folder": tasks[0].drive_folder if tasks else None,
    }
    (output_dir / "export_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _build_ee_export(
    ee: object,
    stations: list[Station],
    task: ExportTask,
    label_dates: set[str] | None = None,
    label_cells_by_date: dict[str, set[str]] | None = None,
    *,
    tile_scale: int = 16,
    s1_speckle_radius: int = 0,
) -> object:
    """Build one station-centered, quota-aware daily TFRecord export."""
    contract = GlobalDataContract()
    # Earth Engine's projection arguments are server-side Projection objects.
    # Passing the CRS string through sampleRegions can fail with
    # "The CRS of a map projection could not be parsed".
    export_projection = ee.Projection(EARTH_ENGINE_GLOBAL_CRS)
    points = ee.FeatureCollection(
        [
            ee.Feature(
                ee.Geometry.Point([station.longitude, station.latitude]),
                {
                    "station_id": station.station_id,
                    "cell_id": station.cell_id,
                    "network": station.network,
                },
            )
            for station in stations
        ]
    )
    dated_points = ee.FeatureCollection(
        [
            ee.Feature(
                ee.Geometry.Point([station.longitude, station.latitude]),
                {
                    "station_id": station.station_id,
                    "cell_id": station.cell_id,
                    "network": station.network,
                    "sample_date": day,
                },
            )
            for day, cells in (label_cells_by_date or {}).items()
            for station in stations
            if station.cell_id in cells
        ]
    ) if label_cells_by_date is not None else points
    start = ee.Date(task.start_date)
    end = ee.Date(task.end_date)
    buffered_start = start.advance(-30, "day")
    aoi = points.geometry().bounds()

    def safe_median(collection: object, bands: list[str]) -> object:
        return ee.Image(
            ee.Algorithms.If(
                collection.size().gt(0),
                collection.median(),
                ee.Image.constant([0] * len(bands)).rename(bands),
            )
        )

    def window(collection: object, bands: list[str], current: object, days: int) -> object:
        subset = collection.filterDate(ee.Date(current).advance(-days, "day"), current)
        return safe_median(subset, bands).select(bands)

    def daily(collection: object, bands: list[str], current: object) -> object:
        subset = collection.filterDate(current, ee.Date(current).advance(1, "day"))
        return safe_median(subset, bands).select(bands)

    def antecedent_precipitation(current: object, days: int = 10, alpha: float = 0.85) -> object:
        """Compute API from the existing ERA5 precipitation collection."""
        subset = era5.filterDate(
            ee.Date(current).advance(-days, "day"), current
        ).select("precipitation_mm").sort("system:time_start")
        images = subset.toList(subset.size())

        def add_weighted(index: object, accumulated: object) -> object:
            image = ee.Image(images.get(index))
            return ee.Image(accumulated).add(
                image.multiply(ee.Number(alpha).pow(ee.Number(index)))
            )

        return ee.Image(
            ee.Algorithms.If(
                subset.size().gt(0),
                ee.List.sequence(0, subset.size().subtract(1)).iterate(
                    add_weighted, ee.Image.constant(0)
                ),
                ee.Image.constant(0),
            )
        ).rename("antecedent_precipitation_mm")

    ndvi = ee.ImageCollection("MODIS/061/MOD13A1").filterBounds(aoi).filterDate(
        buffered_start, end
    ).map(lambda image: ee.Image(image).select("NDVI").multiply(0.0001).rename("ndvi"))
    if tile_scale < 1:
        raise ValueError("tile_scale must be positive")
    if s1_speckle_radius < 0:
        raise ValueError("s1_speckle_radius must be non-negative")

    def process_s1(image: object) -> object:
        image = ee.Image(image)
        bands = image.bandNames()
        vv = ee.Image(
            ee.Algorithms.If(bands.contains("VV"), image.select("VV"), ee.Image.constant(-20))
        )
        vh = ee.Image(
            ee.Algorithms.If(bands.contains("VH"), image.select("VH"), vv)
        )
        if s1_speckle_radius:
            vv = vv.focal_median(radius=s1_speckle_radius, units="pixels")
            vh = vh.focal_median(radius=s1_speckle_radius, units="pixels")
        radar_bands = [
            vv.rename("vv"),
            vh.rename("vh"),
            vv.subtract(vh).rename("vv_vh_difference"),
        ]
        return ee.Image.cat(radar_bands)

    s1 = ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(aoi).filterDate(
        buffered_start, end
    ).filter(ee.Filter.eq("instrumentMode", "IW")).map(process_s1)
    lst = ee.ImageCollection("MODIS/061/MOD11A1").filterBounds(aoi).filterDate(
        start, end
    ).map(lambda image: ee.Image(image).select("LST_Day_1km").multiply(0.02).subtract(273.15).rename("lst_celsius"))
    smap = ee.ImageCollection("NASA/SMAP/SPL4SMGP/008").filterBounds(aoi).filterDate(
        start, end
    ).map(lambda image: ee.Image(image).select("sm_surface").rename("smap"))
    albedo = ee.ImageCollection("MODIS/061/MCD43A3").filterBounds(aoi).filterDate(
        buffered_start, end
    ).map(lambda image: ee.Image(image).select("Albedo_BSA_shortwave").multiply(0.001).rename("albedo"))
    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterBounds(aoi).filterDate(
        start, end
    ).map(
        lambda image: ee.Image.cat(
            [
                ee.Image(image).select("total_precipitation_sum").rename("precipitation_mm"),
                ee.Image(image).select("temperature_2m").subtract(273.15).rename("temperature_2m"),
                ee.Image(image).select("dewpoint_temperature_2m").subtract(273.15).rename(
                    "dewpoint_temperature_2m"
                ),
            ]
        )
    )
    soil = ee.Image.cat(
        [
            ee.Image("projects/soilgrids-isric/clay_mean").select("clay_0-5cm_mean").rename("clay_pct"),
            ee.Image("projects/soilgrids-isric/sand_mean").select("sand_0-5cm_mean").rename("sand_pct"),
            ee.Image("projects/soilgrids-isric/silt_mean").select("silt_0-5cm_mean").rename("silt_pct"),
        ]
    )
    dem = ee.Image("NASA/NASADEM_HGT/001").select("elevation")
    topo = ee.Image.cat(
        [dem.rename("elevation_m"), ee.Terrain.slope(dem).rename("slope_deg"), ee.Terrain.aspect(dem).rename("aspect_deg")]
    )

    if label_dates is None:
        dates = ee.List.sequence(0, end.difference(start, "day").subtract(1)).map(
            lambda offset: start.advance(offset, "day")
        )
    else:
        dates = ee.List([ee.Date(day) for day in sorted(label_dates)])

    # Preserve the Spain model's spatial feature shape. Derived meteorological
    # channels are computed from the existing ERA5 collection, so no extra
    # Earth Engine source or exported temperature/dewpoint columns are needed.
    radar_select_bands = ["vv", "vh", "vv_vh_difference"]
    patch_bands = [
        "ndvi",
        "vv",
        "vh",
        "vv_vh_difference",
        "lst_celsius",
        "smap",
        "precipitation_mm",
        "vpd_kpa",
        "antecedent_precipitation_mm",
        "albedo",
    ]
    scalar_bands: list[str] = []

    def build_stack(current: object) -> object:
        current = ee.Date(current)
        precipitation = daily(era5, ["precipitation_mm"], current)
        era_window = safe_median(
            era5.filterDate(current.advance(-6, "day"), current.advance(1, "day")),
            ["temperature_2m", "dewpoint_temperature_2m"],
        )
        temperature = era_window.select("temperature_2m")
        dewpoint = era_window.select("dewpoint_temperature_2m")
        saturation_vapor_pressure = temperature.expression(
            "0.6108 * exp((17.27 * T) / (T + 237.3))", {"T": temperature}
        )
        actual_vapor_pressure = dewpoint.expression(
            "0.6108 * exp((17.27 * Td) / (Td + 237.3))", {"Td": dewpoint}
        )
        vpd = saturation_vapor_pressure.subtract(actual_vapor_pressure).rename("vpd_kpa")
        dynamic = ee.Image.cat(
            [
                window(ndvi, ["ndvi"], current, 30),
                window(s1, radar_select_bands, current, 12),
                window(lst, ["lst_celsius"], current, 7),
                daily(smap, ["smap"], current),
                precipitation,
                vpd,
                antecedent_precipitation(current),
                window(albedo, ["albedo"], current, 16),
            ]
        )
        base = ee.Image.cat([dynamic, soil, topo]).clip(
            aoi.buffer(contract.patch_size * contract.grid_scale_m / 2)
        )
        patches = base.select(patch_bands).neighborhoodToArray(
            ee.Kernel.square(radius=contract.patch_size // 2)
        )
        scalars = base.select(scalar_bands)
        return ee.Image.cat([patches, scalars]).set(
            "system:time_start", current.millis()
        )

    collection = ee.ImageCollection(dates.map(build_stack))

    def sample(image: object) -> object:
        sample_date = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd")
        sample_points = (
            dated_points.filter(ee.Filter.eq("sample_date", sample_date))
            if label_cells_by_date is not None
            else points
        )
        return ee.Image(image).sampleRegions(
            collection=sample_points,
            properties=["station_id", "cell_id", "network"],
            scale=contract.grid_scale_m,
            projection=export_projection,
            tileScale=tile_scale,
            geometries=False,
        ).map(lambda feature: feature.set("timestamp", image.get("system:time_start")))

    dataset = collection.map(sample).flatten()
    selectors = [
        *patch_bands,
        *scalar_bands,
        "station_id",
        "cell_id",
        "network",
        "timestamp",
    ]
    return ee.batch.Export.table.toDrive(
        collection=dataset,
        description=task.description,
        folder=task.drive_folder,
        fileNamePrefix=task.file_prefix,
        fileFormat="TFRecord",
        selectors=selectors,
    )


def _build_static_export(
    ee: object,
    stations: list[Station],
    task: ExportTask,
    *,
    tile_scale: int = 16,
) -> object:
    """Export static features once per resolution cell/tile."""
    points = ee.FeatureCollection(
        [
            ee.Feature(
                ee.Geometry.Point([station.longitude, station.latitude]),
                {"station_id": station.station_id, "cell_id": station.cell_id, "network": station.network},
            )
            for station in stations
        ]
    )
    dem = ee.Image("NASA/NASADEM_HGT/001").select("elevation")
    static = ee.Image.cat(
        [
            ee.Image("projects/soilgrids-isric/clay_mean").select("clay_0-5cm_mean").rename("clay_pct"),
            ee.Image("projects/soilgrids-isric/sand_mean").select("sand_0-5cm_mean").rename("sand_pct"),
            ee.Image("projects/soilgrids-isric/silt_mean").select("silt_0-5cm_mean").rename("silt_pct"),
            dem.rename("elevation_m"),
            ee.Terrain.slope(dem).rename("slope_deg"),
            ee.Terrain.aspect(dem).rename("aspect_deg"),
        ]
    )
    return ee.batch.Export.table.toDrive(
        collection=static.sampleRegions(
            collection=points,
            properties=["station_id", "cell_id", "network"],
            scale=1000,
            projection=ee.Projection(EARTH_ENGINE_GLOBAL_CRS),
            tileScale=tile_scale,
            geometries=False,
        ),
        description=task.description,
        folder=task.drive_folder,
        fileNamePrefix=task.file_prefix,
        fileFormat="TFRecord",
        selectors=["clay_pct", "sand_pct", "silt_pct", "elevation_m", "slope_deg", "aspect_deg",
                   "station_id", "cell_id", "network"],
    )


def submit_tasks(
    tasks: list[ExportTask],
    tiles: dict[str, list[Station]],
    *,
    project: str | None,
    max_tasks: int | None,
    authenticate: bool = False,
    label_dates: dict[str, set[str]] | None = None,
    label_cells_by_date: dict[str, dict[str, set[str]]] | None = None,
    tile_scale: int = 16,
    s1_speckle_radius: int = 0,
) -> None:
    try:
        import ee
    except ImportError as error:
        raise RuntimeError("Install earthengine-api to submit exports, or use --dry-run") from error
    if authenticate:
        ee.Authenticate()
    if project:
        if project in PROJECT_PLACEHOLDERS:
            raise ValueError(
                f"--ee-project is still a placeholder ({project!r}); "
                "pass the Google Cloud project ID that has Earth Engine enabled"
            )
        ee.Initialize(project=project)
    else:
        ee.Initialize()
    selected_tasks = tasks if max_tasks is None else tasks[:max_tasks]
    stations_by_tile = {tile_id: items for tile_id, items in tiles.items()}
    for task in selected_tasks:
        tile_stations = stations_by_tile[task.tile_id]
        if task.export_kind == "static":
            export = _build_static_export(ee, tile_stations, task, tile_scale=tile_scale)
            export.start()
            print(f"Started {task.task_id}: {export.id}")
            continue
        tile_dates = (
            {
                day
                for station in tile_stations
                for day in (label_dates or {}).get(station.cell_id, set())
                if task.start_date <= day < task.end_date
            }
            if label_dates is not None
            else None
        )
        tile_cells_by_date = (
            {
                day: cells
                for day, cells in (label_cells_by_date or {}).get(task.tile_id, {}).items()
                if task.start_date <= day < task.end_date
            }
            if label_cells_by_date is not None
            else None
        )
        export = _build_ee_export(
            ee,
            tile_stations,
            task,
            tile_dates,
            tile_cells_by_date,
            tile_scale=tile_scale,
            s1_speckle_radius=s1_speckle_radius,
        )
        export.start()
        print(f"Started {task.task_id}: {export.id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("station_manifest", type=Path, help="Catalog or split manifest CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("global_gee_exports"))
    parser.add_argument("--max-stations-per-tile", type=int, default=100)
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2015, 4, 1))
    parser.add_argument("--end-date", type=date.fromisoformat, default=date(2026, 10, 3))
    parser.add_argument("--drive-folder", default="global_ismn_exports")
    parser.add_argument(
        "--window-years",
        type=int,
        default=1,
        help="Years per Earth Engine export window (default: 1)",
    )
    parser.add_argument("--ee-project")
    parser.add_argument("--year", type=int, help="Export only this calendar year")
    parser.add_argument(
        "--label-dates",
        type=Path,
        required=True,
        help="CSV from build_global_label_dates.py; required to avoid exporting unlabeled dates",
    )
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument(
        "--skip-static",
        action="store_true",
        help="Do not recreate static features; use this for subsequent yearly runs",
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Submit static features and no daily exports; use this once before yearly runs",
    )
    parser.add_argument(
        "--tile-scale",
        type=int,
        default=16,
        help="Earth Engine sampling tileScale; higher values reduce worker memory use (default: 16)",
    )
    parser.add_argument(
        "--s1-speckle-radius",
        type=int,
        default=0,
        help="Apply focal-median speckle filtering with this radius; 0 relies on temporal medians",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.skip_static and args.static_only:
        parser.error("--skip-static and --static-only cannot be used together")
    stations = read_stations(args.station_manifest)
    stations, collision_map = collapse_resolution_cells(stations)
    tiles = adaptive_tiles(stations, args.max_stations_per_tile)
    label_dates: dict[str, set[str]] = {}
    label_cells_by_date: dict[str, dict[str, set[str]]] = {}
    label_rows: list[dict[str, str]] = []
    with args.label_dates.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label_dates.setdefault(row["cell_id"], set()).add(row["sample_date"])
            label_rows.append(row)
    if not label_rows:
        raise SystemExit("The label-date manifest contains no usable rows")
    tasks = build_tasks(
        tiles,
        args.start_date,
        args.end_date,
        args.drive_folder,
        years=args.window_years,
        only_year=args.year,
        label_dates=label_dates,
        include_static=not args.skip_static,
        static_only=args.static_only,
    )
    cells_by_date: dict[str, set[str]] = {}
    for row in label_rows:
        cells_by_date.setdefault(row["sample_date"], set()).add(row["cell_id"])
    label_cells_by_date = {
        tile_id: {
            day: {station.cell_id for station in stations_in_tile if station.cell_id in cells}
            for day, cells in cells_by_date.items()
        }
        for tile_id, stations_in_tile in tiles.items()
    }
    write_manifest(
        tasks,
        tiles,
        args.output_dir,
        start=args.start_date,
        end=args.end_date,
        maximum=args.max_stations_per_tile,
        collision_map=collision_map,
    )
    print(f"Wrote {len(tiles)} adaptive tiles and {len(tasks)} yearly export tasks to {args.output_dir}")
    if not args.dry_run:
        submit_tasks(
            tasks,
            tiles,
            project=args.ee_project,
            max_tasks=args.max_tasks,
            label_dates=label_dates,
            label_cells_by_date=label_cells_by_date,
            tile_scale=args.tile_scale,
            s1_speckle_radius=args.s1_speckle_radius,
        )


if __name__ == "__main__":
    main()
