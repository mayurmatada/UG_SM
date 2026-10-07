"""Shared configuration for the first global MultiModalSoilNet data contract."""

from __future__ import annotations

from dataclasses import dataclass


SPATIAL_CHANNELS = (
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
)

STATIC_CHANNELS = (
    "clay_pct",
    "sand_pct",
    "silt_pct",
    "elevation_m",
    "slope_deg",
    "aspect_deg",
)

TARGET_DEPTH_MIN_M = 0.0
TARGET_DEPTH_MAX_M = 0.05
FALLBACK_DEPTH_MAX_M = 0.10
SOIL_MOISTURE_UNIT = "m3/m3"
SOIL_MOISTURE_MIN = 0.0
SOIL_MOISTURE_MAX = 0.6
PATCH_SIZE = 13
GRID_SCALE_M = 1000
TIMEZONE = "UTC"
QUALITY_FLAG_GOOD = "G"
CONTRACT_VERSION = "global-v2"


@dataclass(frozen=True)
class GlobalDataContract:
    """Versioned shape and label rules used by global dataset builders."""

    version: str = CONTRACT_VERSION
    patch_size: int = PATCH_SIZE
    grid_scale_m: int = GRID_SCALE_M
    target_depth_min_m: float = TARGET_DEPTH_MIN_M
    target_depth_max_m: float = TARGET_DEPTH_MAX_M
    fallback_depth_max_m: float = FALLBACK_DEPTH_MAX_M
    soil_moisture_unit: str = SOIL_MOISTURE_UNIT
    soil_moisture_min: float = SOIL_MOISTURE_MIN
    soil_moisture_max: float = SOIL_MOISTURE_MAX
    quality_flag: str = QUALITY_FLAG_GOOD
    spatial_channels: tuple[str, ...] = SPATIAL_CHANNELS
    static_channels: tuple[str, ...] = STATIC_CHANNELS

    @property
    def spatial_channel_count(self) -> int:
        return len(self.spatial_channels)

    @property
    def static_channel_count(self) -> int:
        return len(self.static_channels)

    @property
    def total_channel_count(self) -> int:
        return self.spatial_channel_count + self.static_channel_count

    def depth_class(self, depth_from_m: float, depth_to_m: float) -> str:
        """Classify a layer without silently treating deeper data as surface data."""
        if depth_to_m < depth_from_m:
            return "invalid"
        if depth_from_m >= self.target_depth_min_m and depth_to_m <= self.target_depth_max_m:
            return "near_surface"
        if depth_from_m >= self.target_depth_max_m and depth_to_m <= self.fallback_depth_max_m:
            return "fallback_deeper"
        return "excluded"

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "patch_size": self.patch_size,
            "grid_scale_m": self.grid_scale_m,
            "timezone": TIMEZONE,
            "target_depth_min_m": self.target_depth_min_m,
            "target_depth_max_m": self.target_depth_max_m,
            "fallback_depth_max_m": self.fallback_depth_max_m,
            "soil_moisture_unit": self.soil_moisture_unit,
            "soil_moisture_min": self.soil_moisture_min,
            "soil_moisture_max": self.soil_moisture_max,
            "quality_flag": self.quality_flag,
            "spatial_channels": list(self.spatial_channels),
            "static_channels": list(self.static_channels),
        }
