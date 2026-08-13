"""Label data structures for the starter dataset.

Two scales of label live here:

1. **Pixel labels** — per-pixel land cover (e.g. ESA WorldCover class codes).
   Stored as single-band uint8 GeoTIFFs aligned to the chip.
2. **Object labels** — per-field polygons extracted from the pixel mask via
   watershed + morphology (or SAM2 later). Stored as geoparquet so
   geopandas can read them with a single line.

The ``FieldSizeBin`` enum is the controlled axis for the entire paper: all evaluation metrics will be stratified by these bins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable


class FieldSizeBin(str, Enum):
    """Field-size bins used throughout the benchmark.

    Boundaries are in hectares. Picked to match smallholder-ag literature:
    <0.1 ha is the "very small" tier reported in Marshak et al. 2025;
    0.1-0.5 ha covers most of the most smallholder regions globally;
    >0.5 ha is "small-but-mappable"; >1 ha is "industrial-comparable".
    """

    UNDER_0_1 = "<0.1ha"
    BIN_0_1_TO_0_3 = "0.1-0.3ha"
    BIN_0_3_TO_0_5 = "0.3-0.5ha"
    BIN_0_5_TO_1 = "0.5-1ha"
    ABOVE_1 = ">1ha"

    @classmethod
    def from_area_m2(cls, area_m2: float) -> "FieldSizeBin":
        ha = area_m2 / 10_000.0
        if ha < 0.1:
            return cls.UNDER_0_1
        if ha < 0.3:
            return cls.BIN_0_1_TO_0_3
        if ha < 0.5:
            return cls.BIN_0_3_TO_0_5
        if ha < 1.0:
            return cls.BIN_0_5_TO_1
        return cls.ABOVE_1

    @classmethod
    def ordered(cls) -> list["FieldSizeBin"]:
        return [cls.UNDER_0_1, cls.BIN_0_1_TO_0_3, cls.BIN_0_3_TO_0_5, cls.BIN_0_5_TO_1, cls.ABOVE_1]


# ESA WorldCover v200 class codes (https://esa-worldcover.org).
WORLDCOVER_CLASSES: dict[int, str] = {
    10: "tree_cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built_up",
    60: "bare_or_sparse",
    70: "snow_ice",
    80: "permanent_water",
    90: "herbaceous_wetland",
    95: "mangroves",
    100: "moss_lichen",
}
WORLDCOVER_CROPLAND_CODE: int = 40


@dataclass(frozen=True)
class RegionTag:
    """Coarse-grained region identifier carried through every chip."""

    country: str
    province: str | None
    district: str


@dataclass
class ChipManifestEntry:
    """One row in the chip manifest. Bridges index -> imagery -> labels.

    Path fields are relative to the data root so the manifest survives moves.
    """

    chip_id: str
    region: RegionTag
    lon: float
    lat: float
    gsd_m: float
    chip_size_px: int
    scene_id: str
    scene_date: str
    cloud_pct: float
    s2_path: Path
    s1_path: Path | None
    worldcover_path: Path | None

    def to_dict(self) -> dict:
        d = {
            "chip_id": self.chip_id,
            "country": self.region.country,
            "province": self.region.province,
            "district": self.region.district,
            "lon": self.lon,
            "lat": self.lat,
            "gsd_m": self.gsd_m,
            "chip_size_px": self.chip_size_px,
            "scene_id": self.scene_id,
            "scene_date": self.scene_date,
            "cloud_pct": self.cloud_pct,
            "s2_path": str(self.s2_path),
            "s1_path": str(self.s1_path) if self.s1_path else None,
            "worldcover_path": str(self.worldcover_path) if self.worldcover_path else None,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ChipManifestEntry":
        return cls(
            chip_id=d["chip_id"],
            region=RegionTag(country=d.get("country", ""),
                             province=d.get("province"),
                             district=d["district"]),
            lon=d["lon"], lat=d["lat"], gsd_m=d["gsd_m"],
            chip_size_px=d["chip_size_px"],
            scene_id=d["scene_id"], scene_date=d["scene_date"], cloud_pct=d["cloud_pct"],
            s2_path=Path(d["s2_path"]),
            s1_path=Path(d["s1_path"]) if d.get("s1_path") else None,
            worldcover_path=Path(d["worldcover_path"]) if d.get("worldcover_path") else None,
        )


def write_manifest(entries: Iterable[ChipManifestEntry], out: Path) -> None:
    import json
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for e in entries:
            f.write(json.dumps(e.to_dict()) + "\n")


def read_manifest(path: Path) -> list[ChipManifestEntry]:
    import json
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(ChipManifestEntry.from_dict(json.loads(line)))
    return rows


# ---- Field polygons ---------------------------------------------------------


POLYGON_COLUMNS: list[str] = [
    "chip_id", "district", "polygon_id",
    "area_m2", "size_bin",
    "centroid_lon", "centroid_lat",
    "geometry",
]


def empty_polygon_frame():
    """Return an empty geopandas.GeoDataFrame with the canonical schema."""
    import geopandas as gpd
    return gpd.GeoDataFrame(
        {c: [] for c in POLYGON_COLUMNS},
        geometry="geometry",
        crs="EPSG:4326",
    )
