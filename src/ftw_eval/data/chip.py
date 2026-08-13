from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Self

import numpy as np


# Canonical Sentinel-2 L2A band order (12 bands, excludes B10 which is L1C-only).
# TerraMind requires all 12; Clay/Prithvi use a subset via `select_s2_bands`.
S2_BAND_ORDER: list[str] = [
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B09", "B11", "B12",
]

S1_BAND_ORDER: list[str] = ["VV", "VH"]


@dataclass
class Chip:
    """A geospatial patch with optional multi-modality and temporal axes.

    Shape convention:
        s2:  [T, C, H, W] when ``has_time`` else [C, H, W]; C ordered by ``s2_bands``.
        s1:  same convention; bands by ``s1_bands``.
        dem: [H, W] (single-time, single-channel).

    Reflectance values are scaled to the [0, 1] range (i.e. raw L2A divided by 10_000).
    Wrappers are responsible for any further per-model normalization.
    """

    s2: np.ndarray | None = None
    s2_bands: list[str] = field(default_factory=lambda: list(S2_BAND_ORDER))
    s1: np.ndarray | None = None
    s1_bands: list[str] = field(default_factory=lambda: list(S1_BAND_ORDER))
    dem: np.ndarray | None = None

    lat: float = 0.0
    lon: float = 0.0
    date: datetime | None = None
    dates: list[datetime] | None = None
    gsd_m: float = 10.0
    crs: str = "EPSG:4326"
    transform: tuple[float, float, float, float, float, float] | None = None

    @property
    def has_time(self) -> bool:
        if self.s2 is not None:
            return self.s2.ndim == 4
        if self.s1 is not None:
            return self.s1.ndim == 4
        return self.dates is not None and len(self.dates) > 1

    @property
    def height_px(self) -> int:
        for arr in (self.s2, self.s1):
            if arr is not None:
                return arr.shape[-2]
        if self.dem is not None:
            return self.dem.shape[-2]
        raise ValueError("Chip has no spatial array")

    @property
    def width_px(self) -> int:
        for arr in (self.s2, self.s1):
            if arr is not None:
                return arr.shape[-1]
        if self.dem is not None:
            return self.dem.shape[-1]
        raise ValueError("Chip has no spatial array")

    @property
    def height_m(self) -> float:
        return self.height_px * self.gsd_m

    @property
    def width_m(self) -> float:
        return self.width_px * self.gsd_m

    def select_s2_bands(self, names: list[str]) -> np.ndarray:
        if self.s2 is None:
            raise ValueError("Chip has no S2 data")
        try:
            idx = [self.s2_bands.index(n) for n in names]
        except ValueError as e:
            missing = [n for n in names if n not in self.s2_bands]
            raise ValueError(f"S2 bands missing from chip: {missing}") from e
        return self.s2[..., idx, :, :] if self.has_time else self.s2[idx]

    def validate(self) -> None:
        if self.s2 is not None:
            if self.s2.ndim not in (3, 4):
                raise ValueError(f"s2 must be 3D or 4D, got shape {self.s2.shape}")
            c = self.s2.shape[-3]
            if c != len(self.s2_bands):
                raise ValueError(
                    f"s2 has {c} channels but s2_bands has {len(self.s2_bands)} entries"
                )
        if self.s1 is not None:
            if self.s1.ndim not in (3, 4):
                raise ValueError(f"s1 must be 3D or 4D, got shape {self.s1.shape}")
            c = self.s1.shape[-3]
            if c != len(self.s1_bands):
                raise ValueError(
                    f"s1 has {c} channels but s1_bands has {len(self.s1_bands)} entries"
                )
        if self.dem is not None and self.dem.ndim != 2:
            raise ValueError(f"dem must be 2D, got shape {self.dem.shape}")
        if not (-90.0 <= self.lat <= 90.0):
            raise ValueError(f"lat out of range: {self.lat}")
        if not (-180.0 <= self.lon <= 180.0):
            raise ValueError(f"lon out of range: {self.lon}")

    @classmethod
    def synthetic(
        cls,
        size_px: int = 224,
        with_s1: bool = True,
        with_dem: bool = True,
        time_steps: int | None = None,
        seed: int = 0,
    ) -> Self:
        """Random chip for smoke testing. Values are in [0, 1]."""
        rng = np.random.default_rng(seed)
        if time_steps is None:
            s2 = rng.random((len(S2_BAND_ORDER), size_px, size_px), dtype=np.float32)
            s1 = rng.random((len(S1_BAND_ORDER), size_px, size_px), dtype=np.float32) if with_s1 else None
        else:
            s2 = rng.random(
                (time_steps, len(S2_BAND_ORDER), size_px, size_px), dtype=np.float32
            )
            s1 = (
                rng.random((time_steps, len(S1_BAND_ORDER), size_px, size_px), dtype=np.float32)
                if with_s1
                else None
            )
        dem = rng.random((size_px, size_px), dtype=np.float32) * 500.0 if with_dem else None
        return cls(
            s2=s2,
            s1=s1,
            dem=dem,
            lat=20.0,
            lon=78.0,
            date=datetime(2024, 6, 15),
            gsd_m=10.0,
            crs="EPSG:32643",
        )

    @classmethod
    def from_geotiff(cls, path: str | Path) -> Self:
        """Load a Sentinel-2 chip from a single multi-band GeoTIFF (bands ordered as S2_BAND_ORDER)."""
        import rasterio
        from rasterio.warp import transform as warp_transform

        path = Path(path)
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32) / 10_000.0
            cy, cx = src.height // 2, src.width // 2
            x, y = src.xy(cy, cx)
            lon_arr, lat_arr = warp_transform(src.crs, "EPSG:4326", [x], [y])
            return cls(
                s2=arr,
                s2_bands=list(S2_BAND_ORDER[: arr.shape[0]]),
                lat=lat_arr[0],
                lon=lon_arr[0],
                gsd_m=abs(src.transform.a),
                crs=str(src.crs),
                transform=tuple(src.transform)[:6],  # type: ignore[arg-type]
            )
