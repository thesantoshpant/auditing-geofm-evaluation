"""ESA WorldCover label loader.

WorldCover is a single-band uint8 raster with class codes in [10, 100].
This loader is intentionally thin: read the file, expose the array, and
provide a couple of derived views the pipeline asks for repeatedly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ftw_eval.data.labels import WORLDCOVER_CLASSES, WORLDCOVER_CROPLAND_CODE


@dataclass
class WorldCoverChip:
    classes: np.ndarray         # [H, W] uint8 class codes (10..100)
    transform: tuple            # rasterio Affine as 6-tuple
    crs: str

    @property
    def shape(self) -> tuple[int, int]:
        return self.classes.shape

    def binary_cropland_mask(self) -> np.ndarray:
        """Return uint8 mask: 1 where class == cropland (40), else 0."""
        return (self.classes == WORLDCOVER_CROPLAND_CODE).astype(np.uint8)

    def class_distribution(self) -> dict[str, float]:
        """Fraction of pixels per class label. Sums to ~1.0."""
        total = float(self.classes.size)
        out: dict[str, float] = {}
        for code, name in WORLDCOVER_CLASSES.items():
            n = int((self.classes == code).sum())
            if n > 0:
                out[name] = n / total
        return out

    def cropland_fraction(self) -> float:
        return float(self.binary_cropland_mask().mean())

    @classmethod
    def from_geotiff(cls, path: str | Path) -> "WorldCoverChip":
        import rasterio
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.uint8)
            return cls(
                classes=arr,
                transform=tuple(src.transform)[:6],  # type: ignore[arg-type]
                crs=str(src.crs),
            )
