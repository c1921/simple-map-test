"""地图生成器核心类。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from . import noise


@dataclass
class MapGeneratorConfig:
    width: int = 768
    height: int = 512
    seed: Optional[int] = None
    octaves: int = 5
    persistence: float = 0.5
    lacunarity: float = 2.0
    base_scale: float = 160.0
    sea_level: float = 0.45
    snow_level: float = 0.9


class MapGenerator:
    """生成高度图、地形分类并负责渲染。"""

    def __init__(self, config: MapGeneratorConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.seed)

    def generate_heightmap(self) -> np.ndarray:
        cfg = self.config
        fbm = noise.fractal_noise(
            cfg.width,
            cfg.height,
            octaves=cfg.octaves,
            persistence=cfg.persistence,
            lacunarity=cfg.lacunarity,
            base_scale=cfg.base_scale,
            seed=self._rng.integers(0, 2**32 - 1),
        )
        mask = noise.continent_mask(cfg.width, cfg.height)
        heightmap = fbm * 0.65 + mask * 0.35
        heightmap = (heightmap - heightmap.min()) / (np.ptp(heightmap) + 1e-6)
        return heightmap.astype(np.float32)

    def classify_terrain(self, heightmap: np.ndarray) -> np.ndarray:
        sea = self.config.sea_level
        snow = self.config.snow_level
        colors = np.zeros((heightmap.shape[0], heightmap.shape[1], 3), dtype=np.float32)

        def apply_mask(condition: np.ndarray, rgb: tuple[float, float, float]) -> None:
            colors[condition] = rgb

        apply_mask(heightmap < sea * 0.75, (12 / 255, 32 / 255, 89 / 255))
        apply_mask((heightmap >= sea * 0.75) & (heightmap < sea), (28 / 255, 74 / 255, 142 / 255))
        apply_mask((heightmap >= sea) & (heightmap < sea + 0.05), (222 / 255, 202 / 255, 166 / 255))
        apply_mask((heightmap >= sea + 0.05) & (heightmap < 0.65), (87 / 255, 140 / 255, 62 / 255))
        apply_mask((heightmap >= 0.65) & (heightmap < 0.8), (109 / 255, 93 / 255, 57 / 255))
        apply_mask((heightmap >= 0.8) & (heightmap < snow), (0.75, 0.75, 0.75))
        apply_mask(heightmap >= snow, (0.95, 0.95, 0.97))
        return colors

    def render(
        self,
        colors: np.ndarray,
        heightmap: np.ndarray,
        output: Path,
        *,
        dpi: int = 150,
    ) -> None:
        fig_w = self.config.width / dpi
        fig_h = self.config.height / dpi
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        ax.imshow(colors, origin="lower")
        ax.contour(heightmap, levels=[self.config.sea_level], colors="#102542", linewidths=0.75)
        land_levels = np.linspace(self.config.sea_level + 0.05, 0.95, 4)
        ax.contour(heightmap, levels=land_levels, colors="black", linewidths=0.3, alpha=0.35)
        ax.axis("off")
        fig.tight_layout(pad=0)
        fig.savefig(output, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    def save_heightmap(self, heightmap: np.ndarray, path: Path) -> None:
        plt.imsave(path, heightmap, cmap="gray", origin="lower")
