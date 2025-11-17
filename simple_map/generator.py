"""地图生成器核心类。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from . import noise, rendering


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

    # 侵蚀模拟参数
    enable_erosion: bool = False
    erosion_iterations: int = 100
    erosion_rain_rate: float = 0.0008
    erosion_evaporation_rate: float = 0.0005
    erosion_min_height_delta: float = 0.05
    erosion_repose_slope: float = 0.03
    erosion_gravity: float = 30.0
    erosion_sediment_capacity: float = 50.0
    erosion_dissolving_rate: float = 0.25
    erosion_deposition_rate: float = 0.001
    erosion_cell_width: float = 1.0

    # 侵蚀快照参数
    erosion_snapshot_interval: int = 0  # 0表示不保存快照
    erosion_snapshots_dir: Optional[Path] = None  # 快照保存目录

    # 渲染相关参数
    render_mode: str = "classic"
    gradient_preset: str = "ocean_land"
    light_direction: Tuple[float, float, float] = (-0.2, -0.5, 0.7)
    ambient_light: float = 0.35
    normal_strength: float = 6.0


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

        # 应用侵蚀模拟(如果启用)
        if cfg.enable_erosion:
            from .erosion import ErosionSimulator, ErosionConfig

            erosion_config = ErosionConfig(
                iterations=cfg.erosion_iterations,
                rain_rate=cfg.erosion_rain_rate,
                evaporation_rate=cfg.erosion_evaporation_rate,
                min_height_delta=cfg.erosion_min_height_delta,
                repose_slope=cfg.erosion_repose_slope,
                gravity=cfg.erosion_gravity,
                sediment_capacity_constant=cfg.erosion_sediment_capacity,
                dissolving_rate=cfg.erosion_dissolving_rate,
                deposition_rate=cfg.erosion_deposition_rate,
                cell_width=cfg.erosion_cell_width,
                sea_level=cfg.sea_level,
            )

            simulator = ErosionSimulator(erosion_config)

            # 准备快照回调（如果需要）
            callback = None
            if cfg.erosion_snapshot_interval > 0 and cfg.erosion_snapshots_dir:
                snapshots_dir = Path(cfg.erosion_snapshots_dir)
                snapshots_dir.mkdir(parents=True, exist_ok=True)

                def snapshot_callback(step: int, terrain: np.ndarray) -> None:
                    """在特定步骤保存快照"""
                    if step % cfg.erosion_snapshot_interval == 0:
                        # 归一化地形
                        normalized = (terrain - terrain.min()) / (np.ptp(terrain) + 1e-6)
                        normalized = normalized.astype(np.float32)

                        # 保存彩色图
                        color_path = snapshots_dir / f"erosion_step{step:04d}.png"
                        self.render_map(normalized, color_path)

                        # 保存高度图
                        height_path = snapshots_dir / f"erosion_step{step:04d}_height.png"
                        self.save_heightmap(normalized, height_path)

                        print(f"  已保存快照: 步骤 {step}")

                callback = snapshot_callback

            heightmap = simulator.simulate(heightmap, verbose=True, callback=callback)

            # 重新归一化
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

    def render_map(self, heightmap: np.ndarray, output: Path, *, dpi: int = 150) -> None:
        mode = (self.config.render_mode or "classic").lower()
        if mode == "realistic":
            gradient = rendering.resolve_gradient(
                self.config.gradient_preset,
                self.config.sea_level,
                self.config.snow_level,
            )
            rendering.render_realistic(
                heightmap,
                output,
                gradient=gradient,
                sea_level=self.config.sea_level,
                light_direction=self.config.light_direction,
                ambient_light=self.config.ambient_light,
                normal_strength=self.config.normal_strength,
            )
            return

        colors = self.classify_terrain(heightmap)
        self.render(colors, heightmap, output, dpi=dpi)
