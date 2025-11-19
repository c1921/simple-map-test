"""地图生成器核心类。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

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
    edge_falloff_margin: float = 0.08
    edge_falloff_power: float = 1.8

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
    output_scale: float = 1.0
    output_interpolation: str = "bicubic"
    output_detail_noise_strength: float = 0.0
    output_detail_noise_scale: float = 24.0
    output_detail_noise_octaves: int = 3
    output_detail_noise_persistence: float = 0.55
    output_detail_noise_lacunarity: float = 2.2
    output_detail_noise_min_level: float = 0.0
    output_detail_noise_full_level: float = 1.0
    enable_domain_warp: bool = False
    domain_warp_strength: float = 60.0
    domain_warp_scale: float = 250.0
    domain_warp_octaves: int = 3
    domain_warp_persistence: float = 0.5
    domain_warp_lacunarity: float = 2.0
    preview_base_noise: bool = False
    pre_erosion_map: Optional[Path] = None
    pre_erosion_heightmap: Optional[Path] = None
    pre_detail_map: Optional[Path] = None
    pre_detail_heightmap: Optional[Path] = None


class MapGenerator:
    """生成高度图、地形分类并负责渲染。"""

    def __init__(self, config: MapGeneratorConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        self._detail_noise_seed = self._init_detail_noise_seed()
        self._pre_erosion_heightmap: Optional[np.ndarray] = None

    def generate_heightmap(self) -> np.ndarray:
        # 记录总开始时间
        total_start = time.time()

        cfg = self.config

        # 1. 生成噪声
        print("开始生成噪声 (Ridged Multifractal)...")
        noise_start = time.time()
        ridged = noise.ridged_multifractal(
            cfg.width,
            cfg.height,
            octaves=cfg.octaves,
            persistence=cfg.persistence,
            lacunarity=cfg.lacunarity,
            base_scale=cfg.base_scale,
            seed=self._rng.integers(0, 2**32 - 1),
        )
        ridged = self._apply_domain_warp(ridged)
        mask = noise.continent_mask(cfg.width, cfg.height)
        heightmap = ridged * 0.65 + mask * 0.35
        heightmap = (heightmap - heightmap.min()) / (np.ptp(heightmap) + 1e-6)
        noise_time = time.time() - noise_start
        print(f"噪声生成完成，耗时: {noise_time:.2f}s")

        # 2. 边缘衰减
        edge_mask: Optional[np.ndarray] = None
        if cfg.edge_falloff_power > 0:
            print("应用边缘衰减...")
            edge_start = time.time()
            edge_mask = noise.edge_falloff_mask(
                cfg.width,
                cfg.height,
                margin=cfg.edge_falloff_margin,
                exponent=cfg.edge_falloff_power,
            )
            heightmap *= edge_mask
            edge_time = time.time() - edge_start
            print(f"边缘衰减完成，耗时: {edge_time:.2f}s")

        self._pre_erosion_heightmap = heightmap.copy()
        if cfg.preview_base_noise:
            total_time = time.time() - total_start
            print(f"\n=== 地形生成总耗时: {total_time:.2f}s ===\n")
            return heightmap.astype(np.float32)

        # 应用侵蚀模拟(如果启用)
        if cfg.enable_erosion:
            print("启动侵蚀模拟...")
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
            print("归一化侵蚀后的地形...")
            norm_start = time.time()
            heightmap = (heightmap - heightmap.min()) / (np.ptp(heightmap) + 1e-6)
            if edge_mask is not None:
                heightmap *= edge_mask
            norm_time = time.time() - norm_start
            print(f"归一化完成，耗时: {norm_time:.2f}s")

        # 计算总耗时
        total_time = time.time() - total_start
        print(f"\n=== 地形生成总耗时: {total_time:.2f}s ===\n")

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
        height, width = colors.shape[:2]
        fig_w = width / dpi
        fig_h = height / dpi
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        ax.imshow(colors, origin="lower")
        ax.contour(heightmap, levels=[self.config.sea_level], colors="#102542", linewidths=0.75)
        land_levels = np.linspace(self.config.sea_level + 0.05, 0.95, 4)
        ax.contour(heightmap, levels=land_levels, colors="black", linewidths=0.3, alpha=0.35)
        ax.axis("off")
        fig.tight_layout(pad=0)
        fig.savefig(output, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    def save_heightmap(
        self,
        heightmap: np.ndarray,
        path: Path,
        *,
        apply_output_scale: bool = True,
        log_label: str | None = None,
    ) -> None:
        start = time.time()
        if apply_output_scale:
            heightmap = self.prepare_heightmap_for_output(heightmap)
        plt.imsave(path, heightmap, cmap="gray", origin="lower")
        duration = time.time() - start
        label = log_label or f"高度图保存 ({path})"
        print(f"{label}耗时: {duration:.2f}s")

    def render_map(
        self,
        heightmap: np.ndarray,
        output: Path,
        *,
        dpi: int = 150,
        apply_output_scale: bool = True,
        log_label: str | None = None,
    ) -> None:
        start = time.time()
        if apply_output_scale:
            heightmap = self.prepare_heightmap_for_output(heightmap)
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
        duration = time.time() - start
        label = log_label or f"渲染输出 ({output})"
        print(f"{label}耗时: {duration:.2f}s")

    def prepare_heightmap_for_output(
        self,
        heightmap: np.ndarray,
        *,
        apply_detail_noise: bool = True,
    ) -> np.ndarray:
        scaled = self._scale_heightmap_for_output(heightmap)
        detail_strength = float(self.config.output_detail_noise_strength)
        if apply_detail_noise and detail_strength > 0.0 and self._detail_noise_seed is not None:
            scaled = self._apply_output_detail_noise(scaled, detail_strength)
        return np.clip(scaled, 0.0, 1.0)

    def apply_output_detail_noise_to_scaled(self, scaled_heightmap: np.ndarray) -> np.ndarray:
        detail_strength = float(self.config.output_detail_noise_strength)
        if detail_strength <= 0.0 or self._detail_noise_seed is None:
            return scaled_heightmap
        working = scaled_heightmap.astype(np.float32, copy=True)
        working = self._apply_output_detail_noise(working, detail_strength)
        return np.clip(working, 0.0, 1.0)

    def _scale_heightmap_for_output(self, heightmap: np.ndarray) -> np.ndarray:
        scale = float(self.config.output_scale)
        if scale <= 0:
            raise ValueError("output_scale 必须大于 0")
        needs_scaling = abs(scale - 1.0) >= 1e-6
        working = heightmap.astype(np.float32, copy=True)
        if not needs_scaling:
            return working
        h, w = working.shape[:2]
        target_h = max(1, int(round(h * scale)))
        target_w = max(1, int(round(w * scale)))
        zoom_h = target_h / h
        zoom_w = target_w / w
        order = _resolve_interpolation_order(self.config.output_interpolation)
        zoom_factors = (zoom_h, zoom_w)
        working = ndimage.zoom(
            working,
            zoom_factors,
            order=order,
            mode="reflect",
        ).astype(np.float32, copy=False)
        return working

    def _apply_output_detail_noise(self, data: np.ndarray, strength: float) -> np.ndarray:
        height, width = data.shape[:2]
        octaves = max(1, int(self.config.output_detail_noise_octaves))
        base_scale = max(float(self.config.output_detail_noise_scale), 1.0)
        persistence = float(self.config.output_detail_noise_persistence)
        lacunarity = float(self.config.output_detail_noise_lacunarity)
        seed = self._detail_noise_seed_for_shape(width, height)
        detail = noise.fractal_noise(
            width,
            height,
            octaves=octaves,
            persistence=persistence,
            lacunarity=lacunarity,
            base_scale=base_scale,
            seed=seed,
        ).astype(np.float32)
        detail -= float(np.mean(detail))
        max_abs = float(np.max(np.abs(detail))) + 1e-6
        detail /= max_abs
        weights = self._detail_noise_weights(data)
        if np.all(weights <= 0.0):
            return data
        weighted_detail = detail * weights
        return np.clip(data + strength * weighted_detail, 0.0, 1.0)

    def _init_detail_noise_seed(self) -> Optional[int]:
        if self.config.output_detail_noise_strength <= 0:
            return None
        base_seed = self.config.seed
        if base_seed is None:
            base_seed = int(np.random.default_rng().integers(0, 2**32 - 1))
        detail_seed = int((int(base_seed) ^ 0xA511E9B3) & 0xFFFFFFFF)
        if detail_seed == 0:
            detail_seed = 1
        return detail_seed

    def _detail_noise_seed_for_shape(self, width: int, height: int) -> int:
        base = self._detail_noise_seed or 1
        mix = (base ^ (width * 0x9E3779B1) ^ (height * 0x85EBCA77)) & 0xFFFFFFFF
        if mix == 0:
            mix = base
        return mix

    def _detail_noise_weights(self, heightmap: np.ndarray) -> np.ndarray:
        min_level = float(np.clip(self.config.output_detail_noise_min_level, 0.0, 1.0))
        full_level = float(np.clip(self.config.output_detail_noise_full_level, 0.0, 1.0))
        epsilon = 1e-6
        if full_level <= min_level + epsilon:
            weights = (heightmap >= full_level).astype(np.float32)
        else:
            weights = (heightmap - min_level) / (full_level - min_level)
            weights = np.clip(weights, 0.0, 1.0)
        return weights.astype(np.float32, copy=False)

    def _apply_domain_warp(self, values: np.ndarray) -> np.ndarray:
        cfg = self.config
        if not cfg.enable_domain_warp or cfg.domain_warp_strength <= 0:
            return values
        height, width = values.shape
        octaves = max(1, int(cfg.domain_warp_octaves))
        persistence = float(cfg.domain_warp_persistence)
        lacunarity = float(cfg.domain_warp_lacunarity)
        scale = max(float(cfg.domain_warp_scale), 1.0)
        seed_x = int(self._rng.integers(0, 2**32 - 1))
        seed_y = int(self._rng.integers(0, 2**32 - 1))
        offset_x = noise.fractal_noise(
            width,
            height,
            octaves=octaves,
            persistence=persistence,
            lacunarity=lacunarity,
            base_scale=scale,
            seed=seed_x,
        ).astype(np.float32)
        offset_y = noise.fractal_noise(
            width,
            height,
            octaves=octaves,
            persistence=persistence,
            lacunarity=lacunarity,
            base_scale=scale,
            seed=seed_y,
        ).astype(np.float32)
        offset_x -= offset_x.mean()
        offset_y -= offset_y.mean()
        offset_x /= np.maximum(np.max(np.abs(offset_x)), 1e-6)
        offset_y /= np.maximum(np.max(np.abs(offset_y)), 1e-6)
        strength = float(cfg.domain_warp_strength)
        grid_y, grid_x = np.meshgrid(
            np.arange(height, dtype=np.float32),
            np.arange(width, dtype=np.float32),
            indexing="ij",
        )
        sample_y = grid_y + offset_y * strength
        sample_x = grid_x + offset_x * strength
        warped = ndimage.map_coordinates(
            values,
            [sample_y, sample_x],
            order=1,
            mode="reflect",
        )
        return warped.astype(np.float32, copy=False)

    @property
    def pre_erosion_heightmap(self) -> Optional[np.ndarray]:
        return None if self._pre_erosion_heightmap is None else self._pre_erosion_heightmap.copy()


def _resolve_interpolation_order(method: str | None) -> int:
    lookup = {
        "nearest": 0,
        "linear": 1,
        "bilinear": 1,
        "quadratic": 2,
        "bicubic": 3,
        "cubic": 3,
        "quartic": 4,
        "quintic": 5,
    }
    key = (method or "bicubic").strip().lower()
    return lookup.get(key, 3)
