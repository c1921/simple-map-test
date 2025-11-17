"""不同渲染模式所需的工具函数。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence, Tuple

import numpy as np
from PIL import Image

# --------------------------- 颜色梯度 ---------------------------


Color = Tuple[float, float, float]


@dataclass(frozen=True)
class GradientStop:
    position: float
    color: Color


GradientBuilder = Callable[[float, float], Sequence[GradientStop]]


def _reference_gradient(_sea_level: float, _snow_level: float) -> Sequence[GradientStop]:
    return (
        GradientStop(0.00, (20 / 255, 60 / 255, 20 / 255)),
        GradientStop(0.25, (50 / 255, 100 / 255, 40 / 255)),
        GradientStop(0.45, (90 / 255, 130 / 255, 60 / 255)),
        GradientStop(0.65, (120 / 255, 110 / 255, 70 / 255)),
        GradientStop(0.80, (160 / 255, 160 / 255, 160 / 255)),
        GradientStop(0.92, (220 / 255, 220 / 255, 220 / 255)),
        GradientStop(1.00, (1.00, 1.00, 1.00)),
    )


def _ocean_land_gradient(sea_level: float, snow_level: float) -> Sequence[GradientStop]:
    sea = float(np.clip(sea_level, 0.05, 0.95))
    snow = float(np.clip(snow_level, 0.5, 1.0))
    shore = min(sea + 0.02, 0.98)
    plains = min(sea + 0.18, 0.99)
    highlands = min(snow - 0.08, 0.995)
    return (
        GradientStop(0.00, (8 / 255, 20 / 255, 65 / 255)),
        GradientStop(max(sea * 0.35, 0.02), (17 / 255, 46 / 255, 110 / 255)),
        GradientStop(max(sea * 0.8, 0.04), (32 / 255, 86 / 255, 143 / 255)),
        GradientStop(sea, (223 / 255, 211 / 255, 180 / 255)),
        GradientStop(shore, (209 / 255, 196 / 255, 158 / 255)),
        GradientStop(plains, (97 / 255, 145 / 255, 82 / 255)),
        GradientStop(highlands, (123 / 255, 113 / 255, 73 / 255)),
        GradientStop(snow, (200 / 255, 200 / 255, 200 / 255)),
        GradientStop(1.0, (1.0, 1.0, 1.0)),
    )


def _desert_gradient(sea_level: float, snow_level: float) -> Sequence[GradientStop]:
    sea = float(np.clip(sea_level, 0.05, 0.95))
    snow = float(np.clip(snow_level, 0.5, 1.0))
    dunes = min(sea + 0.25, snow - 0.2)
    plateau = min(snow - 0.05, 0.995)
    return (
        GradientStop(0.00, (6 / 255, 18 / 255, 52 / 255)),
        GradientStop(max(sea * 0.8, 0.03), (18 / 255, 46 / 255, 96 / 255)),
        GradientStop(sea, (220 / 255, 210 / 255, 170 / 255)),
        GradientStop(dunes, (207 / 255, 171 / 255, 113 / 255)),
        GradientStop(plateau, (158 / 255, 126 / 255, 82 / 255)),
        GradientStop(snow, (210 / 255, 210 / 255, 210 / 255)),
        GradientStop(1.0, (1.0, 1.0, 1.0)),
    )


GRADIENT_PRESETS: dict[str, GradientBuilder] = {
    "reference": _reference_gradient,
    "default": _ocean_land_gradient,
    "ocean_land": _ocean_land_gradient,
    "desert": _desert_gradient,
}


def resolve_gradient(name: str, sea_level: float, snow_level: float) -> Sequence[GradientStop]:
    builder = GRADIENT_PRESETS.get(name.lower()) if isinstance(name, str) else None
    if builder is None:
        builder = _ocean_land_gradient
    gradient = tuple(builder(sea_level, snow_level))
    if not gradient:
        return _reference_gradient(sea_level, snow_level)
    return gradient


def apply_gradient(values: np.ndarray, gradient: Sequence[GradientStop]) -> np.ndarray:
    positions = np.array([stop.position for stop in gradient], dtype=np.float32)
    colors = np.array([stop.color for stop in gradient], dtype=np.float32)
    flat = values.reshape(-1)
    r = np.interp(flat, positions, colors[:, 0])
    g = np.interp(flat, positions, colors[:, 1])
    b = np.interp(flat, positions, colors[:, 2])
    rgb = np.stack((r, g, b), axis=1).reshape(values.shape + (3,))
    return rgb


# --------------------------- 法线与光照 ---------------------------


def compute_normals_sobel(heightmap: np.ndarray, strength: float) -> np.ndarray:
    dx = (
        -1 * np.roll(heightmap, (0, -1), (0, 1))
        + 1 * np.roll(heightmap, (0, 1), (0, 1))
    ) * 0.5
    dy = (
        -1 * np.roll(heightmap, (-1, 0), (0, 1))
        + 1 * np.roll(heightmap, (1, 0), (0, 1))
    ) * 0.5
    nx = -dx * strength
    ny = -dy * strength
    nz = np.ones_like(heightmap)
    normals = np.stack([nx, ny, nz], axis=-1)
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals /= np.maximum(norms, 1e-6)
    return normals


def apply_lighting(colors: np.ndarray, normals: np.ndarray, light_dir: Sequence[float], ambient: float) -> np.ndarray:
    light = np.asarray(light_dir, dtype=np.float32)
    if light.shape != (3,):
        raise ValueError("光照方向需要 3 个分量")
    light /= np.maximum(np.linalg.norm(light), 1e-6)
    diffuse = np.clip(np.sum(normals * light, axis=2), 0.0, 1.0)
    shading = ambient + (1.0 - ambient) * diffuse
    return np.clip(colors * shading[..., None], 0.0, 1.0)


# --------------------------- 写实渲染入口 ---------------------------


def render_realistic(
    heightmap: np.ndarray,
    output: Path,
    *,
    gradient: Sequence[GradientStop],
    light_direction: Sequence[float],
    ambient_light: float,
    normal_strength: float,
) -> None:
    normalized = np.clip(heightmap, 0.0, 1.0).astype(np.float32)
    # 轻微提升对比度，使山脉与平原差异更加明显
    toned = np.power(normalized, 1.3)
    normals = compute_normals_sobel(toned, strength=normal_strength)
    colors = apply_gradient(toned, gradient)
    lit = apply_lighting(colors, normals, light_direction, ambient_light)
    # numpy -> Image 默认左上角为原点，需要翻转保持与 classic 一致
    pixels = np.flipud(lit)
    Image.fromarray((pixels * 255).astype(np.uint8)).save(output)
