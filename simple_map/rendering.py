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
    land_span = max(snow - sea, 1e-4)

    def land_pos(relative: float) -> float:
        return min(sea + relative * land_span, 0.999)

    return (
        GradientStop(0.00, (8 / 255, 20 / 255, 65 / 255)),
        GradientStop(max(sea * 0.35, 0.02), (17 / 255, 46 / 255, 110 / 255)),
        GradientStop(max(sea * 0.8, 0.04), (34 / 255, 78 / 255, 138 / 255)),
        GradientStop(sea, (52 / 255, 112 / 255, 64 / 255)),
        GradientStop(land_pos(0.08), (66 / 255, 131 / 255, 62 / 255)),
        GradientStop(land_pos(0.2), (122 / 255, 154 / 255, 60 / 255)),
        GradientStop(land_pos(0.4), (213 / 255, 191 / 255, 101 / 255)),
        GradientStop(land_pos(0.55), (210 / 255, 143 / 255, 65 / 255)),
        GradientStop(land_pos(0.85), (147 / 255, 72 / 255, 33 / 255)),
        GradientStop(land_pos(0.95), (128 / 255, 128 / 255, 128 / 255)),
        GradientStop(land_pos(0.98), (210 / 255, 210 / 255, 210 / 255)),
        GradientStop(snow, (0.94, 0.94, 0.94)),
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


def _sample_gradient_color(gradient: Sequence[GradientStop], position: float) -> Color:
    """在梯度上采样指定位置的颜色"""
    if not gradient:
        raise ValueError("梯度为空，无法采样颜色")
    positions = np.array([stop.position for stop in gradient], dtype=np.float32)
    colors = np.array([stop.color for stop in gradient], dtype=np.float32)
    pos = float(np.clip(position, positions[0], positions[-1]))
    r = np.interp(pos, positions, colors[:, 0])
    g = np.interp(pos, positions, colors[:, 1])
    b = np.interp(pos, positions, colors[:, 2])
    return float(r), float(g), float(b)


def _build_segment(
    gradient: Sequence[GradientStop],
    start: float,
    end: float,
    start_color: Color,
    end_color: Color,
) -> Sequence[GradientStop]:
    if end <= start:
        return ()
    stops: list[GradientStop] = [GradientStop(start, start_color)]
    for stop in gradient:
        if start < stop.position < end:
            stops.append(stop)
    stops.append(GradientStop(end, end_color))
    return tuple(stops)


def _split_gradient_by_sea_level(
    gradient: Sequence[GradientStop],
    sea_level: float,
    *,
    epsilon: float = 1e-4,
) -> tuple[Sequence[GradientStop], Sequence[GradientStop]]:
    """根据海平面将梯度拆分为海洋与陆地两段，避免颜色在海岸线处混合"""
    if not gradient:
        return (), ()
    lower_bound = gradient[0].position
    upper_bound = gradient[-1].position
    sea = float(np.clip(sea_level, lower_bound, upper_bound))
    if sea <= lower_bound:
        return (), gradient
    if sea >= upper_bound:
        return gradient, ()

    ocean_end_pos = max(sea - epsilon, lower_bound)
    land_start_pos = min(sea + epsilon, upper_bound)
    ocean_segment = _build_segment(
        gradient,
        lower_bound,
        sea,
        _sample_gradient_color(gradient, lower_bound),
        _sample_gradient_color(gradient, ocean_end_pos),
    )
    land_segment = _build_segment(
        gradient,
        sea,
        upper_bound,
        _sample_gradient_color(gradient, land_start_pos),
        _sample_gradient_color(gradient, upper_bound),
    )
    return ocean_segment, land_segment


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
    sea_level: float,
    light_direction: Sequence[float],
    ambient_light: float,
    normal_strength: float,
    land_mask: np.ndarray | None = None,
) -> None:
    normalized = np.clip(heightmap, 0.0, 1.0).astype(np.float32)
    toned = np.power(normalized, 1.3)
    normals = compute_normals_sobel(toned, strength=normal_strength)

    ocean_gradient, land_gradient = _split_gradient_by_sea_level(gradient, sea_level)

    # 如果提供了陆地mask,则使用它来确定海洋区域
    # 海洋 = 不在陆地mask内的区域
    if land_mask is not None:
        ocean_mask = ~land_mask.astype(bool)
    else:
        # 回退到基于高度的判断
        ocean_mask = normalized <= sea_level

    colors = np.zeros(toned.shape + (3,), dtype=np.float32)
    if ocean_gradient:
        ocean_values = np.clip(toned, ocean_gradient[0].position, sea_level)
        ocean_colors = apply_gradient(ocean_values, ocean_gradient)
        colors[ocean_mask] = ocean_colors[ocean_mask]
    if land_gradient:
        land_values = np.clip(toned, sea_level, land_gradient[-1].position)
        land_colors = apply_gradient(land_values, land_gradient)
        colors[~ocean_mask] = land_colors[~ocean_mask]
    if not ocean_gradient and land_gradient:
        colors = land_colors
    elif not land_gradient and ocean_gradient:
        colors = ocean_colors

    coastline_mask = _coastline_mask(ocean_mask)
    if np.any(coastline_mask):
        coast_color = np.array(_sample_gradient_color(gradient, min(sea_level + 0.02, 1.0)), dtype=np.float32)
        blend_ratio = 0.65
        colors[coastline_mask] = np.clip(
            blend_ratio * coast_color + (1.0 - blend_ratio) * colors[coastline_mask],
            0.0,
            1.0,
        )

    lit = apply_lighting(colors, normals, light_direction, ambient_light)
    # numpy -> Image 默认左上角为原点，需要翻转保持与 classic 一致
    pixels = np.flipud(lit)
    Image.fromarray((pixels * 255).astype(np.uint8)).save(output)


def _coastline_mask(ocean_mask: np.ndarray) -> np.ndarray:
    """检测与海洋相邻的陆地区域，用于额外强调海岸线"""
    if ocean_mask.size == 0:
        return ocean_mask
    neighbors = (
        np.roll(ocean_mask, 1, axis=0)
        | np.roll(ocean_mask, -1, axis=0)
        | np.roll(ocean_mask, 1, axis=1)
        | np.roll(ocean_mask, -1, axis=1)
    )
    return (~ocean_mask) & neighbors


# --------------------------- 宜居度计算 ---------------------------


def compute_habitability_score(heightmap: np.ndarray, land_mask: np.ndarray) -> np.ndarray:
    """计算陆地区域的宜居度分数。

    宜居度基于地形平坦程度，平坦的平原、谷底、高地等区域宜居度高。

    Args:
        heightmap: 高度图数据 (0-1)
        land_mask: 陆地mask (True=陆地, False=海洋)

    Returns:
        宜居度分数 (0-1)，0=不宜居，1=最宜居
    """
    # 计算地形梯度（坡度）
    # 使用Sobel算子计算x和y方向的梯度
    from scipy.ndimage import sobel

    dx = sobel(heightmap, axis=1, mode='reflect')
    dy = sobel(heightmap, axis=0, mode='reflect')

    # 计算梯度幅值（坡度）
    gradient_magnitude = np.sqrt(dx**2 + dy**2)

    # 归一化梯度
    if gradient_magnitude.max() > 0:
        gradient_magnitude = gradient_magnitude / gradient_magnitude.max()

    # 计算宜居度：坡度越小，宜居度越高
    # 使用指数函数使变化更平滑
    habitability = np.exp(-gradient_magnitude * 8.0)

    # 仅保留陆地区域的宜居度
    habitability = habitability * land_mask.astype(np.float32)

    return habitability.astype(np.float32)


def render_habitability_map(
    heightmap: np.ndarray,
    output: Path,
    *,
    land_mask: np.ndarray,
) -> None:
    """渲染宜居度地图。

    在陆地区域内，根据地形平坦程度使用红绿渐变着色：
    - 绿色 = 宜居（平坦的平原、谷底、平坦高地）
    - 红色 = 不宜居（陡峭的山地、峡谷）
    - 黑色 = 海洋区域

    Args:
        heightmap: 高度图数据
        output: 输出路径
        land_mask: 陆地mask (True=陆地, False=海洋)
    """
    # 计算宜居度分数
    habitability = compute_habitability_score(heightmap, land_mask)

    # 创建RGB图像
    # 海洋区域：黑色
    # 陆地区域：红绿渐变 (低宜居度=红色, 高宜居度=绿色)
    colors = np.zeros(heightmap.shape + (3,), dtype=np.float32)

    # 陆地区域：使用红绿渐变
    # habitability: 0 (不宜居) -> 红色 (1, 0, 0)
    # habitability: 1 (宜居) -> 绿色 (0, 1, 0)
    land_pixels = land_mask.astype(bool)
    colors[land_pixels, 0] = 1.0 - habitability[land_pixels]  # R: 越不宜居越红
    colors[land_pixels, 1] = habitability[land_pixels]        # G: 越宜居越绿
    colors[land_pixels, 2] = 0.0                              # B: 保持为0

    # numpy -> Image 默认左上角为原点，需要翻转保持与 classic 一致
    pixels = np.flipud(colors)
    Image.fromarray((pixels * 255).astype(np.uint8)).save(output)
