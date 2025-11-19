"""噪声与地形基础函数。"""

from __future__ import annotations

import numpy as np


def fade(t: np.ndarray) -> np.ndarray:
    """Perlin 噪声的平滑插值曲线。"""

    return t * t * t * (t * (t * 6 - 15) + 10)


def perlin(width: int, height: int, scale: float, seed: int) -> np.ndarray:
    """生成单层二维 Perlin 噪声。"""

    scale = max(scale, 1.0)
    grid_x = int(np.ceil(width / scale)) + 2
    grid_y = int(np.ceil(height / scale)) + 2

    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, (grid_y, grid_x))
    gradients = np.dstack((np.cos(theta), np.sin(theta)))

    yv, xv = np.meshgrid(
        np.arange(height, dtype=np.float32) / scale,
        np.arange(width, dtype=np.float32) / scale,
        indexing="ij",
    )

    xi = np.floor(xv).astype(int)
    yi = np.floor(yv).astype(int)
    xf = xv - xi
    yf = yv - yi

    g00 = gradients[yi, xi]
    g10 = gradients[yi, xi + 1]
    g01 = gradients[yi + 1, xi]
    g11 = gradients[yi + 1, xi + 1]

    d00 = np.stack((xf, yf), axis=-1)
    d10 = np.stack((xf - 1, yf), axis=-1)
    d01 = np.stack((xf, yf - 1), axis=-1)
    d11 = np.stack((xf - 1, yf - 1), axis=-1)

    n00 = np.sum(g00 * d00, axis=-1)
    n10 = np.sum(g10 * d10, axis=-1)
    n01 = np.sum(g01 * d01, axis=-1)
    n11 = np.sum(g11 * d11, axis=-1)

    u = fade(xf)
    v = fade(yf)

    nx0 = n00 * (1 - u) + n10 * u
    nx1 = n01 * (1 - u) + n11 * u
    nxy = nx0 * (1 - v) + nx1 * v

    return nxy


def fractal_noise(
    width: int,
    height: int,
    *,
    octaves: int,
    persistence: float,
    lacunarity: float,
    base_scale: float,
    seed: int,
) -> np.ndarray:
    """多层 Perlin 噪声 (fBm)。"""

    rng = np.random.default_rng(seed)
    amplitude = 1.0
    frequency = 1.0
    noise = np.zeros((height, width), dtype=np.float32)
    max_amplitude = 0.0

    for _ in range(octaves):
        octave_seed = rng.integers(0, 2**32 - 1)
        scale = base_scale / frequency
        noise += amplitude * perlin(width, height, scale, int(octave_seed))
        max_amplitude += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    if max_amplitude == 0:
        return noise
    return noise / max_amplitude


def ridged_multifractal(
    width: int,
    height: int,
    *,
    octaves: int,
    persistence: float,
    lacunarity: float,
    base_scale: float,
    seed: int,
) -> np.ndarray:
    """Ridged Multifractal 噪声。

    与 fBm 不同，Ridged Multifractal 会锐化噪声值，
    创建锐利的山谷特征。
    """
    rng = np.random.default_rng(seed)
    amplitude = 1.0
    frequency = 1.0
    noise = np.zeros((height, width), dtype=np.float32)
    max_amplitude = 0.0

    for _ in range(octaves):
        octave_seed = rng.integers(0, 2**32 - 1)
        scale = base_scale / frequency
        octave_noise = perlin(width, height, scale, int(octave_seed))

        # Ridged Multifractal 的关键：取绝对值创建锐利山谷
        octave_noise = np.abs(octave_noise)

        noise += amplitude * octave_noise
        max_amplitude += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    if max_amplitude == 0:
        return noise
    return noise / max_amplitude


def continent_mask(width: int, height: int, falloff: float = 2.5) -> np.ndarray:
    """构造岛屿式大陆轮廓。"""

    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    distance = np.sqrt(x * x + y * y)
    mask = 1.0 - np.power(distance, falloff)
    return np.clip(mask, 0.0, 1.0)


def edge_falloff_mask(
    width: int,
    height: int,
    *,
    margin: float = 0.08,
    exponent: float = 1.5,
) -> np.ndarray:
    """生成一个靠近边缘快速衰减的遮罩，用于压低海岸线附近的噪声。

    margin 控制从边缘开始衰减的宽度（占总尺寸的比例），exponent 控制衰减的速度。
    """

    if exponent <= 0:
        return np.ones((height, width), dtype=np.float32)

    margin = float(np.clip(margin, 0.0, 0.49))
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    distance_to_edge = np.minimum(
        np.minimum(x, 1.0 - x),
        np.minimum(y, 1.0 - y),
    )
    max_distance = 0.5 - margin
    if max_distance <= 1e-6:
        normalized = np.zeros_like(distance_to_edge)
    else:
        normalized = np.clip((distance_to_edge - margin) / max_distance, 0.0, 1.0)
    normalized = normalized.astype(np.float32)
    return normalized**exponent
