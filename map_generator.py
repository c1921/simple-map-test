"""Simple procedural map generator producing continents, terrain, and renderings.

This module exposes a MapGenerator that can fabricate a believable landmass
heightmap using multi-octave Perlin-like noise blended with a radial mask. The
results can be rendered to colorful terrain maps along with coastline and
topography contours.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def _fade(t: np.ndarray) -> np.ndarray:
    """Smooth interpolation curve used by Perlin noise."""

    return t * t * t * (t * (t * 6 - 15) + 10)


def _perlin(width: int, height: int, scale: float, seed: int) -> np.ndarray:
    """Return a single octave of 2D Perlin noise."""

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

    u = _fade(xf)
    v = _fade(yf)

    nx0 = n00 * (1 - u) + n10 * u
    nx1 = n01 * (1 - u) + n11 * u
    nxy = nx0 * (1 - v) + nx1 * v

    return nxy


def _fractal_noise(
    width: int,
    height: int,
    *,
    octaves: int,
    persistence: float,
    lacunarity: float,
    base_scale: float,
    seed: int,
) -> np.ndarray:
    """Blend multiple Perlin octaves into fractal Brownian motion noise."""

    rng = np.random.default_rng(seed)
    amplitude = 1.0
    frequency = 1.0
    noise = np.zeros((height, width), dtype=np.float32)
    max_amplitude = 0.0

    for _ in range(octaves):
        octave_seed = rng.integers(0, 2**32 - 1)
        scale = base_scale / frequency
        noise += amplitude * _perlin(width, height, scale, int(octave_seed))
        max_amplitude += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    if max_amplitude == 0:
        return noise

    return noise / max_amplitude


def _continent_mask(width: int, height: int, falloff: float = 2.5) -> np.ndarray:
    """Encourage island-like landmasses by fading elevation near the borders."""

    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    distance = np.sqrt(x * x + y * y)
    mask = 1.0 - np.power(distance, falloff)
    return np.clip(mask, 0.0, 1.0)


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
    """High-level API for creating and rendering simple maps."""

    def __init__(self, config: MapGeneratorConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.seed)

    def generate_heightmap(self) -> np.ndarray:
        cfg = self.config
        fbm = _fractal_noise(
            cfg.width,
            cfg.height,
            octaves=cfg.octaves,
            persistence=cfg.persistence,
            lacunarity=cfg.lacunarity,
            base_scale=cfg.base_scale,
            seed=int(self._rng.integers(0, 2**32 - 1)),
        )
        mask = _continent_mask(cfg.width, cfg.height)
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
        apply_mask(
            (heightmap >= sea * 0.75) & (heightmap < sea),
            (28 / 255, 74 / 255, 142 / 255),
        )
        apply_mask(
            (heightmap >= sea) & (heightmap < sea + 0.05),
            (222 / 255, 202 / 255, 166 / 255),
        )
        apply_mask(
            (heightmap >= sea + 0.05) & (heightmap < 0.65),
            (87 / 255, 140 / 255, 62 / 255),
        )
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
        ax.contour(
            heightmap,
            levels=[self.config.sea_level],
            colors="#102542",
            linewidths=0.75,
        )
        land_levels = np.linspace(self.config.sea_level + 0.05, 0.95, 4)
        ax.contour(
            heightmap,
            levels=land_levels,
            colors="black",
            linewidths=0.3,
            alpha=0.35,
        )
        ax.axis("off")
        fig.tight_layout(pad=0)
        fig.savefig(output, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    def save_heightmap(self, heightmap: np.ndarray, path: Path) -> None:
        plt.imsave(path, heightmap, cmap="gray", origin="lower")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a colorful procedural map.")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: random)")
    parser.add_argument("--out", type=Path, default=Path("map.png"), help="Rendered map output path")
    parser.add_argument(
        "--heightmap",
        type=Path,
        default=None,
        help="Optional grayscale heightmap output path",
    )
    parser.add_argument("--octaves", type=int, default=5)
    parser.add_argument("--persistence", type=float, default=0.5)
    parser.add_argument("--lacunarity", type=float, default=2.0)
    parser.add_argument("--base-scale", type=float, default=160.0)
    parser.add_argument("--sea-level", type=float, default=0.45)
    parser.add_argument("--snow-level", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MapGeneratorConfig(
        width=args.width,
        height=args.height,
        seed=args.seed,
        octaves=args.octaves,
        persistence=args.persistence,
        lacunarity=args.lacunarity,
        base_scale=args.base_scale,
        sea_level=args.sea_level,
        snow_level=args.snow_level,
    )

    generator = MapGenerator(config)
    heightmap = generator.generate_heightmap()
    colors = generator.classify_terrain(heightmap)

    output_path = args.out
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generator.render(colors, heightmap, output_path)

    if args.heightmap:
        args.heightmap.parent.mkdir(parents=True, exist_ok=True)
        generator.save_heightmap(heightmap, args.heightmap)

    print(f"Map saved to {output_path}")
    if args.heightmap:
        print(f"Heightmap saved to {args.heightmap}")


if __name__ == "__main__":
    main()
