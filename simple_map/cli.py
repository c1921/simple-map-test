"""命令行入口。"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict

from .generator import MapGenerator, MapGeneratorConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成彩色大陆地图。")
    parser.add_argument("--config", type=Path, default=None, help="配置文件路径，默认 map_config.json")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="随机种子（默认随机）")
    parser.add_argument("--use-seed", action="store_true", help="启用固定种子")
    parser.add_argument("--out", type=Path, default=None, help="渲染输出路径")
    parser.add_argument("--heightmap", type=Path, default=None, help="可选的高度图输出路径")
    parser.add_argument("--octaves", type=int, default=None)
    parser.add_argument("--persistence", type=float, default=None)
    parser.add_argument("--lacunarity", type=float, default=None)
    parser.add_argument("--base-scale", type=float, default=None)
    parser.add_argument("--sea-level", type=float, default=None)
    parser.add_argument("--snow-level", type=float, default=None)
    return parser.parse_args()


def load_config(path: Path | None, *, require_exists: bool) -> Dict[str, Any]:
    target = path if path is not None else Path("map_config.json")
    if not target.exists():
        if require_exists:
            raise FileNotFoundError(f"指定的配置文件不存在：{target}")
        # 如果是默认配置文件且不存在，尝试从示例文件复制
        if path is None:
            example_path = Path("map_config.json.example")
            if example_path.exists():
                shutil.copy(example_path, target)
                print(f"已从 {example_path} 创建配置文件：{target}")
            else:
                return {}
        else:
            return {}
    with target.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def pick(setting: str, args: argparse.Namespace, config: Dict[str, Any], default: Any) -> Any:
    value = getattr(args, setting)
    if value is not None:
        return value
    if setting in config:
        return config[setting]
    return default


def run() -> None:
    args = parse_args()
    config_data = load_config(args.config, require_exists=args.config is not None)
    config = MapGeneratorConfig(
        width=int(pick("width", args, config_data, 768)),
        height=int(pick("height", args, config_data, 512)),
        seed=_resolve_seed(args, config_data),
        octaves=int(pick("octaves", args, config_data, 5)),
        persistence=float(pick("persistence", args, config_data, 0.5)),
        lacunarity=float(pick("lacunarity", args, config_data, 2.0)),
        base_scale=float(pick("base_scale", args, config_data, 160.0)),
        sea_level=float(pick("sea_level", args, config_data, 0.45)),
        snow_level=float(pick("snow_level", args, config_data, 0.9)),
    )

    generator = MapGenerator(config)
    heightmap = generator.generate_heightmap()
    colors = generator.classify_terrain(heightmap)

    output_path = Path(pick("out", args, config_data, "map.png"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generator.render(colors, heightmap, output_path)

    heightmap_path = pick("heightmap", args, config_data, None)
    if heightmap_path:
        heightmap_path = Path(heightmap_path)
        heightmap_path.parent.mkdir(parents=True, exist_ok=True)
        generator.save_heightmap(heightmap, heightmap_path)

    print(f"地图已保存：{output_path}")
    if heightmap_path:
        print(f"高度图已保存：{heightmap_path}")


def _resolve_seed(args: argparse.Namespace, config: Dict[str, Any]) -> int | None:
    use_seed = bool(pick("use_seed", args, config, False))
    if not use_seed:
        return None
    return pick("seed", args, config, None)
