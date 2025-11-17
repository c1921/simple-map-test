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

    # 侵蚀模拟参数
    parser.add_argument("--enable-erosion", action="store_true", help="启用侵蚀模拟")
    parser.add_argument("--erosion-iterations", type=int, default=None, help="侵蚀迭代次数")
    parser.add_argument("--erosion-rain-rate", type=float, default=None, help="降雨率")
    parser.add_argument("--erosion-evaporation-rate", type=float, default=None, help="蒸发率")
    parser.add_argument("--erosion-min-height-delta", type=float, default=None, help="最小高度差")
    parser.add_argument("--erosion-repose-slope", type=float, default=None, help="休止角坡度")
    parser.add_argument("--erosion-gravity", type=float, default=None, help="重力常数")
    parser.add_argument("--erosion-sediment-capacity", type=float, default=None, help="沉积物容量系数")
    parser.add_argument("--erosion-dissolving-rate", type=float, default=None, help="溶蚀率")
    parser.add_argument("--erosion-deposition-rate", type=float, default=None, help="沉积率")
    parser.add_argument("--erosion-cell-width", type=float, default=None, help="网格单元宽度")

    # 侵蚀快照参数
    parser.add_argument("--erosion-snapshot-interval", type=int, default=None, help="快照间隔步数（0表示不保存快照）")
    parser.add_argument("--erosion-snapshots-dir", type=Path, default=None, help="快照保存目录")

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
    # 对于 action="store_true" 的参数，只有显式提供时才使用命令行值
    # 否则应该从配置文件读取
    if setting == "enable_erosion":
        # 检查是否通过命令行显式设置了 --enable-erosion
        if value is True:  # 只有真正设置了才使用
            return value
        # 否则从配置文件读取
        if setting in config:
            return config[setting]
        return default

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
        # 侵蚀模拟参数
        enable_erosion=bool(pick("enable_erosion", args, config_data, False)),
        erosion_iterations=int(pick("erosion_iterations", args, config_data, 100)),
        erosion_rain_rate=float(pick("erosion_rain_rate", args, config_data, 0.0008)),
        erosion_evaporation_rate=float(pick("erosion_evaporation_rate", args, config_data, 0.0005)),
        erosion_min_height_delta=float(pick("erosion_min_height_delta", args, config_data, 0.05)),
        erosion_repose_slope=float(pick("erosion_repose_slope", args, config_data, 0.03)),
        erosion_gravity=float(pick("erosion_gravity", args, config_data, 30.0)),
        erosion_sediment_capacity=float(pick("erosion_sediment_capacity", args, config_data, 50.0)),
        erosion_dissolving_rate=float(pick("erosion_dissolving_rate", args, config_data, 0.25)),
        erosion_deposition_rate=float(pick("erosion_deposition_rate", args, config_data, 0.001)),
        erosion_cell_width=float(pick("erosion_cell_width", args, config_data, 1.0)),
        # 侵蚀快照参数
        erosion_snapshot_interval=int(pick("erosion_snapshot_interval", args, config_data, 0)),
        erosion_snapshots_dir=_resolve_path(pick("erosion_snapshots_dir", args, config_data, None)),
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


def _resolve_path(value: Any) -> Path | None:
    """将字符串路径转换为Path对象，None保持为None"""
    if value is None:
        return None
    return Path(value)
