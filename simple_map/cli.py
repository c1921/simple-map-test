"""命令行入口。"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict
import time

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
    parser.add_argument(
        "--edge-falloff-margin",
        type=float,
        default=None,
        help="距边缘多少比例开始强制降低高度(0-0.49)",
    )
    parser.add_argument(
        "--edge-falloff-power",
        type=float,
        default=None,
        help="边缘高度衰减速度(>0)",
    )

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

    # 渲染参数
    parser.add_argument(
        "--render-mode",
        type=str,
        choices=["classic", "realistic"],
        default=None,
        help="选择渲染模式：classic(默认) 或 realistic",
    )
    parser.add_argument("--gradient-preset", type=str, default=None, help="写实渲染使用的颜色梯度预设")
    parser.add_argument(
        "--light-direction",
        type=float,
        nargs=3,
        metavar=("LX", "LY", "LZ"),
        default=None,
        help="写实渲染光照方向",
    )
    parser.add_argument("--ambient-light", type=float, default=None, help="写实渲染环境光强度(0-1)")
    parser.add_argument("--normal-strength", type=float, default=None, help="写实渲染法线强度")
    parser.add_argument("--output-scale", type=float, default=None, help="输出插值放大倍数(默认 1.0 表示不变)")
    parser.add_argument(
        "--output-interpolation",
        type=str,
        default=None,
        help="输出插值方法：nearest/bilinear/bicubic 等",
    )
    parser.add_argument("--output-detail-noise-strength", type=float, default=None, help="输出阶段叠加的细节噪声强度(0 表示关闭)")
    parser.add_argument("--output-detail-noise-scale", type=float, default=None, help="细节噪声基础尺度(像素越小细节越密集)")
    parser.add_argument("--output-detail-noise-octaves", type=int, default=None, help="细节噪声叠加层数")
    parser.add_argument(
        "--output-detail-noise-persistence",
        type=float,
        default=None,
        help="细节噪声持续性(每层振幅衰减系数)",
    )
    parser.add_argument(
        "--output-detail-noise-lacunarity",
        type=float,
        default=None,
        help="细节噪声频率增长系数(>1.0)",
    )
    parser.add_argument("--pre-erosion-map", type=Path, default=None, help="额外保存侵蚀前的彩色地图")
    parser.add_argument("--pre-erosion-heightmap", type=Path, default=None, help="额外保存侵蚀前的高度图")
    parser.add_argument("--pre-detail-map", type=Path, default=None, help="额外保存添加细节噪声前的彩色地图")
    parser.add_argument("--pre-detail-heightmap", type=Path, default=None, help="额外保存添加细节噪声前的高度图")

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
    overall_start = time.time()
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
        edge_falloff_margin=float(pick("edge_falloff_margin", args, config_data, 0.08)),
        edge_falloff_power=float(pick("edge_falloff_power", args, config_data, 1.8)),
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
        # 渲染参数
        render_mode=str(pick("render_mode", args, config_data, "classic")),
        gradient_preset=str(pick("gradient_preset", args, config_data, "ocean_land")),
        light_direction=_to_float_tuple(pick("light_direction", args, config_data, (-0.2, -0.5, 0.7)), 3),
        ambient_light=float(pick("ambient_light", args, config_data, 0.35)),
        normal_strength=float(pick("normal_strength", args, config_data, 6.0)),
        output_scale=float(pick("output_scale", args, config_data, 1.0)),
        output_interpolation=str(pick("output_interpolation", args, config_data, "bicubic")),
        output_detail_noise_strength=float(pick("output_detail_noise_strength", args, config_data, 0.0)),
        output_detail_noise_scale=float(pick("output_detail_noise_scale", args, config_data, 24.0)),
        output_detail_noise_octaves=int(pick("output_detail_noise_octaves", args, config_data, 3)),
        output_detail_noise_persistence=float(pick("output_detail_noise_persistence", args, config_data, 0.55)),
        output_detail_noise_lacunarity=float(pick("output_detail_noise_lacunarity", args, config_data, 2.2)),
        pre_erosion_map=_resolve_path(pick("pre_erosion_map", args, config_data, None)),
        pre_erosion_heightmap=_resolve_path(pick("pre_erosion_heightmap", args, config_data, None)),
        pre_detail_map=_resolve_path(pick("pre_detail_map", args, config_data, None)),
        pre_detail_heightmap=_resolve_path(pick("pre_detail_heightmap", args, config_data, None)),
    )

    generator = MapGenerator(config)
    heightmap = generator.generate_heightmap()
    pre_detail_heightmap = generator.prepare_heightmap_for_output(heightmap, apply_detail_noise=False)
    output_heightmap = generator.apply_output_detail_noise_to_scaled(pre_detail_heightmap)

    output_path = Path(pick("out", args, config_data, "map.png"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generator.render_map(output_heightmap, output_path, apply_output_scale=False, log_label="最终地图渲染")

    heightmap_path = pick("heightmap", args, config_data, None)
    if heightmap_path:
        heightmap_path = Path(heightmap_path)
        heightmap_path.parent.mkdir(parents=True, exist_ok=True)
        generator.save_heightmap(output_heightmap, heightmap_path, apply_output_scale=False, log_label="最终高度图保存")

    _maybe_save_stage_outputs(generator, pre_detail_heightmap, stage="pre_detail")

    pre_erosion = generator.pre_erosion_heightmap
    if pre_erosion is not None:
        pre_erosion_prepared = generator.prepare_heightmap_for_output(pre_erosion, apply_detail_noise=False)
        _maybe_save_stage_outputs(generator, pre_erosion_prepared, stage="pre_erosion")

    print(f"地图已保存：{output_path}")
    if heightmap_path:
        print(f"高度图已保存：{heightmap_path}")
    total_duration = time.time() - overall_start
    print(f"=== 全流程总耗时: {total_duration:.2f}s ===")


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


def _stage_paths(config: MapGeneratorConfig, stage: str) -> tuple[Path | None, Path | None]:
    if stage == "pre_detail":
        return config.pre_detail_map, config.pre_detail_heightmap
    if stage == "pre_erosion":
        return config.pre_erosion_map, config.pre_erosion_heightmap
    return None, None


def _maybe_save_stage_outputs(generator: MapGenerator, heightmap: np.ndarray, *, stage: str) -> None:
    map_path, height_path = _stage_paths(generator.config, stage)
    if map_path:
        map_path.parent.mkdir(parents=True, exist_ok=True)
        generator.render_map(
            heightmap,
            map_path,
            apply_output_scale=False,
            log_label=f"{stage} 阶段地图渲染",
        )
        print(f"{stage} 阶段地图已保存：{map_path}")
    if height_path:
        height_path.parent.mkdir(parents=True, exist_ok=True)
        generator.save_heightmap(
            heightmap,
            height_path,
            apply_output_scale=False,
            log_label=f"{stage} 阶段高度图保存",
        )
        print(f"{stage} 阶段高度图已保存：{height_path}")


def _to_float_tuple(value: Any, size: int) -> tuple[float, ...]:
    if isinstance(value, (list, tuple)):
        if len(value) != size:
            raise ValueError(f"向量长度应为 {size}，收到 {len(value)}")
        return tuple(float(v) for v in value)
    if value is None:
        raise ValueError("缺少必需的向量参数")
    # 单值也可以——复制为所有分量
    return tuple(float(value) for _ in range(size))
