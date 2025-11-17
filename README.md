# Simple Map Generator

使用 `map_generator.py` 可以快速生成具有大陆轮廓、地形以及渲染效果的简易地图。

## 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate  # Windows 使用 .venv\\Scripts\\activate
pip install -r requirements.txt
```

## 生成地图

```bash
python map_generator.py --width 1024 --height 768 --seed 123 --out renders/map.png --heightmap renders/height.png
# 或者
python -m simple_map.cli --config map_config.json
```

`map_config.json` 已预先提供，可直接修改参数（宽高、噪声设置、输出路径等）；命令行参数与配置文件同名时会覆盖配置值。若要固定种子，可在配置中或命令行启用 `use_seed` 并设置 `seed`；默认关闭则每次运行随机生成。

可用参数：

- `--octaves`、`--persistence`、`--lacunarity`、`--base-scale` 控制噪声细节。
- `--sea-level` 和 `--snow-level` 控制海岸线和雪线高度。

脚本会输出彩色渲染图，包含大陆轮廓与地形等高线；如果提供 `--heightmap`，还会额外保存灰度高度图。
