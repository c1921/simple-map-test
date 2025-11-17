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
```

可用参数：

- `--octaves`、`--persistence`、`--lacunarity`、`--base-scale` 控制噪声细节。
- `--sea-level` 和 `--snow-level` 控制海岸线和雪线高度。

脚本会输出彩色渲染图，包含大陆轮廓与地形等高线；如果提供 `--heightmap`，还会额外保存灰度高度图。
