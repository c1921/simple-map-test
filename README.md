# Simple Map Generator

使用 `map_generator.py` 可以快速生成具有大陆轮廓、地形以及渲染效果的简易地图。

## 特性

- 基于 Perlin 噪声的程序化地形生成
- **水力侵蚀模拟**：可选的真实感侵蚀效果,使地形更自然
- 可配置的地形参数(海平面、雪线、噪声参数等)
- 支持配置文件和命令行参数

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

## 可用参数

### 基础参数

- `--width`, `--height` - 地图尺寸
- `--seed` - 随机种子
- `--use-seed` - 启用固定种子
- `--out` - 渲染图输出路径
- `--heightmap` - 高度图输出路径

### 噪声参数

- `--octaves` - 噪声层数(默认 5)
- `--persistence` - 持续性(默认 0.5)
- `--lacunarity` - 间隙性(默认 2.0)
- `--base-scale` - 基础缩放(默认 160.0)

### 地形参数

- `--sea-level` - 海平面高度(默认 0.45)
- `--snow-level` - 雪线高度(默认 0.9)

### 侵蚀模拟参数

启用侵蚀模拟可以让地形更加真实,产生河谷、冲积平原等自然地貌。

- `--enable-erosion` - 启用侵蚀模拟(默认关闭)
- `--erosion-iterations` - 侵蚀迭代次数(默认 100,值越大效果越明显)
- `--erosion-rain-rate` - 降雨率(默认 0.0008)
- `--erosion-evaporation-rate` - 蒸发率(默认 0.0005)
- `--erosion-sediment-capacity` - 沉积物容量系数(默认 50.0,控制侵蚀强度)
- `--erosion-dissolving-rate` - 溶蚀率(默认 0.25)
- `--erosion-deposition-rate` - 沉积率(默认 0.001)
- `--erosion-repose-slope` - 休止角坡度(默认 0.03,控制坡度平滑)
- `--erosion-gravity` - 重力常数(默认 30.0)
- `--erosion-cell-width` - 网格单元宽度(默认 1.0)

### 侵蚀模拟示例

```bash
# 启用侵蚀模拟
python -m simple_map.cli --enable-erosion --erosion-iterations 150

# 强侵蚀效果
python -m simple_map.cli --enable-erosion --erosion-iterations 200 --erosion-sediment-capacity 80.0

# 使用配置文件
# 编辑 map_config.json,设置 "enable_erosion": true
python -m simple_map.cli --config map_config.json
```

脚本会输出彩色渲染图，包含大陆轮廓与地形等高线；如果提供 `--heightmap`，还会额外保存灰度高度图。

## 侵蚀模拟原理

本项目集成了基于物理的水力侵蚀模拟算法,模拟过程包括:

1. **降雨** - 随机降水分布
2. **水流** - 沿梯度流动
3. **侵蚀/沉积** - 根据水流速度和沉积物容量决定侵蚀或沉积
4. **坡度坍塌** - 平滑过陡的坡面(热侵蚀)
5. **蒸发** - 水量逐渐减少

该算法参考自:
- http://ranmantaru.com/blog/2011/10/08/water-erosion-on-heightmap-terrain/
- https://hal.inria.fr/inria-00402079/document
