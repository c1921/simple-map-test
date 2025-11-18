# Simple Map Generator

使用 `map_generator.py` 可以快速生成具有大陆轮廓、地形以及渲染效果的简易地图。

## 特性

- 基于 Perlin 噪声的程序化地形生成
- **水力侵蚀模拟**：可选的真实感侵蚀效果,使地形更自然
- 可配置的地形参数(海平面、雪线、噪声参数等)
- **双渲染模式**：经典等高线风格与参考 `reference/main.py` 的写实光照风格
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
- `--edge-falloff-margin` - 从边缘往内多少比例开始降低噪声(0~0.49，默认 0.08)，可保证四周海面
- `--edge-falloff-power` - 边缘衰减速度(>0，默认 1.8)，值越大海岸线越窄

### 渲染参数

- `--render-mode` - `classic`(默认) 使用 Matplotlib 等高线渲染；`realistic` 使用参考脚本的写实渲染
- `--gradient-preset` - 写实渲染的颜色梯度预设，如 `ocean_land`、`reference`、`desert`
- `--light-direction` - 写实渲染光源方向，示例：`--light-direction -0.2 -0.5 0.7`
- `--ambient-light` - 写实渲染环境光强度(0~1)
- `--normal-strength` - 写实渲染法线强度，增大可强化明暗反差
- `--output-scale` - 对渲染结果与高度图进行插值放大，例如 `--output-scale 2.0` 输出 2 倍尺寸
- `--output-interpolation` - 放大时使用的插值方式，可选 `nearest`/`bilinear`/`bicubic`/`quadratic`/`quartic`/`quintic`
- `--output-detail-noise-strength` - 放大后叠加轻微分型噪声的强度(0 表示关闭)，缓解像素块感
- `--output-detail-noise-scale` / `--output-detail-noise-octaves` - 控制细节噪声的尺度与叠加层数
- `--output-detail-noise-persistence` / `--output-detail-noise-lacunarity` - 调整细节噪声每层振幅衰减与频率倍率
- `--pre-erosion-map` / `--pre-erosion-heightmap` - 额外导出侵蚀模拟之前的彩色图与高度图
- `--pre-detail-map` / `--pre-detail-heightmap` - 额外导出添加细节噪声之前（仍含侵蚀效果）的彩色图与高度图

### 输出插值放大

当 `width`/`height` 设置为计算网格分辨率但希望导出更大的 PNG 时，可使用 `output_scale` 与 `output_interpolation`。运行结束后会在保存前对高度图做一次 `scipy.ndimage.zoom` 插值，默认采用 `bicubic`。例如：

```bash
python -m simple_map.cli --config map_config.json --output-scale 2.5 --output-interpolation quintic
```

这样即使仍按 1024×1024 网格进行噪声/侵蚀计算，最终输出会被平滑放大为 2560×2560，并保持高度图与彩色图尺寸一致。若放大后仍能看到像素块，可再叠加分型噪声增强细节：

```bash
python -m simple_map.cli --output-scale 4 --output-interpolation quartic \
    --output-detail-noise-strength 0.035 --output-detail-noise-scale 18
```

细节噪声在输出阶段添加，默认只要 `output-detail-noise-strength > 0` 即可启用。根据放大倍数可适当增大强度或减小 `output-detail-noise-scale`（数字越小细节越密集），一般 0.02~0.05 的强度即可明显缓解块状感。

若需要比较不同阶段的效果，可同时开启阶段输出。例如：

```bash
python -m simple_map.cli --config map_config.json \
    --pre-erosion-map renders/pre_erosion_map.png \
    --pre-erosion-heightmap renders/pre_erosion_height.png \
    --pre-detail-map renders/pre_detail_map.png \
    --pre-detail-heightmap renders/pre_detail_height.png
```

其中“侵蚀前”会导出尚未经过侵蚀模拟、但已经完成插值缩放的地图；“添加细节噪声前”导出已完成侵蚀与缩放但未叠加细节噪声的结果，方便与最终输出做对比。

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

### 渲染模式说明

- **classic**：沿用最初的配色与等高线渲染方式，适合快速查看地貌结构。
- **realistic**：复用了 `reference/main.py` 中的颜色梯度、法线估计与光照计算，输出具有柔和漫反射效果的 PNG。可通过梯度预设与光源参数自定义风格。

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
