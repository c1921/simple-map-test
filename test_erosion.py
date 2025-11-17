"""测试侵蚀功能的简单脚本"""

from simple_map.generator import MapGenerator, MapGeneratorConfig
from pathlib import Path

# 创建输出目录
Path("renders").mkdir(exist_ok=True)

# 测试1: 不带侵蚀
print("生成不带侵蚀的地图...")
config1 = MapGeneratorConfig(
    width=512,
    height=512,
    seed=1142,
    enable_erosion=False,
)

gen1 = MapGenerator(config1)
heightmap1 = gen1.generate_heightmap()
colors1 = gen1.classify_terrain(heightmap1)
gen1.render(colors1, heightmap1, Path("renders/test_no_erosion.png"))
gen1.save_heightmap(heightmap1, Path("renders/test_no_erosion_height.png"))
print("OK - 不带侵蚀的地图已保存")

# 测试2: 带侵蚀
print("\n生成带侵蚀的地图 (100次迭代)...")
config2 = MapGeneratorConfig(
    width=512,
    height=512,
    seed=1142,
    enable_erosion=True,
    erosion_iterations=100,
)

gen2 = MapGenerator(config2)
heightmap2 = gen2.generate_heightmap()
colors2 = gen2.classify_terrain(heightmap2)
gen2.render(colors2, heightmap2, Path("renders/test_with_erosion.png"))
gen2.save_heightmap(heightmap2, Path("renders/test_with_erosion_height.png"))
print("OK - 带侵蚀的地图已保存")

print("\n所有测试完成!")
print("- renders/test_no_erosion.png - 无侵蚀地图")
print("- renders/test_with_erosion.png - 侵蚀地图")
