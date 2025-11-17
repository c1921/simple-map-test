"""
水力侵蚀模拟器
基于参考项目 reference/simulation.py 的半物理侵蚀算法
"""

from dataclasses import dataclass
import numpy as np
from . import erosion_utils as utils


@dataclass
class ErosionConfig:
    """侵蚀模拟参数配置"""

    # 迭代参数
    iterations: int = 100  # 模拟迭代次数

    # 水文参数
    rain_rate: float = 0.0008  # 降雨率(会乘以cell_area)
    evaporation_rate: float = 0.0005  # 蒸发率

    # 坡度参数
    min_height_delta: float = 0.05  # 最小高度差
    repose_slope: float = 0.03  # 休止角坡度(用于坍塌)
    gravity: float = 30.0  # 重力常数

    # 沉积物参数
    sediment_capacity_constant: float = 50.0  # 沉积物容量系数
    dissolving_rate: float = 0.25  # 溶蚀率
    deposition_rate: float = 0.001  # 沉积率

    # 网格参数
    cell_width: float = 1.0  # 网格单元宽度(用于物理计算)


class ErosionSimulator:
    """水力侵蚀模拟器"""

    def __init__(self, config: ErosionConfig):
        """
        初始化侵蚀模拟器

        参数:
            config: 侵蚀配置参数
        """
        self.config = config

    def simulate(self, terrain: np.ndarray, verbose: bool = True, callback=None) -> np.ndarray:
        """
        执行侵蚀模拟

        参数:
            terrain: 输入地形高度图 (归一化到[0,1])
            verbose: 是否打印进度信息
            callback: 可选的回调函数，签名为 callback(step: int, terrain: np.ndarray)
                     在每次迭代后调用，用于保存中间结果等操作

        返回:
            侵蚀后的地形高度图
        """
        # 复制地形避免修改原始数据
        terrain = terrain.copy().astype(np.float64)
        shape = terrain.shape

        # 计算网格单元面积
        cell_area = self.config.cell_width ** 2

        # 初始化模拟变量
        sediment = np.zeros_like(terrain)  # 水中悬浮的沉积物
        water = np.zeros_like(terrain)  # 水量
        velocity = np.zeros_like(terrain)  # 水流速度

        # 主模拟循环
        for i in range(self.config.iterations):
            if verbose and (i + 1) % 10 == 0:
                print(f'侵蚀模拟进度: {i + 1} / {self.config.iterations}')

            # 1. 降雨
            water += np.random.rand(*shape) * self.config.rain_rate * cell_area

            # 2. 计算地形梯度(归一化为单位向量)
            gradient = utils.simple_gradient(terrain)
            # 处理梯度为零的情况(随机方向)
            gradient = np.select(
                [np.abs(gradient) < 1e-10],
                [np.exp(2j * np.pi * np.random.rand(*shape))],
                gradient
            )
            gradient /= np.abs(gradient)

            # 3. 计算高度差
            neighbor_height = utils.sample(terrain, -gradient)
            height_delta = terrain - neighbor_height

            # 4. 计算沉积物容量
            sediment_capacity = (
                (np.maximum(height_delta, self.config.min_height_delta) / self.config.cell_width) *
                velocity * water * self.config.sediment_capacity_constant
            )

            # 5. 计算沉积/侵蚀量
            deposited_sediment = np.select(
                [
                    height_delta < 0,  # 上坡 -> 沉积
                    sediment > sediment_capacity,  # 沉积物过量 -> 沉积
                ],
                [
                    np.minimum(height_delta, sediment),  # 沉积部分沉积物
                    self.config.deposition_rate * (sediment - sediment_capacity),
                ],
                # 沉积物不足容量 -> 侵蚀
                self.config.dissolving_rate * (sediment - sediment_capacity)
            )

            # 6. 限制侵蚀量不超过当前地形高度
            deposited_sediment = np.maximum(-height_delta, deposited_sediment)

            # 7. 更新地形和沉积物
            sediment -= deposited_sediment
            terrain += deposited_sediment

            # 8. 沉积物和水沿梯度移动
            sediment = utils.displace(sediment, gradient)
            water = utils.displace(water, gradient)

            # 9. 坡度坍塌(平滑陡坡)
            terrain = self._apply_slippage(terrain)

            # 10. 更新速度
            velocity = self.config.gravity * height_delta / self.config.cell_width

            # 11. 蒸发
            water *= (1 - self.config.evaporation_rate)

            # 12. 调用回调函数（如果提供）
            if callback is not None:
                callback(i + 1, terrain.copy())

        if verbose:
            print(f'侵蚀模拟完成: {self.config.iterations} 次迭代')

        return terrain.astype(np.float32)

    def _apply_slippage(self, terrain: np.ndarray) -> np.ndarray:
        """
        应用坡度坍塌(热侵蚀)

        对于坡度超过休止角的区域,使用高斯模糊平滑

        参数:
            terrain: 地形高度图

        返回:
            应用坍塌后的地形
        """
        # 计算坡度
        delta = utils.simple_gradient(terrain) / self.config.cell_width

        # 对陡坡区域应用平滑
        smoothed = utils.gaussian_blur(terrain, sigma=1.5)
        result = np.select(
            [np.abs(delta) > self.config.repose_slope],
            [smoothed],
            terrain
        )

        return result
