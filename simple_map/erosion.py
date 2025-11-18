"""
水力侵蚀模拟器
基于参考项目 reference/simulation.py 的半物理侵蚀算法
"""

from dataclasses import dataclass
from typing import Optional
import time
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

    # 海拔参数
    sea_level: Optional[float] = None  # 低于此高度视为海洋(终止侵蚀)


class ErosionSimulator:
    """水力侵蚀模拟器"""

    def __init__(self, config: ErosionConfig):
        """
        初始化侵蚀模拟器

        参数:
            config: 侵蚀配置参数
        """
        self.config = config
        # 预计算的网格坐标和高斯核将在第一次模拟时初始化
        self._base_coords = None
        self._blur_kernel = None
        self._cached_shape = None

    def _initialize_precomputed_data(self, shape):
        """
        预计算meshgrid和高斯核以提高性能

        参数:
            shape: 地形数组的形状
        """
        # 只在形状改变时重新计算
        if self._cached_shape == shape:
            return

        # 预计算sample()使用的基础坐标网格
        self._base_coords = np.array(np.meshgrid(*map(range, shape)))

        # 预计算gaussian_blur()使用的高斯核(sigma=1.5)
        freqs = tuple(np.fft.fftfreq(n, d=1.0 / n) for n in shape)
        freq_radial = np.hypot(*np.meshgrid(*freqs))
        sigma = 1.5
        sigma2 = sigma**2
        g = lambda x: ((2 * np.pi * sigma2) ** -0.5) * np.exp(-0.5 * (x / sigma)**2)
        self._blur_kernel = g(freq_radial)
        self._blur_kernel /= self._blur_kernel.sum()

        self._cached_shape = shape

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
        # 记录总耗时
        start_time = time.time()

        # 复制地形避免修改原始数据
        terrain = terrain.copy().astype(np.float64)
        shape = terrain.shape

        # 预计算meshgrid和高斯核以提高性能
        self._initialize_precomputed_data(shape)

        # 计算网格单元面积
        cell_area = self.config.cell_width ** 2

        # 初始化模拟变量
        sediment = np.zeros_like(terrain)  # 水中悬浮的沉积物
        water = np.zeros_like(terrain)  # 水量
        velocity = np.zeros_like(terrain)  # 水流速度

        ocean_mask = None
        if self.config.sea_level is not None:
            ocean_mask = terrain <= self.config.sea_level
            terrain = self._clamp_ocean_surface(terrain, ocean_mask)

        # 主模拟循环
        iteration_start_time = time.time()
        for i in range(self.config.iterations):
            step_start_time = time.time()

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
            neighbor_height = utils.sample(terrain, -gradient, self._base_coords)
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
            ocean_mask = self._drain_to_ocean(terrain, water, sediment, ocean_mask)

            # 9. 坡度坍塌(平滑陡坡)
            terrain = self._apply_slippage(terrain)
            terrain = self._clamp_ocean_surface(terrain, ocean_mask)

            # 10. 更新速度
            velocity = self.config.gravity * height_delta / self.config.cell_width

            # 11. 蒸发
            water *= (1 - self.config.evaporation_rate)

            # 12. 调用回调函数（如果提供）
            if callback is not None:
                callback(i + 1, terrain.copy())

            # 打印每一步的进度和计时
            if verbose:
                step_time = time.time() - step_start_time
                elapsed = time.time() - iteration_start_time
                avg_time = elapsed / (i + 1)
                remaining = avg_time * (self.config.iterations - i - 1)
                print(f'步骤 {i + 1}/{self.config.iterations} | '
                      f'本步: {step_time:.4f}s | '
                      f'已用时: {elapsed:.2f}s | '
                      f'预计剩余: {remaining:.2f}s')

        terrain = self._clamp_ocean_surface(terrain, ocean_mask)

        # 计算总耗时
        total_time = time.time() - start_time

        if verbose:
            print(f'侵蚀模拟完成: {self.config.iterations} 次迭代 | 总耗时: {total_time:.2f}s | '
                  f'平均每次迭代: {total_time / self.config.iterations:.4f}s')

        return terrain.astype(np.float32)

    def _drain_to_ocean(self, terrain, water, sediment, ocean_mask):
        sea_level = self.config.sea_level
        if sea_level is None:
            return ocean_mask

        current_ocean = terrain <= sea_level
        if np.any(current_ocean):
            water[current_ocean] = 0.0
            sediment[current_ocean] = 0.0

        if ocean_mask is None:
            ocean_mask = current_ocean
        else:
            ocean_mask |= current_ocean
        return ocean_mask

    def _clamp_ocean_surface(self, terrain, ocean_mask):
        sea_level = self.config.sea_level
        if sea_level is None or ocean_mask is None:
            return terrain
        if np.any(ocean_mask):
            terrain[ocean_mask] = np.minimum(terrain[ocean_mask], sea_level)
        return terrain

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

        # 对陡坡区域应用平滑(使用预计算的高斯核)
        smoothed = utils.gaussian_blur(terrain, sigma=1.5, kernel=self._blur_kernel)
        result = np.select(
            [np.abs(delta) > self.config.repose_slope],
            [smoothed],
            terrain
        )

        return result
