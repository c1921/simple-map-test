"""
侵蚀模拟工具函数（Numba 优化版）
完全兼容原版 API，数值等价，但性能大幅提升
"""

import numpy as np
from numba import njit, prange


# ---------------------------------------------------------
#  simple_gradient (保留原实现)
# ---------------------------------------------------------

def simple_gradient(a):
    """
    计算地形梯度(使用复数编码向量)

    返回:
        复数数组,实部=dx,虚部=dy
        (注意：你的算法里 real=dy, imag=dx, 我们严格保持)
    """
    dx = 0.5 * (np.roll(a, 1, axis=0) - np.roll(a, -1, axis=0))
    dy = 0.5 * (np.roll(a, 1, axis=1) - np.roll(a, -1, axis=1))
    return 1j * dx + dy


# ---------------------------------------------------------
#  Numba 核心：sample_numba（严格复刻你的 sample 行为）
# ---------------------------------------------------------

@njit(parallel=True, fastmath=True)
def sample_numba(a, off_real, off_imag):
    """
    Numba 加速版 sample()

    - off_real = offset.real = dy
    - off_imag = offset.imag = dx
    - 原版 sample() 使用 base_coords = meshgrid(range(H), range(W))
      并根据 coords = base_coords - delta 做双线性插值。
      在这里完全复刻其行为（避免 meshgrid 的巨大开销）。
    """
    H, W = a.shape
    out = np.empty_like(a)

    for i in prange(H):
        for j in range(W):

            # 等价于：
            # coords_x = j - off_real[i, j]
            # coords_y = i - off_imag[i, j]
            xx = j - off_real[i, j]
            yy = i - off_imag[i, j]

            # 下界
            x0 = int(np.floor(xx))
            y0 = int(np.floor(yy))
            x1 = x0 + 1
            y1 = y0 + 1

            # 周期性边界
            x0 %= W
            x1 %= W
            y0 %= H
            y1 %= H

            # 插值系数
            tx = xx - np.floor(xx)
            ty = yy - np.floor(yy)

            # 双线性插值
            a00 = a[y0, x0]
            a01 = a[y0, x1]
            a10 = a[y1, x0]
            a11 = a[y1, x1]

            out[i, j] = (
                (1 - ty) * ((1 - tx) * a00 + tx * a01) +
                 ty      * ((1 - tx) * a10 + tx * a11)
            )

    return out


# ---------------------------------------------------------
#  外壳：sample() 兼容 erosion.py 的 API
# ---------------------------------------------------------

def sample(a, offset, base_coords=None):
    """
    双线性插值采样 API（外壳）
    erosion.py 调用格式：sample(a, offset, base_coords)

    base_coords 在 Numba 版本中不再使用，但必须保留参数。
    """
    off_real = offset.real  # dy
    off_imag = offset.imag  # dx
    return sample_numba(a, off_real, off_imag)


# ---------------------------------------------------------
#  Numba 核心：displace_numba（完全等价于原 roll 版本）
# ---------------------------------------------------------

@njit(parallel=True, fastmath=True)
def displace_numba(a, delta_real, delta_imag):
    H, W = a.shape
    out = np.zeros_like(a)

    for i in prange(H):
        for j in range(W):

            accum = 0.0

            # dx,dy 的语义：与原版 roll 的顺序一致
            for dx in range(-1, 2):    # 对应 axis=1（列偏移）
                for dy in range(-1, 2):  # 对应 axis=0（行偏移）

                    # roll 等价：原位置是 [i-dy, j-dx]
                    ii = (i - dy) % H
                    jj = (j - dx) % W

                    # 权重必须取“原位置”的 delta，而不是当前 (i,j)
                    v_real = delta_real[ii, jj]
                    v_imag = delta_imag[ii, jj]

                    # fns[dx](real)
                    if dx == -1:
                        wx = -v_real
                    elif dx == 0:
                        wx = 1.0 - abs(v_real)
                    else:
                        wx = v_real
                    if wx < 0: wx = 0

                    # fns[dy](imag)
                    if dy == -1:
                        wy = -v_imag
                    elif dy == 0:
                        wy = 1.0 - abs(v_imag)
                    else:
                        wy = v_imag
                    if wy < 0: wy = 0

                    w = wx * wy
                    if w > 0:
                        accum += w * a[ii, jj]

            out[i, j] = accum

    return out




# ---------------------------------------------------------
#  外壳：displace()
# ---------------------------------------------------------

def displace(a, delta):
    """
    API：保持 erosion.py 中 displace(a, gradient) 调用不变
    """
    return displace_numba(a, delta.real, delta.imag)


# ---------------------------------------------------------
#  gaussian_blur（保持原实现，不动）
# ---------------------------------------------------------

def gaussian_blur(a, sigma=1.0, kernel=None):
    """
    高斯模糊(FFT实现)

    注：你已经在 erosion.py 用 kernel 预计算提升性能，
        这里保持原功能即可。
    """
    if kernel is None:
        freqs = tuple(np.fft.fftfreq(n, d=1.0 / n) for n in a.shape)
        freq_radial = np.hypot(*np.meshgrid(*freqs))
        sigma2 = sigma**2
        g = lambda x: ((2 * np.pi * sigma2) ** -0.5) * np.exp(-0.5 * (x / sigma)**2)
        kernel = g(freq_radial)
        kernel /= kernel.sum()

    return np.fft.ifft2(np.fft.fft2(a) * np.fft.fft2(kernel)).real


# ---------------------------------------------------------
# normalize
# ---------------------------------------------------------

def normalize(x, bounds=(0, 1)):
    return np.interp(x, (x.min(), x.max()), bounds)
