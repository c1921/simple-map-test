"""
侵蚀模拟工具函数
移植自参考项目 reference/util.py
"""

import numpy as np


def lerp(x, y, a):
    """线性插值: (1-a)*x + a*y"""
    return (1.0 - a) * x + a * y


def simple_gradient(a):
    """
    计算地形梯度(使用复数编码向量)

    参数:
        a: 地形高度数组

    返回:
        复数数组,实部=dx,虚部=dy
        使用周期性边界条件
    """
    dx = 0.5 * (np.roll(a, 1, axis=0) - np.roll(a, -1, axis=0))
    dy = 0.5 * (np.roll(a, 1, axis=1) - np.roll(a, -1, axis=1))
    return 1j * dx + dy


def sample(a, offset):
    """
    双线性插值采样

    参数:
        a: 输入数组
        offset: 偏移量(复数编码: real=dy, imag=dx)

    返回:
        在偏移位置的插值结果,使用周期性边界
    """
    shape = np.array(a.shape)
    delta = np.array((offset.real, offset.imag))
    coords = np.array(np.meshgrid(*map(range, shape))) - delta

    lower_coords = np.floor(coords).astype(int)
    upper_coords = lower_coords + 1
    coord_offsets = coords - lower_coords
    lower_coords %= shape[:, np.newaxis, np.newaxis]
    upper_coords %= shape[:, np.newaxis, np.newaxis]

    result = lerp(lerp(a[lower_coords[1], lower_coords[0]],
                       a[lower_coords[1], upper_coords[0]],
                       coord_offsets[0]),
                  lerp(a[upper_coords[1], lower_coords[0]],
                       a[upper_coords[1], upper_coords[0]],
                       coord_offsets[0]),
                  coord_offsets[1])
    return result


def displace(a, delta):
    """
    将数组值按偏移量位移到相邻网格点

    参数:
        a: 输入数组
        delta: 位移向量(复数编码: real=dy, imag=dx)

    返回:
        位移后的数组,值分配到相邻格点(使用权重)
    """
    fns = {
        -1: lambda x: -x,
        0: lambda x: 1 - np.abs(x),
        1: lambda x: x,
    }
    result = np.zeros_like(a)
    for dx in range(-1, 2):
        wx = np.maximum(fns[dx](delta.real), 0.0)
        for dy in range(-1, 2):
            wy = np.maximum(fns[dy](delta.imag), 0.0)
            result += np.roll(np.roll(wx * wy * a, dy, axis=0), dx, axis=1)

    return result


def gaussian_blur(a, sigma=1.0):
    """
    高斯模糊(FFT实现)

    参数:
        a: 输入数组
        sigma: 高斯核标准差

    返回:
        模糊后的数组
    """
    freqs = tuple(np.fft.fftfreq(n, d=1.0 / n) for n in a.shape)
    freq_radial = np.hypot(*np.meshgrid(*freqs))
    sigma2 = sigma**2
    g = lambda x: ((2 * np.pi * sigma2) ** -0.5) * np.exp(-0.5 * (x / sigma)**2)
    kernel = g(freq_radial)
    kernel /= kernel.sum()
    return np.fft.ifft2(np.fft.fft2(a) * np.fft.fft2(kernel)).real


def normalize(x, bounds=(0, 1)):
    """
    归一化数组值到指定范围

    参数:
        x: 输入数组
        bounds: 目标范围 (min, max)

    返回:
        归一化后的数组
    """
    return np.interp(x, (x.min(), x.max()), bounds)
