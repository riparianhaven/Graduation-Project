# -*- coding: utf-8 -*-
"""内核回归算例矩阵（黄金样本测试、基准脚本、重冻结脚本共用一份）。

为什么单独放一个模块：算例一旦三处各写一份，就会"测试用的输入"和
"冻结样本时的输入"悄悄分家，黄金测试变成自欺欺人。

算例覆盖运行时真正会出现的形状：
    m=1（单振子）/ m=2（H0 标定 + 周期态）/ m=3（三相位阵列，GUI 默认）/
    m=32 与 m=45（微内核与阵列路径的切换边界两侧）/ m=180（估幅网格）/
    序列短于积分步数（i ≥ n 的零阶保持分支）/ burn_frac 非默认 /
    R=7.3、3.7（变尺度扫频）/ trace=True。

输入全部由固定种子生成，**不许改**（改了就等于换了题，黄金样本失效）。
"""
from __future__ import annotations

import math

import numpy as np

TWOPI = 2.0 * math.pi
B3 = (0.0, -2.0 * math.pi / 3.0, 2.0 * math.pi / 3.0)
B4 = (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0)

#: 与阶段 2–5 完全一致的链路参数（ω=10 rad/s、fs=500 Hz）
FS = 500.0
W = 10.0
DUR = 94.2


def cycles(n: int, R: float, dt: float) -> float:
    """由样本数反推周期数，使内核的 steps 恰好等于 n（接收端用的就是这条）。"""
    return n * R * dt / TWOPI


def frame(n: int = 47100, seed: int = 20261126) -> np.ndarray:
    """一帧「信号 + 噪声」接收序列（A=0.02、σ=0.05，力口径 −7.4 dB）。"""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    return 0.02 * np.cos(W * t) + 0.05 * math.sqrt(W / FS) * rng.standard_normal(n)


def cases() -> dict[str, tuple[np.ndarray, dict]]:
    """name → (sig, kwargs)。kwargs 直接喂给 simulate_seq。"""
    fr = frame()
    return {
        'idle-lam-072-076': (
            np.zeros(1),
            dict(R=1.0, dt=0.02, lam=[0.72, 0.76], n_cycles=400.0 / TWOPI)),
        'scalar-lock': (
            fr[:20000],
            dict(R=10.0, dt=0.002, lam=0.73, beta=0.0,
                 n_cycles=cycles(20000, 10.0, 0.002))),
        'three-phase': (
            fr,
            dict(R=10.0, dt=0.002, lam=0.73, beta=list(B3),
                 n_cycles=cycles(fr.size, 10.0, 0.002))),
        'three-phase-trace': (
            fr,
            dict(R=10.0, dt=0.002, lam=0.73, beta=list(B3),
                 n_cycles=cycles(fr.size, 10.0, 0.002), trace=True)),
        'm32-broadcast': (
            fr,
            dict(R=10.0, dt=0.002, lam=[0.73] * 32,
                 beta=[0.11 * j for j in range(32)],
                 n_cycles=cycles(fr.size, 10.0, 0.002))),
        'm45-vector': (
            fr,
            dict(R=10.0, dt=0.002, lam=[0.73] * 45,
                 beta=[0.07 * j for j in range(45)],
                 n_cycles=cycles(fr.size, 10.0, 0.002))),
        'short-sig-long-int': (
            fr[:5000],
            dict(R=10.0, dt=0.002, lam=0.72, n_cycles=300.0)),
        'burn-0.3': (
            fr[:20000],
            dict(R=10.0, dt=0.002, lam=0.73, n_cycles=cycles(20000, 10.0, 0.002),
                 burn_frac=0.3)),
        'R-7.3': (
            fr[:20000],
            dict(R=7.3, dt=0.002, lam=0.73, n_cycles=cycles(20000, 7.3, 0.002))),
        'R-3.7': (
            fr[:20000],
            dict(R=3.7, dt=0.002, lam=0.73, n_cycles=cycles(20000, 3.7, 0.002))),
        'amp-grid': (
            fr[:8000],
            dict(R=10.0, dt=0.002,
                 lam=[0.68 + 0.0025 * (k % 45) for k in range(180)],
                 beta=[B4[k // 45] for k in range(180)],
                 n_cycles=cycles(8000, 10.0, 0.002))),
    }
