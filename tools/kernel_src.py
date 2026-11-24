# -*- coding: utf-8 -*-
"""★ 全项目唯一的 Duffing 数值内核（single source of truth）。

项目里所有跑仿真的地方——`learn/2,3,4`、根目录 `复现-2`、`s2-jupyterbook` 的
八章实验——都从这里的 `KERNEL_SRC` 取同一段源码，不允许再各写一份：

1. `tools/build_stage2_notebook.py` / `build_stage3_notebook.py` /
   `build_stage4_notebook.py` 把 `KERNEL_SRC` 原样拼进 notebook（notebook 保持
   自包含，不 import 本模块）；
2. `s2-jupyterbook/_src/s2sim.py` 用 `load_kernel()` 把同一段源码 exec 进模块
   命名空间，于是 `s2sim.simulate_seq` 与 notebook 里的实现逐字相同。

宿主必须事先定义四个全局名（内核体里引用它们）：
    TWOPI = 2*pi          # 内驱周期（数值时间轴上恒为 2π）
    K     = 0.5           # 阻尼（有些 notebook 里叫 DAMP，自行别名）
    R0    = 1.0           # 无待测信号时的尺度系数，使 h1 = dt
    SIG0  = np.zeros(1)   # 全零采样序列 = 「没有待测信号，只有内驱」
    LAM0  = 0.72          # 内驱幅值默认值（内核签名里的默认参数）
    H     = 0.02          # 采样间隔默认值

`tests/test_kernel_contract.py` 会校验各 notebook 里的实现与 `KERNEL_SRC` 一致。
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["KERNEL_SRC", "load_kernel", "DEFAULTS"]

DEFAULTS = dict(TWOPI=2.0 * math.pi, K=0.5, R0=1.0, LAM0=0.72, H=0.02)


def load_kernel(**over):
    """把 KERNEL_SRC exec 进一个新命名空间，返回 (simulate_seq, ns)。

    over 用于覆盖 DEFAULTS 里的常量（例如 K=0.5 换成别的阻尼）。
    """
    consts = dict(DEFAULTS)
    consts.update(over)
    ns = dict(math=math, np=np, SIG0=np.zeros(1))
    ns.update(consts)
    exec(compile(KERNEL_SRC, "<kernel_src>", "exec"), ns)
    return ns["simulate_seq"], ns


KERNEL_SRC = r'''
def simulate_seq(sig, R=R0, dt=H, lam=LAM0, beta=0.0,
                 n_cycles=150.0, burn_frac=0.5, trace=False):
    """数据驱动（盲测）RK4：x'' = -k x' + x³ - x⁵ + λcos(τ+β) + s(τ)

    sig  : 采样序列，(N,) 或 (N, m)；第 i 个样本是物理时刻 i*dt 的值。
           传 SIG0（长度 1 的全零序列）表示「没有待测信号」，只留内驱。
    R    : 变尺度系数。数值步长 h1 = R*dt —— 第 i 步在数值时间轴上走 h1，
           读的却是物理时刻 i*dt 的样本。于是序列里角频率 ω 的分量在新时间
           轴上变成 ω/R。本函数里既没有 ω，也没有 ν：ν 只是 ω/R 的别名。
    lam, beta : 内驱幅值与初相，可为标量或等长数组（阵列并行）。
    n_cycles  : 积分多少个内驱周期（数值轴上的周期恒为 2π）。
    burn_frac : 前多少比例的样本当作瞬态丢掉（默认前一半）。
    返回 dict(pstd, mx, p, t, X, Y)；t/X/Y 仅在 trace=True 时非 None。

    单振子（m=1）走标量快路径：同一套 RK4，只把 numpy 小数组换成 Python
    float，快约 10 倍；阵列（m>1）走向量路径。两条路径公式逐行一致，只有
    驱动项取值用 math.cos 还是 np.cos 的区别，结果在 1 ulp 内相同。
    """
    lam, beta = [a.ravel().astype(float) for a in np.broadcast_arrays(
        np.asarray(lam, float), np.asarray(beta, float))]
    m = lam.size
    scalar = (m == 1)

    sig = np.asarray(sig, float)
    if sig.ndim == 1:
        sig = sig.reshape(-1, 1)
    if sig.shape[1] == 1 and not scalar:            # 单列按振子数广播
        sig = np.broadcast_to(sig, (sig.shape[0], m))
    elif sig.shape[1] != m:
        raise ValueError(f'sig 有 {sig.shape[1]} 列，与振子数 {m} 不符')
    n = sig.shape[0]

    h1 = R * dt
    steps = int(round(n_cycles * TWOPI / h1))
    bsteps = int(round(steps * burn_frac))

    if scalar:
        lam_s, beta_s = float(lam[0]), float(beta[0])
        col = sig[:, 0]
        x = y = x_prev = 0.0
        mx = 0.0
    else:
        zero = np.zeros(m)
        x, y = np.zeros(m), np.zeros(m)
        mx, x_prev = np.zeros(m), np.zeros(m)
    idx_prev = 0
    pts, tr_t, tr_x, tr_y = [], [], [], []

    for i in range(steps):
        tau = i * h1
        # ---- 驱动项：内驱 + 第 i 个样本（零阶保持，中点取相邻两样本均值）
        if i < n:
            s0 = col[i] if scalar else sig[i]
            s1 = (col[i + 1] if scalar else sig[i + 1]) if i + 1 < n else s0
        elif scalar:
            s0 = s1 = 0.0
        else:
            s0 = s1 = zero
        if scalar:
            f1 = lam_s * math.cos(tau + beta_s) + s0
            fh = lam_s * math.cos(tau + h1 / 2 + beta_s) + 0.5 * (s0 + s1)
            f4 = lam_s * math.cos(tau + h1 + beta_s) + s1
        else:
            f1 = lam * np.cos(tau + beta) + s0
            fh = lam * np.cos(tau + h1 / 2 + beta) + 0.5 * (s0 + s1)
            f4 = lam * np.cos(tau + h1 + beta) + s1

        # ---- RK4：标量与数组共用同一段公式
        k1x = y;              k1y = -K * y + x ** 3 - x ** 5 + f1
        x2, y2 = x + h1 / 2 * k1x, y + h1 / 2 * k1y
        k2x = y2;             k2y = -K * y2 + x2 ** 3 - x2 ** 5 + fh
        x3, y3 = x + h1 / 2 * k2x, y + h1 / 2 * k2y
        k3x = y3;             k3y = -K * y3 + x3 ** 3 - x3 ** 5 + fh
        x4, y4 = x + h1 * k3x, y + h1 * k3y
        k4x = y4;             k4y = -K * y4 + x4 ** 3 - x4 ** 5 + f4
        x_prev = x                                  # 更新前存档，供 2π 时刻插值
        x = x + h1 / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        y = y + h1 / 6 * (k1y + 2 * k2y + 2 * k3y + k4y)

        if i % 256 == 0 and not (math.isfinite(x) if scalar
                                 else np.all(np.isfinite(x))):
            return dict(pstd=np.full(m, np.nan), mx=np.atleast_1d(mx),
                        p=np.empty((0, m)), t=None, X=None, Y=None)

        if i >= bsteps:
            if scalar:
                mx = max(mx, abs(x))
            else:
                np.maximum(mx, np.abs(x), out=mx)
            # 精确落在 τ=2πk 上取截面：h1 不整除 2π（2π/0.02=314.159）时，
            # 按「每 314 步取一次」会让采样相位每周期漂 0.003 rad，
            # 周期态 pstd 被凭空抬到 1e-2 量级，阈值标定随之失真。
            idx = int(math.floor((tau + h1) / TWOPI + 1e-12))
            if idx != idx_prev and idx > 0:
                frac = min(max((TWOPI * idx - tau) / h1, 0.0), 1.0)
                pts.append(x_prev + frac * (x - x_prev))
                idx_prev = idx
            if trace:
                tr_t.append(tau)
                tr_x.append(x if scalar else x.copy())
                tr_y.append(y if scalar else y.copy())

    p_all = np.asarray(pts)
    if p_all.size:
        p = p_all[len(p_all) // 2:]
        if p.ndim == 1:
            p = p.reshape(-1, 1)
        pstd = p.std(axis=0)
    else:
        p = np.empty((0, m))
        pstd = np.full(m, np.nan)
    if trace:
        T_ = np.asarray(tr_t)
        X_ = np.asarray(tr_x)
        Y_ = np.asarray(tr_y)
        if scalar:
            X_ = X_.reshape(-1, 1)
            Y_ = Y_.reshape(-1, 1)
    else:
        T_ = X_ = Y_ = None
    return dict(pstd=pstd, mx=np.atleast_1d(mx), p=p, t=T_, X=X_, Y=Y_)
'''

# 去掉定界用的首尾空行：拼接进 notebook 时不能多出换行（否则与既有单元差一个空行）
KERNEL_SRC = KERNEL_SRC.strip("\n") + "\n"
