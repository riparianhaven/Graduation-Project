"""★[2] 阵列 Duffing 复现的数值内核。

两个关键修正（相对项目早期的 `复现-2.ipynb`）：
1. Poincaré 采样在精确的 τ=2πk 处对 x 做线性插值，而不是「每 N 步采一次」。
   后者在 h 不整除 2π 时会引入 0.1~0.17 的伪 pstd，把周期态误判成混沌。
2. 变尺度用「数据驱动」实现：只给采样序列 sig 与采样间隔 dt，第 i 步读 sig[i]，
   步长 h1 = R*dt。全程不出现待测角频率，因此可以真正盲测。
"""
from __future__ import annotations

import math

import numpy as np

TWOPI = 2.0 * math.pi

DEFAULT = dict(
    k=0.5,          # 阻尼
    lam=0.72,       # 内驱幅值（论文工作点）
    beta=0.0,       # 内驱初相
    h=TWOPI / 300,  # 步长：整除 2π，Poincaré 无相位漂移
    t_end=400.0,
    burn=200.0,
    gamma=0.02,     # 待测信号幅值
    theta=0.0,      # 待测信号初相
    nu=1.0,         # 信号/内驱 角频率比（变尺度后 = omega/R）
    sigma=0.0,      # 噪声标准差
)


def _as1d(*vals):
    arrs = np.broadcast_arrays(*[np.atleast_1d(np.asarray(v, float)).ravel()
                                 for v in vals])
    return [a.ravel().copy() for a in arrs]


def simulate(lam, beta=0.0, gamma=0.0, theta=0.0, nu=1.0, sigma=0.0,
             seed=0, k=None, h=None, t_end=None, burn=None, trace=False):
    """解析驱动版：x'' + k x' - x^3 + x^5 = lam cos(t+beta) + gamma cos(nu t+theta) + n.

    所有参数可为标量或等长数组，一次循环并行跑完整组。
    返回 pstd（Poincaré 精确采样点的标准差）、mx、以及可选轨迹。
    """
    k = DEFAULT["k"] if k is None else k
    h = DEFAULT["h"] if h is None else h
    t_end = DEFAULT["t_end"] if t_end is None else t_end
    burn = DEFAULT["burn"] if burn is None else burn
    lam, beta, gamma, theta, nu, sigma = _as1d(lam, beta, gamma, theta, nu, sigma)
    m = lam.size

    x = np.zeros(m)
    y = np.zeros(m)
    rng = np.random.default_rng(seed)
    steps = int(round(t_end / h))
    bsteps = int(round(burn / h))
    sqh = math.sqrt(h)
    noisy = bool(np.any(sigma))
    coherent = bool(np.all(np.abs(nu - 1.0) < 1e-12))

    poincare = []
    mx = np.zeros(m)
    sq = np.zeros(m)
    cnt = 0
    x_prev = x.copy()
    idx_prev = -1
    tr_t, tr_x, tr_y = [], [], []

    for i in range(steps):
        t = i * h
        nz = rng.standard_normal(m) * sqh if noisy else 0.0
        if coherent:
            # 同频时用相量合成，省一半三角函数（数学上与直接相加等价）
            A = lam * np.cos(beta) + gamma * np.cos(theta)
            B = lam * np.sin(beta) + gamma * np.sin(theta)
            f1 = A * np.cos(t) - B * np.sin(t) + sigma * nz
            fh = A * np.cos(t + h / 2) - B * np.sin(t + h / 2) + sigma * nz
            f4 = A * np.cos(t + h) - B * np.sin(t + h) + sigma * nz
        else:
            f1 = lam * np.cos(t + beta) + gamma * np.cos(nu * t + theta) + sigma * nz
            fh = lam * np.cos(t + h / 2 + beta) + gamma * np.cos(nu * (t + h / 2) + theta) + sigma * nz
            f4 = lam * np.cos(t + h + beta) + gamma * np.cos(nu * (t + h) + theta) + sigma * nz

        x_prev = x.copy()
        k1x = y; k1y = -k * y + x ** 3 - x ** 5 + f1
        x2, y2 = x + h / 2 * k1x, y + h / 2 * k1y
        k2x = y2; k2y = -k * y2 + x2 ** 3 - x2 ** 5 + fh
        x3, y3 = x + h / 2 * k2x, y + h / 2 * k2y
        k3x = y3; k3y = -k * y3 + x3 ** 3 - x3 ** 5 + fh
        x4, y4 = x + h * k3x, y + h * k3y
        k4x = y4; k4y = -k * y4 + x4 ** 3 - x4 ** 5 + f4
        x = x + h / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        y = y + h / 6 * (k1y + 2 * k2y + 2 * k3y + k4y)

        if i >= bsteps:
            np.maximum(mx, np.abs(x), out=mx)
            sq += x * x
            cnt += 1
            # 精确 2π 时刻插值采样
            idx = int(math.floor((t + h) / TWOPI + 1e-12))
            if idx != idx_prev and idx > 0:
                tp = TWOPI * idx
                frac = (tp - t) / h
                frac = min(max(frac, 0.0), 1.0)
                poincare.append(x_prev + frac * (x - x_prev))
                idx_prev = idx
            if trace:
                tr_t.append(t); tr_x.append(x.copy()); tr_y.append(y.copy())

    p = np.asarray(poincare) if poincare else np.empty((0, m))
    p = p[len(p) // 2:] if p.shape[0] else p
    pstd = p.std(axis=0) if p.shape[0] else np.full(m, np.nan)
    out = dict(pstd=pstd, mx=mx, rms=np.sqrt(sq / max(cnt, 1)))
    if trace:
        out.update(t=np.asarray(tr_t), X=np.asarray(tr_x), Y=np.asarray(tr_y), p=p)
    return out


def simulate_seq(sig, R, dt, lam=0.72, beta=0.0, k=0.5, n_cycles=80,
                 burn_frac=0.5, sigma_extra=0.0, seed=0):
    """数据驱动版（盲测）：只给采样序列 sig 与采样间隔 dt。

    变尺度 R 的物理含义：第 i 步在数值时间轴上走 h1 = R*dt，
    而它读的是 sig[i]（对应物理时间 i*dt）。于是 sig 中角频率 ω 的分量
    在新时间轴上是 ω/R —— 这正是 ν，但代码里从未出现 ω。
    """
    h1 = R * dt
    steps = int(round(n_cycles * TWOPI / h1))
    bsteps = int(steps * burn_frac)
    N = len(sig)
    rng = np.random.default_rng(seed)
    x = y = 0.0
    poincare = []
    mx = 0.0
    x_prev = 0.0
    idx_prev = -1
    sig = np.asarray(sig, float)
    for i in range(steps):
        tau = i * h1
        s0 = sig[i] if i < N else 0.0
        s1 = sig[i + 1] if i + 1 < N else s0
        extra = sigma_extra * rng.standard_normal() if sigma_extra else 0.0
        f1 = lam * math.cos(tau + beta) + s0 + extra
        fh = lam * math.cos(tau + h1 / 2 + beta) + 0.5 * (s0 + s1) + extra
        f4 = lam * math.cos(tau + h1 + beta) + s1 + extra
        x_prev = x
        k1x = y; k1y = -k * y + x ** 3 - x ** 5 + f1
        x2, y2 = x + h1 / 2 * k1x, y + h1 / 2 * k1y
        k2x = y2; k2y = -k * y2 + x2 ** 3 - x2 ** 5 + fh
        x3, y3 = x + h1 / 2 * k2x, y + h1 / 2 * k2y
        k3x = y3; k3y = -k * y3 + x3 ** 3 - x3 ** 5 + fh
        x4, y4 = x + h1 * k3x, y + h1 * k3y
        k4x = y4; k4y = -k * y4 + x4 ** 3 - x4 ** 5 + f4
        x = x + h1 / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        y = y + h1 / 6 * (k1y + 2 * k2y + 2 * k3y + k4y)
        if not np.isfinite(x) or abs(x) > 100:
            return dict(pstd=np.nan, mx=np.nan, diverged=True)
        if i >= bsteps:
            mx = max(mx, abs(x))
            idx = int(math.floor((tau + h1) / TWOPI + 1e-12))
            if idx != idx_prev and idx > 0:
                tp = TWOPI * idx
                frac = min(max((tp - tau) / h1, 0.0), 1.0)
                poincare.append(x_prev + frac * (x - x_prev))
                idx_prev = idx
    p = np.asarray(poincare)
    p = p[len(p) // 2:] if p.size else p
    return dict(pstd=float(p.std()) if p.size else np.nan, mx=mx,
                diverged=False, n_p=int(p.size))


def phi_diagnostic(fc=None):
    """扫 λ 找真实临界（精确 2π 采样，无伪迹）。"""
    lam = np.round(np.arange(0.700, 0.7601, 0.002), 4)
    r = simulate(lam, gamma=0.0, t_end=600.0, burn=300.0)
    return lam, r["pstd"]
