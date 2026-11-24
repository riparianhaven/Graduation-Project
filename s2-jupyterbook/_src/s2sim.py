"""★[2] 阵列 Duffing 复现的数值内核（薄封装）。

本模块**不再自带积分子**。项目里只有一个浮点内核，源码在 `tools/kernel_src.py`
的 `KERNEL_SRC` 里；这里用 `load_kernel()` 把它 exec 进本模块命名空间，于是
`s2sim.simulate_seq` 与 `learn/2`、`learn/3`、`learn/4`、根目录 `复现-2` 里的
实现逐字相同。

两条历史修正保留在共享内核里：
1. Poincaré 采样在精确的 τ=2πk 处对 x 做线性插值，而不是「每 N 步采一次」。
2. 变尺度用「数据驱动」实现：只给采样序列 sig 与采样间隔 dt，第 i 步读 sig[i]，
   步长 h1 = R*dt。全程不出现待测角频率，因此可以真正盲测。

本模块额外提供：
- `tone()`：按**指定**角频率造一段采样序列（验模台架用：已知答案，检查方法能否复现）。
  它是"造数据"，不是"检测"；检测端只吃 `tone()` 吐出来的那一列样本。
- `phi_diagnostic()`：无信号扫 λ，给临界带。
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_TOOLS = _ROOT / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from kernel_src import load_kernel  # noqa: E402

# ---- 共享内核要求的全局名（见 tools/kernel_src.py 的模块文档） -----------------
TWOPI = 2.0 * math.pi
K = 0.5                 # 阻尼
LAM0 = 0.72             # 内驱幅值（论文工作点）
H = TWOPI / 300         # 论文工作步长：整除 2π，Poincaré 无相位漂移
R0 = 1.0                # 不做变尺度时的尺度系数：h1 = R0*dt = dt
SIG0 = np.zeros(1)      # 全零采样序列 = 没有待测信号，只剩内驱

simulate_seq, _NS = load_kernel(TWOPI=TWOPI, K=K, LAM0=LAM0, H=H, R0=R0)

__all__ = ["simulate_seq", "tone", "phi_diagnostic", "TWOPI", "K", "LAM0",
           "H", "R0", "SIG0"]


def tone(gamma=0.0, theta=0.0, omega=1.0, n=1, dt=None, sigma=0.0, seed=None):
    """造一段采样序列 sig[k] = γcos(ω·k·dt+θ) + σ√dt·N(0,1)。

    omega 是**造数据用的真值**，只决定样本长什么样；检测端拿到的只有 sig 与 dt。
    sigma 为连续噪声强度，离散后每个样本的标准差是 σ√dt（与内核的力口径一致）。
    """
    dt = H if dt is None else dt
    idx = np.arange(int(n)) * dt
    s = np.asarray(gamma, float) * np.cos(omega * idx + theta)
    if sigma:
        rng = np.random.default_rng(seed)
        s = s + sigma * math.sqrt(dt) * rng.standard_normal(s.shape)
    return s


def phi_diagnostic(h=None, t_end=600.0, burn=300.0):
    """无信号扫 λ，返回 (lam, pstd)。走同一个盲测内核，喂全零序列。"""
    h = H if h is None else h
    lam = np.round(np.arange(0.700, 0.7601, 0.002), 4)
    r = simulate_seq(SIG0, R0, h, lam=lam,
                     n_cycles=t_end / TWOPI, burn_frac=burn / t_end)
    return lam, r["pstd"]
