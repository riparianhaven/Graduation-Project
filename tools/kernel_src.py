# -*- coding: utf-8 -*-
"""★ 全项目唯一的 Duffing 数值内核（single source of truth）。

**一个内核，两个实现**，签名与返回值完全相同，只有算术域不同：

- `KERNEL_SRC`        → `simulate_seq()`        浮点（双精度），主力实现；
- `FIXED_KERNEL_SRC`  → `simulate_fixed_seq()`  Q2.F 定点复刻，用于阶段 5
  的嵌入式验证。它不是"另一套模型"：同一方程、同一 RK4、同一份采样序列，
  唯一差别是每步算术都落在整型上（乘法后舍入）。所以可以**把同一个序列
  喂给两个函数直接比 pstd**，量化代价就是这么量出来的。

项目里所有跑仿真的地方——`learn/2,3,4,5`、根目录 `复现-2`、`s2-jupyterbook`
的八章实验——都从这里的源码取，不允许再各写一份：

1. `tools/build_stage2/3/4/5_notebook.py` 把源码原样拼进 notebook（notebook
   保持自包含，不 import 本模块）；
2. `s2-jupyterbook/_src/s2sim.py` 用 `load_kernel()` 把同一段源码 exec 进模块
   命名空间，于是 `s2sim.simulate_seq` 与 notebook 里的实现逐字相同。

宿主必须事先定义四个全局名（内核体里引用它们，定点版还需要 `qmul/to_q/from_q`
与小数位宽 `F`）：
    TWOPI = 2*pi          # 内驱周期（数值时间轴上恒为 2π）
    K     = 0.5           # 阻尼（有些 notebook 里叫 DAMP，自行别名）
    R0    = 1.0           # 无待测信号时的尺度系数，使 h1 = dt
    SIG0  = np.zeros(1)   # 全零采样序列 = 「没有待测信号，只有内驱」
    LAM0  = 0.72          # 内驱幅值默认值（内核签名里的默认参数）
    H     = 0.02          # 采样间隔默认值

`tests/test_kernel_contract.py` 会校验各 notebook 里的实现与源码逐字一致。
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["KERNEL_SRC", "FIXED_KERNEL_SRC", "QTOOLS_SRC",
           "load_kernel", "load_fixed_kernel", "DEFAULTS",
           "qmul", "to_q", "from_q"]

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

    两条路径，**同一套 RK4、逐位相同的算术**，只是求值方式不同：

    * 微内核（m ≤ 32）：驱动项 λcos(τ+β)+s(τ) 在循环外按列预算成 Python
      float 列表，循环体只剩纯标量算术。阵列路径每步约 70 次 numpy 调用，
      在小数组上全是调用开销；换成标量后单振子快约 3 倍、3 路快约 11 倍。
    * 阵列路径（m > 32）：沿通道维一次算 m 路，把 numpy 的调用开销摊薄。
      实测交叉点在 40 路附近（32 路时微内核仍略快，64 路起阵列快 1.5 倍）。

    两条路径对同一问题给出**逐位相同**的 pstd（tests 里有专门一条，
    在 m=32/33 的边界上比对），所以换路径不会动任何已验证的数值。
    """
    lam, beta = [a.ravel().astype(float) for a in np.broadcast_arrays(
        np.asarray(lam, float), np.asarray(beta, float))]
    m = lam.size

    sig = np.asarray(sig, float)
    if sig.ndim == 1:
        sig = sig.reshape(-1, 1)
    if sig.shape[1] == 1 and m > 1:                 # 单列按振子数广播
        sig = np.broadcast_to(sig, (sig.shape[0], m))
    elif sig.shape[1] != m:
        raise ValueError(f'sig 有 {sig.shape[1]} 列，与振子数 {m} 不符')
    n = sig.shape[0]

    h1 = R * dt
    steps = int(round(n_cycles * TWOPI / h1))
    bsteps = int(round(steps * burn_frac))
    h12 = h1 / 2                                    # 预算一次，与逐步 h1/2 同值
    h16 = h1 / 6

    if m <= 32:
        # ---------------------------------------------------------- 微内核
        # 驱动表按列生成：每列只有 3 条 Python float 列表（O(steps) 内存），
        # 不在循环里做 np.cos、也不在循环里索引 numpy 数组。
        k = min(n, steps)
        tauv = np.arange(steps, dtype=np.float64) * h1   # 与逐步 i*h1 逐位相同
        PJ, MX, TX, TY = [], [], [], []
        bad = False
        for j in range(m):
            lam_j, beta_j = float(lam[j]), float(beta[j])
            col = sig[:, j]
            s0 = np.zeros(steps)
            s1 = np.zeros(steps)
            if k:
                s0[:k] = col[:k]
                if k > 1:
                    s1[:k - 1] = col[1:k]
                s1[k - 1] = col[k - 1]              # 末点 s1 = s0（与阵列路径同）
            a1 = (lam_j * np.cos(tauv + beta_j) + s0).tolist()
            a2 = (lam_j * np.cos((tauv + h12) + beta_j) + 0.5 * (s0 + s1)).tolist()
            a3 = (lam_j * np.cos((tauv + h1) + beta_j) + s1).tolist()

            x = y = x_prev = 0.0
            mx = 0.0
            idx_prev = 0
            pj = []
            tj = [] if trace else None
            xj = [] if trace else None
            yj = [] if trace else None
            for i in range(steps):
                f1 = a1[i]
                fh = a2[i]
                f4 = a3[i]
                # ---- RK4：k2/k3 取 h/2 处、k4 取 h 处
                k1x = y
                k1y = -K * y + x ** 3 - x ** 5 + f1
                x2 = x + h12 * k1x
                y2 = y + h12 * k1y
                k2x = y2
                k2y = -K * y2 + x2 ** 3 - x2 ** 5 + fh
                x3 = x + h12 * k2x
                y3 = y + h12 * k2y
                k3x = y3
                k3y = -K * y3 + x3 ** 3 - x3 ** 5 + fh
                x4 = x + h1 * k3x
                y4 = y + h1 * k3y
                k4x = y4
                k4y = -K * y4 + x4 ** 3 - x4 ** 5 + f4
                x_prev = x                          # 更新前存档，供 2π 时刻插值
                x = x + h16 * (k1x + 2 * k2x + 2 * k3x + k4x)
                y = y + h16 * (k1y + 2 * k2y + 2 * k3y + k4y)

                if not (i & 255) and not math.isfinite(x):
                    bad = True
                    break

                if i >= bsteps:
                    ax = abs(x)
                    if ax > mx:
                        mx = ax
                    # 精确落在 τ=2πk 上取截面（按步数取整会让相位每周期漂
                    # 0.003 rad，周期态 pstd 从 3e-6 抬到 1e-2）
                    tau_i = i * h1
                    idx = int(math.floor((tau_i + h1) / TWOPI + 1e-12))
                    if idx != idx_prev and idx > 0:
                        frac = min(max((TWOPI * idx - tau_i) / h1, 0.0), 1.0)
                        pj.append(x_prev + frac * (x - x_prev))
                        idx_prev = idx
                    if trace:
                        tj.append(tau_i)
                        xj.append(x)
                        yj.append(y)
            if bad:
                break
            PJ.append(pj)
            MX.append(mx)
            if trace:
                TX.append(xj)
                TY.append(yj)

        if bad:
            return dict(pstd=np.full(m, np.nan), mx=np.full(m, np.nan),
                        p=np.empty((0, m)), t=None, X=None, Y=None)
        if PJ:
            half = [q[len(q) // 2:] for q in PJ]        # 与阵列路径同：丢掉前一半
            L = min(len(q) for q in half)
        else:
            L = 0
        if L:
            p = np.empty((L, m))                        # C 序，std 的归约顺序与阵列同
            for j in range(m):
                p[:, j] = half[j][:L]
            pstd = p.std(axis=0)
        else:
            p = np.empty((0, m))
            pstd = np.full(m, np.nan)
        mx = np.asarray(MX, float) if MX else np.zeros(m)
        if trace:
            T_ = np.asarray(TX[0])                      # 各通道的 τ 完全相同，取第一列
            X_ = (np.asarray(TX[0]).reshape(-1, 1) if m == 1
                  else np.ascontiguousarray(np.asarray(TX).T))
            Y_ = (np.asarray(TY[0]).reshape(-1, 1) if m == 1
                  else np.ascontiguousarray(np.asarray(TY).T))
        else:
            T_ = X_ = Y_ = None
        return dict(pstd=pstd, mx=np.atleast_1d(mx), p=p, t=T_, X=X_, Y=Y_)

    # ---------------------------------------------------------- 阵列路径
    zero = np.zeros(m)
    x, y = np.zeros(m), np.zeros(m)
    mx, x_prev = np.zeros(m), np.zeros(m)
    ab = np.empty(m)                                # 复用缓冲：abs 不每次分配
    idx_prev = 0
    pts, tr_t, tr_x, tr_y = [], [], [], []

    for i in range(steps):
        tau = i * h1
        # ---- 驱动项：内驱 + 第 i 个样本（零阶保持，中点取相邻两样本均值）
        if i < n:
            s0 = sig[i]
            s1 = sig[i + 1] if i + 1 < n else s0
        else:
            s0 = s1 = zero
        f1 = lam * np.cos(tau + beta) + s0
        fh = lam * np.cos(tau + h12 + beta) + 0.5 * (s0 + s1)
        f4 = lam * np.cos(tau + h1 + beta) + s1

        # ---- RK4：与微内核逐行同构
        k1x = y
        k1y = -K * y + x ** 3 - x ** 5 + f1
        x2, y2 = x + h12 * k1x, y + h12 * k1y
        k2x = y2
        k2y = -K * y2 + x2 ** 3 - x2 ** 5 + fh
        x3, y3 = x + h12 * k2x, y + h12 * k2y
        k3x = y3
        k3y = -K * y3 + x3 ** 3 - x3 ** 5 + fh
        x4, y4 = x + h1 * k3x, y + h1 * k3y
        k4x = y4
        k4y = -K * y4 + x4 ** 3 - x4 ** 5 + f4
        x_prev = x                                  # 更新前存档，供 2π 时刻插值
        x = x + h16 * (k1x + 2 * k2x + 2 * k3x + k4x)
        y = y + h16 * (k1y + 2 * k2y + 2 * k3y + k4y)

        if not (i & 255) and not np.all(np.isfinite(x)):
            return dict(pstd=np.full(m, np.nan), mx=np.atleast_1d(mx),
                        p=np.empty((0, m)), t=None, X=None, Y=None)

        if i >= bsteps:
            np.abs(x, out=ab)
            np.maximum(mx, ab, out=mx)
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
                tr_x.append(x.copy())
                tr_y.append(y.copy())

    p_all = np.asarray(pts)
    if p_all.size:
        p = p_all[len(p_all) // 2:]
        if p.ndim == 1:
            p = p.reshape(-1, 1)
        p = np.ascontiguousarray(p)                 # std 按 C 序归约，与微内核一致
        pstd = p.std(axis=0)
    else:
        p = np.empty((0, m))
        pstd = np.full(m, np.nan)
    if trace:
        T_ = np.asarray(tr_t)
        X_ = np.asarray(tr_x)
        Y_ = np.asarray(tr_y)
    else:
        T_ = X_ = Y_ = None
    return dict(pstd=pstd, mx=np.atleast_1d(mx), p=p, t=T_, X=X_, Y=Y_)
'''

# 去掉定界用的首尾空行：拼接进 notebook 时不能多出换行（否则与既有单元差一个空行）
KERNEL_SRC = KERNEL_SRC.strip("\n") + "\n"


# ---------------------------------------------------------------- Q 格式工具
# 定点内核与 notebook 的 0.4 单元共用这一份实现：抄两份必然对不上。
QTOOLS_SRC = r'''
def qmul(a, b, S):
    """定点乘：a,b 为 Q2.F 整型，S=2^F；结果舍入回整型。"""
    return np.rint(a.astype(np.float64) * b.astype(np.float64) / S).astype(np.int64)


def to_q(v, S):
    """浮点 → Q2.F 整型。"""
    return np.rint(np.asarray(v) * S).astype(np.int64)


def from_q(q, S):
    """Q2.F 整型 → 浮点。"""
    return np.asarray(q, dtype=np.float64) / S
'''

_qns = {"np": np}
exec(compile(QTOOLS_SRC, "<qtools_src>", "exec"), _qns)
qmul, to_q, from_q = _qns["qmul"], _qns["to_q"], _qns["from_q"]


# ---------------------------------------------------------------- 定点复刻
FIXED_KERNEL_SRC = r'''
def _qacc(xq, yq, fq, S):
    """定点加速度 -K·y + x³ - x⁵ + f（全程 Q2.F 整型）。

    K·y 用 round 而非右移：右移对负数是向下取整，会引入负向偏置，
    在混沌边缘把轨迹系统性地推向一侧。
    """
    x2 = qmul(xq, xq, S)
    x3 = qmul(x2, xq, S)
    x5 = qmul(x3, x2, S)
    return -np.rint(K * yq).astype(np.int64) + x3 - x5 + fq


def simulate_fixed_seq(sig, R=R0, dt=H, lam=LAM0, beta=0.0,
                       n_cycles=150.0, burn_frac=0.5, F=12, trace=False):
    """定点复刻 simulate_seq：同方程、同签名、同返回，算术落在 Q2.F 整型上。

    参数与 simulate_seq 逐项相同，只多一个 F（小数位宽，总位宽 W=2+F）。
    本实现只走向量路径（m=1 也当 1 元素数组）——硬件里单振子与阵列走的是
    同一条数据通路，没有"快路径"这回事。

    模拟的是**舍入误差**，不模拟位宽溢出与饱和（用 int64 承载），所以 F 在这里
    是"精度"刻度而非"范围"约束：F 越小，等效阈值漂得越多，直至动力学塌缩。
    """
    lam, beta = [a.ravel().astype(float) for a in np.broadcast_arrays(
        np.asarray(lam, float), np.asarray(beta, float))]
    m = lam.size

    sig = np.asarray(sig, float)
    if sig.ndim == 1:
        sig = sig.reshape(-1, 1)
    if sig.shape[1] == 1 and m > 1:
        sig = np.broadcast_to(sig, (sig.shape[0], m))
    elif sig.shape[1] != m:
        raise ValueError(f'sig 有 {sig.shape[1]} 列，与振子数 {m} 不符')
    n = sig.shape[0]

    h1 = R * dt
    steps = int(round(n_cycles * TWOPI / h1))
    bsteps = int(round(steps * burn_frac))
    S = 1 << F
    zero = np.zeros(m)

    x = np.zeros(m, dtype=np.int64)
    y = np.zeros(m, dtype=np.int64)
    mx = np.zeros(m)
    idx_prev = 0
    pts, tr_t, tr_x, tr_y = [], [], [], []

    for i in range(steps):
        tau = i * h1
        # ---- 驱动项：内驱 + 第 i 个样本，整段量化（与浮点版同一位置取样本）
        if i < n:
            s0 = sig[i]
            s1 = sig[i + 1] if i + 1 < n else s0
        else:
            s0 = s1 = zero
        f1 = to_q(lam * np.cos(tau + beta) + s0, S)
        fh = to_q(lam * np.cos(tau + h1 / 2 + beta) + 0.5 * (s0 + s1), S)
        f4 = to_q(lam * np.cos(tau + h1 + beta) + s1, S)

        # ---- RK4：k2/k3 取 h/2 处、k4 取 h 处（与浮点内核逐行同构）
        k1x = y
        k1y = _qacc(x, y, f1, S)
        xa = x + to_q(h1 / 2 * k1x, 1)
        ya = y + to_q(h1 / 2 * k1y, 1)
        k2x = ya
        k2y = _qacc(xa, ya, fh, S)
        xb = x + to_q(h1 / 2 * k2x, 1)
        yb = y + to_q(h1 / 2 * k2y, 1)
        k3x = yb
        k3y = _qacc(xb, yb, fh, S)
        xc = x + to_q(h1 * k3x, 1)
        yc = y + to_q(h1 * k3y, 1)
        k4x = yc
        k4y = _qacc(xc, yc, f4, S)
        x_prev = from_q(x, S)                       # 更新前存档，供 2π 时刻插值
        x = x + to_q(h1 / 6 * (k1x + 2 * k2x + 2 * k3x + k4x), 1)
        y = y + to_q(h1 / 6 * (k1y + 2 * k2y + 2 * k3y + k4y), 1)

        if i >= bsteps:
            np.maximum(mx, np.abs(from_q(x, S)), out=mx)
            idx = int(math.floor((tau + h1) / TWOPI + 1e-12))
            if idx != idx_prev and idx > 0:
                frac = min(max((TWOPI * idx - tau) / h1, 0.0), 1.0)
                pts.append(x_prev + frac * (from_q(x, S) - x_prev))
                idx_prev = idx
            if trace:
                tr_t.append(tau)
                tr_x.append(from_q(x, S).copy())
                tr_y.append(from_q(y, S).copy())

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
        T_, X_, Y_ = np.asarray(tr_t), np.asarray(tr_x), np.asarray(tr_y)
    else:
        T_ = X_ = Y_ = None
    return dict(pstd=pstd, mx=np.atleast_1d(mx), p=p, t=T_, X=X_, Y=Y_)
'''


def load_fixed_kernel(**over):
    """把 FIXED_KERNEL_SRC exec 进新命名空间，返回 (simulate_fixed_seq, ns)。

    over 可覆盖 DEFAULTS 里的常量；Q 格式工具与浮点版共用同一份源码。
    """
    consts = dict(DEFAULTS)
    consts.update(over)
    ns = dict(math=math, np=np, SIG0=np.zeros(1),
              qmul=qmul, to_q=to_q, from_q=from_q)
    ns.update(consts)
    exec(compile(FIXED_KERNEL_SRC, "<fixed_kernel_src>", "exec"), ns)
    return ns["simulate_fixed_seq"], ns
