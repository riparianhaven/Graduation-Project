"""C2 强耦合 Duffing 振子复现核心（批量 RK4）。

论文：曹保锋 等, 物理学报 2019, 68(8): 080501.
方程编号沿用原文，见 `paper_eq` 注释。

设计要点
--------
1. 批量推进：状态为 (N, 4)，N 个独立样本共享同一组系统参数、各自不同的噪声实现。
2. 只保存观测量 so = ẋ1 − ẋ2，不存完整轨迹，避免 100 样本 × 4 万步的内存爆炸。
3. 噪声采用零阶保持（hold 步更新一次），等效带宽 1/(2·hold·h)；
   论文未交代 n(t) 的离散化方式，这一点在书里单独讨论。

与阶段 2 内核的关系
------------------
本模块**不是** `tools/kernel_src.py` 那个单振子内核的副本，也不该并入：C2 是强耦合
双振子（4 维状态），方程不同；且 `simulate(t, drive=...)` 由外部时间序列驱动，
不存在"待测频率 ω 未知、要用 R 去扫"的变尺度问题。项目里做极低频磁信号检测
（★[2] 单振子）的那条线，唯一内核是 `simulate_seq(sig, R, dt)`。
"""
from __future__ import annotations

import numpy as np

# 论文表 1 默认值
DEFAULT = dict(F=0.2, omega=5e6, k=10.0, xi=0.7, h=1e-9)
X0 = (1.4, 4.0, 1.4, 4.0)          # 论文 5.2 节给定初始值


def _acc(x1, v1, x2, v2, t, d1, p):
    """方程 (4) 的加速度项。d1 为振子 1 的附加驱动（s+n），振子 2 无附加项。

    p["weak"]=True 时去掉阻尼力耦合项 ωk(ẋ1−ẋ2)，退化为文献 [19] 的弱耦合模型，
    用于原文 5.3 节的对比实验（图 11）。
    """
    w, xi, k, F = p["omega"], p["xi"], p["k"], p["F"]
    w2 = w * w
    dk = 0.0 if p.get("weak") else 1.0
    a1 = (-w * xi * v1 - w2 * xi * k * (x1 - x2) - dk * w * k * (v1 - v2)
          + w2 * (x1 - x1 ** 3) + w2 * (F * np.cos(w * t) + d1))
    a2 = (-w * xi * v2 - w2 * xi * k * (x2 - x1) - dk * w * k * (v2 - v1)
          + w2 * (x2 - x2 ** 3) + w2 * F * np.cos(w * t))
    return a1, a2


def simulate(t, drive=None, batch=1, hold=1, sigma=0.0, seed=0,
             x0=X0, return_state=False, **kw):
    """在给定时间栅格上积分方程 (4)。

    Parameters
    ----------
    t : (M,) 时间栅格（等间隔）
    drive : callable(t)->(M,) 或 (M,) 确定性附加驱动 s(t)（不含噪声）
    batch : 独立样本数（各自独立的噪声实现）
    hold : 噪声零阶保持步数；等效带宽 = 1/(2·hold·h)
    sigma : 每步噪声标准差（仅当 batch>1 或 sigma>0 时注入）
    return_state : 为 True 时同时返回 (x1,v1,x2,v2) 末态与轨迹采样

    Returns
    -------
    so : (batch, M) 输出 ẋ1 − ẋ2
    X  : (batch, M, 4) 仅在 return_state=True 时返回
    """
    p = dict(DEFAULT); p.update({k: v for k, v in kw.items() if k in DEFAULT})
    p["weak"] = bool(kw.get("weak", False))
    h = float(t[1] - t[0]); n = len(t) - 1
    w = p["omega"]

    if drive is None:
        d = np.zeros(len(t))
    elif callable(drive):
        d = np.asarray(drive(t), dtype=float)
    else:
        d = np.asarray(drive, dtype=float)

    rng = np.random.default_rng(seed)
    if sigma > 0:
        reps = int(np.ceil(len(t) / hold))
        z = rng.normal(0.0, sigma, (batch, reps))
        d = d[None, :] + np.repeat(z, hold, axis=1)[:, :len(t)]
    else:
        d = np.repeat(d[None, :], batch, axis=0)

    x1 = np.full(batch, x0[0]); v1 = np.full(batch, x0[1])
    x2 = np.full(batch, x0[2]); v2 = np.full(batch, x0[3])
    so = np.empty((batch, len(t)), dtype=np.float64)
    so[:, 0] = v1 - v2
    X = np.empty((batch, len(t), 4), dtype=np.float32) if return_state else None
    if return_state:
        X[:, 0] = np.stack([x1, v1, x2, v2], axis=1)

    for i in range(n):
        ti = t[i]; di = d[:, i]; dn = d[:, i + 1]
        a1, a2 = _acc(x1, v1, x2, v2, ti, di, p)
        K1 = (v1, a1, v2, a2)
        a1, a2 = _acc(x1 + h / 2 * K1[0], v1 + h / 2 * K1[1],
                      x2 + h / 2 * K1[2], v2 + h / 2 * K1[3],
                      ti + h / 2, 0.5 * (di + dn), p)
        K2 = (v1 + h / 2 * K1[1], a1, v2 + h / 2 * K1[3], a2)
        a1, a2 = _acc(x1 + h / 2 * K2[0], v1 + h / 2 * K2[1],
                      x2 + h / 2 * K2[2], v2 + h / 2 * K2[3],
                      ti + h / 2, 0.5 * (di + dn), p)
        K3 = (v1 + h / 2 * K2[1], a1, v2 + h / 2 * K2[3], a2)
        a1, a2 = _acc(x1 + h * K3[0], v1 + h * K3[1],
                      x2 + h * K3[2], v2 + h * K3[3],
                      ti + h, dn, p)
        K4 = (v1 + h * K3[1], a1, v2 + h * K3[3], a2)

        x1 = x1 + h / 6 * (K1[0] + 2 * K2[0] + 2 * K3[0] + K4[0])
        v1 = v1 + h / 6 * (K1[1] + 2 * K2[1] + 2 * K3[1] + K4[1])
        x2 = x2 + h / 6 * (K1[2] + 2 * K2[2] + 2 * K3[2] + K4[2])
        v2 = v2 + h / 6 * (K1[3] + 2 * K2[3] + 2 * K3[3] + K4[3])
        so[:, i + 1] = v1 - v2
        if return_state:
            X[:, i + 1, 0] = x1; X[:, i + 1, 1] = v1
            X[:, i + 1, 2] = x2; X[:, i + 1, 3] = v2
    return (so, X) if return_state else so


# ---------- 被检测波形（论文 5.1 节表 2） ----------

def gauss(t, A=0.632, c=3e-6, b=None, T=None):
    """高斯脉冲，论文式 (8)。"""
    b = (T / 2 if T is not None else 20e-6) if b is None else b
    return A * np.exp(-((t - b) / c) ** 2)


def square(t, A=0.2, t0=1.0, width=1.0, rise=1e-4):
    """方波：上升时间 0.1 ms、半高宽 1 s（表 2）。"""
    up = np.clip((t - t0) / rise, 0, 1)
    down = np.clip((t - (t0 + width)) / rise, 0, 1)
    return A * (up - down)


def double_exp(t, A=0.2, t0=100e-6, tau1=3e-6, tau2=25e-6):
    """双指数脉冲：上升 3 μs、半高宽 25 μs（表 2）。"""
    s = np.where(t >= t0, np.exp(-(t - t0) / tau2) - np.exp(-(t - t0) / tau1), 0.0)
    return A * s / (s.max() if s.max() > 0 else 1.0)


def gauss_deriv(t, A=0.2, t0=100e-6, c=1.45e-6):
    """高斯导数脉冲：峰峰宽 1.45 μs（表 2）。"""
    u = (t - t0) / c
    s = -u * np.exp(-0.5 * u ** 2)
    return A * s / (np.abs(s).max() if np.abs(s).max() > 0 else 1.0)


# ---------- 指标 ----------

def waveform_similarity(x, y):
    """论文式 (5)：皮尔逊相关系数。"""
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x - x.mean(); y = y - y.mean()
    den = np.sqrt((x ** 2).sum() * (y ** 2).sum())
    return float((x * y).sum() / den) if den > 0 else 0.0


def snr_db(a, D):
    """论文式 (7)：SNR = 10 log10(a²/(4D))。"""
    return 10 * np.log10(a ** 2 / (4 * D))


# ---------- 信号链：输出量 → 限带 → 判据 → 指标 ----------
#
# 这一层是为了解决"同一组数据算出两套结论"的问题：原文只给了差分输出与 5σ 判据，
# 没有写输出是否限带、SNR 用哪个口径。下面每个函数都把口径写在签名里，调用处不许再改。

def lowpass(x, fs, fc, order=4):
    """零相位 FFT 低通，幅频为 order 阶 Butterworth。x 可为 (..., M)。

    用 FFT 而非 IIR 的原因：零相位，不会把脉冲峰挪位；且无 scipy 依赖。
    边界为镜像延拓，避免两端振铃污染判据窗口。
    """
    x = np.asarray(x, float)
    n = x.shape[-1]
    pad = min(n // 10, int(round(0.05 * n)) + 1)
    xp = np.concatenate([x[..., pad:0:-1], x, x[..., -2:-pad - 2:-1]], axis=-1)
    f = np.fft.rfftfreq(xp.shape[-1], 1.0 / fs)
    H = 1.0 / np.sqrt(1.0 + (f / fc) ** (2 * order))
    y = np.fft.irfft(np.fft.rfft(xp, axis=-1) * H, n=xp.shape[-1], axis=-1)
    return y[..., pad:pad + n]


def quiet_mask(t, t0, t1, guard=0.0):
    """脉冲之外（可加保护间隔）的静默段，用于估计噪声。"""
    return (t < t0 - guard) | (t > t1 + guard)


def integrate(so, h, quiet):
    """去基线后累积积分（还原被微分的波形）。quiet 为静默段掩码。"""
    so = np.asarray(so, float)
    base = so[:, quiet].mean(axis=1, keepdims=True)
    return np.cumsum(so - base, axis=1) * h


def ratio_5sigma(x, quiet, k=5.0):
    """原文判据：输出脉冲幅度 / 噪声均方差，超过 k 记为检出。"""
    x = np.asarray(x, float)
    r = np.abs(x).max(axis=1) / x[:, quiet].std(axis=1)
    return r, r > k


def snr_out(ratio):
    """把峰噪比换成与式 (7) 同一口径的输出 SNR。

    式 (7) 是 10log10(a²/4D)。令 a=峰值、D=σ²，则 10log10(a²/4σ²)
    = 20log10(ratio) − 6.02。于是 ratio=5 恰好给出 7.96 dB，与原文
    "5 倍噪声均方差（输出信噪比大于 7.96 dB）"严格对齐。
    """
    return 20 * np.log10(np.asarray(ratio, float)) - 20 * np.log10(2.0)


def chain_metrics(so, X, s, t, h, fc=200e3, quiet=None, t_pulse=None):
    """统一口径的完整指标。返回 dict，键名即口径。

    Parameters
    ----------
    so : (B, M) 速度差 ẋ1−ẋ2（原文 5.2 节所写）
    X  : (B, M, 4) 状态轨迹（取 x1−x2 作位移差，原文图 8–10 纵轴所标）
    s  : (M,) 输入脉冲（不含噪声）
    fc : 限带截止频率；None 表示不限带
    quiet : 噪声估计窗；None 时用脉冲两侧
    """
    so = np.asarray(so, float)
    dx = X[:, :, 0] - X[:, :, 2]
    if quiet is None:
        if t_pulse is None:
            t_pulse = (0.5 * (t[0] + t[-1]),) * 2
        quiet = quiet_mask(t, t_pulse[0], t_pulse[1], guard=3e-6)

    if fc:
        dx_b = lowpass(dx, 1.0 / h, fc)
        so_b = lowpass(so, 1.0 / h, fc)
    else:
        dx_b, so_b = dx, so
    it_b = lowpass(integrate(so, h, quiet), 1.0 / h, fc) if fc else integrate(so, h, quiet)

    r_dx = float(np.mean([waveform_similarity(dx_b[i], s) for i in range(dx.shape[0])]))
    r_it = float(np.mean([waveform_similarity(it_b[i], s) for i in range(dx.shape[0])]))
    ratio, hit = ratio_5sigma(dx_b, quiet)
    return dict(fc=fc, r_dx=r_dx, r_int=r_it,
                ratio=float(np.median(ratio)), pd=float(hit.mean()),
                snr_o=float(snr_out(np.median(ratio))),
                snr_o_db=float(snr_out(np.median(ratio))))
