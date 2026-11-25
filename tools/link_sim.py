#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""端到端链路仿真：一端发射、一端接收并判定「目标是否存在」。

链路
    发射端 TX   载波 A·cos(ωt+φ)（目标特征磁场；A、φ 对接收端都未知）
      │  衰减（近场磁偶极 1/r³）+ 传播相位
      ▼
    信道        + 环境磁噪声（白噪 / 1/f^α 色噪）+ ADC 量化
      ▼
    接收端 RX   FFT 粗估频率 → 变尺度细扫 R 定 ω → 三相位阵列 pstd 判态
                → 「有目标 / 无目标」 → λ* 差分估计幅值与初相

接收端为什么是"盲"的
    它只拿到采样序列 sig[i] 与采样间隔 dt，**不知道 ω**。变尺度系数 R 由序列
    自身扫出来：R=ω 时序列角频率在新时间轴上恰好等于内驱的 1，发生共振，
    pstd 从混沌的 ~0.4 塌到 1e-5 量级。整条接收链只调用 simulate_seq()，
    与阶段 2–5 同一个内核、同一段 RK4。

两个必须知道的物理事实（本脚本会实测出来）
    1. 单振子存在**相位盲区**：相变由合成幅值决定，λ0+A·cosθ 必须越过 λc。
       θ 随机时检出率被"相位覆盖率"封顶（λ0=0.72、λc=0.7335、A=0.02 → 约 26%），
       与噪声无关。所以接收机默认用三相位阵列（β=0, ±120°，取 min pstd）。
    2. 频率选择带宽只有 ±0.1%，必须 FFT 给初值再在小窗内细扫；全域穷举会漏。

运行
    .venv\\Scripts\\python.exe tools\\link_sim.py            # 全量（约 2 min）
    .venv\\Scripts\\python.exe tools\\link_sim.py --quick    # 冒烟（约 40 s）

输出
    复现输出/端到端-*.png（4 张）
    复现输出/端到端链路-发射接收检测汇总.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "复现输出"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kernel_src import load_kernel  # noqa: E402

TWOPI = 2.0 * math.pi
SIM, NS = load_kernel()
PSTD_TH = 0.08                      # 判态门限（阶段 2/3 口径）
BETAS3 = np.array([0.0, -2.0 * math.pi / 3.0, 2.0 * math.pi / 3.0])
BETAS4 = np.array([0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0])


# --------------------------------------------------------------------------- 配置
@dataclass
class LinkCfg:
    """一条链路的全部参数。幅值用归一化策动力单位（与 λ 同量纲）。"""

    w: float = 10.0          # 载波角频率 rad/s（真值，只有发射端知道）
    fs: float = 500.0        # 接收端采样率 Hz
    n_cycles: float = 150.0  # 观测窗 = 多少个载波周期
    amp: float = 0.02        # 发射幅值 A
    sigma: float = 0.05      # 噪声强度（力口径：每步标准差 = σ·√h₁）
    alpha: float = 0.0       # 0 白噪；1 为 1/f 色噪
    lam0: float = NS["LAM0"]
    atten_db: float = 0.0    # 信道衰减
    adc_bits: int = 0        # 0 = 理想；否则按满量程量化
    pstd_th: float = PSTD_TH
    f_band: tuple = (0.5, 1.5)   # 接收端已知的"目标特征频段"（相对 f0 的倍数）

    @property
    def dt(self) -> float:
        return 1.0 / self.fs

    @property
    def f0(self) -> float:
        return self.w / TWOPI

    @property
    def h1(self) -> float:
        """数值步长 h₁ = R·dt；共振时 R=ω。也是噪声口径里的步长。"""
        return self.w * self.dt

    @property
    def n(self) -> int:
        return int(round(self.n_cycles * TWOPI / self.h1))

    @property
    def duration(self) -> float:
        return self.n * self.dt

    def snr_db(self, amp: float | None = None) -> float:
        """SNR_E = 10lg[(A²/2)/σ²]，阶段 1/4 的整段能量比口径。"""
        a = self.amp if amp is None else amp
        return 10.0 * math.log10((a ** 2 / 2.0) / self.sigma ** 2)

    def amp_of_snr(self, snr_db: float) -> float:
        return math.sqrt(2.0 * self.sigma ** 2 * 10.0 ** (snr_db / 10.0))


# --------------------------------------------------------------------------- 噪声
def make_noise(n: int, k: int, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """(n, k) 单位方差噪声，逐列独立；alpha=0 白噪，否则 1/f^alpha（与阶段 4 同）。"""
    w = rng.standard_normal((n, k))
    if alpha == 0.0:
        return w
    W = np.fft.rfft(w, axis=0)
    f = np.fft.rfftfreq(n, d=1.0)
    scale = np.zeros_like(f)
    nz = f > 0
    scale[nz] = f[nz] ** (-alpha / 2.0)
    scale[0] = 0.0                                  # 去 DC，避免随机游走
    x = np.fft.irfft(W * scale[:, None], n=n, axis=0)
    return x / (x.std(axis=0) + 1e-12)


# --------------------------------------------------------------------------- 发射端
def tx_emit(cfg: LinkCfg, phases, amp: float | None = None) -> np.ndarray:
    """发射端：(n, k) 的目标特征磁场，每列一个独立目标（相位可不同）。"""
    a = cfg.amp if amp is None else amp
    ph = np.atleast_1d(np.asarray(phases, float))
    t = np.arange(cfg.n) * cfg.dt
    return a * np.cos(cfg.w * t[:, None] + ph[None, :])


# --------------------------------------------------------------------------- 信道
def channel(cfg: LinkCfg, tx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """衰减 + 加性噪声 + ADC 量化，返回接收端拿到的采样序列 (n, k)。"""
    rx = tx * 10.0 ** (-cfg.atten_db / 20.0)
    k = rx.shape[1]
    rx = rx + cfg.sigma * math.sqrt(cfg.h1) * make_noise(cfg.n, k, cfg.alpha, rng)
    if cfg.adc_bits:
        fsr = 4.0 * max(float(np.abs(rx).max()), 1e-9)   # ±2 倍峰值满量程
        lsb = 2.0 * fsr / (2 ** cfg.adc_bits)
        rx = np.round(rx / lsb) * lsb
    return rx


# --------------------------------------------------------------------------- 接收端
def fft_guess(cfg: LinkCfg, rx: np.ndarray) -> float:
    """从一段接收序列粗估角频率 ω̂（Hann 窗 + 8× 补零 + 对数抛物线插值）。

    只在"目标特征频段"内找峰：检测系统知道目标大概在哪个频段，但不知道确切频率。
    """
    x = np.asarray(rx, float).ravel()
    x = x - x.mean()
    nfft = 1 << int(math.ceil(math.log2(x.size * 8)))
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size), nfft))
    freqs = np.fft.rfftfreq(nfft, 1.0 / cfg.fs)
    lo, hi = cfg.f_band[0] * cfg.f0, cfg.f_band[1] * cfg.f0
    band = (freqs >= lo) & (freqs <= hi)
    k = int(np.argmax(np.where(band, spec, 0.0)))
    d = 0.0
    if 0 < k < spec.size - 1:
        a, b, c = (math.log(max(v, 1e-30)) for v in (spec[k - 1], spec[k], spec[k + 1]))
        den = a - 2 * b + c
        if abs(den) > 1e-12:
            d = float(np.clip(0.5 * (a - c) / den, -1.0, 1.0))
    return TWOPI * (k + d) * cfg.fs / nfft


def pstd_batch(cfg: LinkCfg, sig: np.ndarray, R: float,
               betas: np.ndarray | None = None,
               lams: np.ndarray | None = None) -> np.ndarray:
    """把 (n, m) 段序列一次喂给内核，返回 m 个 pstd（阵列并行，比逐路快 10 倍）。

    betas/lams 可以逐列不同——估计器靠"每列一个 λ、每行一个 β"的二维网格
    一次算出整张相变图，不用循环。
    """
    m = sig.shape[1]
    beta = np.zeros(m) if betas is None else np.asarray(betas, float)
    if beta.size == 1:
        beta = np.full(m, float(beta[0]))
    lam = np.full(m, cfg.lam0) if lams is None else np.asarray(lams, float)
    return SIM(sig, R, cfg.dt, lam=lam, beta=beta,
               n_cycles=cfg.n_cycles)["pstd"]


def array_signals(rx: np.ndarray) -> np.ndarray:
    """同一物理场分给 3 个相位振子：列序 [trial0×3, trial1×3, ...]。"""
    k = rx.shape[1]
    return np.repeat(rx, BETAS3.size, axis=1)


def array_betas(k: int) -> np.ndarray:
    return np.tile(BETAS3, k)


def scan_scale(cfg: LinkCfg, rx: np.ndarray, w0: float,
               rel: float = 0.01, step: float = 0.0005) -> dict:
    """在 w0 的 ±rel 内以 step 相对步长细扫 R，取 pstd 最小点。

    注意：**带宽测量必须单路做**（β=0）。三路取 min 会把别的振子在锁频态上的
    伪相变捞进来，把带宽边界弄糊（阶段 2 实测 0.20% → 0.55%）。
    """
    half = int(round(rel / step))
    grid = w0 * (1.0 + np.arange(-half, half + 1) * step)
    vals = np.array([float(pstd_batch(cfg, rx[:, :1], R)[0]) for R in grid])
    k = int(np.argmin(vals))
    d = 0.0
    if 0 < k < vals.size - 1:
        y = np.log10(np.maximum(vals, 1e-12))
        den = y[k - 1] - 2 * y[k] + y[k + 1]
        if abs(den) > 1e-12:
            d = float(np.clip(0.5 * (y[k - 1] - y[k + 1]) / den, -1.0, 1.0))
    R_hat = float(grid[k] + d * (grid[1] - grid[0]))
    return dict(grid=grid, pstd=vals, R_hat=R_hat,
                pstd_min=float(vals[k]), w0=w0)


def estimate_amp(cfg: LinkCfg, rx: np.ndarray, R: float,
                 lam_grid: np.ndarray | None = None) -> dict:
    """λ* 差分估计器：四个参考相位各扫到相变点，再作差消去 λc。

        γ̂cos θ̂ = (λ*₁₈₀ − λ*₀)/2,   γ̂sin θ̂ = (λ*₂₇₀ − λ*₉₀)/2

    λc 未知且随平台漂移，差值把它消掉——这是相对"测绝对阈值"的关键好处。
    """
    if lam_grid is None:
        lam_grid = np.arange(0.680, 0.7901, 0.0025)
    m = BETAS4.size * lam_grid.size
    betas = np.repeat(BETAS4, lam_grid.size)
    lams = np.tile(lam_grid, BETAS4.size)
    sig = np.repeat(rx[:, :1], m, axis=1)
    p = pstd_batch(cfg, sig, R, betas, lams).reshape(BETAS4.size, lam_grid.size)
    lam_star = []
    for b in range(BETAS4.size):
        idx = np.where(p[b] < cfg.pstd_th)[0]
        lam_star.append(float(lam_grid[idx[0]]) if idx.size else float(lam_grid[-1]))
    gc = (lam_star[2] - lam_star[0]) / 2.0
    gs = (lam_star[3] - lam_star[1]) / 2.0
    return dict(gamma=math.hypot(gc, gs), theta=math.atan2(gs, gc),
                lam_star=lam_star)


def full_receive(cfg: LinkCfg, rx: np.ndarray) -> dict:
    """完整接收机：FFT 粗估 → 细扫定频 → 三相位阵列判态。"""
    w0 = fft_guess(cfg, rx)
    sc = scan_scale(cfg, rx, w0)
    w_hat = sc["R_hat"]
    p = pstd_batch(cfg, array_signals(rx), w_hat, array_betas(rx.shape[1]))
    return dict(w0=w0, w_hat=w_hat, f_hat=w_hat / TWOPI,
                err_pct=(w_hat - cfg.w) / cfg.w * 100.0,
                pstd_ch=p, pstd=float(p.min()),
                detected=bool(p.min() < cfg.pstd_th),
                scan=sc, best_ch=int(np.argmin(p)))


# --------------------------------------------------------------------------- 实验：单次演示
def exp_demo(cfg: LinkCfg, snr_db: float) -> dict:
    """有目标 / 无目标各接收一次，走完整链路。"""
    rng = np.random.default_rng(20261125)
    amp = cfg.amp_of_snr(snr_db)
    phase = float(rng.uniform(-math.pi, math.pi))

    tx = tx_emit(cfg, [phase], amp)
    rx_on = channel(cfg, tx, rng)
    rx_off = channel(cfg, np.zeros_like(tx), rng)

    on = full_receive(cfg, rx_on)
    off = full_receive(cfg, rx_off)
    est = estimate_amp(cfg, rx_on, on["w_hat"])

    # ADC 位数敏感性：同一段接收数据换位数
    adc = {}
    for bits in (8, 12, 16):
        cfg2 = LinkCfg(**{**cfg.__dict__, "adc_bits": bits})
        rxq = channel(cfg2, tx, np.random.default_rng(1))
        adc[bits] = float(pstd_batch(cfg2, rxq, cfg.w)[0])

    return dict(cfg=cfg, snr_db=snr_db, amp=amp, phase=phase,
                tx=tx[:, 0], rx_on=rx_on[:, 0], rx_off=rx_off[:, 0],
                on=on, off=off, est=est,
                est_err=abs(est["gamma"] - amp) / amp * 100.0, adc=adc)


# --------------------------------------------------------------------------- 实验：Pd / Pfa
def batch_case(cfg: LinkCfg, rng: np.random.Generator, k: int,
               amp: float | None = None) -> np.ndarray:
    """一批 k 次试验，返回 (k, 3) 的三相位 pstd。amp=None 即 H0（无目标）。"""
    if amp is None:
        rx = cfg.sigma * math.sqrt(cfg.h1) * make_noise(cfg.n, k, cfg.alpha, rng)
    else:
        phases = rng.uniform(-math.pi, math.pi, k)
        rx = channel(cfg, tx_emit(cfg, phases, amp), rng)
    p = pstd_batch(cfg, array_signals(rx), cfg.w, array_betas(k))
    return p.reshape(k, BETAS3.size)


def phase_coverage(cfg: LinkCfg, amp: float, n_osc: int = 1) -> float:
    """解析相位覆盖率：总驱动幅值越过 λc 的初相占比。

    共振时信号与内驱同频，合成幅值 λ_eff = √(λ0² + A² + 2λ0A·cosθ)。
    单振子要求 λ_eff > λc，对应 |θ| < θ_h；三相位阵列的 β=0, ±120° 各覆盖
    一段 ±θ_h，取并集。Pd 的上限就是这个数——与噪声无关。
    """
    lam_c = 0.7335
    c = (lam_c ** 2 - cfg.lam0 ** 2 - amp ** 2) / (2.0 * cfg.lam0 * amp)
    if c <= -1.0:
        return 1.0
    if c >= 1.0:
        return 0.0
    half = math.acos(c)
    return float(min(1.0, n_osc * half / math.pi))


def exp_sweep(cfg: LinkCfg, snr_grid, n_cal=200, n_val=200, n_trial=100) -> dict:
    """Pd / Pfa vs SNR：固定 σ 扫 A（阶段 4 的正确做法）。

    门限只用 H0 标定（n_cal 次），再用**另一批** H0（n_val 次）验证 Pfa——
    同一批既标定又验证会把 Pfa 系统性低估（阶段 4 已记为隐患）。
    由于 σ 在整个扫描中不变，H0 分布与 A 无关，门限标定一次即可。
    """
    rng = np.random.default_rng(31415)

    p_cal = batch_case(cfg, rng, n_cal)
    thr_s = float(np.quantile(p_cal[:, 0], 0.05))
    thr_a = float(np.quantile(p_cal.min(axis=1), 0.05))
    p_val = batch_case(cfg, rng, n_val)
    pfa_s = float(np.mean(p_val[:, 0] < thr_s))
    pfa_a = float(np.mean(p_val.min(axis=1) < thr_a))
    se = math.sqrt(max(pfa_s * (1 - pfa_s), 1e-4) / n_val)
    print(f"  H0 标定（{n_cal} 次）：单振子门限={thr_s:.3f}  阵列门限={thr_a:.3f}")
    print(f"  H0 验证（{n_val} 次）：Pfa 单={pfa_s:.3f} 阵={pfa_a:.3f}"
          f"（二项标准差 ≈{se:.3f}）")

    rows = []
    for snr in snr_grid:
        amp = cfg.amp_of_snr(snr)
        p = batch_case(cfg, rng, n_trial, amp)
        pd_s = float(np.mean(p[:, 0] < thr_s))
        pd_a = float(np.mean(p.min(axis=1) < thr_a))
        rows.append(dict(snr_db=float(snr), amp=float(amp),
                         pd_single=pd_s, pd_array=pd_a,
                         cover_single=phase_coverage(cfg, amp, 1),
                         cover_array=phase_coverage(cfg, amp, 3),
                         n=n_trial))
        print(f"  SNR={snr:6.1f} dB  A={amp:.4f} | 单 Pd={pd_s:.2f}"
              f"（覆盖上限 {rows[-1]['cover_single']:.2f}）| 阵 Pd={pd_a:.2f}"
              f"（{rows[-1]['cover_array']:.2f}）")
    return dict(rows=rows, thr_single=thr_s, thr_array=thr_a,
                pfa_single=pfa_s, pfa_array=pfa_a, n_val=n_val, n_cal=n_cal)


# --------------------------------------------------------------------------- 实验：幅值估计
def exp_estimate(cfg: LinkCfg, snr_list, n_rep=5) -> dict:
    rng = np.random.default_rng(1618)
    rows = []
    for snr in snr_list:
        amp = cfg.amp_of_snr(snr)
        g_err, t_err, g_hat = [], [], []
        for _ in range(n_rep):
            ph = float(rng.uniform(-math.pi, math.pi))
            rx = channel(cfg, tx_emit(cfg, [ph], amp), rng)
            e = estimate_amp(cfg, rx, cfg.w)
            g_hat.append(e["gamma"])
            g_err.append(abs(e["gamma"] - amp) / amp * 100.0)
            dth = (e["theta"] - ph + math.pi) % TWOPI - math.pi
            t_err.append(abs(math.degrees(dth)))
        rows.append(dict(snr_db=float(snr), amp=float(amp), gamma_hat=g_hat,
                         gamma_rel_err_pct=float(np.mean(g_err)),
                         theta_err_deg=float(np.mean(t_err))))
        print(f"  SNR={snr:6.1f} dB  γ̂={np.mean(g_hat):.4f}（真值 {amp:.4f}）"
              f" 相对误差 {np.mean(g_err):.1f}%  θ 误差 {np.mean(t_err):.1f}°")
    return dict(rows=rows)


# --------------------------------------------------------------------------- 出图
def make_figures(demo: dict, sweep: dict, est: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 105
    plt.rcParams["savefig.dpi"] = 135
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.28
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    RED, BLUE, GREY, GREEN = "#c0392b", "#2c7fb8", "#7f8c8d", "#1e8449"

    cfg: LinkCfg = demo["cfg"]
    t = np.arange(cfg.n) * cfg.dt
    m = int(round(12.0 / cfg.f0 / cfg.dt))          # 12 个载波周期

    # ---- 图 1：发射 / 接收波形 + 频谱
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9), constrained_layout=True)
    axes[0].plot(t[:m], demo["tx"][:m], color=RED, lw=1.8)
    axes[0].set(title="(a) 发射端：目标特征磁场", xlabel="t / s", ylabel="归一化幅值")

    axes[1].plot(t[:m], demo["rx_on"][:m], color=BLUE, lw=0.7, alpha=0.55,
                 label="接收序列（含噪）")
    axes[1].plot(t[:m], demo["tx"][:m] * 10 ** (-cfg.atten_db / 20), color=RED,
                 lw=1.4, label="发射真值 × 衰减")
    axes[1].set(title=f"(b) 接收端：SNR={demo['snr_db']:.1f} dB", xlabel="t / s")
    axes[1].legend(fontsize=8)

    nfft = 1 << int(math.ceil(math.log2(cfg.n)))
    sp = np.abs(np.fft.rfft((demo["rx_on"] - demo["rx_on"].mean()) * np.hanning(cfg.n),
                            nfft))
    fr = np.fft.rfftfreq(nfft, cfg.dt)
    band = (fr > 0.4) & (fr < 3.2)
    axes[2].semilogy(fr[band], sp[band] / sp[band].max(), color=BLUE, lw=0.9)
    axes[2].axvline(demo["on"]["f_hat"], color=RED, ls="--", lw=1.3,
                    label=f"$\\hat f$={demo['on']['f_hat']:.4f} Hz")
    axes[2].axvline(cfg.f0, color=GREY, ls=":", lw=1.2, label=f"真值 {cfg.f0:.4f} Hz")
    axes[2].set(title="(c) 接收频谱（归一化）", xlabel="f / Hz", ylim=(1e-4, 1.2))
    axes[2].legend(fontsize=8)
    fig.savefig(OUT / "端到端-发射与接收.png", bbox_inches="tight")
    plt.close(fig)

    # ---- 图 2：阵列相图 + 单振子相图 + 细扫 R
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.1), constrained_layout=True)
    best = demo["on"]["best_ch"]
    beta_best = float(BETAS3[best % BETAS3.size])

    r = SIM(demo["rx_on"], demo["on"]["w_hat"], cfg.dt, lam=cfg.lam0,
            beta=beta_best, n_cycles=cfg.n_cycles, burn_frac=0.8, trace=True)
    axes[0].plot(r["X"], r["Y"], color=RED, lw=0.25, alpha=0.75)
    axes[0].set(title=f"(a) 有目标：阵列第 {best + 1} 路（β={math.degrees(beta_best):+.0f}°）\n"
                      f"pstd={demo['on']['pstd']:.1e} → 判「有」",
                xlabel="x", ylabel="y = dx/dτ")

    sc = demo["on"]["scan"]
    axes[1].semilogy((sc["grid"] / cfg.w - 1.0) * 100.0, sc["pstd"], "o-", ms=3,
                     lw=1.2,
                     color=RED)
    axes[1].axhline(cfg.pstd_th, color=GREY, ls="--", lw=1,
                    label=f"判态门限 {cfg.pstd_th}")
    axes[1].axvline(demo["on"]["err_pct"], color=BLUE, ls=":", lw=1.2,
                    label=f"$\\hat\\omega$ 偏差 {demo['on']['err_pct']:+.3f}%")
    axes[1].set(title="(b) 细扫 R：共振点唯一（单路）",
                xlabel="R 相对偏移 / %", ylabel="pstd")
    axes[1].legend(fontsize=8)

    r2 = SIM(demo["rx_off"], demo["off"]["w_hat"], cfg.dt, lam=cfg.lam0,
             n_cycles=cfg.n_cycles, burn_frac=0.8, trace=True)
    axes[2].plot(r2["X"], r2["Y"], color=BLUE, lw=0.25, alpha=0.75)
    axes[2].set(title=f"(c) 无目标：pstd={demo['off']['pstd']:.2f} → 判「无」",
                xlabel="x", ylabel="y = dx/dτ")
    fig.savefig(OUT / "端到端-相图与频率扫描.png", bbox_inches="tight")
    plt.close(fig)

    # ---- 图 3：Pd / Pfa
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.2), constrained_layout=True)
    x = [r["snr_db"] for r in sweep["rows"]]
    axes[0].plot(x, [r["cover_single"] for r in sweep["rows"]], "-", color=GREY,
                 lw=1.2, alpha=0.55, label="单振子相位覆盖估计（解析）")
    axes[0].plot(x, [r["pd_single"] for r in sweep["rows"]], "o", color=GREY, ms=5,
                 label="单振子实测 $P_d$")
    axes[0].plot(x, [r["cover_array"] for r in sweep["rows"]], "-", color=BLUE,
                 lw=1.2, alpha=0.55, label="三相位覆盖估计（解析）")
    axes[0].plot(x, [r["pd_array"] for r in sweep["rows"]], "s-", color=RED,
                 lw=2, ms=6, label="三相位阵列实测 $P_d$")
    axes[0].axhline(0.9, color=GREY, ls=":", lw=1)
    axes[0].set(xlabel="SNR (dB)", ylabel="$P_d$", ylim=(-0.05, 1.05),
                title="(a) 检测概率被相位覆盖率封顶（未计噪声助相变）")
    axes[0].legend(fontsize=8)

    pfa_s, pfa_a, nv = sweep["pfa_single"], sweep["pfa_array"], max(sweep["n_val"], 2)
    se = [math.sqrt(max(p * (1 - p), 1e-4) / nv) for p in (pfa_s, pfa_a)]
    axes[1].bar([0, 1], [pfa_s, pfa_a], width=0.45, color=[GREY, BLUE],
                yerr=se, capsize=5)
    axes[1].axhline(0.05, color=RED, ls="--", lw=1.4, label="标称 $P_{fa}$=0.05")
    axes[1].set(xticks=[0, 1], xticklabels=["单振子", "三相位阵列"],
                ylabel="$P_{fa}$", ylim=(0, max(0.2, pfa_s * 1.8, pfa_a * 1.8)),
                title=f"(b) 虚警率：H0 标定 + 独立验证（{nv} 次）")
    axes[1].legend(fontsize=9)
    fig.savefig(OUT / "端到端-Pd与Pfa.png", bbox_inches="tight")
    plt.close(fig)

    # ---- 图 4：幅值估计（误差条 + 估计值 vs 真值）
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2), constrained_layout=True)
    if est["rows"]:
        xs = [r["snr_db"] for r in est["rows"]]
        means, sds, amps, ghats = [], [], [], []
        for r in est["rows"]:
            errs = [abs(g - r["amp"]) / r["amp"] * 100.0 for g in r["gamma_hat"]]
            means.append(float(np.mean(errs)))
            sds.append(float(np.std(errs)))
            amps.append(r["amp"])
            ghats.append(float(np.mean(r["gamma_hat"])))

        axes[0].errorbar(xs, means, yerr=sds, fmt="o-", color=GREEN, lw=2, ms=6,
                         capsize=4)
        for xv, yv in zip(xs, means):
            axes[0].annotate(f"{yv:.1f}%", (xv, yv), textcoords="offset points",
                             xytext=(0, 10), ha="center", fontsize=8)
        axes[0].axhline(5, color=GREY, ls=":", lw=1, label="5% 参考线")
        axes[0].set(xlabel="SNR (dB)", ylabel="幅值相对误差 / %", ylim=(0, 8),
                    title="(a) λ* 差分估计器：幅值相对误差")
        axes[0].legend(fontsize=9)

        lim = [0, max(amps) * 1.25]
        axes[1].plot(lim, lim, "--", color=GREY, lw=1.2, label="理想 $\hat\gamma=A$")
        axes[1].plot(amps, ghats, "o", color=GREEN, ms=8)
        for a_, g_, s_ in zip(amps, ghats, xs):
            axes[1].annotate(f"{s_:.1f} dB", (a_, g_), textcoords="offset points",
                             xytext=(6, -12), fontsize=8)
        axes[1].set(xlim=lim, ylim=lim, xlabel="真值 A", ylabel="$\hat\gamma$",
                    title="(b) 估计值 vs 真值")
        axes[1].legend(fontsize=9, loc="upper left")
    fig.savefig(OUT / "端到端-幅值估计.png", bbox_inches="tight")
    plt.close(fig)

    for name in ("端到端-发射与接收", "端到端-相图与频率扫描",
                 "端到端-Pd与Pfa", "端到端-幅值估计"):
        print(f"[图] 复现输出/{name}.png")


# --------------------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description="端到端链路仿真：发射→信道→接收检测")
    ap.add_argument("--quick", action="store_true", help="冒烟模式（少试验）")
    args = ap.parse_args()

    t0 = time.time()
    cfg = LinkCfg()
    print("=" * 78)
    print("端到端链路仿真：一端发射，一端接收并判定目标是否存在")
    print("=" * 78)
    print(f"载波 f0={cfg.f0:.4f} Hz (ω={cfg.w} rad/s)   采样率 {cfg.fs:.0f} Hz   "
          f"步长 h₁=R·dt={cfg.h1:.3f}")
    print(f"观测窗 {cfg.n_cycles:.0f} 个载波周期 = {cfg.duration:.1f} s，{cfg.n} 点；"
          f"工作点 λ0={cfg.lam0}（λc≈0.7335），σ={cfg.sigma}，噪声 α={cfg.alpha}")
    print("链路预算（近场磁偶极 B∝1/r³，每 ×10 距离 −60 dB）："
          "r=1 m 0 dB | 10 m −60 dB | 100 m −120 dB（绝对标定需实测 B_ref）")

    print("\n[1/4] 单次演示：有目标 vs 无目标（完整接收链路）")
    demo = exp_demo(cfg, snr_db=-7.4)
    on, off = demo["on"], demo["off"]
    print(f"  有目标：ω̂={on['w_hat']:.4f}（真值 {cfg.w}，误差 {on['err_pct']:+.4f}%）"
          f" 三路 pstd={np.round(on['pstd_ch'], 5).tolist()} → "
          f"min={on['pstd']:.2e} {'检出' if on['detected'] else '漏检'}")
    print(f"  无目标：三路 pstd={np.round(off['pstd_ch'], 3).tolist()} → "
          f"min={off['pstd']:.3f} {'误报' if off['detected'] else '正确判无'}")
    print(f"  幅值估计：γ̂={demo['est']['gamma']:.4f}（真值 {demo['amp']:.4f}，"
          f"误差 {demo['est_err']:.1f}%），θ̂={math.degrees(demo['est']['theta']):.1f}°"
          f"（真值 {math.degrees(demo['phase']):.1f}°）")
    print(f"  ADC：8 bit pstd={demo['adc'][8]:.2e}  12 bit={demo['adc'][12]:.2e}  "
          f"16 bit={demo['adc'][16]:.2e}（位宽够就不影响判决）")

    n_cal, n_val, n_trial = (40, 40, 20) if args.quick else (200, 200, 100)
    snr_grid = ([-18.9, -13.5, -11.0, -7.4] if args.quick
                else [-23.0, -18.9, -17.0, -15.4, -13.5, -11.0, -7.4, -3.0])

    print(f"\n[2/4] 检测性能（H0 标定 {n_cal} 次 + H0 验证 {n_val} 次，"
          f"每个 SNR {n_trial} 次 H1）")
    sweep = exp_sweep(cfg, snr_grid, n_cal, n_val, n_trial)

    est_grid = [-11.0, -7.4, -3.0] if not args.quick else [-7.4]
    print(f"\n[3/4] 幅值/相位估计（每个 SNR {5 if not args.quick else 2} 次）")
    est = exp_estimate(cfg, est_grid, n_rep=5 if not args.quick else 2)

    print("\n[4/4] 出图 ...")
    make_figures(demo, sweep, est)

    def first_hit(key, target=0.9):
        for r in sweep["rows"]:
            if r[key] >= target:
                return r["snr_db"]
        return None

    summary = dict(
        link=dict(f_carrier_hz=round(cfg.f0, 6), omega_rad_s=cfg.w,
                  fs_hz=cfg.fs, dt_s=cfg.dt, h1=cfg.h1, n_samples=cfg.n,
                  duration_s=round(cfg.duration, 2), n_cycles=cfg.n_cycles,
                  lam0=cfg.lam0, lambda_c=0.7335, sigma=cfg.sigma, alpha=cfg.alpha,
                  snr_def="SNR_E = 10*log10((A^2/2)/sigma^2)",
                  bands="近场磁偶极 1/r^3：每 ×10 距离 −60 dB"),
        demo=dict(snr_db=demo["snr_db"], amp=round(demo["amp"], 5),
                  w_hat=round(on["w_hat"], 5),
                  freq_err_pct=round(on["err_pct"], 5),
                  pstd_ch_on=[float(v) for v in on["pstd_ch"]],
                  pstd_ch_off=[float(v) for v in off["pstd_ch"]],
                  detected_on=on["detected"], detected_off=off["detected"],
                  gamma_hat=round(demo["est"]["gamma"], 5),
                  gamma_true=round(demo["amp"], 5),
                  gamma_rel_err_pct=round(demo["est_err"], 2),
                  theta_hat_deg=round(math.degrees(demo["est"]["theta"]), 1),
                  theta_true_deg=round(math.degrees(demo["phase"]), 1),
                  adc_pstd={str(k): v for k, v in demo["adc"].items()}),
        sweep=dict(rows=[{k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in r.items()} for r in sweep["rows"]],
                   thr_single=round(sweep["thr_single"], 4),
                   thr_array=round(sweep["thr_array"], 4),
                   pfa_single=round(sweep["pfa_single"], 4),
                   pfa_array=round(sweep["pfa_array"], 4),
                   n_cal=sweep["n_cal"], n_val=sweep["n_val"],
                   min_snr_pd90_single=first_hit("pd_single"),
                   min_snr_pd90_array=first_hit("pd_array")),
        estimate=dict(rows=[{k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in r.items()} for r in est["rows"]]),
        budgets=dict(
            model="B = (mu0/4pi)*m/r^3，近场磁偶极",
            example="m=100 A·m²（N=2000, I=1 A, S=0.05 m²）：r=10 m ≈ 0.1 nT，"
                    "r=100 m ≈ 0.1 pT",
            caveat="归一化幅值 A 与绝对磁感应强度的换算系数 B_ref 需实测标定，"
                   "本仿真不作绝对灵敏度断言",
        ),
        notes=[
            "接收端只拿采样序列与 dt，不知道 ω；R 由序列自身扫出，R=ω 时共振",
            "门限由 H0 单独标定、另用 H0 验证集复核，不拿 H1 调阈值",
            "单振子随机初相下 Pd 被相位覆盖率封顶（λ0+A·cosθ 须越过 λc），与噪声无关",
            "三相位阵列（β=0, ±120°）取 min pstd（OR 融合），把相位盲区补上",
            "λ* 差分估计器不需要知道 λc，差值把平台相关偏置消掉",
            "无板级实测：全部为数值仿真；定点/FPGA 见阶段 5",
        ],
    )
    jp = OUT / "端到端链路-发射接收检测汇总.json"
    jp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("结论")
    print("=" * 78)
    print(f"· 单次演示（SNR={demo['snr_db']} dB）：有目标 min pstd={on['pstd']:.1e} 判「有」，"
          f"无目标 {off['pstd']:.2f} 判「无」；频率误差 {on['err_pct']:+.3f}%")
    print(f"· 三相位阵列最低可检测 SNR（Pd≥0.9）= "
          f"{summary['sweep']['min_snr_pd90_array']} dB；"
          f"单振子 = {summary['sweep']['min_snr_pd90_single']}")
    print(f"· 幅值估计："
          + "；".join(f"{r['snr_db']:.1f} dB → {r['gamma_rel_err_pct']:.1f}%"
                      for r in est["rows"]))
    print(f"· 用时 {time.time() - t0:.0f} s，JSON 见 {jp.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
