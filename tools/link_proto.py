#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""发射端 / 接收端共用的「协议 + 算法」层（不含任何 UI 代码）。

上面挂两个 GUI 脚本：
    tools/tx_app.py    发送端界面（不 import 本模块的内核部分也要 import 它）
    tools/rx_app.py    接收端界面（要用本模块的内核封装）

分层的理由
    UI 只负责「读参数、画图、报错」，所有与物理/协议有关的算术都在这里。
    于是两个界面可以各自独立启动、单独测试，且与命令行版 tools/link_sim.py
    共用同一个数值内核（kernel_src.simulate_seq），不存在第二套积分器。

一、数据交互（UDP，默认 127.0.0.1:50500）
    三类包，统一 4 字节魔数 "WBDX" + 1 字节类型：

      H 头包    b"WBDX" + b"H" + UTF-8 JSON
                字段：v, frame, n, fs, w_hint, amp, sigma, alpha,
                      snr_db, has_sig, has_noise, duration, ts
      D 数据包  b"WBDX" + b"D" + <4sIIIf> + float32[chunk]
                <4sIIIf> = (魔数, frame, seq, nseq, fs)
      T 尾包    b"WBDX" + b"T" + UTF-8 JSON（{frame, sent}）

    一帧 = 1 个 H + ceil(n/CHUNK) 个 D + 1 个 T。CHUNK=1024 样本
    （float32 → 4117 字节/包），远小于单包上限，跨机也不分片。

    w_hint 是"联调旁路"：接收端可以选择直接用它当 R 对表，但默认不用
    ——正常流程里接收端只拿序列与 dt，靠扫 R 自己找 ω。

二、接收端的 n_cycles 怎么来
    内核需要「积分多少个载波周期」。接收端不知道 ω，所以不能预设 150。
    做法是从长度反推：h₁ = R·dt，n 个样本对应 n_cycles = n·R·dt/2π。
    于是 steps = round(n_cycles·2π/h₁) 恰好等于 n，整段序列一点不浪费，
    任意 R 都能算——这是扫 R 与手动指定 R 都能工作的前提。

三、噪声口径（必须与阶段 2–5 一致）
    每样本标准差 = σ·√h₁，h₁ 取**共振时**的值 ω·dt。这是"力口径"
    （每步叠加 σ√h·N(0,1)），不是"连续白噪声 PSD 口径"（σ/√Δt）。
    两者差一个 h₁，会让虚警率从 85% 变成 0%，阶段 2 踩过这个坑。
"""
from __future__ import annotations

import json
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from kernel_src import load_kernel  # noqa: E402

# --------------------------------------------------------------------------- 常量
TWOPI = 2.0 * math.pi
SIM, NS = load_kernel()
LAM0 = float(NS["LAM0"])
BETAS3 = np.array([0.0, -2.0 * math.pi / 3.0, 2.0 * math.pi / 3.0])
BETAS4 = np.array([0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0])

MAGIC = b"WBDX"
DATA_HDR = struct.Struct("<4scIIIf")    # magic, 类型'bD', frame, seq, nseq, fs
CHUNK = 1024                            # 每包样本数
PORT_DEFAULT = 50500
PROTO_V = 1

PSTD_TH = 0.08          # 判态门限（阶段 2/3 口径：混沌 ~0.4，周期 ~1e-5）


# --------------------------------------------------------------------------- 协议
def pack_head(meta: dict) -> bytes:
    """H 头包：一帧的元信息（JSON 便于人眼直接 tcpdump 看）。"""
    meta = dict(meta, v=PROTO_V, kind="head")
    return MAGIC + b"H" + json.dumps(meta, ensure_ascii=False).encode("utf-8")


def pack_data(frame: int, seq: int, nseq: int, fs: float, samples) -> bytes:
    """D 数据包：固定头 + float32 小端样本。"""
    s = np.asarray(samples, dtype=np.float32).ravel()
    return DATA_HDR.pack(MAGIC, b"D", frame, seq, nseq, float(fs)) + s.tobytes()


def pack_tail(meta: dict) -> bytes:
    meta = dict(meta, v=PROTO_V, kind="tail")
    return MAGIC + b"T" + json.dumps(meta, ensure_ascii=False).encode("utf-8")


def parse_packet(buf: bytes):
    """返回 (kind, payload)。kind ∈ {"head", "data", "tail"}；非法包返回 (None, None)。"""
    if buf is None or len(buf) < 5 or buf[:4] != MAGIC:
        return None, None
    kind = buf[4:5]
    if kind == b"D":
        if len(buf) < DATA_HDR.size:
            return None, None
        _, _, frame, seq, nseq, fs = DATA_HDR.unpack_from(buf, 0)
        x = np.frombuffer(buf, dtype=np.float32, offset=DATA_HDR.size)
        return "data", dict(frame=int(frame), seq=int(seq), nseq=int(nseq),
                            fs=float(fs), x=x.astype(np.float64))
    if kind in (b"H", b"T"):
        try:
            obj = json.loads(buf[5:].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None, None
        return ("head" if kind == b"H" else "tail"), obj
    return None, None


def split_frame(frame: int, x: np.ndarray, fs: float):
    """把一帧切成若干 D 包（生成器），供发送线程逐包发。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    nseq = int(math.ceil(n / CHUNK))
    for seq in range(nseq):
        seg = x[seq * CHUNK:(seq + 1) * CHUNK]
        yield pack_data(frame, seq, nseq, fs, seg)


# --------------------------------------------------------------------------- 帧装配
class FrameBuf:
    """一帧的装配缓冲：按 seq 把数据包填进定长数组（接收端用）。

    放在协议层而不是 UI 层，是因为它只跟包和数组打交道：
    测试可以直接喂包，不必启动 tkinter。
    """

    def __init__(self, meta: dict):
        self.meta = meta
        self.n = int(meta.get("n") or 0)
        self.fs = float(meta.get("fs") or 0.0)
        self.buf = np.zeros(max(self.n, 1))
        self.seqs: set[int] = set()
        self.got = 0
        self.t0 = time.time()
        self.t_last = self.t0

    def put(self, seq: int, x: np.ndarray, nseq: int = 0) -> None:
        """写入第 seq 个包；重复包直接丢弃（UDP 可能重传或回环重复）。"""
        if seq in self.seqs:
            return
        lo = seq * CHUNK
        hi = min(lo + np.asarray(x).size, self.n)
        if lo >= self.n:
            return
        self.buf[lo:hi] = np.asarray(x, float)[:hi - lo]
        self.seqs.add(seq)
        self.got += hi - lo
        self.t_last = time.time()

    @property
    def complete(self) -> bool:
        return self.n > 0 and self.got >= self.n

    @property
    def missing(self) -> int:
        return max(self.n - self.got, 0)


# --------------------------------------------------------------------------- 信号与噪声
def snr_of_amp(amp: float, sigma: float) -> float:
    """SNR_E = 10lg[(A²/2)/σ²]，与阶段 1/4 的整段能量比口径一致。"""
    if sigma <= 0:
        return float("inf")
    return 10.0 * math.log10(max(amp ** 2 / 2.0, 1e-300) / sigma ** 2)


def amp_of_snr(snr_db: float, sigma: float) -> float:
    return math.sqrt(2.0 * sigma ** 2 * 10.0 ** (snr_db / 10.0))


def make_noise(n: int, k: int, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """(n, k) 单位方差噪声；alpha=0 白噪，alpha>0 为 1/f^alpha 色噪（去 DC）。"""
    w = rng.standard_normal((n, k))
    if alpha == 0.0:
        return w
    W = np.fft.rfft(w, axis=0)
    f = np.fft.rfftfreq(n, d=1.0)
    scale = np.zeros_like(f)
    nz = f > 0
    scale[nz] = f[nz] ** (-alpha / 2.0)
    scale[0] = 0.0
    x = np.fft.irfft(W * scale[:, None], n=n, axis=0)
    return x / (x.std(axis=0) + 1e-12)


def make_frame(*, w: float, fs: float, duration: float, amp: float,
               sigma: float, alpha: float = 0.0,
               has_sig: bool = True, has_noise: bool = True,
               phase=None, atten_db: float = 0.0, adc_bits: int = 0,
               rng: np.random.Generator | None = None) -> dict:
    """生成一帧：发射序列 tx 与接收序列 rx（含信道衰减 + 噪声 + 可选 ADC）。

    返回 dict(tx, rx, phase, snr_db, n, fs, h1)。
    """
    rng = np.random.default_rng() if rng is None else rng
    n = max(64, int(round(duration * fs)))
    t = np.arange(n, dtype=np.float64) / fs
    ph = float(rng.uniform(-math.pi, math.pi)) if phase is None else float(phase)

    tx = amp * np.cos(w * t + ph) if has_sig else np.zeros(n)
    if not has_sig:
        tx = np.zeros(n)                    # 关信号时相位无意义，置 0 便于对照

    rx = tx * 10.0 ** (-atten_db / 20.0)
    h1 = w / fs                             # 共振时的数值步长（噪声口径用）
    if has_noise:
        rx = rx + sigma * math.sqrt(h1) * make_noise(n, 1, alpha, rng)[:, 0]
    if adc_bits:
        fsr = 4.0 * max(float(np.abs(rx).max()), 1e-9)
        lsb = 2.0 * fsr / (2 ** adc_bits)
        rx = np.round(rx / lsb) * lsb

    return dict(tx=tx, rx=rx, phase=ph, n=n, fs=fs, h1=h1,
                snr_db=(snr_of_amp(amp, sigma) if (has_sig and has_noise) else None))


# --------------------------------------------------------------------------- 内核封装
def _cycles(n: int, R: float, dt: float) -> float:
    """由「样本数 + 尺度系数 R + 采样间隔」反推积分周期数，使 steps 恰为 n。"""
    return max(n * R * dt / TWOPI, 1e-6)


def pstd_one(x: np.ndarray, R: float, dt: float, lam: float = LAM0,
             beta: float = 0.0, burn: float = 0.5) -> float:
    """单振子（标量快路径）的 pstd。"""
    n = np.asarray(x).size
    r = SIM(np.asarray(x, float).reshape(-1, 1), R, dt, lam=float(lam),
            beta=float(beta), n_cycles=_cycles(n, R, dt), burn_frac=burn)
    return float(r["pstd"][0])


def pstd_array(x: np.ndarray, R: float, dt: float, lam: float = LAM0,
               betas=BETAS3, burn: float = 0.5, trace: bool = False) -> dict:
    """阵列并行算 pstd：同一段序列喂给 len(betas) 个振子。

    trace=True 时额外返回 t/X/Y（用于相图）。m 路一次算完，比逐路快约 7 倍。
    """
    x = np.asarray(x, float).ravel()
    b = np.atleast_1d(np.asarray(betas, float))
    return SIM(x.reshape(-1, 1), R, dt, lam=np.full(b.size, float(lam)),
               beta=b, n_cycles=_cycles(x.size, R, dt), burn_frac=burn,
               trace=trace)


def fft_guess(x: np.ndarray, fs: float, band=(0.3, 5.0)) -> float:
    """接收端在「目标特征频段」内粗估角频率：Hann 窗 + 8× 补零 + 对数抛物线。

    接收端不知道 ω，只用绝对频段（默认 0.3–5 Hz）。误差约 0.002%，
    远小于频率选择带宽 ±0.1%，够当细扫的初值。
    """
    x = np.asarray(x, float).ravel()
    x = x - x.mean()
    if x.size < 16 or not np.any(x):
        return float("nan")
    nfft = 1 << int(math.ceil(math.log2(x.size * 8)))
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size), nfft))
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    m = (freqs >= band[0]) & (freqs <= band[1])
    if not m.any():
        return float("nan")
    k = int(np.argmax(np.where(m, spec, 0.0)))
    d = 0.0
    if 0 < k < spec.size - 1:
        a, b, c = (math.log(max(v, 1e-30)) for v in (spec[k - 1], spec[k], spec[k + 1]))
        den = a - 2 * b + c
        if abs(den) > 1e-12:
            d = float(np.clip(0.5 * (a - c) / den, -1.0, 1.0))
    return TWOPI * (k + d) * fs / nfft


def scan_R(x: np.ndarray, dt: float, w0: float, lam: float = LAM0,
           rel: float = 0.01, coarse: float = 0.005, fine: float = 0.0005,
           zoom: int = 5) -> dict:
    """在 w0 附近两级细扫 R，返回 pstd 最小的 R̂。

    一级 ±rel、步长 coarse；二级在一级最优点 ±zoom·fine 内、步长 fine；
    最后三点抛物线插值。共 1+2·2 + 1+2·zoom = 16 次单振子，约 1.1 s。

    必须单路做（β=0）：三路取 min 会把别的振子在锁频态上的伪相变捞进来，
    把带宽边界从 0.20% 糊成 0.55%（阶段 2 实测）。
    """
    half = int(round(rel / coarse))
    g1 = w0 * (1.0 + np.arange(-half, half + 1) * coarse)
    v1 = np.array([pstd_one(x, float(R), dt, lam) for R in g1])
    R1 = float(g1[int(np.argmin(v1))])

    g2 = R1 * (1.0 + np.arange(-zoom, zoom + 1) * fine)
    v2 = np.array([pstd_one(x, float(R), dt, lam) for R in g2])
    k = int(np.argmin(v2))
    d = 0.0
    if 0 < k < v2.size - 1:
        y = np.log10(np.maximum(v2, 1e-15))
        den = y[k - 1] - 2 * y[k] + y[k + 1]
        if abs(den) > 1e-12:
            d = float(np.clip(0.5 * (y[k - 1] - y[k + 1]) / den, -1.0, 1.0))
    R_hat = float(g2[k] + d * (g2[1] - g2[0]))
    return dict(grid=g2, pstd=v2, R_hat=R_hat, pstd_min=float(v2[k]),
                w0=w0, coarse_grid=g1, coarse_pstd=v1)


def estimate_amp(x: np.ndarray, R: float, dt: float,
                 lam_grid: np.ndarray | None = None,
                 betas=BETAS4, lam_c: float = 0.7335) -> dict:
    """λ* 差分估计器：四个参考相位各扫到相变点，再作差消去 λc。

        γ̂cos θ̂ = (λ*₁₈₀ − λ*₀)/2,   γ̂sin θ̂ = (λ*₂₇₀ − λ*₉₀)/2

    好处是不必知道 λc 的绝对值（它会随平台/字长漂），差值把它消掉。
    """
    if lam_grid is None:
        lam_grid = np.arange(0.680, 0.7901, 0.0025)
    b = np.atleast_1d(np.asarray(betas, float))
    x = np.asarray(x, float).ravel()
    m = b.size * lam_grid.size
    lams = np.tile(lam_grid, b.size)
    sig = np.repeat(x.reshape(-1, 1), m, axis=1)
    p = SIM(sig, R, dt, lam=lams, beta=np.repeat(b, lam_grid.size),
            n_cycles=_cycles(x.size, R, dt), burn_frac=0.5)["pstd"]
    p = p.reshape(b.size, lam_grid.size)

    lam_star = []
    for i in range(b.size):
        hit = np.where(p[i] < PSTD_TH)[0]
        lam_star.append(float(lam_grid[hit[0]]) if hit.size else float(lam_grid[-1]))
    gc = (lam_star[2] - lam_star[0]) / 2.0
    gs = (lam_star[3] - lam_star[1]) / 2.0
    return dict(gamma=math.hypot(gc, gs), theta=math.atan2(gs, gc),
                lam_star=lam_star, lam_c_assumed=lam_c)


def analyze(x: np.ndarray, dt: float, *, mode: str = "auto", R_manual: float = 10.0,
            w_hint: float | None = None, lam: float = LAM0, betas=BETAS3,
            th: float = PSTD_TH, do_est: bool = False, band=(0.3, 5.0),
            trace: bool = True) -> dict:
    """接收机的完整判决流程（纯计算，供 UI 线程调用）。

    mode = "known"  已知 R，直接算（默认，最快，不用扫）
           "auto"   FFT 粗估 → 两级细扫 R（盲测，论文流程）
           "hint"   用发送端包头里的 w_hint（联调时省去手填）
    """
    x = np.asarray(x, float).ravel()
    fs = 1.0 / dt
    out = dict(n=x.size, dt=dt, mode=mode, lam=lam, th=th, x=x,
               betas=np.asarray(betas, float))
    sc = None
    if not np.any(x):
        out.update(ok=False, msg="全零序列（信号与噪声都关了）", detected=False,
                   R=float("nan"), w0=float("nan"), pstd_all=np.full(len(betas), np.nan),
                   pstd=float("nan"), peak=0.0, spectrum=None, freq=None)
        return out

    if mode == "auto":
        w0 = fft_guess(x, fs, band)
        if not math.isfinite(w0):
            out.update(ok=False, msg="频段内没有可辨的谱峰", detected=False,
                       R=float("nan"), w0=float("nan"),
                       pstd_all=np.full(len(betas), np.nan), pstd=float("nan"),
                       peak=0.0, spectrum=None, freq=None)
            return out
        sc = scan_R(x, dt, w0, lam)
        R = sc["R_hat"]
    elif mode == "hint":
        R = float(w_hint or 0.0)
        w0 = R
        if R <= 0:
            out.update(ok=False, msg="没有收到发送端的频率提示", detected=False,
                       R=float("nan"), w0=float("nan"),
                       pstd_all=np.full(len(betas), np.nan), pstd=float("nan"),
                       peak=0.0, spectrum=None, freq=None)
            return out
    elif mode in ("manual", "known"):       # known = 已知 R，不扫；manual 是旧名
        R = float(R_manual)
        w0 = float("nan")
    else:
        raise ValueError(f"未知的 R 模式：{mode}")

    arr = pstd_array(x, R, dt, lam=lam, betas=betas, trace=trace)
    pstd_all = np.asarray(arr["pstd"], float)

    # 频谱（画图用）
    nfft = 1 << int(math.ceil(math.log2(max(x.size, 64) * 4)))
    spec = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(x.size), nfft))
    freq = np.fft.rfftfreq(nfft, dt)

    out.update(ok=True, msg="", R=float(R), w0=float(w0),
               pstd_all=pstd_all, pstd=float(np.nanmin(pstd_all)),
               detected=bool(np.nanmin(pstd_all) < th),
               best_ch=int(np.nanargmin(pstd_all)),
               trace=(arr["t"], arr["X"], arr["Y"]) if trace else None,
               scan=sc, spectrum=spec, freq=freq,
               peak=float(x.max() - x.min()))

    # 只有判出「有」才估幅：无信号时 λ* 全落在网格端点，估出来的 γ/θ 没有意义
    if do_est and out["detected"]:
        out["est"] = estimate_amp(x, R, dt)
    return out
