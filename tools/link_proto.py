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
                字段：v, frame, n, fs, scale, w_hint, amp, sigma, alpha,
                      snr_db, has_sig, has_noise, cov, t0, sig_t0, sig_dur,
                      sig_epoch, phase, duration, ts
      D 数据包  b"WBDX" + b"D" + <4sIIIf> + int16[chunk]
                <4sIIIf> = (魔数, frame, seq, nseq, fs)
      T 尾包    b"WBDX" + b"T" + UTF-8 JSON（{frame, sent, nseq}）

    一帧 = 1 个 H + ceil(n/CHUNK) 个 D + 1 个 T。CHUNK=1024 样本
    （int16 → 2069 字节/包），远小于单包上限，跨机也不分片。

    **样本用 int16 而不是 float32**（协议 v2）：定点数正好对应 ESP32 那边
    ADC 的 int16 计数，固件与上位机不必各写一套浮点序列化；带宽也减半
    （UART 回放 ×10 只要 ~97 kbaud）。标定系数 scale 放头包里：
    物理量 = int16 / scale，默认 scale=32768（满量程 ±1.0，1 LSB=3.05e-5，
    比 A=0.02 低 3 个量级，量化对判决无影响）。

    w_hint 是"联调旁路"：接收端可以选择直接用它当 R 对表，但默认不用
    ——正常流程里接收端只拿序列与 dt，靠扫 R 自己找 ω。固件按"已知 R"
    实现（用户要求），也不解析 w_hint。

二、连续发送（协议 v2 的行为变化）
    发送端启动后**一直发到手动停止**，帧号从 0 单调递增、没有总数。
    目标信号默认不出现在链路上，由操作员在界面点「附加目标信号」手动
    注入一段 [sig_t0, sig_t0+sig_dur) 的窗口，窗口按会话绝对时间定义
    （不是按帧对齐），相位以窗口起点为零点连续推进、跨帧不重置。

    于是每帧都有一个覆盖度 cov ∈ [0, 1]：
        cov = 1    整帧都在目标窗内 —— 判决为「有」才算检出
        cov = 0    整帧都在窗外     —— 判决为「有」就是虚警
        0 < cov < 1 窗口切在帧中间  —— 部分覆盖，能量不足，不计入统计
    接收端不信任这个布尔量，自己用 t0/sig_t0/sig_dur 重算覆盖度
    （window_coverage），避免"发送端算错、接收端跟着错"。

三、接收端的 n_cycles 怎么来
    内核需要「积分多少个载波周期」。接收端不知道 ω，所以不能预设 150。
    做法是从长度反推：h₁ = R·dt，n 个样本对应 n_cycles = n·R·dt/2π。
    于是 steps = round(n_cycles·2π/h₁) 恰好等于 n，整段序列一点不浪费，
    任意 R 都能算——这是扫 R 与手动指定 R 都能工作的前提。

四、噪声口径（必须与阶段 2–5 一致）
    每样本标准差 = σ·√h₁，h₁ 取**共振时**的值 ω·dt。这是"力口径"
    （每步叠加 σ√h·N(0,1)），不是"连续白噪声 PSD 口径"（σ/√Δt）。
    两者差一个 h₁，会让虚警率从 85% 变成 0%，阶段 2 踩过这个坑。
"""
from __future__ import annotations

import json
import math
import struct
import sys
import threading
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
PROTO_V = 2                             # v2: 样本 int16 + 连续发送 + 目标窗字段

SAMPLE_DTYPE = np.int16                 # 线上样本类型（对应 ESP32 ADC 的 int16）
ADC_SCALE = 32768.0                     # 满量程 ±1.0 → 1 LSB = 3.05e-5
ADC_MAX = 32767
ADC_MIN = -32768

PSTD_TH = 0.08          # 判态门限（阶段 2/3 口径：混沌 ~0.4，周期 ~1e-5）


# --------------------------------------------------------------------------- 协议
def pack_head(meta: dict) -> bytes:
    """H 头包：一帧的元信息（JSON 便于人眼直接 tcpdump 看）。"""
    meta = dict(meta, v=PROTO_V, kind="head")
    return MAGIC + b"H" + json.dumps(meta, ensure_ascii=False).encode("utf-8")


def to_int16(x, scale: float = ADC_SCALE) -> np.ndarray:
    """浮点序列 → int16 计数（饱和，不环绕）。"""
    a = np.asarray(x, dtype=np.float64).ravel() * float(scale)
    return np.clip(np.round(a), ADC_MIN, ADC_MAX).astype(SAMPLE_DTYPE)


def from_int16(q, scale: float = ADC_SCALE) -> np.ndarray:
    return np.asarray(q, dtype=np.float64).ravel() / float(scale)


def pack_data(frame: int, seq: int, nseq: int, fs: float,
              samples, scale: float = ADC_SCALE) -> bytes:
    """D 数据包：固定头 + int16 小端样本。"""
    return (DATA_HDR.pack(MAGIC, b"D", frame, seq, nseq, float(fs))
            + to_int16(samples, scale).tobytes())


def pack_tail(meta: dict) -> bytes:
    meta = dict(meta, v=PROTO_V, kind="tail")
    return MAGIC + b"T" + json.dumps(meta, ensure_ascii=False).encode("utf-8")


def parse_packet(buf: bytes):
    """返回 (kind, payload)。kind ∈ {"head", "data", "tail"}；非法包返回 (None, None)。

    data 包给三个字段：`x`（已按默认 scale 换算成浮点）、`raw`（原始 int16，
    头包 scale 与默认值不同时用它重算）、`fs`。
    """
    if buf is None or len(buf) < 5 or buf[:4] != MAGIC:
        return None, None
    kind = buf[4:5]
    if kind == b"D":
        if len(buf) < DATA_HDR.size:
            return None, None
        _, _, frame, seq, nseq, fs = DATA_HDR.unpack_from(buf, 0)
        q = np.frombuffer(buf, dtype=SAMPLE_DTYPE, offset=DATA_HDR.size)
        return "data", dict(frame=int(frame), seq=int(seq), nseq=int(nseq),
                            fs=float(fs), raw=q.copy(), x=from_int16(q))
    if kind in (b"H", b"T"):
        try:
            obj = json.loads(buf[5:].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None, None
        return ("head" if kind == b"H" else "tail"), obj
    return None, None


def split_frame(frame: int, x: np.ndarray, fs: float, scale: float = ADC_SCALE):
    """把一帧切成若干 D 包（生成器），供发送线程逐包发。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    nseq = int(math.ceil(n / CHUNK))
    for seq in range(nseq):
        seg = x[seq * CHUNK:(seq + 1) * CHUNK]
        yield pack_data(frame, seq, nseq, fs, seg, scale)


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
        self.scale = float(meta.get("scale") or ADC_SCALE)   # 头包给的标定系数
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

    def put_raw(self, seq: int, q) -> None:
        """写入 int16 计数包，按本帧头包的 scale 换算回物理量。"""
        self.put(seq, from_int16(q, self.scale))

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
               rng: np.random.Generator | None = None,
               t_start: float = 0.0, win=None, theta0: float = 0.0) -> dict:
    """生成一帧：发射序列 tx 与接收序列 rx（含信道衰减 + 噪声 + 可选 ADC）。

    win = (t_on, t_off) 给目标信号的时间窗（会话绝对时间，秒）：
        None       整帧有信号（旧行为，相位 phase 每帧独立随机）
        (a, b)     只在这段窗口内叠 A·cos(ω(t−t_on)+θ0)，窗口外是纯背景
    窗口模式下相位以 t_on 为零点连续推进，跨帧不重置——所以一帧的初相
    由 t_start 决定（θ_frame = ω(t_start−t_on)+θ0），不是随机抽的。

    返回 dict(tx, rx, phase, cov, sig_on, snr_db, n, fs, h1)，其中
    phase 是**帧起点**的内驱相位（诊断用），cov 是本帧信号覆盖率。
    """
    rng = np.random.default_rng() if rng is None else rng
    n = max(64, int(round(duration * fs)))
    t = np.arange(n, dtype=np.float64) / fs
    t_abs = t_start + t

    if not has_sig:
        gate = np.zeros(n)
        ph = 0.0                            # 关信号时相位无意义，置 0 便于对照
    elif win is None:
        gate = np.ones(n)
        ph = float(rng.uniform(-math.pi, math.pi)) if phase is None else float(phase)
    else:
        t_on, t_off = float(win[0]), float(win[1])
        gate = ((t_abs >= t_on) & (t_abs < t_off)).astype(np.float64)
        ph = float(phase) if phase is not None else w * (t_start - t_on) + theta0
        ph = math.atan2(math.sin(ph), math.cos(ph))      # 折到 [-π, π]

    tx = amp * np.cos(w * t + ph) * gate
    rx = tx * 10.0 ** (-atten_db / 20.0)
    h1 = w / fs                             # 共振时的数值步长（噪声口径用）
    if has_noise:
        rx = rx + sigma * math.sqrt(h1) * make_noise(n, 1, alpha, rng)[:, 0]
    if adc_bits:
        fsr = 4.0 * max(float(np.abs(rx).max()), 1e-9)
        lsb = 2.0 * fsr / (2 ** adc_bits)
        rx = np.round(rx / lsb) * lsb

    return dict(tx=tx, rx=rx, phase=ph, n=n, fs=fs, h1=h1,
                cov=float(gate.mean()), sig_on=bool(gate.size and gate[0] > 0),
                snr_db=(snr_of_amp(amp, sigma) if (has_sig and has_noise) else None))


# --------------------------------------------------------------------------- 假设检验框架演示
def coherent_proj(x: np.ndarray, fs: float, f0: float) -> float:
    """已知频率处的相干投影幅值：2·|mean(x·e^{−jωt})|。

    对纯正弦 `A·cos(ωt+θ)` 无偏地给出 A（与 θ 无关），这是它被选作
    "框架演示"统计量的原因——不牵涉动力学，纯线性、可解析。
    **它不是本课题的检测器**（检测判据是阶段 3 的相变），只用来演示
    「阈值只由 H0 标定」这条纪律。
    """
    x = np.asarray(x, float).ravel()
    n = x.size
    if n == 0:
        return float("nan")
    t = np.arange(n) / float(fs)
    w = 2.0 * math.pi * float(f0)
    return float(2.0 * math.hypot(np.mean(x * np.cos(w * t)),
                                 np.mean(x * np.sin(w * t))))


def framework_demo(*, fs: float, f0: float, amp: float, sigma_samp: float,
                   dur: float = 20.0, n_mc: int = 200, theta: float = 0.0,
                   seed: int = 0, thr_pct: float = 95.0,
                   seg_s: float = 2.0, h1: float | None = None) -> dict:
    """H0/H1 两组独立试验的相干投影统计（阶段一口径的框架演示）。

    参数
        fs, f0     采样率与已知载频（Hz）
        amp        信号幅值（与 sigma_samp 同一套单位）
        sigma_samp **每样本**噪声标准差（= σ√h₁，见 make_frame 的口径说明）
        dur        单次试验的观测时长（s）；n_mc 每组试验次数
        thr_pct    阈值取 H0 的分位数（默认 95% → 标称 Pfa = 0.05）
        h1         可选，给出时额外回报「力口径」SNR，便于与界面滑块对表

    返回 dict(thr, pfa, pd, stat_h0, stat_h1, seg_h0, seg_h1, t_seg,
              snr_e_db, snr_force_db, snr_samp_db, n, fs, f0, dur, n_mc, thr_pct)

    口径提醒：**能量比 SNR_E 与"每样本"SNR 是同一个数**（两者都是
    Σs²/Σn²，样本数相同），而界面滑块上的 SNR 是"力口径"（A 对 σ 而非
    σ√h₁），比它低 `10lg h₁`（本链路 h₁=0.02 → 17 dB）。

    纪律：**阈值只由 H0 样本标定**（`np.quantile(stat_h0, thr_pct/100)`），
    拿 H1 调阈值会让 Pd 虚高、ROC 不可信。这条在上位机与固件两侧都必须守。
    """
    fs = float(fs)
    f0 = float(f0)
    if fs <= 0 or f0 <= 0:
        raise ValueError("fs 与 f0 必须为正")
    if dur <= 0 or n_mc < 10:
        raise ValueError("dur 必须为正，n_mc 至少 10")
    n = max(64, int(round(dur * fs)))
    t = np.arange(n) / fs
    w = 2.0 * math.pi * f0
    c, s = np.cos(w * t), np.sin(w * t)
    rng = np.random.default_rng(seed)

    sig = float(amp) * np.cos(w * t + float(theta))
    sd = float(sigma_samp)

    def _stat(x2d: np.ndarray) -> np.ndarray:
        # 2·|mean| = 2/n · |x·basis|，矩阵一次算完（200×10000 只要几十毫秒）
        return 2.0 / n * np.hypot(x2d @ c, x2d @ s)

    if sd > 0:
        s0 = _stat(sd * rng.standard_normal((n_mc, n)))
        s1 = _stat(sig[None, :] + sd * rng.standard_normal((n_mc, n)))
    else:                                   # 无噪声：统计量退化，H0 全 0
        s0 = np.zeros(n_mc)
        s1 = _stat(np.repeat(sig[None, :], n_mc, axis=0))

    thr = float(np.quantile(s0, thr_pct / 100.0))
    pfa = float(np.mean(s0 > thr))
    pd = float(np.mean(s1 > thr))

    m = min(n, max(16, int(round(seg_s * fs))))
    # 片段各取第 0 次试验；H0/H1 的噪声独立抽样——两条曲线不该重合
    seg_h0 = sd * rng.standard_normal(m) if sd > 0 else np.zeros(m)
    seg_h1 = sig[:m] + (sd * rng.standard_normal(m) if sd > 0 else 0.0)

    snr_e = snr_of_amp(amp, sd) if sd > 0 else float("inf")
    snr_force = (snr_e + 10.0 * math.log10(h1)) if (h1 and h1 > 0 and sd > 0) else None

    return dict(thr=thr, pfa=pfa, pd=pd, stat_h0=s0, stat_h1=s1,
                seg_h0=np.asarray(seg_h0, float), seg_h1=np.asarray(seg_h1, float),
                t_seg=np.arange(m) / fs, sigma_samp=sd, amp=float(amp),
                snr_e_db=snr_e, snr_samp_db=snr_e, snr_force_db=snr_force,
                n=n, fs=fs, f0=f0, dur=dur, n_mc=n_mc, thr_pct=thr_pct)


# --------------------------------------------------------------------------- 目标窗
def window_coverage(t0: float, dur: float, sig_t0, sig_dur) -> float:
    """一帧 [t0, t0+dur) 落在目标窗 [sig_t0, sig_t0+sig_dur) 内的比例。

    接收端用它**独立重算**覆盖度，不直接信发送端包头里的 cov——
    两边算出来的值必须一致（tests 里有专门一条），这才叫"同步一致"。
    sig_t0 为 None 表示没有附加目标，返回 0。
    """
    if sig_t0 is None or not sig_dur or sig_dur <= 0:
        return 0.0
    a, b = float(t0), float(t0) + float(dur)
    lo, hi = max(a, float(sig_t0)), min(b, float(sig_t0) + float(sig_dur))
    return max(0.0, min(1.0, (hi - lo) / float(dur)))


def coverage_of(meta: dict) -> float:
    """接收端从包头独立算覆盖度（"同步"的唯一入口）。

    优先级：有 t0 与 sig_t0 就自己算；否则退回发送端声明的 cov。
    "常开"模式没有窗口，包头 cov=1 且 sig_t0 为空，走退回分支。
    """
    rep = meta.get("cov")
    rep = 0.0 if rep is None else float(rep)
    t0, n, fs = meta.get("t0"), meta.get("n"), meta.get("fs")
    if t0 is None or meta.get("sig_t0") is None or not n or not fs:
        return rep
    return window_coverage(float(t0), float(n) / float(fs),
                           meta.get("sig_t0"), meta.get("sig_dur"))


class SignalSchedule:
    """连续发送时的目标信号闸门（会话绝对时间，秒）。

    默认不附加；`attach(t_now, dur)` 从"现在"起开一段 dur 秒的窗口。
    窗口不按帧对齐——一帧可能被窗口切一半（cov∈(0,1)），接收端据此
    把该帧标成"部分覆盖"并排除出统计。

    加锁是因为 `attach()` 在界面线程、`coverage()` 在发送线程。
    """

    def __init__(self, always: bool = False):
        self._lk = threading.Lock()
        self.always = bool(always)
        self.t_on: float | None = None
        self.dur = 0.0
        self.theta0 = 0.0
        self.epoch = 0
        self.t_end = 0.0

    @property
    def active(self) -> bool:
        with self._lk:
            return self.t_on is not None and self.dur > 0

    def attach(self, t_now: float, dur: float, theta0: float = 0.0) -> float:
        """从 t_now 起附加 dur 秒；已在附加中则重新计时（窗口顺延）。返回窗口终点。"""
        if not (dur and dur > 0):
            raise ValueError("附加时长必须为正")
        with self._lk:
            self.t_on = float(t_now)
            self.dur = float(dur)
            self.t_end = self.t_on + self.dur
            self.theta0 = float(theta0)
            self.epoch += 1
            return self.t_end

    def cancel(self) -> None:
        with self._lk:
            self.t_on = None
            self.dur = 0.0
            self.t_end = 0.0

    def snapshot(self) -> dict:
        with self._lk:
            return dict(always=self.always, t_on=self.t_on, dur=self.dur,
                        theta0=self.theta0, epoch=self.epoch, t_end=self.t_end)

    def coverage(self, t0: float, dur: float) -> float:
        if self.always:
            return 1.0
        s = self.snapshot()
        if s["t_on"] is None:
            return 0.0
        return window_coverage(t0, dur, s["t_on"], s["dur"])

    def remains(self, t_now: float) -> float:
        s = self.snapshot()
        if s["t_on"] is None:
            return 0.0
        return max(0.0, s["t_end"] - t_now)


class TxStream:
    """连续发送的帧发生器：一次一帧，帧号与绝对时间都不回退、不封顶。

    把「发多少帧」这件事从发送端界面里拿掉——界面只管调用 frame()，
    想停就停。信号闸门挂在 stream 上，界面线程点「附加目标信号」时
    改的就是它，下一帧生成时自然生效。
    """

    def __init__(self, *, w: float, fs: float, duration: float, amp: float,
                 sigma: float, alpha: float = 0.0, has_noise: bool = True,
                 always_sig: bool = False, seed: int = 0,
                 atten_db: float = 0.0, scale: float = ADC_SCALE):
        self.w, self.fs, self.duration = float(w), float(fs), float(duration)
        self.amp, self.sigma, self.alpha = float(amp), float(sigma), float(alpha)
        self.has_noise, self.atten_db = bool(has_noise), float(atten_db)
        self.scale = float(scale)
        self.rng = np.random.default_rng(seed if seed else None)
        self.sched = SignalSchedule(always=always_sig)
        self.k = 0
        self.t0 = 0.0                        # 下一帧的起始绝对时间
        self._lk = threading.Lock()

    # ---- 闸门（界面线程调用）
    def attach(self, dur: float, theta0: float = 0.0) -> float:
        with self._lk:
            return self.sched.attach(self.t0, dur, theta0)

    def cancel(self) -> None:
        self.sched.cancel()

    @property
    def t_now(self) -> float:
        """会话时间：下一帧的起点（也就是已经发出去的时间长度）。"""
        with self._lk:
            return self.t0

    # ---- 出帧（发送线程调用）
    def frame(self) -> dict:
        with self._lk:
            k, t0 = self.k, self.t0
            s = self.sched.snapshot()
            always = s["always"]
            win = None if (always or s["t_on"] is None) else (s["t_on"], s["t_end"])
            theta0 = s["theta0"]

        # 闸门关着时必须显式关信号：win=None 在 make_frame 里表示"整帧有信号"
        gate_on = bool(always or win is not None)
        fr = make_frame(w=self.w, fs=self.fs, duration=self.duration, amp=self.amp,
                        sigma=self.sigma, alpha=self.alpha, has_sig=gate_on,
                        has_noise=self.has_noise, rng=self.rng, t_start=t0,
                        win=win, theta0=theta0)
        fr["frame"] = k
        fr["t0"] = t0
        fr["sig_t0"] = None if (always or s["t_on"] is None) else s["t_on"]
        fr["sig_dur"] = 0.0 if always else s["dur"]
        fr["sig_epoch"] = s["epoch"]
        with self._lk:
            self.k += 1
            self.t0 += fr["n"] / self.fs
        return fr


class VerdictStats:
    """按覆盖度分类的判决统计：背景帧算虚警，整帧覆盖算检出。

    分开算是为了不让"部分覆盖帧"污染两个率：它的能量本来就不够，
    判成"无"是必然的，混进去会把 Pd 拉低、还让人以为是灵敏度不够。
    """

    COV_FULL = 0.99
    COV_NONE = 1e-9

    def __init__(self):
        self.bg = self.bg_hit = 0
        self.sig = self.sig_hit = 0
        self.part = 0
        self.bad = 0

    def add(self, cov: float, detected) -> str:
        """记一帧，返回分类标签：sig / bg / part / bad。"""
        if detected is None:
            self.bad += 1
            return "bad"
        hit = bool(detected)
        if cov >= self.COV_FULL:
            self.sig += 1
            self.sig_hit += int(hit)
            return "sig"
        if cov <= self.COV_NONE:
            self.bg += 1
            self.bg_hit += int(hit)
            return "bg"
        self.part += 1
        return "part"

    @property
    def pd(self) -> float:
        return self.sig_hit / self.sig if self.sig else float("nan")

    @property
    def pfa(self) -> float:
        return self.bg_hit / self.bg if self.bg else float("nan")

    def text(self) -> str:
        pd = "—" if not self.sig else f"{self.pd * 100:.0f}%"
        pfa = "—" if not self.bg else f"{self.pfa * 100:.0f}%"
        s = (f"背景 {self.bg} 帧/虚警 {self.bg_hit}（{pfa}）　"
             f"目标窗 {self.sig} 帧/判有 {self.sig_hit}（{pd}）")
        if self.part:
            s += f"　部分覆盖 {self.part}"
        if self.bad:
            s += f"　无法判定 {self.bad}"
        return s


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
    # 0 拷贝广播：旧写法 np.repeat 要给 180 列各复制一份序列（47100×180×8
    # = 68 MB），纯属白花的内存带宽；内核只按行读 sig[i]，视图中取一行
    # 与复制品逐位相同。
    sig = np.broadcast_to(x.reshape(-1, 1), (x.size, m))
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
