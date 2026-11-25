#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""发送端 GUI：模拟目标发出的极低频磁场，通过 UDP 持续打给接收端。

启动
    .venv\\Scripts\\python.exe tools\\tx_app.py
    .venv\\Scripts\\python.exe tools\\tx_app.py --host 192.168.1.20 --port 50500

行为（协议 v2：连续发送）
    点「开始持续发送」后一直发，**不会自动停**，直到点「停止」或关窗口。
    帧号从 0 单调递增，没有总数。

    目标信号默认**不出来**：链路上只有背景噪声。要测"有目标"，在发送
    过程中点「附加目标信号」——按界面上的「附加时长」注入一段窗
    [sig_t0, sig_t0+sig_dur)。窗口从**下一帧起点**开始算（当前帧已经
    生成并正在播，不能回头改），所以起算延迟 ≤ 1 个帧长；结束时刻按
    请求的时长精确切断，最后一帧可能是"部分覆盖"。

    勾「常开目标信号」等价于旧行为：每帧整帧都有信号，用来做单帧调试。

界面能配什么
    信号：载波角频率 ω、SNR(dB)、基线噪声 σ、采样率 fs、帧长（每帧秒数）
    目标：附加时长、常开开关、手动附加 / 撤销
    发送：随机种子、节奏（尽快 / 实时×倍率）、目标地址与端口

发送什么
    一帧 = A·cos(ωt+φ)·窗（窗内为 1）→ 乘信道衰减 → 加噪声 → 量化成 int16
    噪声每样本标准差 = σ·√(ω/fs)（力口径，与阶段 2–5 一致）。
    A 由 SNR 反算：A = σ·√(2·10^(SNR/10))。
    相位以窗口起点为零点连续推进，跨帧不重置——真实目标不会在帧边界跳相。

数据交互（详见 tools/link_proto.py 顶部）
    UDP 127.0.0.1:50500，一帧 = 1 个 JSON 头包 + ceil(n/1024) 个 int16 数据包
    + 1 个 JSON 尾包。头包里带 n / fs / scale / t0 / sig_t0 / sig_dur / cov，
    接收端据此独立重算覆盖度，把帧分成背景 / 目标窗 / 部分覆盖三类。
"""
from __future__ import annotations

import argparse
import math
import queue
import socket
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import link_proto as LP  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C_TX = "#1f77b4"
C_RX = "#d62728"
C_BG = "#9e9e9e"
C_GRID = "#cccccc"
SNR_E_DEFAULT = "-15"        # 框架演示的默认信噪比（能量比口径，dB）


class TxApp:
    def __init__(self, root: tk.Tk, host: str, port: int):
        self.root = root
        self.q: queue.Queue = queue.Queue()
        self.stop_evt = threading.Event()
        self.worker: threading.Thread | None = None
        self.stream: LP.TxStream | None = None      # 界面线程读它的闸门状态
        self.sent_frames = 0
        self._ui_t = 0.0                            # 上次刷界面的时刻（节流用）

        root.title("发送端 TX — 极低频磁场目标模拟（连续发送）")
        root.geometry("1400x1010")
        root.minsize(1250, 900)
        style = ttk.Style()
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass
        style.configure(".", font=("Microsoft YaHei UI", 9))
        style.configure("Big.TLabel", font=("Microsoft YaHei UI", 11, "bold"))
        style.configure("Go.TButton", font=("Microsoft YaHei UI", 10, "bold"))

        root.columnconfigure(0, weight=0, minsize=478)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        self._build_left(host, port)
        self._build_right()
        self._build_bottom()

        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._after_id = self.root.after(120, self._drain)
        self._refresh_derived()
        self.preview()

    # ------------------------------------------------------------------ 左：参数
    def _build_left(self, host: str, port: int) -> None:
        f = ttk.Frame(self.root, padding=(10, 6, 6, 8), width=470)
        f.grid(row=0, column=0, sticky="nsew")
        f.grid_propagate(False)          # 左栏定宽：不让长说明文字把面板撑宽、把右图挤扁
        f.columnconfigure(0, weight=1)

        # ---- 信号参数
        g = ttk.LabelFrame(f, text=" 信号参数 ", padding=(10, 4, 10, 6))
        g.grid(row=0, column=0, sticky="ew")
        g.columnconfigure(1, weight=0, minsize=150)

        self.v_w = tk.StringVar(value="10.0")
        self.v_snr = tk.DoubleVar(value=-7.4)
        self.v_sigma = tk.StringVar(value="0.05")
        self.v_fs = tk.StringVar(value="500")
        self.v_dur = tk.StringVar(value="94.2")
        self.v_alpha = tk.StringVar(value="0")
        self.v_sig = tk.BooleanVar(value=False)     # 默认不常开：靠手动附加
        self.v_noise = tk.BooleanVar(value=True)

        r = 0
        r = self._row(g, r, "载波角频率 ω", self.v_w, "rad/s")
        self.lb_f0 = ttk.Label(g, text="", foreground="#555555", wraplength=230,
                               font=("Microsoft YaHei UI", 8))
        self.lb_f0.grid(row=r, column=1, sticky="w", pady=(0, 4))
        r += 1

        ttk.Label(g, text="信噪比 SNR").grid(row=r, column=0, sticky="w", pady=3)
        self.sc_snr = ttk.Scale(g, from_=-30.0, to=5.0, variable=self.v_snr, length=152,
                                orient="horizontal", command=lambda _=None: self._refresh_derived())
        self.sc_snr.grid(row=r, column=1, columnspan=2, sticky="w", padx=(6, 4))
        r += 1

        self.lb_amp = ttk.Label(g, text="", foreground="#555555", wraplength=260,
                                font=("Microsoft YaHei UI", 8))
        self.lb_amp.grid(row=r, column=1, sticky="w", pady=(0, 4))
        r += 1

        r = self._row(g, r, "基线噪声 σ", self.v_sigma, "")
        r = self._row(g, r, "采样率 fs", self.v_fs, "Hz")
        r = self._row(g, r, "帧长（每帧）", self.v_dur, "s")
        self.lb_n = ttk.Label(g, text="", foreground="#555555", wraplength=260,
                              font=("Microsoft YaHei UI", 8))
        self.lb_n.grid(row=r, column=1, sticky="w", pady=(0, 4))
        r += 1

        self.lb_snr = ttk.Label(g, text="", wraplength=260,
                                font=("Microsoft YaHei UI", 8))
        self.lb_snr.grid(row=r, column=1, sticky="w", pady=(0, 4))
        r += 1

        ttk.Label(g, text="噪声类型").grid(row=r, column=0, sticky="w", pady=3)
        nf = ttk.Frame(g)
        nf.grid(row=r, column=1, columnspan=2, sticky="w", padx=(6, 0))
        ttk.Radiobutton(nf, text="白噪", value="0", variable=self.v_alpha).pack(side="left")
        ttk.Radiobutton(nf, text="1/f 色噪", value="1", variable=self.v_alpha).pack(side="left", padx=(10, 0))
        r += 1

        sw = ttk.Frame(g)
        sw.grid(row=r, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Checkbutton(sw, text="常开目标信号", variable=self.v_sig,
                        command=self._refresh_derived).pack(side="left")
        ttk.Checkbutton(sw, text="发送噪声", variable=self.v_noise,
                        command=self._refresh_derived).pack(side="left", padx=(14, 0))

        # ---- 目标信号控制
        w = ttk.LabelFrame(f, text=" 目标信号（发送中可手动附加） ", padding=(10, 4, 10, 6))
        w.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        w.columnconfigure(1, weight=0, minsize=150)

        self.v_on = tk.StringVar(value="180")
        rr = 0
        rr = self._row(w, rr, "附加时长", self.v_on, "s")
        self.lb_on = ttk.Label(w, text="", foreground="#555555", wraplength=260,
                               font=("Microsoft YaHei UI", 8))
        self.lb_on.grid(row=rr, column=1, sticky="w", pady=(0, 4))
        rr += 1

        bf = ttk.Frame(w)
        bf.grid(row=rr, column=0, columnspan=2, sticky="w", pady=(2, 0))
        self.bt_attach = ttk.Button(bf, text="附加目标信号", style="Go.TButton",
                                    command=self.attach_signal, state="disabled")
        self.bt_attach.pack(side="left")
        self.bt_detach = ttk.Button(bf, text="撤销附加", command=self.detach_signal,
                                    state="disabled")
        self.bt_detach.pack(side="left", padx=(8, 0))

        self.lb_sig = ttk.Label(w, text="无目标（链路只发背景）", foreground="#616161",
                                wraplength=420)
        self.lb_sig.grid(row=rr + 1, column=0, columnspan=2, sticky="w", pady=(6, 0))

        # ---- 发送参数
        h = ttk.LabelFrame(f, text=" 发送参数 ", padding=(10, 4, 10, 6))
        h.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        h.columnconfigure(1, weight=0, minsize=150)

        self.v_seed = tk.StringVar(value="0")
        self.v_rate = tk.StringVar(value="realtime")
        self.v_speed = tk.StringVar(value="10")
        self.v_host = tk.StringVar(value=host)
        self.v_port = tk.StringVar(value=str(port))

        r = 0
        r = self._row(h, r, "随机种子（0=随机）", self.v_seed, "")

        ttk.Label(h, text="发送节奏").grid(row=r, column=0, sticky="w", pady=3)
        rf = ttk.Frame(h)
        rf.grid(row=r, column=1, columnspan=2, sticky="w", padx=(6, 0))
        ttk.Radiobutton(rf, text="尽快", value="fast", variable=self.v_rate,
                        command=self._refresh_derived).pack(side="left")
        ttk.Radiobutton(rf, text="实时", value="realtime", variable=self.v_rate,
                        command=self._refresh_derived).pack(side="left", padx=(8, 0))
        ttk.Label(rf, text="×").pack(side="left", padx=(6, 2))
        self.en_speed = ttk.Entry(rf, textvariable=self.v_speed, width=3)
        self.en_speed.pack(side="left")
        r += 1

        ttk.Label(h, text="目标地址").grid(row=r, column=0, sticky="w", pady=3)
        hf = ttk.Frame(h)
        hf.grid(row=r, column=1, columnspan=2, sticky="w", padx=(6, 0))
        ttk.Entry(hf, textvariable=self.v_host, width=10).grid(row=0, column=0, sticky="w")
        ttk.Label(hf, text=":").grid(row=0, column=1, padx=2)
        ttk.Entry(hf, textvariable=self.v_port, width=5).grid(row=0, column=2)

        # 左栏不再放长提示：150% DPI 下它会把最后一行挤到窗口外，
        # 「先开接收端」这句并到状态行里（见 _build_bottom）
        f.rowconfigure(3, weight=1)

    def _row(self, parent, r: int, label: str, var, unit: str) -> int:
        """一行 = 标签（单位并在括号里）+ 输入框。单位不单独占列，省得挤。"""
        txt = f"{label} ({unit})" if unit else label
        ttk.Label(parent, text=txt).grid(row=r, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=var, width=13).grid(row=r, column=1, sticky="w", padx=(6, 4))
        return r + 1

    # ------------------------------------------------------------------ 右：图
    def _build_right(self) -> None:
        f = ttk.Frame(self.root, padding=(0, 8, 16, 4))
        f.grid(row=0, column=1, sticky="nsew")
        f.rowconfigure(0, weight=1)
        f.rowconfigure(1, weight=0)   # 图吃掉多余高度，日志按自身请求高度
        f.columnconfigure(0, weight=1)

        self.fig = plt.Figure(figsize=(8.6, 4.6), dpi=100, constrained_layout=True)
        # constrained_layout 默认 pad 只有 0.04 in（≈4 px），频谱的右边框会紧贴
        # 画布边缘，看着像被裁掉。给到 0.16 in。
        self.fig.set_constrained_layout_pads(w_pad=0.16, h_pad=0.08)
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0])
        self.ax_t = self.fig.add_subplot(gs[0, 0])
        self.ax_f = self.fig.add_subplot(gs[0, 1])
        # 画布用 grid_propagate(False) 的容器 + place 铺满：DPI 150% 时
        # TkAgg 的请求高度会到 1300+ px，直接把下面的日志框挤没
        holder = ttk.Frame(f)
        holder.grid(row=0, column=0, sticky="nsew")
        holder.grid_propagate(False)
        holder.rowconfigure(0, weight=1)
        holder.columnconfigure(0, weight=1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=holder)
        self.canvas.get_tk_widget().place(x=0, y=0, relwidth=1, relheight=1)

        box = ttk.LabelFrame(f, text=" 发送日志 ", padding=(6, 4))
        box.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        box.rowconfigure(0, weight=1)
        box.columnconfigure(0, weight=1)
        self.txt = tk.Text(box, height=7, wrap="none", font=("Consolas", 9),
                           background="#fbfbfb", relief="flat")
        self.txt.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(box, orient="vertical", command=self.txt.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.txt.configure(yscrollcommand=sb.set, state="disabled")

    def _build_bottom(self) -> None:
        b = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        b.grid(row=1, column=0, columnspan=2, sticky="ew")
        b.columnconfigure(3, weight=1)

        ttk.Button(b, text="预览波形", command=self.preview).grid(row=0, column=0, padx=(0, 6))
        self.bt_send = ttk.Button(b, text="开始持续发送", style="Go.TButton", command=self.start)
        self.bt_send.grid(row=0, column=1, padx=(0, 6))
        self.bt_stop = ttk.Button(b, text="停止", command=self.stop, state="disabled")
        self.bt_stop.grid(row=0, column=2)
        ttk.Button(b, text="框架演示 Pfa/Pd", command=self.open_demo).grid(
            row=0, column=3, sticky="w", padx=(16, 0))
        ttk.Button(b, text="打开接收端", command=self.launch_rx).grid(row=0, column=4, padx=(6, 0))

        self.lb_stat = ttk.Label(b, text="就绪（先开接收端，再点「开始持续发送」）",
                                 style="Big.TLabel", foreground="#2e7d32")
        self.lb_stat.grid(row=1, column=0, columnspan=5, sticky="w", pady=(6, 0))

    # ------------------------------------------------------------------ 参数
    def _f(self, var, name: str) -> float:
        try:
            return float(str(var.get()).strip())
        except (ValueError, tk.TclError):
            raise ValueError(f"「{name}」不是合法数字") from None

    def params(self) -> dict:
        p = dict(
            w=self._f(self.v_w, "载波角频率"),
            snr=self._f(self.v_snr, "信噪比"),
            sigma=self._f(self.v_sigma, "基线噪声"),
            fs=self._f(self.v_fs, "采样率"),
            duration=self._f(self.v_dur, "帧长"),
            alpha=float(self.v_alpha.get()),
            always=bool(self.v_sig.get()),
            has_noise=bool(self.v_noise.get()),
            on_dur=self._f(self.v_on, "附加时长"),
            seed=int(float(self.v_seed.get())),
            rate=self.v_rate.get(),
            speed=self._f(self.v_speed, "实时倍率"),
            host=str(self.v_host.get()).strip(),
            port=int(float(self.v_port.get())),
        )
        if p["w"] <= 0:
            raise ValueError("载波角频率必须为正")
        if p["fs"] <= 0:
            raise ValueError("采样率必须为正")
        if p["duration"] <= 0:
            raise ValueError("帧长必须为正")
        if p["sigma"] < 0:
            raise ValueError("基线噪声不能为负")
        if p["on_dur"] <= 0:
            raise ValueError("附加时长必须为正")
        if p["rate"] == "realtime" and p["speed"] <= 0:
            raise ValueError("实时倍率必须为正")
        if p["port"] <= 0 or p["port"] > 65535:
            raise ValueError("端口须在 1–65535")
        p["amp"] = LP.amp_of_snr(p["snr"], p["sigma"]) if p["sigma"] > 0 else 0.02
        return p

    def _refresh_derived(self) -> None:
        try:
            w = self._f(self.v_w, "ω")
            fs = self._f(self.v_fs, "fs")
            dur = self._f(self.v_dur, "时长")
            sigma = self._f(self.v_sigma, "σ")
            on = self._f(self.v_on, "附加时长")
        except ValueError:
            return
        f0 = w / LP.TWOPI
        self.lb_f0.configure(text=f"f0 = {f0:.4f} Hz / T = {1.0 / f0:.3f} s" if f0 > 0 else "")
        snr = self.v_snr.get()
        amp = LP.amp_of_snr(snr, sigma) if sigma > 0 else float("nan")
        self.lb_amp.configure(text=f"{snr:+.1f} dB  →  A = {amp:.4f}"
                                   + ("（关噪声时不生效）" if not self.v_noise.get() else ""))
        n = int(round(dur * fs))
        cyc = dur * f0
        if cyc < 30:
            note, col = "   ← 太短：判据不稳，建议 ≥48 周期", "#c62828"
        elif cyc < 60:
            note, col = "   ← 偏短：命中率开始抖动，建议 ≥48 周期", "#ef6c00"
        else:
            note, col = "", "#555555"
        self.lb_n.configure(text=f"{n} 点 / {cyc:.1f} 个周期{note}", foreground=col)

        # 两个 SNR 口径差 10lg(h1)：界面上的 SNR 是"力口径"（与阶段 2–5 一致），
        # 波形上能不能看出正弦要看"每样本"口径 —— 两者差 10lg(ω/fs) ≈ −17 dB
        h1 = w / fs if fs > 0 else 0.0
        s_samp = sigma * math.sqrt(h1) if (h1 > 0 and sigma > 0) else 0.0
        if s_samp > 0:
            snr_s = 10.0 * math.log10(max(amp ** 2 / 2.0, 1e-30) / s_samp ** 2)
            vis = "波形可见正弦" if snr_s >= 6 else ("勉强可辨" if snr_s >= 0 else "波形被噪声淹没")
            self.lb_snr.configure(
                text=f"每样本 SNR = {snr_s:+.1f} dB → {vis}",
                foreground="#2e7d32" if snr_s >= 6 else ("#ef6c00" if snr_s >= 0 else "#c62828"))
        else:
            self.lb_snr.configure(text="", foreground="#555555")

        nfr = on / dur if dur > 0 else 0.0
        self.lb_on.configure(
            text=f"≈{nfr:.1f} 帧，从下一帧起算" if not self.v_sig.get()
                 else "常开时不生效",
            foreground="#555555")
        self.en_speed.configure(state="normal" if self.v_rate.get() == "realtime" else "disabled")

    # ------------------------------------------------------------------ 预览
    def preview(self) -> None:
        try:
            p = self.params()
        except ValueError as e:
            messagebox.showerror("参数有误", str(e))
            return
        seed = p["seed"] if p["seed"] else 7
        fr_t = LP.make_frame(w=p["w"], fs=p["fs"], duration=p["duration"], amp=p["amp"],
                             sigma=p["sigma"], alpha=p["alpha"], has_sig=True,
                             has_noise=p["has_noise"], rng=np.random.default_rng(seed))
        self._draw(fr_t, p, cov=1.0)
        snr_txt = "—" if fr_t["snr_db"] is None else f"{fr_t['snr_db']:+.2f} dB"
        self._log(f"预览：n={fr_t['n']}，f0={p['w'] / LP.TWOPI:.4f} Hz，"
                  f"A={p['amp']:.4f}，SNR={snr_txt}")

    def _bg_frame(self, p: dict) -> dict:
        """参考背景帧（只有噪声），缓存起来——每帧重算 4.7 万点没必要。"""
        key = (p["w"], p["fs"], p["duration"], p["sigma"], p["alpha"],
               p["has_noise"], p["seed"])
        if getattr(self, "_bg_key", None) != key or getattr(self, "_bg_fr", None) is None:
            self._bg_fr = LP.make_frame(w=p["w"], fs=p["fs"], duration=p["duration"],
                                        amp=p["amp"], sigma=p["sigma"], alpha=p["alpha"],
                                        has_sig=False, has_noise=p["has_noise"],
                                        rng=np.random.default_rng(11))
            self._bg_key = key
        return self._bg_fr

    def _draw(self, fr: dict, p: dict, cov: float) -> None:
        """画当前帧，并叠一条**参考正弦**——否则发背景帧时蓝线是 0，看着像"没信号"。

        蓝粗线 = 整帧有目标时该有的干净波形（A·cos(ωt+φ)）；
        红线   = 当前帧到达的样本；
        灰细线 = 纯背景参考（仅在当前帧有目标时画，用来对比）。
        """
        t = np.arange(fr["n"]) / fr["fs"]
        show = min(fr["n"], int(round(3.2 * fr["fs"])))       # 3.2 s ≈ 5 个载波周期
        ph = fr["phase"] if cov > 0 else 0.0
        tx_ref = p["amp"] * np.cos(p["w"] * t + ph)

        self.ax_t.clear()
        if cov > 0:
            bg = self._bg_frame(p)
            self.ax_t.plot(t[:show], bg["rx"][:show], color=C_BG, lw=0.6, alpha=0.45,
                           label="纯背景（参考）", zorder=1)
        self.ax_t.plot(t[:show], fr["rx"][:show], color=C_RX, lw=0.7, alpha=0.6,
                       label=f"当前帧（cov={cov:.2f}）", zorder=2)
        self.ax_t.plot(t[:show], tx_ref[:show], color=C_TX, lw=2.0,
                       label="参考正弦（非接收数据）", zorder=3)

        tag = "背景（无目标）" if cov <= 1e-9 else ("整帧覆盖" if cov >= 0.99 else "部分覆盖")
        self.ax_t.set(title="波形（前 3.2 s ≈ 5 个周期）", xlabel="时间 / s", ylabel="幅值")
        # 帧类型写在轴内，别塞标题——标题太长会被两侧裁掉
        self.ax_t.text(0.02, 0.04, f"当前帧：{tag}", transform=self.ax_t.transAxes,
                       fontsize=8, color=("#c62828" if cov > 0 else "#616161"))
        self.ax_t.grid(alpha=0.25, color=C_GRID)
        self.ax_t.legend(fontsize=8, loc="upper right")

        x = fr["rx"] - fr["rx"].mean()
        if np.any(x):
            nfft = 1 << int(math.ceil(math.log2(max(fr["n"], 64) * 4)))
            spec = np.abs(np.fft.rfft(x * np.hanning(fr["n"]), nfft))
            freq = np.fft.rfftfreq(nfft, 1.0 / fr["fs"])
            band = freq <= 6.0
            self.ax_f.clear()
            self.ax_f.semilogy(freq[band], np.maximum(spec[band], 1e-12), color=C_TX, lw=1.0)
            self.ax_f.axvline(p["w"] / LP.TWOPI, color=C_RX, ls="--", lw=1.2,
                              label=f"f0 = {p['w'] / LP.TWOPI:.4f} Hz")
            self.ax_f.set(title="当前帧频谱", xlabel="频率 / Hz", ylabel="|X(f)|",
                          xlim=(0.0, 6.2))
            self.ax_f.grid(alpha=0.25, color=C_GRID, which="both")
            self.ax_f.legend(fontsize=8)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ 发送
    def start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        try:
            p = self.params()
        except ValueError as e:
            messagebox.showerror("参数有误", str(e))
            return
        self.stop_evt.clear()
        self.sent_frames = 0
        self._ui_t = 0.0
        self.stream = None
        self.bt_send.configure(state="disabled")
        self.bt_stop.configure(state="normal")
        self.bt_attach.configure(state="normal")
        self.bt_detach.configure(state="disabled")
        self.lb_stat.configure(text="持续发送中…", foreground="#ef6c00")
        self.lb_sig.configure(text="无目标（链路只发背景）", foreground="#616161")
        self.worker = threading.Thread(target=self._worker, args=(p,), daemon=True)
        self.worker.start()

    def _worker(self, p: dict) -> None:
        st = LP.TxStream(w=p["w"], fs=p["fs"], duration=p["duration"], amp=p["amp"],
                         sigma=p["sigma"], alpha=p["alpha"], has_noise=p["has_noise"],
                         always_sig=p["always"], seed=p["seed"],
                         atten_db=0.0, scale=LP.ADC_SCALE)
        self.stream = st                       # 界面线程从这里读闸门状态

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
        addr = (p["host"], p["port"])
        gap = 0.002 if p["rate"] == "fast" else (LP.CHUNK / p["fs"]) / p["speed"]

        try:
            while not self.stop_evt.is_set():
                fr = st.frame()
                k = fr["frame"]
                meta = dict(frame=k, n=fr["n"], fs=p["fs"], scale=st.scale,
                            w_hint=p["w"], amp=p["amp"], sigma=p["sigma"],
                            alpha=p["alpha"],
                            snr_db=None if fr["snr_db"] is None else round(fr["snr_db"], 3),
                            has_sig=bool(fr["cov"] > 0), has_noise=p["has_noise"],
                            cov=round(fr["cov"], 6), t0=round(fr["t0"], 6),
                            sig_t0=None if fr["sig_t0"] is None else round(fr["sig_t0"], 6),
                            sig_dur=round(fr["sig_dur"], 6), sig_epoch=fr["sig_epoch"],
                            duration=round(fr["n"] / p["fs"], 4),
                            phase=round(fr["phase"], 5), ts=time.time(),
                            host=p["host"], port=p["port"])
                sock.sendto(LP.pack_head(meta), addr)
                for pkt in LP.split_frame(k, fr["rx"], p["fs"], st.scale):
                    if self.stop_evt.is_set():
                        break
                    sock.sendto(pkt, addr)
                    time.sleep(gap)
                if self.stop_evt.is_set():
                    break
                nseq = int(math.ceil(fr["n"] / LP.CHUNK))
                sock.sendto(LP.pack_tail(dict(frame=k, sent=fr["n"], nseq=nseq)), addr)

                self.sent_frames += 1
                # 界面最多 ~3 帧/s：'尽快'模式一秒出上百帧，全塞进 Tk 会把
                # 事件队列撑爆（实测 _draw 一次约 120 ms，update() 退不回来）。
                # 计数永远准，画面只是抽样。
                now = time.time()
                if now - self._ui_t >= 0.35:
                    self._ui_t = now
                    self.q.put(("frame", dict(meta=meta, fr=fr, p=p)))
        except OSError as e:
            self.q.put(("err", f"发送失败：{e}"))
        finally:
            sock.close()
            self.q.put(("done", self.sent_frames))

    def stop(self) -> None:
        self.stop_evt.set()
        self.bt_stop.configure(state="disabled")
        self.bt_attach.configure(state="disabled")
        self.bt_detach.configure(state="disabled")
        self._log("已请求停止…")

    # ------------------------------------------------------------------ 目标信号
    def attach_signal(self) -> None:
        if not (self.stream and self.worker and self.worker.is_alive()):
            messagebox.showinfo("还没开始", "先点「开始持续发送」，再附加目标信号。")
            return
        try:
            dur = self._f(self.v_on, "附加时长")
        except ValueError as e:
            messagebox.showerror("参数有误", str(e))
            return
        if dur <= 0:
            messagebox.showerror("参数有误", "附加时长必须为正")
            return
        t_end = self.stream.attach(dur)
        self.bt_detach.configure(state="normal")
        self._refresh_sig_state()          # 立刻刷新，别等下一次 120 ms 节拍
        self._log(f"附加目标：窗口 [{self.stream.t_now:.1f}, {t_end:.1f}) s，"
                  f"时长 {dur:.1f} s（从下一帧起算）")

    def detach_signal(self) -> None:
        if self.stream:
            self.stream.cancel()
            self.bt_detach.configure(state="disabled")
            self._refresh_sig_state()
            self._log("已撤销附加，链路回到纯背景")

    # ------------------------------------------------------------------ 队列与日志
    def _drain(self) -> None:
        events = []
        try:
            while True:
                events.append(self.q.get_nowait())
        except queue.Empty:
            pass

        last_draw = None
        finished = False
        for kind, obj in events:
            if kind == "frame":
                last_draw = obj
            elif kind == "done":
                finished = True
                self.bt_send.configure(state="normal")
                self.bt_stop.configure(state="disabled")
                self.bt_attach.configure(state="disabled")
                self.bt_detach.configure(state="disabled")
                self.lb_stat.configure(text=f"已停止，共发 {obj} 帧", foreground="#2e7d32")
                self.lb_sig.configure(text="发送已停止", foreground="#616161")
                self._log(f"发送停止，共 {obj} 帧")
            elif kind == "err":
                finished = True
                self.bt_send.configure(state="normal")
                self.bt_stop.configure(state="disabled")
                self.bt_attach.configure(state="disabled")
                self.bt_detach.configure(state="disabled")
                self.lb_stat.configure(text="发送失败", foreground="#c62828")
                self._log(str(obj))

        if last_draw:
            meta, fr, p = last_draw["meta"], last_draw["fr"], last_draw["p"]
            cov = meta["cov"]
            self._draw(fr, p, cov)
            tag = ("整帧覆盖" if cov >= 0.99 else "部分覆盖" if cov > 0 else "背景")
            self._log(f"[帧 {meta['frame']}] t0={meta['t0']:.1f}s n={meta['n']} "
                      f"cov={cov:.2f}（{tag}）φ={meta['phase']:+.3f} rad")
            if not finished:                      # 别把"已停止"覆盖回"发送中"
                self.lb_stat.configure(
                    text=f"持续发送中：已发 {self.sent_frames} 帧　"
                         f"{p['host']}:{p['port']}", foreground="#ef6c00")

        self._refresh_sig_state()

        try:
            self._after_id = self.root.after(120, self._drain)
        except tk.TclError:
            pass

    def _refresh_sig_state(self) -> None:
        st = self.stream
        if not (st and self.worker and self.worker.is_alive()):
            return
        s = st.sched.snapshot()
        if s["always"]:
            self.lb_sig.configure(text="常开目标信号：每帧整帧都有", foreground="#2e7d32")
            return
        if s["t_on"] is None:
            self.lb_sig.configure(text="无目标（链路只发背景）", foreground="#616161")
            return
        rem = st.sched.remains(st.t_now)
        if rem <= 0:
            self.lb_sig.configure(text="目标窗口已结束，回到纯背景", foreground="#616161")
        else:
            self.lb_sig.configure(text=f"附加中：剩余 {rem:.0f} s（窗至 "
                                       f"{s['t_end']:.0f} s）",
                                  foreground="#c62828")

    def _log(self, msg: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.txt.configure(state="normal")
        self.txt.insert("end", f"[{stamp}] {msg}\n")
        self.txt.see("end")
        self.txt.configure(state="disabled")

    def launch_rx(self) -> None:
        try:
            port = str(int(float(self.v_port.get())))
        except (ValueError, tk.TclError):
            port = str(LP.PORT_DEFAULT)
        script = TOOLS / "rx_app.py"
        try:
            subprocess.Popen([sys.executable, str(script), "--port", port])
            self._log(f"已启动接收端：python tools/rx_app.py --port {port}")
        except OSError as e:
            self._log(f"启动接收端失败：{e}")

    def open_demo(self) -> None:
        """打开「框架演示」窗口（H0/H1 输入片段 + 相干投影统计）。

        窗口是单例：再点一次只是把它提到前面并重算，不叠一堆窗口。
        """
        win = getattr(self, "demo_win", None)
        if win is not None and win.winfo_exists():
            win.deiconify()
            win.lift()
            win.recompute()
            return
        try:
            self.demo_win = DemoWindow(self.root, self)
        except (ValueError, tk.TclError) as e:
            messagebox.showerror("框架演示无法打开", str(e))
            self.demo_win = None
            return
        self._log("已打开框架演示窗口（H0/H1 与 Pfa/Pd）")

    def _on_close(self) -> None:
        self.stop_evt.set()
        try:
            self.root.after_cancel(self._after_id)
        except (tk.TclError, AttributeError):
            pass
        if self.worker:
            self.worker.join(timeout=1.0)
        win = getattr(self, "demo_win", None)
        if win is not None:
            try:
                win.destroy()
            except tk.TclError:
                pass
        plt.close(self.fig)
        self.root.destroy()


# --------------------------------------------------------------------------- 框架演示
class DemoWindow(tk.Toplevel):
    """H0 / H1 两幅图：输入片段 + 相干投影统计分布（含 Pfa / Pd）。

    复现 `learn/1.ipynb` 里那张「输入片段 / 框架演示」图，但**默认用当前
    发送参数**算 —— 于是它展示的是"这套链路自己"的统计分离，而不是抄一张
    别处的图。下拉框里的第二项保留阶段一算例（γ=1 nT、σ=7.07 nT、−20 dB），
    用来一对一对照那张图。

    纪律（图里也写着）：**阈值只由 H0 样本的 95% 分位标定**，绝不拿 H1 调；
    统计量是已知频率处的相干投影，只演示假设检验框架，**不是本课题的检测器**
    （真正的判据是阶段 3 的相变，实现见 rx_app 的 pstd）。
    """

    PRESET_TX = "当前发送参数（归一化）"
    PRESET_S1 = "阶段一算例（nT 口径）"

    def __init__(self, master: tk.Tk, app: "TxApp"):
        super().__init__(master)
        self.app = app
        self.res: dict | None = None
        self.unit = "（归一化）"
        self.title("框架演示：H0 / H1 与相干投影统计")
        self.geometry("1240x760")
        self.minsize(1000, 700)
        try:
            self.attributes("-topmost", False)
        except tk.TclError:
            pass

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.fig = plt.Figure(figsize=(12.0, 4.0), dpi=100, constrained_layout=True)
        self.fig.set_constrained_layout_pads(w_pad=0.12, h_pad=0.06)
        self.ax_x = self.fig.add_subplot(1, 2, 1)
        self.ax_h = self.fig.add_subplot(1, 2, 2)

        holder = ttk.Frame(self)
        holder.grid(row=0, column=0, sticky="nsew")
        holder.grid_propagate(False)
        holder.rowconfigure(0, weight=1)
        holder.columnconfigure(0, weight=1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=holder)
        self.canvas.get_tk_widget().place(x=0, y=0, relwidth=1, relheight=1)

        bar = ttk.Frame(self, padding=(10, 4, 10, 8))
        bar.grid(row=1, column=0, sticky="ew")

        # 用 pack(side="right") 先把按钮占到右边：grid 的列宽分配在窗口变窄时
        # 会把最后一个按钮挤出画布（看起来像被裁掉），pack 不会。
        btns = ttk.Frame(bar)
        btns.grid(row=1, column=0, sticky="e", pady=(6, 0))
        ttk.Button(btns, text="关闭", command=self.destroy).pack(side="right", padx=(6, 0))
        ttk.Button(btns, text="保存", command=self.save).pack(side="right", padx=(6, 0))
        ttk.Button(btns, text="重算", style="Go.TButton",
                   command=self.recompute).pack(side="right")

        left = ttk.Frame(bar)
        left.grid(row=0, column=0, sticky="w")

        # 默认就是用户要的那张图（阶段一算例），"当前发送参数"作为第二项
        self.v_preset = tk.StringVar(value=self.PRESET_S1)
        cb = ttk.Combobox(left, textvariable=self.v_preset, width=21, state="readonly",
                          values=(self.PRESET_S1, self.PRESET_TX))
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda _e: self.on_preset())

        # 两个独立输入：**幅值 A** 与 **SNR_E**（能量比口径，默认 −15 dB）。
        # σ 不填，由这两者反算：σ = √((A²/2)/10^(SNR_E/10))，只读显示。
        # 这里不用界面滑块上的"力口径 SNR"——那是 A 对 σ，两者差 10lg h₁。
        ttk.Label(left, text="幅值 A").pack(side="left", padx=(12, 4))
        self.v_A = tk.StringVar(value="1.0")
        self.en_A = ttk.Entry(left, textvariable=self.v_A, width=9)
        self.en_A.pack(side="left")
        self.en_A.bind("<Return>", lambda _e: self.recompute())

        ttk.Label(left, text="SNR_E").pack(side="left", padx=(10, 4))
        self.v_snr = tk.StringVar(value=SNR_E_DEFAULT)
        self.en_snr = ttk.Entry(left, textvariable=self.v_snr, width=7)
        self.en_snr.pack(side="left")
        self.en_snr.bind("<Return>", lambda _e: self.recompute())

        ttk.Label(left, text="σ").pack(side="left", padx=(10, 4))
        self.lb_sigma = ttk.Label(left, text="—", foreground="#455a64", width=9)
        self.lb_sigma.pack(side="left")

        ttk.Label(left, text="T (s)").pack(side="left", padx=(12, 4))
        self.v_T = tk.StringVar(value="20")
        self.sp_T = ttk.Spinbox(left, from_=0.1, to=200.0, increment=1.0, width=6,
                                textvariable=self.v_T, command=self.recompute)
        self.sp_T.pack(side="left")

        ttk.Label(left, text="N/组").pack(side="left", padx=(12, 4))
        self.v_nmc = tk.StringVar(value="200")
        self.sp_nmc = ttk.Spinbox(left, from_=50, to=2000, increment=50, width=6,
                                  textvariable=self.v_nmc, command=self.recompute)
        self.sp_nmc.pack(side="left")
        for sp in (self.sp_T, self.sp_nmc):
            sp.bind("<Return>", lambda _e: self.recompute())

        # 说明文字单独占一行、单独一个 Frame：塞进控件那一行会参与列宽分配，
        # 把「关闭」按钮挤到窗口边缘外（看起来像被裁了）
        inf = ttk.Frame(self, padding=(10, 0, 10, 8))
        inf.grid(row=2, column=0, sticky="ew")
        self.lb_info = ttk.Label(inf, text="", foreground="#455a64", justify="left",
                                 wraplength=1180)
        self.lb_info.grid(row=0, column=0, sticky="w")

        self.recompute()

    # ------------------------------------------------------------------ 计算
    def _defaults(self) -> dict:
        """两套口径的固定量 + 两个输入的默认值（A、SNR_E 都是可改的）。"""
        if self.v_preset.get() == self.PRESET_S1:
            # 阶段一算例：nT 物理单位
            return dict(unit=" / nT", fs=200.0, f0=1.0, A=1.0,
                        snr=float(SNR_E_DEFAULT), h1=None)
        p = self.app.params()                    # 顺带做参数合法性检查
        h1 = p["w"] / p["fs"]
        sd = p["sigma"] * math.sqrt(h1)          # 每样本标准差（力口径 → 每样本）
        # 发送端当前的等效 SNR_E（能量比 = 每样本），供输入框起点
        snr = LP.snr_of_amp(p["amp"], sd) if sd > 0 else float("inf")
        return dict(unit="（归一化）", fs=p["fs"], f0=p["w"] / LP.TWOPI,
                    A=p["amp"], snr=snr, h1=h1)

    def on_preset(self) -> None:
        """换口径时把 A 与 SNR_E 填回该口径的默认值（两个都可改，但要有个起点）。"""
        try:
            d = self._defaults()
        except ValueError as e:
            messagebox.showerror("参数有误", str(e), parent=self)
            return
        self.v_A.set(f"{d['A']:.4g}")
        self.v_snr.set(f"{d['snr']:.2f}")
        self.recompute()

    def _cond(self) -> dict:
        """按当前下拉框 + 两个输入框决定这次演示的条件。

        A 与 SNR_E 是输入，σ 由它们反算：σ = √((A²/2)/10^(SNR_E/10))。
        """
        try:
            n_mc = int(float(self.v_nmc.get()))
        except (ValueError, tk.TclError):
            n_mc = 200
        n_mc = max(20, min(5000, n_mc))
        try:
            dur = float(str(self.v_T.get()).strip())
        except (ValueError, tk.TclError):
            dur = 20.0
        dur = max(0.05, min(400.0, dur))

        d = self._defaults()
        try:
            amp = float(str(self.v_A.get()).strip())
        except (ValueError, tk.TclError):
            raise ValueError("「幅值 A」不是合法数字") from None
        if amp <= 0:
            raise ValueError("「幅值 A」必须为正")
        try:
            snr = float(str(self.v_snr.get()).strip())
        except (ValueError, tk.TclError):
            raise ValueError("「SNR_E」不是合法数字") from None
        if not (-80.0 <= snr <= 40.0):
            raise ValueError("「SNR_E」请填 −80…+40 dB 之间")

        sd = math.sqrt((amp ** 2 / 2.0) / (10.0 ** (snr / 10.0)))
        self.lb_sigma.configure(text=f"{sd:.4g}")
        return dict(unit=d.pop("unit"), n_mc=n_mc, fs=d["fs"], f0=d["f0"],
                    amp=amp, sigma_samp=sd, dur=dur, h1=d["h1"],
                    thr_pct=95.0, seg_s=2.0)

    def recompute(self) -> None:
        try:
            c = self._cond()
        except ValueError as e:
            messagebox.showerror("参数有误", str(e), parent=self)
            return
        self.unit = c.pop("unit")
        r = LP.framework_demo(**c)
        self.res = r
        self._draw(r)
        self._info(r)

    # ------------------------------------------------------------------ 画图
    def _draw(self, r: dict) -> None:
        ax = self.ax_x
        ax.clear()
        ax.plot(r["t_seg"], r["seg_h0"], lw=0.85, alpha=0.85, color="#1f77b4",
                label="H0：仅噪声")
        ax.plot(r["t_seg"], r["seg_h1"], lw=0.85, alpha=0.85, color="#ff7f0e",
                label="H1：信号+噪声")
        ax.set_xlabel("时间 t / s", fontsize=9)
        ax.set_ylabel("x(t)" + self.unit, fontsize=9)
        ax.set_title(f"输入片段（σ={r['sigma_samp']:.3g}，SNR_E={r['snr_e_db']:+.1f} dB）",
                     fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25, color=C_GRID)
        ax.legend(fontsize=8, loc="upper right")

        ah = self.ax_h
        ah.clear()
        hi = max(float(r["stat_h0"].max()), float(r["stat_h1"].max())) * 1.05
        bins = np.linspace(0.0, hi if hi > 0 else 1.0, 30)
        ah.hist(r["stat_h0"], bins=bins, alpha=0.72, color="#1f77b4",
                label=r"H0（$P_{fa}$）")
        ah.hist(r["stat_h1"], bins=bins, alpha=0.72, color="#ff7f0e",
                label=r"H1（$P_d$）")
        ah.axvline(r["thr"], color="k", ls="--", lw=1.2,
                   label=f"H0 的 95% 分位 = {r['thr']:.3g}")
        ah.set_xlabel("相干投影幅值" + self.unit, fontsize=9)
        ah.set_ylabel("次数", fontsize=9)
        ah.set_title(f"框架演示：Pfa={r['pfa']:.3f}，Pd={r['pd']:.3f}"
                     f"（{r['n_mc']} 次/组）", fontsize=10)
        ah.tick_params(labelsize=8)
        ah.grid(alpha=0.25, color=C_GRID)
        ah.legend(fontsize=8, loc="upper right")
        self.canvas.draw_idle()

    def _info(self, r: dict) -> None:
        l1 = [f"A = {r['amp']:.4g}", f"σ（每样本）= {r['sigma_samp']:.4g}",
              f"SNR_E = {r['snr_e_db']:+.1f} dB", f"fs = {r['fs']:g} Hz",
              f"f0 = {r['f0']:.4f} Hz", f"T = {r['dur']:g} s", f"n = {r['n']} 点"]
        l2 = [f"阈值（仅 H0 标定，{r['thr_pct']:g}% 分位）= {r['thr']:.4g}"]
        if r.get("snr_force_db") is not None:
            l2.append(f"（同一份数据按力口径算是 {r['snr_force_db']:+.1f} dB，"
                      f"差 10lg h₁；界面滑块用的是那个）")
        # 两组分布的相对距离：>10 倍就说明这套参数下已经完全分开，看不到过渡带
        ratio = (float(np.median(r["stat_h1"])) / r["thr"]) if r["thr"] > 0 else float("inf")
        if ratio > 10:
            l2.append(f"两组相差 {ratio:.0f} 倍（已完全分离；想看过渡带把 T 调小）")
        elif ratio > 2:
            l2.append(f"两组相差 {ratio:.1f} 倍（过渡带，Pd 随 Pfa 变化）")
        else:
            l2.append(f"两组相差 {ratio:.1f} 倍（几乎重叠，检测已到极限）")

        l3 = ("统计量 = 已知频率处的相干投影幅值 2|⟨x·e^(-jωt)⟩|；只演示假设检验框架，"
              "阈值只由 H0 定（拿 H1 调会让 Pd 虚高），本课题真正的判据是阶段 3 的相变。")
        self.lb_info.configure(text="　·　".join(l1) + "\n" + "　·　".join(l2) + "\n" + l3)

    # ------------------------------------------------------------------ 存盘
    def save(self) -> None:
        if self.res is None:
            return
        out = TOOLS.parent / "复现输出"
        out.mkdir(exist_ok=True)
        name = ("框架演示-PfaPd-阶段一算例.png" if self.v_preset.get() == self.PRESET_S1
                else "框架演示-PfaPd-发送端参数.png")
        p = out / name
        self.fig.savefig(p, dpi=140)
        self.app._log(f"框架演示图已存：{p}")


def main() -> int:
    ap = argparse.ArgumentParser(description="发送端 GUI（连续发送含噪载波，可手动附加目标窗）")
    ap.add_argument("--host", default="127.0.0.1", help="目标地址（默认本机回环）")
    ap.add_argument("--port", type=int, default=LP.PORT_DEFAULT, help="目标端口")
    args = ap.parse_args()
    root = tk.Tk()
    TxApp(root, args.host, args.port)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
