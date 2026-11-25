#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""接收端 GUI：连续监听 UDP，每凑齐一帧就判「有没有目标」并显示阵列图与波形图。

启动
    .venv\\Scripts\\python.exe tools\\rx_app.py
    .venv\\Scripts\\python.exe tools\\rx_app.py --port 50500 --host 0.0.0.0

界面能配什么
    网络：监听地址与端口
    时基：采样间隔 dt（勾「跟随发送端」时取头包里的 1/fs）
    定标：R 模式三选一
           已知 R   直接填 R，不扫频（默认；也是 ESP32 固件的实现口径）
           跟随发送端 用头包里的 w_hint（联调对表旁路，正常检测不用）
           自动扫 R  FFT 粗估 → ±1% 两级细扫（论文的盲测流程，最慢）
    判据：λ0（默认 0.73）、判态门限 pstd_th、阵列（三相位 / 单路）、是否估幅值

    λ0 默认取 0.73 而不是内核的 0.72：离相变点 λc=0.7335 只剩 0.0035，目标
    只要把合成幅值推过线就能判出，背景仍稳在 0.23~0.40（离门限 0.08 有 3 倍
    余量）。30 s 帧实测：目标帧 β=0 路 pstd=2.2e-3，背景帧最小 0.23。

连续接收（协议 v2）
    发送端不再"发 N 帧就停"，而是一直发；接收端一帧接一帧地收，
    帧号应当连续。丢帧、残帧、分析跟不上这三种情况都会在日志里报出来，
    不会静默跳过：

        新头包顶掉未收完的旧帧  → 报"残帧丢弃"
        frame 号跳变            → 报"缺 k 帧"
        分析队列积压            → 丢最旧的一帧（保最新），报累计数

    每帧按目标窗覆盖度分类，统计分开算，互不污染：

        cov = 1      目标窗帧  → 判「有」记检出（算 Pd）
        cov = 0      背景帧    → 判「有」记虚警（算 Pfa）
        0< cov <1    部分覆盖  → 只记数，不进 Pd/Pfa（能量本来就不够）

    覆盖度不是照抄头包里的 cov：接收端用 t0/sig_t0/sig_dur 自己重算
    （link_proto.coverage_of），两边算出来必须一致。

界面显示什么
    波形图    接收序列（时域，前 2 s）+ 频谱（标出粗估频率）
    阵列图    三路 pstd 柱状（对数轴）+ 门限线；三路相轨迹
    时序条    最近若干帧的判决色块（绿=目标窗判有、红=漏检、灰=背景、
              橙=虚警、黄=部分覆盖），一眼看出目标窗何时开始、何时结束

数据交互
    UDP 收包，按 frame → seq 装配成整帧；收到尾包或凑够 n 点即提交。
    3 秒内没凑齐的残帧丢弃并报缺包数。
"""
from __future__ import annotations

import argparse
import math
import queue
import socket
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
from matplotlib.patches import Rectangle  # noqa: E402

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import link_proto as LP  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C_WAVE = "#37474f"
C_SPEC = "#1f77b4"
C_HIT = "#2e7d32"
C_MISS = "#9e9e9e"
C_TH = "#d62728"
C_GRID = "#cccccc"
TRACE_PTS = 2000           # 相图只画尾部这么多点，避免 4 万点把界面拖慢
LAM_DEFAULT = "0.73"       # 判决工作点（内核默认 0.72，这里按联调要求抬到 0.73）


class RxApp:
    def __init__(self, root: tk.Tk, host: str, port: int):
        self.root = root
        self.q: queue.Queue = queue.Queue()          # 工作线程 → UI
        self.frame_q: queue.Queue = queue.Queue(maxsize=1)   # 接收线程 → 工作线程
        self.stop_evt = threading.Event()
        self.rx_thread: threading.Thread | None = None
        self.wk_thread: threading.Thread | None = None
        self.buf: LP.FrameBuf | None = None
        self.stats = LP.VerdictStats()               # 背景 / 目标窗 / 部分覆盖分开算
        self.hist: list[dict] = []                   # 时序条用：最近若干帧的判决
        self.dropped = 0                             # 分析跟不上、被丢掉的帧数
        self.last_frame = -1                         # 上一帧帧号（查跳帧）
        self.n_recv = 0
        self.n_hit = 0
        self.last_res: dict | None = None
        self.cur: dict | None = None        # 主线程刷新的参数快照（工作线程每帧读它）

        root.title("接收端 RX — 混沌振子阵列检测（连续接收）")
        root.geometry("1330x1010")
        root.minsize(1160, 780)
        style = ttk.Style()
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass
        style.configure(".", font=("Microsoft YaHei UI", 9))
        style.configure("Verdict.TLabel", font=("Microsoft YaHei UI", 17, "bold"))
        style.configure("Info.TLabel", font=("Consolas", 10))
        style.configure("Go.TButton", font=("Microsoft YaHei UI", 10, "bold"))

        root.columnconfigure(0, weight=0, minsize=398)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(1, weight=1)

        self._build_top()
        self._build_left(host, port)
        self._build_plots()

        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._refresh_derived()
        self._after_id = self.root.after(100, self._drain)

    # ------------------------------------------------------------------ 顶部状态
    def _build_top(self) -> None:
        bar = ttk.Frame(self.root, padding=(12, 10, 12, 4))
        bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        bar.columnconfigure(1, weight=1)

        self.lb_verdict = ttk.Label(bar, text="待命", style="Verdict.TLabel",
                                    foreground="#757575")
        self.lb_verdict.grid(row=0, column=0, sticky="w")

        self.lb_count = ttk.Label(bar, text="已收 0 帧", style="Info.TLabel",
                                  foreground="#455a64")
        self.lb_count.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))

        self.lb_key = ttk.Label(bar, text="", style="Info.TLabel")
        self.lb_key.grid(row=0, column=1, sticky="e")

        self.lb_sub = ttk.Label(bar, text="", foreground="#607d8b")
        self.lb_sub.grid(row=2, column=0, columnspan=2, sticky="w", pady=(2, 0))

    # ------------------------------------------------------------------ 左：参数
    def _build_left(self, host: str, port: int) -> None:
        f = ttk.Frame(self.root, padding=(10, 4, 6, 8), width=390)
        f.grid(row=1, column=0, sticky="nsew")
        f.grid_propagate(False)          # 左栏定宽
        f.columnconfigure(0, weight=1)

        g = ttk.LabelFrame(f, text=" 接收参数 ", padding=(10, 6, 10, 8))
        g.grid(row=0, column=0, sticky="ew")
        g.columnconfigure(1, weight=0, minsize=150)

        self.v_host = tk.StringVar(value=host)
        self.v_port = tk.StringVar(value=str(port))
        self.v_dt = tk.StringVar(value="0.002")
        self.v_follow = tk.BooleanVar(value=True)
        self.v_mode = tk.StringVar(value="known")
        self.v_R = tk.StringVar(value="10.0")
        # 判决工作点默认 0.73（不是内核的 0.72）：离 λc=0.7335 只剩 0.0035，
        # 目标一侧更陡、背景一侧仍稳在 0.23 以上，实测分离度不变（见 docstring）
        self.v_lam = tk.StringVar(value=LAM_DEFAULT)
        self.v_th = tk.StringVar(value="0.08")
        self.v_arr = tk.StringVar(value="3")
        self.v_est = tk.BooleanVar(value=False)      # 估幅 4 s/帧，连续接收时默认关

        r = 0
        ttk.Label(g, text="监听地址").grid(row=r, column=0, sticky="w", pady=3)
        hf = ttk.Frame(g)
        hf.grid(row=r, column=1, columnspan=2, sticky="w", padx=(6, 0))
        ttk.Entry(hf, textvariable=self.v_host, width=13).grid(row=0, column=0, sticky="w")
        ttk.Label(hf, text=":").grid(row=0, column=1, padx=2)
        ttk.Entry(hf, textvariable=self.v_port, width=6).grid(row=0, column=2)
        r += 1

        r = self._row(g, r, "采样间隔 dt", self.v_dt, "s")
        ttk.Checkbutton(g, text="跟随发送端的 fs（推荐）", variable=self.v_follow,
                        command=self._refresh_derived).grid(row=r, column=0, columnspan=3,
                                                            sticky="w", pady=(0, 4))
        r += 1

        ttk.Separator(g, orient="horizontal").grid(row=r, column=0, columnspan=3,
                                                   sticky="ew", pady=6)
        r += 1

        ttk.Label(g, text="R 模式").grid(row=r, column=0, sticky="w", pady=3)
        mf = ttk.Frame(g)
        mf.grid(row=r, column=1, columnspan=2, sticky="w", padx=(6, 0))
        for txt, val in (("已知 R（默认）", "known"),
                         ("跟随发送端", "hint"),
                         ("自动扫 R（较慢）", "auto")):
            ttk.Radiobutton(mf, text=txt, value=val, variable=self.v_mode,
                            command=self._refresh_derived).pack(anchor="w")
        r += 1

        r = self._row(g, r, "已知 R", self.v_R, "rad/s")
        r = self._row(g, r, "工作点 λ0", self.v_lam, "")
        r = self._row(g, r, "判态门限", self.v_th, "pstd")

        ttk.Label(g, text="阵列").grid(row=r, column=0, sticky="w", pady=3)
        af = ttk.Frame(g)
        af.grid(row=r, column=1, columnspan=2, sticky="w", padx=(6, 0))
        ttk.Radiobutton(af, text="三相位 ±120°", value="3",
                        variable=self.v_arr).pack(anchor="w")
        ttk.Radiobutton(af, text="单路 β=0°", value="1",
                        variable=self.v_arr).pack(anchor="w")
        r += 1

        ck = ttk.Frame(g)
        ck.grid(row=r, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Checkbutton(ck, text="估计幅值/相位（+4 s/帧）", variable=self.v_est).pack(side="left")
        r += 1

        b = ttk.Frame(f)
        b.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        self.bt_start = ttk.Button(b, text="开始监听", style="Go.TButton", command=self.start)
        self.bt_start.pack(side="left")
        self.bt_stop = ttk.Button(b, text="停止", command=self.stop, state="disabled")
        self.bt_stop.pack(side="left", padx=(6, 0))
        ttk.Button(b, text="清空", command=self.reset).pack(side="left", padx=(6, 0))

        self.lb_hint = ttk.Label(f, text="", foreground="#555555", justify="left",
                                 wraplength=360)
        self.lb_hint.grid(row=2, column=0, sticky="w", pady=(8, 0))

        b2 = ttk.Frame(f)
        b2.grid(row=3, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(b2, text="保存本帧 npz", command=self.save_npz).pack(side="left")

        box = ttk.LabelFrame(f, text=" 日志 ", padding=(6, 4))
        box.grid(row=4, column=0, sticky="nsew", pady=(8, 0))
        f.rowconfigure(4, weight=1)
        box.rowconfigure(0, weight=1)
        box.columnconfigure(0, weight=1)
        self.txt = tk.Text(box, height=6, wrap="none", font=("Consolas", 9),
                           background="#fbfbfb", relief="flat")
        self.txt.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(box, orient="vertical", command=self.txt.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.txt.configure(yscrollcommand=sb.set, state="disabled")

    def _row(self, parent, r: int, label: str, var, unit: str) -> int:
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=var, width=13).grid(row=r, column=1, sticky="w", padx=(6, 4))
        ttk.Label(parent, text=unit, foreground="#666666").grid(row=r, column=2, sticky="w")
        return r + 1

    # ------------------------------------------------------------------ 右：四联图
    def _build_plots(self) -> None:
        f = ttk.Frame(self.root, padding=(0, 4, 10, 10))
        f.grid(row=1, column=1, sticky="nsew")
        f.rowconfigure(0, weight=1)
        f.columnconfigure(0, weight=1)

        self.fig = plt.Figure(figsize=(9.4, 6.0), dpi=100, constrained_layout=True)
        # 同上：默认 pad 太窄，(b)(d) 的右边框会贴在画布边缘
        self.fig.set_constrained_layout_pads(w_pad=0.16, h_pad=0.08)
        gs = self.fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.40])
        self.ax_w = self.fig.add_subplot(gs[0, 0])
        self.ax_s = self.fig.add_subplot(gs[0, 1])
        self.ax_a = self.fig.add_subplot(gs[1, 0])
        self.ax_p = self.fig.add_subplot(gs[1, 1])
        self.ax_tl = self.fig.add_subplot(gs[2, :])       # 判决时序条（连续模式专用）
        # 画布放进一个 grid_propagate(False) 的容器再用 place 铺满：
        # DPI 150% 时 TkAgg 会把自己的请求高度顶到 1500+ px，直接 grid 会把
        # 整行撑爆（左栏日志被压到 26 px）。
        holder = ttk.Frame(f)
        holder.grid(row=0, column=0, sticky="nsew")
        holder.grid_propagate(False)
        holder.rowconfigure(0, weight=1)
        holder.columnconfigure(0, weight=1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=holder)
        self.canvas.get_tk_widget().place(x=0, y=0, relwidth=1, relheight=1)
        self._blank()

    def _blank(self) -> None:
        for ax, t in ((self.ax_w, "(a) 接收波形"), (self.ax_s, "(b) 频谱"),
                      (self.ax_a, "(c) 阵列 pstd"), (self.ax_p, "(d) 相图"),
                      (self.ax_tl, "(e) 判决时序")):
            ax.clear()
            ax.set_title(t, fontsize=10)
            ax.tick_params(labelsize=8)
        for ax in (self.ax_w, self.ax_s, self.ax_p, self.ax_tl):
            ax.text(0.5, 0.5, "等待数据…", ha="center", va="center",
                    color="#9e9e9e", fontsize=11, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        self.ax_a.text(0.5, 0.5, "等待数据…", ha="center", va="center",
                       color="#9e9e9e", fontsize=11, transform=self.ax_a.transAxes)
        self.ax_a.set_xticks([])
        self.ax_a.set_yticks([])
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ 参数
    def _f(self, var, name: str) -> float:
        try:
            return float(str(var.get()).strip())
        except (ValueError, tk.TclError):
            raise ValueError(f"「{name}」不是合法数字") from None

    def params(self) -> dict:
        p = dict(host=str(self.v_host.get()).strip(),
                 port=int(float(self.v_port.get())),
                 dt=self._f(self.v_dt, "采样间隔"),
                 follow=bool(self.v_follow.get()),
                 mode=self.v_mode.get(),
                 R=self._f(self.v_R, "手动 R"),
                 lam=self._f(self.v_lam, "λ0"),
                 th=self._f(self.v_th, "门限"),
                 n_arr=int(self.v_arr.get()),
                 do_est=bool(self.v_est.get()))
        if not (0 < p["port"] <= 65535):
            raise ValueError("端口须在 1–65535")
        if p["dt"] <= 0:
            raise ValueError("采样间隔必须为正")
        if p["mode"] in ("known", "manual") and p["R"] <= 0:
            raise ValueError("手动 R 必须为正")
        if p["th"] <= 0:
            raise ValueError("门限必须为正")
        return p

    def _refresh_derived(self) -> None:
        m = self.v_mode.get()
        self.lb_hint.configure(
            text=("已知 R：不扫频，约 2 s 出结果；R 与发送端 ω 差 5% 就判不出。"
                  if m == "known" else
                  "跟随发送端：用包头里的 ω 当 R，联调时不用两边各填一次。"
                  if m == "hint" else
                  "自动扫 R：只拿序列与 dt，FFT 粗估后 ±1% 两级细扫，约 6 s 一帧。"))

    # ------------------------------------------------------------------ 监听
    def start(self) -> None:
        if self.rx_thread and self.rx_thread.is_alive():
            return
        try:
            p = self.params()
        except ValueError as e:
            messagebox.showerror("参数有误", str(e))
            return
        self.stop_evt.clear()
        self.buf = None
        self.cur = p
        self.bt_start.configure(state="disabled")
        self.bt_stop.configure(state="normal")
        self.lb_verdict.configure(text="监听中…", foreground="#ef6c00")
        self.wk_thread = threading.Thread(target=self._worker, args=(p,), daemon=True)
        self.wk_thread.start()
        self.rx_thread = threading.Thread(target=self._recv, args=(p,), daemon=True)
        self.rx_thread.start()
        self._log(f"开始监听 {p['host']}:{p['port']}（R 模式 {p['mode']}）")

    def _recv(self, p: dict) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
        try:
            sock.bind((p["host"], p["port"]))
        except OSError as e:
            self.q.put(("err", f"绑定 {p['host']}:{p['port']} 失败：{e}"))
            return
        sock.settimeout(0.4)
        self.q.put(("info", f"已绑定 {p['host']}:{p['port']}，等待发送端…"))

        while not self.stop_evt.is_set():
            try:
                raw, addr = sock.recvfrom(65535)
            except socket.timeout:
                if self.buf and time.time() - self.buf.t_last > 3.0:
                    self.q.put(("warn", f"帧 {self.buf.meta.get('frame')} 超时丢弃"
                                        f"（已收 {self.buf.got}/{self.buf.n} 点）"))
                    self.buf = None
                continue
            except OSError:
                break

            kind, obj = LP.parse_packet(raw)
            if kind == "head":
                # 连续模式下帧是背靠背来的：旧帧没凑齐就被新帧顶掉，必报
                if self.buf is not None and not self.buf.complete:
                    self.q.put(("warn", f"帧 {self.buf.meta.get('frame')} 未收完即被"
                                        f"帧 {obj.get('frame')} 顶掉，丢 {self.buf.missing} 点"))
                k = obj.get("frame")
                if isinstance(k, int) and self.last_frame >= 0 and k > self.last_frame + 1:
                    self.q.put(("warn", f"帧号跳变：{self.last_frame} → {k}，"
                                        f"中间缺 {k - self.last_frame - 1} 帧"))
                self.buf = LP.FrameBuf(obj)
                self.q.put(("info", f"← 来自 {addr[0]}:{addr[1]} 帧 {k}，"
                                    f"n={obj.get('n')}，fs={obj.get('fs')}，"
                                    f"cov={float(obj.get('cov') or 0):.2f}，"
                                    f"SNR={obj.get('snr_db')}，"
                                    f"信号={'开' if obj.get('has_sig') else '关'} "
                                    f"噪声={'开' if obj.get('has_noise') else '关'}"))
            elif kind == "data":
                if self.buf is None or obj["frame"] != self.buf.meta.get("frame"):
                    continue                       # 丢了头包的残包，忽略
                if "raw" in obj:
                    self.buf.put_raw(obj["seq"], obj["raw"])
                else:
                    self.buf.put(obj["seq"], obj["x"], obj["nseq"])
                if self.buf.complete:
                    self._submit()
            elif kind == "tail":
                if self.buf and obj.get("frame") == self.buf.meta.get("frame"):
                    self._submit()

        sock.close()
        self.q.put(("stopped", None))

    def _submit(self) -> None:
        b, self.buf = self.buf, None
        if not b:
            return
        miss = b.n - b.got
        if miss > 0:
            self.q.put(("warn", f"帧 {b.meta.get('frame')} 缺 {miss} 点（{miss / b.n * 100:.1f}%），"
                                f"缺口补零后仍参与计算"))
        self.last_frame = int(b.meta.get("frame") or 0)
        # 分析比采集慢时（"尽快"模式尤其明显）丢最旧的帧，保最新——否则队列
        # 会无限增长，界面看到的永远是几分钟前的帧
        if self.frame_q.full():
            try:
                self.frame_q.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
            self.q.put(("warn", f"分析跟不上，丢弃旧帧（累计 {self.dropped} 帧）"))
        try:
            self.frame_q.put_nowait(dict(meta=b.meta, x=b.buf.copy()))
        except queue.Full:
            self.dropped += 1

    def _worker(self, p: dict) -> None:
        while not self.stop_evt.is_set():
            try:
                job = self.frame_q.get(timeout=0.4)
            except queue.Empty:
                continue
            meta, x = job["meta"], job["x"]
            p = dict(self.cur or p)         # 监听中改参数立刻生效
            dt = 1.0 / float(meta.get("fs") or 0) if p["follow"] else p["dt"]
            if not (dt > 0):
                dt = p["dt"]
            betas = LP.BETAS3 if p["n_arr"] == 3 else LP.BETAS3[:1]

            # 覆盖度自己重算，不照抄头包——发送端算错时这里能当场发现
            cov = LP.coverage_of(meta)
            rep = meta.get("cov")
            if rep is not None and abs(float(rep) - cov) > 1e-3:
                self.q.put(("warn", f"帧 {meta.get('frame')} 覆盖度不一致：发送端 "
                                    f"{float(rep):.3f}，本地重算 {cov:.3f}"))

            t0 = time.time()
            try:
                res = LP.analyze(x, dt, mode=p["mode"], R_manual=p["R"],
                                 w_hint=meta.get("w_hint"), lam=p["lam"], betas=betas,
                                 th=p["th"], do_est=p["do_est"])
            except Exception as e:                    # noqa: BLE001 —— 任何算错都要报到界面上
                self.q.put(("err", f"计算失败：{type(e).__name__}: {e}"))
                continue
            res.update(meta=meta, cost=time.time() - t0, dt=dt, cov=cov)
            self.q.put(("result", res))

    def stop(self) -> None:
        self.stop_evt.set()
        self.bt_stop.configure(state="disabled")
        self.lb_verdict.configure(text="已停止", foreground="#757575")
        self._log("停止监听")

    def reset(self) -> None:
        self.n_recv = self.n_hit = 0
        self.stats = LP.VerdictStats()
        self.hist = []
        self.dropped = 0
        self.last_frame = -1
        self._count()
        self._blank()
        self._log("已清空统计与时序条")

    # ------------------------------------------------------------------ 队列
    def _drain(self) -> None:
        try:
            while True:
                kind, obj = self.q.get_nowait()
                if kind == "result":
                    self._on_result(obj)
                elif kind == "info":
                    self._log(str(obj))
                elif kind == "warn":
                    self._log("⚠ " + str(obj))
                elif kind == "err":
                    self.lb_verdict.configure(text="出错", foreground="#c62828")
                    self.lb_sub.configure(text=str(obj))
                    self._log(str(obj))
                    self.bt_start.configure(state="normal")
                    self.bt_stop.configure(state="disabled")
                elif kind == "stopped":
                    self.bt_start.configure(state="normal")
        except queue.Empty:
            pass
        try:
            self.cur = self.params()        # 输入框非法时抛错，保留上一次有效值
        except (ValueError, tk.TclError):
            pass
        try:
            self._after_id = self.root.after(100, self._drain)
        except tk.TclError:
            pass

    def _count(self) -> None:
        self.lb_count.configure(text=f"已收 {self.n_recv} 帧　{self.stats.text()}")

    # ------------------------------------------------------------------ 出结果
    def _on_result(self, r: dict) -> None:
        meta = r["meta"]
        self.last_res = r
        self.n_recv += 1

        cov = float(r.get("cov") or 0.0)
        detected = bool(r["detected"]) if r.get("ok") else None
        cls = self.stats.add(cov, detected)
        if r.get("ok"):
            self.n_hit += int(detected)
        self._count()
        self.hist.append(dict(frame=int(meta.get("frame") or 0), t0=meta.get("t0"),
                              cov=cov, detected=detected,
                              pstd=r.get("pstd") if r.get("ok") else None))
        self.hist = self.hist[-120:]

        if not r.get("ok"):
            self.lb_verdict.configure(text="○ 无法判定", foreground="#757575")
            self.lb_key.configure(text="")
            self.lb_sub.configure(text=r.get("msg", ""))
            self._draw(r, meta)
            self._log(f"帧 {meta.get('frame')}：{r.get('msg')}")
            return

        hit = detected
        if hit:
            if cls == "bg":
                txt, col = "● 虚警：背景帧判有", "#ef6c00"
            elif cls == "part":
                txt, col = f"● 判有（部分覆盖 {cov * 100:.0f}%，不计统计）", "#ef6c00"
            else:
                txt, col = "● 检测到目标（目标窗内）", C_HIT
        else:
            if cls == "sig":
                txt, col = "○ 漏检：目标窗内判无", "#c62828"
            elif cls == "part":
                txt, col = f"○ 判无（部分覆盖 {cov * 100:.0f}%，不计统计）", "#757575"
            else:
                txt, col = "○ 未检测到目标（背景帧）", "#757575"
        self.lb_verdict.configure(text=txt, foreground=col)

        R_hat = r["R"]
        f_hat = R_hat / LP.TWOPI
        self.lb_key.configure(
            text=f"R_hat={R_hat:.4f} rad/s   min pstd={r['pstd']:.2e} "
                 f"{'<' if hit else '>='} {r['th']:.2g}")

        bits = [f"f_hat = {f_hat:.4f} Hz", f"cov = {cov:.2f}",
                f"耗时 {r['cost']:.1f} s"]
        # 真值只有发送端知道；仅在"确实发了信号"时才拿它对表，否则误差没意义
        true_w = meta.get("w_hint") if meta.get("has_sig") else None
        if true_w:
            bits.append(f"真值 {true_w:.4f}（误差 {(R_hat - true_w) / true_w * 100:+.3f}%）")
        if r.get("est"):
            e = r["est"]
            bits.append(f"gamma_hat = {e['gamma']:.4f}")
            bits.append(f"theta_hat = {math.degrees(e['theta']):+.1f}°")
            if meta.get("amp") and meta.get("has_sig"):
                bits.append(f"A = {meta['amp']:.4f}（误差 "
                            f"{(e['gamma'] - meta['amp']) / meta['amp'] * 100:+.1f}%）")
        self.lb_sub.configure(text="　·　".join(bits))

        self._draw(r, meta)

        ch = " ".join(f"{v:.2e}" for v in r["pstd_all"])
        self._log(f"帧 {meta.get('frame')}：{'有' if hit else '无'}目标 [{cls}]　"
                  f"cov={cov:.2f}　R_hat={R_hat:.4f}　pstd=[{ch}]　"
                  f"min={r['pstd']:.3e}　{r['cost']:.1f} s")

    # ------------------------------------------------------------------ 绘图
    def _draw(self, r: dict, meta: dict) -> None:
        x = r.get("x")
        fs = 1.0 / r["dt"] if r.get("dt") else 0.0

        # (a) 波形
        self.ax_w.clear()
        if x is not None and np.any(x):
            n = min(x.size, int(round(2.0 * fs)))
            self.ax_w.plot(np.arange(n) / fs, x[:n], color=C_WAVE, lw=0.9)
            self.ax_w.set_ylabel("幅值", fontsize=9)
        elif x is not None:
            self.ax_w.text(0.5, 0.5, "全零序列", ha="center", va="center",
                           color="#9e9e9e", transform=self.ax_w.transAxes)
        # 标题里点名「信号+噪声」：接收序列只有这两项，策动力是振子内驱，
        # 不参与收发（见 docs「接收序列口径」一节）
        self.ax_w.set_title(f"(a) 接收波形：信号+噪声（前 2 s，共 "
                            f"{0 if x is None else x.size} 点）", fontsize=10)
        self.ax_w.set_xlabel("时间 / s", fontsize=9)
        self.ax_w.tick_params(labelsize=8)
        self.ax_w.grid(alpha=0.25, color=C_GRID)

        # (b) 频谱
        self.ax_s.clear()
        if r.get("freq") is not None:
            freq, spec = r["freq"], r["spectrum"]
            band = freq <= 6.0
            self.ax_s.semilogy(freq[band], np.maximum(spec[band], 1e-12), color=C_SPEC, lw=0.9)
            if math.isfinite(r.get("w0") or float("nan")):
                self.ax_s.axvline(r["w0"] / LP.TWOPI, color=C_TH, ls="--", lw=1.1,
                                  label=f"FFT 初值 {r['w0'] / LP.TWOPI:.4f} Hz")
                self.ax_s.legend(fontsize=8)
            if math.isfinite(r.get("R") or float("nan")):
                self.ax_s.axvline(r["R"] / LP.TWOPI, color=C_HIT, ls=":", lw=1.3,
                                  label=f"R_hat {r['R'] / LP.TWOPI:.4f} Hz")
                self.ax_s.legend(fontsize=8)
        if r.get("freq") is not None:
            self.ax_s.set_xlim(0.0, 6.2)      # 留出右边距，否则 6 Hz 的刻度被裁半个
        self.ax_s.set_title("(b) 频谱", fontsize=10)
        self.ax_s.set_xlabel("频率 / Hz", fontsize=9)
        self.ax_s.set_ylabel("|X(f)|", fontsize=9)
        self.ax_s.tick_params(labelsize=8)
        self.ax_s.grid(alpha=0.25, color=C_GRID, which="both")

        # (c) 阵列 pstd
        self.ax_a.clear()
        pv = np.atleast_1d(np.asarray(r.get("pstd_all", []), float))
        if pv.size:
            th = r.get("th", LP.PSTD_TH)
            lbl = ([f"{d:+.0f}°" for d in (0, -120, 120)] if pv.size == 3
                   else ["β=0°"] * pv.size)
            cols = [C_HIT if (np.isfinite(v) and v < th) else C_MISS for v in pv]
            xs = np.arange(pv.size)
            self.ax_a.bar(xs, [max(v, 1e-7) if np.isfinite(v) else 1e-7 for v in pv],
                          color=cols, width=0.6)
            self.ax_a.set_yscale("log")
            self.ax_a.axhline(th, color=C_TH, ls="--", lw=1.3, label=f"门限 {th:g}")
            for xi, v in zip(xs, pv):
                if np.isfinite(v):
                    self.ax_a.annotate(f"{v:.1e}", (xi, max(v, 1e-7)),
                                       textcoords="offset points", xytext=(0, 4),
                                       ha="center", fontsize=8)
            self.ax_a.set_xticks(xs)
            self.ax_a.set_xticklabels(lbl, fontsize=9)
            self.ax_a.legend(fontsize=8, loc="lower right")
            # 纵轴上界留 3 倍余量，否则柱顶标注会压到标题上；无检出时把下界
            # 收窄，不然 1e-6 的下界会把 0.4 的柱子顶到天花板
            fin = pv[np.isfinite(pv)]
            if fin.size:
                pmin, pmax = float(fin.min()), float(fin.max())
                if pmin < th:                       # 有通道共振：要看到 1e-6 级的差别
                    lo, hi = 1e-6, max(pmax * 8.0, th * 2.0)
                else:                               # 全混沌：把下界收窄，柱顶标注才不顶到标题
                    lo, hi = max(pmin / 20.0, 1e-6), max(pmax * 3.0, 1e-3)
                self.ax_a.set_ylim(lo, hi)
        self.ax_a.set_title("(c) 阵列 pstd（低于门限 = 共振）", fontsize=10)
        self.ax_a.set_ylabel("pstd", fontsize=9)
        self.ax_a.tick_params(labelsize=8)
        self.ax_a.grid(alpha=0.25, color=C_GRID, axis="y")

        # (d) 相图
        self.ax_p.clear()
        tr = r.get("trace")
        if tr:
            _, X, Y = tr
            X = np.atleast_2d(X)
            Y = np.atleast_2d(Y)
            m = X.shape[1]
            tail = min(TRACE_PTS, X.shape[0])
            cols = [C_HIT, "#1e88e5", "#8e24aa"] if m >= 3 else [C_HIT]
            pv2 = np.atleast_1d(np.asarray(r.get("pstd_all", []), float))
            for i in range(m):
                c = cols[i % len(cols)]
                # 图例别写 "pstd="：字数一多，相图右边界会被图例顶出画布
                self.ax_p.plot(X[-tail:, i], Y[-tail:, i], color=c, lw=0.6, alpha=0.85,
                               label=f"β{i}={pv2[i]:.1e}" if i < pv2.size else f"β{i}")
            self.ax_p.legend(fontsize=6.5, loc="upper right", handlelength=1.2,
                             borderpad=0.3, labelspacing=0.25)
        self.ax_p.set_title("(d) 相图（闭合 = 周期）", fontsize=10)
        # 相图是振子状态 (x,y)，由内驱 lam·cos t 推动 —— 这条内驱不进接收序列，
        # 只在判决积分里加。写在 xlabel 里（轴内没空地，写了会压曲线）。
        self.ax_p.set_xlabel("x　（内驱 lam·cos t 不进序列）", fontsize=7.5)
        self.ax_p.set_ylabel("y", fontsize=9)
        self.ax_p.tick_params(labelsize=8)
        self.ax_p.grid(alpha=0.25, color=C_GRID)

        self._draw_timeline()
        self.canvas.draw_idle()

    def _draw_timeline(self) -> None:
        """(e) 判决时序：每帧一个色块，连续模式下一眼看目标窗何时起止。"""
        ax = self.ax_tl
        ax.clear()
        h = self.hist[-80:]
        if not h:
            ax.set_title("(e) 判决时序", fontsize=10)
            ax.text(0.5, 0.5, "等待数据…", ha="center", va="center",
                    color="#9e9e9e", fontsize=11, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            return

        for i, e in enumerate(h):
            cov, det = e["cov"], e["detected"]
            if det is None:
                face = "#b0bec5"
            elif cov >= LP.VerdictStats.COV_FULL:
                face = C_HIT if det else "#ef5350"          # 有 / 漏检
            elif cov <= LP.VerdictStats.COV_NONE:
                face = "#e0e0e0" if not det else "#ff9800"  # 背景 / 虚警
            else:
                face = "#ffe082"                            # 部分覆盖
            ax.add_patch(Rectangle((i, 0), 0.92, 1.0, facecolor=face,
                                   edgecolor="#78909c", lw=0.35))
        ax.set_xlim(-0.5, len(h) - 0.5)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        step = max(1, len(h) // 10)
        ticks = list(range(0, len(h), step))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(h[i]["frame"]) for i in ticks], fontsize=7)
        ax.set_xlabel("帧号　　绿=判有　红=漏检　灰=背景　橙=虚警　黄=部分覆盖", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title("(e) 判决时序（每格一帧）", fontsize=9)
        ax.grid(alpha=0.2, color=C_GRID, axis="x")

    # ------------------------------------------------------------------ 存盘
    def save_npz(self) -> None:
        if not self.last_res or self.last_res.get("x") is None:
            messagebox.showinfo("没有数据", "还没有收到任何一帧。")
            return
        r = self.last_res
        out = Path(__file__).resolve().parent.parent / "复现输出" / "接收端-最新一帧.npz"
        out.parent.mkdir(exist_ok=True)
        np.savez_compressed(out, x=r["x"], dt=r["dt"], R=r["R"],
                            pstd_all=r["pstd_all"], detected=r["detected"],
                            cov=r.get("cov", 0.0), t0=r["meta"].get("t0"),
                            frame=r["meta"].get("frame"))
        self._log(f"已保存 {out}")

    # ------------------------------------------------------------------ 杂项
    def _log(self, msg: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.txt.configure(state="normal")
        self.txt.insert("end", f"[{stamp}] {msg}\n")
        self.txt.see("end")
        self.txt.configure(state="disabled")

    def _on_close(self) -> None:
        self.stop_evt.set()
        try:
            self.root.after_cancel(self._after_id)
        except (tk.TclError, AttributeError):
            pass
        for th in (self.rx_thread, self.wk_thread):
            if th:
                th.join(timeout=1.0)
        plt.close(self.fig)
        self.root.destroy()


def main() -> int:
    ap = argparse.ArgumentParser(description="接收端 GUI（监听 UDP 并判定有无目标）")
    ap.add_argument("--host", default="0.0.0.0", help="监听地址（0.0.0.0 收所有网卡）")
    ap.add_argument("--port", type=int, default=LP.PORT_DEFAULT, help="监听端口")
    ap.add_argument("--mode", default="known", choices=["known", "hint", "auto", "manual"],
                    help="R 模式：known=已知 R（默认），hint=跟随发送端，auto=盲测扫 R")
    args = ap.parse_args()
    root = tk.Tk()
    app = RxApp(root, args.host, args.port)
    app.v_mode.set("known" if args.mode == "manual" else args.mode)
    app._refresh_derived()
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
