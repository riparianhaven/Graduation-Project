#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""接收端 GUI：监听 UDP，收到一帧后判定「有没有目标」，并显示阵列图与波形图。

启动
    .venv\\Scripts\\python.exe tools\\rx_app.py
    .venv\\Scripts\\python.exe tools\\rx_app.py --port 50500 --host 0.0.0.0

界面能配什么
    网络：监听地址与端口
    时基：采样间隔 dt（勾「跟随发送端」时取头包里的 1/fs）
    定标：R 模式三选一
           自动    FFT 粗估 → ±1% 两级细扫 R（盲测，正常流程）
           手动    直接填 R（演示"R 不对会怎样"）
           提示    用头包里的 w_hint（联调对表旁路，正常检测不用）
    判据：λ0、判态门限 pstd_th、阵列（三相位 / 单路）、是否估幅值

界面显示什么
    波形图    接收序列（时域，前 2 s）+ 频谱（标出粗估频率）
    阵列图    三路 pstd 柱状（对数轴）+ 门限线；三路相轨迹
    状态条    判决、ω̂、R̂、min pstd、累计帧数与检出数

数据交互
    UDP 收包，按 frame → seq 顺序装配成整帧；收到尾包或凑够 n 点即提交。
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


class RxApp:
    def __init__(self, root: tk.Tk, host: str, port: int):
        self.root = root
        self.q: queue.Queue = queue.Queue()          # 工作线程 → UI
        self.frame_q: queue.Queue = queue.Queue()    # 接收线程 → 工作线程
        self.stop_evt = threading.Event()
        self.rx_thread: threading.Thread | None = None
        self.wk_thread: threading.Thread | None = None
        self.buf: LP.FrameBuf | None = None
        self.n_recv = 0
        self.n_hit = 0
        self.last_res: dict | None = None
        self.cur: dict | None = None        # 主线程刷新的参数快照（工作线程每帧读它）

        root.title("接收端 RX — 混沌振子阵列检测")
        root.geometry("1330x950")
        root.minsize(1160, 720)
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

        self.lb_key = ttk.Label(bar, text="", style="Info.TLabel")
        self.lb_key.grid(row=0, column=1, sticky="w", padx=(16, 0))

        self.lb_count = ttk.Label(bar, text="已收 0 帧 / 判有 0 帧",
                                  style="Info.TLabel", foreground="#455a64")
        self.lb_count.grid(row=0, column=2, sticky="e")

        self.lb_sub = ttk.Label(bar, text="", foreground="#607d8b")
        self.lb_sub.grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))

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
        self.v_lam = tk.StringVar(value=f"{LP.LAM0:.2f}")
        self.v_th = tk.StringVar(value="0.08")
        self.v_arr = tk.StringVar(value="3")
        self.v_est = tk.BooleanVar(value=True)
        self.v_auto = tk.BooleanVar(value=True)

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
        ttk.Checkbutton(ck, text="估计幅值/相位", variable=self.v_est).pack(side="left")
        ttk.Checkbutton(ck, text="每帧自动重算",
                        variable=self.v_auto).pack(side="left", padx=(14, 0))
        r += 1

        b = ttk.Frame(f)
        b.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        self.bt_start = ttk.Button(b, text="开始监听", style="Go.TButton", command=self.start)
        self.bt_start.pack(side="left")
        self.bt_stop = ttk.Button(b, text="停止", command=self.stop, state="disabled")
        self.bt_stop.pack(side="left", padx=(6, 0))
        ttk.Button(b, text="清空统计", command=self.reset).pack(side="left", padx=(6, 0))

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
        self.txt = tk.Text(box, height=7, wrap="none", font=("Consolas", 9),
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

        self.fig = plt.Figure(figsize=(9.4, 6.2), dpi=100, constrained_layout=True)
        gs = self.fig.add_gridspec(2, 2)
        self.ax_w = self.fig.add_subplot(gs[0, 0])
        self.ax_s = self.fig.add_subplot(gs[0, 1])
        self.ax_a = self.fig.add_subplot(gs[1, 0])
        self.ax_p = self.fig.add_subplot(gs[1, 1])
        self.canvas = FigureCanvasTkAgg(self.fig, master=f)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._blank()

    def _blank(self) -> None:
        for ax, t in ((self.ax_w, "(a) 接收波形"), (self.ax_s, "(b) 频谱"),
                      (self.ax_a, "(c) 阵列 pstd"), (self.ax_p, "(d) 相图")):
            ax.clear()
            ax.set_title(t, fontsize=10)
            ax.tick_params(labelsize=8)
        for ax in (self.ax_w, self.ax_s, self.ax_p):
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
            text=("已知 R：不扫频，约 2 s 出结果。R 必须与发送端的 ω 一致，"
                  "偏 5% 就判不出来。"
                  if m == "known" else
                  "跟随发送端：用包头里的 ω 当 R，联调时不用两边各填一次。"
                  if m == "hint" else
                  "自动扫 R：只拿序列与 dt，FFT 粗估后在 ±1% 内两级细扫；"
                  "这是论文的盲测流程，约 6 s 一帧。"))

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
                self.buf = LP.FrameBuf(obj)
                self.q.put(("info", f"← 来自 {addr[0]}:{addr[1]} 帧 {obj.get('frame')}，"
                                    f"n={obj.get('n')}，fs={obj.get('fs')}，"
                                    f"SNR={obj.get('snr_db')}，"
                                    f"信号={'开' if obj.get('has_sig') else '关'} "
                                    f"噪声={'开' if obj.get('has_noise') else '关'}"))
            elif kind == "data":
                if self.buf is None or obj["frame"] != self.buf.meta.get("frame"):
                    continue                       # 丢了头包的残包，忽略
                self.buf.put(obj["seq"], obj["x"], obj["nseq"])
                if self.buf.complete:
                    self._submit()
            elif kind == "tail":
                if self.buf:
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
        self.frame_q.put(dict(meta=b.meta, x=b.buf.copy()))

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
            t0 = time.time()
            try:
                res = LP.analyze(x, dt, mode=p["mode"], R_manual=p["R"],
                                 w_hint=meta.get("w_hint"), lam=p["lam"], betas=betas,
                                 th=p["th"], do_est=p["do_est"])
            except Exception as e:                    # noqa: BLE001 —— 任何算错都要报到界面上
                self.q.put(("err", f"计算失败：{type(e).__name__}: {e}"))
                continue
            res.update(meta=meta, cost=time.time() - t0, dt=dt)
            self.q.put(("result", res))

    def stop(self) -> None:
        self.stop_evt.set()
        self.bt_stop.configure(state="disabled")
        self.lb_verdict.configure(text="已停止", foreground="#757575")
        self._log("停止监听")

    def reset(self) -> None:
        self.n_recv = self.n_hit = 0
        self._count()

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
        rate = (self.n_hit / self.n_recv * 100.0) if self.n_recv else 0.0
        self.lb_count.configure(text=f"已收 {self.n_recv} 帧 / 判有 {self.n_hit} 帧"
                                      f"（{rate:.0f}%）")

    # ------------------------------------------------------------------ 出结果
    def _on_result(self, r: dict) -> None:
        meta = r["meta"]
        self.last_res = r
        self.n_recv += 1
        if r.get("ok"):
            self.n_hit += int(r["detected"])
            self._count()

        if not r.get("ok"):
            self.lb_verdict.configure(text="○ 无法判定", foreground="#757575")
            self.lb_key.configure(text="")
            self.lb_sub.configure(text=r.get("msg", ""))
            self._draw(r, meta)
            self._log(f"帧 {meta.get('frame')}：{r.get('msg')}")
            return

        hit = r["detected"]
        self.lb_verdict.configure(text="● 检测到目标" if hit else "○ 未检测到目标",
                                  foreground=C_HIT if hit else "#757575")
        w_hat, R_hat = r.get("w0", float("nan")), r["R"]
        f_hat = R_hat / LP.TWOPI
        self.lb_key.configure(
            text=f"R_hat={R_hat:.4f} rad/s   min pstd={r['pstd']:.2e} "
                 f"{'<' if hit else '>='} {r['th']:.2g}")

        bits = [f"f_hat = {f_hat:.4f} Hz", f"耗时 {r['cost']:.1f} s"]
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
        self._log(f"帧 {meta.get('frame')}：{'有' if hit else '无'}目标　"
                  f"R_hat={R_hat:.4f}　pstd=[{ch}]　min={r['pstd']:.3e}　"
                  f"{r['cost']:.1f} s")

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
        self.ax_w.set_title(f"(a) 接收波形（前 2 s，共 {0 if x is None else x.size} 点）",
                            fontsize=10)
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
            self.ax_a.set_ylim(1e-7, max(1.0, float(np.nanmax(pv)) * 3))
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
                self.ax_p.plot(X[-tail:, i], Y[-tail:, i], color=c, lw=0.6, alpha=0.85,
                               label=f"β{i} pstd={pv2[i]:.1e}" if i < pv2.size else f"β{i}")
            self.ax_p.legend(fontsize=7, loc="upper right")
        self.ax_p.set_title("(d) 相图（闭合 = 周期）", fontsize=10)
        self.ax_p.set_xlabel("x", fontsize=9)
        self.ax_p.set_ylabel("y", fontsize=9)
        self.ax_p.tick_params(labelsize=8)
        self.ax_p.grid(alpha=0.25, color=C_GRID)

        self.canvas.draw_idle()

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
