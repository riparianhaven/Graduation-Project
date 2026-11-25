#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""发送端 GUI：模拟目标发出的极低频磁场，通过 UDP 打给接收端。

启动
    .venv\\Scripts\\python.exe tools\\tx_app.py
    .venv\\Scripts\\python.exe tools\\tx_app.py --host 192.168.1.20 --port 50500

界面能配什么
    信号：载波角频率 ω、SNR(dB)、基线噪声 σ、采样率 fs、发送时长
    开关：**信号**与**噪声**各自独立开关（都关 = 全零帧，用来测"无目标/静默"）
    发送：帧数、随机种子、节奏（尽快 / 实时×倍率）、目标地址与端口

发送什么
    一帧 = A·cos(ωt+φ)（φ 每次随机，接收端不知道）→ 乘信道衰减 → 加噪声
    噪声每样本标准差 = σ·√(ω/fs)（力口径，与阶段 2–5 一致）。
    A 由 SNR 反算：A = σ·√(2·10^(SNR/10))；关掉噪声时 SNR 无意义，A 仍按
    同一个 σ 参考给出，便于和"有噪声"的条件对照。

数据交互（详见 tools/link_proto.py 顶部）
    UDP 127.0.0.1:50500，一帧 = 1 个 JSON 头包 + ceil(n/1024) 个 float32 数据包
    + 1 个 JSON 尾包。头包里的 n / fs 让接收端知道要收多少点、采样间隔多少。
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
C_GRID = "#cccccc"


class TxApp:
    def __init__(self, root: tk.Tk, host: str, port: int):
        self.root = root
        self.q: queue.Queue = queue.Queue()
        self.stop_evt = threading.Event()
        self.worker: threading.Thread | None = None
        self.sent_frames = 0

        root.title("发送端 TX — 极低频磁场目标模拟")
        root.geometry("1370x910")
        root.minsize(1250, 800)
        style = ttk.Style()
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass
        style.configure(".", font=("Microsoft YaHei UI", 9))
        style.configure("Big.TLabel", font=("Microsoft YaHei UI", 11, "bold"))
        style.configure("Go.TButton", font=("Microsoft YaHei UI", 10, "bold"))

        root.columnconfigure(0, weight=0, minsize=468)
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
        f = ttk.Frame(self.root, padding=(10, 8, 6, 8), width=460)
        f.grid(row=0, column=0, sticky="nsew")
        f.grid_propagate(False)          # 左栏定宽：不让长说明文字把面板撑宽、把右图挤扁
        f.columnconfigure(0, weight=1)

        # ---- 信号参数
        g = ttk.LabelFrame(f, text=" 信号参数 ", padding=(10, 6, 10, 8))
        g.grid(row=0, column=0, sticky="ew")
        g.columnconfigure(1, weight=0, minsize=150)

        self.v_w = tk.StringVar(value="10.0")
        self.v_snr = tk.DoubleVar(value=-11.0)
        self.v_sigma = tk.StringVar(value="0.05")
        self.v_fs = tk.StringVar(value="500")
        self.v_dur = tk.StringVar(value="94.2")
        self.v_alpha = tk.StringVar(value="0")
        self.v_sig = tk.BooleanVar(value=True)
        self.v_noise = tk.BooleanVar(value=True)

        r = 0
        r = self._row(g, r, "载波角频率 ω", self.v_w, "rad/s")
        self.lb_f0 = ttk.Label(g, text="", foreground="#555555", wraplength=210)
        self.lb_f0.grid(row=r, column=1, sticky="w", pady=(0, 4))
        r += 1

        ttk.Label(g, text="信噪比 SNR").grid(row=r, column=0, sticky="w", pady=3)
        self.sc_snr = ttk.Scale(g, from_=-30.0, to=5.0, variable=self.v_snr, length=152,
                                orient="horizontal", command=lambda _=None: self._refresh_derived())
        self.sc_snr.grid(row=r, column=1, columnspan=2, sticky="w", padx=(6, 4))
        r += 1

        self.lb_amp = ttk.Label(g, text="", foreground="#555555", wraplength=250)
        self.lb_amp.grid(row=r, column=1, sticky="w", pady=(0, 4))
        r += 1

        r = self._row(g, r, "基线噪声 σ", self.v_sigma, "")
        r = self._row(g, r, "采样率 fs", self.v_fs, "Hz")
        r = self._row(g, r, "发送时长", self.v_dur, "s")
        self.lb_n = ttk.Label(g, text="", foreground="#555555", wraplength=210)
        self.lb_n.grid(row=r, column=1, sticky="w", pady=(0, 4))
        r += 1

        ttk.Label(g, text="噪声类型").grid(row=r, column=0, sticky="w", pady=3)
        nf = ttk.Frame(g)
        nf.grid(row=r, column=1, columnspan=2, sticky="w", padx=(6, 0))
        ttk.Radiobutton(nf, text="白噪", value="0", variable=self.v_alpha).pack(side="left")
        ttk.Radiobutton(nf, text="1/f 色噪", value="1", variable=self.v_alpha).pack(side="left", padx=(10, 0))
        r += 1

        sw = ttk.Frame(g)
        sw.grid(row=r, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Checkbutton(sw, text="发送信号", variable=self.v_sig,
                        command=self._refresh_derived).pack(side="left")
        ttk.Checkbutton(sw, text="发送噪声", variable=self.v_noise,
                        command=self._refresh_derived).pack(side="left", padx=(16, 0))

        # ---- 发送参数
        h = ttk.LabelFrame(f, text=" 发送参数 ", padding=(10, 6, 10, 8))
        h.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        h.columnconfigure(1, weight=0, minsize=150)

        self.v_frames = tk.StringVar(value="1")
        self.v_seed = tk.StringVar(value="0")
        self.v_rate = tk.StringVar(value="fast")
        self.v_speed = tk.StringVar(value="10")
        self.v_host = tk.StringVar(value=host)
        self.v_port = tk.StringVar(value=str(port))

        r = 0
        r = self._row(h, r, "连发帧数", self.v_frames, "帧")
        r = self._row(h, r, "随机种子", self.v_seed, "0=每次随机")

        ttk.Label(h, text="发送节奏").grid(row=r, column=0, sticky="w", pady=3)
        rf = ttk.Frame(h)
        rf.grid(row=r, column=1, columnspan=2, sticky="w", padx=(6, 0))
        ttk.Radiobutton(rf, text="尽快", value="fast", variable=self.v_rate,
                        command=self._refresh_derived).pack(side="left")
        ttk.Radiobutton(rf, text="实时", value="realtime", variable=self.v_rate,
                        command=self._refresh_derived).pack(side="left", padx=(10, 0))
        ttk.Label(rf, text="×").pack(side="left", padx=(10, 2))
        self.en_speed = ttk.Entry(rf, textvariable=self.v_speed, width=4)
        self.en_speed.pack(side="left")
        r += 1

        ttk.Label(h, text="目标地址").grid(row=r, column=0, sticky="w", pady=3)
        hf = ttk.Frame(h)
        hf.grid(row=r, column=1, columnspan=2, sticky="w", padx=(6, 0))
        ttk.Entry(hf, textvariable=self.v_host, width=10).grid(row=0, column=0, sticky="w")
        ttk.Label(hf, text=":").grid(row=0, column=1, padx=2)
        ttk.Entry(hf, textvariable=self.v_port, width=5).grid(row=0, column=2)

        # ---- 联调提示
        tip = ttk.Label(f, text="先开接收端，再点「开始发送」。接收端默认用已知 R，也可改成扫 R。",
                        foreground="#777777", justify="left", wraplength=330)
        tip.grid(row=2, column=0, sticky="w", pady=(10, 0))
        f.rowconfigure(2, weight=1)

    def _row(self, parent, r: int, label: str, var, unit: str) -> int:
        """一行 = 标签（单位并在括号里）+ 输入框。单位不单独占列，省得挤。"""
        txt = f"{label} ({unit})" if unit else label
        ttk.Label(parent, text=txt).grid(row=r, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=var, width=13).grid(row=r, column=1, sticky="w", padx=(6, 4))
        return r + 1

    # ------------------------------------------------------------------ 右：图
    def _build_right(self) -> None:
        f = ttk.Frame(self.root, padding=(0, 8, 10, 4))
        f.grid(row=0, column=1, sticky="nsew")
        f.rowconfigure(0, weight=1)
        f.rowconfigure(1, weight=0)   # 图吃掉多余高度，日志按自身请求高度
        f.columnconfigure(0, weight=1)

        self.fig = plt.Figure(figsize=(8.6, 4.8), dpi=100, constrained_layout=True)
        self.ax_t = self.fig.add_subplot(1, 2, 1)
        self.ax_f = self.fig.add_subplot(1, 2, 2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=f)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

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
        self.bt_send = ttk.Button(b, text="开始发送", style="Go.TButton", command=self.start)
        self.bt_send.grid(row=0, column=1, padx=(0, 6))
        self.bt_stop = ttk.Button(b, text="停止", command=self.stop, state="disabled")
        self.bt_stop.grid(row=0, column=2)
        ttk.Button(b, text="打开接收端", command=self.launch_rx).grid(row=0, column=4, padx=(6, 0))

        self.lb_stat = ttk.Label(b, text="就绪", style="Big.TLabel", foreground="#2e7d32")
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
            duration=self._f(self.v_dur, "发送时长"),
            alpha=float(self.v_alpha.get()),
            has_sig=bool(self.v_sig.get()),
            has_noise=bool(self.v_noise.get()),
            frames=int(float(self.v_frames.get())),
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
            raise ValueError("发送时长必须为正")
        if p["sigma"] < 0:
            raise ValueError("基线噪声不能为负")
        if p["frames"] < 1:
            raise ValueError("帧数至少为 1")
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
        self.en_speed.configure(state="normal" if self.v_rate.get() == "realtime" else "disabled")

    # ------------------------------------------------------------------ 预览
    def preview(self) -> None:
        try:
            p = self.params()
        except ValueError as e:
            messagebox.showerror("参数有误", str(e))
            return
        rng = np.random.default_rng(p["seed"] if p["seed"] else None)
        fr = LP.make_frame(w=p["w"], fs=p["fs"], duration=p["duration"], amp=p["amp"],
                           sigma=p["sigma"], alpha=p["alpha"], has_sig=p["has_sig"],
                           has_noise=p["has_noise"], rng=rng)
        self._draw(fr, p)
        snr_txt = "—" if fr["snr_db"] is None else f"{fr['snr_db']:+.2f} dB"
        self._log(f"预览：n={fr['n']}，f0={p['w'] / LP.TWOPI:.4f} Hz，"
                  f"A={p['amp']:.4f}，σ={p['sigma']}，SNR={snr_txt}")

    def _draw(self, fr: dict, p: dict) -> None:
        t = np.arange(fr["n"]) / fr["fs"]
        show = min(fr["n"], int(round(2.0 * fr["fs"])))       # 时域只画前 2 s
        self.ax_t.clear()
        self.ax_t.plot(t[:show], fr["tx"][:show], color=C_TX, lw=1.2, label="发射（干净）")
        self.ax_t.plot(t[:show], fr["rx"][:show], color=C_RX, lw=0.9, alpha=0.8, label="到达接收端")
        self.ax_t.set(title="发射与接收波形（前 2 s）", xlabel="时间 / s", ylabel="幅值（归一化策动力）")
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
            self.ax_f.set(title="到达信号频谱", xlabel="频率 / Hz", ylabel="|X(f)|")
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
        self.bt_send.configure(state="disabled")
        self.bt_stop.configure(state="normal")
        self.lb_stat.configure(text="发送中…", foreground="#ef6c00")
        self.worker = threading.Thread(target=self._worker, args=(p,), daemon=True)
        self.worker.start()

    def _worker(self, p: dict) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
        addr = (p["host"], p["port"])
        rng = np.random.default_rng(p["seed"] if p["seed"] else None)
        gap = 0.002 if p["rate"] == "fast" else (LP.CHUNK / p["fs"]) / p["speed"]

        try:
            for k in range(p["frames"]):
                if self.stop_evt.is_set():
                    break
                fr = LP.make_frame(w=p["w"], fs=p["fs"], duration=p["duration"],
                                   amp=p["amp"], sigma=p["sigma"], alpha=p["alpha"],
                                   has_sig=p["has_sig"], has_noise=p["has_noise"], rng=rng)
                meta = dict(frame=k, n=fr["n"], fs=p["fs"], w_hint=p["w"],
                            amp=p["amp"], sigma=p["sigma"], alpha=p["alpha"],
                            snr_db=None if fr["snr_db"] is None else round(fr["snr_db"], 3),
                            has_sig=p["has_sig"], has_noise=p["has_noise"],
                            duration=round(fr["n"] / p["fs"], 4),
                            phase=round(fr["phase"], 5), ts=time.time(),
                            host=p["host"], port=p["port"])
                sock.sendto(LP.pack_head(meta), addr)
                for pkt in LP.split_frame(k, fr["rx"], p["fs"]):
                    if self.stop_evt.is_set():
                        break
                    sock.sendto(pkt, addr)
                    time.sleep(gap)
                nseq = int(math.ceil(fr["n"] / LP.CHUNK))
                sock.sendto(LP.pack_tail(dict(frame=k, sent=fr["n"], nseq=nseq)), addr)

                self.sent_frames += 1
                self.q.put(("frame", dict(p=p, fr=fr, meta=meta, total=p["frames"])))
        except OSError as e:
            self.q.put(("err", f"发送失败：{e}"))
        finally:
            sock.close()
            self.q.put(("done", self.sent_frames))

    def stop(self) -> None:
        self.stop_evt.set()
        self._log("已请求停止…")

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

    # ------------------------------------------------------------------ 队列与日志
    def _drain(self) -> None:
        try:
            while True:
                kind, obj = self.q.get_nowait()
                if kind == "frame":
                    p, fr, meta = obj["p"], obj["fr"], obj["meta"]
                    self._draw(fr, p)
                    snr = meta["snr_db"]
                    self._log(f"[帧 {meta['frame'] + 1}/{obj['total']}] n={meta['n']} "
                              f"φ={meta['phase']:+.3f} rad　"
                              f"SNR={'—' if snr is None else f'{snr:+.2f} dB'}　"
                              f"信号={'开' if p['has_sig'] else '关'} "
                              f"噪声={'开' if p['has_noise'] else '关'}")
                    self.lb_stat.configure(
                        text=f"已发送 {self.sent_frames}/{obj['total']} 帧　"
                             f"{p['host']}:{p['port']}", foreground="#ef6c00")
                elif kind == "done":
                    self.bt_send.configure(state="normal")
                    self.bt_stop.configure(state="disabled")
                    self.lb_stat.configure(text=f"发送完成：{obj} 帧", foreground="#2e7d32")
                    self._log(f"发送完成，共 {obj} 帧")
                elif kind == "err":
                    self.bt_send.configure(state="normal")
                    self.bt_stop.configure(state="disabled")
                    self.lb_stat.configure(text="发送失败", foreground="#c62828")
                    self._log(str(obj))
        except queue.Empty:
            pass
        try:
            self._after_id = self.root.after(120, self._drain)
        except tk.TclError:
            pass

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
        if self.worker:
            self.worker.join(timeout=1.0)
        plt.close(self.fig)
        self.root.destroy()


def main() -> int:
    ap = argparse.ArgumentParser(description="发送端 GUI（UDP 打一帧帧的含噪载波）")
    ap.add_argument("--host", default="127.0.0.1", help="目标地址（默认本机回环）")
    ap.add_argument("--port", type=int, default=LP.PORT_DEFAULT, help="目标端口")
    args = ap.parse_args()
    root = tk.Tk()
    TxApp(root, args.host, args.port)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
