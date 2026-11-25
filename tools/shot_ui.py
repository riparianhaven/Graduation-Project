#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""给双端界面各截一张图（写文档用，不进测试集）。

    .venv\\Scripts\\python.exe .workbuddy/tmp/shot_ui.py

输出：复现输出/界面-发送端.png、复现输出/界面-接收端.png
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes
import socket
import sys
import threading
import time
import tkinter as tk
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402
from PIL import ImageGrab  # noqa: E402

import link_proto as LP  # noqa: E402
import rx_app  # noqa: E402
import tx_app  # noqa: E402

# 关掉 DPI 虚拟化：否则 GetWindowRect 给逻辑像素、ImageGrab 按物理像素截，
# 结果只截到窗口左上 2/3（150% 缩放下这个坑很显眼）。
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:                                    # 老系统没有 shcore
    ctypes.windll.user32.SetProcessDPIAware()

OUT = ROOT / "复现输出"
OUT.mkdir(exist_ok=True)


def window_rect(root: tk.Tk) -> tuple[int, int, int, int]:
    """窗口外框的物理像素矩形（含标题栏与边框）。

    不能直接用 winfo_rootx/winfo_width：系统 DPI 150% 时 Tk 报的是逻辑像素，
    而 ImageGrab 按物理像素截，结果只截到窗口左上 2/3。
    """
    hwnd = int(root.wm_frame(), 16)
    r = ctypes.wintypes.RECT()
    ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(r))
    return r.left, r.top, r.right, r.bottom


def grab(root: tk.Tk, name: str) -> None:
    root.attributes("-topmost", True)      # 否则会被其他窗口盖住，截出半张
    root.lift()
    root.focus_force()
    root.update_idletasks()
    root.update()
    time.sleep(0.6)
    root.update()
    x0, y0, x1, y1 = window_rect(root)
    img = ImageGrab.grab(bbox=(x0, y0, x1, y1), all_screens=True)
    p = OUT / name
    img.save(p)
    print(f"  已存 {p}  {img.size}")


def shot_tx() -> None:
    root = tk.Tk()
    root.geometry("+16+8")
    app = tx_app.TxApp(root, "127.0.0.1", LP.PORT_DEFAULT)
    app.v_dur.set("94.2")
    app.v_snr.set(-7.4)
    app._refresh_derived()
    app.preview()
    for _ in range(4):
        root.update()
        time.sleep(0.08)
    grab(root, "界面-发送端.png")
    app._on_close()


def shot_rx(port: int = 50790) -> None:
    root = tk.Tk()
    root.geometry("+16+8")
    app = rx_app.RxApp(root, "127.0.0.1", port)
    app.v_port.set(str(port))
    app.v_R.set("10.0")
    app.v_est.set(True)
    app.start()
    for _ in range(8):
        root.update()
        time.sleep(0.05)

    fr = LP.make_frame(w=10.0, fs=500.0, duration=94.2, amp=0.02, sigma=0.05,
                       phase=0.0, rng=np.random.default_rng(4))

    def send() -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.sendto(LP.pack_head(dict(frame=0, n=fr["n"], fs=500.0, w_hint=10.0,
                                   amp=0.02, sigma=0.05, alpha=0, has_sig=True,
                                   has_noise=True, snr_db=round(fr["snr_db"], 2))),
                 ("127.0.0.1", port))
        for p in LP.split_frame(0, fr["rx"], 500.0):
            s.sendto(p, ("127.0.0.1", port))
            time.sleep(0.002)
        s.sendto(LP.pack_tail(dict(frame=0, sent=fr["n"])), ("127.0.0.1", port))
        s.close()

    threading.Thread(target=send, daemon=True).start()
    t0 = time.time()
    while time.time() - t0 < 25 and app.last_res is None:
        root.update()
        time.sleep(0.03)
    for _ in range(6):
        root.update()
        time.sleep(0.08)
    grab(root, "界面-接收端.png")
    app._on_close()


if __name__ == "__main__":
    print("截发送端…")
    shot_tx()
    print("截接收端…")
    shot_rx()
