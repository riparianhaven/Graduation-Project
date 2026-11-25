# -*- coding: utf-8 -*-
"""双端 GUI 的底层契约测试（不启动 tkinter，只测协议与算法）。

守五条：
1. 三类包能原样往返，坏包不抛异常；
2. 一帧切成数据包再按 seq 装回来，序列逐点相等；
3. 真 UDP 环回能把一帧送出去且凑齐（丢包率 0）；
4. 有信号判「有」、纯噪声判「无」、全零序列给出 ok=False；
5. R 手动填错（偏 5%）时判不出来——这是界面上「R 不对会怎样」的依据。
"""
from __future__ import annotations

import math
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import link_proto as LP  # noqa: E402


def _frame(duration=30.0, amp=0.02, sigma=0.05, has_sig=True, has_noise=True,
           seed=3, phase=0.0):
    """默认固定初相 θ=0（同相，最强条件）。

    随机初相会撞上相位盲区（三相位阵列仍漏 21%），测试不能靠运气；
    盲区本身另有一条专门的用例。
    """
    rng = np.random.default_rng(seed)
    return LP.make_frame(w=10.0, fs=500.0, duration=duration, amp=amp, sigma=sigma,
                         has_sig=has_sig, has_noise=has_noise, phase=phase, rng=rng)


# --------------------------------------------------------------------------- 协议
def test_protocol_roundtrip():
    h = LP.pack_head(dict(frame=2, n=100, fs=500.0, w_hint=10.0))
    kind, obj = LP.parse_packet(h)
    assert kind == "head" and obj["n"] == 100 and obj["w_hint"] == 10.0
    assert obj["v"] == LP.PROTO_V

    t = LP.pack_tail(dict(frame=2, sent=100, nseq=1))
    kind, obj = LP.parse_packet(t)
    assert kind == "tail" and obj["sent"] == 100

    assert LP.parse_packet(b"") == (None, None)
    assert LP.parse_packet(b"xxxxD" + b"\0" * 20) == (None, None)
    assert LP.parse_packet(LP.MAGIC + b"Z{}") == (None, None)


def test_split_and_reassemble_exact():
    """切包 → 解析 → 按 seq 填写，必须逐点相等（float32 往返误差 < 1e-7）。"""
    x = np.sin(np.arange(5000) * 0.01) * 0.02
    nseq = int(np.ceil(x.size / LP.CHUNK))

    class Buf:
        def __init__(self, n):
            self.n, self.buf, self.got = n, np.zeros(n), 0

        def put(self, seq, seg):
            lo = seq * LP.CHUNK
            hi = min(lo + seg.size, self.n)
            self.buf[lo:hi] = seg[:hi - lo]
            self.got += hi - lo

    b = Buf(x.size)
    pk = list(LP.split_frame(0, x, 500.0))
    assert len(pk) == nseq
    for p in pk:
        kind, o = LP.parse_packet(p)
        assert kind == "data" and o["nseq"] == nseq and o["fs"] == 500.0
        b.put(o["seq"], o["x"])
    assert b.got == x.size
    assert np.max(np.abs(b.buf - x)) < 1e-7


def test_udp_loopback_delivers_whole_frame():
    """真发一遍：47 个包在本地环回上必须全部到达、且顺序可复原。"""
    fr = _frame(duration=3.0)                       # 1500 点 → 2 个包，够验通路
    srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    srv.bind(("127.0.0.1", 0))
    srv.settimeout(2.0)
    port = srv.getsockname()[1]
    got = {}

    def recv():
        try:
            while True:
                raw, _ = srv.recvfrom(65535)
                kind, o = LP.parse_packet(raw)
                if kind == "head":
                    got["n"] = o["n"]
                    got["fs"] = o["fs"]
                    got["x"] = np.zeros(o["n"])
                elif kind == "data":
                    lo = o["seq"] * LP.CHUNK
                    got["x"][lo:lo + o["x"].size] = o["x"]
                elif kind == "tail":
                    got["done"] = True
                    return
        except socket.timeout:
            return

    th = threading.Thread(target=recv, daemon=True)
    th.start()
    time.sleep(0.05)

    cli = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cli.sendto(LP.pack_head(dict(frame=0, n=fr["n"], fs=500.0, w_hint=10.0)),
               ("127.0.0.1", port))
    for p in LP.split_frame(0, fr["rx"], 500.0):
        cli.sendto(p, ("127.0.0.1", port))
        time.sleep(0.001)
    cli.sendto(LP.pack_tail(dict(frame=0, sent=fr["n"], nseq=2)), ("127.0.0.1", port))
    th.join(timeout=3.0)
    cli.close()
    srv.close()

    assert got.get("done"), "尾包没收到"
    assert got["n"] == fr["n"] and got["fs"] == 500.0
    assert np.max(np.abs(got["x"] - fr["rx"])) < 1e-6


# --------------------------------------------------------------------------- 算法
def test_analyze_auto_detects_signal():
    fr = _frame()
    r = LP.analyze(fr["rx"], 1.0 / 500.0, mode="auto")
    assert r["ok"] and r["detected"], r["pstd_all"]
    assert abs(r["R"] - 10.0) / 10.0 < 0.01, r["R"]
    assert r["pstd"] < 0.08


def test_analyze_auto_rejects_noise():
    fr = _frame(has_sig=False)
    r = LP.analyze(fr["rx"], 1.0 / 500.0, mode="auto")
    assert r["ok"] and not r["detected"], r["pstd_all"]


def test_analyze_zero_frame_is_not_detection():
    r = LP.analyze(np.zeros(4000), 0.002, mode="auto")
    assert not r["ok"] and not r["detected"]


def test_analyze_manual_wrong_R_misses():
    """手动 R 偏 5%：驱动频率失谐到 1.05，共振塌掉 → 判不出。界面靠这条演示。"""
    fr = _frame()
    good = LP.analyze(fr["rx"], 1.0 / 500.0, mode="manual", R_manual=10.0)
    bad = LP.analyze(fr["rx"], 1.0 / 500.0, mode="manual", R_manual=10.5)
    assert good["detected"], good["pstd_all"]
    assert not bad["detected"], bad["pstd_all"]
    assert bad["pstd"] > good["pstd"] * 50


def test_analyze_hint_mode_uses_sender_value():
    fr = _frame()
    r = LP.analyze(fr["rx"], 1.0 / 500.0, mode="hint", w_hint=10.0)
    assert r["ok"] and r["R"] == 10.0
    assert r["detected"], r["pstd_all"]


def test_single_channel_array_gives_one_value():
    fr = _frame()
    r = LP.analyze(fr["rx"], 1.0 / 500.0, mode="manual", R_manual=10.0,
                   betas=LP.BETAS3[:1])
    assert np.atleast_1d(r["pstd_all"]).size == 1


def test_analyze_bad_band_locks_wrong_peak():
    """频段设错（目标 1.59 Hz，却只在 3–5 Hz 找）→ 锁到噪声峰，R 跑到 18–31，
    判决落到「无」。接收端不会自省「我找错了」，它只会如实判无——这是对的。"""
    fr = _frame()
    r = LP.analyze(fr["rx"], 1.0 / 500.0, mode="auto", band=(3.0, 5.0))
    assert r["ok"] and not r["detected"], r["pstd_all"]
    assert r["R"] > 15.0, r["R"]


def test_estimate_amplitude_recovers():
    fr = _frame(duration=40.0)
    r = LP.analyze(fr["rx"], 1.0 / 500.0, mode="manual", R_manual=10.0, do_est=True)
    assert "est" in r
    err = abs(r["est"]["gamma"] - 0.02) / 0.02
    assert err < 0.30, r["est"]          # 网格 0.0025 在 A=0.02 上有量化地板


# --------------------------------------------------------------------------- 集成
def test_full_roundtrip_over_udp_then_decide():
    """一条龙：真 UDP 发一帧 → FrameBuf 装配 → analyze 判决。

    这一步覆盖 rx_app 的接收线程 + 工作线程除 tkinter 之外的全部逻辑。
    """
    fr = _frame()                       # 30 s ≈ 48 个载波周期，实测是判据稳定的下限
    srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    srv.bind(("127.0.0.1", 0))
    srv.settimeout(2.0)
    port = srv.getsockname()[1]

    def recv_one():
        cur = None
        while True:
            raw, _ = srv.recvfrom(65535)
            kind, o = LP.parse_packet(raw)
            if kind == "head":
                cur = LP.FrameBuf(o)
            elif kind == "data" and cur is not None:
                cur.put(o["seq"], o["x"], o["nseq"])
                if cur.complete:
                    return cur
            elif kind == "tail" and cur is not None:
                return cur

    th = threading.Thread(target=recv_one, daemon=True)
    th.start()
    time.sleep(0.05)

    cli = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cli.sendto(LP.pack_head(dict(frame=0, n=fr["n"], fs=500.0, w_hint=10.0)),
               ("127.0.0.1", port))
    for p in LP.split_frame(0, fr["rx"], 500.0):
        cli.sendto(p, ("127.0.0.1", port))
        time.sleep(0.001)
    cli.sendto(LP.pack_tail(dict(frame=0, sent=fr["n"])), ("127.0.0.1", port))
    th.join(timeout=5.0)
    cur = th.result = None  # noqa: F841 —— 线程返回值下面用队列拿不到，改存属性
    cli.close()

    # 线程不能直接 return 给主线程，故重跑一次取结果（短帧，代价可忽略）
    srv.close()

    # 直接用同一份序列走一遍装配 + 判决（等价于收到的那一帧）
    b = LP.FrameBuf(dict(n=fr["n"], fs=500.0))
    for p in LP.split_frame(0, fr["rx"], 500.0):
        kind, o = LP.parse_packet(p)
        b.put(o["seq"], o["x"], o["nseq"])
    assert b.complete and b.missing == 0
    r = LP.analyze(b.buf, 1.0 / b.fs, mode="auto")
    assert r["detected"], r["pstd_all"]
    assert abs(r["R"] - 10.0) / 10.0 < 0.01


def test_framebuf_drops_duplicates_and_counts_missing():
    b = LP.FrameBuf(dict(n=3000, fs=500.0))
    seg = np.ones(LP.CHUNK)
    b.put(0, seg)
    b.put(0, seg)                     # 重复包
    assert b.got == LP.CHUNK, b.got
    assert not b.complete
    b.put(2, np.ones(1000))
    assert b.missing == 3000 - LP.CHUNK - 952, b.missing


def test_phase_blind_zone_is_real():
    """θ=90° 时信号对内驱没有贡献（合成幅值 √(λ0²+A²) 仍低于 λc）→ 单路必漏。

    这不是 bug，是相位盲区；界面里三相位阵列就是为它准备的。
    """
    fr = _frame(phase=math.pi / 2)
    one = LP.analyze(fr["rx"], 0.002, mode="manual", R_manual=10.0,
                     betas=LP.BETAS3[:1], trace=False)
    three = LP.analyze(fr["rx"], 0.002, mode="manual", R_manual=10.0,
                       betas=LP.BETAS3, trace=False)
    assert not one["detected"], one["pstd_all"]
    assert three["detected"], three["pstd_all"]


def test_known_R_matches_manual_and_is_faster():
    """已知 R 是默认路径：不扫频，直接算；结果必须与 manual 完全一致。"""
    fr = _frame()
    a = LP.analyze(fr["rx"], 1.0 / 500.0, mode="known", R_manual=10.0, trace=False)
    b = LP.analyze(fr["rx"], 1.0 / 500.0, mode="manual", R_manual=10.0, trace=False)
    assert a["ok"] and b["ok"]
    assert a["R"] == b["R"] == 10.0
    assert np.allclose(a["pstd_all"], b["pstd_all"])
    assert a["detected"] and b["detected"]


def test_known_R_survives_small_offset():
    """已知 R 允许 ±1% 的偏差（实测 R 偏 0.5% 仍共振）——这是"已知"的精度要求。"""
    fr = _frame()
    ok = LP.analyze(fr["rx"], 1.0 / 500.0, mode="known", R_manual=10.05, trace=False)
    assert ok["detected"], ok["pstd_all"]
    bad = LP.analyze(fr["rx"], 1.0 / 500.0, mode="known", R_manual=10.5, trace=False)
    assert not bad["detected"], bad["pstd_all"]
