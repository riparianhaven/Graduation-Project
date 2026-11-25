# -*- coding: utf-8 -*-
"""双端 GUI 的底层契约测试（不启动 tkinter，只测协议与算法）。

守这些：
1. 三类包能原样往返，坏包不抛异常，样本是 int16 且往返误差 ≤ 1 LSB；
2. 一帧切成数据包再按 seq 装回来，序列逐点相等（到 1 LSB）；
3. 真 UDP 环回能把一帧送出去且凑齐（丢包率 0）；
4. 有信号判「有」、纯噪声判「无」、全零序列给出 ok=False；
5. R 手动填错（偏 5%）时判不出来——这是界面上「R 不对会怎样」的依据；
6. 连续发送：帧号单调、绝对时间连续、附加窗口的覆盖度与接收端重算一致；
7. 判决统计把背景帧（算虚警）与目标窗帧（算 Pd）分开，部分覆盖单列。
"""
from __future__ import annotations

import math
import queue
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

LSB = 1.0 / LP.ADC_SCALE          # int16 一个最低位对应的物理量


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
    h = LP.pack_head(dict(frame=2, n=100, fs=500.0, w_hint=10.0, cov=1.0, t0=94.2))
    kind, obj = LP.parse_packet(h)
    assert kind == "head" and obj["n"] == 100 and obj["w_hint"] == 10.0
    assert obj["v"] == LP.PROTO_V == 2 and obj["cov"] == 1.0 and obj["t0"] == 94.2

    t = LP.pack_tail(dict(frame=2, sent=100, nseq=1))
    kind, obj = LP.parse_packet(t)
    assert kind == "tail" and obj["sent"] == 100

    assert LP.parse_packet(b"") == (None, None)
    assert LP.parse_packet(b"xxxxD" + b"\0" * 20) == (None, None)
    assert LP.parse_packet(LP.MAGIC + b"Z{}") == (None, None)


def test_int16_payload_roundtrip():
    """样本走 int16：包长 = 21 + 2·N，往返误差 ≤ 1 LSB，超量程要饱和不环绕。"""
    x = np.sin(np.arange(2048) * 0.01) * 0.02
    pk = list(LP.split_frame(0, x, 500.0))
    assert len(pk) == 2
    assert len(pk[0]) == LP.DATA_HDR.size + LP.CHUNK * 2          # 21 + 2048
    k, o = LP.parse_packet(pk[0])
    assert k == "data" and o["raw"].dtype == np.int16
    assert np.max(np.abs(o["x"] - x[:LP.CHUNK])) <= LSB

    sat = LP.to_int16(np.array([0.0, 1.5, -1.5, 0.99999]))
    assert sat[0] == 0 and sat[1] == LP.ADC_MAX and sat[2] == LP.ADC_MIN
    assert sat[3] == 32767


def test_split_and_reassemble_exact():
    """切包 → 解析 → 按 seq 填写，误差只剩 int16 量化（≤1 LSB）。"""
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
    assert np.max(np.abs(b.buf - x)) <= LSB


def test_framebuf_scales_by_head_scale():
    """头包给不同 scale 时，put_raw 必须按它换算——这是接 ESP32 ADC 计数的入口。"""
    b = LP.FrameBuf(dict(n=4, fs=500.0, scale=2048.0))
    b.put_raw(0, np.array([2048, -2048, 1024, 0], dtype=np.int16))
    assert np.allclose(b.buf[:4], [1.0, -1.0, 0.5, 0.0])


def test_udp_loopback_delivers_whole_frame():
    """真发一遍：int16 包在本地环回上必须全部到达、且顺序可复原。"""
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
    cli.sendto(LP.pack_head(dict(frame=0, n=fr["n"], fs=500.0, w_hint=10.0,
                                 scale=LP.ADC_SCALE)), ("127.0.0.1", port))
    for p in LP.split_frame(0, fr["rx"], 500.0):
        cli.sendto(p, ("127.0.0.1", port))
        time.sleep(0.001)
    cli.sendto(LP.pack_tail(dict(frame=0, sent=fr["n"], nseq=2)), ("127.0.0.1", port))
    th.join(timeout=3.0)
    cli.close()
    srv.close()

    assert got.get("done"), "尾包没收到"
    assert got["n"] == fr["n"] and got["fs"] == 500.0
    assert np.max(np.abs(got["x"] - fr["rx"])) <= LSB


# --------------------------------------------------------------------------- 连续发送
def test_txstream_never_stops_and_time_is_contiguous():
    """连续流：帧号单调 +1，t0 每帧前移 n/fs，没有总数上限。"""
    st = LP.TxStream(w=10.0, fs=500.0, duration=2.0, amp=0.02, sigma=0.05, seed=1)
    prev_t = -1.0
    for k in range(50):
        fr = st.frame()
        assert fr["frame"] == k
        assert fr["cov"] == 0.0                 # 没附加目标 = 纯背景
        if k:
            assert fr["t0"] == pytest.approx(prev_t + 2.0, abs=1e-9)
        prev_t = fr["t0"]
    assert st.k == 50 and st.t_now == pytest.approx(100.0, abs=1e-9)


def test_attach_window_starts_at_next_frame_boundary():
    """附加目标从"下一帧起点"起算，不是从当前时刻中间切开。"""
    st = LP.TxStream(w=10.0, fs=500.0, duration=2.0, amp=0.02, sigma=0.05, seed=2)
    st.frame()                                   # 帧 0 已生成，t0 前进到 2.0
    t_end = st.attach(3.0)
    assert t_end == pytest.approx(5.0)
    assert st.frame()["cov"] == 1.0              # 帧 1 [2,4)
    assert st.frame()["cov"] == pytest.approx(0.5)   # 帧 2 [4,6) 只剩 4–5
    assert st.frame()["cov"] == 0.0


def test_attach_then_cancel_returns_to_background():
    st = LP.TxStream(w=10.0, fs=500.0, duration=2.0, amp=0.02, sigma=0.05, seed=3)
    st.attach(10.0)
    assert st.frame()["cov"] == 1.0
    st.cancel()
    assert st.frame()["cov"] == 0.0


def test_signal_is_phase_continuous_across_frames():
    """窗口跨帧时相位连续：第二帧起点的 φ = ω·帧长 + 第一帧起点的 φ（模 2π）。

    真实目标不会在帧边界跳相；跳相会让同一段目标在不同帧里随机落进盲区。
    """
    st = LP.TxStream(w=10.0, fs=500.0, duration=2.0, amp=0.02, sigma=0.05, seed=4)
    st.attach(4.0)                               # 窗口 [0, 4)，跨两帧
    f0, f1 = st.frame(), st.frame()
    expect = (f0["phase"] + 10.0 * 2.0) % (2 * math.pi)
    got = f1["phase"] % (2 * math.pi)
    d = abs(math.atan2(math.sin(got - expect), math.cos(got - expect)))
    assert d < 1e-9, (f0["phase"], f1["phase"])


def test_receiver_recomputes_coverage_from_timestamps():
    """接收端用 t0/sig_t0/sig_dur 重算覆盖度，必须与发送端声明的一致。"""
    st = LP.TxStream(w=10.0, fs=500.0, duration=2.0, amp=0.02, sigma=0.05, seed=5)
    st.frame()
    st.attach(5.0)
    for _ in range(4):
        fr = st.frame()
        meta = dict(n=fr["n"], fs=st.fs, t0=fr["t0"], cov=fr["cov"],
                    sig_t0=fr["sig_t0"], sig_dur=fr["sig_dur"])
        assert LP.coverage_of(meta) == pytest.approx(fr["cov"], abs=1e-9)


def test_coverage_of_falls_back_to_declared_cov():
    """常开模式没有窗口字段，只能用发送端声明的 cov。"""
    assert LP.coverage_of(dict(n=100, fs=500.0, cov=1.0, t0=0.0, sig_t0=None)) == 1.0
    assert LP.coverage_of(dict(n=100, fs=500.0)) == 0.0


def test_always_sig_mode_is_full_coverage():
    st = LP.TxStream(w=10.0, fs=500.0, duration=2.0, amp=0.02, sigma=0.05,
                     always_sig=True, seed=6)
    for _ in range(3):
        fr = st.frame()
        assert fr["cov"] == 1.0 and fr["sig_t0"] is None


# --------------------------------------------------------------------------- 统计
def test_verdict_stats_separates_bg_and_window():
    s = LP.VerdictStats()
    assert s.add(1.0, True) == "sig"
    assert s.add(1.0, False) == "sig"
    assert s.add(0.0, False) == "bg"
    assert s.add(0.0, True) == "bg"
    assert s.add(0.4, True) == "part"
    assert s.add(1.0, None) == "bad"
    assert s.sig == 2 and s.sig_hit == 1 and s.pd == pytest.approx(0.5)
    assert s.bg == 2 and s.bg_hit == 1 and s.pfa == pytest.approx(0.5)
    assert s.part == 1 and s.bad == 1
    txt = s.text()
    assert "背景 2" in txt and "目标窗 2" in txt and "部分覆盖 1" in txt


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


# --------------------------------------------------------------------------- 集成
def test_continuous_udp_background_then_target():
    """连续两帧走真 UDP：帧 0 纯背景判无，帧 1 在目标窗内判有。

    覆盖 rx_app 接收线程 + 工作线程除 tkinter 之外的全部逻辑：头包定帧长、
    int16 装配、覆盖度重算、已知 R 判决。
    """
    st = LP.TxStream(w=10.0, fs=500.0, duration=30.0, amp=0.0302, sigma=0.05, seed=7)

    srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    srv.bind(("127.0.0.1", 0))
    port = srv.getsockname()[1]
    srv.settimeout(5.0)

    def send():
        cli = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for i in range(2):
            if i == 1:
                st.attach(30.0)          # 帧 0 已出，t0=30 → 帧 1 整帧覆盖
            fr = st.frame()
            meta = dict(frame=fr["frame"], n=fr["n"], fs=500.0, scale=st.scale,
                        w_hint=10.0, cov=fr["cov"], t0=fr["t0"],
                        sig_t0=fr["sig_t0"], sig_dur=fr["sig_dur"], has_sig=True)
            cli.sendto(LP.pack_head(meta), ("127.0.0.1", port))
            for pkt in LP.split_frame(fr["frame"], fr["rx"], 500.0, st.scale):
                cli.sendto(pkt, ("127.0.0.1", port))
                time.sleep(0.0005)
            cli.sendto(LP.pack_tail(dict(frame=fr["frame"], sent=fr["n"])),
                       ("127.0.0.1", port))
        cli.close()

    th = threading.Thread(target=send, daemon=True)
    th.start()

    done: queue.Queue = queue.Queue()

    def on_frame(meta, x):
        done.put((meta, x))

    buf = None
    while done.qsize() < 2:
        raw, _ = srv.recvfrom(65535)
        kind, o = LP.parse_packet(raw)
        if kind == "head":
            buf = LP.FrameBuf(o)
        elif kind == "data" and buf is not None:
            buf.put_raw(o["seq"], o["raw"])
            if buf.complete:
                on_frame(buf.meta, buf.buf.copy())
                buf = None
        elif kind == "tail" and buf is not None:
            on_frame(buf.meta, buf.buf.copy())
            buf = None
    th.join(timeout=5.0)
    srv.close()

    frames = [done.get(), done.get()]
    assert [m["frame"] for m, _ in frames] == [0, 1]
    assert [m["cov"] for m, _ in frames] == [0.0, 1.0]
    # 接收端自己重算的覆盖度必须与发送端一致
    assert LP.coverage_of(frames[1][0]) == 1.0

    r0 = LP.analyze(frames[0][1], 0.002, mode="known", R_manual=10.0, trace=False)
    r1 = LP.analyze(frames[1][1], 0.002, mode="known", R_manual=10.0, trace=False)
    assert not r0["detected"], r0["pstd_all"]
    assert r1["detected"], r1["pstd_all"]

    s = LP.VerdictStats()
    s.add(LP.coverage_of(frames[0][0]), r0["detected"])
    s.add(LP.coverage_of(frames[1][0]), r1["detected"])
    assert s.bg == 1 and s.bg_hit == 0
    assert s.sig == 1 and s.sig_hit == 1 and s.pd == 1.0


def test_framebuf_drops_duplicates_and_counts_missing():
    b = LP.FrameBuf(dict(n=3000, fs=500.0))
    seg = np.ones(LP.CHUNK)
    b.put(0, seg)
    b.put(0, seg)                     # 重复包
    assert b.got == LP.CHUNK, b.got
    assert not b.complete
    b.put(2, np.ones(1000))
    assert b.missing == 3000 - LP.CHUNK - 952, b.missing


# --------------------------------------------------------------------------- 框架演示
def test_coherent_proj_is_unbiased_in_phase():
    """相干投影对纯正弦无偏：任何初相都给回 A（这是它当演示统计量的理由）。"""
    fs, f0, A = 500.0, 1.6, 0.02
    t = np.arange(int(20 * fs)) / fs
    for deg in (0.0, 37.0, 90.0, 180.0, -125.0):
        x = A * np.cos(2 * math.pi * f0 * t + math.radians(deg))
        assert abs(LP.coherent_proj(x, fs, f0) - A) < 1e-3 * A * 5, deg


def test_framework_demo_threshold_comes_from_h0_only():
    """阈值必须等于 H0 样本的 95% 分位——拿 H1 调阈值会让 Pd 虚高。"""
    r = LP.framework_demo(fs=200.0, f0=1.0, amp=1.0, sigma_samp=3.976,
                          dur=20.0, n_mc=400, seed=3)
    assert r["thr"] == pytest.approx(float(np.quantile(r["stat_h0"], 0.95)), rel=1e-12)
    assert r["pfa"] == pytest.approx(float(np.mean(r["stat_h0"] > r["thr"])))
    assert 0.02 <= r["pfa"] <= 0.09, r["pfa"]        # 标称 0.05，允许分位数抽样误差
    assert -15.5 <= r["snr_e_db"] <= -14.5, r["snr_e_db"]


def test_framework_demo_pd_grows_with_snr_and_collapses_when_no_signal():
    """Pd 随 SNR 单调升；A=0 时 H1 退化成 H0，Pd 应回落到 Pfa 附近。"""
    lo = LP.framework_demo(fs=200.0, f0=1.0, amp=0.0, sigma_samp=3.976,
                           dur=20.0, n_mc=300, seed=5)
    mid = LP.framework_demo(fs=200.0, f0=1.0, amp=1.0, sigma_samp=3.976,
                            dur=20.0, n_mc=300, seed=5)
    hi = LP.framework_demo(fs=200.0, f0=1.0, amp=6.0, sigma_samp=3.976,
                           dur=20.0, n_mc=300, seed=5)
    assert lo["pd"] <= 0.10, lo["pd"]                # 无信号 ≈ 虚警率
    assert lo["pd"] < mid["pd"] <= hi["pd"], (lo["pd"], mid["pd"], hi["pd"])
    assert hi["pd"] >= 0.99


def test_framework_demo_sigma_consistent_with_energy_snr():
    """A 与 SNR_E 定 σ：σ = √((A²/2)/10^(SNR_E/10))，两边算出来的 SNR_E 要闭合。"""
    A, snr = 1.0, -15.0
    sd = math.sqrt((A ** 2 / 2.0) / 10 ** (snr / 10.0))
    r = LP.framework_demo(fs=200.0, f0=1.0, amp=A, sigma_samp=sd,
                          dur=10.0, n_mc=50, seed=1)
    assert r["snr_e_db"] == pytest.approx(snr, abs=1e-9)
    assert r["sigma_samp"] == pytest.approx(sd)
