# -*- coding: utf-8 -*-
"""端到端链路仿真的契约测试。

守四条：
1. 相位覆盖率的解析式与几何直觉一致（单振子 < 阵列，强信号收敛到 1）；
2. λ* 差分估计器能还原幅值——**这一条专门防"把 λ 网格忘了传"的回归**，
   那个 bug 会让估计器只扫 β 不扫 λ，结果恒为 2 倍真值；
3. 有信号判「有」、纯噪声判「无」；
4. 链路只调用 simulate_seq，不自己再写一套积分器。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from link_sim import (LinkCfg, channel, estimate_amp, full_receive,  # noqa: E402
                      phase_coverage, tx_emit)


def test_phase_coverage_geometry():
    cfg = LinkCfg()
    c1 = phase_coverage(cfg, 0.02, 1)
    c3 = phase_coverage(cfg, 0.02, 3)
    assert 0.20 < c1 < 0.32, c1            # λ0=0.72、λc=0.7335、A=0.02 → 约 26%
    assert 0.7 < c3 < 0.9, c3              # 三相位补到约 80%
    assert c3 > c1
    assert phase_coverage(cfg, 0.05, 3) == pytest.approx(1.0)   # 强信号铺满一圈
    assert phase_coverage(cfg, 1e-6, 1) == pytest.approx(0.0)   # 信号太弱，全盲


def test_estimator_recovers_amplitude_noiseless():
    """λ* 差分器在无噪时必须还原 γ（网格 0.0025 → 允许 12% 量化误差）。"""
    cfg = LinkCfg()
    amp, theta = 0.02, 0.7
    est = estimate_amp(cfg, tx_emit(cfg, [theta], amp), cfg.w)
    assert abs(est["gamma"] - amp) / amp < 0.12, est
    dth = abs((est["theta"] - theta + math.pi) % (2 * math.pi) - math.pi)
    assert math.degrees(dth) < 10.0, est


@pytest.fixture(scope="module")
def short_cfg():
    """短窗版本：60 个载波周期，够判态、跑得快。"""
    return LinkCfg(n_cycles=60.0)


def test_end_to_end_detects_signal_and_rejects_noise(short_cfg):
    cfg = short_cfg
    rng = np.random.default_rng(42)

    # 同相（θ=0.2 rad）：λ0 + A·cosθ 越过 λc → 判「有」
    on = full_receive(cfg, channel(cfg, tx_emit(cfg, [0.2], 0.02), rng))
    assert on["detected"], on["pstd_ch"]
    assert abs(on["w_hat"] - cfg.w) / cfg.w < 2e-3, on["err_pct"]

    # 正交（θ=90°）：信号对合成幅值没有贡献 → 相位盲区，单振子必漏
    blind = full_receive(cfg, channel(cfg, tx_emit(cfg, [math.pi / 2], 0.02), rng))
    assert not blind["detected"], blind["pstd_ch"]

    # 纯噪声：判「无」
    off = full_receive(cfg, channel(cfg, np.zeros((cfg.n, 1)), rng))
    assert not off["detected"], off["pstd_ch"]


def test_link_does_not_grow_a_second_integrator():
    src = (ROOT / "tools" / "link_sim.py").read_text(encoding="utf-8")
    assert "def simulate(" not in src          # 不许自己写积分器
    assert "def simulate_seq(" not in src      # 内核只能来自 kernel_src
    assert "load_kernel" in src
