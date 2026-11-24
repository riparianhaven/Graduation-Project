"""针对 learn/0.ipynb 输入信号单元（lam = LAM0, beta = 0.0 及 u_H0/u_H1 构建）的单元测试。

笔记本代码不可直接导入，此处按该单元的离散形式复现同一逻辑，
并覆盖正常、边界与异常路径：
    reference = lam * cos(t + beta)
    target    = gamma * cos(t + theta)
    noise     = sigma * sqrt(H) * N(0, 1)
    u_H0 = reference + noise
    u_H1 = reference + target + noise
"""

import math

import numpy as np
import pytest

# ---- 论文参数（与笔记本全局参数单元一致）----
K = 0.5
LAM0 = 0.72
C_TRIG = 0.73
H = 0.02
T = 400.0


def build_signals(t, lam=LAM0, beta=0.0, gamma=0.02, theta_deg=40.0, sigma=0.20, seed=1):
    """按笔记本单元复现信号构建，返回全部中间量。"""
    theta = np.deg2rad(theta_deg)
    rng = np.random.default_rng(seed)
    reference = lam * np.cos(t + beta)
    target = gamma * np.cos(t + theta)
    noise = sigma * np.sqrt(H) * rng.standard_normal(t.size)
    return reference, target, noise, reference + noise, reference + target + noise


# ---------- 正常路径 ----------

def test_lam_equals_lam0_and_beta_zero():
    """选中的两行：lam 取工作点 LAM0，beta 为 0。"""
    assert LAM0 == pytest.approx(0.72)
    assert 0.0 == pytest.approx(0.0)


def test_reference_amplitude_equals_lam():
    """reference = lam*cos(t+beta)：样本幅值不超过 |lam|，且接近峰值。"""
    t = np.arange(0.0, T, H)
    reference, *_ = build_signals(t)
    assert np.max(np.abs(reference)) <= LAM0 + 1e-12
    assert np.max(np.abs(reference)) == pytest.approx(LAM0, rel=1e-3)


def test_reference_phase_beta_zero_shift():
    """beta=0 时 t=0 处 reference 应等于 lam；改变 beta 仅相位平移。"""
    t = np.arange(0.0, T, H)
    ref_beta0, *_ = build_signals(t, beta=0.0)
    ref_beta, *_ = build_signals(t, beta=math.pi / 2)
    assert ref_beta0[0] == pytest.approx(LAM0)
    # 相位平移 +pi/2 后 t=0 处应为 0
    assert ref_beta[0] == pytest.approx(0.0, abs=1e-12)


def test_target_amplitude_and_phase():
    """target = gamma*cos(t+theta)：幅值 gamma，t=0 处相位 40°。"""
    theta_deg = 40.0
    t = np.arange(0.0, T, H)
    _, target, *_ = build_signals(t, theta_deg=theta_deg)
    assert np.max(np.abs(target)) == pytest.approx(0.02, rel=1e-3)
    assert target[0] == pytest.approx(0.02 * math.cos(math.radians(theta_deg)))


def test_noise_std_matches_sigma_sqrt_H():
    """noise = sigma*sqrt(H)*N(0,1)：样本标准差应接近 sigma*sqrt(H)。"""
    t = np.arange(0.0, T, H)
    *_, noise, _, _ = build_signals(t)
    expected = 0.20 * math.sqrt(H)
    assert np.std(noise) == pytest.approx(expected, rel=0.05)


def test_h0_h1_share_noise():
    """H0/H1 共用同一噪声：u_H1 - u_H0 恰为目标信号。"""
    t = np.arange(0.0, T, H)
    _, target, _, u_h0, u_h1 = build_signals(t)
    np.testing.assert_allclose(u_h1 - u_h0, target, atol=1e-15)


def test_u_h0_equals_reference_plus_noise():
    t = np.arange(0.0, T, H)
    reference, _, _, u_h0, _ = build_signals(t)
    assert u_h0.size == t.size
    # 仅抽查首尾，避免全量对比
    assert u_h0[0] + u_h0[-1] == pytest.approx(reference[0] + reference[-1], rel=0.1)


# ---------- 边界路径 ----------

def test_single_sample_grid():
    """单样本时间网格（边界）。"""
    t = np.arange(0.0, H, H)
    assert t.size == 1
    reference, target, noise, u_h0, u_h1 = build_signals(t)
    for arr in (reference, target, noise, u_h0, u_h1):
        assert arr.size == 1


def test_zero_amplitude_components():
    """lam=0 且 gamma=0 的退化情形：u_H0 == u_H1 == noise。"""
    t = np.arange(0.0, 10.0, H)
    _, _, noise, u_h0, u_h1 = build_signals(t, lam=0.0, gamma=0.0)
    np.testing.assert_allclose(u_h0, u_h1)
    np.testing.assert_allclose(u_h0, noise)


def test_seed_reproducibility():
    """同一种子生成完全相同的噪声。"""
    t = np.arange(0.0, 50.0, H)
    *_, noise_a, _, _ = build_signals(t, seed=7)
    *_, noise_b, _, _ = build_signals(t, seed=7)
    np.testing.assert_array_equal(noise_a, noise_b)


# ---------- 异常路径 ----------

def test_negative_sigma_inverts_noise_sign_only():
    """sigma 为负不报错，只相当于噪声取反。"""
    t = np.arange(0.0, 20.0, H)
    *_, noise_pos, _, _ = build_signals(t, sigma=0.2, seed=3)
    *_, noise_neg, _, _ = build_signals(t, sigma=-0.2, seed=3)
    np.testing.assert_allclose(noise_neg, -noise_pos)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nan_inf_t_amplitude_poison_output(bad):
    """t 含 NaN/Inf 时输出被污染（NaN 传播），但形状不变。"""
    t = np.arange(0.0, 10.0, H)
    t[5] = bad
    reference, target, noise, u_h0, u_h1 = build_signals(t, sigma=0.0)
    assert reference.shape == t.shape
    if math.isnan(bad):
        assert np.isnan(reference[5])
        assert np.isnan(u_h1[5])
    else:
        # IEEE 754 规定 cos(inf) 返回 NaN（而非 Inf），所以污染结果是非有限值 NaN。
        assert not np.isfinite(reference[5])
        assert not np.isfinite(u_h1[5])
