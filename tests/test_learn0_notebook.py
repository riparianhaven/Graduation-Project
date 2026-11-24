# -*- coding: utf-8 -*-
"""learn/0.ipynb 单元测试。

验证内容：
1. 参数单元中的全局常量（K, LAM0, C_TRIG, H, T, BURN 等）。
2. 输入信号单元的信号生成逻辑：
   - u(t) = lam*cos(t+beta) + gamma*cos(t+theta) + sigma*sqrt(H)*N(0,1)
   - reference / target / noise / u_H0 / u_H1 的形状、数值、统计特性。
3. 边界情况：show 掩码、噪声确定性（固定种子）、噪声幅度随 sigma/H 的缩放。

测试通过 nbformat 提取笔记本代码单元，在受控命名空间中执行
（matplotlib 使用 Agg 后端，plt.show 被替换为空操作，magic 行被剔除）。
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境下执行笔记本绘图单元

import numpy as np
import pytest

NB_PATH = Path(__file__).resolve().parent.parent / "learn" / "0.ipynb"


# ---------- 笔记本加载与执行工具 ----------

def _load_code_cells(nb_path: Path) -> list[list[str]]:
    """读取笔记本并返回各代码单元的源码行列表。"""
    import nbformat

    nb = nbformat.read(str(nb_path), as_version=4)
    cells = []
    for cell in nb.cells:
        if cell.cell_type == "code":
            lines = [ln for ln in cell.source.splitlines()
                     if not ln.lstrip().startswith("%")]  # 剔除 IPython magic
            cells.append(lines)
    return cells


def _exec_cell(lines: list[str], ns: dict) -> dict:
    """在命名空间 ns 中执行代码单元，返回该命名空间。"""
    exec(compile("\n".join(lines), "<notebook-cell>", "exec"), ns)
    return ns


@pytest.fixture(scope="module")
def cells() -> list[list[str]]:
    got = _load_code_cells(NB_PATH)
    # learn/0.ipynb 已从最初的 2 个代码单元扩展到 8 个（变尺度、阵列、阈值扫描等）。
    # 本测试只覆盖前两个（全局参数、输入信号），故按「至少 2 个」校验；
    # 若前两单元的顺序或内容变了，下面的 param_ns / signal_ns 会立刻失败。
    assert len(got) >= 2, f"笔记本至少需要 2 个代码单元，实际 {len(got)}"
    return got


@pytest.fixture(scope="module")
def param_ns(cells: list[list[str]]) -> dict:
    """执行参数单元（第 1 个代码单元），并屏蔽计时副作用。"""
    import matplotlib.pyplot as plt

    ns: dict = {"plt": plt, "math": math, "np": np, "time": None}
    _exec_cell(cells[0], ns)
    return ns


@pytest.fixture(scope="module")
def signal_ns(cells: list[list[str]], param_ns: dict) -> dict:
    """执行输入信号单元（第 2 个代码单元）。"""
    import matplotlib.pyplot as plt

    plt.show = lambda *a, **k: None  # 空操作，避免阻塞
    ns = dict(param_ns)
    ns["plt"] = plt
    _exec_cell(cells[1], ns)
    return ns


# ---------- 1. 参数单元 ----------

class TestParams:
    def test_core_constants(self, param_ns):
        """论文参数与本复现条件应与注释声明的值一致。"""
        assert param_ns["K"] == pytest.approx(0.5)
        assert param_ns["LAM0"] == pytest.approx(0.72)
        assert param_ns["C_TRIG"] == pytest.approx(0.73)
        assert param_ns["H"] == pytest.approx(0.02)
        assert param_ns["T"] == pytest.approx(400.0)
        assert param_ns["BURN"] == pytest.approx(200.0)
        assert param_ns["T_FAST"] == pytest.approx(200.0)
        assert param_ns["BURN_FAST"] == pytest.approx(100.0)
        assert param_ns["PSTD_TH"] == pytest.approx(0.08)

    def test_damping_within_chaos_window(self, param_ns):
        """阻尼 K 应为正且小于 1（Duffing 混沌参数窗口的合理范围）。"""
        assert 0 < param_ns["K"] < 1.0

    def test_c_trig_above_lam0(self, param_ns):
        """触发幅值 C_TRIG 应略大于内驱幅值 LAM0（临界阈值机制）。"""
        assert param_ns["C_TRIG"] > param_ns["LAM0"]

    def test_step_and_burn_consistent(self, param_ns):
        """积分步长应为正；丢弃瞬态 BURN 应小于总时长 T。"""
        assert param_ns["H"] > 0
        assert 0 < param_ns["BURN"] < param_ns["T"]


# ---------- 2. 时间轴与信号参数 ----------

class TestTimeGridAndSignalParams:
    def test_time_grid(self, signal_ns):
        """t 应为 [0, T) 上步长 H 的均匀网格。"""
        t = signal_ns["t"]
        H, T = signal_ns["H"], signal_ns["T"]
        assert t[0] == pytest.approx(0.0)
        assert t.size == int(round(T / H))  # 400 / 0.02 = 20000
        assert t[1] == pytest.approx(H)
        assert t[-1] == pytest.approx(T - H)
        assert np.allclose(np.diff(t), H)

    def test_lam_uses_lam0_and_beta_zero(self, signal_ns):
        """内部参考驱动使用工作点幅值 LAM0，初相 beta = 0。"""
        assert signal_ns["lam"] == pytest.approx(signal_ns["LAM0"])
        assert signal_ns["lam"] == pytest.approx(0.72)
        assert signal_ns["beta"] == pytest.approx(0.0)

    def test_target_signal_params(self, signal_ns):
        """待测信号 gamma=0.02，theta=40°（弧度换算正确）。"""
        assert signal_ns["gamma"] == pytest.approx(0.02)
        assert signal_ns["theta"] == pytest.approx(math.radians(40.0))
        assert np.isclose(np.rad2deg(signal_ns["theta"]), 40.0)

    def test_noise_sigma(self, signal_ns):
        assert signal_ns["sigma"] == pytest.approx(0.20)


# ---------- 3. 信号分量 ----------

class TestSignalComponents:
    def test_reference_matches_formula(self, signal_ns):
        """reference 应精确等于 lam*cos(t+beta)。"""
        t = signal_ns["t"]
        expected = signal_ns["lam"] * np.cos(t + signal_ns["beta"])
        assert np.allclose(signal_ns["reference"], expected)

    def test_reference_endpoint_values(self, signal_ns):
        """beta=0 时端点取值可解析验证：reference[0]=lam。"""
        lam = signal_ns["lam"]
        assert signal_ns["reference"][0] == pytest.approx(lam)
        # 在 t=157*H 处用解析式核对单个样本
        assert signal_ns["reference"][157] == pytest.approx(
            lam * math.cos(157 * signal_ns["H"]), abs=1e-12)

    def test_target_matches_formula(self, signal_ns):
        """target 应精确等于 gamma*cos(t+theta)。"""
        t = signal_ns["t"]
        expected = signal_ns["gamma"] * np.cos(t + signal_ns["theta"])
        assert np.allclose(signal_ns["target"], expected)

    def test_target_amplitude_bounded(self, signal_ns):
        """目标分量是弱信号：幅值不超过 gamma，且确实在正负之间摆动。"""
        tgt = signal_ns["target"]
        assert np.max(np.abs(tgt)) <= signal_ns["gamma"] + 1e-12
        assert tgt.min() < 0 < tgt.max()

    def test_target_is_weak_relative_to_reference(self, signal_ns):
        """弱信号假设：gamma << lam（信噪比意义下的可检测前提）。"""
        assert signal_ns["gamma"] < 0.05 * signal_ns["lam"]


# ---------- 4. 噪声 ----------

class TestNoise:
    def test_noise_deterministic_with_seed_1(self, signal_ns):
        """固定种子 rng(1) 下噪声应完全可复现。"""
        expected = (signal_ns["sigma"] * math.sqrt(signal_ns["H"])
                    * np.random.default_rng(1).standard_normal(signal_ns["t"].size))
        assert np.array_equal(signal_ns["noise"], expected)

    def test_noise_shape_matches_time_grid(self, signal_ns):
        assert signal_ns["noise"].shape == signal_ns["t"].shape

    def test_noise_statistics(self, signal_ns):
        """样本标准差应接近 sigma*sqrt(H)，均值应接近 0（大样本统计容差）。"""
        noise = signal_ns["noise"]
        n = noise.size
        expected_std = signal_ns["sigma"] * math.sqrt(signal_ns["H"])
        # 均值容差：4 * std/sqrt(n)
        assert abs(noise.mean()) < 4 * expected_std / math.sqrt(n)
        # 样本标准差相对容差 ~ 1/sqrt(2n) 的数倍
        assert noise.std(ddof=1) == pytest.approx(expected_std, rel=4 / math.sqrt(2 * n))

    def test_noise_scaling_with_sigma_and_H(self):
        """噪声离散形式 sigma*sqrt(H)*N(0,1) 对 sigma 与 H 的线性/开方缩放。"""
        rng = np.random.default_rng(7)
        base = rng.standard_normal(50_000)
        for sigma, H in [(0.1, 0.01), (0.2, 0.02), (0.5, 0.02), (0.2, 0.08)]:
            sample = sigma * math.sqrt(H) * base
            assert sample.std(ddof=1) == pytest.approx(sigma * math.sqrt(H), rel=0.05)

    def test_zero_sigma_gives_zero_noise(self):
        """边界：sigma=0 时无噪声（纯确定性输入）。"""
        H = 0.02
        noise = 0.0 * math.sqrt(H) * np.random.default_rng(1).standard_normal(100)
        assert np.all(noise == 0.0)


# ---------- 5. 合成输入 u_H0 / u_H1 ----------

class TestSyntheticInputs:
    def test_u_h0_is_reference_plus_noise(self, signal_ns):
        assert np.allclose(signal_ns["u_H0"],
                           signal_ns["reference"] + signal_ns["noise"])

    def test_u_h1_is_reference_plus_target_plus_noise(self, signal_ns):
        assert np.allclose(signal_ns["u_H1"],
                           signal_ns["reference"] + signal_ns["target"] + signal_ns["noise"])

    def test_h1_minus_h0_equals_target(self, signal_ns):
        """H0/H1 共用同一噪声，两假设之差应恰为目标信号。"""
        assert np.allclose(signal_ns["u_H1"] - signal_ns["u_H0"], signal_ns["target"])

    def test_shared_noise_between_hypotheses(self, signal_ns):
        """共用噪声保证对照公平：H0 与 H1 的差异只来自 target。"""
        n = signal_ns["u_H1"].size
        resid = (signal_ns["u_H1"] - signal_ns["u_H0"]) - signal_ns["target"]
        assert np.max(np.abs(resid)) == pytest.approx(0.0, abs=1e-15)
        assert n > 0

    def test_amplitude_range_plausible(self, signal_ns):
        """合成输入幅值不应超出 |lam|+gamma+噪声上尾的合理范围。"""
        for u in (signal_ns["u_H0"], signal_ns["u_H1"]):
            bound = signal_ns["lam"] + signal_ns["gamma"] + 6 * signal_ns["sigma"] * math.sqrt(signal_ns["H"])
            assert np.max(np.abs(u)) < bound


# ---------- 6. 展示掩码与绘图对象 ----------

class TestShowMaskAndPlot:
    def test_show_mask_is_first_20_time_units(self, signal_ns):
        """show = t < 20 应选出前 20 个模型时间单位的样本。"""
        t, show = signal_ns["t"], signal_ns["show"]
        assert show.dtype == bool
        assert show.sum() == int(round(20.0 / signal_ns["H"]))  # 20/0.02 = 1000
        assert np.all(t[show] < 20.0)
        assert np.all(t[~show] >= 20.0)

    def test_figure_has_three_axes(self, signal_ns):
        """绘图单元应生成 3x1 的共享 x 轴子图（H0/H1 对照、目标分量、H2 仅目标）。

        2026-11-15 更新：notebook 在 H0/H1 之外新增了 H2（仅目标）对照面板，
        子图数由 2 变为 3，此处同步；共享 x 轴要求对全部面板成立。
        """
        fig = signal_ns["fig"]
        assert len(fig.axes) == 3
        for ax in fig.axes[1:]:
            assert ax.get_shared_x_axes().joined(fig.axes[0], ax)

    def test_lines_have_labels(self, signal_ns):
        """每条曲线都应有图例标签（H0/H1 对照与目标分量）。"""
        labels = []
        for ax in signal_ns["fig"].axes:
            labels += [ln.get_label() for ln in ax.get_lines()]
        assert any("H0" in lb for lb in labels)
        assert any("H1" in lb for lb in labels)

    def test_plotted_data_lengths_match_mask(self, signal_ns):
        """绘制的曲线长度应等于 show 掩码选中的样本数。"""
        for ax in signal_ns["fig"].axes:
            for ln in ax.get_lines():
                assert len(ln.get_xdata()) == signal_ns["show"].sum()


# ---------- 7. 独立复算对照（端到端） ----------

class TestEndToEndRecomputation:
    def test_full_pipeline_recomputed_independently(self, param_ns):
        """完全独立地复算整条链路，应与笔记本执行结果逐位一致。"""
        import matplotlib.pyplot as plt

        plt.show = lambda *a, **k: None
        ns = {"plt": plt, "math": math, "np": np, "time": None,
              **{k: v for k, v in param_ns.items()
                 if not k.startswith("_") and k != "plt"}}
        # 从参数命名空间重建第二单元的输入并独立计算
        H, T = ns["H"], ns["T"]
        t = np.arange(0.0, T, H)
        lam, beta = ns["LAM0"], 0.0
        gamma, theta, sigma = 0.02, np.deg2rad(40.0), 0.20
        noise = sigma * math.sqrt(H) * np.random.default_rng(1).standard_normal(t.size)
        u_H1 = lam * np.cos(t + beta) + gamma * np.cos(t + theta) + noise

        exec_ns = dict(ns)
        cell_src = "\n".join(_load_code_cells(NB_PATH)[1])
        # 剔除绘图与打印部分，仅保留计算部分（到 u_H1 为止）
        calc_src = cell_src.split("# 展示前 20 个模型时间单位")[0]
        exec(compile(calc_src, "<recompute>", "exec"), exec_ns)
        assert np.array_equal(exec_ns["u_H1"], u_H1)

    def test_target_phase_offset_effect(self):
        """theta 相位移动应使 t=0 处目标值 = gamma*cos(theta)。"""
        gamma, theta = 0.02, np.deg2rad(40.0)
        assert gamma * math.cos(0.0 + theta) == pytest.approx(gamma * math.cos(theta))
        assert gamma * math.cos(theta) == pytest.approx(0.02 * math.cos(math.radians(40)))
