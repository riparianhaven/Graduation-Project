# -*- coding: utf-8 -*-
"""内核一致性契约测试（全项目单一内核 simulate_seq）。

守的是四条纪律，任何一条被破坏都说明"又长出了第二套积分器"：

1. 没有解析内核：`def simulate(` 在项目自有代码与 notebook 里一处都不许有
   （包括 `s2-jupyterbook/_src`）。`simulate_fixed` 是定点复刻，不在此列，
   因为它的存在意义就是和浮点内核对比量化代价。
2. 每份含内核的 notebook 恰好一个 `def simulate_seq`。
3. notebook 里的内核正文与 `tools/kernel_src.KERNEL_SRC` 逐字相同
   （生成器改了却没重新生成、或手改 notebook 漏同步，都会被这条抓住）。
4. 数值契约：全零序列等价于"只有内驱"，λ=0.72 混沌 / 0.76 周期；
   标量快路径与向量路径在同一问题上给出相同 pstd。
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"

import sys

sys.path.insert(0, str(TOOLS))
from kernel_src import KERNEL_SRC, load_kernel  # noqa: E402

# 含内核的 notebook（learn/0,1,5 与 s2 书里的 notebook 不含内核，只有 JSON 读取）
NB_WITH_KERNEL = ["learn/2.ipynb", "learn/3.ipynb", "learn/4.ipynb",
                  "learn/all.ipynb", "复现-2-混沌振子建模与临界阈值.ipynb"]

RETURN_LINE = "    return dict(pstd=pstd, mx=np.atleast_1d(mx)"


def _nb_sources(path: Path) -> list[str]:
    import nbformat

    nb = nbformat.read(str(path), as_version=4)
    return [c.source for c in nb.cells if c.cell_type == "code"]


def _extract_kernel(src: str) -> str:
    """从一段单元源码里截出 simulate_seq 的定义正文。"""
    lines = src.splitlines()
    i0 = next(i for i, ln in enumerate(lines) if ln.startswith("def simulate_seq("))
    i1 = next(i for i, ln in enumerate(lines) if ln.startswith(RETURN_LINE))
    return "\n".join(lines[i0:i1 + 1])


# ---------------------------------------------------------------- 纪律 1
def test_no_analytic_kernel_in_python():
    """tools/ 与 s2 书源码里不许再有 def simulate(（解析写法）。"""
    bad = []
    for p in list(TOOLS.glob("*.py")) + list((ROOT / "s2-jupyterbook/_src").glob("*.py")):
        txt = p.read_text(encoding="utf-8")
        if re.search(r"^def simulate\(", txt, re.M):
            bad.append(p.name)
    assert not bad, f"仍有解析内核定义：{bad}"


def test_no_analytic_calls_in_notebooks():
    """所有 notebook 里不许出现调用 simulate(（simulate_seq / simulate_fixed 不算）。"""
    bad = {}
    for p in sorted(ROOT.glob("learn/*.ipynb")) + [ROOT / NB_WITH_KERNEL[-1]]:
        n = len(re.findall(r"(?<![_a-zA-Z])simulate\(", p.read_text(encoding="utf-8")))
        if n:
            bad[p.name] = n
    assert not bad, f"仍有 simulate( 调用：{bad}"


# ---------------------------------------------------------------- 纪律 2
@pytest.mark.parametrize("rel", NB_WITH_KERNEL)
def test_one_kernel_per_notebook(rel):
    srcs = _nb_sources(ROOT / rel)
    n = sum(len(re.findall(r"^def simulate_seq\(", s, re.M)) for s in srcs)
    assert n == 1, f"{rel} 里有 {n} 个 def simulate_seq，应恰好 1 个"


# ---------------------------------------------------------------- 纪律 3
@pytest.mark.parametrize("rel", NB_WITH_KERNEL)
def test_notebook_kernel_matches_shared_source(rel):
    want = KERNEL_SRC.strip("\n")
    for s in _nb_sources(ROOT / rel):
        if "def simulate_seq(" in s:
            assert _extract_kernel(s) == want, (
                f"{rel} 的内核与 tools/kernel_src.py 不一致（改了生成器没重新生成？）")
            return
    pytest.fail(f"{rel} 里找不到内核单元")


def test_s2sim_reexports_shared_kernel():
    """s2 书的内核必须是同一段源码，而不是自己抄一份。"""
    sys.path.insert(0, str(ROOT / "s2-jupyterbook/_src"))
    import s2sim

    assert not hasattr(s2sim, "simulate"), "s2sim 仍导出解析内核 simulate"
    # 字节码相同 = 同一段源码 exec 出来的（exec 的函数没有源文件，不能 inspect）
    ref, _ = load_kernel(TWOPI=s2sim.TWOPI, K=s2sim.K, LAM0=s2sim.LAM0,
                         H=s2sim.H, R0=s2sim.R0)
    assert s2sim.simulate_seq.__code__.co_code == ref.__code__.co_code
    assert s2sim.simulate_seq.__code__.co_consts == ref.__code__.co_consts


# ---------------------------------------------------------------- 纪律 4
@pytest.fixture(scope="module")
def kernel():
    fn, ns = load_kernel()
    return fn, ns


def test_zero_sequence_gives_known_states(kernel):
    """全零序列 = 只有内驱：λ=0.72 混沌、0.76 大周期（生产常数，莫随手改）。"""
    fn, ns = kernel
    r = fn(ns["SIG0"], 1.0, 0.02, lam=[0.72, 0.76],
           n_cycles=400.0 / ns["TWOPI"])
    assert r["pstd"][0] == pytest.approx(0.35223, abs=5e-4)
    assert r["pstd"][1] < 1e-5


def test_scalar_path_matches_vector_path(kernel):
    """m=1 标量快路径与 m=2 向量路径在同一输入上必须一致（1e-12 量级）。"""
    fn, ns = kernel
    n = 20000
    sig = 0.02 * np.cos(np.arange(n) * 0.02 + 0.3)
    one = fn(sig, 1.0, 0.02, lam=0.72, beta=0.0, n_cycles=120.0)["pstd"][0]
    two = fn(np.column_stack([sig, sig]), 1.0, 0.02, lam=[0.72, 0.72],
             beta=[0.0, 0.0], n_cycles=120.0)["pstd"]
    assert two[0] == pytest.approx(one, rel=1e-12)
    assert two[1] == pytest.approx(one, rel=1e-12)


def test_sig_column_mismatch_raises(kernel):
    """给 3 列的序列却只有 2 路振子，必须报错而不是静默广播。"""
    fn, ns = kernel
    sig = np.zeros((100, 3))
    with pytest.raises(ValueError):
        fn(sig, 1.0, 0.02, lam=[0.72, 0.76], n_cycles=5.0)
