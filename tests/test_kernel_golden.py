# -*- coding: utf-8 -*-
"""内核黄金样本测试：优化前后必须**逐位相同**。

`tests/golden/kernel_ref.json` 是 2026-11-26 性能优化**之前**用当时的
`simulate_seq` 冻结下来的（算例见 `tools/kernel_cases.py`），里面存了

    pstd / mx       —— 用 JSON 浮点（repr）存，round-trip 精确
    p.shape、p 的 sha256
    t.shape、X 的 sha256（trace=True 的算例）

为什么用 sha256 而不是容差：优化只允许改**求值方式**（预算驱动项、换
Python float、复用缓冲、C 序归约），不允许改任何一个浮点运算。混沌态下
1 ulp 的差异会被指数放大，只有哈希相等才真的说明"数值没动"。

**不要为了让测试过而重新冻结**。它过不了就说明数值被改动了；只有在内核
被有意识地升级（且论文里的数都已同步）之后，才该按
`tests/golden/README.md` 重做一次。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import kernel_cases  # noqa: E402
from kernel_src import load_kernel  # noqa: E402

SIM, NS = load_kernel()
REF = json.loads((ROOT / "tests/golden/kernel_ref.json").read_text(encoding="utf-8"))
CASES = kernel_cases.cases()


def test_golden_covers_all_cases():
    """JSON 里的算例与算例矩阵必须一一对应（防止加了算例忘了冻结）。"""
    assert sorted(CASES) == sorted(c["name"] for c in REF)


@pytest.mark.parametrize("case", REF, ids=[c["name"] for c in REF])
def test_matches_frozen_output_bitwise(case):
    sig, kw = CASES[case["name"]]
    assert int(np.asarray(sig).size) == case["args"]["n"], "输入构造与冻结时不一致"
    r = SIM(sig, **kw)

    assert np.array_equal(r["pstd"], np.array(case["pstd"])), (
        f"{case['name']}: pstd 不再逐位相同 —— 内核数值被改动了")
    assert np.array_equal(r["mx"], np.array(case["mx"])), f"{case['name']}: mx 变了"
    assert list(r["p"].shape) == case["p_shape"], f"{case['name']}: p 形状变了"
    got = hashlib.sha256(np.ascontiguousarray(r["p"]).tobytes()).hexdigest()
    assert got == case["p_sha256"], f"{case['name']}: Poincaré 点逐位不同"

    if case["t_shape"] is None:
        assert r["t"] is None and r["X"] is None and r["Y"] is None
    else:
        assert list(np.asarray(r["t"]).shape) == case["t_shape"]
        assert (hashlib.sha256(np.ascontiguousarray(r["X"]).tobytes()).hexdigest()
                == case["X_sha256"]), f"{case['name']}: trace 的 X 逐位不同"


def test_two_paths_agree_at_the_switch_boundary():
    """微内核（m≤32）与阵列路径（m>32）在同一问题上必须逐位相同。

    这是"换求值方式不改数值"的核心契约：把同一列复制到 32 路（微内核）
    与 33 路（阵列）上，前 32 路的结果必须完全一样。
    """
    rng = np.random.default_rng(3)
    n = 6000
    sig = 0.02 * np.cos(np.arange(n) * 0.02) + 0.01 * rng.standard_normal(n)
    nc = kernel_cases.cycles(n, 10.0, 0.002)
    beta = [0.02 * j for j in range(33)]
    small = SIM(sig, 10.0, 0.002, lam=[0.73] * 32, beta=beta[:32], n_cycles=nc)
    big = SIM(sig, 10.0, 0.002, lam=[0.73] * 33, beta=beta, n_cycles=nc)
    assert np.array_equal(small["pstd"], big["pstd"][:32])
    assert np.array_equal(small["mx"], big["mx"][:32])
    assert np.array_equal(small["p"], big["p"][:, :32])
