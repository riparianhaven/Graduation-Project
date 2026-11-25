# -*- coding: utf-8 -*-
"""性能基准 + 数值一致性校验（一条命令出完整证据）。

    .venv\\Scripts\\python.exe tools\\bench_speed.py
    .venv\\Scripts\\python.exe tools\\bench_speed.py --reps 3

做两件事：

1. **一致性**：把 `tools/kernel_cases.py` 的每个算例跑一遍，与
   `tests/golden/kernel_ref.json`（优化前冻结的 Poincaré 点 sha256）比对。
   逐位相同才算通过——不是"误差小于多少"。
2. **速度**：每个场景多跑几遍取最小值，与 `BASELINE`（同一台机器、优化前
   实测）对比。**绝对值只在同一台机器上有意义，跨机看加速比**。

结果同时写到 `复现输出/性能基准.json`，供文档引用。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import kernel_cases as KC  # noqa: E402
import link_proto as LP  # noqa: E402
from kernel_src import load_kernel  # noqa: E402

SIM, _ = load_kernel()
OUT = ROOT / "复现输出/性能基准.json"

#: 优化前（2026-11-26，Python 3.13.9 / numpy 2.4.6 / 本机）实测，单位秒。
#: 只用来算加速比；换机器后绝对值不可比。
BASELINE = {
    "kernel m=1（单振子，47100 步）": 0.133,
    "kernel m=3（三相位阵列）": 1.448,
    "kernel m=45（阵列路径）": 1.751,
    "kernel m=180（估幅网格）": 2.591,
    "analyze(known) 一帧判决": 1.513,
    "analyze(auto) 盲测扫 R": 3.631,
    "scan_R 两级细扫（16 次单振子）": 2.047,
    "estimate_amp λ* 差分": 2.650,
    "framework_demo 200×2 次": 0.045,
    "make_frame 一帧 94.2 s": 0.001,
}


def timed(fn, reps: int) -> float:
    fn()                                   # 预热（首次调用有 import/缓存开销）
    best = float("inf")
    for _ in range(max(1, reps)):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def check_golden() -> dict:
    ref_p = ROOT / "tests/golden/kernel_ref.json"
    if not ref_p.exists():
        return dict(ok=False, msg="缺少 tests/golden/kernel_ref.json")
    ref = json.loads(ref_p.read_text(encoding="utf-8"))
    cases = KC.cases()
    bad = []
    for c in ref:
        sig, kw = cases[c["name"]]
        r = SIM(sig, **kw)
        got = hashlib.sha256(np.ascontiguousarray(r["p"]).tobytes()).hexdigest()
        if (got != c["p_sha256"]
                or not np.array_equal(r["pstd"], np.array(c["pstd"]))):
            bad.append(c["name"])
    return dict(ok=not bad, total=len(ref), bad=bad)


def main() -> int:
    ap = argparse.ArgumentParser(description="内核/链路性能基准")
    ap.add_argument("--reps", type=int, default=1, help="每个场景重复次数（取最小值）")
    ap.add_argument("--skip-golden", action="store_true", help="跳过逐位校验")
    a = ap.parse_args()

    g = dict(ok=None)
    if not a.skip_golden:
        t0 = time.perf_counter()
        g = check_golden()
        g["seconds"] = round(time.perf_counter() - t0, 2)

    fr = KC.frame()
    n = fr.size
    nc = KC.cycles(n, 10.0, 0.002)
    b3 = np.array(KC.B3)
    lam3 = np.full(3, 0.73)
    # 估幅网格：4 相位 × 45 个 λ（与 link_proto.estimate_amp 同构）
    lams = np.tile(np.arange(0.680, 0.7901, 0.0025), 4)
    betas = np.repeat([0.0, math.pi / 2, math.pi, 3 * math.pi / 2], 45)

    scen = [
        ("kernel m=1（单振子，47100 步）",
         lambda: SIM(fr, 10.0, 0.002, lam=0.73, beta=0.0, n_cycles=nc)),
        ("kernel m=3（三相位阵列）",
         lambda: SIM(fr, 10.0, 0.002, lam=lam3, beta=b3, n_cycles=nc)),
        ("kernel m=45（阵列路径）",
         lambda: SIM(fr, 10.0, 0.002, lam=np.full(45, 0.73),
                     beta=np.arange(45) * 0.07, n_cycles=nc)),
        ("kernel m=180（估幅网格）",
         lambda: SIM(np.broadcast_to(fr.reshape(-1, 1), (n, 180)), 10.0, 0.002,
                     lam=lams, beta=betas, n_cycles=nc)),
        ("analyze(known) 一帧判决",
         lambda: LP.analyze(fr, 0.002, mode="known", R_manual=10.0, lam=0.73)),
        ("analyze(auto) 盲测扫 R",
         lambda: LP.analyze(fr, 0.002, mode="auto", lam=0.73)),
        ("scan_R 两级细扫（16 次单振子）",
         lambda: LP.scan_R(fr, 0.002, 10.0, 0.73)),
        ("estimate_amp λ* 差分",
         lambda: LP.estimate_amp(fr, 10.0, 0.002)),
        ("framework_demo 200×2 次",
         lambda: LP.framework_demo(fs=500.0, f0=10.0 / KC.TWOPI, amp=0.03,
                                   sigma_samp=0.00707, dur=20.0, n_mc=200,
                                   seed=9, h1=0.02)),
        ("make_frame 一帧 94.2 s",
         lambda: LP.make_frame(w=10.0, fs=500.0, duration=94.2, amp=0.02,
                               sigma=0.05, rng=np.random.default_rng(1))),
    ]

    rows = []
    print("\n%-34s %10s %10s %8s" % ("场景", "优化前/s", "现在/s", "加速"))
    print("-" * 68)
    for name, fn in scen:
        t = timed(fn, a.reps)
        base = BASELINE.get(name)
        speed = (base / t) if base else float("nan")
        rows.append(dict(name=name, before_s=base, now_s=round(t, 4),
                         speedup=round(speed, 2)))
        print("%-34s %10.3f %10.3f %7.1f×" % (name, base, t, speed))

    tot_b = sum(r["before_s"] for r in rows if r["name"].startswith("analyze"))
    tot_n = sum(r["now_s"] for r in rows if r["name"].startswith("analyze"))
    print("-" * 68)
    print("一帧「盲测判决」合计：%.3f s → %.3f s（%.1f×）"
          % (tot_b, tot_n, tot_b / tot_n))

    if g.get("ok") is not None:
        print("\n逐位校验：%s（%d 个算例，%.2f s）"
              % ("全部一致" if g["ok"] else f"不一致 {g['bad']}",
                 g.get("total", 0), g.get("seconds", 0)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(rows=rows, golden=g, reps=a.reps),
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print("已写", OUT)
    return 0 if g.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
