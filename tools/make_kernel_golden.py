# -*- coding: utf-8 -*-
"""重冻结内核黄金样本（`tests/golden/kernel_ref.json`）。

    .venv\\Scripts\\python.exe tools\\make_kernel_golden.py          # 只看差异，不落盘
    .venv\\Scripts\\python.exe tools\\make_kernel_golden.py --yes    # 真的覆盖

**这不是"让测试变绿"的工具**。黄金样本是性能优化/重构前后数值不变的
证据链，只有在内核被*有意识地*升级（换了离散化、改了步长口径、论文里的
数也跟着改了）之后才该重做一次，然后在提交说明里写清"为什么数值该变"。

算例来自 `tools/kernel_cases.py`（测试与基准脚本共用同一份，不许各写一套）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import kernel_cases  # noqa: E402
from kernel_src import load_kernel  # noqa: E402

OUT = ROOT / "tests/golden/kernel_ref.json"


def build() -> list[dict]:
    sim, _ = load_kernel()
    ref = []
    for name, (sig, kw) in kernel_cases.cases().items():
        r = sim(sig, **kw)
        p = r["p"]
        ref.append(dict(
            name=name,
            args=dict(n=int(np.asarray(sig).size), m=int(np.atleast_1d(r["pstd"]).size),
                      R=kw["R"], dt=kw["dt"], n_cycles=kw["n_cycles"],
                      burn_frac=kw.get("burn_frac", 0.5),
                      trace=bool(kw.get("trace", False)),
                      lam=[float(v) for v in np.atleast_1d(kw["lam"])],
                      beta=[float(v) for v in np.atleast_1d(kw.get("beta", 0.0))]),
            pstd=[float(v) for v in r["pstd"]],
            mx=[float(v) for v in r["mx"]],
            p_shape=list(p.shape),
            p_sha256=hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest(),
            t_shape=None if r["t"] is None else list(np.asarray(r["t"]).shape),
            X_sha256=None if r["X"] is None else hashlib.sha256(
                np.ascontiguousarray(r["X"]).tobytes()).hexdigest(),
        ))
    return ref


def main() -> int:
    ap = argparse.ArgumentParser(description="重冻结内核黄金样本")
    ap.add_argument("--yes", action="store_true", help="真的覆盖 JSON（否则只比对）")
    a = ap.parse_args()

    new = build()
    old = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else []
    old_by = {c["name"]: c for c in old}

    changed = [c["name"] for c in new
               if old_by.get(c["name"], {}).get("p_sha256") != c["p_sha256"]]
    for c in new:
        print(f"{c['name']:<20} pstd={np.array2string(np.array(c['pstd']), precision=6)}")
    print()
    if not old:
        print("没有旧样本，将首次写入。")
    elif changed:
        print(f"与旧样本**不同**的算例 {len(changed)} 个：{changed}")
        print("如果这次改动本来不该动数值，先别覆盖——去查内核。")
    else:
        print("与旧样本逐位相同。")

    if not a.yes:
        print("\n（未落盘；确认后加 --yes）")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(new, indent=1), encoding="utf-8")
    print("\n已写入", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
