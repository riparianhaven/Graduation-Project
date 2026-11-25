# -*- coding: utf-8 -*-
"""把 `tools/kernel_src.py` 的内核正文同步进各 notebook 的代码单元。

    .venv\\Scripts\\python.exe tools\\sync_kernel_notebooks.py           # 只报差异
    .venv\\Scripts\\python.exe tools\\sync_kernel_notebooks.py --write   # 真的写回

为什么需要它：内核在 notebook 里是**内嵌源码**（notebook 保持自包含，不
import tools），所以改了 `KERNEL_SRC` 之后 6 份 notebook 的正文会过期，
`tests/test_kernel_contract.py` 会抓出来。正规流程是"改内核 → 重新生成
（build_stage*_notebook.py）→ 重跑（run_ipynb.py）"，那会重算所有输出。

本脚本走的是另一条等价路径：**只替换内核单元里的那段正文，保留已有输出**。
适用于"这次改动逐位不改变数值"的重构（性能优化、注释、变量名），
`tests/test_kernel_golden.py` 就是这条前提的证明。**改了数值的升级不许用
这个脚本**——那样 notebook 里留的是旧输出，图和数会对不上。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from kernel_src import KERNEL_SRC  # noqa: E402

NB = ["learn/2.ipynb", "learn/3.ipynb", "learn/4.ipynb", "learn/5.ipynb",
      "learn/all.ipynb", "复现-2-混沌振子建模与临界阈值.ipynb"]

RETURN_LINE = "    return dict(pstd=pstd, mx=np.atleast_1d(mx)"
HEAD = "def simulate_seq("


def swap(src: str, want: str) -> tuple[str, bool]:
    """把单元源码里的内核正文换成 want，返回 (新源码, 是否改过)。

    结尾要**按行首锚定**去找：内核里还有一处 `        return dict(pstd=pstd, …)`
    （微内核分支，8 空格缩进），用 str.index 找子串会误命中它，于是"内核正文"
    被截在分支 return 上、脚本永远认为需要更新，--write 还会把后半个函数重复
    粘一遍。
    """
    if HEAD not in src:
        return src, False
    i0 = src.index(HEAD)
    m = re.search(r"^" + re.escape(RETURN_LINE), src[i0:], re.M)
    if m is None:
        return src, False
    j = i0 + m.start()
    k = src.find("\n", j)
    i1 = len(src) if k < 0 else k + 1           # 含换行，后面原样保留
    if src[i0:i1].rstrip("\n") == want:
        return src, False
    return src[:i0] + want + "\n" + src[i1:], True


def main() -> int:
    ap = argparse.ArgumentParser(description="同步 notebook 内嵌内核正文")
    ap.add_argument("--write", action="store_true", help="写回（否则只报差异）")
    a = ap.parse_args()

    want = KERNEL_SRC.strip("\n")
    n_changed = 0
    for rel in NB:
        p = ROOT / rel
        nb = nbformat.read(str(p), as_version=4)
        touched = 0
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            new, changed = swap(cell.source, want)
            if changed:
                cell.source = new
                touched += 1
        if touched:
            n_changed += 1
            print(f"{rel}: 需要更新 {touched} 个单元")
            if a.write:
                nbformat.write(nb, str(p))
                print(f"    已写回（输出保留）")
        else:
            print(f"{rel}: 已是最新")
    print(f"\n共 {n_changed} 份需要更新" + ("" if a.write else "（加 --write 落盘）"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
