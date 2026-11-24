"""在 .venv 中实跑 s2-jupyterbook 的全部 notebook，输出内嵌到 ipynb。"""
from __future__ import annotations

import pathlib
import sys

import nbformat
from nbclient import NotebookClient

BOOK = pathlib.Path(__file__).resolve().parent.parent
KERNEL = "c2book"


def main() -> int:
    only = set(sys.argv[1:])
    bad = 0
    for p in sorted(BOOK.glob("0*.ipynb")):
        if only and p.name not in only and p.stem not in only:
            continue
        nb = nbformat.read(p, as_version=4)
        client = NotebookClient(nb, timeout=1800, kernel_name=KERNEL,
                                resources={"metadata": {"path": str(BOOK)}},
                                allow_errors=True)
        client.execute()
        nbformat.write(nb, p)
        errs = [o for c in nb.cells for o in c.get("outputs", [])
                if o.get("output_type") == "error"]
        n_img = sum(1 for c in nb.cells for o in c.get("outputs", [])
                    if "data" in o and any(k.startswith("image/") for k in o["data"]))
        flag = "OK " if not errs else f"ERR({len(errs)})"
        print(f"{flag} {p.name:<24} 单元 {len(nb.cells):2d}  图 {n_img}")
        for e in errs:
            print("     ", e.get("ename"), e.get("evalue"))
        bad += len(errs)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
