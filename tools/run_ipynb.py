# -*- coding: utf-8 -*-
"""用 .venv 重跑 notebook：执行全部单元，可选把输出写回文件。

在虚拟环境里运行（双击 run_ipynb.bat 更省事）：
    .venv\\Scripts\\python.exe run_ipynb.py                    # 重跑并写回原 ipynb
    .venv\\Scripts\\python.exe run_ipynb.py --no-write         # 只验证，不改动 ipynb
    .venv\\Scripts\\python.exe run_ipynb.py --workdir <dir>    # 图纸另存到指定目录
    .venv\\Scripts\\python.exe run_ipynb.py --input other.ipynb --output out.ipynb

默认工作目录是工作区根目录，notebook 会把 10 张 PNG 写进「复现输出/」，
并更新「复现输出/结果汇总.json」；这是重跑的预期行为，写论文前请注意版本一致。
"""
import argparse
import pathlib
import sys
import time

import nbformat
from nbclient import NotebookClient

# 本脚本位于 tools/；notebook 与 复现输出/ 都在项目根，因此默认工作目录取上一级。
ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_INPUT = '复现-2-混沌振子建模与临界阈值.ipynb'


def main():
    ap = argparse.ArgumentParser(description='Run a notebook with the .venv kernel')
    ap.add_argument('--input', default=DEFAULT_INPUT, help='notebook 路径（相对工作区根目录）')
    ap.add_argument('--output', default=None, help='另存为；默认写回 --input')
    ap.add_argument('--no-write', action='store_true', help='只执行、不写回 ipynb')
    ap.add_argument('--workdir', default=None, help='notebook 的工作目录，默认工作区根目录')
    ap.add_argument('--timeout', type=int, default=600, help='单个单元超时秒数')
    args = ap.parse_args()

    src = (ROOT / args.input).resolve()
    if not src.is_file():
        sys.exit('[ERROR] notebook 不存在: %s' % src)

    work = pathlib.Path(args.workdir).resolve() if args.workdir else ROOT
    work.mkdir(parents=True, exist_ok=True)
    (work / '复现输出').mkdir(exist_ok=True)

    nb = nbformat.read(src, as_version=4)
    print('notebook :', src.name)
    print('kernel   :', nb.metadata.get('kernelspec', {}).get('name', 'python3'), '(venv)')
    print('workdir  :', work, flush=True)

    client = NotebookClient(
        nb, timeout=args.timeout, kernel_name='python3',
        resources={'metadata': {'path': str(work)}},
        allow_errors=False,
    )
    t0 = time.time()
    client.execute()

    code = [c for c in nb.cells if c.cell_type == 'code']
    errors = [o for c in code for o in c.get('outputs', []) if o.output_type == 'error']
    imgs = [o for c in code for o in c.get('outputs', []) if 'image/png' in o.get('data', {})]
    print('\n耗时 %.1f s | 代码单元 %d | 有输出 %d | 内嵌图 %d | 报错 %d'
          % (time.time() - t0, len(code), sum(bool(c.get('outputs')) for c in code),
             len(imgs), len(errors)))
    for e in errors:
        print('  ERROR:', e.get('ename'), e.get('evalue'))
    if errors:
        sys.exit(1)

    if not args.no_write:
        dst = (ROOT / args.output).resolve() if args.output else src
        nbformat.write(nb, dst)
        print('已写回 :', dst)
    else:
        print('已跳过写回（--no-write）')
    print('图纸    :', work / '复现输出')
    print('PASS')


if __name__ == '__main__':
    main()
