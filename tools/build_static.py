# -*- coding: utf-8 -*-
"""把 index.html 引用的 Markdown 预渲染成同名 .html 静态副本。

为什么要这一步：浏览器不认识 Markdown。直接双击 index.html（file://）时，
链接指向 .md 只会显示源码；有了同名 .html 副本，页面脚本会自动把链接切过去，
于是双击也能看到排版后的文档，不必启动本地服务。

用法：
    py build_static.py          # 重新生成全部静态副本
    py build_static.py --check  # 只检查哪些副本过期，不写文件

Markdown 内容改动后重跑一次即可；8765 服务模式下始终是实时渲染，不受影响。
"""
import html
import os
import re
import sys
from datetime import datetime

from serve_index import render_md

# 本脚本位于 tools/；静态副本必须生成在源 Markdown 旁边（项目根与 文献/ 下），
# 所以 ROOT 取项目根，而不是脚本自身目录。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(ROOT, 'index.html')


def md_links():
    s = open(INDEX, encoding='utf-8').read()
    links = re.findall(r'href="([^"#]+\.md)(?:#[^"]*)?"', s)
    out = []
    for x in links:
        rel = html.unescape(x).replace('\\', '/')
        if not rel.startswith(('http', '#')) and rel not in out:
            out.append(rel)
    return out


def rewrite_md_links(page, rel, available=()):
    """静态副本里，正文指向其它 Markdown 的链接改指同名排版页。

    顶栏「查看 Markdown 源码」必须保留 .md，用占位符保护。
    副本已存在、或该 .md 属于本轮要生成的清单时改写；否则不改写，避免死链。
    传 available 是为了修掉「首次新增互链文档时，先生成的副本漏改」的问题。
    """
    guard = re.search(r'(<a href="[^"]+\.md">查看 Markdown 源码</a>)', page)
    token = '\x01RAW_LINK\x01'
    if guard:
        page = page.replace(guard.group(1), token, 1)

    def repl(m):
        pre, path, anchor, post = m.groups()
        target = os.path.normpath(os.path.join(ROOT, os.path.dirname(rel),
                                              html.unescape(path) + '.md'))
        dst = os.path.splitext(target)[0] + '.html'
        key = os.path.relpath(target, ROOT).replace('\\', '/')
        if os.path.isfile(dst) or key in available:
            return pre + path + '.html' + (anchor or '') + post
        return m.group(0)

    page = re.sub(r'(href=")([^"]+)\.md(#[^"]*)?(")', repl, page)
    if guard:
        page = page.replace(token, guard.group(1), 1)
    return page


def main():
    check = '--check' in sys.argv
    stamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    made, stale, skipped = [], [], []
    links = md_links()
    available = {os.path.normpath(x).replace('\\', '/') for x in links}

    for rel in links:
        src = os.path.normpath(os.path.join(ROOT, rel))
        if not os.path.isfile(src):
            skipped.append(rel)
            continue
        dst = os.path.splitext(src)[0] + '.html'
        exists = os.path.isfile(dst)

        if check:
            if not exists or os.path.getmtime(dst) < os.path.getmtime(src):
                stale.append(rel)
            continue

        depth = rel.count('/')
        back = '../' * depth + 'index.html'
        raw = os.path.basename(rel)
        note = (
            '<div class="staticnote">这是 <b>静态排版副本</b>，生成于 %s。'
            '直接双击 <code>index.html</code> 时看到的就是它；'
            '若 Markdown 刚改过，请在工作区运行 <code>py build_static.py</code> 重新生成，'
            '或用 <code>start_index.bat</code> 启动本地服务查看实时渲染版。'
            '页内公式由 MathJax 渲染，需要联网。</div>'
        ) % stamp

        text = open(src, encoding='utf-8').read()
        out = render_md(text, rel, back=back, raw=raw, note=note)
        out = rewrite_md_links(out, rel, available)
        with open(dst, 'w', encoding='utf-8') as f:
            f.write(out)
        made.append((rel, os.path.relpath(dst, ROOT), len(out)))

    if check:
        print('需要重新生成的副本:', stale or '无（全部为最新）')
        return

    print('已生成 %d 份静态排版副本：' % len(made))
    for rel, dst, size in made:
        print('  %-52s -> %-52s %7d 字符' % (rel, dst, size))
    if skipped:
        print('跳过（源文件不存在）:', skipped)
    print('提示：Markdown 改动后重跑本脚本；start_index.bat 会自动执行一次。')


if __name__ == '__main__':
    main()
