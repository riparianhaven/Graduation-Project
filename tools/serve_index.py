# -*- coding: utf-8 -*-
"""毕设资料索引的本地预览服务。

用法：
    py serve_index.py            # 默认 127.0.0.1:8765
    py serve_index.py 9000       # 换端口

只绑定 127.0.0.1，不对外暴露。对 .md 请求返回排版后的 HTML，
PDF / .bib / .txt / .py 按原样提供。?raw=1 可查看 Markdown 源码。
"""
import html
import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

# 本脚本位于 tools/，而 web root 是整个项目根（index.html 与各 Markdown 所在层）。
# 因此向上取一级；若把本文件移回根目录，这里须改回单层 dirname。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def _port():
    """只取第一个纯数字参数，兼容被其它脚本 import 时传入的 --check / --verbose。"""
    for a in sys.argv[1:]:
        if a.isdigit():
            return int(a)
    return 8765


PORT = _port()

CSS = """
:root{--bg:#f6f7f9;--panel:#fff;--line:#e3e7ec;--text:#1b232c;--muted:#63707e;--accent:#2f6fd0;--soft:#eaf1fc}
@media(prefers-color-scheme:dark){:root{--bg:#14181d;--panel:#1d232a;--line:#303a45;--text:#e7edf3;--muted:#a7b3c0;--accent:#7fb0f5;--soft:#21334d}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.8 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
.bar{position:sticky;top:0;background:var(--panel);border-bottom:1px solid var(--line);padding:9px 24px;font-size:13px;display:flex;gap:18px;align-items:center;flex-wrap:wrap}
.bar a{color:var(--accent);text-decoration:none}.bar a:hover{text-decoration:underline}
.bar .fn{color:var(--muted);margin-left:auto;word-break:break-all}
main{max-width:980px;margin:0 auto;padding:26px 28px 70px;background:var(--panel);min-height:calc(100vh - 44px);
  border-left:1px solid var(--line);border-right:1px solid var(--line)}
h1,h2,h3,h4{line-height:1.4;margin:1.5em 0 .6em;font-weight:650}
h1{font-size:25px;border-bottom:1px solid var(--line);padding-bottom:10px;margin-top:.4em}
h2{font-size:20px;border-bottom:1px solid var(--line);padding-bottom:7px}
h3{font-size:17px}h4{font-size:15px}
p,li{color:var(--text)}li{margin:.25em 0}
a{color:var(--accent)}
code{background:var(--soft);padding:1px 5px;border-radius:4px;font:12.6px ui-monospace,Consolas,monospace}
pre{background:var(--soft);border:1px solid var(--line);border-radius:8px;padding:14px 16px;overflow:auto}
pre code{background:none;padding:0}
blockquote{border-left:3px solid var(--accent);background:var(--soft);margin:12px 0;padding:8px 16px;color:var(--muted)}
blockquote p{margin:.3em 0}
.staticnote{max-width:980px;margin:14px auto -8px;padding:9px 14px;border:1px solid var(--line);
  border-left:3px solid var(--accent);border-radius:0 8px 8px 0;background:var(--soft);
  font-size:12.6px;color:var(--muted)}
.staticnote a{color:var(--accent)}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:13.4px;display:block;overflow:auto}
th,td{border:1px solid var(--line);padding:8px 11px;text-align:left;vertical-align:top}
th{background:var(--soft);color:var(--muted);white-space:nowrap}
tr:nth-child(even) td{background:rgba(127,150,180,.045)}
hr{border:none;border-top:1px solid var(--line);margin:26px 0}
strong{font-weight:650}
"""

PAGE = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>{css}</style>
<script>window.MathJax={{tex:{{inlineMath:[['$','$'],['\\\\(','\\\\)']]}},
  options:{{skipHtmlTags:['script','noscript','style','textarea','pre','code','annotation']}}}};</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
</head><body>
<div class="bar"><a href="{back}">← 返回资料索引</a>
<a href="{raw}">查看 Markdown 源码</a><span class="fn">{name}</span></div>
{note}<main>{body}</main></body></html>"""


def inline(s):
    """行内标记：图片、代码、粗体、斜体、链接。先占位避免二次解析。"""
    holds = []

    def hold(tag):
        holds.append(tag)
        return '\x00%d\x00' % (len(holds) - 1)

    def code(m):
        return hold('<code>%s</code>' % html.escape(m.group(1), quote=False))

    def image(m):
        alt = html.escape(m.group(1), quote=True)
        url = html.escape(m.group(2).replace('\\', '/'), quote=True)
        return hold('<img src="%s" alt="%s" style="display:block;max-width:100%%;height:auto;margin:10px 0">'
                    % (url, alt))

    s = re.sub(r'`([^`]+)`', code, s)
    s = re.sub(r'!\[([^\]]*)\]\(([^)\s]+)\)', image, s)
    s = html.escape(s, quote=False)

    def link(m):
        # 文档里常用 Windows 反斜杠写相对路径，浏览器只认正斜杠
        url = m.group(2).replace('\\', '/')
        return '<a href="%s">%s</a>' % (url, m.group(1))

    s = re.sub(r'\[([^\]]+)\]\(([^)\s]+)\)', link, s)
    s = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', s)
    s = re.sub(r'(?<!\*)\*([^*\n]+)\*(?!\*)', r'<em>\1</em>', s)
    s = re.sub(r'\x00(\d+)\x00', lambda m: holds[int(m.group(1))], s)
    return s


def table_row(line):
    cells = [c.strip() for c in line.strip().strip('|').split('|')]
    return cells


def render_md(text, name, back='/index.html', raw=None, note=''):
    lines = text.splitlines()
    out, i, n = [], 0, len(lines)
    while i < n:
        ln = lines[i]
        st = ln.strip()

        if st.startswith('```'):
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith('```'):
                buf.append(html.escape(lines[i], quote=False))
                i += 1
            i += 1
            out.append('<pre><code>' + '\n'.join(buf) + '</code></pre>')
            continue

        if st.startswith('|') and i + 1 < n and re.match(r'^\s*\|[\s:|-]+\|\s*$', lines[i + 1]):
            head = table_row(ln)
            i += 2
            rows = []
            while i < n and lines[i].strip().startswith('|'):
                rows.append(table_row(lines[i]))
                i += 1
            out.append('<table><thead><tr>' + ''.join('<th>%s</th>' % inline(c) for c in head) + '</tr></thead><tbody>')
            for r in rows:
                out.append('<tr>' + ''.join('<td>%s</td>' % inline(c) for c in r) + '</tr>')
            out.append('</tbody></table>')
            continue

        m = re.match(r'^(#{1,6})\s+(.*)$', st)
        if m:
            lv = len(m.group(1))
            out.append('<h%d>%s</h%d>' % (lv, inline(m.group(2)), lv))
            i += 1
            continue

        if re.match(r'^(---+|\*\*\*+|___+)$', st):
            out.append('<hr>')
            i += 1
            continue

        if st.startswith('> '):
            buf = []
            while i < n and lines[i].strip().startswith('>'):
                buf.append(inline(lines[i].strip().lstrip('>').strip()))
                i += 1
            out.append('<blockquote>' + ''.join('<p>%s</p>' % b for b in buf) + '</blockquote>')
            continue

        if re.match(r'^\s*([-*+]|\d+\.)\s+', ln):
            ordered = bool(re.match(r'^\s*\d+\.\s+', ln))
            tag = 'ol' if ordered else 'ul'
            buf = []
            while i < n and re.match(r'^\s*([-*+]|\d+\.)\s+', lines[i]):
                buf.append('<li>%s</li>' % inline(re.sub(r'^\s*([-*+]|\d+\.)\s+', '', lines[i])))
                i += 1
            out.append('<%s>%s</%s>' % (tag, ''.join(buf), tag))
            continue

        if not st:
            i += 1
            continue

        buf = []
        while i < n and lines[i].strip() and not re.match(
                r'^\s*(#{1,6}\s|\||>|```|[-*+]\s|\d+\.\s|---+\s*$)', lines[i]):
            buf.append(inline(lines[i].strip()))
            i += 1
        if buf:
            out.append('<p>' + '<br>'.join(buf) + '</p>')
        else:
            i += 1

    if raw is None:
        raw = '/' + name + '?raw=1'
    return PAGE.format(title=html.escape(name), css=CSS, body='\n'.join(out),
                       back=back, raw=raw, note=note, name=html.escape(name))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path.endswith('.md') and 'raw=1' not in self.path:
            full = os.path.normpath(os.path.join(ROOT, path.lstrip('/')))
            if full.startswith(ROOT) and os.path.isfile(full):
                try:
                    with open(full, encoding='utf-8') as f:
                        body = render_md(f.read(), path.lstrip('/'))
                    data = body.encode('utf-8')
                except Exception as e:
                    return super().do_GET()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
        return super().do_GET()

    def log_message(self, fmt, *args):
        if '--verbose' in sys.argv:
            super().log_message(fmt, *args)


if __name__ == '__main__':
    url = 'http://127.0.0.1:%d/index.html' % PORT
    print('毕设资料索引:', url, flush=True)
    print('停止服务: Ctrl+C', flush=True)
    ThreadingHTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
