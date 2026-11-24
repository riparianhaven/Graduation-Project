# 环境说明：用 .venv 跑 ipynb

- 环境位置：`C:\Users\22902\WorkBuddy\毕设\.venv`（Python 3.13.9，Windows，约 353 MB）
- **最省事的一条路**：双击 [start_notebook.bat](../start_notebook.bat) → 浏览器自动打开 JupyterLab → 打开 `复现-2-混沌振子建模与临界阈值.ipynb` → Kernel 选 `Python 3 (毕设 .venv)` → Run All。
- 已实跑验证（2026-11-14）：在虚拟环境里跑 notebook 的**临时副本**，39.8 秒、13 个代码单元全部有输出、10 张内嵌图、0 报错；验证过程未改动已交付的 ipynb 与 `复现输出/` 里的图纸。
- 本环境只服务于 [notebook 复现](../复现-2-混沌振子建模与临界阈值.html)；[阅读路线](阅读路线.md)、[第 0 步方框图](第0步-技术路线方框图.md)、`index.html` 的排版服务用的是系统 `py`，两者互不干扰。

| 文件 | 作用 |
|---|---|
| `requirements.txt` | 顶层依赖与锁定版本，换机器重装用 |
| `start_notebook.bat` | 一键启动 JupyterLab（虚拟环境的 Python） |
| `run_ipynb.bat` / `tools/run_ipynb.py` | 不开浏览器，直接重跑 notebook 并把输出写回 |
| `.venv\share\jupyter\kernels\python3\kernel.json` | Kernel 定义，已指向本工作区的虚拟环境 |

---

## 一、三种打开方式

| 方式 | 怎么做 | 适合 |
|---|---|---|
| **① JupyterLab（推荐）** | 双击 `start_notebook.bat`，浏览器自动打开 `127.0.0.1:8888/lab`；左侧双击 ipynb，右上角 Kernel 选 `Python 3 (毕设 .venv)`，菜单 Run → Run All Cells | 想边看边改参数、当场看 10 张图 |
| **② VS Code** | 打开 `.ipynb` → 右上角「选择内核」→「Python 环境」→ 选 `.\.venv\Scripts\python.exe` → 全部运行 | 已经习惯 VS Code、要断点调试 |
| **③ 命令行重跑** | 双击 `run_ipynb.bat`（等价于 `.venv\Scripts\python.exe tools/run_ipynb.py`），等约 40 秒看 PASS | 只想刷新结果、不想开界面；可传给别人复现 |

`tools/run_ipynb.py` 常用参数：

```
.venv\Scripts\python.exe tools/run_ipynb.py --no-write                  # 只验证，不改动 ipynb
.venv\Scripts\python.exe tools/run_ipynb.py --workdir D:\tmp\repro     # 图纸另存到别处
.venv\Scripts\python.exe tools/run_ipynb.py --input foo.ipynb --output bar.ipynb
```

> ⚠️ 默认重跑会把输出**写回原 ipynb**，并覆盖 `复现输出\` 下 10 张 PNG 与 `结果汇总.json`。这是预期行为，但**写论文前请确认曲线数字与文档里引用的版本一致**；只想检查环境就用 `--no-write --workdir` 指到临时目录。

---

## 二、装了什么（顶层依赖，版本已锁定）

| 包 | 版本 | 用途 |
|---|---|---|
| numpy | 2.4.6 | RK4 积分、扫描、统计 |
| matplotlib | 3.11.0 | 10 张图，中文字体 Microsoft YaHei |
| ipykernel | 6.30.0 | notebook 的 Python 内核 |
| nbformat | 5.11.1 | 读写 .ipynb |
| nbclient | 0.11.0 | 无界面执行全部单元 |
| nbconvert | 7.17.2 | 导出 HTML 预览 |
| jupyterlab | 4.4.5 | 浏览器界面 |

版本与交付 notebook 实跑时**逐一对齐**，避免升级 numpy / matplotlib 后曲线与论文里的数字漂移。完整依赖树可用 `.venv\Scripts\python.exe -m pip freeze` 查看。

> 系统 `py` 里原本就有 numpy / matplotlib / ipykernel / nbformat / nbclient / nbconvert，只是**没有** JupyterLab。虚拟环境是独立副本，不改动、也不依赖系统 Python。

---

## 三、换电脑 / 删了 .venv 怎么重建

在工作区根目录执行（Windows、Python 3.13）：

```
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

装完即可双击 `start_notebook.bat`。若 pip 默认源不通，保留上面的清华源参数。

**关于 Kernel 的绝对路径**：当前 `.venv\share\jupyter\kernels\python3\kernel.json` 里 `argv[0]` 写死为
`C:/Users/22902/WorkBuddy/毕设/.venv/Scripts/python.exe`，`display_name` 为 `Python 3 (毕设 .venv)`。这样即使从系统 Jupyter 或 VS Code 启动，选 `python3` 也会落到本虚拟环境。**如果工作区换了路径或重建了 venv，把 `argv[0]` 改成新的 `python.exe` 绝对路径即可**（或重新执行
`.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name python3 --display-name "Python 3 (毕设 .venv)"` 生成默认 kernelspec，此时依赖激活 venv 后的 PATH）。

---

## 四、常见问题

- **中文标题变方块**：一般不会，系统字体 Microsoft YaHei / SimHei 在虚拟环境的 matplotlib 里已确认可用。若换了机器，把 notebook 顶部的 `plt.rcParams['font.sans-serif']` 改成该机器有的中文字体。
- **`Proactor event loop does not implement add_reader` 警告**：Windows + pyzmq 的常见提示，出现在 stderr，**不影响执行结果**，可忽略。
- **8888 端口被占用**：JupyterLab 会自动换端口，以控制台里打印的带 token URL 为准；也可以编辑 `start_notebook.bat` 里的 `--port=8888`。
- **重跑后数字和文档对不上**：先确认没有 `--no-write` 之外的版本混用；`复现输出/结果汇总.json` 记录了当次参数与耗时。
- **`.venv` 不要提交 / 打包**：约 353 MB，且含绝对路径；交付给别人的只需 `requirements.txt` + ipynb + 图纸。
- **关闭服务**：JupyterLab 那个黑色控制台窗口关闭即停止；`index.html` 的 8765 排版服务是另一回事，不受影响。

---

## 五、相关文件

[notebook 实跑预览（网页，无需环境）](../复现-2-混沌振子建模与临界阈值.html)｜[ipynb 源文件](../复现-2-混沌振子建模与临界阈值.ipynb)｜[「主要内容 2」逐句解读](../02-阶段二/解读-主要内容2-混沌振子建模与临界阈值.md)｜[结果汇总.json](../复现输出/结果汇总.json)｜[返回资料索引](../index.html)
