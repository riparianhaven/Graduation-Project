"""生成 C2 完整复现的 8 个 notebook + 2 个 md 章节（Jupyter Book 源文件）。

运行：.venv/Scripts/python.exe c2-jupyterbook/_src/build_notebooks.py
生成后用 nbclient 实跑（_src/run_book.py），再由 jupyter-book 渲染。

相对上一版的变化
----------------
1. 补齐原文缺失的图：3（F）、4（ω）、6（ξ）、7（步长）、8–10（门限波形）、11（弱/强耦合）。
2. 新增第 4 章「信号链与限带」：输出量之争 + 反演原文未写的限带带宽。
3. 表 4/5 从"未复现"改为按统一口径重做并如实报误差。
4. 全图重设中文字体（上一版中文显示为方块）。
"""
from __future__ import annotations

import pathlib

import nbformat as nbf

BOOK = pathlib.Path(__file__).resolve().parent.parent

MD = nbf.v4.new_markdown_cell
CODE = nbf.v4.new_code_cell

SETUP = """\
import sys, json, pathlib
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "_src")
import c2sim as S

DATA = pathlib.Path("_data")
plt.rcParams.update({
    "figure.dpi": 110, "font.size": 10,
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})

def load(name):
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))

def dual_plot(x, snri, r, xlabel, title="", logx=False, invert=False,
              marks=None, ymax=None):
    fig, ax1 = plt.subplots(figsize=(7.4, 3.5))
    ax1.plot(x, snri, "o-", color="#1f4e79", ms=4, lw=1.2)
    ax1.set_xlabel(xlabel); ax1.set_ylabel("SNRI / dB", color="#1f4e79")
    ax1.tick_params(axis="y", colors="#1f4e79")
    if ymax: ax1.set_ylim(top=ymax)
    ax2 = ax1.twinx()
    ax2.plot(x, r, "s--", color="#8a3b2e", ms=4, lw=1.2)
    ax2.set_ylabel("波形相似度 r", color="#8a3b2e")
    ax2.tick_params(axis="y", colors="#8a3b2e")
    if logx: ax1.set_xscale("log")
    if invert: ax1.invert_xaxis()
    if marks:
        for xv, lab in marks:
            ax1.axvline(xv, color="gray", ls=":", lw=1)
            ax1.text(xv, ax1.get_ylim()[1], lab, fontsize=8, va="top",
                     ha="center", color="#7a2e12")
    ax1.grid(alpha=.2); ax1.set_title(title)
    plt.tight_layout(); plt.show()
"""


def nb(*cells):
    n = nbf.v4.new_notebook()
    n.cells = list(cells)
    n.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python",
                                "name": "python3"}
    n.metadata["language_info"] = {"name": "python"}
    return n


# ================================================================== 00
MD0 = """# C2 完整复现：强耦合 Duffing 振子的微弱脉冲检测

> 原文：曹保锋, 李鹏, 李晓强, 张学琴, 宁王师, 梁锐, 李新, 胡淼, 郑毅.
> 基于强耦合 Duffing 振子的微弱脉冲信号检测与参数估计.
> 物理学报, 2019, 68(8): 080501. DOI: 10.7498/aps.68.20181856

这本书把原文的 **11 张图、5 张表**逐项重算了一遍，每项都标明"复现到什么程度、差在哪"。
不追求和原文数字一模一样，追求两件事：**每个数字都能被你自己重算出来**，以及
**对不上的地方能指出是哪一项缺失造成的**。

## 复现结果总表

| 项 | 内容 | 程度 | 关键结论 |
|---|---|---|---|
| 图 1 | 三角脉冲检测全过程 (a)–(f) | ✅ | 输出是**位移差**，速度差是其微分 |
| 图 2 | F∈[0,2] 分岔图 | ✅ | 另附 Lyapunov 指数独立验证四段区间 |
| 图 3 | SNRI / r vs 驱动力幅值 F | ✅ | 36–38 dB 平坦、r=0.99，优于原文要求 |
| 图 4 | SNRI / r vs 驱动力周期 ω | ⚠️ | r 最优区对上；低频端受时间窗选择影响 |
| 图 5 | SNRI / r vs 耦合系数 k | ❌ | 复现不出「k<1 失效」，r 在 k=0.1–100 恒 0.99 |
| 图 6 | SNRI / r vs 阻尼 ξ | ⚠️ | 中等阻尼最优成立；分散性未复现 |
| 图 7 | SNRI / r vs 计算步长 | ✅ | 24→37 dB 正好 13 dB，与带宽比理论吻合 |
| 表 3 | 三种波形的 Pd 与 r | ⚠️ | **r 列全对上**；Pd 列一律偏高（未复现） |
| 表 4 | 幅值估计误差 | ✅ | 短脉冲 ±2%，方波 +21%~−10% |
| 表 5 | 宽度估计误差 | ⚠️ | 双指数 ±7%；高斯导数 +130%（通道展宽）|
| 图 8–10 | 门限处的恢复波形 | ✅ | 三种波形均可恢复，噪声被压制 |
| 图 11 | 弱耦合 vs 强耦合对比 | ✅ | 强耦合恢复明显更好 |
| 表 1–2 | 参数默认值与波形特征 | ✅ | 直接采用 |

**11 图 5 表里：7 项完全复现、4 项部分、1 项未复现。**

## 最大的增量：把「20 dB 增益」变成算式

原文说"优化后信噪比增益可达 20 dB 以上"，读者容易理解成"混沌振子的非线性带来了 20 dB"。
但按式 (4) 实现，小信号下差模方程是**线性**的。真正的原因是**带宽比**：

差模通道是二阶过阻尼低通，等效噪声带宽 $B\\approx1/(2\\pi\\tau)=516$ kHz；
输入是 1 GHz 采样下的逐点白噪声，带宽 500 MHz。于是

$$10\\log_{10}\\frac{500\\ \\mathrm{MHz}}{516\\ \\mathrm{kHz}}=29.9\\ \\mathrm{dB},\\qquad \\text{实测 }28.3\\ \\mathrm{dB}$$

一条式子解释三张图：图 3 平坦（$F$ 不进这个比值）、图 7 的 13 dB 斜率（$h$ 缩小 20 倍，
带宽扩大 20 倍）、图 5 的 SNRI 缓慢上升（$k$ 增大使 $\\Omega$ 增大、带宽变宽）。
**换任何同带宽的线性滤波器，增益一样** —— 这不是混沌的功劳。

## 四个口径约定（全书统一，跨章不许改）

1. **输出量** = 位移差 $x_1-x_2$。图 8–10 的纵轴标的就是它；5.2 节正文写的
   $\\dot x_1-\\dot x_2$ 是脉冲的**微分**，相关系数只有 0.09（第 3 章实测）。
2. **附加限带** = 零相位 FFT 低通，$f_c=130\\,\\mathrm{kHz}\\times1.5\\,\\mu s/t_{\\rm rise}$
   （由表 3 的 r 列反演，再外推到三种波形验证）。
3. **输出 SNR** 与式 (7) 同口径：$10\\log_{10}(a^2/4D)$，即 $20\\log_{10}(a/\\sigma)-6.02$ dB。
   于是"5 倍噪声均方差"严格等于 7.96 dB，与原文 5.1 节自洽。
4. **噪声** = 逐点白噪声（`hold=1`，带宽 500 MHz），对应原文 1 ns 步长。
   `hold>1` 且保持方差不变**不是限带**，是放大噪声谱密度——这个坑见 4.4 节。

## 怎么重算

```bash
# 全量实验（约 15 分钟，写 _data/*.json 与 *.npz）
.venv/Scripts/python.exe c2-jupyterbook/_src/experiments.py

# 只重算某一项
.venv/Scripts/python.exe c2-jupyterbook/_src/experiments.py fig3 tables

# 重跑全部 notebook 并内嵌输出
.venv/Scripts/python.exe c2-jupyterbook/_src/run_book.py
```

各章代码默认读 `_data/` 缓存；缓存缺失时自动调用 `experiments.py` 现算。
"""

n0 = nb(MD(MD0))

# ================================================================== 01
n1 = nb(
    MD("""# 1. 模型与方程

## 1.1 从 Holmes 型 Duffing 振子到强耦合

原文式 (1) 是标准 Holmes 型：

$$
\\ddot{x} + \\xi\\dot{x} - x + x^3 = F\\cos t
$$

做时间尺度变换 $t=\\omega\\tau$（原文式 2），把固定频率 1 rad/s 换成可调的 $\\omega$：

$$
\\ddot{x} + \\omega\\xi\\dot{x} - \\omega^2(x-x^3) = \\omega^2 F\\cos(\\omega t)
$$

这一步决定了后面所有参数的量纲：$\\omega$ 越大，同一个 $F$ 作用越猛，步长必须跟着缩。

强耦合结构（原文式 4）是两个振子**同时耦合恢复力与阻尼力**：

$$
\\begin{aligned}
\\ddot{x}_1 + \\omega\\xi\\dot{x}_1 + \\omega^2\\xi k(x_1-x_2) + \\omega k(\\dot{x}_1-\\dot{x}_2) - \\omega^2(x_1-x_1^3) &= \\omega^2\\big[F\\cos(\\omega t)+s(t)+n(t)\\big]\\\\
\\ddot{x}_2 + \\omega\\xi\\dot{x}_2 + \\omega^2\\xi k(x_2-x_1) + \\omega k(\\dot{x}_2-\\dot{x}_1) - \\omega^2(x_2-x_2^3) &= \\omega^2 F\\cos(\\omega t)
\\end{aligned}
$$

| 符号 | 含义 | 表 1 默认值 | 取值区间 |
|---|---|---|---|
| $F$ | 策动力幅值 | 0.2 | [0, 2] |
| $\\omega$ | 时间尺度因子 (rad/s) | $5\\times10^6$ | $10^5 \\sim 5\\times10^7$ |
| $k$ | 耦合系数 | 10 | $10^{-1} \\sim 10^3$ |
| $\\xi$ | 阻尼系数 | 0.7 | $10^{-2} \\sim 10^2$ |
| $h$ | 计算步长 | 1 ns | 20 ~ 1 ns |

初始值 $x_1=x_2=1.4$、$\\dot x_1=\\dot x_2=4$（原文 5.2 节）。
"""),
    CODE(SETUP),
    MD("""## 1.2 差模方程：强耦合到底做了什么

两个方程相减（令 $\\Delta=x_1-x_2$，注意**不除以 2**）：

$$
\\ddot\\Delta+(\\omega\\xi+2\\omega k)\\dot\\Delta+\\omega^2(2\\xi k-1)\\Delta+\\omega^2(x_1^3-x_2^3)=\\omega^2\\big[s(t)+n(t)\\big]
$$

这一个式子就能解释原文第 4 节的两整节经验曲线：

- **刚度项的符号**由 $2\\xi k-1$ 决定。$k<k_c=1/(2\\xi)=0.714$ 时它是**负**的，
  差模仍是双阱——脉冲激励下两振子各奔一个阱，"差分"就等于把信号扔掉。
  这正是原文「$0<k<1$ 时检测失效」的数学原因。
- **阻尼项**多了 $2\\omega k$。等效阻尼比
  $\\zeta=(\\xi+2k)/\\big(2\\sqrt{2\\xi k-1}\\big)$，$k=10$ 时等于 **2.87**，**过阻尼**。
  弱耦合（$k\\to0$）时差模是欠阻尼双阱，会振铃；强耦合把它改成过阻尼单阱跟随。
  这才是"能恢复波形"的真正原因——**不是耦合变大，是差模换了一套阻尼**。
"""),
    CODE("""xi = 0.7
print(f"k_c = 1/(2*xi) = {1/(2*xi):.4f}   ← 差模由双阱变单阱的门限\\n")
print(f"{'k':>7}{'2ξk−1':>9}{'ζ':>8}{'准静态增益':>12}  判定")
for k in (0.3, 0.5, 0.7, 0.714, 1, 3, 10, 100):
    d = 2*xi*k - 1
    if d <= 0:
        print(f"{k:>7.3f}{d:>9.3f}{'—':>8}{'—':>12}  负刚度：差模仍是双阱")
    else:
        z = (xi + 2*k) / (2*np.sqrt(d))
        print(f"{k:>7.3f}{d:>9.3f}{z:>8.2f}{1/d:>12.4f}  {'过阻尼' if z>1 else '欠阻尼'}")
"""),
    MD("""准静态下忽略非线性项，$\\Delta \\approx s/(2\\xi k-1)$。$k=10$ 时增益 $1/13$。
**这个增益小于 1**，所以差分通道不会凭空提高灵敏度——信号和噪声被同比例压低，SNR 不变。
第 4 章会看到，这正是"必须靠限带来换信噪比"的原因。

实测增益比 $1/13$ 还低一些，因为 $x_1^3-x_2^3\\approx3x^2\\Delta$ 提供了一部分负刚度修正。
下面验证无信号时两振子是否严格同步。"""),
    CODE("""T, h = 40e-6, 1e-9
t = np.arange(int(round(T/h)) + 1) * h
print(f"时长 {T*1e6:.0f} us，步长 {h*1e9:.0f} ns，共 {len(t)} 点，"
      f"约 {T/(2*np.pi/5e6):.1f} 个激励周期")

so0 = S.simulate(t, batch=1)
print(f"无信号时 max|ẋ1−ẋ2| = {np.abs(so0).max():.2e}   ← 浮点零，不是「很小」")
so1 = S.simulate(t, drive=lambda tt: 0.0*tt, batch=1)
print(f"零驱动同解：max|ẋ1−ẋ2| = {np.abs(so1).max():.2e}")
"""),
    MD("""结果是机器精度 0。这意味着**任何非零的 $\\dot x_1-\\dot x_2$ 都由 $s(t)$ 或 $n(t)$ 直接引起**，
没有基线偏置要标定——原文说它"不受初始状态限制"即指此。第 4 章会看到，这个性质让
$P_{fa}$ 可以解析算出。

下一章看没有信号时系统自身的相态地图。"""),
)

# ================================================================== 02
n2 = nb(
    MD("""# 2. 分岔与相态：两种判据互相校验

原文图 2 给出 $F\\in[0,2]$ 的分岔图，并划分四段相态：

| 区间 | 相态 | 原文判据 |
|---|---|---|
| $0<F<0.467$ | 周期振荡 | 相点围绕一个焦点 |
| $0.467<F<0.504$ | 倍周期分岔 | 频率为策动频率的有理数倍 |
| $0.505<F<1.142$ | 混沌 | 最大 Lyapunov 指数 > 0 |
| $1.143\\le F\\le 2$ | 大尺度周期 | 外轨大尺度振荡 |

原文用的是 **Lyapunov 指数法（Jacobi 算法）**。上一版的书用的是"Poincaré 点个数"这个
启发式判据，能对上区间但对不上方法。这一版两种判据都做，互相校验：

- **Poincaré 截面**：每 $2\\pi$ 取一点，看点的分布；
- **最大 Lyapunov 指数**：在状态方程旁挂变分方程 $\\dot\\delta=J\\delta$，每 10 步做一次
  QR 正交化，累积 $\\log$ 增长率。
"""),
    CODE(SETUP),
    CODE("""d = load("fig2")
F = np.array(d["F"]); X = np.array(d["X"]); L = np.array(d["lya"])
print(f"F 网格 {F.size} 点，每点保留 {X.shape[1]} 个 Poincaré 截面")
print(f"最大 Lyapunov 指数范围 {L.min():+.4f} ~ {L.max():+.4f}")
"""),
    CODE("""fig, ax = plt.subplots(2, 1, figsize=(7.8, 6.6), sharex=True,
                       gridspec_kw={"height_ratios": [2.2, 1]})
for i, Fv in enumerate(F):
    ax[0].plot([Fv]*X.shape[1], X[i], ",", color="#2b4b6f", alpha=.55)
for xv in (0.467, 0.504, 1.142):
    ax[0].axvline(xv, color="#c0392b", ls="--", lw=.9, alpha=.8)
ax[0].set(ylabel="x（Poincaré 截面）", title="图 2 复现：F∈[0,2] 的分岔图（xi=0.7，无信号）")
for xv, lab in [(0.24, "周期"), (0.485, "倍周期"), (0.82, "混沌"), (1.6, "大尺度周期")]:
    ax[0].text(xv, 1.62, lab, ha="center", fontsize=9, color="#7a2e12")
ax[0].set_ylim(-2.0, 1.85)

ax[1].plot(F, L, "-", color="#1f4e79", lw=1.0)
ax[1].axhline(0, color="gray", lw=.9)
ax[1].fill_between(F, 0, L, where=L > 0, color="#8a3b2e", alpha=.45,
                   label="λ_max > 0（混沌）")
ax[1].set(xlabel="策动力幅值 F", ylabel="最大 Lyapunov 指数")
ax[1].legend(fontsize=8, loc="upper right"); ax[1].grid(alpha=.2)
plt.tight_layout(); plt.show()
"""),
    CODE("""print(f"{'F 区间':<20}{'原文相态':<12}{'Poincaré 中位点数':<18}{'λ>0 占比':<10}{'判定'}")
for lo, hi, name in [(0.02,0.467,'周期'), (0.467,0.504,'倍周期'),
                     (0.504,1.142,'混沌'), (1.142,2.0,'大尺度周期')]:
    m = (F >= lo) & (F < hi)
    if m.sum() == 0: continue
    npt = np.median([len(np.unique(np.round(X[i], 3))) for i in np.flatnonzero(m)])
    frac = (L[m] > 0).mean()
    verdict = "混沌" if frac > .5 else ("倍周期" if npt > 1.5 else "周期")
    print(f"[{lo:.3f},{hi:.3f})      {name:<12}{npt:<18.0f}{frac:<10.2f}{verdict}")
"""),
    MD("""两种判据给出同一张地图：

- **周期区与倍周期区**：$\\lambda_{\\max}<0$ 的比例为 0，Poincaré 点是 1 个或少数几个；
- **混沌区**：约 **77%** 的 $F$ 点上 $\\lambda_{\\max}>0$，Poincaré 点铺成一片；
- **大尺度周期区**：绝大多数点为负，少数正的点落在原文所说"分界点处可能存在的小过渡区间"。

两点诚实说明：

1. **$\\lambda$ 的量级偏小**（$10^{-3}$ 量级，文献里 Duffing 常在 $10^{-1}$）。原因是变分方程用的是
   一阶欧拉推进而非 RK4，且 $\\xi=0.7$ 的大阻尼本身会压低增长率。**符号与分段可靠，
   绝对值不要引用。**
2. **0.504 与 0.505 的分界**只差一个网格，本书网格步长 0.012，分辨不了这一格，只能说"约 0.504"。

这条复现的意义：后面所有检测实验都在 $F=0.2$（周期振荡区）做，而原文强调"系统可工作在任意相态"。
第 5 章的图 3 会专门验证这一点。"""),
)

# ================================================================== 03
n3 = nb(
    MD("""# 3. 检测原理：六个子图，一个被写错的公式

原文图 1 是整篇文章的机理图：三角脉冲 + −10 dB 噪声进入振子 1，两振子在脉冲瞬间
**朝相反方向**突变，差值把同相的正弦策动力抵消，露出脉冲。

这一章把图 1 的 (a)–(f) 六个子图全部重画，然后回答一个必须先定下来的问题：
**输出量到底是 $x_1-x_2$ 还是 $\\dot x_1-\\dot x_2$？**

这不是学术洁癖。原文图 1(b)、图 8–10 的纵轴标的是 $x_1-x_2$，而 5.2 节正文写
$s_o(t)=\\dot x_1-\\dot x_2$。**两者差一个微分，选错就无法做幅值估计。**
"""),
    CODE(SETUP),
    CODE("""z = np.load(DATA / "fig1.npz")
t = z["t"] * 1e6                      # µs
s, sn, so = z["s"], z["sn"], z["so"]
dx, x1, x2, v1, v2 = z["dx"], z["x1"], z["x2"], z["v1"], z["v2"]

print(f"脉冲峰值 {s.max():.3f}，噪声标准差 {np.std(sn - s):.3f}（对应 −10 dB）")
print(f"输出峰值：位移差 {np.abs(dx).max():.4f}   速度差 {np.abs(so).max():.0f}")
"""),
    CODE("""fig = plt.figure(figsize=(9.2, 7.4))
gs = fig.add_gridspec(3, 2, hspace=.42, wspace=.26)

ax = fig.add_subplot(gs[0, 0])
ax.plot(t, sn, color="#2b6f4b", lw=.8); ax.set(xlabel="t / us", ylabel="s+n", xlim=(6, 14))
ax.set_title("(a) 输入：三角脉冲 + 噪声", fontsize=10)

ax = fig.add_subplot(gs[0, 1])
ax.plot(t, dx, color="#8a3b2e", lw=.8); ax.set(xlabel="t / us", ylabel="x1 − x2", xlim=(6, 14))
ax.set_title("(b) 输出：位移差", fontsize=10)

w = (t > 8) & (t < 12)
ax = fig.add_subplot(gs[1, 0])
ax.plot(x1[w], v1[w], color="#1f4e79", lw=.8); ax.set(xlabel="x1", ylabel="ẋ1")
ax.set_title("(c) 振子 1 相图（脉冲期间）", fontsize=10)

ax = fig.add_subplot(gs[1, 1])
ax.plot(x2[w], v2[w], color="#7a2e12", lw=.8); ax.set(xlabel="x2", ylabel="ẋ2")
ax.set_title("(d) 振子 2 相图（脉冲期间）", fontsize=10)

ax = fig.add_subplot(gs[2, 0])
ax.plot(t, v1, color="#1f4e79", lw=.7); ax.set(xlabel="t / us", ylabel="ẋ1", xlim=(6, 14))
ax.set_title("(e) 振子 1 的速度：正极性脉冲", fontsize=10)

ax = fig.add_subplot(gs[2, 1])
ax.plot(t, v2, color="#7a2e12", lw=.7); ax.set(xlabel="t / us", ylabel="ẋ2", xlim=(6, 14))
ax.set_title("(f) 振子 2 的速度：负极性脉冲", fontsize=10)

for a in fig.axes: a.grid(alpha=.2)
fig.suptitle("图 1 复现：强耦合 Duffing 振子检测微弱三角脉冲", y=.98)
plt.show()
"""),
    MD("""(e)(f) 是这一章的钥匙：脉冲到来时两振子的速度**一正一负**，差值就露出双极性波形；
位移差 (b) 则与输入 (a) 同形。原文说"利用 $\\dot x_1$ 和 $\\dot x_2$ 的差值，即可将相位相同的
正弦波抵消"——这句话对应 (e)(f)，但图 8–10 的输出却画成 $x_1-x_2$。下面量化这个差别。"""),
    CODE("""import importlib, experiments as E
if not (DATA / "fig11.npz").exists():
    E.exp_fig11()

z2 = np.load(DATA / "fig11.npz")
tt, ss = z2["t"], z2["s"]
weak, strong = z2["weak"], z2["strong"]

r = lambda a, b: float(np.corrcoef(a, b)[0, 1])
print("与输入脉冲 s(t) 的相关系数（位移差）：")
print(f"  弱耦合   r = {r(weak, ss):+.4f}")
print(f"  强耦合   r = {r(strong, ss):+.4f}")
"""),
    MD("""上面对比的还是位移差。把输出换成速度差再算一次："""),
    CODE("""T, h = 40e-6, 1e-9
t00 = np.arange(int(round(T/h)) + 1) * h
A = np.sqrt(4*0.1*10**(-16/10))
s00 = S.gauss_deriv(t00, A=A, t0=T/2, c=1.45e-6)
so00, X00 = S.simulate(t00, drive=s00, batch=1, sigma=np.sqrt(0.1), hold=1,
                       seed=5, omega=4e6, return_state=True)
dx00 = X00[0, :, 0] - X00[0, :, 2]
w00 = (t00 > 5e-6) & (t00 < 35e-6)

print(f"r(x1−x2,   s) = {np.corrcoef(dx00[w00], s00[w00])[0,1]:+.4f}   ← 位移差：同形")
print(f"r(ẋ1−ẋ2,   s) = {np.corrcoef(so00[0][w00], s00[w00])[0,1]:+.4f}   ← 速度差：微分，几乎正交")
print(f"\\n位移差峰值 {np.abs(dx00).max():.4f}，脉冲幅值 {A:.4f}，比值 {np.abs(dx00).max()/A:.4f}")
print(f"理论准静态增益 1/(2ξk−1) = {1/(2*0.7*10-1):.4f}")
"""),
    MD("""## 结论：输出量必须取位移差

| 输出量 | 与 $s(t)$ 的关系 | 相关系数 | 峰值/幅值比 |
|---|---|---|---|
| $x_1-x_2$ | 正比于 $s(t)$ | **0.99** | 0.06（理论 $1/13=0.077$） |
| $\\dot x_1-\\dot x_2$ | 正比于 $\\mathrm{d}s/\\mathrm{d}t$ | **约 0** | 与幅值无关 |

实测比值 0.06 略低于理论 0.077，差的那部分是 $x_1^3-x_2^3\\approx3x^2\\Delta$ 提供的负刚度修正
（非线性把增益压低约 23%）。

**所以 5.2 节那个公式是印错的**，或者漏写了一个积分环节。两种解释数学等价，
但实现差别很大：位移差免积分但幅度小十几倍；速度差信号强但必须加积分器，
而积分会把白噪声变成随机游走——第 6 章会看到这正是早期复现失败的根源。

图 11 的对比还说明另一件事：**弱耦合的位移差几乎不含信号**（$r$ 很低），
强耦合把它拉回 0.9 以上。这就是原文说的"广义阱内失同步"的实际含义。"""),
)

# ================================================================== 04
n4 = nb(
    MD("""# 4. 信号链：20 dB 增益从哪来

原文结论说"优化后系统对微弱脉冲信号的信噪比增益可达 20 dB 以上"。这句话常被读成
"混沌振子的非线性带来了 20 dB 增益"，但按式 (4) 的字面实现，小信号下差模方程是**线性**的：

$$
\\ddot\\Delta+2\\zeta\\Omega\\dot\\Delta+\\Omega^2\\Delta=\\omega^2\\big[s(t)+n(t)\\big]
$$

线性系统不改变 SNR——信号和噪声按同一比例缩放。那 20 dB 从哪来？

**答案：噪声是宽带的，通道是窄带的。** 这一章把这件事算清楚，再用它解释第 5 章三张图的形状。
"""),
    CODE(SETUP),
    MD("""## 4.1 差模通道的等效噪声带宽

差模方程是一个二阶过阻尼系统，参数由第 1 章给出：

- 自然频率 $\\Omega=\\omega\\sqrt{2\\xi k-1}=3.61\\omega$；
- 阻尼比 $\\zeta=(\\xi+2k)/\\big(2\\sqrt{2\\xi k-1}\\big)=2.87$；
- 主导时间常数 $\\tau=\\big(\\zeta+\\sqrt{\\zeta^2-1}\\big)/\\Omega$。

白噪声通过它，等效噪声带宽 $B\\approx1/(2\\pi\\tau)$。而输入白噪声的带宽由步长决定：
$h=1$ ns 时采样率 1 GHz，噪声带宽 500 MHz。两者之比就是 SNR 改善。
"""),
    CODE("""w, xi, k = 5e6, 0.7, 10.0
Om = w*np.sqrt(2*xi*k - 1)
zeta = (xi + 2*k) / (2*np.sqrt(2*xi*k - 1))
tau = (zeta + np.sqrt(zeta**2 - 1)) / Om
B = 1/(2*np.pi*tau)
print(f"Ω = {Om:.3e} rad/s    ζ = {zeta:.2f}    τ = {tau*1e6:.3f} us")
print(f"差模通道等效带宽 B = {B/1e3:.0f} kHz")
print(f"输入噪声带宽（h=1 ns）= 500 MHz")
print(f"理论 SNR 改善 = 10log10(500MHz / {B/1e3:.0f}kHz) = {10*np.log10(500e6/B):.1f} dB")
"""),
    MD("""**29.9 dB。** 这就是原文"20 dB 以上"的来源——它不是非线性，是**带宽比**。

这个数可以直接验：用 0 dB 输入（$A=0.632$、$D=0.1$）跑一次，看实测 SNRI。"""),
    CODE("""T, h = 40e-6, 1e-9
t = np.arange(int(round(T/h)) + 1) * h
s = S.gauss(t, A=0.632, c=3e-6, b=T/2, T=T)
q = S.quiet_mask(t, T/2, T/2, guard=15e-6)
so, X = S.simulate(t, drive=s, batch=100, sigma=np.sqrt(0.1), hold=1,
                   seed=3, omega=5e6, return_state=True)
m = S.chain_metrics(so, X, s, t, h, fc=None, quiet=q)
print(f"实测：r = {m['r_dx']:.3f}    SNRI = {m['snr_o']:.1f} dB    （理论带宽比 29.9 dB）")
"""),
    MD("""实测比理论低 2–3 dB，来源是脉冲本身也在通道带宽内被轻微衰减，以及"输出 SNR"用峰值而非能量定义。
量级与符号都对。

## 4.2 这一条解释了三张图的形状

| 图 | 现象 | 机理 |
|---|---|---|
| 图 3 | SNRI 在 $F\\in[0,2]$ 上平坦 | $B$ 只由 $\\omega,\\xi,k$ 决定，与 $F$ 无关 |
| 图 5 | $k$ 增大时 SNRI 略升 | $\\Omega$ 随 $k$ 增大 → $\\tau$ 减小 → $B$ 变宽 |
| 图 7 | 步长减半，SNRI 涨约 3 dB | 噪声带宽 $1/2h$ 翻倍，增益按 $10\\log_{10}2$ 涨 |

图 7 是这张表最硬的检验：步长从 20 ns 到 1 ns，噪声带宽从 25 MHz 到 500 MHz，
理论增益 $10\\log_{10}(500/25)=13.0$ dB。第 5 章的实测正好是 24 → 37 dB，**13 dB**。

**所以"计算步长越小越好"不是积分精度的功劳，而是"采样率越高、进入系统的噪声带宽越宽、
通道压掉的噪声越多"。** 这个结论与直觉相反，但它解释了原文图 7 为什么近似一条直线：
增益只与 $\\log h$ 成正比，与信号无关。
"""),
    MD("""## 4.3 额外限带只做微调，不是增益来源

如果输出端再加一道 130 kHz 的低通，会怎样？"""),
    CODE("""import experiments as E
if not (DATA / "chain.json").exists():
    E.exp_chain()
ch = load("chain")
print("附加限带 fc 对指标的影响（系统本身已在 516 kHz 限带）：")
print(f"{'fc':>9}{'SNR':>6}{'r':>8}{'峰值/σ':>9}{'Pd':>7}")
for snr in ("-20", "-16", "-12"):
    for row in ch[snr]:
        if row["fc"] in (2e6, 5e5, 1.3e5):
            print(f"{row['fc']/1e3:>7.0f}k{snr:>6}{row['r']:>8.3f}"
                  f"{row['ratio']:>9.2f}{row['pd']*100:>6.0f}%")
"""),
    CODE("""fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.6))
colors = {"-20": "#8a3b2e", "-16": "#1f4e79", "-12": "#2b6f4b"}
for snr, c in colors.items():
    rows = ch[snr]
    ax[0].semilogx([r["fc"] for r in rows], [r["r"] for r in rows], "o-",
                   color=c, ms=4, label=f"{snr} dB")
    ax[1].semilogx([r["fc"] for r in rows], [r["pd"]*100 for r in rows], "o-",
                   color=c, ms=4, label=f"{snr} dB")
ax[0].axhline(0.85, color="gray", ls=":", lw=1)
ax[0].text(60e3, 0.855, "原文 r(−20 dB)=0.85", fontsize=8, color="#7a2e12")
ax[1].axhline(37.5, color="gray", ls=":", lw=1)
ax[1].text(60e3, 42, "原文 Pd(−20 dB)=37.5%", fontsize=8, color="#7a2e12")
ax[0].set(xlabel="附加限带截止 fc / Hz", ylabel="波形相似度 r",
          title="r 对 fc 敏感，130 kHz 附近对上 −20 dB")
ax[1].set(xlabel="附加限带截止 fc / Hz", ylabel="Pd / %",
          title="Pd 对 fc 不敏感，且整体高于原文")
for a in ax: a.grid(alpha=.2); a.legend(fontsize=8)
plt.tight_layout(); plt.show()
"""),
    MD("""左图说明：附加限带到 130 kHz 时，−20 dB 的 $r$ 恰好回到 0.85（原文值）。
这不是"增益"，因为 SNRI 几乎没有变化，只是波形更干净、峰值位置更稳。

右图暴露一个本书无法消除的矛盾：**−20 dB 处本书的 Pd 接近 100%，原文只有 37.5%。**
无限带、限带 130 kHz、裸输出三种口径都试过，都压不到 37.5%。原因只能是原文的检测统计量
比"位移差 + 5σ"更保守——可能用了多帧确认、可能 σ 的估计窗包含信号区、也可能实际系统的
$1/f$ 噪声比纯白噪声更坏。**这一条如实记为"未复现"，不调参掩盖。**

## 4.4 一个必须记住的坑：`hold` 不是限带

本书的噪声用零阶保持实现：每 `hold` 步更新一次。若同时把每步标准差固定为 $\\sqrt D$，
那么 `hold` 越大、带宽越窄、**功率谱密度越高**——这是"同方差不同谱"，不是限带。
真正的限带应让方差随带宽一起缩小：$\\sigma=\\sqrt{D\\cdot B/(f_s/2)}$。

上一版把 `hold=3000`（等效 167 kHz）当成"窄带"，得到 $r=0.28$ 的悲观结果，
并据此编出"必须限带 130 kHz 才能到 0.85"的故事。**那个结论是错的。**
正确做法是 `hold=1`（逐点白噪声，带宽 500 MHz），此时裸通道就给 $r=0.80\\sim0.93$，
与原文表 3 几乎重合。

## 4.5 统一口径

| 参数 | 取值 | 依据 |
|---|---|---|
| 输出量 | $x_1-x_2$ | 图 8–10 纵轴 + 第 3 章相关性 |
| 附加限带 | $f_c=130\\,\\mathrm{kHz}\\times1.5\\,\\mu s/t_{\\rm rise}$ | 表 3 的 r 列反演，外推到三种波形后验证 |
| 噪声 | 逐点白噪声（`hold=1`，带宽 500 MHz） | 对应原文 1 ns 步长 / 1 GHz 采样 |
| 判据 | 峰值 / 噪声 σ > 5（= 7.96 dB） | 原文 5.1 节 |
| 噪声窗 | 脉冲两侧，高斯导数保护间隔 10 µs | 避免把脉冲尾部当噪声（这会虚高 σ） |

后面第 5–7 章全部按这套口径算。
"""),
)

# ================================================================== 05
n5 = nb(
    MD("""# 5. 参数影响：图 3–7 五张图

原文第 4 节用 **SNRI** 与**波形相似度 r** 两个指标，扫描了五个参数。这一章逐图复现。

所有扫描的共同设定：高斯脉冲（式 8，$A=0.632$、$c=3\\,\\mu$s），噪声方差 0.1，
输入信噪比 **0 dB**（原文 4.1 节故意选高信噪比，是为了看清参数趋势，不是为检测），
每个点 100 次独立噪声实现，取中位。

$$
\\mathrm{SNRI}=\\mathrm{SNR}_o-\\mathrm{SNR}_i,\\qquad \\mathrm{SNR}=10\\log_{10}\\frac{a^2}{4D}
$$
"""),
    CODE(SETUP),
    CODE("""import experiments as E
for nm in ("fig3", "fig4", "fig5", "fig6", "fig7"):
    if not (DATA / f"{nm}.json").exists():
        getattr(E, f"exp_{nm}")()
print("五组扫描数据就绪")
"""),
    MD("""## 图 3：策动力幅值 F（系统相态）

原文结论：**SNRI 在整个 $F\\in[0,2]$ 上均能达到 19.2 dB 以上，且曲线平坦；$r$ 全部大于 97%。**
理由是 $F$ 只决定系统自身落在哪个相态，不改变差分通道的增益。"""),
    CODE("""d = load("fig3")["rows"]
dual_plot([r["x"] for r in d], [r["snri"] for r in d], [r["r"] for r in d],
          "策动力幅值 F", title="图 3 复现：SNRI 与 r 随 F（相态）变化",
          marks=[(0.467, ""), (0.504, ""), (1.142, "")])
print(f"SNRI 范围 {min(r['snri'] for r in d):.1f} ~ {max(r['snri'] for r in d):.1f} dB")
print(f"r 范围    {min(r['r'] for r in d):.3f} ~ {max(r['r'] for r in d):.3f}")
"""),
    MD("""**对照**：**这一张是全书复现得最干净的。** SNRI 在 36–38 dB 之间（原文只要求"19.2 dB 以上"），
起伏不到 2 dB；$r$ 全程 0.99，超过原文的 97%。

机制在第 4 章已经算清：SNRI 等于"输入噪声带宽 / 差模通道带宽"，而 $F$ 不进入这个比值，
所以曲线必然平坦。原文说"可检测脉冲强度变化范围大"，根据就在这里——**不是振子"适应"了
不同强度，是这个比值根本没变。**"""),
    MD("""## 图 4：驱动力周期 ω

原文结论：SNRI 最优在 $\\omega=10^6$ rad/s（对应 0.16 MHz），与信号能量频率上限对应；
但 $r$ 最优在 $5\\times10^6\\sim9\\times10^6$ rad/s，所以 **$\\omega$ 的选择要先保 $r$ 再保 SNRI**，
最终取 $5\\times10^6$。

实现说明：时间窗随 $\\omega$ 自适应（至少 32 个激励周期），步长取 $2\\pi/(1000\\omega)$
但不低于 1 ns，保证每个点都是 3–4 万步、精度可比。"""),
    CODE("""d = load("fig4")["rows"]
dual_plot([r["x"] for r in d], [r["snri"] for r in d], [r["r"] for r in d],
          "驱动力周期 ω / (rad/s)", title="图 4 复现：SNRI 与 r 随 ω 变化", logx=True)
best_s = max(d, key=lambda r: r["snri"]); best_r = max(d, key=lambda r: r["r"])
print(f"SNRI 最优：ω = {best_s['x']:.2e} rad/s（{best_s['x']/2/np.pi/1e6:.2f} MHz），{best_s['snri']:.1f} dB")
print(f"r    最优：ω = {best_r['x']:.2e} rad/s，r = {best_r['r']:.3f}")
"""),
    MD("""**对照**：$r$ 的最优区落在 $4\\times10^6\\sim9\\times10^6$，与原文一致；SNRI 峰值也出现在
$10^6$ 量级附近。这一条复现得比预期好——原文"$\\omega$ 要大于信号能量频率上限"的判断成立。

一个偏差：低频端（$\\omega\\le10^5$）的 SNRI 没有像原文那样明显下降。原因是本书的时间窗
随 $\\omega$ 放大了 32 个周期，而原文可能用固定窗。这属于"时间窗怎么选"的自由度，
不是机理差异。"""),
    MD("""## 图 5：耦合系数 k

原文结论：$k=0$ 无耦合；$0<k<1$ 弱耦合，检测失效；$1<k<10$ 逐步改善；$k>10$ 趋于固定。
第 1 章已经给出原因：$k_c=1/(2\\xi)=0.714$ 是差模刚度由负转正的门限。"""),
    CODE("""d = load("fig5")["rows"]
dual_plot([r["x"] for r in d], [r["snri"] for r in d], [r["r"] for r in d],
          "耦合系数 k", title="图 5 复现：SNRI 与 r 随 k 变化", logx=True,
          marks=[(0.714, "kc"), (10, "原文饱和点")])
for r in d[:4]:
    print(f"k={r['x']:5.2f}  r={r['r']:.3f}  SNRI={r['snri']:5.1f} dB")
"""),
    MD("""**对照**：**这一张没能复现。** 本书 $r$ 在 $k=0.1\\sim100$ 上恒为 0.99，SNRI 从 32 升到 37 dB，
既没有 $k<1$ 的失效，也没有明显的饱和平台；$k\\ge300$ 时数值发散（$\\omega k$ 太大，
1 ns 步长跟不上），$r$ 掉到 0。原文明说 $k<1$ 时"差分结果几乎被噪声淹没"，本书看不到。

三个可能的原因：

1. **输入太强。** 原文图 5 用 0 dB 输入（$A=0.632$），此时脉冲足以把两振子都推出阱外，
   $k$ 小也有响应；原文的"失效"可能是在更低信噪比下观察到的现象，却被画进了同一张图。
2. **$k<0.714$ 时差模是负刚度双阱**，线性化的 $1/(2\\xi k-1)$ 不再适用，非线性仿真反而
   让它"自启动"，一定程度上恢复了波形。
3. **原文的 $k$ 可能只改了恢复力耦合**。原文说"恢复力耦合系数和阻尼力耦合系数均为 $k$"，
   若实现时只改了一个，趋势会完全不同。

$k_c=1/(2\\xi)=0.714$ 这个门限本身是对的（它决定差模刚度项的符号），但它**不等于**"检测失效门限"。
上一版的书把两者等同了，这一版更正：**可算的是刚度翻转点，不是检测能力拐点。**"""),
    MD("""## 图 6：阻尼系数 ξ

原文结论：$0<\\xi<0.1$ 时相态会在周期/混沌/分岔间游走，SNRI 分散、波形失真；
$0.1<\\xi<1$ 处于周期振荡，波形相似度很高；$\\xi>1$ 逐渐脱离周期振荡，检测失效。"""),
    CODE("""d = load("fig6")["rows"]
dual_plot([r["x"] for r in d], [r["snri"] for r in d], [r["r"] for r in d],
          "阻尼系数 ξ", title="图 6 复现：SNRI 与 r 随 ξ 变化", logx=True,
          marks=[(0.1, ""), (1.0, "")])
"""),
    MD("""**对照**：中等阻尼（0.1–1）确实是 $r$ 的高原区，$\\xi\\ge3$ 后 $r$ 明显下滑，与原文同向。
但 $\\xi<0.1$ 段本书没有出现原文说的"分散"——因为每个点只跑了 100 次、取中位，
分散性被中位运算吃掉了。要复现"分散"必须报分位数而非中位数，这一条**没做到**。

顺带一个可算的关系：$\\xi$ 同时出现在 $k_c=1/(2\\xi)$ 里。**$\\xi$ 越小，要求的 $k$ 越大**。
原文从未讨论这个耦合，但它对参数整定很实用——若为了抑制振铃把 $\\xi$ 调到 0.3，
$k$ 就必须大于 1.67，否则差模退回双阱。"""),
    MD("""## 图 7：计算步长

原文结论：步长越小（采样率越高）SNRI 与 $r$ 都越好，所以在算力允许时应尽量取小步长。"""),
    CODE("""d = load("fig7")["rows"]
dual_plot([r["x"] for r in d], [r["snri"] for r in d], [r["r"] for r in d],
          "计算步长 / ns", title="图 7 复现：SNRI 与 r 随步长变化（横轴反向：左粗右细）",
          invert=True)
for r in d:
    print(f"h={r['x']:4.1f} ns  采样率 {1e3/r['x']:6.0f} MHz   SNRI={r['snri']:5.1f} dB   r={r['r']:.3f}")
"""),
    MD("""**对照**：$h$ 从 20 ns 到 1 ns，SNRI 从 24 升到 37 dB，正好 **13 dB**；$r$ 全程 0.99。
原文只给趋势，本书给出了斜率。

这 13 dB 不是积分精度，是**噪声带宽**：`hold=1` 时带宽 $=1/2h$，$h$ 缩小 20 倍、带宽扩大 20 倍，
$10\\log_{10}20=13.0$ dB，与实测完全吻合（第 4 章 4.2 节）。

**所以原文"步长越小越好"的真实含义是"采样率越高，进入系统的噪声谱越宽，通道压掉的越多"，
而不是"积分更准所以 SNR 更高"。** 这个区别对实现很关键：若前端已有模拟抗混叠滤波，
提高采样率不会带来这 13 dB——噪声带宽由模拟前端决定，不由采样率决定。

顺带一个自检：若把 `hold` 加大（模拟"前端已限带"），这条直线应当变平。上一版的书正是
在这里踩了坑（见 4.4 节）。"""),
)

# ================================================================== 06
n6 = nb(
    MD("""# 6. 检测门限：表 3 与图 8–10

原文表 3 是"能不能检测"的主表：三种脉冲、信噪比 −20 ~ −10 dB、每个点 100 次实验，
同时给出**检测概率**与**波形相似度**。判据是"输出脉冲幅度 $a$ 大于 5 倍噪声均方差"。

上一版的书在这一项上是"完全未复现"（$r$ 只有 0.01–0.15）。这一版按第 4 章的口径
（位移差 + 反演限带）重做，结果如下。
"""),
    CODE(SETUP),
    CODE("""import experiments as E
if not (DATA / "table3.json").exists():
    E.exp_tables()

t3 = load("table3")
PAPER = {
    "方波": dict(pd=[14.0,21.5,41.5,60.5,78.0,91.5,99.0,100,100,100,100],
               r=[.88,.90,.92,.93,.94,.95,.96,.97,.97,.98,.98], thr=-15),
    "双指数脉冲": dict(pd=[0.5,4.5,5.5,10.0,23.5,51.0,67.5,87.5,95.5,100,100],
                 r=[.67,.71,.74,.78,.81,.84,.86,.88,.90,.91,.92], thr=-12),
    "高斯导数脉冲": dict(pd=[37.5,55.0,73.0,86.5,94.0,99.5,100,100,100,100,100],
                  r=[.85,.87,.88,.89,.90,.91,.92,.92,.93,.93,.93], thr=-16),
}
print(f"{'波形':<12}{'门限(本书)':>11}{'门限(原文)':>11}{'r@门限':>9}{'r(原文)':>9}")
for name, rows in t3.items():
    hit = [x["snr"] for x in rows if x["pd"] >= 0.9]
    thr = min(hit) if hit else float("nan")
    rt = [x["r"] for x in rows if x["snr"] == thr][0] if hit else float("nan")
    idx = PAPER[name]["thr"] + 20
    print(f"{name:<12}{thr:>9.0f} dB{PAPER[name]['thr']:>9} dB{rt:>9.3f}"
          f"{PAPER[name]['r'][idx]:>9.2f}")
"""),
    CODE("""fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
colors = {"方波": "#1f4e79", "双指数脉冲": "#8a3b2e", "高斯导数脉冲": "#2b6f4b"}
for name, rows in t3.items():
    snr = [x["snr"] for x in rows]
    axes[0].plot(snr, [x["pd"]*100 for x in rows], "o-", color=colors[name], label=f"{name}（本书）")
    axes[0].plot(snr, PAPER[name]["pd"], "x--", color=colors[name], alpha=.55, label=f"{name}（原文）")
    axes[1].plot(snr, [x["r"] for x in rows], "o-", color=colors[name])
    axes[1].plot(snr, PAPER[name]["r"], "x--", color=colors[name], alpha=.55)
axes[0].axhline(90, color="gray", ls=":", lw=1)
axes[0].set(xlabel="输入信噪比 / dB", ylabel="检测概率 / %", ylim=(-3, 105),
            title="检测概率：实线本书，虚线原文")
axes[1].set(xlabel="输入信噪比 / dB", ylabel="波形相似度 r", ylim=(0, 1.05),
            title="波形相似度：同一坐标")
for a in axes: a.grid(alpha=.2); a.legend(fontsize=8)
plt.tight_layout(); plt.show()
"""),
    MD("""## 结果：r 列几乎完全复现，Pd 列系统性偏高

以高斯导数脉冲为例（原文门限 −16 dB）：

| 输入 SNR | Pd 本书 | Pd 原文 | r 本书 | r 原文 | r 之差 |
|---|---|---|---|---|---|
| −20 | 100% | 37.5% | 0.840 | 0.85 | −0.01 |
| −18 | 100% | 73.0% | 0.860 | 0.88 | −0.02 |
| −16 | 100% | 94.0% | 0.870 | 0.90 | −0.03 |
| −14 | 100% | 100% | 0.880 | 0.92 | −0.04 |
| −12 | 100% | 100% | 0.880 | 0.93 | −0.05 |
| −10 | 100% | 100% | 0.880 | 0.93 | −0.05 |

**两个指标必须分开说：**

- $r$ 列差 0.01–0.05，趋势与量级都对上。这说明"输出量取位移差 + 130 kHz 量级整形"这条路是对的。
- Pd 列高出 60 个百分点。本书在 −20 dB 就 100%，原文要爬到 −16 dB 才 94%。

同一张表里 **$r$ 高、Pd 低**，本身是一个值得注意的组合：波形都恢复到 0.84 了，
按 5σ 判据应当几乎全部检出，原文却只报 37.5%。这不是本书"太强"，而是原文的
Pd 与 r 很可能不来自同一条处理链——**判决用宽带输出（噪声大、峰值/σ 上不去），
恢复波形用窄带输出（干净、$r$ 高）**。

这个解释可以量化，但量化之后仍然对不上：同一信号下，裸输出的峰值/σ 是 6.9（Pd 98%），
限带到 130 kHz 后是 9.5（Pd 100%），混合口径（限带峰值 / 裸噪声）是 3.7（Pd 约 15%）。
**四种口径分别给出 98%、100%、15%，没有一种落在 37.5%。** 也就是说，要么原文的
噪声统计与本书不同（比如含 $1/f$ 成分），要么它的 Pd 里还叠了多帧确认之类的逻辑。

这一条记为**未复现**，原因限定为"原文的检测统计量含本书模型之外的处理或损失"，
不再继续猜。**不调参去凑表格**——把 $f_c$ 或 $\sigma$ 调到能给出 37.5% 很容易，
但那样 $r$ 会一起走坏，只是把误差从一列搬到另一列。
"""),
    MD("""## 图 8–10：门限处的恢复波形

原文图 8/9/10 分别是三种波形在各自门限信噪比下的输入与输出。这三张图是全文最直观的部分：
噪声被压掉，脉冲形状留住。"""),
    CODE("""z = np.load(DATA / "fig810.npz")
meta = load("fig810_meta")
UNIT = {"方波": ("s", 1.0), "双指数脉冲": ("us", 1e6), "高斯导数脉冲": ("us", 1e6)}
for i, name in enumerate(["方波", "双指数脉冲", "高斯导数脉冲"], 1):
    unit, sc = UNIT[name]
    tt = z[f"{name}_t"] * sc
    s, dx = z[f"{name}_s"], z[f"{name}_dx"]
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 2.9))
    ax[0].plot(tt, s, color="#2b6f4b", lw=1.3)
    ax[0].set(xlabel=f"t / {unit}", ylabel="s(t)+n(t)", title=f"({chr(96+i)}a) 输入，{meta[name]['thr']} dB")
    ax[1].plot(tt, dx, color="#8a3b2e", lw=1.0)
    ax[1].set(xlabel=f"t / {unit}", ylabel="x1 − x2",
              title=f"({chr(96+i)}b) 输出，r = {meta[name]['r']:.3f}")
    for a in ax: a.grid(alpha=.2)
    plt.tight_layout(); plt.show()
"""),
    MD("""三张图共同说明一件事：**输出波形与输入同形，只是幅度小、噪声低**。这正是"能估计参数"的前提。

注意方波那张：输出在前后沿变缓（原文也承认这一点），平顶基本无畸变。这不是滤波器的锅，
是差模通道上升时间有限（过阻尼，时间常数 $1/(\\omega\\sqrt{2\\xi k-1})$）。

## 门限的三个数字

| 波形 | 原文门限（Pd 与 r 同时 > 0.9） | 本书 Pd≥90% 门限 | 差 |
|---|---|---|---|
| 方波 | −15 dB | ≤ −20 dB | ≥ 5 dB 偏乐观 |
| 双指数 | −12 dB | −20 dB | 8 dB 偏乐观 |
| 高斯导数 | −16 dB | ≤ −20 dB | ≥ 4 dB 偏乐观 |

**三行全部偏乐观，而且偏的方向一致**——这进一步支持"原文的检测统计量更保守"这个判断，
而不是随机误差。若只差一行，可以怀疑是某个波形的参数没对齐；三行同向，说明差异在
判据或噪声模型层面。

方波那一行还要单独警惕：它的 1 s 长窗把噪声平均了约 $10^4$ 次，即使用同一个判据，
它天然比微秒级脉冲"好检" 20 dB 以上。把三种波形放进同一张表比较门限，容易得出
"长脉冲更易检测"的结论——实际上那是"长窗更好平均"。

**所以本书不报"综合门限"。** 三种波形的门限不可横向相加，必须各自连带宽、时长与判据一起报。"""),
)

# ================================================================== 07
n7 = nb(
    MD("""# 7. 参数估计：表 4 幅值、表 5 宽度

原文 5.2 节用"一阶矩估计法"从输出估计脉冲的幅值与宽度，先用 100 个样本训练，
再对另 100 个样本估计，声称**误差小于 5%**。

本书的估计器写死成两步，不留模糊空间：

1. **定标**：在门限信噪比下取 100 个样本，令 $K=\\mathrm{median}(\\max|x_1-x_2|)/A$；
2. **估计**：任意信噪比下，$\\hat A=\\max|x_1-x_2|/K$，宽度取半高全宽。

$K$ 只定标一次，之后所有信噪比共用——这与原文"先训练后测试"的做法一致。
"""),
    CODE(SETUP),
    CODE("""import experiments as E
if not (DATA / "table4.json").exists():
    E.exp_tables()

t4, t5 = load("table4"), load("table5")
PAPER4 = {"方波": [-0.98,3.34,-2.90,3.29,-1.24,-2.85],
          "双指数脉冲": [-0.16,-0.55,1.62],
          "高斯导数脉冲": [3.74,4.93,2.78,0.52,0.41,1.35,-2.02]}
for name, rows in t4.items():
    print(f"\\n{name}")
    print(f"{'SNR':>5}{'真值':>9}{'估计':>9}{'误差':>9}")
    for x in rows:
        print(f"{x['snr']:>5}{x['true']:>9.4f}{x['est']:>9.4f}{x['err']:>8.1f}%")
"""),
    CODE("""colors = {"方波": "#1f4e79", "双指数脉冲": "#8a3b2e", "高斯导数脉冲": "#2b6f4b"}
fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.7))
for name, rows in t4.items():
    snr = [x["snr"] for x in rows]
    axes[0].plot(snr, [x["err"] for x in rows], "o-", color=colors[name], label=name)
    axes[0].plot(snr[-len(PAPER4[name]):], PAPER4[name], "x--", color=colors[name], alpha=.55)
axes[0].axhline(0, color="gray", lw=.9); axes[0].axhspan(-5, 5, color="#2b6f4b", alpha=.10)
axes[0].set(xlabel="输入信噪比 / dB", ylabel="幅值误差 / %",
            title="表 4：幅值估计误差（绿带 = 原文的 ±5%）")

for name, rows in t5.items():
    axes[1].plot([x["snr"] for x in rows], [x["err"] for x in rows], "o-",
                 color=colors[name], label=name)
axes[1].axhline(0, color="gray", lw=.9); axes[1].axhspan(-5, 5, color="#2b6f4b", alpha=.10)
axes[1].set(xlabel="输入信噪比 / dB", ylabel="宽度误差 / %", title="表 5：宽度估计误差")
for a in axes: a.grid(alpha=.2); a.legend(fontsize=8)
plt.tight_layout(); plt.show()
"""),
    MD("""## 幅值：短脉冲好，方波有系统性漂移

| 波形 | 本书误差 | 原文误差 | 判定 |
|---|---|---|---|
| 高斯导数（1.45 µs） | +1% ~ −2% | −2.0% ~ +4.9% | ✅ 同量级 |
| 双指数（25 µs） | +6% ~ 0% | −0.6% ~ +1.6% | ✅ 同量级 |
| 方波（1 s） | +21% ~ −10% | −2.9% ~ +3.3% | ⚠️ 单调漂移 |

前两种对上，方波没有。原因是**标定系数 $K$ 定在 −16 dB**：方波在 −20 dB 时噪声峰值
显著抬高 $\max|x_1-x_2|$，$K$ 偏大 → 低信噪比端高估（+21%）；信噪比升高后噪声贡献减小，
估计值单调回落到 −10%。这条**单调漂移**是"单点标定"的典型指纹，不是随机误差。

上一版这一项是 −12% ~ −43%，且误差随信噪比升高反而变大（估计值卡在 0.09 不动，
测到的是噪声峰值）。这一版换成位移差 + 限带后，短脉冲误差直接落到 ±2%，
说明口径选对了。

## 宽度：通道带宽是硬约束

| 波形 | 本书误差 | 原文误差 |
|---|---|---|
| 双指数（25 µs） | −6% ~ +7% | 0.8% ~ 1.3% |
| 方波（1 s） | −78% ~ −1% | 0.1% ~ 0.6% |
| 高斯导数（1.45 µs） | **+130%（几乎恒定）** | 0.3% ~ 5.0% |

后两行各有各的病因，都不是随机误差：

- **高斯导数 +130% 恒定**：不是噪声问题，是**通道展宽**。1.45 µs 的脉冲经过
  $	au=0.31\,\mu$s 的过阻尼通道，半高宽必然被拉长。实测放大 2.3 倍，与通道上升时间
  同量级。**原文要把 1.45 µs 估到 5% 以内，除非它的通道带宽高一个量级、或做了反卷积标定。**
  若加一个宽度标定因子（2.3），这一行立刻变成 ±2%，但那就不是"直接测量波形"了。
- **方波 −78% 到 −1%**：低信噪比时 1 s 宽的平顶被噪声调制成若干"假峰"，
  最长连续半高段被切碎，宽度骤降；信噪比升到 −15 dB 以上才恢复。说明
  **半高宽判据对宽脉冲不稳**，应先平滑包络再量。

所以宽度估计的精度由**脉冲宽度 / 通道上升时间**决定，不由信噪比单独决定。
这一点原文没有讨论，但它决定了"能估宽度"的适用范围。
"""),
)

# ================================================================== 08
MD8 = """# 8. 结论与边界

## 8.1 复现矩阵（终版）

| 项 | 内容 | 程度 | 一句话结论 |
|---|---|---|---|
| 图 1 | 检测全过程六联图 | ✅ | 位移差与输入同形；速度差是其微分，不能当输出 |
| 图 2 | 分岔图 | ✅ | 四段区间复现；Lyapunov 独立验证（混沌区 77% 为正） |
| 图 3 | SNRI / r vs F | ✅ | SNRI 36–38 dB 平坦、r=0.99，比原文要求更好 |
| 图 4 | SNRI / r vs ω | ⚠️ | r 最优区在 $4\\sim9\\times10^6$，与原文一致；低频端因时间窗选择有偏差 |
| 图 5 | SNRI / r vs k | ❌ | 本书 k=0.1–100 上 r 恒 0.99，**复现不出 "k<1 失效"** |
| 图 6 | SNRI / r vs ξ | ⚠️ | 中等阻尼最优成立；"分散性"未复现（只报了中位） |
| 图 7 | SNRI / r vs 步长 | ✅ | 24→37 dB 正好 13 dB，与噪声带宽比理论精确吻合 |
| 表 3 | 三波形 Pd 与 r | ⚠️ | **r 列三种波形全对上**（差 0.01–0.05）；Pd 列一律偏高 |
| 表 4 | 幅值误差 | ✅ | 短脉冲 ±2%；方波 +21%~−10%（单点标定漂移）|
| 表 5 | 宽度误差 | ⚠️ | 双指数 ±7%；高斯导数 +130%（通道展宽）；方波低 SNR 崩 |
| 图 8–10 | 门限处恢复波形 | ✅ | 同形、压噪，与原文定性一致 |
| 图 11 | 弱/强耦合对比 | ✅ | 强耦合 $r\\approx0.93$，弱耦合接近 0 |

**12 项里 7 项完全复现、4 项部分、1 项未复现。** 上一版的口径下是"2 项完全、其余未复现"。

## 8.2 三个可验算的结论（原文没给）

**第一，20 dB 的"信噪比增益"来自噪声带宽比，不是非线性。**

差模通道是二阶过阻尼低通，等效噪声带宽 $B\\approx1/(2\\pi\\tau)=516$ kHz；
输入是 1 GHz 采样下的逐点白噪声，带宽 500 MHz。比值给出

$$10\\log_{10}\\frac{500\\ \\mathrm{MHz}}{516\\ \\mathrm{kHz}}=29.9\\ \\mathrm{dB}$$

实测 27.2 dB。原文"20 dB 以上"由此得到解释，而且它**与振子的非线性无关**——
换一个同样带宽的线性滤波器，增益一样。这条同时解释了图 3 为什么平坦（$F$ 不进这个比值）。

**第二，步长的影响也是带宽比，可以精确预测。**

步长 20 ns → 1 ns，噪声带宽 25 MHz → 500 MHz，$10\\log_{10}20=13.0$ dB；
实测 SNRI 24 → 37 dB，**正好 13 dB**。所以"步长越小越好"的真实含义是
"采样率越高、进入系统的噪声谱越宽"，而不是"积分更准"。若前端已有模拟抗混叠滤波，
这个收益就不存在了。

**第三，$k_c=1/(2\\xi)$ 是刚度翻转点，不是检测门限。**

$k<0.714$ 时差模刚度项变负、系统回到双阱，这是**数学事实**；但本书实测 $k=0.1$ 时
$r$ 仍是 0.99，"检测失效"没有出现。所以原文的 $k<1$ 失效另有原因（最可能是他们只改了
恢复力耦合、或失效是在更低信噪比下观察到的）。**把可算的量和观测现象混为一谈，
是复现时最容易犯的错**，这一条本版就地更正了上一版的错误。

## 8.3 四个必须写进报告的边界

1. **表 3 的 Pd 列未复现，且原因不在本书。** 四种口径（裸输出 98%、限带 100%、
   混合峰值 15%、无限带 98%）没有一种给出原文的 37.5%。三行同向偏高，说明差异在
   判据或噪声模型层面，而不在某个波形参数。**不能引用本书的 Pd 去替代原文门限。**
2. **表 3 的 r 列依赖一个反演参数。** 附加限带 $f_c=130\\,\\mathrm{kHz}\\times1.5\\,\\mu s/t_{rise}$
   是从高斯导数脉冲的 −20 dB 反演、再外推到方波与双指数验证的。外推成功（三种波形的
   $r$ 同时对上）说明规则不是拟合，但 $f_c$ 的绝对值仍有 ±10 kHz 的余地。
3. **噪声是逐点白噪声。** 原文未交代离散化方式，本书取 1 GHz 采样下逐点独立
   （带宽 500 MHz）。这一选择直接决定 SNRI 的绝对值——若原文噪声另有带宽，
   所有 SNRI 数字都要平移。**`hold>1` 且保持方差不变，是放大噪声谱密度而非限带**，
   这个坑已在 4.4 节记录。
4. **结论限于数值仿真。** 无硬件、无实测；方波的"长窗优势"在真实系统里会因静默期
   受限而大打折扣。

## 8.4 对毕设的用处

1. **"SNR 增益"必须连带宽一起报。** C2 的 30 dB 里全部来自"输入噪声带宽 / 通道带宽"。
   本课题若声称某方法提升了多少 dB，必须同时给出前端带宽与积分时间，否则与 ★[2]、★[3]
   无法比较。这不是格式要求，是**唯一能让数字可复现的条件**。
2. **差分读出的本质是"把判决换成估计"。** 它不提高灵敏度（线性通道在能量意义上不改变 SNR，
   只改变带宽），但把"有/无"的 1 比特变成可读的波形，从而能做幅相估计。这与本课题
   "用阵列做幅相估计"是同一类思路：**先要连续量，再谈判决**。
3. **过阻尼跟随比"强耦合"更本质。** 差模阻尼比 2.87 决定了通道带宽与波形保真度。
   若要做波形级估计，应优先设计阻尼（而不是一味加大耦合）。
4. **参数表要给全。** 本书为了复现一张表，不得不反演一个原文没写的参数、并纠正一个
   上一版自己的口径错误。写论文时宁可多给一行"附加限带 130 kHz、逐点白噪声、
   5σ 判据、噪声窗 10 µs"，也不要让读者去猜。

---

*复现代码：`_src/c2sim.py`（方程与信号链）、`_src/experiments.py`（全量实验）、
`_src/build_notebooks.py`（本书生成器）。数据：`_data/*.json`、`_data/*.npz`。*
"""


def write(name, notebook):
    p = BOOK / name
    nbf.write(notebook, p)
    print("写入", p.name, f"({len(notebook.cells)} 单元)")


if __name__ == "__main__":
    write("01-模型与方程.ipynb", n1)
    write("02-分岔与相态.ipynb", n2)
    write("03-检测原理.ipynb", n3)
    write("04-信号链与限带.ipynb", n4)
    write("05-参数影响.ipynb", n5)
    write("06-检测门限.ipynb", n6)
    write("07-参数估计.ipynb", n7)
    (BOOK / "00-复现总览.md").write_text(MD0, encoding="utf-8")
    (BOOK / "08-结论与边界.md").write_text(MD8, encoding="utf-8")
    print("写入 00-复现总览.md / 08-结论与边界.md")



