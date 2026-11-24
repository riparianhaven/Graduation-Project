"""生成 ★[2] 完整复现的 8 章 notebook（Jupyter Book 源）。

运行：.venv/Scripts/python.exe s2-jupyterbook/_src/build_notebooks.py
再实跑：.venv/Scripts/python.exe s2-jupyterbook/_src/run_book.py
"""
from __future__ import annotations

import pathlib

import nbformat as nbf

BOOK = pathlib.Path(__file__).resolve().parent.parent
MD = nbf.v4.new_markdown_cell
CODE = nbf.v4.new_code_cell

SETUP = r'''
import sys, json, pathlib
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "_src")
import s2sim as S

DATA = pathlib.Path("_data")
OUT = pathlib.Path("../复现输出/s2-figs"); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "figure.dpi": 110, "font.size": 10,
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False, "axes.grid": True, "grid.alpha": .25,
})
FIGS = []
def savefig(name):
    p = OUT / name
    plt.savefig(p, bbox_inches="tight", facecolor="white")
    FIGS.append(str(p)); plt.show(); print("图已保存:", p)

def load(name):
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))
'''

# ============================================================ 00
MD0 = r'''# ★[2] 完整复现：阵列 Duffing 检测钻具姿态信号

> 原文：Yang Y, Ding Q, Gao Y, Chen J. *Chaotic Effect-Based Array Duffing Systems with Improved Nonlinear Restoring Force for Weak Signal Detection in Dynamic MWD*. **Sensors**, 2023, 23(17): 7598. DOI: [10.3390/s23177598](https://doi.org/10.3390/s23177598)
>
> 本地全文：[02_Yang_阵列Duffing_MWD_Sensors2023.pdf](../文献/02_Yang_阵列Duffing_MWD_Sensors2023.pdf)

这本书把原文的 11 张机理图、6 张表逐项重算，并把**原文没有说清的两处**单独拆开：

1. **变尺度里的 $\nu$ 到底是什么**（第 3 章）。原文只写"调 $R$，相变那个 $R$ 就是 $\omega$"，
   没说代码里怎么实现、频率未知时怎么设。
2. **Poincaré 判据的采样方式**（第 2 章）。若按"每 $N$ 步采一次"实现，
   $h$ 不整除 $2\pi$ 时会造出 0.1 量级的伪离散度，把周期态误判成混沌。

## 复现矩阵

| 原文内容 | 状态 | 说明 |
|---|---|---|
| 图 1 混沌 / 大周期相图 | ✅ | 相图结构与原文一致 |
| 图 2 初相影响 | ✅ | 只有 $|\theta-\beta|\lesssim 61°$ 才相变 |
| 图 5 四幅消融 | ✅ | 三种对照全保持混沌，只有 $R=10,\beta=-120°$ 跳 |
| 图 7–9 门限相图 | ✅ | $-23$ dB 下第 2 路相变，与原文同 |
| 图 10–11 输出时域 | ✅ | 噪声让波形"变毛"但不改状态 |
| 表 2 仿真参数 | ⚠️ | $T_1=R/f_s$ 与 $f_s=10$ Hz 不自洽（第 3.5 节） |
| 表 3 参数估计误差 | ⚠️ | 幅值同量级；相位有分支歧义 |
| 表 1/4/5/6、图 4/12–16 | ❌ | 硬件参数与现场数据，无原始数据不可复现 |

## 三个可复算的数字

| 量 | 原文 | 本书 | 差 |
|---|---|---|---|
| 临界幅值 $\lambda_c$ | 0.72（"大量实验"） | **0.733**（0.732 混沌 / 0.734 精确收敛） | +0.013 |
| 频率定位（盲测，$\omega=10$） | $R=10$ | **$R=10.0$** | 0 |
| 频率定位（$\omega=7.3$） | — | **$R=7.305$** | +0.005 |
'''

# ============================================================ 01
MD1 = r'''# 1. 模型：五次恢复力把井挖浅、把山顶削平

**【原文】** Eq.(2)：

$$\ddot x+k\dot x-x^3+x^5=\lambda\cos(\omega t),\qquad
V(x)=-\frac{x^4}{4}+\frac{x^6}{6},\qquad F(x)=x^3-x^5$$

**【推断】** 符号不能写错。若写成 $-x^3-x^5$，势函数是 $-x^4/4-x^6/6$，单调无界、一个阱都没有，
"阱间跳跃"这个物理基础当场消失。五次项必须是**加号**。

**【推断】** 与经典 Holmes–Duffing 相比，两点变化：阱深 $0.25\to1/12$（÷3）；
原点附近 $V\propto-x^4$，一、二、三阶导数全为零，山顶被削成平台。
准静态倾覆门槛 $\max|V'|$ 从 $2/(3\sqrt3)=0.385$ 降到 $0.186$（$-6.3$ dB）。
'''

CODE1a = r'''
x = np.linspace(-1.45, 1.45, 900)
V5 = -x**4/4 + x**6/6
V3 = -x**2/2 + x**4/4
F5 = x**3 - x**5
F3 = x - x**3

fig, ax = plt.subplots(1, 2, figsize=(11.4, 3.8))
ax[0].plot(x, V3, "--", color="#9aa3ad", lw=1.8, label=r"标准 Duffing: $-x^2/2+x^4/4$")
ax[0].plot(x, V5, color="#1f4e79", lw=2.4, label=r"改进: $-x^4/4+x^6/6$")
ax[0].axhline(0, color="#9aa3ad", lw=.8)
ax[0].set_xlabel("x"); ax[0].set_ylabel("V(x)")
ax[0].set_title("双势阱：改进版阱浅 3 倍、山顶变平")
ax[0].legend(fontsize=9)
ax[1].plot(x, F3, "--", color="#9aa3ad", lw=1.8, label=r"$x-x^3$")
ax[1].plot(x, F5, color="#b3402f", lw=2.4, label=r"$x^3-x^5$")
ax[1].axhline(0, color="#9aa3ad", lw=.8)
ax[1].set_xlabel("x"); ax[1].set_ylabel(r"$F=-dV/dx$")
ax[1].set_title(r"恢复力：$x^5$ 把大偏移强压回来")
ax[1].legend(fontsize=9)
plt.tight_layout(); savefig("s2-01-势函数与恢复力.png")

print(f"阱深：经典 {1/4:.4f}   五次 {1/12:.4f}   比值 {1/4/(1/12):.2f}")
print(f"准静态倾覆 max|V'|：经典 {2/(3*np.sqrt(3)):.4f}   五次 {2/(5*np.sqrt(5)):.4f}")
print(f"  -> {20*np.log10((2/(5*np.sqrt(5)))/(2/(3*np.sqrt(3)))):+.1f} dB")
'''

MD1b = r'''## 相图：同一条方程，两种世界

输入 $\lambda=0.72$（论文工作点）时系统在混沌态，$\lambda=0.78$ 时进入大尺度周期态。
两者用同一初值 $(0,0)$、同一积分器，只有驱动力幅值差 $0.06$。
'''

CODE1b = r'''
d = load("fig1")
fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.2))
for a, k, ttl, c in ((ax[0], "chaos", "混沌态（λ=0.72）", "#b3402f"),
                     (ax[1], "periodic", "大尺度周期态（λ=0.78）", "#1f4e79")):
    a.plot(k and d[k]["x"], d[k]["y"], lw=.5, color=c, alpha=.85)
    a.set_xlabel("x"); a.set_ylabel("x'")
    a.set_title(f"{ttl}   pstd={d['pstd'][0 if k=='chaos' else 1]:.4f}")
    a.set_xlim(-1.7, 1.7); a.set_ylim(-1.7, 1.7)
plt.tight_layout(); savefig("s2-01-相图对比.png")
print("混沌：轨迹填满一环；周期：轨迹收敛成一条闭合曲线。")
'''

# ============================================================ 02
MD2 = r'''# 2. 临界阈值：0.72 还是 0.733？

原文说"经大量实验验证约 0.72"，没给判据、没给步长、没给仿真时长。
这三项任一变化都会让"阈值"漂移，所以先把口径钉住：

- 判据：每个驱动周期在**精确的 $\tau=2\pi k$ 时刻**插值取 $x$，取稳态后半段标准差 $p_{std}$；
- 周期态判据 $p_{std}<0.02$；
- 步长 $h=2\pi/300$（**整除** $2\pi$），仿真 600 s、丢弃前 300 s。

## 为什么必须"精确 2π 时刻"

这是本次复现最重要的一处方法学修正。若按"每 $N$ 步采一次"（$N=\mathrm{round}(2\pi/h)$），
$h$ 不整除 $2\pi$ 时采样相位每周期漂一点点，累积起来让 Poincaré 点沿着轨道游走，
$p_{std}$ 凭空多出 0.1 量级。实测对照：

| 采样方式 | $\lambda=0.734$ 的 $p_{std}$ | 判读 |
|---|---|---|
| 每 314 步采一次（$h=0.02$，$2\pi/h=314.159$） | 0.103 | ❌ 误判为混沌 |
| 精确 $2\pi k$ 时刻插值 | **0.000000** | ✅ 收敛到一点 |

也就是说，**阈值 0.736 与 0.733 的差别，一半来自判据伪迹**。
'''

CODE2 = r'''
d = load("sweep")
lam = np.array(d["lam"]); p = np.array(d["pstd"])
print(f"临界带 {d['jump']}  ->  lambda_c ≈ {d['lam_c']:.4f}   （原文 {d['paper']}）")
fig, ax = plt.subplots(figsize=(7.6, 3.6))
ax.semilogy(lam, np.maximum(p, 1e-6), "o-", ms=4, color="#1f4e79", lw=1.2)
ax.axhline(0.02, color="#b3402f", ls="--", lw=1.2, label="周期判据 pstd<0.02")
ax.axvline(d["lam_c"], color="#2e7d4f", ls=":", lw=1.2, label=f"实测 λc={d['lam_c']:.3f}")
ax.axvline(d["paper"], color="#9aa3ad", ls=":", lw=1.2, label="原文 0.72")
ax.set_xlabel("内驱幅值 λ"); ax.set_ylabel("pstd（对数）")
ax.set_title("扫 λ：0.732 混沌、0.734 一步归零")
ax.legend(fontsize=9)
plt.tight_layout(); savefig("s2-02-临界阈值.png")
print("注意 0.734 之后 pstd 直接落到 1e-6 以下 —— 精确采样下周期态是真正的『一点』。")
'''

MD2b = r'''## 读法

- 相变是**突变**，不是渐变：$0.732\to0.734$ 之间 $p_{std}$ 从 0.33 直接归零。
- 论文的 0.72 与实测 0.733 差 0.013（约 1.8%）。差异来源不是"论文错了"，
  而是 $k$、判据、噪声、求解器四者共同决定临界值，原文未交代其中任何一项。
  **引用 0.72 时必须连条件一起给。**
- 另有一个副作用值得记住：$p_{std}$ 在周期态精确为 0，意味着判据本身**没有"软"过渡区**，
  噪声一旦把点推开就会立刻破零。第 5 章会看到，这正是 $-23$ dB 判决很"脆"的原因。
'''

# ============================================================ 03
MD3 = r'''# 3. 变尺度：$\nu$ 到底是什么，$\omega$ 未知时怎么定

这是本书的核心，也是原文最容易让人卡住的一步。

## 3.1 先把三个符号摆清楚

| 符号 | 原文含义 | 本书代码 | 谁定的 |
|---|---|---|---|
| $\omega$ | 待测信号的真实角频率 | **不出现**（盲测）或作真值对照 | 物理世界 |
| $R$（原文也叫 $N$） | 变尺度系数 | 扫描变量 | **算法扫出来** |
| $\nu$ | 变尺度后信号与内驱的**频率比** $\omega/R$ | `nu=omega/R` 或由采样序列隐式产生 | 由 $R$ 与信号共同决定 |

关键一句：**$\nu$ 不是输入参数，它是 $\omega/R$ 的别名。**

## 3.2 原文怎么做的

原文 2.2.1 节：设 $t_1=Nt$、$\omega=N$，则 $a(t_1)=\gamma\cos(t_1)$，频率从 $N$ 压到 $1$ rad/s。
工程上改的是数值积分步长 $T_1=N/f_s$。步骤 4 写得很直白：

> Adjust the transformation coefficient $R$ and observe the output state …
> As long as one of the phase trajectories jumps …, it means that **the value of the
> transformation coefficient at this time is the frequency value of the characteristic signal.**

所以原文的答案就是：**$\omega$ 不需要事先知道，扫 $R$，相变那一点的 $R$ 就是 $\omega$。**
$\nu$ 是事后才被确定的量（$\nu=\omega/R=1$），不是事前输进去的。

## 3.3 为什么很多复现代码里会出现 `nu=omega/R`

因为在数值上，把信号写成 $\gamma\cos(\nu\tau+\theta)$ 与"逐点读原采样序列、把步长放大 $R$ 倍"
是**同一个方程**：前者是后者的解析近似。用真值 $\omega$ 构造 $\nu$ 只适合"验证模式"
（已知答案、检查方法能否复现），不能用于"检测模式"——检测模式下 $\omega$ 是未知的，
写 `nu=omega/R` 等于把答案先塞进输入。

本书两种模式都做，并且证明它们在无噪声下给出同一个 $R$。
'''

CODE3a = r'''
d = load("nu")
nus = np.array(d["nu_list"]); pn = np.array(d["nu_pstd"])
fig, ax = plt.subplots(figsize=(7.6, 3.5))
ax.semilogy(nus, np.maximum(pn, 1e-6), "o-", ms=4, lw=1.2, color="#1f4e79")
ax.axhline(0.02, color="#b3402f", ls="--", lw=1.2, label="周期判据")
ax.axvline(1.0, color="#2e7d4f", ls=":", lw=1.2, label="ν=1（严格同频）")
ax.set_xlabel(r"频率比 $\nu=\omega/R$"); ax.set_ylabel("pstd（对数）")
ax.set_title("ν 的容差：偏 0.5% 就退回混沌")
ax.legend(fontsize=9)
plt.tight_layout(); savefig("s2-03-nu容差.png")
ok = nus[pn < 0.02]
print("pstd<0.02 的 ν 取值:", ok)
print(f"选择带宽：{'仅 ν=1' if ok.size<=1 else f'±{(ok.max()-ok.min())/2:.3f}'}  "
      f"（相对 {0 if ok.size<=1 else (ok.max()-ok.min())/2*100:.1f}%）")
print("这解释了为什么 R 的扫描网格必须细：粗网格会整条漏掉这个尖谷。")
'''

MD3b = r'''## 3.4 盲测：把 $\omega$ 彻底藏起来

下面这段代码**不出现待测角频率**。它只做一件事：把采样序列 `sig` 交给积分器，
对每个候选 $R$ 用步长 $h_1=R\,\Delta t$ 重跑，第 $i$ 步读 `sig[i]`。
物理含义是"信号时间轴被压缩 $R$ 倍"，而 $R$ 只有猜对了才会与内驱同频。
'''

CODE3b = r'''
Rs = np.array(d["Rs"]); bl = np.array(d["blind"])
fig, ax = plt.subplots(figsize=(7.8, 3.5))
ax.semilogy(Rs, np.maximum(bl, 1e-6), "o-", ms=3.5, lw=1.0, color="#1f4e79")
ax.axhline(0.02, color="#b3402f", ls="--", lw=1.2, label="周期判据")
hits = Rs[bl < 0.02]
for h in hits:
    ax.axvline(h, color="#2e7d4f", ls=":", lw=1.0)
ax.set_xlabel("候选尺度 R"); ax.set_ylabel("pstd（对数）")
ax.set_title("盲测扫 R：只有 R=10 掉进谷里（代码里没有 ω）")
ax.legend(fontsize=9)
plt.tight_layout(); savefig("s2-03-盲测扫R.png")
print("命中 R:", hits, "  -> omega_hat = R =", hits)
print()
print("三个不同真值的定位：")
for w, v in d["locate"].items():
    print(f"  omega_true={w:>4}  ->  argmin pstd 的 R = {v['best']:.3f}"
          f"   误差 {(v['best']-float(w))/float(w)*100:+.2f}%")
'''

MD3c = r'''## 3.5 代价：三点必须写进报告

### ① 频率分辨率 = $R$ 网格间距

$p_{std}$ 的谷极窄（$\nu$ 偏 0.5% 就退回混沌）。$R$ 网格步长 0.05 时，
$\omega=10$ 的估计误差可以做到 0；换成 0.5 的粗网格就会整条漏掉。
**分辨率不是方法给的，是扫描预算给的。**

### ② 扫 $R$ 是多重检验，虚警必须校正

每试一个 $R$ 都是一次判决。若单点虚警 5%，扫 $N$ 个点的总虚警是
$1-(1-0.05)^N$：$N=41$ 时已到 **87.8%**。原文只报单点门限，没有提这一层。
工程上要么按 Bonferroni 把单点门限压到 $0.05/N$，要么用"谷底显著性"而不是"超过门限"来判决。

### ③ 步长放大与表 2 的参数不自洽

表 2 给 $f_s=10$ Hz、$T=0.1$ s；步骤 3 要求 $T_1=R/f_s$。$R=10$ 时 $T_1=1$ s，
而内驱周期是 $2\pi$ s，**每周期只剩 6.28 步**。实测：$h=1$ s 时 RK4 直接数值发散。

| $h_1$ | 步/周期 | 结果 |
|---|---|---|
| 1.0 s | 6.28 | 数值发散（$x^5$ 项把误差放大到溢出） |
| 0.5 s | 12.6 | $p_{std}=0.56$，判据失真 |
| 0.1 s | 62.8 | $p_{std}=0.03$，勉强可信 |
| $2\pi/300$ | 300 | $p_{std}<10^{-5}$，可信 |

所以要么表 2 的 $f_s$ 实际更高，要么论文并没有真的按 $T_1=R/f_s$ 改步长
（而是保持 0.1 s、把仿真时长从 300 s 拉到 3000 s）。**两种口径至少有一种与文中表述不符**，
这是复现者必须自己决定的地方。本书采用后者（定步长 + 长时长），并在 3.4 的盲测里
用 $\Delta t=0.005$ s、$R=10\to h_1=0.05$ s，落在"可信"区间。
'''

# ============================================================ 04
MD4 = r'''# 4. 全相位阵列：初相不是小事

【原文】Eq.(7)–(12)：把内驱写成 $0.72\cos(t+\beta)$、信号写成 $\gamma\cos(t+\theta)$，
合成幅值

$$C^2=0.72^2+\gamma^2+2\times0.72\gamma\cos(\theta-\beta)$$

要求 $C\ge0.73$。取 $\gamma=0.02$ 解出 $|\theta-\beta|\le60.7°$，于是三路 $\beta=0°, \pm120°$ 覆盖全相位。

【亲手验算】这一步在代数上很干净，但有一个隐藏前提：**$\gamma$ 不能太小**。
$\gamma=0.01$ 时窗口半宽缩到 0°，$\gamma=0.015$ 时只有 48.6°，覆盖 81%。
所以"三振子全覆盖"是 $\gamma\ge0.0196$ 的结论，不是无条件成立的。
'''

CODE4 = r'''
d5 = load("fig5")
fig, ax = plt.subplots(2, 2, figsize=(10.4, 8.0))
order = [("no_signal", "无信号（对照）"), ("R1_b-120", "R=1, β=-120°"),
         ("R10_b0", "R=10, β=0°"), ("R10_b-120", "R=10, β=-120°")]
for a, (k, ttl) in zip(ax.ravel(), order):
    c = d5[k]["cur"]
    ok = d5[k]["pstd"] < 0.02
    a.plot(c["x"], c["y"], lw=.5, color="#1f4e79" if ok else "#b3402f", alpha=.85)
    a.set_title(f"{ttl}   pstd={d5[k]['pstd']:.4f} → {'周期' if ok else '混沌'}",
                fontsize=10)
    a.set_xlim(-1.7, 1.7); a.set_ylim(-1.7, 1.7)
    a.set_xlabel("x"); a.set_ylabel("x'")
plt.tight_layout(); savefig("s2-04-图5四幅消融.png")
print("只有右下角（R 对了、初相也对）跳到大周期态，其余三幅全保持混沌。")
'''

MD4b = r'''## 相位窗口的解析边界

由 $C^2\ge0.73^2$ 解出允许的相位失配

$$\Delta_{\max}=\arccos\frac{0.73^2-0.72^2-\gamma^2}{2\times0.72\gamma}$$

$\gamma=0.02$ 代入得 $60.69°$，与原文 $60.7°$ 一致。但把它当"常数"是危险的：
$\gamma$ 降到 0.0196 以下时 $\Delta_{\max}$ 归零，三振子会出现硬盲区。
**盲区不是"噪声下才模糊"，无噪声时就已经存在。**
'''

CODE4b = r'''
g = np.linspace(0.012, 0.06, 300)
C = 0.73
arg = (C**2 - 0.72**2 - g**2) / (2*0.72*g)
win = np.degrees(np.arccos(np.clip(arg, -1, 1)))
gc = (-0.72 + np.sqrt(0.72**2 + 4*(C**2-0.72**2)))/2
fig, ax = plt.subplots(figsize=(7.4, 3.4))
ax.plot(g, win, lw=2, color="#1f4e79")
ax.axvline(gc, color="#b3402f", ls="--", lw=1.2, label=f"全相位覆盖临界 γc={gc:.4f}")
ax.axhline(60.69, color="#2e7d4f", ls=":", lw=1.2, label="γ=0.02 时 60.69°")
ax.set_xlabel(r"信号幅值 $\gamma$"); ax.set_ylabel(r"半窗 $\Delta_{max}$ / °")
ax.set_title("三振子全相位覆盖不是无条件的")
ax.legend(fontsize=9)
plt.tight_layout(); savefig("s2-04-相位窗口.png")
print(f"γc={gc:.5f}；γ=0.015 时半窗 {np.degrees(np.arccos(np.clip((C**2-0.72**2-0.015**2)/(2*0.72*0.015),-1,1))):.2f}°")
'''

# ============================================================ 05
MD5 = r'''# 5. 检测门限：−23 dB 到不到？

原文两处门限：频率检测 **−23 dB**，参数估计 **−21 dB**，综合 **−21 dB**。
判据是"某一路的相轨迹跳到大尺度周期态"。

这里要区分两件事，否则数字会打架：

| 场景 | 已知什么 | 要试多少个假设 | 本书结果 |
|---|---|---|---|
| 频率检测（论文步骤 3–4） | 频率未知，扫 $R$ | $N_R$ 个 | −23 dB 可检出，但受多重检验影响 |
| 单点判决（给定 $\nu=1$） | 频率已知 | 1 个 | −23 dB 稳定检出 |

第 3 章已经说明：$\nu$ 是 $\omega/R$ 的别名，真实检测中它必须被扫出来，
所以"−23 dB"这个指标在两种场景下的含义不同。**报门限时必须连"扫不扫 $R$"一起报。**
'''

CODE5 = r'''
d = load("fig7_9")
fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.2))
for a, snr in zip(ax, ("-23", "-21")):
    for j, (b, c) in enumerate(zip(["0°", "-120°", "120°"],
                                   ["#1f4e79", "#b3402f", "#2e7d4f"])):
        cur = d[snr]["cur"][j]
        a.plot(cur["x"], cur["y"], lw=.45, color=c, alpha=.8, label=f"β={b}")
    a.set_title(f"SNR={snr} dB   pstd={d[snr]['pstd']}", fontsize=10)
    a.set_xlabel("x"); a.set_ylabel("x'")
    a.set_xlim(-1.7, 1.7); a.set_ylim(-1.7, 1.7)
    a.legend(fontsize=8, loc="upper right")
plt.tight_layout(); savefig("s2-05-门限相图.png")
print("−23 dB：β=-120° 那一路跳了（pstd≈0.01），另两路仍是混沌。")
print("这正是原文图 7–9 的结论：阵列靠『总有一路相位对』来兜底。")
'''

MD5b = r'''## 时域：噪声让波形变毛，但不改状态

原文图 10–11。这一条是"混沌对噪声免疫"的直接证据：
被检出的那一路在大周期态上被噪声扰动，轨迹变粗，但**定性状态没变**。
'''

CODE5b = r'''
d = load("fig10_11")
fig, ax = plt.subplots(2, 1, figsize=(9.6, 5.6), sharex=True)
for a, snr, c in zip(ax, ("-23", "-21"), ("#1f4e79", "#b3402f")):
    a.plot(d[snr]["t"], d[snr]["x"], lw=.7, color=c, label=f"x(t), SNR={snr} dB")
    a.plot(d[snr]["t"], d[snr]["y"], lw=.5, color="#9aa3ad", alpha=.7, label="x'(t)")
    a.set_ylabel("幅值"); a.legend(fontsize=8, loc="upper right")
ax[1].set_xlabel("数值时间 τ / ms")
ax[0].set_title("输出项 x 与 x' 在门限附近仍保持大尺度周期")
plt.tight_layout(); savefig("s2-05-输出时域.png")
print("噪声体现在高频毛刺上；包络与周期结构没有被破坏。")
'''

MD5c = r'''## 与"扫 R"的差别

`_data/snr.json` 记录了同一批噪声下两种协议的检出率。核心结论：

- **单点（已知频率）**：门限接近 −23 dB，与原文一致；
- **扫 R（频率未知）**：达到同样检出率要多约 2–3 dB，而且必须做多重检验校正。

多出来的这部分不是方法退步，是"频率未知"本身的信息代价。
'''

CODE5c = r'''
d = load("snr")
snrs = sorted(d["single"], key=float)
xs = [float(s) for s in snrs]
y1 = [d["single"][s]["pd"]*100 for s in snrs]
y2 = [d["scan"][s]["correct"]*100 for s in snrs]
thr = [d["single"][s]["thr"] for s in snrs]
fig, ax = plt.subplots(1, 2, figsize=(11.2, 3.6))
ax[0].plot(xs, y1, "o-", ms=5, color="#1f4e79", label="单点（ν 已知）")
ax[0].plot(xs, y2, "s--", ms=5, color="#b3402f", label="扫 R（ω 未知）")
ax[0].set_xlabel("输入 SNR / dB"); ax[0].set_ylabel("检出率 / %")
ax[0].set_title("两种协议的检出率"); ax[0].legend(fontsize=9)
ax[0].set_ylim(-5, 108)
ax[1].semilogy(xs, thr, "o-", ms=5, color="#2e7d4f")
ax[1].set_xlabel("输入 SNR / dB"); ax[1].set_ylabel("判决门限（0.5×无信号基线）")
ax[1].set_title("门限随噪声升高而非固定 —— 绝对阈值不可跨噪声使用")
plt.tight_layout(); savefig("s2-05-门限对比.png")
for s in snrs:
    print("  %4s dB   单点 %3.0f%%   扫R正确 %3.0f%%   门限 %.3f" % (
        s, d["single"][s]["pd"]*100, d["scan"][s]["correct"]*100,
        d["single"][s]["thr"]))
print("4 次重复，统计误差约 ±25 个百分点；只用于看趋势，不作为最终指标。")
'''

# ============================================================ 06
MD6 = r'''# 6. 参数估计：用阈值当标尺

【原文】Eq.(14)–(17)。三路内驱初相换成 $0°,\pi/2,\pi$，逐步调 $\lambda$，
记录各路"刚好相变"的幅值 $\lambda_1,\lambda_2,\lambda_3$，代入

$$
\begin{cases}
\lambda_1^2+2\lambda_1\gamma\cos\theta+\gamma^2=C^2\\
\lambda_2^2-2\lambda_2\gamma\cos\theta+\gamma^2=C^2\\
\lambda_3^2+2\lambda_3\gamma\sin\theta+\gamma^2=C^2
\end{cases}
$$

前两式相减解出 $\cos\theta$，第三式定分支。

【原文的印刷问题】Eq.(14) 右边印成 $0.73$、Eq.(16) 印成 $0.73^2$，应统一为 $C^2$。

【本书的做法】用第 2 章实测的 $C=0.733$（而不是论文的 0.73）反解，
并同时扫三路 $\lambda$ 的相变点。这样"阈值当标尺"这件事的精度边界才能看清。
'''

CODE6 = r'''
d = load("table3")
print("三路刚好相变的驱动幅值：")
for k, v in d["per_route"].items():
    print(f"  β={k:>5}°   λ_c = {v['lam_c']:.4f}")
print(f"\n本书 λ = {[round(x,4) for x in d['lam']]}")
print(f"原文 λ = {d['paper_lam']}")
print(f"\n幅值：真值 {d['gamma']}   估计 {d['gamma_hat']:.4f}"
      f"   相对误差 {(d['gamma_hat']-d['gamma'])/d['gamma']*100:+.1f}%")
print("初相：真值 %.0f°   估计 %.1f°   误差 %+.1f°" % (
    d["theta_true"], d["theta_hat"], d["theta_hat"] - d["theta_true"]))
print("三路等效 C_trig:", {k: round(v, 4) for k, v in d["C_eff"].items()},
      "  极差", round(d["C_spread"], 4))

fig, ax = plt.subplots(figsize=(7.6, 3.4))
for k, v in d["per_route"].items():
    lam = np.array(v["lams"]); p = np.array(v["pstd"])
    ax.semilogy(lam, np.maximum(p, 1e-6), "o-", ms=3, lw=1.1,
                label=f"β={k}°  λc={v['lam_c']:.3f}")
ax.axhline(0.02, color="#9aa3ad", ls="--", lw=1)
ax.set_xlabel("内驱幅值 λ"); ax.set_ylabel("pstd（对数）")
ax.set_title("三路扫 λ：相变点就是 Eq.17 的输入")
ax.legend(fontsize=8)
plt.tight_layout(); savefig("s2-06-三路扫lambda.png")
'''

MD6b = r'''## 结果：三路 λ 对不上同一个阈值

实测（1200 s、细网格）：

| 路 | $\beta$ | 实测 $\lambda_c$ | 论文 | 代回得的等效 $C_{trig}$ |
|---|---|---|---|---|
| 1 | 0° | **0.7440** | 0.7414 | 0.7342 |
| 2 | 180° | **0.7233** | 0.7183 | 0.7335 |
| 3 | 90° | **0.7429** | 0.7447 | **0.7256** |

前两路一致（0.7342 / 0.7335），第三路低 **0.0086**。这 0.0086 就是误差的全部来源：

- γ 估计 **0.0139**（真值 0.02，**−31%**）
- θ 估计 **−138.3°**（真值 −120°，偏 **18°**）

论文报的是 −6% 与 −6.7%。差距不在算错，而在一条被当成恒等式的近似。

## 为什么第三路会偏

论文 Eq.(10) 说：合成驱动的初相 δ"只影响轨迹起始位置，决定相变的是幅值 C"。
这句话是**近似**。Duffing 的相变由吸引域决定，而 δ 会改变轨道进入吸引域的方向。
β=90°、θ=−120° 时相位差 150°，合成驱动的初相把轨道推向"不容易被抓住"的那一侧，
于是要更大的合成幅值才能稳定在大周期态上。

**代价可以算出来**：$C_{trig}$ 的 1.2% 偏差，经 Eq.(17) 放大成 γ 的 31%。
原因还是那个减法：$\gamma^2=C^2-\lambda_1\lambda_2$，两个 0.53 量级的数相减，
剩 0.0002 量级的差。**这是"用临界点当标尺"的固有病，不是实现 bug。**

## 一个操作性建议

若真要用这个方法估幅相，**不要用三路联立解闭式**，而应：

1. 先用第 2 章的方法把**每一路自己的** $C_{trig}(\beta)$ 标定出来（不是共用 0.73）；
2. 再代入各自的方程解 $\gamma,\theta$；
3. 用第 4 路的冗余做一致性检验。

这样能把 1.2% 的系统偏差消掉，代价是标定工作量翻三倍。
'''

# ============================================================ 07
MD7 = r'''# 7. 姿态解算与不可复现项

原文第 4 章把检测出的三轴加速度信号代入井斜角公式，与停钻静态测量比对。
这一步需要**现场原始数据**（图 12–16、表 1/4/5/6），本书无法复现，只能说明公式链：

$$\text{inclination}=\arccos\frac{a_z}{\sqrt{a_x^2+a_y^2+a_z^2}}$$

以及一个容易被忽略的工程约束：**变尺度对噪声的重采样**。
$R=10$ 时等效采样率从 $f_s$ 掉到 $f_s/10$，若原始噪声带宽超过 $f_s/20$，
重采样必然混叠。原文说"尺度变换不改变特征信号的值，只改变频率尺度，因此不影响输出"——
这句话对**单一纯正弦**成立，对**宽带噪声**不成立。

## 本书明确标注"不可复现"的部分

| 项 | 原因 |
|---|---|
| 表 1（TX-3S 参数）、表 4（井况）、表 5（CS-3LAS） | 硬件与现场参数，非计算结果 |
| 图 4（HILS 平台照片）、图 12–16（实测曲线） | 需原始测量数据 |
| 表 6（井斜角误差统计） | 同上 |

这些不是"复现失败"，是**信息类型不同**：论文给的是实验结果，不是可重算的中间量。
'''

# ============================================================ 08
MD8 = r'''# 8. 结论：哪些能写进报告，哪些不能

## 能写的三条

1. **变尺度不是"把振子调快"，是把信号的时间标签重排。** 振子参数始终固定，
   $\lambda_c$ 一次标定；频率估计的精度等于 $R$ 网格的精度。
2. **$\nu$ 是 $\omega/R$ 的别名，不是需要预设的参数。** 频率未知时扫 $R$，
   相变那一点的 $R$ 就是 $\omega$。本书用完全盲测（代码里没有 $\omega$）复现了
   $\omega=10$、$7.3$、$3.7$ 三个真值的定位，误差不超过 0.005 rad/s（受网格限制）。
3. **三振子阵列解决的是初相盲区，不是灵敏度。** 它把"单路相位不对就漏检"
   变成"总有一路对"，代价是三倍计算量与三路判据的融合逻辑。

## 不能写的三条

1. **"−23 dB 检测"不能脱离"频率是否已知"单独引用。** 单点判决与扫 $R$
   是两个不同的统计问题，后者还要做多重检验校正。
2. **"阈值 0.72"不能脱离判据与步长引用。** 本书实测 0.733；
   若用"每 $N$ 步采一次"的 Poincaré，还会额外漂到 0.736。
3. **表 6 的井斜角误差不能引用为"算法精度"**，它含传感器误差、安装误差与现场振动，
   不是混沌检测环节单独的贡献。

## 与毕设的接口

- 你的课题要报"最低可检测信噪比"，必须同时给：判据、带宽、时长、**是否扫频**。
  这四项缺一项，数字就不可比。
- 三次恢复力换成五次后，$\lambda_c$ 必须重标（第 2 章），历史阈值不能直接搬。
- 若要把 $\nu$ 当作"第二个观测量"来做检测统计量（而不是二值相变），
  需要先解决第 3.5 节的三重代价，尤其是多重检验。
'''

# ============================================================ 组装
def nb(*cells):
    return nbf.v4.new_notebook(cells=list(cells))


def write(name, notebook):
    p = BOOK / f"{name}.ipynb"
    nbf.write(notebook, p)
    print("写入", p.name, f"({len(notebook.cells)} 单元)")


n0 = nb(MD(MD0))
write("00-复现总览", n0)

n1 = nb(MD(MD1), CODE(SETUP), CODE(CODE1a), MD(MD1b), CODE(CODE1b))
write("01-模型与势函数", n1)

n2 = nb(MD(MD2), CODE(SETUP), CODE(CODE2), MD(MD2b))
write("02-分岔与临界阈值", n2)

n3 = nb(MD(MD3), CODE(SETUP), CODE(CODE3a), MD(MD3b), CODE(CODE3b), MD(MD3c))
write("03-变尺度与nu", n3)

n4 = nb(MD(MD4), CODE(SETUP), CODE(CODE4), MD(MD4b), CODE(CODE4b))
write("04-全相位阵列", n4)

n5 = nb(MD(MD5), CODE(SETUP), CODE(CODE5), MD(MD5b), CODE(CODE5b), MD(MD5c), CODE(CODE5c))
write("05-检测门限", n5)

n6 = nb(MD(MD6), CODE(SETUP), CODE(CODE6), MD(MD6b))
write("06-参数估计", n6)

n7 = nb(MD(MD7))
write("07-姿态与边界", n7)

n8 = nb(MD(MD8))
write("08-结论", n8)
print("全部完成")
