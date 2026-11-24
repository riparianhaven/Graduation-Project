# -*- coding: utf-8 -*-
"""生成 learn/2.ipynb（阶段 2：建模与相变，回答「带着问题读」的五问）。

写法参考 learn/0.ipynb 与 learn/1.ipynb：markdown 讲条件 + 代码给可复核数字。
内核代码改编自 复现-2-混沌振子建模与临界阈值.ipynb 的 simulate()，只取阶段 2 地基部分；
§7 的变尺度反例保留，因为它是「相变 ≠ 频率估计」的关键证据。

运行：
    .venv\\Scripts\\python.exe tools\\build_stage2_notebook.py
    .venv\\Scripts\\python.exe tools\\run_ipynb.py --input learn/2.ipynb --timeout 600
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / 'learn' / '2.ipynb'

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip('\n')))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip('\n')))


# ---------------------------------------------------------------- 开场
md(r'''
# 阶段 2：建模与相变 —— 回答「方程是什么、相图怎么变」

> **阶段定位**：第 3–5 周｜过关产出 = 单振子 RK4 仿真程序 + 混沌态/大尺度周期态两幅并列相图 + x(t) 时序图 + 一张可复现参数表（[阅读路线](../00-总览/阅读路线.md) 阶段 2）。对应课题「主要内容 2」的三个动词：**建立模型 / 分析相轨迹演变 / 确定临界阈值**。
>
> **本号要回答的五个问题**（阅读路线阶段 2「带着问题读」）：
> 1. Holmes–Duffing 方程各项物理含义？
> 2. 阻尼 $k$、策动力 $\lambda$、内置频率 $\omega$ 各怎么取？
> 3. RK4 步长取多少才稳定？
> 4. $\lambda$ 从小到大时相轨迹如何从混沌跃到大尺度周期态？
> 5. 「变尺度」为什么能解决极低频？
>
> **边界**：本号做到「阈值带标定 + 变尺度原理与反例」为止。判态伪代码与初相位盲区属于阶段 3；$P_d/P_{fa}$、ROC、门限数值属于阶段 4；定点化属于阶段 5。
>
> **纪律**：经验阈值报**带**不报点；本复现检验的是论文 claim 的**敏感性与可复现性**，不是「证明论文错了」——求解器、步长、判据、噪声条件都不同。
''')

# ---------------------------------------------------------------- 环境
code(r'''
# 环境、全局参数、绘图设置（与 learn/0.ipynb / 1.ipynb 同款）
%matplotlib inline
import math, json, time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 105
plt.rcParams['savefig.dpi'] = 135
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.28
plt.rcParams['mathtext.fontset'] = 'dejavusans'

def find_root():
    p = Path.cwd().resolve()
    for q in [p, *p.parents]:
        if ((q / '00-总览/阅读路线.md').is_file() or (q / '阅读路线.md').is_file()):
            return q
    return p

ROOT = find_root()
OUT = ROOT / '复现输出'
OUT.mkdir(exist_ok=True)

def savefig(name):
    path = OUT / name
    plt.savefig(path, bbox_inches='tight')
    print(f'[图] {path}')

# ---- 可复现参数表（★[2] Eq.2/3 + 本机复现条件）----
K        = 0.5      # 阻尼 k（论文给定）
OMEGA    = 1.0      # 内驱角频率（rad/s）；极低频靠变尺度折到这里
LAM0     = 0.72     # 论文预置工作点（经验值，本号重新标定）
C_TRIG   = 0.73     # 论文声称的触发合成幅值（经验值）
H        = 0.02     # RK4 步长（s）；论文用 Simulink Ode4 / h=0.1
T        = 400.0    # 总仿真时长（s）
BURN     = 200.0    # 丢弃瞬态（s）
T_FAST   = 200.0    # 批量扫描短时长
BURN_FAST= 100.0
PSTD_TH  = 0.08     # Poincaré 离散度判态阈值（本机条件）
_START = time.time()

print(f'ROOT = {ROOT}')
print(f'k={K}, ω={OMEGA} rad/s, λ0={LAM0}, h={H}s, T={T}s, burn={BURN}s, pstd阈值={PSTD_TH}')
''')

# ---------------------------------------------------------------- Q1
md(r'''
## 1｜问 1：方程各项的物理含义

标准 Holmes–Duffing 是 $\ddot x+k\dot x-x+x^3=\lambda\cos\omega t$。母本 ★[2] 把线性恢复力换成五次项（Eq.2/3）：

$$\ddot x+0.5\dot x-x^3+x^5=\underbrace{\lambda\cos(t+\beta)}_{\text{内置参考}}+\underbrace{\gamma\cos(t+\theta)}_{\text{待测信号}}+n(t)$$

写成一阶系统令 $y=\dot x$：$\dot x=y,\ \dot y=-0.5y+x^3-x^5+u(t)$。恢复力 $F(x)=x^3-x^5=-V'(x)$ 来自双阱势

$$V(x)=-\frac{x^4}{4}+\frac{x^6}{6}$$

| 项 | 作用 | 取值 / 出处 |
|---|---|---|
| $\ddot x$ | 惯性 | 状态 $x$ 位移、$y$ 速度，二阶拆两个一阶才能积分 |
| $0.5\dot x$ | 阻尼（耗散） | $k=0.5$【★[2] Eq.2】；没有它系统不收敛到吸引子 |
| $-x^3+x^5$ | 改进恢复力 | 双阱、阱壁比标准 Duffing 更陡【★[2]】；势阱底在 $x=\pm1$，$V(\pm1)=-1/12$，势垒在 $x=0$，$V(0)=0$ |
| $\lambda\cos(t+\beta)$ | 内驱 / 相干参考 | $\omega=1$ rad/s、$\beta$ 已知；$\lambda$ 是**工作点**，本号要标定 |
| $\gamma\cos(t+\theta)+n(t)$ | 待测 + 噪声 | $\gamma\ll\lambda$、$\theta$ 未知；只有同频分量能与内驱相干叠加 |

**直觉**：振子是在双阱里被周期性推搡的小球。推力小，在两阱间无规则翻跳（混沌）；推力越过临界，跟着外力整圈绕行（大尺度周期）。检测不是放大信号，是**让系统跨过一条分界线**——这条线得自己标定。
''')

code(r'''
# 势函数与恢复力：核实双阱位置与势垒高度
x = np.linspace(-1.55, 1.55, 800)
V = -x**4 / 4 + x**6 / 6
F = x**3 - x**5

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
axes[0].plot(x, V, lw=1.8, color='#2f6fd0')
axes[0].axvline(-1, color='#1a8a5a', ls=':', lw=1.2)
axes[0].axvline(1, color='#1a8a5a', ls=':', lw=1.2)
axes[0].set_xlabel('x'); axes[0].set_ylabel('V(x)')
axes[0].set_title('双阱势 $V(x)=-x^4/4+x^6/6$')
axes[1].plot(x, F, lw=1.8, color='#c0392b')
axes[1].axhline(0, color='k', lw=0.8)
axes[1].set_xlabel('x'); axes[1].set_ylabel('F(x) = x³ − x⁵')
axes[1].set_title('恢复力：±1 与 0 是平衡点')
plt.show()

print(f'势阱底 x=±1 时 V = {(-1/4 + 1/6):.6f}；势垒 V(0)=0，高度 = {1/12:.6f}')
print('平衡点由 x³(1−x²)=0 给出：x=0（不稳）与 x=±1（稳定）——这是"双阱"的代数来源。')
''')

# ---------------------------------------------------------------- Q2 + solver
md(r'''
## 2｜问 2：参数怎么取 + 求解器

**可复现参数表**（抄进论文时把「本号实测」列填进去）：

| 参数 | 值 | 来源 / 条件 |
|---|---|---|
| 阻尼 $k$ | 0.5 | 【★[2] Eq.2】论文给定 |
| 内驱频率 $\omega$ | 1 rad/s | 【★[2]】极低频靠变尺度折算，不另设参数 |
| 内驱幅值 $\lambda_0$ | 0.72 | 【★[2]】经验工作点，本号重标 |
| 触发合成幅值 | 0.73 | 【★[2]】经验值，本号重标 |
| 求解器 / 步长 | RK4 / $h=0.02$ s | 【本号】论文 Simulink Ode4 / $h=0.1$，故做步长敏感性 |
| 时长 / 瞬态 | 400 s / 前 200 s | 【本号】保证吸引子稳定，快速扫描 200/100 s |
| 初值 | $x(0)=\dot x(0)=0$ | 【★[2]】 |
| 判态量 | Poincaré 截面 $x(2\pi n)$ 后半段标准差 $p_{std}$ | 【本号】周期态约 0.01、混沌态 0.16–0.63，阈值 0.08 |
| 噪声离散 | $\sigma\sqrt h\,N(0,1)$ | 【本号】$\sigma$ 为连续强度；§1–§6 先不注入噪声 |

**求解器**：一次循环并行跑多路参数，支持同频相量合成（$\nu=1$ 走 Eq.(8)）与双频直接积分（$\nu\neq1$，变尺度用）。
''')

code(r'''
def simulate(lam, beta=0.0, gamma=0.0, theta=0.0, sigma=0.0, seed=0,
             h=H, t_end=T, burn=BURN, nu=1.0, trace=False):
    """向量化 RK4 积分 x'' = -k x' + x³ - x⁵ + u(t)。

    lam/beta/gamma/theta/sigma 可为标量或等长数组，一次循环并行跑完整组实验。
    nu = 待测信号频率 / 内驱频率；nu=1 走相量合成 A cos t − B sin t，否则双频直接积分。
    返回 pstd（Poincaré 后半段标准差）、mx（稳态最大 |x|）、轨迹（可选）。
    """
    lam, beta, gamma, theta, sigma = [a.ravel().astype(float).copy() for a in
        np.broadcast_arrays(np.asarray(lam, float), np.asarray(beta, float),
                            np.asarray(gamma, float), np.asarray(theta, float),
                            np.asarray(sigma, float))]
    m = lam.size
    coherent = abs(nu - 1.0) < 1e-12
    if coherent:   # Eq.(8)：两路同频余弦合成相量，顺带验证推导
        A = lam * np.cos(beta) + gamma * np.cos(theta)
        B = lam * np.sin(beta) + gamma * np.sin(theta)

    x = np.zeros(m); y = np.zeros(m)
    rng = np.random.default_rng(seed)
    steps, burn_steps = int(round(t_end / h)), int(round(burn / h))
    TWOPI = 2 * math.pi
    idx_prev = 0                      # 已采到的内驱周期序号（Poincaré 截面）
    sqrt_h = math.sqrt(h)
    noisy = bool(np.any(sigma))
    poincare, mx = [], np.zeros(m)
    tr_t, tr_x, tr_y = [], [], []

    for i in range(steps):
        t = i * h
        nz = rng.standard_normal(m) * sqrt_h if noisy else 0.0
        if coherent:
            f1 = A * math.cos(t)       - B * math.sin(t)       + sigma * nz
            fh = A * math.cos(t+h/2)   - B * math.sin(t+h/2)   + sigma * nz
            f4 = A * math.cos(t+h)     - B * math.sin(t+h)     + sigma * nz
        else:
            f1 = lam*np.cos(t+beta)       + gamma*np.cos(nu*t+theta)       + sigma*nz
            fh = lam*np.cos(t+h/2+beta)   + gamma*np.cos(nu*(t+h/2)+theta) + sigma*nz
            f4 = lam*np.cos(t+h+beta)     + gamma*np.cos(nu*(t+h)+theta)   + sigma*nz

        k1x = y;            k1y = -K*y  + x**3  - x**5  + f1
        x2, y2 = x+h/2*k1x, y+h/2*k1y
        k2x = y2;           k2y = -K*y2 + x2**3 - x2**5 + fh
        x3, y3 = x+h/2*k2x, y+h/2*k2y
        k3x = y3;           k3y = -K*y3 + x3**3 - x3**5 + fh
        x4, y4 = x+h*k3x,   y+h*k3y
        k4x = y4;           k4y = -K*y4 + x4**3 - x4**5 + f4
        x_prev = x.copy()                      # 更新前存档，供 2π 时刻插值
        x += h/6*(k1x + 2*k2x + 2*k3x + k4x)
        y += h/6*(k1y + 2*k2y + 2*k3y + k4y)

        if i >= burn_steps:
            np.maximum(mx, np.abs(x), out=mx)
            # 精确落在 τ=2πk 上取截面：h 不整除 2π（如 2π/0.02=314.159）时，
            # 按步数取整会让采样相位每周期漂，pstd 凭空多出 1e-2 量级。
            idx = int(math.floor((t + h) / TWOPI + 1e-12))
            if idx != idx_prev and idx > 0:
                frac = min(max((TWOPI * idx - t) / h, 0.0), 1.0)
                poincare.append(x_prev + frac * (x - x_prev))
                idx_prev = idx
            if trace:
                tr_t.append(t); tr_x.append(x.copy()); tr_y.append(y.copy())

    p_all = np.asarray(poincare)
    p = p_all[len(p_all)//2:] if p_all.size else np.empty((0, m))
    pstd = p.std(axis=0) if p.size else np.full(m, np.nan)
    return dict(pstd=pstd, mx=mx, p=p,
                t=np.array(tr_t) if trace else None,
                X=np.array(tr_x) if trace else None,
                Y=np.array(tr_y) if trace else None)

# 冒烟：0.72 应保持混沌，0.76 应进入大周期态
r = simulate([LAM0, 0.76], seed=0)
print('冒烟 pstd:', np.round(r['pstd'], 4),
      '-> 判态:', ['周期' if v < PSTD_TH else '混沌' for v in r['pstd']])
''')

# ---------------------------------------------------------------- Q3
md(r'''
## 3｜问 3：RK4 步长取多少才稳定

**这里「稳定」有两层意思，别混**：

1. **数值不发散**：因为 $-x^3+x^5$ 在大 $|x|$ 时强抑制，粗到 $h=0.5$ s 也不会爆掉——所以「没 NaN」不能当精度证据。
2. **结论不随步长漂**：同一套 $\lambda$ 扫描，换 $h$ 后阈值带位置是否一致。这才是要检验的。

做法：§5 先用 $h=0.02$ s 定出阈值带，再用 $h=0.1/0.05/0.02/0.01$ s 四种步长在同一窗口重扫，看跃变是否落在同一带内。**本机选 $h=0.02$ s**（比论文的 0.1 细一档），结论对步长不敏感，但跃变区间随 $h$ 最多平移 0.01，因此阈值报区间不报点。
''')

# ---------------------------------------------------------------- Q4 phase portraits
md(r'''
## 4｜问 4：相轨迹怎么从混沌跃到大周期态

固定内驱 $\lambda$，无目标、无噪声，跑两条轨迹：$\lambda=0.70$（预期混沌）与 $\lambda=0.76$（预期大周期）。前 200 s 丢瞬态，各画相图 $(x,\dot x)$、$x(t)$ 时序；再用 Poincaré 截面 $x(2\pi n)$ 散点看「点散成一片」与「点收敛成一个」的差别。

> 判态必须可计算：后面会看到，拿 $\max|x|>1.2$ 当判据会把所有状态都判成周期态（混沌轨道本来就偶尔跑很远），虚警 100%。**目测相图不是判据。**
''')

code(r'''
r_tr = simulate([0.70, 0.76], seed=1, trace=True)

fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
names = ['λ=0.70（混沌侧）', 'λ=0.76（周期侧）']
for j in range(2):
    axes[0, j].plot(r_tr['X'][:, j], r_tr['Y'][:, j], lw=0.45)
    axes[0, j].set_title(f'相图：{names[j]}')
    axes[0, j].set_xlabel('x'); axes[0, j].set_ylabel('$\\dot{x}$')
    axes[1, j].plot(r_tr['t'], r_tr['X'][:, j], lw=0.6)
    axes[1, j].set_title(f'时序：{names[j]}（pstd={r_tr["pstd"][j]:.4f}）')
    axes[1, j].set_xlabel('模型时间 t'); axes[1, j].set_ylabel('x(t)')
savefig('阶段2-混沌与周期相图时序.png')
plt.show()

fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
for j, lab in zip([0, 1], ['λ=0.70 混沌', 'λ=0.76 大周期']):
    ax.plot(r_tr['p'][:, j], np.full(r_tr['p'].shape[0], j) + 0.12 * np.random.default_rng(j).standard_normal(r_tr['p'].shape[0]),
            '.', ms=3, alpha=0.6, label=lab)
ax.set_yticks([0, 1], ['λ=0.70', 'λ=0.76'])
ax.set_xlabel('Poincaré 截面点 x(2πn)'); ax.set_title('截面散点：一片 vs 一点')
ax.legend(fontsize=9)
savefig('阶段2-Poincare截面散点.png')
plt.show()

print(f'λ=0.70: pstd={r_tr["pstd"][0]:.4f}（判{"周期" if r_tr["pstd"][0]<PSTD_TH else "混沌"}）')
print(f'λ=0.76: pstd={r_tr["pstd"][1]:.4f}（判{"周期" if r_tr["pstd"][1]<PSTD_TH else "混沌"}）')
print('注：截面散点的纵向抖动只为分层显示，不是物理量。')
''')

# ---------------------------------------------------------------- Q4 threshold
md(r'''
## 5｜问 4 续：临界阈值怎么定（报带不报点）

**定义先写死**（三件不同的事，论文里常混着用）：

| 记号 | 含义 | 论文值 | 本号实测 |
|---|---|---|---|
| $\lambda_c$ | 无目标、无噪声时跃入大周期态的分岔点 | 0.72 / 0.73 | 见下方扫描 |
| $\lambda_0$ | 检测时预置的工作点（略低于 $\lambda_c$） | 0.72 | 0.72 |
| $C_{trig}$ | 内驱 + 待测信号相量合成后刚好越界的总幅值 | 0.73 | 见下方精扫 |

**标定流程**：固定 $k,\omega,h,T,\beta=0$、无噪声、$\gamma=0$ → 在 $[0.60,0.85]$ 以 0.005 粗扫 → 找连续三点进入周期态的位置 → 附近加密到 0.001 精扫，再用 0.0002 网格 + 1200 s 长时窗复核 → 报「跨越区间 + 中点 $\lambda_c$」。
''')

code(r'''
lams = np.round(np.arange(0.60, 0.8501, 0.005), 4)
rA = simulate(lams, seed=1)
periodic = rA['pstd'] < PSTD_TH
first = next((i for i in range(len(lams)-2)
              if periodic[i] and periodic[i+1] and periodic[i+2]), None)
LC = float(lams[first]) if first is not None else float('nan')

print('  λ      pstd      max|x|   判态   （每 0.02 打一行）')
for lam, p, a in zip(lams, rA['pstd'], rA['mx']):
    if abs(lam/0.02 - round(lam/0.02)) < 1e-6:
        print(f'  {lam:5.3f}  {p:8.1e}  {a:6.3f}   {"周期" if p < PSTD_TH else "混沌"}')
print(f'\n首个稳定进入大周期态: λ = {LC:.3f}')
if first:
    print(f'跃变前 pstd = {rA["pstd"][first-1]:.2e} → 跃变后 {rA["pstd"][first]:.2e}（数量级落差 = 悬崖）')

# 精扫 1：0.001 网格，定位跨越发生在哪个 0.001 区间
fine = np.round(np.arange(LC-0.012, LC+0.0051, 0.001), 4)
rF = simulate(fine, seed=11)
per_f = rF['pstd'] < PSTD_TH
ip = int(np.argmax(per_f))
first_p = float(fine[ip])
print('\n精扫 0.001（λ, pstd）:', [(float(a), float('%.2e' % b)) for a, b in zip(fine, rF['pstd'])])
if ip > 0:
    print(f'0.001 网格：{first_p-0.001:.3f} 仍混沌（{rF["pstd"][ip-1]:.2e}），'
          f'{first_p:.3f} 已周期 → λc ∈ ({first_p-0.001:.3f}, {first_p:.3f}]')

# 精扫 2：0.0002 网格 + 长时窗。λc 附近瞬态很长，
# T=400 会把 λ=0.7345/0.735 误判成混沌；跑到 1200 s 才收敛。
sub = np.round(np.arange(first_p-0.0014, first_p+0.00101, 0.0002), 4)
rS = simulate(sub, seed=3, t_end=1200.0, burn=600.0)
per_s = rS['pstd'] < PSTD_TH
LC_LO = float(sub[~per_s].max()) if (~per_s).any() else float('nan')
LC_HI = float(sub[per_s].min()) if per_s.any() else float('nan')
print('\n精扫 0.0002 + T=1200（λ, pstd）:', [(float(a), float('%.2e' % b)) for a, b in zip(sub, rS['pstd'])])
print(f'λc ∈ ({LC_LO:.4f}, {LC_HI:.4f}]；取 λc ≈ 0.7335')
print('  边界在 ±2e-4 内随积分实现抖动（0.7334 处两种写法分别给 3e-6 与 0.38），报带不报点。')
p_med = float(np.median(rS['pstd'][per_s]))
print(f'两态落差：混沌 {rS["pstd"][~per_s].min():.1e}–{rS["pstd"][~per_s].max():.1e}，'
      f'周期中位 {p_med:.1e}（约 5 个量级；个别点 {rS["pstd"][per_s].max():.1e} 仍在门限 {PSTD_TH} 之下）')
print('注：旧版按步数取整采截面时，周期态 pstd 被抬到 1e-2 量级，两态只差一位数；'
      '且「0.734–0.736 内混沌/周期交替」其实是 T=400 时瞬态未收敛造成的假象。')

fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), constrained_layout=True)
axes[0].plot(lams, rA['pstd'], 'o-', ms=3, color='#2f6fd0')
axes[0].axhline(PSTD_TH, color='#c0392b', ls='--', label=f'判态阈值 {PSTD_TH}')
axes[0].axvline(0.72, color='#b7791f', ls=':', lw=2, label='论文 λ=0.72')
axes[0].axvline(LC, color='#1a8a5a', lw=1.6, label=f'实测 {LC:.3f}')
axes[0].set_yscale('log'); axes[0].set_xlabel('内驱幅值 λ'); axes[0].set_ylabel('pstd (log)')
axes[0].set_title('λ 粗扫：悬崖在哪里'); axes[0].legend(fontsize=8)
axes[1].plot(fine, rF['pstd'], 's-', ms=4, color='#1a8a5a', label='T=400，网格 0.001')
axes[1].plot(sub, rS['pstd'], 'o', ms=6, mfc='none', mew=1.4, color='#c0392b', label='T=1200，网格 0.0002')
axes[1].axhline(PSTD_TH, color='#c0392b', ls='--', lw=1.0)
axes[1].axvspan(first_p-0.001, first_p, color='#e7f6ee',
                label=f'跨越区间 {first_p-0.001:.3f}–{first_p:.3f}')
axes[1].set_yscale('log')
axes[1].set_xlabel('内驱幅值 λ'); axes[1].set_ylabel('pstd (log)')
axes[1].set_title(f'精扫：单次跨越，λc≈{(LC_LO+LC_HI)/2:.4f}'); axes[1].legend(fontsize=8)
savefig('阶段2-临界阈值扫描.png')
plt.show()
''')

# ---------------------------------------------------------------- Q3 step sensitivity
md(r'''
## 6｜问 3 续：步长敏感性实测

四种步长在同一窗口 $[0.720,0.760]$ 重扫，比较跃变区间。若四条曲线的跃变都落在 §5 的跨越区间内，就说明阈值不是数值假象；跃变区间随 $h$ 最多平移 0.01，这正是「报区间不报点」的理由。
''')

code(r'''
hs = [0.1, 0.05, 0.02, 0.01]
probe = np.round(np.arange(0.720, 0.7601, 0.005), 4)
step_results, step_jumps = {}, []
for h in hs:
    rr = simulate(probe, h=h, t_end=T, burn=BURN, seed=5)
    step_results[h] = rr['pstd']
    per = rr['pstd'] < PSTD_TH
    jump = next((f'{probe[i]:.3f}→{probe[i+1]:.3f}' for i in range(len(probe)-1)
                 if not per[i] and per[i+1]), '窗口内未跃变')
    step_jumps.append(jump)
    print(f'h={h:5.3f} s  首个跃变区间 {jump}   pstd(0.730)={rr["pstd"][2]:.4f}  pstd(0.740)={rr["pstd"][4]:.4f}')

fig, ax = plt.subplots(figsize=(7.6, 4.0), constrained_layout=True)
for h in hs:
    ax.plot(probe, step_results[h], 'o-', ms=3.5, label=f'h={h}')
ax.axhline(PSTD_TH, color='#c0392b', ls='--', lw=1.2)
ax.axvspan(first_p-0.001, first_p, color='#e7f6ee', alpha=0.8,
           label=f'§5 跨越区间 {first_p-0.001:.3f}–{first_p:.3f}')
ax.set_yscale('log'); ax.set_xlabel('内驱幅值 λ'); ax.set_ylabel('pstd (log)')
ax.set_title('步长敏感性：四种 h 的跃变位置'); ax.legend(fontsize=8)
savefig('阶段2-步长敏感性.png')
plt.show()

print('判读：λ_c 不是数值假象，但也不是可照抄的普适常数——论文 0.72 与本号差异来自')
print('      求解器(Ode4/RK4)、步长(0.1/0.02)、判据(目测/Poincaré)、噪声条件的整体不同。')
''')

# ---------------------------------------------------------------- Q5
md(r'''
## 7｜问 5：变尺度为什么能解决极低频

振子的内驱被钉死在 $\omega=1$ rad/s，而真实待测信号频率未知且可能低到 0.1 Hz。变尺度**不改振子，改信号的时间标签**：原始采样步长 $1/f_s$ 改读成 $R/f_s$，第 $n$ 个样本被读成时刻 $nR/f_s$，于是角频率 $\nu$ 被压缩成 $\nu/R$。取 $R=\nu$ 就把信号折到 1 rad/s，与内驱同频。

副产品是「阈值只标一次」；但**重采样同样作用于噪声**，等效采样率降为 $f_s/R$，宽带噪声会混叠（阶段一 Q8）。

**必须记住的反例**：本机复现发现，扫 $R$ 时不止真值 $R=\omega$ 会出现低 $p_{std}$——锁频/准周期态也会。内驱幅值远大于信号，响应相干指数 $\rho\approx1$ 分不开真假。**所以「哪个 $R$ 相变，$\nu$ 就是 $R$」这个推断不能靠混沌/周期二分类自证**；这一条留给阶段 3/4 用能看见待测信号分量的统计量去解决。
''')

code(r'''
omega_true, gam_sig, th_sig = 10.0, 0.02, np.deg2rad(-120.0)
Rs = np.array([6, 7, 8, 9, 9.5, 10, 10.5, 11, 12, 13, 14], float)
betas3 = np.deg2rad([0.0, -120.0, 120.0])

def scan_R(lam_work, seed=21):
    vals, cohs = [], []
    for R in Rs:
        rr = simulate(np.full(3, lam_work), betas3, np.full(3, gam_sig),
                      np.full(3, th_sig), nu=omega_true/R,
                      t_end=T_FAST, burn=BURN_FAST, seed=seed, trace=True)
        vals.append(rr['pstd'].min())
        j = int(np.argmin(rr['pstd']))
        xs, tt = rr['X'][:, j], rr['t']
        cohs.append(np.sqrt((xs @ np.cos(tt))**2 + (xs @ np.sin(tt))**2) /
                    (np.linalg.norm(xs) * math.sqrt(tt.size/2)))
    return np.array(vals), np.array(cohs)

minp072, coh072 = scan_R(0.72)
minp073, coh073 = scan_R(0.73)
print('   R      ν=ω/R   min pstd@0.72  @0.73   ρ@0.73   低pstd?')
for R, a, b, c in zip(Rs, minp072, minp073, coh073):
    print(f'{R:6.1f}  {omega_true/R:7.3f}     {a:8.4f}   {b:8.4f}  {c:6.3f}    '
          f'{"是" if b < PSTD_TH else "否"}')
spurious = [float(R) for R, p in zip(Rs, minp073)
            if abs(R-omega_true) > 1e-9 and p < PSTD_TH]
print(f'\n非真值 R 也出现低 pstd 的锁频伪相变: {spurious}')
print(f'相干指数 ρ 范围 {min(coh073):.3f}–{max(coh073):.3f}，分不开 R=10 与伪相变。')

fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), constrained_layout=True)
axes[0].plot(Rs, minp072, 'o-', color='#8a94a0', label='λ=0.72')
axes[0].plot(Rs, minp073, 's-', color='#2f6fd0', label='λ=0.73')
axes[0].axhline(PSTD_TH, color='#c0392b', ls='--', lw=1.2, label='判态阈值')
axes[0].axvline(omega_true, color='#1a8a5a', ls=':', lw=2, label='真值 R=10')
axes[0].set_yscale('log'); axes[0].set_xlabel('尺度系数 R'); axes[0].set_ylabel('min pstd (log)')
axes[0].set_title('扫 R：不止 R=10 会「收敛」'); axes[0].legend(fontsize=8)
axes[1].plot(Rs, coh072, 'o-', color='#8a94a0', label='λ=0.72')
axes[1].plot(Rs, coh073, 's-', color='#1a8a5a', label='λ=0.73')
axes[1].axvline(omega_true, color='#1a8a5a', ls=':', lw=2)
axes[1].set_ylim(0, 1.05); axes[1].set_xlabel('尺度系数 R'); axes[1].set_ylabel('相干指数 ρ')
axes[1].set_title('内驱分量占比：几乎所有 R 都 ≈1'); axes[1].legend(fontsize=8)
savefig('阶段2-变尺度扫描.png')
plt.show()

print('提醒：R=10 时等效采样率降到 1/10，对 0–10 Hz 噪声必然混叠；论文未讨论这一点【推断】。')
''')

# ---------------------------------------------------------------- 汇总
md(r'''
## 8｜五问速答 + 与阶段一问题清单的衔接

| 问 | 一句话答案 | 证据 |
|---|---|---|
| Q1 方程各项 | 双阱势 $V=-x^4/4+x^6/6$；阻尼耗散、五次恢复力成陡壁、内驱提供相干参考、待测信号只负责「临门一脚」 | §1 势函数图 |
| Q2 参数怎么取 | $k=0.5,\ \omega=1,\ \lambda_0=0.72,\ h=0.02,\ T=400$（丢 200），初值 0，判据 pstd<0.08 | §2 参数表 |
| Q3 步长 | 「不发散」不是判据；四种 $h$ 的跃变落在同一带内，取 $h=0.02$，跃变区间随 $h$ 最多平移 0.01 | §6 步长敏感性 |
| Q4 相变过程 | 小 λ 在两阱乱跳、Poincaré 点散开；越界后整圈绕行、点收敛；pstd 断崖式下跌（5 个量级），无「带内交替」——旧观测到的交替是瞬态未收敛的假象 | §4–§5 |
| Q5 变尺度 | 只改时间标签，$R=\nu$ 把未知频率压到 1 rad/s；阈值只标一次，但噪声混叠 + 锁频伪相变是其软肋 | §7 |

**阶段一问题清单销账**：Q6（$\lambda_c$ 换参数漂多少）→ 本号给出步长维度的 ±0.005 抖动与阈值带，恢复力/噪声维度留阶段 3；Q8（变尺度是否缩放噪声带宽）→ 本号指出混叠风险并留待阶段 3 做 PSD 对照；Q10（频段取哪一段）→ **仍待导师**，本号不给假答案。
''')

code(r'''
# 落盘：阶段 2 的可引用数字与条件
summary = {
    'stage': 2,
    'equation': "x'' + 0.5x' - x^3 + x^5 = lambda*cos(t+beta) + gamma*cos(t+theta) + n(t)",
    'potential': 'V(x) = -x^4/4 + x^6/6; wells x=+-1 (V=-1/12), barrier x=0 (V=0)',
    'parameters': {'k': K, 'omega_rad_s': OMEGA, 'lambda0': LAM0, 'lambda_paper': 0.72,
                   'C_trig_paper': C_TRIG, 'h_s': H, 'T_s': T, 'burn_s': BURN,
                   'init': [0, 0], 'pstd_threshold': PSTD_TH,
                   'verdict': 'pstd < 0.08 -> large-scale periodic (signal present)'},
    'phase_portraits': {'lambda_chaos': 0.70, 'lambda_periodic': 0.76,
                        'pstd': [round(float(v), 4) for v in r_tr['pstd']]},
    'threshold': {'coarse_grid': [0.60, 0.85, 0.005], 'first_stable_lambda': LC,
                  'crossing_band_0p001': [first_p-0.001, first_p],
                  'lc_fine_0p0002_T1200': [LC_LO, LC_HI],
                  'lc_estimate': round((LC_LO+LC_HI)/2, 5),
                  'poincare_sampling': 'exact linear interpolation at tau=2*pi*k',
                  'note': 'report a band, not a point; conditions: clean, h=0.02, T=1200 for the fine scan; '
                          'the old 0.734-0.736 alternation was a T=400 transient artifact'},
    'step_sensitivity': {str(h): {'first_jump': jump} for h, jump in zip(hs, step_jumps)},
    'rescale': {'true_omega_rad_s': omega_true, 'R_grid': list(map(float, Rs)),
                'spurious_low_pstd_R': spurious,
                'coherence_range': [round(float(min(coh073)), 3), round(float(max(coh073)), 3)],
                'verdict': 'phase transition alone cannot identify frequency; noise aliasing risk'},
    'stage1_questions_answered': ['Q6 (step-size dimension only)', 'Q8 (aliasing risk flagged)'],
    'stage1_questions_pending': ['Q10 (frequency band — mentor decision)'],
}
path = OUT / '阶段2-建模与阈值汇总.json'
path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'已写入：{path}')
print(f'阶段 2 notebook 全部完成，耗时 {time.time() - _START:.1f} s')
''')

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {
    'kernelspec': {'display_name': '.venv (3.13.9)', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.13.9'},
}
nbf.write(nb, TARGET)
print(f'WROTE {TARGET} ({len(cells)} cells)')
