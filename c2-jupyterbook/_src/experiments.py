"""C2 全量复现：图 1–11、表 3–5 的数据生成。

跑法：.venv/Scripts/python.exe c2-jupyterbook/_src/experiments.py [目标名 ...]
产出：_data/*.json（画图用）、复现输出/c2-figs/*.png（成品图）

口径统一说明
------------
1. 输出量取位移差 x1−x2（原文图 8–10 纵轴所标），速度差 ẋ1−ẋ2 只作对照。
2. 输出限带是原文未写的一环，本书从表 3 的 r 列与 Pd 列联合反演得 fc≈130 kHz
   （高斯导数脉冲，上升时间 1.5 μs）。其余波形按 fc ∝ 1/上升时间 缩放。
3. 输出 SNR 与式 (7) 同口径：10log10(a²/4D)，即 20log10(峰值/σ) − 6.02 dB。
   于是"5σ"严格等于 7.96 dB，与原文 5.1 节一致。
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
BOOK = HERE.parent
ROOT = BOOK.parent
sys.path.insert(0, str(HERE))
import c2sim as S

DATA = BOOK / "_data"
DATA.mkdir(exist_ok=True)
FIGS = ROOT / "复现输出" / "c2-figs"
FIGS.mkdir(parents=True, exist_ok=True)

D = 0.1
FC_REF = 130e3          # 反演得到的参考截止频率（t_rise = 1.5 μs 时）
TRISE_REF = 1.5e-6


def fc_for(t_rise):
    """限带截止频率：按上升时间反比缩放（以高斯导数为标定基准）。"""
    return FC_REF * TRISE_REF / t_rise


def dump(name, obj):
    p = DATA / f"{name}.json"
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  → {p.name}", flush=True)


def amp_of(snr):
    return float(np.sqrt(4 * D * 10 ** (snr / 10)))


# ------------------------------------------------------------------ 图 1
def exp_fig1():
    """图 1(a)–(f)：三角脉冲 + −10 dB 噪声下的检测过程。"""
    om, h, T = 5e6, 1e-9, 20e-6
    t = np.arange(int(round(T / h)) + 1) * h
    A, snr = 0.63, -10
    sig = np.sqrt(4 * D * 10 ** (snr / 10))          # 由式 (7) 反推的噪声标准差
    # 三角脉冲：上升沿、下降沿各 0.5 μs
    s = A * np.clip(1 - np.abs(t - 10e-6) / 0.5e-6, 0, 1)
    rng = np.random.default_rng(2019)
    n = rng.normal(0, sig, t.size)
    so, X = S.simulate(t, drive=s + n, batch=1, omega=om, return_state=True)
    x1, v1, x2, v2 = (X[0, :, i] for i in range(4))
    np.savez_compressed(
        DATA / "fig1.npz", t=t, s=s, sn=s + n, so=so[0],
        dx=x1 - x2, x1=x1, x2=x2, v1=v1, v2=v2)
    print("  图1 数据已存 fig1.npz", flush=True)


# ------------------------------------------------------------------ 图 2
def exp_fig2():
    """图 2：F∈[0,2] 分岔图 + 最大 Lyapunov 指数。"""
    Fs = np.linspace(0.02, 2.0, 160)
    xi, h, per = 0.7, 0.02, int(round(2 * np.pi / 0.02))
    n_trans, n_keep = 140, 70

    # 批量 RK4：状态 (N,2) + 变分 (N,2,2)
    x = np.zeros(Fs.size); v = np.zeros(Fs.size)
    Q = np.tile(np.eye(2), (Fs.size, 1, 1))
    out = np.empty((Fs.size, n_keep))
    lya = np.zeros(Fs.size); n_ly = 0; n_qr = 0
    tcur = 0.0; j = 0

    def f(x, v, t, F):
        return v, -xi * v + x - x ** 3 + F * np.cos(t)

    def jac(x, v, t):
        return np.array([[0.0, 1.0], [1 - 3 * x ** 2, -xi]])

    for i in range((n_trans + n_keep) * per):
        a1, b1 = f(x, v, tcur, Fs)
        a2, b2 = f(x + h / 2 * a1, v + h / 2 * b1, tcur + h / 2, Fs)
        a3, b3 = f(x + h / 2 * a2, v + h / 2 * b2, tcur + h / 2, Fs)
        a4, b4 = f(x + h * a3, v + h * b3, tcur + h, Fs)
        # 变分方程：dδ/dt = J δ
        for k in range(2):
            qx, qv = Q[:, 0, k], Q[:, 1, k]
            dqx = qv
            dqv = (1 - 3 * x ** 2) * qx - xi * qv
            Q[:, 0, k] = qx + h * dqx
            Q[:, 1, k] = qv + h * dqv
        x = x + h / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        v = v + h / 6 * (b1 + 2 * b2 + 2 * b3 + b4)
        tcur += h
        n_ly += 1
        if n_ly % 10 == 0:                            # 每 10 步做一次 QR 正交化
            norm = np.sqrt((Q ** 2).sum(axis=1))      # (N,2)
            lya += np.log(norm[:, 0] + 1e-300)
            Q = Q / norm[:, None, :]
            n_qr += 1
        if (i + 1) % per == 0:
            if i + 1 > n_trans * per:
                out[:, j] = x; j += 1
            tcur = 0.0
    lya /= max(1, n_qr * 10)                          # 每步的对数增长率
    dump("fig2", dict(F=Fs.tolist(), X=out.tolist(), lya=lya.tolist()))
    print(f"  Lyapunov>0 的点数 {int((lya > 0).sum())}/{Fs.size}", flush=True)


# ------------------------------------------------------------------ 图 3–7
def _sweep(name, key, values, kwargs_of, fc, n=100, T=40e-6):
    """通用参数扫描：每个取值跑 n 个噪声样本，返回 SNRI 与 r。"""
    rows = []
    for val in values:
        kw = kwargs_of(val)
        h = kw.pop("h", 1e-9); TT = kw.pop("T", T)
        t = np.arange(int(round(TT / h)) + 1) * h
        A = 0.632
        s = S.gauss(t, A=A, c=3e-6, b=TT / 2, T=TT)
        t0 = time.time()
        so, X = S.simulate(t, drive=s, batch=n, sigma=np.sqrt(D), hold=1,
                           seed=17 + int(abs(val) * 7) % 999, return_state=True,
                           **kw)
        quiet = S.quiet_mask(t, TT / 2, TT / 2, guard=15e-6)
        m = S.chain_metrics(so, X, s, t, h, fc=fc, quiet=quiet)
        rows.append(dict(x=float(val), snri=float(m["snr_o"]),
                         r=float(m["r_dx"]), pd=float(m["pd"]),
                         sec=round(time.time() - t0, 1)))
        print(f"  {key}={val:<10g} SNRI={m['snr_o']:6.1f} dB  r={m['r_dx']:.3f}"
              f"  Pd={m['pd']*100:3.0f}%  ({rows[-1]['sec']}s)", flush=True)
    dump(name, dict(key=key, rows=rows))
    return rows


def exp_fig3():
    """图 3：SNRI 与 r 随策动力幅值 F（相态）的变化。"""
    _sweep("fig3", "F", np.round(np.linspace(0.05, 2.0, 20), 3),
           lambda v: dict(F=float(v)), fc=FC_REF)


def exp_fig4():
    """图 4：SNRI 与 r 随驱动力周期 ω 的变化（时间窗随 ω 自适应）。"""
    omegas = [1e5, 2e5, 5e5, 1e6, 2e6, 3e6, 4e6, 5e6, 7e6, 9e6, 1e7, 2e7, 5e7]

    def kw(v):
        h = max(1e-9, 2 * np.pi / (v * 1000))
        T = max(40e-6, 32 * 2 * np.pi / v)
        return dict(omega=float(v), h=h, T=T)

    _sweep("fig4", "omega", omegas, kw, fc=FC_REF)


def exp_fig5():
    """图 5：SNRI 与 r 随耦合系数 k 的变化。"""
    _sweep("fig5", "k", [0.1, 0.3, 0.5, 0.7, 1, 2, 3, 5, 7, 10, 20, 50, 100, 300, 1000],
           lambda v: dict(k=float(v)), fc=FC_REF)


def exp_fig6():
    """图 6：SNRI 与 r 随阻尼系数 ξ 的变化。"""
    _sweep("fig6", "xi", [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 3, 10],
           lambda v: dict(xi=float(v)), fc=FC_REF)


def exp_fig7():
    """图 7：SNRI 与 r 随计算步长的变化（采样率 50–1000 MHz）。"""
    hs = [20e-9, 10e-9, 5e-9, 2e-9, 1e-9]
    _sweep("fig7", "h_ns", [h * 1e9 for h in hs],
           lambda v: dict(h=float(v) * 1e-9), fc=FC_REF)


# ------------------------------------------------------------------ 表 3/4/5
WAVES = {
    "方波": dict(omega=50.0, h=1e-4, T=3.0, t_rise=1e-4, thr=-15, guard=0.6,
               fn=lambda t, A, T: S.square(t, A=A, t0=1.0, width=1.0),
               centre=1.5),
    "双指数脉冲": dict(omega=5e5, h=1e-8, T=600e-6, t_rise=3e-6, thr=-12, guard=100e-6,
                  fn=lambda t, A, T: S.double_exp(t, A=A, t0=225e-6),
                  centre=250e-6),
    "高斯导数脉冲": dict(omega=4e6, h=1e-9, T=40e-6, t_rise=1.5e-6, thr=-16, guard=10e-6,
                   fn=lambda t, A, T: S.gauss_deriv(t, A=A, t0=20e-6, c=1.45e-6),
                   centre=20e-6),
}


def _hwhm(t, y):
    """半高宽：取峰值一半以上的最宽连续区间。"""
    y = np.abs(y)
    pk = y.max()
    if pk <= 0:
        return np.nan
    m = y >= 0.5 * pk
    idx = np.flatnonzero(m)
    if idx.size < 2:
        return np.nan
    # 找最长连续段
    splits = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    seg = max(splits, key=len)
    return float(t[seg[-1]] - t[seg[0]])


def exp_tables():
    """表 3（检测概率与相似度）、表 4（幅值）、表 5（宽度）。"""
    snrs = np.arange(-20, -9)
    t3, t4, t5 = {}, {}, {}
    for name, cfg in WAVES.items():
        h, T, om = cfg["h"], cfg["T"], cfg["omega"]
        t = np.arange(int(round(T / h)) + 1) * h
        fc = fc_for(cfg["t_rise"])
        quiet = S.quiet_mask(t, cfg["centre"], cfg["centre"], guard=cfg["guard"])
        rows3, rows4, rows5 = [], [], []
        # 训练集：门限 SNR 的 100 个样本，定标位移差峰值→幅值
        A0 = amp_of(cfg["thr"])
        s0 = cfg["fn"](t, A0, T)
        so0, X0 = S.simulate(t, drive=s0, batch=100, sigma=np.sqrt(D), hold=1,
                             seed=555, omega=om, return_state=True)
        dx0 = S.lowpass(X0[:, :, 0] - X0[:, :, 2], 1 / h, fc)
        K = float(np.median(np.abs(dx0).max(axis=1)) / A0)
        print(f"  {name}: fc={fc/1e3:.0f} kHz  标定系数 K={K:.4f}", flush=True)

        for snr in snrs:
            A = amp_of(snr)
            s = cfg["fn"](t, A, T)
            t0 = time.time()
            so, X = S.simulate(t, drive=s, batch=100, sigma=np.sqrt(D), hold=1,
                               seed=100 - int(snr), omega=om, return_state=True)
            m = S.chain_metrics(so, X, s, t, h, fc=fc, quiet=quiet)
            dx = S.lowpass(X[:, :, 0] - X[:, :, 2], 1 / h, fc)
            est = np.abs(dx).max(axis=1) / K
            wid = np.array([_hwhm(t, dx[i]) for i in range(dx.shape[0])])
            rows3.append(dict(snr=float(snr), a=float(A), pd=float(m["pd"]),
                              r=float(m["r_dx"]), ratio=float(m["ratio"]),
                              snri=float(m["snr_o"] - snr)))
            rows4.append(dict(snr=float(snr), true=float(A),
                              est=float(np.median(est)),
                              err=float((np.median(est) - A) / A * 100),
                              seed_std=float(np.std(est))))
            true_w = {"方波": 1.0, "双指数脉冲": 25e-6, "高斯导数脉冲": 1.45e-6}[name]
            rows5.append(dict(snr=float(snr), true=true_w,
                              est=float(np.nanmedian(wid)),
                              err=float((np.nanmedian(wid) - true_w) / true_w * 100)))
            print(f"  {name:7s} SNR={snr:3d}  Pd={m['pd']*100:5.1f}%  r={m['r_dx']:.3f}"
                  f"  a^={np.median(est):.4f}/{A:.4f}  w^={np.nanmedian(wid)*1e6:.2f}us"
                  f"  ({round(time.time()-t0,1)}s)", flush=True)
        t3[name] = rows3; t4[name] = rows4; t5[name] = rows5
    dump("table3", t3); dump("table4", t4); dump("table5", t5)


# ------------------------------------------------------------------ 图 8–11
def exp_fig8_10():
    """图 8/9/10：三种波形在各自门限 SNR 处的输入与恢复波形。"""
    arr, meta = {}, {}
    for name, cfg in WAVES.items():
        h, T, om = cfg["h"], cfg["T"], cfg["omega"]
        t = np.arange(int(round(T / h)) + 1) * h
        fc = fc_for(cfg["t_rise"])
        A = amp_of(cfg["thr"])
        s = cfg["fn"](t, A, T)
        so, X = S.simulate(t, drive=s, batch=1, sigma=np.sqrt(D), hold=1,
                           seed=77, omega=om, return_state=True)
        dx = S.lowpass(X[0, :, 0] - X[0, :, 2], 1 / h, fc)
        arr[f"{name}_t"] = t
        arr[f"{name}_s"] = s
        arr[f"{name}_dx"] = dx
        m = S.chain_metrics(so, X, s, t, h, fc=fc,
                            quiet=S.quiet_mask(t, cfg["centre"], cfg["centre"],
                                               guard=cfg["guard"]))
        meta[name] = dict(thr=cfg["thr"], fc=fc, r=float(m["r_dx"]))
    np.savez_compressed(DATA / "fig810.npz", **arr)
    dump("fig810_meta", meta)


def exp_fig11():
    """图 11：弱耦合 vs 强耦合对高斯导数脉冲的恢复对比（−16 dB）。"""
    om, h, T = 4e6, 1e-9, 40e-6
    t = np.arange(int(round(T / h)) + 1) * h
    A = amp_of(-16)
    s = S.gauss_deriv(t, A=A, t0=T / 2, c=1.45e-6)
    arr = dict(t=t, s=s)
    for tag, weak in (("weak", True), ("strong", False)):
        so, X = S.simulate(t, drive=s, batch=1, sigma=np.sqrt(D), hold=1,
                           seed=99, omega=om, weak=weak, return_state=True)
        arr[tag] = S.lowpass(X[0, :, 0] - X[0, :, 2], 1 / h, 130e3)
    np.savez_compressed(DATA / "fig11.npz", **arr)
    print("  → fig11.npz", flush=True)


# ------------------------------------------------------------------ 限带反演
def exp_chain():
    """限带截止频率 fc 的敏感性：用表 3 的 r 列与 Pd 列联合反演 fc。"""
    om, h, T = 4e6, 1e-9, 40e-6
    t = np.arange(int(round(T / h)) + 1) * h
    fcs = [60e3, 80e3, 100e3, 120e3, 130e3, 140e3, 160e3, 200e3, 300e3,
           500e3, 1e6, 2e6]
    out = {}
    for snr in (-20, -16, -12):
        A = amp_of(snr)
        s = S.gauss_deriv(t, A=A, t0=T / 2, c=1.45e-6)
        so, X = S.simulate(t, drive=s, batch=100, sigma=np.sqrt(D), hold=1,
                           seed=100 - snr, omega=om, return_state=True)
        quiet = S.quiet_mask(t, T / 2, T / 2, guard=10e-6)
        rows = []
        for fc in fcs:
            m = S.chain_metrics(so, X, s, t, h, fc=fc, quiet=quiet)
            rows.append(dict(fc=float(fc), r=float(m["r_dx"]),
                             pd=float(m["pd"]), ratio=float(m["ratio"]),
                             snri=float(m["snr_o"] - snr)))
        out[str(snr)] = rows
        print(f"  {snr} dB 完成", flush=True)
    dump("chain", out)


TARGETS = {
    "fig1": exp_fig1, "fig2": exp_fig2, "fig3": exp_fig3, "fig4": exp_fig4,
    "fig5": exp_fig5, "fig6": exp_fig6, "fig7": exp_fig7,
    "tables": exp_tables, "fig810": exp_fig8_10, "fig11": exp_fig11,
    "chain": exp_chain,
}

if __name__ == "__main__":
    want = sys.argv[1:] or list(TARGETS)
    t_start = time.time()
    for k in want:
        print(f"=== {k} ===", flush=True)
        TARGETS[k]()
    print(f"\n全部完成，用时 {(time.time()-t_start)/60:.1f} min")
