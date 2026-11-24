"""★[2] 复现的全部数值实验，结果落 _data/*.json 供 notebook 读。

覆盖：
  sweep   扫 λ 的 Poincaré 判据（真实临界，精确 2π 采样）
  fig1    混沌态 / 大周期态的相图与时域
  fig2    信号初相对检测的影响
  fig5    四幅消融：无信号 / R=1,β=-120 / R=10,β=0 / R=10,β=-120
  fig7_9  SNR=-23/-21 dB 下三路阵列的相图
  fig10_11 输出项 x、y 的时域波形
  table3  参数估计：三路扫 λ 得 λ1/λ2/λ3 -> Eq.17 -> γ,θ 与误差
  nu      ν 与 R 的关系、盲测扫 R、频率分辨率
  snr     SNR 门限：已知频率单点 vs 未知频率扫 R
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s2sim as S  # noqa: E402

DATA = HERE.parent / "_data"
DATA.mkdir(exist_ok=True)
TWOPI = S.TWOPI
LAM_WORK = 0.72
GAM, THETA = 0.02, math.radians(-120)
BETAS = np.deg2rad([0.0, -120.0, 120.0])
DT = 0.005          # 数据驱动模式的采样间隔
FS = 1 / DT


def dump(name, obj):
    p = DATA / f"{name}.json"
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  -> {p.name}  {p.stat().st_size/1024:.0f} KB")


def curve(r, step=20):
    """把轨迹抽稀成可落盘的曲线。"""
    t = r["t"] * 1e3            # ms
    return dict(t=np.round(t[::step], 3).tolist(),
                x=np.round(r["X"][::step], 4).tolist(),
                y=np.round(r["Y"][::step], 4).tolist())


# ---------------------------------------------------------------- sweep
def exp_sweep():
    print("[sweep] 扫 λ")
    lam = np.round(np.arange(0.700, 0.7601, 0.002), 4)
    r = S.simulate(lam, gamma=0.0, t_end=600.0, burn=300.0)
    pstd = r["pstd"]
    per = pstd < 0.02
    jump = next(([float(lam[i]), float(lam[i + 1])] for i in range(len(lam) - 1)
                 if not per[i] and per[i + 1]), None)
    dump("sweep", dict(lam=lam.tolist(), pstd=np.round(pstd, 6).tolist(),
                       jump=jump, lam_c=float((jump[0] + jump[1]) / 2) if jump else None,
                       paper=0.72))
    print(f"   临界带 {jump} -> lam_c ≈ {(jump[0]+jump[1])/2:.4f}")


# ---------------------------------------------------------------- fig1
def exp_fig1():
    print("[fig1] 混沌 / 大周期相图")
    r0 = S.simulate(LAM_WORK, gamma=0.0, trace=True, t_end=400.0, burn=200.0)
    r1 = S.simulate(0.78, gamma=0.0, trace=True, t_end=400.0, burn=200.0)
    dump("fig1", dict(chaos=curve(r0, 4), periodic=curve(r1, 4),
                      pstd=[float(r0["pstd"][0]), float(r1["pstd"][0])]))


# ---------------------------------------------------------------- fig2
def exp_fig2():
    print("[fig2] 初相影响")
    ths = np.deg2rad([0, 45, 90, 135, 180])
    out = {}
    for th in ths:
        r = S.simulate(LAM_WORK, 0.0, GAM, th, t_end=400.0, burn=200.0, trace=True)
        out[f"{int(np.rad2deg(th))}"] = dict(pstd=float(r["pstd"][0]),
                                            cur=curve(r, 6))
    dump("fig2", out)


# ---------------------------------------------------------------- fig5
def exp_fig5():
    print("[fig5] 四幅消融")
    cases = {
        "no_signal": dict(lam=LAM_WORK, beta=0.0, gamma=0.0, theta=0.0, nu=1.0),
        "R1_b-120": dict(lam=LAM_WORK, beta=np.deg2rad(-120), gamma=GAM,
                         theta=THETA, nu=10.0),
        "R10_b0": dict(lam=LAM_WORK, beta=0.0, gamma=GAM, theta=THETA, nu=1.0),
        "R10_b-120": dict(lam=LAM_WORK, beta=np.deg2rad(-120), gamma=GAM,
                          theta=THETA, nu=1.0),
    }
    out = {}
    for name, kw in cases.items():
        r = S.simulate(trace=True, t_end=400.0, burn=200.0, **kw)
        out[name] = dict(pstd=float(r["pstd"][0]), cur=curve(r, 6))
        print(f"   {name:12s} pstd={r['pstd'][0]:.4f} "
              f"{'周期' if r['pstd'][0] < 0.02 else '混沌'}")
    dump("fig5", out)


# ---------------------------------------------------------------- fig7_9
def exp_fig7_9():
    print("[fig7_9] 门限附近的相图")
    out = {}
    for snr in (-23, -21):
        D = 0.0002 / 10 ** (snr / 10)
        sig = math.sqrt(D)
        r = S.simulate(LAM_WORK, BETAS, GAM, np.full(3, THETA),
                       sigma=sig, seed=100 + snr, t_end=1200.0, burn=600.0,
                       trace=True)
        out[str(snr)] = dict(pstd=np.round(r["pstd"], 5).tolist(),
                             sigma=sig, cur=[curve_chan(r, j, 8) for j in range(3)])
        print(f"   SNR={snr} dB  pstd={np.round(r['pstd'],4)}")
    dump("fig7_9", out)


def curve_chan(r, j, step=8):
    n = r["t"].size
    t = r["t"][:n][::step] * 1e3
    return dict(t=np.round(t, 2).tolist(),
                x=np.round(r["X"][::step, j], 4).tolist(),
                y=np.round(r["Y"][::step, j], 4).tolist())


# ---------------------------------------------------------------- fig10_11
def exp_fig10_11():
    print("[fig10_11] 输出项时域")
    out = {}
    for snr in (-23, -21):
        D = 0.0002 / 10 ** (snr / 10)
        r = S.simulate(LAM_WORK, np.deg2rad(-120), GAM, THETA,
                       sigma=math.sqrt(D), seed=7, t_end=1200.0, burn=600.0,
                       trace=True)
        st = 10
        out[str(snr)] = dict(t=np.round(r["t"][::st] * 1e3, 2).tolist(),
                             x=np.round(r["X"][::st, 0], 4).tolist(),
                             y=np.round(r["Y"][::st, 0], 4).tolist())
    dump("fig10_11", out)


# ---------------------------------------------------------------- table3
def crit_lambda(beta, theta, gamma, lam_lo=0.715, lam_hi=0.760, n=46,
                t_end=1200.0, burn=600.0):
    """扫内驱幅值，找刚好相变的那一点（二分细化）。"""
    lams = np.round(np.linspace(lam_lo, lam_hi, n), 4)
    r = S.simulate(lams, beta, gamma, theta, t_end=t_end, burn=burn)
    per = r["pstd"] < 0.02
    if not per.any():
        return None, lams, r["pstd"]
    i = int(np.argmax(per))
    if i == 0:
        return float(lams[0]), lams, r["pstd"]
    lo, hi = lams[i - 1], lams[i]
    for _ in range(6):
        mid = round((lo + hi) / 2, 6)
        v = S.simulate(mid, beta, gamma, theta, t_end=t_end, burn=burn)["pstd"][0]
        if v < 0.02:
            hi = mid
        else:
            lo = mid
    return float(hi), lams, r["pstd"]


def exp_table3():
    print("[table3] 参数估计（三路扫 λ）")
    betas3 = [0.0, math.pi, math.pi / 2]
    res = {}
    lam_est = []
    for b in betas3:
        lc, lams, pstd = crit_lambda(b, THETA, GAM)
        lam_est.append(lc)
        res[f"{math.degrees(b):.0f}"] = dict(lam_c=lc,
                                             lams=lams.tolist(),
                                             pstd=np.round(pstd, 5).tolist())
        print(f"   beta={math.degrees(b):5.0f} deg -> lambda_c={lc}")
    l1, l2, l3 = lam_est
    # 三路联立，不依赖 C（论文 Eq.17 的等价形式）：
    #   λ1²+2λ1γcosθ+γ² = λ2²-2λ2γcosθ+γ² = λ3²+2λ3γsinθ+γ²
    #   => cosθ=(λ2-λ1)/(2γ),  γ²=[(λ2-λ1)² + (λ1λ2-λ3²)²/λ3²]/4
    g2 = ((l2 - l1) ** 2 + (l1 * l2 - l3 ** 2) ** 2 / l3 ** 2) / 4
    g_hat = math.sqrt(g2) if g2 > 0 else float("nan")
    cos_t = (l2 - l1) / (2 * g_hat) if g_hat == g_hat else float("nan")
    sin_t = (l1 * l2 - l3 ** 2) / (2 * l3 * g_hat) if g_hat == g_hat else float("nan")
    th_hat = math.degrees(math.atan2(sin_t, cos_t)) if g_hat == g_hat else float("nan")
    # 三路各自的等效 C_trig：把实测 λi 代回，看它们是否一致
    C_eff = {}
    for b, lc in zip((0, 180, 90), lam_est):
        cc = math.cos(THETA - math.radians(b))
        C_eff[str(b)] = math.sqrt(max(lc ** 2 + 2 * lc * GAM * cc + GAM ** 2, 0))
    out = dict(lam=[l1, l2, l3], paper_lam=[0.7414, 0.7183, 0.7447],
               gamma=GAM, gamma_hat=g_hat, gamma_err=(g_hat - GAM) / GAM * 100,
               theta_true=math.degrees(THETA), theta_hat=th_hat,
               cos_hat=cos_t, sin_hat=sin_t,
               C_eff=C_eff, C_spread=max(C_eff.values()) - min(C_eff.values()),
               per_route=res)
    dump("table3", out)
    print(f"   lambda = {[round(v,5) for v in lam_est]}")
    print(f"   gamma_hat={g_hat:.5f} (真值 {GAM}, {out['gamma_err']:+.1f}%)   "
          f"theta_hat={th_hat:.1f}deg (真值 {math.degrees(THETA):.1f})")
    print(f"   三路等效 C_trig: {[(k, round(v,4)) for k,v in C_eff.items()]}  "
          f"极差 {out['C_spread']:.4f}")


# ---------------------------------------------------------------- nu
def exp_nu():
    print("[nu] ν / R / 盲测")
    dt = DT
    # 1) 固定 R=10，扫 nu，看分支连续性与选择带宽
    nus = np.round(np.arange(0.96, 1.041, 0.005), 4)
    r = S.simulate(LAM_WORK, np.deg2rad(-120), GAM, THETA, nu=nus,
                   t_end=600.0, burn=300.0)
    # 2) 盲测：只给采样序列
    ks = np.arange(900000)
    sig = GAM * np.cos(10.0 * ks * dt + THETA)
    Rs = np.round(np.arange(8.0, 12.01, 0.05), 3)
    blind = [S.simulate_seq(sig, R, dt, lam=LAM_WORK, beta=np.deg2rad(-120),
                            n_cycles=150)["pstd"] for R in Rs]
    # 3) 三个真值，看定位
    loc = {}
    for w in (10.0, 7.3, 3.7):
        s2 = GAM * np.cos(w * ks * dt + THETA)
        grid = np.round(np.arange(0.85 * w, 1.15 * w, 0.05), 3)
        v = [S.simulate_seq(s2, R, dt, lam=LAM_WORK, beta=np.deg2rad(-120),
                            n_cycles=150)["pstd"] for R in grid]
        v = np.array(v)
        i = int(np.nanargmin(v))
        loc[str(w)] = dict(grid=grid.tolist(), pstd=np.round(v, 5).tolist(),
                           best=float(grid[i]), ok=bool(v[i] < 0.02))
    dump("nu", dict(nu_list=nus.tolist(), nu_pstd=np.round(r["pstd"], 5).tolist(),
                    Rs=Rs.tolist(), blind=np.round(blind, 5).tolist(),
                    locate=loc))
    print(f"   盲测命中 {[round(R,2) for R,p in zip(Rs,blind) if p<0.02]}")
    print(f"   三真值定位 {[(k, v['best']) for k, v in loc.items()]}")


# ---------------------------------------------------------------- snr
def exp_snr():
    print("[snr] 门限：单点 vs 扫 R（相对判据：pstd < 0.5×无信号基线）")
    dt = DT
    ks = np.arange(700000)
    snrs = [-30, -27, -25, -23, -21, -18, -15]
    rng = np.random.default_rng(2024)
    single, scan = {}, {}
    for snr in snrs:
        D = 0.0002 / 10 ** (snr / 10)
        sg = math.sqrt(D)
        # 基线：同强度纯噪声
        base = np.median([S.simulate_seq(sg * rng.standard_normal(ks.size),
                                         10.0, dt, lam=LAM_WORK,
                                         beta=np.deg2rad(-120), n_cycles=150)["pstd"]
                          for _ in range(4)])
        thr = 0.5 * base
        hit1 = hit2 = corr = 0
        NREP = 4
        for _ in range(NREP):
            sig = GAM * np.cos(10.0 * ks * dt + THETA) + sg * rng.standard_normal(ks.size)
            v1 = S.simulate_seq(sig, 10.0, dt, lam=LAM_WORK,
                                beta=np.deg2rad(-120), n_cycles=150)["pstd"]
            hit1 += (v1 < thr)
            grid = np.round(np.arange(9.4, 10.61, 0.1), 2)
            vs = [S.simulate_seq(sig, R, dt, lam=LAM_WORK,
                                 beta=np.deg2rad(-120), n_cycles=150)["pstd"]
                  for R in grid]
            vs = np.array(vs)
            good = grid[vs < thr]
            if good.size:
                hit2 += 1
                corr += bool(np.any(np.abs(good - 10.0) < 0.25))
        single[str(snr)] = dict(pd=hit1 / NREP, thr=thr, base=base, sigma=sg)
        scan[str(snr)] = dict(hit=hit2 / NREP, correct=corr / NREP,
                              thr=thr, sigma=sg)
        print(f"   SNR={snr:3d} 基线={base:.3f} 门限={thr:.3f}  "
              f"单点 Pd={hit1/NREP*100:3.0f}%  扫R正确={corr/NREP*100:3.0f}%")
    dump("snr", dict(single=single, scan=scan))


TARGETS = dict(sweep=exp_sweep, fig1=exp_fig1, fig2=exp_fig2, fig5=exp_fig5,
               fig7_9=exp_fig7_9, fig10_11=exp_fig10_11, table3=exp_table3,
               nu=exp_nu, snr=exp_snr)

if __name__ == "__main__":
    want = sys.argv[1:] or list(TARGETS)
    t0 = time.time()
    for name in want:
        if name not in TARGETS:
            print(f"未知目标 {name}，可选 {list(TARGETS)}")
            continue
        t = time.time()
        TARGETS[name]()
        print(f"   ({time.time()-t:.1f} s)")
    print(f"总计 {time.time()-t0:.1f} s")
