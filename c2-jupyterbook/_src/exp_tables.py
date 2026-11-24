"""复现论文表 3（检测概率）与表 4（幅值估计）。

在 .venv 下运行：.venv/Scripts/python.exe c2-jupyterbook/_src/exp_tables.py
产出：c2-jupyterbook/_data/table3.json、table4.json、*.npz
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c2sim as S

DATA = HERE.parent / "_data"
DATA.mkdir(exist_ok=True)

D_NOISE = 0.1                      # 论文背景噪声方差
SNRS = np.arange(-20, -9)          # -20 … -10 dB
N_TRIAL = 100

# 三种波形：ω、步长、时长、波形函数、噪声带宽匹配的 hold
WAVES = {
    "方波": dict(omega=50.0, h=1e-4, T=6.0, hold=1,
                 fn=lambda t, A, T: S.square(t, A=A, t0=T / 2 - 0.5, width=1.0),
                 quiet=lambda t, T: t < T / 2 - 0.55,
                 thr=-15),
    "双指数脉冲": dict(omega=5e5, h=1e-8, T=600e-6, hold=300,
                  fn=lambda t, A, T: S.double_exp(t, A=A, t0=T / 2 - 75e-6),
                  quiet=lambda t, T: t < T / 2 - 100e-6,
                  thr=-12),
    "高斯导数脉冲": dict(omega=4e6, h=1e-9, T=40e-6, hold=1500,
                  fn=lambda t, A, T: S.gauss_deriv(t, A=A, t0=T / 2),
                  quiet=lambda t, T: t < T / 2 - 5e-6,
                  thr=-16),
}


def amp_of(snr_db, D=D_NOISE):
    """式 (7) 反推脉冲幅值。"""
    return float(np.sqrt(4 * D * 10 ** (snr_db / 10)))


def run_wave(name, cfg):
    h, T, w = cfg["h"], cfg["T"], cfg["omega"]
    t = np.arange(int(round(T / h)) + 1) * h
    quiet = cfg["quiet"](t, T)
    rows = []
    for snr in SNRS:
        A = amp_of(snr)
        t0 = time.time()
        so = S.simulate(t, drive=lambda tt: cfg["fn"](tt, A, T), batch=N_TRIAL,
                        sigma=np.sqrt(D_NOISE), hold=cfg["hold"], seed=100 + int(snr),
                        omega=w, k=10.0, xi=0.7, F=0.2)
        sig = so[:, quiet].std(axis=1)
        peak = np.abs(so).max(axis=1)
        ratio_raw = peak / sig
        # 统一判据：积分还原后的波形峰值 vs 还原波形的噪声
        # （方波的微分输出只有边沿尖峰，原始峰值判据失效，见第 5 章）
        base = so[:, quiet].mean(axis=1, keepdims=True)
        integ = np.cumsum(so - base, axis=1) * h
        sig_i = integ[:, quiet].std(axis=1)
        ratio = np.abs(integ).max(axis=1) / sig_i
        pd = float((ratio > 5).mean())
        # 波形相似度：积分还原后与 s(t) 比（逐样本平均）
        s_ref = cfg["fn"](t, A, T)
        rs = [S.waveform_similarity(integ[i], s_ref) for i in range(N_TRIAL)]
        rows.append(dict(snr=float(snr), a=A, pd=pd,
                         r=float(np.mean(rs)), ratio=float(np.median(ratio)),
                         ratio_raw=float(np.median(ratio_raw)),
                         sec=round(time.time() - t0, 1)))
        print(f"  {name:8s} SNR={snr:3d} dB  Pd={pd*100:5.1f}%  r={np.mean(rs):.3f}"
              f"  积分峰值/σ={np.median(ratio):5.2f}  (原始 {np.median(ratio_raw):4.2f})"
              f"  ({rows[-1]['sec']}s)", flush=True)
    return rows


if __name__ == "__main__":
    out = {}
    for name, cfg in WAVES.items():
        print(f"=== {name} ===", flush=True)
        out[name] = run_wave(name, cfg)
    (DATA / "table3.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已写入", DATA / "table3.json")
