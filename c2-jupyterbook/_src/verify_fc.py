"""一次性验证：fc=130 kHz 能否复现表 3 的高斯导数脉冲整行。

跑法：.venv/Scripts/python.exe c2-jupyterbook/_src/verify_fc.py
"""
import sys
import pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import c2sim as S

om, h, T, D = 4e6, 1e-9, 40e-6, 0.1
t = np.arange(int(round(T / h)) + 1) * h
PAPER_R = [-20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10]
PAPER_r = [0.85, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.92, 0.93, 0.93, 0.93]
PAPER_Pd = [37.5, 55.0, 73.0, 86.5, 94.0, 99.5, 100, 100, 100, 100, 100]

print(f"{'SNR':>5}{'r(书)':>8}{'r(文)':>7}{'Pd(书)':>8}{'Pd(文)':>8}{'SNRI':>7}")
for snr, rp, pp in zip(PAPER_R, PAPER_r, PAPER_Pd):
    A = np.sqrt(4 * D * 10 ** (snr / 10))
    s = S.gauss_deriv(t, A=A, t0=T / 2, c=1.45e-6)
    so, X = S.simulate(t, drive=s, batch=100, sigma=np.sqrt(D), hold=1,
                       seed=100 - snr, omega=om, return_state=True)
    m = S.chain_metrics(so, X, s, t, h, fc=130e3)
    print(f"{snr:>5}{m['r_dx']:>8.3f}{rp:>7.2f}{m['pd']*100:>8.0f}{pp:>8.1f}"
          f"{m['snr_o']-snr:>7.1f}", flush=True)
