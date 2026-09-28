"""
exp_settling.py
---------------
Empirical validation of settling detection and dual-replica drift attribution:
1. Online Settling Calibration:
   Sweeps client population relaxation rate kappa across 14 fine settings in [0.02, 0.25] (100 reps).
   Compares online plateau and CUSUM stopping times against ground-truth convergence.
2. Dual-Replica Exogenous Drift Attribution:
   Injects a +40% linear exogenous load surge during the evaluation window (100 reps).
   Demonstrates that parallel control differencing neutralizes background bias.
Saves results to results/settling.json.
"""

import os
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import des
import methods as M
import fluid as F

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)

X_TEST      = np.array([26.0, 0.35, 3.0, 200.0])  # Non-trivial test configuration
CONTROL_CFG = np.array([30.0, 2.5, 0.0, 120.0])   # Stable reference control
DRIFT_CFG   = np.array([28.0, 2.0, 0.0, 150.0])   # Target candidate under drift


def true_settle_time(lam, tol=0.04):
    final = lam[-100:].mean()
    band = max(tol * abs(final), 3.5 * lam[-100:].std())
    outside = np.where(np.abs(lam - final) > band)[0]
    return int(outside[-1] + 1) if len(outside) else 0


def eval_kappa_seed(args):
    kap, s, beta = args
    d = des.deploy(X_TEST, beta, n_steps=800, kappa=kap, seed=s)
    t_tru = true_settle_time(d["lam"])
    t_pl = M.detect_settling(d["lam"], method="plateau")
    t_cu = M.detect_settling(d["lam"], method="cusum")
    return kap, t_tru, t_pl, t_cu


def eval_drift_seed(s, beta=0.6, drift_mag=0.40):
    # Nominal reference
    t_ref = F.true_cost(CONTROL_CFG, beta)
    t_tgt = F.true_cost(DRIFT_CFG, beta)

    # Injected drift
    d_ctrl = des.deploy(CONTROL_CFG, beta, n_steps=600, drift=drift_mag, seed=s)
    d_cand = des.deploy(DRIFT_CFG, beta, n_steps=600, drift=drift_mag, seed=s + 1000)

    t_s = max(
        M.detect_settling(d_cand["lam"], method="cusum"),
        M.detect_settling(d_ctrl["lam"], method="cusum"),
        60
    )

    c_naive = float(d_cand["cost"][t_s:].mean())
    c_ctrl_mean = float(d_ctrl["cost"][t_s:].mean())
    c_corr = c_naive - (c_ctrl_mean - t_ref)

    return abs(c_naive - t_tgt), abs(c_corr - t_tgt)


def main():
    kappas = np.round(np.linspace(0.02, 0.25, 14), 3)
    reps = 100
    beta = 0.7

    print(f"Evaluating settling tracking across {len(kappas)} kappa values with {reps} reps...")
    tasks = [(float(k), s, beta) for k in kappas for s in range(reps)]
    res_by_k = {float(k): {"tru": [], "pl": [], "cu": []} for k in kappas}

    with ProcessPoolExecutor() as pool:
        for k, tru, pl, cu in pool.map(eval_kappa_seed, tasks):
            res_by_k[k]["tru"].append(tru)
            res_by_k[k]["pl"].append(pl)
            res_by_k[k]["cu"].append(cu)

    settling_rows = []
    for k in kappas:
        kf = float(k)
        tru = res_by_k[kf]["tru"]
        pl = res_by_k[kf]["pl"]
        cu = res_by_k[kf]["cu"]
        settling_rows.append(dict(
            kappa=kf,
            true_mean=float(np.mean(tru)),
            true_sem=float(np.std(tru) / np.sqrt(reps)),
            plateau_mean=float(np.mean(pl)),
            plateau_sem=float(np.std(pl) / np.sqrt(reps)),
            cusum_mean=float(np.mean(cu)),
            cusum_sem=float(np.std(cu) / np.sqrt(reps))
        ))
        print(f"  kappa={kf:.3f}: True={np.mean(tru):.1f}, Plateau={np.mean(pl):.1f}, CUSUM={np.mean(cu):.1f}")

    print(f"Evaluating drift attribution across {reps} replications (+40% load surge)...")
    naive_errs, corr_errs = [], []
    with ProcessPoolExecutor() as pool:
        for n_err, c_err in pool.map(eval_drift_seed, range(reps)):
            naive_errs.append(n_err)
            corr_errs.append(c_err)

    drift_res = dict(
        naive_err_mean=float(np.mean(naive_errs)),
        naive_err_sem=float(np.std(naive_errs) / np.sqrt(reps)),
        corr_err_mean=float(np.mean(corr_errs)),
        corr_err_sem=float(np.std(corr_errs) / np.sqrt(reps)),
        reduction_pct=float((1.0 - np.mean(corr_errs) / np.mean(naive_errs)) * 100.0)
    )
    print(f"  Naive error: {drift_res['naive_err_mean']:.3f} +/- {drift_res['naive_err_sem']:.3f}")
    print(f"  Drift-corr:  {drift_res['corr_err_mean']:.3f} +/- {drift_res['corr_err_sem']:.3f} ({drift_res['reduction_pct']:.1f}% reduction)")

    out_data = dict(settling=settling_rows, drift=drift_res)
    out_file = os.path.join(OUT_DIR, "settling.json")
    with open(out_file, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"Saved settling and drift results to {out_file}")


if __name__ == "__main__":
    main()
