"""
exp_workload.py
---------------
WORKLOAD-DISTRIBUTION ROBUSTNESS .

Extends Testbed A beyond stationary Poisson demand to bursty ON/OFF and diurnal
(time-varying) exogenous arrivals (heavytail included for completeness). For each
workload we report, at beta=0.60:
  (i)  post-deployment regret of STS-CMA-ES vs single-shot CMA-ES (DES-based
       evaluation on that workload);
  (ii) settling-time detection error (|plateau - true|, |cusum - true|) on a
       fixed test config under that workload;
  (iii)PST detection rate under that workload.

Framed as robustness to workload STOCHASTICITY, not as a claim that any single
model captures all real workloads.

Saves results/workload.json.
"""
import os, json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import fluid as F
import des
import methods as M
import optimize_util as OU
import optimizers as OPT
import workloads as W

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)

BETA = 0.60
N_SEEDS = 40
KINDS = ["poisson", "bursty", "diurnal", "heavytail"]
DES_BUDGET = 200
WARM_SS, WIN_SS = 2, 14
X_TEST = np.array([28.0, 0.40, 4.0, 300.0])
X0 = F.round_cfg(F.X0)


def true_settle_time(lam, tol=0.04):
    final = lam[-80:].mean()
    band = max(tol * abs(final), 3.5 * lam[-80:].std())
    outside = np.where(np.abs(lam - final) > band)[0]
    return int(outside[-1] + 1) if len(outside) else 0


def opt_seed(args):
    kind, seed, c_opt, mode = args
    series = W.make_series(kind, DES_BUDGET, seed=seed)

    ctrl_cfg = np.array([30.0, 2.5, 0.0, 120.0])  # stable control replica

    def evalf(u):
        x = OPT.to_cfg(u)
        d = des.deploy(x, BETA, lam0_series=series, n_steps=DES_BUDGET, seed=seed)
        if mode == "single":
            return float(np.mean(d["cost"][WARM_SS:WARM_SS + WIN_SS]))
        # Full STS: settle + dual-replica control differencing (removes exogenous
        # workload variation, which is essential under time-varying demand).
        t_s = min(M.detect_settling(d["lam"], method="plateau"), DES_BUDGET - 1)
        score = float(np.mean(d["cost"][t_s:]))
        dc = des.deploy(ctrl_cfg, BETA, lam0_series=series, n_steps=DES_BUDGET, seed=seed + 777)
        drift_bias = float(np.mean(dc["cost"][t_s:]) - np.mean(dc["cost"][:WARM_SS + WIN_SS]))
        return score - drift_bias
    cfg, _ = OPT.run_cma(evalf, budget=90, seed=seed)
    return kind, mode, max(0.0, float(F.true_cost(cfg, BETA)) - c_opt)


def settle_seed(args):
    kind, seed = args
    series = W.make_series(kind, 700, seed=seed)
    d = des.deploy(X_TEST, BETA, lam0_series=series, n_steps=700, seed=seed)
    tt = true_settle_time(d["lam"])
    tp = M.detect_settling(d["lam"], method="plateau")
    tc = M.detect_settling(d["lam"], method="cusum")
    return kind, abs(tp - tt), abs(tc - tt)


def screening_workload(x0, beta, kind, k=10, delta=0.15, horizon=200, seed=0):
    """Workload-aware PST: paired micro-perturbations with candidate and control
    replicas both driven by the SAME non-stationary workload series, so the
    control captures the workload's exogenous variability (the null it tests against)."""
    rng = np.random.default_rng(seed)
    span = F.BOUNDS[:, 1] - F.BOUNDS[:, 0]
    signs = np.array([-1.0, 1.0])
    loads, ctrl = [], []
    for i in range(k):
        s = signs[i % 2]
        direction = rng.normal(size=4); direction /= (np.linalg.norm(direction) + 1e-9)
        xp = F._clip_cfg(x0 + s * delta * span * direction)
        ser = W.make_series(kind, horizon, seed=int(rng.integers(1e6)))
        d = des.deploy(xp, beta, lam0_series=ser, n_steps=horizon, seed=int(rng.integers(1e6)))
        loads.append(d["lam"][-horizon // 3:].mean())
        ser2 = W.make_series(kind, horizon, seed=int(rng.integers(1e6)))
        dc = des.deploy(x0, beta, lam0_series=ser2, n_steps=horizon, seed=int(rng.integers(1e6)))
        ctrl.append(dc["lam"][-horizon // 3:].mean())
    loads = np.array(loads); ctrl = np.array(ctrl)
    base = ctrl.mean(); ctrl_std = ctrl.std() + 1e-6
    induced = np.abs(loads - base).mean()
    return float(induced / ctrl_std)


def main():
    c_opt = OU.find_optimum(BETA)[1]
    print(f"Workload robustness at beta={BETA}, c_opt={c_opt:.3f}")

    # (i) optimization regret per workload
    tasks = [(k, s, c_opt, m) for k in KINDS for m in ("single", "sts") for s in range(N_SEEDS)]
    reg = {(k, m): [] for k in KINDS for m in ("single", "sts")}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for k, m, r in pool.map(opt_seed, tasks):
            reg[(k, m)].append(r)

    # (ii) settling detection error per workload
    stasks = [(k, s) for k in KINDS for s in range(N_SEEDS)]
    serr = {k: {"pl": [], "cu": []} for k in KINDS}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for k, ep, ec in pool.map(settle_seed, stasks):
            serr[k]["pl"].append(ep); serr[k]["cu"].append(ec)

    # (iii) workload-aware PST detection: per-workload null threshold + detection
    det = {}
    thr_by = {}
    for k in KINDS:
        null_z = [screening_workload(X0, 0.0, k, seed=9000 + s) for s in range(N_SEEDS)]
        thr = float(np.quantile(null_z, 0.95))
        thr_by[k] = thr
        zs = [screening_workload(X0, BETA, k, seed=s) for s in range(N_SEEDS)]
        det[k] = float(np.mean([z > thr for z in zs]))

    rows = []
    for k in KINDS:
        rows.append(dict(
            workload=k,
            regret_single_mean=float(np.mean(reg[(k, "single")])),
            regret_single_sem=float(np.std(reg[(k, "single")]) / np.sqrt(N_SEEDS)),
            regret_sts_mean=float(np.mean(reg[(k, "sts")])),
            regret_sts_sem=float(np.std(reg[(k, "sts")]) / np.sqrt(N_SEEDS)),
            settle_err_plateau=float(np.mean(serr[k]["pl"])),
            settle_err_cusum=float(np.mean(serr[k]["cu"])),
            pst_detection=det[k],
        ))
        print(f"  {k:10s}: single={np.mean(reg[(k,'single')]):.3f} STS={np.mean(reg[(k,'sts')]):.4f} "
              f"settle_err(pl)={np.mean(serr[k]['pl']):.0f} PST={det[k]*100:.0f}%")

    out = dict(beta=BETA, n_seeds=N_SEEDS, kinds=KINDS, c_opt=c_opt,
               pst_threshold_by_workload=thr_by, rows=rows)
    with open(os.path.join(OUT_DIR, "workload.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved workload.json")


if __name__ == "__main__":
    main()
