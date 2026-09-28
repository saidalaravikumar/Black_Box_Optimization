"""
exp_sensitivity.py
------------------
STS / PST HYPERPARAMETER SENSITIVITY .

Part A -- PST sensitivity:
  Sweeps trial count k in {3,5,10,15,20} and perturbation radius
  delta in {0.05,0.10,0.15,0.20,0.30}. For each (k,delta): detection rate at a
  weak-feedback stress point (beta=0.20) and false-positive rate on the
  non-performative control (beta=0), with the decision threshold calibrated from
  the beta=0 null (95th percentile), over REPS replications.
  NOTE: delta (PST perturbation radius) governs *detection only*; it does not
  enter the STS optimization objective, so post-deployment regret is analysed
  against the stability weight gamma and feedback strength beta below.

Part B -- Stability-weight sensitivity:
  Sweeps gamma in {0,0.025,0.05,0.10,0.20,0.30} (coefficient on R_stab in the STS
  fitness, matching the manuscript's Eq. for f_STS) x beta in {0.0,0.3,0.6,0.8}.
  For each cell: STS-CMA-ES post-deployment regret and final stability residual
  over N_SEEDS seeds. Yields the regret operating-region heatmap (gamma x beta)
  and the regret/residual-vs-gamma curves.

Saves results/sensitivity.json.
"""
import os, json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import fluid as F
import methods as M
import optimize_util as OU
import optimizers as OPT

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)

INCUMBENT = F.round_cfg(F.X0)
K_VALUES = [3, 5, 10, 15, 20]
DELTA_VALUES = [0.05, 0.10, 0.15, 0.20, 0.30]
GAMMA_VALUES = [0.0, 0.025, 0.05, 0.10, 0.20, 0.30]
BETAS_B = [0.0, 0.30, 0.60, 0.80]
PST_REPS = 40
PST_HORIZON = 90
PST_BETA_SIGNAL = 0.20
N_SEEDS = 25
BUDGET = 150


# ---------------- Part A: PST sensitivity ----------------
def _pst_z(args):
    x0, beta, k, delta, seed = args
    return M.screening_test(x0, beta, k=k, delta=delta, horizon=PST_HORIZON, seed=seed)["zscore"]


def run_pst():
    tasks_null = []
    tasks_sig = []
    for k in K_VALUES:
        for dl in DELTA_VALUES:
            for s in range(PST_REPS):
                tasks_null.append((INCUMBENT, 0.0, k, dl, 7000 + s))
                tasks_sig.append((INCUMBENT, PST_BETA_SIGNAL, k, dl, s))
    with ProcessPoolExecutor(max_workers=8) as pool:
        null_z = list(pool.map(_pst_z, tasks_null))
        sig_z = list(pool.map(_pst_z, tasks_sig))

    # reshape
    idx = 0
    null_by = {}
    for k in K_VALUES:
        for dl in DELTA_VALUES:
            null_by[(k, dl)] = null_z[idx:idx + PST_REPS]; idx += PST_REPS
    idx = 0
    sig_by = {}
    for k in K_VALUES:
        for dl in DELTA_VALUES:
            sig_by[(k, dl)] = sig_z[idx:idx + PST_REPS]; idx += PST_REPS

    grid = []
    for k in K_VALUES:
        for dl in DELTA_VALUES:
            thr = float(np.quantile(null_by[(k, dl)], 0.95))
            fpr = float(np.mean([z > thr for z in null_by[(k, dl)]]))
            det = float(np.mean([z > thr for z in sig_by[(k, dl)]]))
            grid.append(dict(k=k, delta=dl, threshold=thr, fpr=fpr, detection_rate=det))
    return grid


# ---------------- Part B: stability-weight sensitivity ----------------
def _stab_eval_run(args):
    gamma, beta, seed, c_opt = args
    rng = np.random.default_rng(seed + 1)

    def evalf(u):
        x = OPT.to_cfg(u)
        fit = F.true_cost(x, beta) + gamma * M.stability_residual(x, beta, n=24, seed=int(rng.integers(1e6)))
        return float(fit + rng.normal() * 0.02)
    cfg, _ = OPT.run_cma(evalf, budget=BUDGET, seed=seed)
    regret = max(0.0, float(F.true_cost(cfg, beta)) - c_opt)
    resid = float(M.stability_residual(cfg, beta, n=48, seed=seed + 3))
    return gamma, beta, regret, resid


def run_gamma():
    opt_costs = {b: OU.find_optimum(b)[1] for b in BETAS_B}
    tasks = [(g, b, s, opt_costs[b]) for g in GAMMA_VALUES for b in BETAS_B for s in range(N_SEEDS)]
    acc = {(g, b): {"regret": [], "resid": []} for g in GAMMA_VALUES for b in BETAS_B}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for g, b, reg, rs in pool.map(_stab_eval_run, tasks):
            acc[(g, b)]["regret"].append(reg)
            acc[(g, b)]["resid"].append(rs)
    rows = []
    for g in GAMMA_VALUES:
        for b in BETAS_B:
            a = acc[(g, b)]
            rows.append(dict(gamma=g, beta=float(b),
                             regret_mean=float(np.mean(a["regret"])),
                             regret_sem=float(np.std(a["regret"]) / np.sqrt(len(a["regret"]))),
                             residual_mean=float(np.mean(a["resid"])),
                             residual_sem=float(np.std(a["resid"]) / np.sqrt(len(a["resid"])))))
    return rows, opt_costs


def main():
    print("Part A: PST sensitivity (k x delta)...")
    pst_grid = run_pst()
    for r in pst_grid:
        if r["k"] in (5, 10) and r["delta"] in (0.10, 0.15):
            print(f"  k={r['k']} d={r['delta']}: det={r['detection_rate']*100:.0f}% fpr={r['fpr']*100:.0f}%")
    print("Part B: stability-weight gamma sensitivity...")
    gamma_rows, opt_costs = run_gamma()
    for r in gamma_rows:
        if r["beta"] in (0.6, 0.8):
            print(f"  gamma={r['gamma']:.3f} b={r['beta']}: regret={r['regret_mean']:.4f} resid={r['residual_mean']:.3f}")

    out = dict(k_values=K_VALUES, delta_values=DELTA_VALUES, gamma_values=GAMMA_VALUES,
               betas_b=BETAS_B, pst_reps=PST_REPS, pst_beta_signal=PST_BETA_SIGNAL,
               n_seeds=N_SEEDS, budget=BUDGET, pst_grid=pst_grid, gamma_rows=gamma_rows,
               opt_costs={str(k): v for k, v in opt_costs.items()})
    with open(os.path.join(OUT_DIR, "sensitivity.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved sensitivity.json")


if __name__ == "__main__":
    main()
