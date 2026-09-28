"""
exp_adaptive_workload.py  
----------------------------------------
ADAPTIVE EVALUATION POLICY ACROSS WORKLOAD REGIMES (reviewer Exp. 2).

Table 6 shows STS helps under (near-)stationary induced load (Poisson,
heavy-tailed) but *hurts* under strongly non-stationary load (bursty ON/OFF,
diurnal), because the settling premise is violated. Instead of hiding this, we
close the loop with the operational guideline already argued in the paper: an
adaptive policy that gates the evaluation semantics.

Policy:
    PST detects performativity.
      - not performative      -> single-shot evaluation
      - performative:
          stationarity gate (CoV of a stable control replica's induced arrival
          rate; ties directly to the STS settling premise, threshold TAU_COV)
            - stationary       -> Settle-Then-Score (STS)
            - non-stationary   -> robust short-window evaluation (deploy the
              candidate, discard warm-up, take the median of short-window costs;
              no settling assumption, robust to bursts / diurnal swings)

We compare, per workload, post-deployment regret (mean-field true cost of the
selected config vs the global optimum) for: single-shot, STS, and Adaptive.
single/STS reproduce the exp_workload semantics so numbers are comparable to
Table 6. beta = 0.60.

Saves ../results/adaptive_workload.json.
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
KINDS = ["poisson", "heavytail", "bursty", "diurnal"]
DES_BUDGET = 200
WARM_SS, WIN_SS = 2, 14
CMA_BUDGET = 90
TAU_COV = 0.06          # stationarity threshold on control-replica arrival-rate CoV
WARM_ROB = 20           # warm-up discarded by robust short-window evaluator
ROB_WIN = 10            # short window length for robust evaluator
CTRL_CFG = np.array([30.0, 2.5, 0.0, 120.0])


def control_cov(series, seed):
    """CoV of a stable control replica's induced arrival rate under the workload.
    High CoV => the system sees persistently non-stationary load (settling premise
    broken)."""
    dc = des.deploy(CTRL_CFG, BETA, lam0_series=series, n_steps=DES_BUDGET, seed=seed + 4321)
    lam = dc["lam"][WARM_ROB:]
    return float(lam.std() / (abs(lam.mean()) + 1e-9))


def eval_single(u, series, seed):
    x = OPT.to_cfg(u)
    d = des.deploy(x, BETA, lam0_series=series, n_steps=DES_BUDGET, seed=seed)
    return float(np.mean(d["cost"][WARM_SS:WARM_SS + WIN_SS]))


def eval_sts(u, series, seed):
    x = OPT.to_cfg(u)
    d = des.deploy(x, BETA, lam0_series=series, n_steps=DES_BUDGET, seed=seed)
    t_s = min(M.detect_settling(d["lam"], method="plateau"), DES_BUDGET - 1)
    score = float(np.mean(d["cost"][t_s:]))
    dc = des.deploy(CTRL_CFG, BETA, lam0_series=series, n_steps=DES_BUDGET, seed=seed + 777)
    drift_bias = float(np.mean(dc["cost"][t_s:]) - np.mean(dc["cost"][:WARM_SS + WIN_SS]))
    return score - drift_bias


def eval_robust(u, series, seed):
    """Robust short-window evaluation: deploy candidate (feedback active), discard
    warm-up, take the median of consecutive short-window mean costs. No settling
    assumption; median is robust to burst/diurnal swings."""
    x = OPT.to_cfg(u)
    d = des.deploy(x, BETA, lam0_series=series, n_steps=DES_BUDGET, seed=seed)
    post = d["cost"][WARM_ROB:]
    nwin = max(1, len(post) // ROB_WIN)
    win_means = [float(np.mean(post[i * ROB_WIN:(i + 1) * ROB_WIN])) for i in range(nwin)]
    return float(np.median(win_means))


def run_seed(args):
    kind, seed, c_opt = args
    series = W.make_series(kind, DES_BUDGET, seed=seed)

    out = {}
    # single-shot
    cfg, _ = OPT.run_cma(lambda u: eval_single(u, series, seed), budget=CMA_BUDGET, seed=seed)
    out["single"] = max(0.0, float(F.true_cost(cfg, BETA)) - c_opt)
    # STS
    cfg, _ = OPT.run_cma(lambda u: eval_sts(u, series, seed), budget=CMA_BUDGET, seed=seed)
    out["sts"] = max(0.0, float(F.true_cost(cfg, BETA)) - c_opt)
    # Adaptive: gate on control-replica CoV
    cov = control_cov(series, seed)
    stationary = cov <= TAU_COV
    ev = eval_sts if stationary else eval_robust
    cfg, _ = OPT.run_cma(lambda u: ev(u, series, seed), budget=CMA_BUDGET, seed=seed)
    out["adaptive"] = max(0.0, float(F.true_cost(cfg, BETA)) - c_opt)
    return kind, out, cov, int(stationary)


def main():
    c_opt = OU.find_optimum(BETA)[1]
    print(f"Adaptive workload policy at beta={BETA}, c_opt={c_opt:.4f}", flush=True)
    tasks = [(k, s, c_opt) for k in KINDS for s in range(N_SEEDS)]
    reg = {k: {"single": [], "sts": [], "adaptive": []} for k in KINDS}
    covs = {k: [] for k in KINDS}
    route_sts = {k: 0 for k in KINDS}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for k, o, cov, stat in pool.map(run_seed, tasks):
            for m in ("single", "sts", "adaptive"):
                reg[k][m].append(o[m])
            covs[k].append(cov)
            route_sts[k] += stat

    rows = []
    for k in KINDS:
        row = dict(workload=k,
                   control_cov_mean=float(np.mean(covs[k])),
                   frac_routed_sts=float(route_sts[k] / N_SEEDS))
        for m in ("single", "sts", "adaptive"):
            a = np.array(reg[k][m])
            row[f"regret_{m}_mean"] = float(np.mean(a))
            row[f"regret_{m}_sem"] = float(np.std(a) / np.sqrt(len(a)))
            row[f"regret_{m}_median"] = float(np.median(a))
        rows.append(row)
        print(f"  {k:10s} CoV={np.mean(covs[k]):.3f} ->STS {route_sts[k]}/{N_SEEDS} | "
              f"single={row['regret_single_mean']:.3f} sts={row['regret_sts_mean']:.3f} "
              f"adaptive={row['regret_adaptive_mean']:.3f}", flush=True)

    out = dict(beta=BETA, n_seeds=N_SEEDS, kinds=KINDS, c_opt=c_opt,
               tau_cov=TAU_COV, rows=rows)
    with open(os.path.join(OUT_DIR, "adaptive_workload.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved adaptive_workload.json")


if __name__ == "__main__":
    main()
