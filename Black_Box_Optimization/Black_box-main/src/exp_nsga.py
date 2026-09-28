"""
exp_nsga.py
-----------
Multi-Objective Evolutionary Search (NSGA-II) under both Single-Shot and
Settle-Then-Score semantics.
Sweeps 10 feedback settings beta in [0.0, 0.90] with 100 seeds each.
Also computes the empirical Pareto frontier of Cost vs. Stability Residual.
Saves results to results/nsga.json.
"""

import os
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import fluid as F
import optimize_util as OU
import optimizers as OPT

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)

BETAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N_SEEDS = 100


def eval_nsga_seed(args):
    beta, seed, kind, opt_cost = args
    incumbent = F.round_cfg(F.X0)

    if kind == "single":
        evalf = OPT.make_eval_fluid("single", beta, incumbent, seed0=seed)
    else:
        evalf = OPT.make_eval_fluid("sts", beta, incumbent, seed0=seed)

    cfg, _ = OPT.run_nsga(evalf, budget=140, seed=seed)
    cost = float(F.true_cost(cfg, beta))
    return max(0.0, cost - opt_cost)


def main():
    print(f"Running NSGA-II comparison across {len(BETAS)} beta values ({N_SEEDS} seeds)...")
    opt_costs = {}
    for b in BETAS:
        _, c_opt = OU.find_optimum(b)
        opt_costs[b] = float(c_opt)

    rows = []
    for b in BETAS:
        tasks_s = [(b, s, "single", opt_costs[b]) for s in range(N_SEEDS)]
        tasks_sts = [(b, s, "sts", opt_costs[b]) for s in range(N_SEEDS)]

        with ProcessPoolExecutor() as pool:
            reg_s = list(pool.map(eval_nsga_seed, tasks_s))
            reg_sts = list(pool.map(eval_nsga_seed, tasks_sts))

        rows.append(dict(
            beta=float(b),
            single_mean=float(np.mean(reg_s)),
            single_sem=float(np.std(reg_s) / np.sqrt(N_SEEDS)),
            sts_mean=float(np.mean(reg_sts)),
            sts_sem=float(np.std(reg_sts) / np.sqrt(N_SEEDS))
        ))
        print(f"  beta={b:.2f} | Single regret: {np.mean(reg_s):.4f} | STS regret: {np.mean(reg_sts):.4f}")

    print("Extracting empirical Pareto frontier at beta=0.60...")
    X_front, F_front = OPT.run_nsga_frontier(0.60, budget=2500, seed=42)
    order = np.argsort(F_front[:, 0])
    pareto_pts = [
        {"cost": float(F_front[i, 0]), "stability_residual": float(F_front[i, 1])}
        for i in order
    ]

    out_data = dict(rows=rows, pareto=pareto_pts, n_seeds=N_SEEDS)
    out_file = os.path.join(OUT_DIR, "nsga.json")
    with open(out_file, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"Saved NSGA-II benchmark and Pareto frontier to {out_file}")


if __name__ == "__main__":
    main()
