"""
exp_cache.py
------------
Evaluates queueing-free cache and prefetch testbed under response-dependent browsing.
Sweeps 16 fine feedback strengths beta_c in [0.0, 0.65] with 100 seeds each.
Demonstrates:
1. Universal emergence of performativity gap in non-queueing systems.
2. Myopic policy over-prefetches due to illusion of high popularity skew.
3. CMA-ES under STS evaluation successfully recovers near-optimal low-prefetch policies.
Saves results to results/cache.json.
"""

import os
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import cma
import cache as Ca

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)


def make_grid(n=25):
    ax = [np.linspace(lo, hi, n) for lo, hi in Ca.BOUNDS]
    G = np.meshgrid(*ax, indexing="ij")
    return np.column_stack([g.ravel() for g in G])


def to_cfg(u):
    u = np.clip(u, 0.0, 1.0)
    return Ca.BOUNDS[:, 0] + u * (Ca.BOUNDS[:, 1] - Ca.BOUNDS[:, 0])


def myopic_cost_vec(Y, s_inc):
    Y = np.atleast_2d(Y)
    C, p, ttl = Y[:, 0], Y[:, 1], Y[:, 2]
    effC = C * (0.6 + 0.4 * np.tanh(ttl))
    h0 = 1.0 - (1.0 - effC) ** (0.6 + s_inc)
    conv = Ca.PREF_EFF * np.clip(s_inc / Ca.S0, 0.0, 1.0)
    h = np.minimum(h0 + p * (1.0 - h0) * conv, 0.999)
    L = h * Ca.L_HIT + (1.0 - h) * Ca.L_MISS
    bw = p * (1.0 - conv)
    return Ca.W_MISS * (1.0 - h) + Ca.W_BW * Ca.BW_COST * bw + Ca.W_LATC * L


def cma_cache_seed(args):
    kind, beta_c, seed, budget, s_inc = args
    rng = np.random.default_rng(seed)

    def f(u):
        y = to_cfg(u)
        if kind == "single":
            v = float(myopic_cost_vec(y, s_inc)[0])
        else:
            v = Ca.true_cost(y, beta_c)
        return v + rng.normal() * 0.015

    es = cma.CMAEvolutionStrategy(
        3 * [0.5], 0.25,
        {"bounds": [0, 1], "maxfevals": budget, "seed": seed + 1,
         "verbose": -9, "popsize": 8}
    )
    best_u, best_f = None, np.inf
    while not es.stop():
        cands = es.ask()
        vals = [f(u) for u in cands]
        es.tell(cands, vals)
        idx = int(np.argmin(vals))
        if vals[idx] < best_f:
            best_f, best_u = vals[idx], cands[idx]
    best_y = to_cfg(best_u)
    return float(Ca.true_cost(best_y, beta_c))


def main():
    betas_c = np.round(np.linspace(0.0, 0.65, 16), 3)
    N_SEEDS = 100
    grid = make_grid(25)

    print(f"Computing ground truth for cache testbed across {len(betas_c)} beta_c settings...")
    gap_rows = []
    opt_costs = {}

    for bc in betas_c:
        bc_f = float(bc)
        s_inc = Ca.settle_cache(Ca.INCUMBENT, bc_f)["s"]

        # Grid search for true optimum and myopic optimum
        true_costs = Ca.true_cost_vec(grid, bc_f)
        idx_opt = int(np.argmin(true_costs))
        y_opt = grid[idx_opt]
        c_opt = float(true_costs[idx_opt])
        opt_costs[bc_f] = c_opt

        myo_costs = myopic_cost_vec(grid, s_inc)
        idx_myo = int(np.argmin(myo_costs))
        y_myo = grid[idx_myo]
        c_myo_realised = float(Ca.true_cost(y_myo, bc_f))
        gap = float(c_myo_realised - c_opt)

        gap_rows.append(dict(
            beta_c=bc_f,
            cost_opt=c_opt,
            cost_myopic=c_myo_realised,
            perf_gap=gap,
            p_opt=float(y_opt[1]),
            p_myopic=float(y_myo[1])
        ))
        print(f"  beta_c={bc_f:.2f}: gap={gap:.4f}, p_opt={y_opt[1]:.2f}, p_myo={y_myo[1]:.2f}")

    print(f"Running 100-seed CMA-ES (Single vs STS) on cache testbed...")
    eval_betas = [0.0, 0.15, 0.30, 0.45, 0.60]
    opt_rows = []

    for bc in eval_betas:
        bc_f = float(bc)
        s_inc = Ca.settle_cache(Ca.INCUMBENT, bc_f)["s"]
        c_opt = float(np.min(Ca.true_cost_vec(grid, bc_f)))

        tasks_s = [("single", bc_f, s, 160, s_inc) for s in range(N_SEEDS)]
        tasks_sts = [("sts", bc_f, s, 160, s_inc) for s in range(N_SEEDS)]

        with ProcessPoolExecutor() as pool:
            costs_s = list(pool.map(cma_cache_seed, tasks_s))
            costs_sts = list(pool.map(cma_cache_seed, tasks_sts))

        reg_s = [max(0.0, c - c_opt) for c in costs_s]
        reg_sts = [max(0.0, c - c_opt) for c in costs_sts]

        opt_rows.append(dict(
            beta_c=bc_f,
            single_mean=float(np.mean(reg_s)),
            single_sem=float(np.std(reg_s) / np.sqrt(N_SEEDS)),
            sts_mean=float(np.mean(reg_sts)),
            sts_sem=float(np.std(reg_sts) / np.sqrt(N_SEEDS))
        ))
        print(f"  beta_c={bc_f:.2f} | Single regret: {np.mean(reg_s):.4f} | STS regret: {np.mean(reg_sts):.4f}")

    out_data = dict(gap=gap_rows, optimizer=opt_rows, n_seeds=N_SEEDS)
    out_file = os.path.join(OUT_DIR, "cache.json")
    with open(out_file, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"Saved cache testbed benchmark to {out_file}")


if __name__ == "__main__":
    main()
