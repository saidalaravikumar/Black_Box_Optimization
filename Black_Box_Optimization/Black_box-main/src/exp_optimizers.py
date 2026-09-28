"""
exp_optimizers.py
-----------------
Executes 100 independent Monte Carlo seeds across 19 feedback levels beta in [0.0, 0.90]
comparing state-of-the-art baselines against proposed Settle-Then-Score (STS) semantics:
  1. Random / single-shot
  2. CMA-ES / single-shot
  3. BO-GP / single-shot (Gaussian Process Expected Improvement)
  4. TUNA-filter / single-shot (Repeated multi-sample variance filtering)
  5. NSGA-II / single-shot
  6. Proposed: CMA-ES / settle-then-score
  7. Proposed: CMA-ES / settle+stability
  8. Proposed: NSGA-II / settle-then-score
Saves results to results/optimizers.json.
"""

import os
import json
import numpy as np
from scipy.stats import wilcoxon
from concurrent.futures import ProcessPoolExecutor
import fluid as F
import optimize_util as OU
import optimizers as OPT

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)

BETAS = np.linspace(0.0, 0.90, 19).tolist()
N_SEEDS = 100
BUDGET = 180


def eval_single_seed(args):
    beta, seed, opt_cost = args
    incumbent = F.round_cfg(F.X0)

    # 1. Single-shot eval callback
    eval_single = OPT.make_eval_fluid("single", beta, incumbent, seed0=seed)
    # 2. STS eval callback
    eval_sts = OPT.make_eval_fluid("sts", beta, incumbent, stability=False, seed0=seed)
    # 3. STS + stability callback
    eval_stab = OPT.make_eval_fluid("sts", beta, incumbent, stability=True, seed0=seed)

    # Run Methods
    cfg_rnd, _ = OPT.run_random(eval_single, budget=BUDGET, seed=seed)
    cfg_cma_s, _ = OPT.run_cma(eval_single, budget=BUDGET, seed=seed)
    cfg_bo_s, _ = OPT.run_bo_gp(eval_single, budget=BUDGET, seed=seed, n_init=12)
    cfg_tuna_s, _ = OPT.run_tuna_filter(eval_single, budget=BUDGET, seed=seed, samples_per_eval=4)
    cfg_nsga_s, _ = OPT.run_nsga(eval_single, budget=BUDGET, seed=seed)

    cfg_cma_sts, _ = OPT.run_cma(eval_sts, budget=BUDGET, seed=seed)
    cfg_cma_stab, _ = OPT.run_cma(eval_stab, budget=BUDGET, seed=seed)
    cfg_nsga_sts, _ = OPT.run_nsga(eval_sts, budget=BUDGET, seed=seed)

    # Evaluate realized true costs
    costs = {
        "Random / single-shot": float(F.true_cost(cfg_rnd, beta)),
        "CMA-ES / single-shot": float(F.true_cost(cfg_cma_s, beta)),
        "BO-GP / single-shot": float(F.true_cost(cfg_bo_s, beta)),
        "TUNA-filter / single-shot": float(F.true_cost(cfg_tuna_s, beta)),
        "NSGA-II / single-shot": float(F.true_cost(cfg_nsga_s, beta)),
        "Proposed: CMA-ES / settle-then-score": float(F.true_cost(cfg_cma_sts, beta)),
        "Proposed: CMA-ES / settle+stability": float(F.true_cost(cfg_cma_stab, beta)),
        "Proposed: NSGA-II / settle-then-score": float(F.true_cost(cfg_nsga_sts, beta)),
    }

    regrets = {k: max(0.0, v - opt_cost) for k, v in costs.items()}
    return beta, regrets


def main():
    print(f"Executing 100-seed Monte Carlo optimization across {len(BETAS)} beta values...")
    # Precompute true optimal cost per beta (instant lookup from ground_truth.json)
    gt_file = os.path.join(OUT_DIR, "ground_truth.json")
    if os.path.exists(gt_file):
        with open(gt_file) as f:
            gt_data = json.load(f)
        gt_betas = np.array([d["beta"] for d in gt_data])
        gt_costs = np.array([d["cost_opt"] for d in gt_data])
        opt_costs = {b: float(np.interp(b, gt_betas, gt_costs)) for b in BETAS}
    else:
        grid = OU.get_grid(10)
        opt_costs = {b: float(np.min(F.true_cost_vec(grid, b))) for b in BETAS}

    tasks = []
    for b in BETAS:
        for s in range(N_SEEDS):
            tasks.append((b, s, opt_costs[b]))

    print(f"Total optimization runs to execute: {len(tasks)} (using multiprocessing)...", flush=True)
    results_by_beta = {b: {} for b in BETAS}

    with ProcessPoolExecutor() as pool:
        for idx, (b, regrets) in enumerate(pool.map(eval_single_seed, tasks)):
            for method, reg in regrets.items():
                if method not in results_by_beta[b]:
                    results_by_beta[b][method] = []
                results_by_beta[b][method].append(reg)
            if (idx + 1) % 190 == 0:
                print(f"  Completed {idx + 1} / {len(tasks)} runs ({((idx + 1) / len(tasks)) * 100:.1f}%)...", flush=True)

    rows = []
    tests = {}

    for b in BETAS:
        methods_data = results_by_beta[b]
        for m, arr in methods_data.items():
            arr = np.array(arr)
            rows.append({
                "beta": float(b),
                "method": m,
                "regret_mean": float(np.mean(arr)),
                "regret_std": float(np.std(arr)),
                "regret_sem": float(np.std(arr) / np.sqrt(len(arr))),
                "regret_median": float(np.median(arr)),
                "regrets": [float(v) for v in arr]
            })

        # Wilcoxon test: Proposed STS CMA-ES vs CMA-ES Single-Shot
        s_arr = np.array(methods_data["CMA-ES / single-shot"])
        sts_arr = np.array(methods_data["Proposed: CMA-ES / settle-then-score"])
        diff = s_arr - sts_arr
        frac_worse = float(np.mean(diff > 0))

        if np.all(diff == 0):
            w_stat, p_val = 0.0, 1.0
        else:
            try:
                res = wilcoxon(s_arr, sts_arr, alternative="greater")
                w_stat, p_val = float(res.statistic), float(res.pvalue)
            except Exception:
                w_stat, p_val = 0.0, 1.0

        tests[str(round(b, 2))] = {
            "W": w_stat,
            "p_value": p_val,
            "frac_single_worse": frac_worse
        }

    out_data = {
        "n_seeds": N_SEEDS,
        "betas": [float(b) for b in BETAS],
        "opt_costs": {str(round(k, 2)): v for k, v in opt_costs.items()},
        "rows": rows,
        "tests": tests
    }

    out_file = os.path.join(OUT_DIR, "optimizers.json")
    with open(out_file, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"Successfully saved 100-seed optimization benchmark to {out_file}!")


if __name__ == "__main__":
    main()
