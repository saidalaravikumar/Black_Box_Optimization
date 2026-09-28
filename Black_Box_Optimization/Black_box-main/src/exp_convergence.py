"""
exp_convergence.py
------------------
Evaluates iteration-by-iteration convergence trajectories across 100 Monte Carlo seeds.
Compares optimization velocity and true realized post-deployment regret over evaluations 1 to 150
under performative feedback (beta = 0.60 and beta = 0.80):
  1. CMA-ES (Single-Shot)
  2. BO-GP (Single-Shot)
  3. TUNA-filter (Single-Shot)
  4. Proposed (STS CMA-ES)
  5. Proposed (Stability-Reg. STS)
Saves results to results/convergence.json.
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

N_SEEDS = 100
BUDGET = 150
TEST_BETAS = [0.60, 0.80]


def run_trajectory_seed(args):
    beta, seed, opt_cost = args
    incumbent = F.round_cfg(F.X0)

    eval_single = OPT.make_eval_fluid("single", beta, incumbent, seed0=seed)
    eval_sts = OPT.make_eval_fluid("sts", beta, incumbent, stability=False, seed0=seed)
    eval_stab = OPT.make_eval_fluid("sts", beta, incumbent, stability=True, seed0=seed)

    true_eval = lambda u: F.true_cost(OPT.to_cfg(u), beta)

    # Track histories of true realized regret
    _, _, h_cma_s = OPT.run_cma(eval_single, budget=BUDGET, seed=seed, track_history=True, true_evalf=true_eval)
    _, _, h_bo_s = OPT.run_bo_gp(eval_single, budget=BUDGET, seed=seed, n_init=12, track_history=True, true_evalf=true_eval)
    _, _, h_tuna_s = OPT.run_tuna_filter(eval_single, budget=BUDGET, seed=seed, samples_per_eval=4, track_history=True, true_evalf=true_eval)
    _, _, h_cma_sts = OPT.run_cma(eval_sts, budget=BUDGET, seed=seed, track_history=True, true_evalf=true_eval)
    _, _, h_cma_stab = OPT.run_cma(eval_stab, budget=BUDGET, seed=seed, track_history=True, true_evalf=true_eval)

    def extract_steps(hist):
        # Interpolate history to dense steps 1..BUDGET
        res = []
        cur_val = hist[0][1]
        hist_dict = dict(hist)
        for s in range(1, BUDGET + 1):
            if s in hist_dict:
                cur_val = hist_dict[s]
            res.append(max(0.0, cur_val - opt_cost))
        return res

    return beta, {
        "CMA-ES (Single-Shot)": extract_steps(h_cma_s),
        "BO-GP (Single-Shot)": extract_steps(h_bo_s),
        "TUNA-filter (Single-Shot)": extract_steps(h_tuna_s),
        "Proposed (STS CMA-ES)": extract_steps(h_cma_sts),
        "Proposed (Stability-Reg. STS)": extract_steps(h_cma_stab)
    }


def main():
    print(f"Executing convergence trajectory benchmark across {N_SEEDS} seeds for beta={TEST_BETAS}...")
    opt_costs = {}
    for b in TEST_BETAS:
        _, c_opt = OU.find_optimum(b)
        opt_costs[b] = float(c_opt)

    tasks = []
    for b in TEST_BETAS:
        for s in range(N_SEEDS):
            tasks.append((b, s, opt_costs[b]))

    data_by_beta = {b: {} for b in TEST_BETAS}

    with ProcessPoolExecutor() as pool:
        for idx, (b, traj_dict) in enumerate(pool.map(run_trajectory_seed, tasks)):
            for method, arr in traj_dict.items():
                if method not in data_by_beta[b]:
                    data_by_beta[b][method] = []
                data_by_beta[b][method].append(arr)
            if (idx + 1) % 50 == 0:
                print(f"  Completed {idx + 1} / {len(tasks)} convergence runs...")

    summary = {
        "budget": BUDGET,
        "n_seeds": N_SEEDS,
        "iterations": list(range(1, BUDGET + 1)),
        "betas": {}
    }

    key_map = {
        "Proposed (STS CMA-ES)": "Proposed_STS_CMA_ES",
        "Proposed (Stability-Reg. STS)": "Proposed_Stability_Reg__STS",
        "BO-GP (Single-Shot)": "BO_GP_Single_Shot_",
        "TUNA-filter (Single-Shot)": "TUNA_filter_Single_Shot_",
        "CMA-ES (Single-Shot)": "CMA_ES_Single_Shot_"
    }

    for b in TEST_BETAS:
        b_key = "x" + str(b).replace(".", "_")
        summary["betas"][b_key] = {}
        summary["betas"][str(b)] = {}
        for m, runs in data_by_beta[b].items():
            runs_arr = np.array(runs)  # (N_SEEDS, BUDGET)
            stat = {
                "mean": [float(v) for v in np.mean(runs_arr, axis=0)],
                "std": [float(v) for v in np.std(runs_arr, axis=0)],
                "sem": [float(v) for v in np.std(runs_arr, axis=0) / np.sqrt(N_SEEDS)],
                "median": [float(v) for v in np.median(runs_arr, axis=0)]
            }
            summary["betas"][str(b)][m] = stat
            # Also store under sanitized identifier for MATLAB
            matlab_m = key_map.get(m, m.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", ""))
            summary["betas"][b_key][matlab_m] = stat

    out_file = os.path.join(OUT_DIR, "convergence.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved convergence benchmark results to {out_file}")


if __name__ == "__main__":
    main()
