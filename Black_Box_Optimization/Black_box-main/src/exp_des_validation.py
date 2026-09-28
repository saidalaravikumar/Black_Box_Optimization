"""
exp_des_validation.py  
-------------------------------------
INDEPENDENT DISCRETE-EVENT VALIDATION (reviewer Exp. 1).

The primary optimizer benchmark (exp_optimizers.py) both *optimizes against* and
*scores* candidates with the deterministic mean-field model, so a reviewer may
ask whether the large regret gaps are an artifact of optimizing the very model
used to grade. This experiment removes that concern: every method still searches
using only the telemetry it would have in practice (single-shot = myopic frozen
workload; STS = settle-then-score), but the FINAL selected configuration of each
method is graded by realized post-deployment cost in the *stochastic* discrete-
event simulator (des.py), averaged over independent DES rollout seeds. The
reference optimum x_o is the mean-field global minimizer, and its regret baseline
is likewise its DES-realized cost, so the comparison is DES-vs-DES.

Grid:  beta in {0.30,0.40,0.50,0.60,0.70,0.80};  100 independent search seeds.
Methods: CMA-ES / single-shot, BO-GP / single-shot, NSGA-II / single-shot,
         Proposed STS-CMA-ES.
Reports (per beta, per method): DES post-deployment regret mean, median, SEM,
         95% bootstrap CI, and paired Wilcoxon p vs CMA-ES single-shot.

Saves ../results/des_validation.json.
"""
import os, json
import numpy as np
from scipy.stats import wilcoxon
from concurrent.futures import ProcessPoolExecutor
import fluid as F
import optimize_util as OU
import optimizers as OPT

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)

BETAS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
N_SEEDS = 100
BUDGET = 180
# DES rollout used for realized-cost grading (independent of the search model)
DES_STEPS = 1000
DES_TAIL = 500
DES_ROLLOUTS = (11, 23, 37)   # independent DES seeds averaged for a stable estimate


def des_realized_cost(x, beta):
    """Mean steady-state realized cost of config x in the stochastic DES."""
    import des
    vals = []
    for s in DES_ROLLOUTS:
        d = des.deploy(x, beta, n_steps=DES_STEPS, seed=int(s))
        vals.append(float(np.mean(d["cost"][-DES_TAIL:])))
    return float(np.mean(vals))


def eval_single_seed(args):
    beta, seed, des_opt = args
    incumbent = F.round_cfg(F.X0)
    e_single = OPT.make_eval_fluid("single", beta, incumbent, seed0=seed)
    e_sts = OPT.make_eval_fluid("sts", beta, incumbent, stability=False, seed0=seed)

    cfg_cma_s, _ = OPT.run_cma(e_single, budget=BUDGET, seed=seed)
    cfg_bo_s, _ = OPT.run_bo_gp(e_single, budget=BUDGET, seed=seed, n_init=12)
    cfg_nsga_s, _ = OPT.run_nsga(e_single, budget=BUDGET, seed=seed)
    cfg_sts, _ = OPT.run_cma(e_sts, budget=BUDGET, seed=seed)

    cfgs = {
        "CMA-ES / single-shot": cfg_cma_s,
        "BO-GP / single-shot": cfg_bo_s,
        "NSGA-II / single-shot": cfg_nsga_s,
        "Proposed: STS-CMA-ES": cfg_sts,
    }
    regrets = {k: max(0.0, des_realized_cost(v, beta) - des_opt) for k, v in cfgs.items()}
    return beta, regrets


def bootstrap_ci(a, n_boot=2000, seed=0):
    a = np.asarray(a, float)
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    print(f"DES validation: {len(BETAS)} betas x {N_SEEDS} seeds x 4 methods", flush=True)
    # Reference optimum per beta (mean-field global min) and its DES-realized cost
    des_opt = {}
    for b in BETAS:
        xo, _ = OU.find_optimum(b)
        des_opt[b] = des_realized_cost(xo, b)
        print(f"  beta={b:.2f}: xo={np.round(xo,2)} DES_opt={des_opt[b]:.4f}", flush=True)

    tasks = [(b, s, des_opt[b]) for b in BETAS for s in range(N_SEEDS)]
    acc = {b: {} for b in BETAS}
    with ProcessPoolExecutor() as pool:
        for i, (b, regs) in enumerate(pool.map(eval_single_seed, tasks)):
            for m, r in regs.items():
                acc[b].setdefault(m, []).append(r)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(tasks)} runs done", flush=True)

    rows = []
    for b in BETAS:
        for m, arr in acc[b].items():
            arr = np.array(arr)
            lo, hi = bootstrap_ci(arr, seed=int(b * 100))
            rows.append(dict(
                beta=float(b), method=m,
                regret_mean=float(np.mean(arr)),
                regret_median=float(np.median(arr)),
                regret_sem=float(np.std(arr) / np.sqrt(len(arr))),
                ci_lo=lo, ci_hi=hi,
                regrets=[float(v) for v in arr],
            ))
        # Wilcoxon: STS vs CMA single-shot
        a = np.array(acc[b]["CMA-ES / single-shot"])
        c = np.array(acc[b]["Proposed: STS-CMA-ES"])
        try:
            p = float(wilcoxon(a, c, alternative="greater").pvalue) if not np.all(a == c) else 1.0
        except Exception:
            p = 1.0
        rows.append(dict(beta=float(b), method="_wilcoxon_sts_vs_cma",
                         p_value=p, frac_single_worse=float(np.mean(a > c))))
        print(f"  beta={b:.2f}: CMA={np.mean(acc[b]['CMA-ES / single-shot']):.4f} "
              f"BO={np.mean(acc[b]['BO-GP / single-shot']):.4f} "
              f"NSGA={np.mean(acc[b]['NSGA-II / single-shot']):.4f} "
              f"STS={np.mean(acc[b]['Proposed: STS-CMA-ES']):.4f} (p={p:.1e})", flush=True)

    out = dict(betas=BETAS, n_seeds=N_SEEDS, budget=BUDGET,
               des_steps=DES_STEPS, des_tail=DES_TAIL, des_rollouts=list(DES_ROLLOUTS),
               des_opt={str(round(b, 2)): des_opt[b] for b in BETAS}, rows=rows)
    with open(os.path.join(OUT_DIR, "des_validation.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved des_validation.json")


if __name__ == "__main__":
    main()
