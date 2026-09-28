"""
exp_repeated_eval.py
--------------------
REPEATED-EVALUATION BASELINE & FAIR DEPLOYMENT-TIME BUDGET .

Addresses the reviewer question: "Could repeated sampling of a candidate solve
the transient/performative problem, i.e., is STS's advantage merely from spending
more time per candidate?"

All methods optimize with CMA-ES on the stochastic serving DES, but differ in how
each candidate is *evaluated*, and each evaluation consumes real deployment time
(simulation steps). Every method is given the SAME total deployment-time budget
B_time (steps); it runs as many candidate deployments as fit in that budget.

  single-shot (N_rep=1) : one short non-settled window (T_SS steps).
  repeated  (N_rep=3,5,10): N_rep short non-settled windows, averaged (N_rep*T_SS steps).
  STS                    : deploy until online settling; score the settled window
                           (steps consumed = settling time, capped at DES_MAX).

Reported per method and per B_time: post-deployment regret, number of candidate
deployments, number of settled/"useful" evaluations, over N_SEEDS seeds.

Saves results/repeated_eval.json.
"""
import os, json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import cma
import fluid as F
import des
import methods as M
import optimize_util as OU
import optimizers as OPT

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)

BETA = 0.60
N_SEEDS = 40
T_SS = 30            # deployment steps for one short (non-settled) window
WARM_SS, WIN_SS = 2, 14   # early transient window (workload not yet settled)
DES_MAX = 200        # cap on a settling deployment
BUDGETS = [1500, 3000, 6000, 12000, 24000]   # total deployment-time budget (steps)
METHODS = ["single-shot", "repeated-3", "repeated-5", "repeated-10", "STS"]


def eval_shortwindow(x, beta, seed):
    d = des.deploy(x, beta, n_steps=T_SS, seed=seed)
    return float(np.mean(d["cost"][WARM_SS:WARM_SS + WIN_SS])), T_SS, False


def eval_repeated(x, beta, seed, n_rep):
    vals = []
    for j in range(n_rep):
        d = des.deploy(x, beta, n_steps=T_SS, seed=seed * 100 + j)
        vals.append(float(np.mean(d["cost"][WARM_SS:WARM_SS + WIN_SS])))
    return float(np.mean(vals)), T_SS * n_rep, False


def eval_sts(x, beta, seed):
    d = des.deploy(x, beta, n_steps=DES_MAX, seed=seed)
    t_s = M.detect_settling(d["lam"], method="plateau")
    censored = t_s >= DES_MAX
    t_eff = min(t_s, DES_MAX - 1)
    steps = min(t_s, DES_MAX)  # deployment time consumed until settling
    post = d["cost"][t_eff:] if t_eff < DES_MAX - WARM_SS else d["cost"][-WARM_SS:]
    return float(np.mean(post)), int(steps), (not censored)


def budget_limited_cma(eval_fn, beta, budget_steps, seed):
    """Run CMA-ES until the cumulative deployment time reaches budget_steps."""
    es = cma.CMAEvolutionStrategy(4 * [0.5], 0.25,
                                  {"bounds": [0, 1], "maxfevals": 100000,
                                   "seed": seed + 1, "verbose": -9, "popsize": 8})
    used = 0
    n_dep = 0
    n_useful = 0
    best_u, best_f = None, np.inf
    ev = 0
    # Evaluate a full population per generation (CMA needs >= mu solutions);
    # stop at the generation boundary once the budget is spent.
    while not es.stop() and used < budget_steps:
        cands = es.ask()
        vals = []
        for u in cands:
            x = OPT.to_cfg(u)
            f, steps, useful = eval_fn(x, beta, seed * 1000 + ev)
            ev += 1
            used += steps
            n_dep += 1
            n_useful += int(useful)
            vals.append(f + np.random.default_rng(seed * 7 + ev).normal() * 0.01)
            if f < best_f:
                best_f, best_u = f, np.array(u)
        es.tell(cands, vals)
    return OPT.to_cfg(best_u), n_dep, n_useful


def run_one(args):
    method, budget, seed, c_opt = args
    if method == "single-shot":
        fn = eval_shortwindow
    elif method.startswith("repeated-"):
        nr = int(method.split("-")[1])
        fn = lambda x, b, s: eval_repeated(x, b, s, nr)
    else:
        fn = eval_sts
    cfg, n_dep, n_useful = budget_limited_cma(fn, BETA, budget, seed)
    regret = max(0.0, float(F.true_cost(cfg, BETA)) - c_opt)
    return method, budget, regret, n_dep, n_useful


def main():
    c_opt = OU.find_optimum(BETA)[1]
    print(f"Repeated-eval / fair-budget at beta={BETA}, c_opt={c_opt:.3f}")
    tasks = [(m, b, s, c_opt) for m in METHODS for b in BUDGETS for s in range(N_SEEDS)]
    acc = {(m, b): {"regret": [], "ndep": [], "nuseful": []} for m in METHODS for b in BUDGETS}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for i, (m, b, reg, nd, nu) in enumerate(pool.map(run_one, tasks)):
            acc[(m, b)]["regret"].append(reg)
            acc[(m, b)]["ndep"].append(nd)
            acc[(m, b)]["nuseful"].append(nu)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(tasks)}", flush=True)

    rows = []
    for m in METHODS:
        for b in BUDGETS:
            a = acc[(m, b)]
            reg = np.array(a["regret"])
            rows.append(dict(method=m, budget=b,
                             regret_mean=float(np.mean(reg)),
                             regret_sem=float(np.std(reg) / np.sqrt(len(reg))),
                             regret_median=float(np.median(reg)),
                             deployments_mean=float(np.mean(a["ndep"])),
                             useful_evals_mean=float(np.mean(a["nuseful"]))))
            print(f"  {m:12s} B={b:6d}: regret={np.mean(reg):.4f} deploys={np.mean(a['ndep']):.0f} "
                  f"useful={np.mean(a['nuseful']):.0f}")

    out = dict(beta=BETA, budgets=BUDGETS, methods=METHODS, n_seeds=N_SEEDS,
               c_opt=c_opt, T_ss=T_SS, des_max=DES_MAX, rows=rows)
    with open(os.path.join(OUT_DIR, "repeated_eval.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved repeated_eval.json")


if __name__ == "__main__":
    main()
