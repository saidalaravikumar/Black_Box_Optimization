"""
exp_ablation.py
---------------
STS COMPONENT ABLATION .

Isolates the contribution of each Settle-Then-Score (STS) component by
optimizing with a black-box optimizer (CMA-ES) whose fitness callback is one of
five ablation variants of the STS evaluation semantics, on the stochastic
discrete-event serving simulator (des.py). To give every component something to
act on, each evaluation is run under a mild exogenous linear load drift injected
during tuning (drift_mag), reflecting realistic non-stationary tuning windows.

Variants:
  A. Single-Shot            : short early window, no settling, no drift corr., no stability.
  B. STS-Settle             : online settling detection, score on settled window.
  C. STS-Settle+Drift       : B + dual-replica control differencing.
  D. STS-Settle+Stability   : B + stability-residual regularization (gamma).
  E. Full STS               : settling + drift correction + stability.

Reported per (variant, beta): post-deployment regret P(x,x)-c_opt, settling time,
number of candidate deployments, corrected-evaluation error |fitness - P(x,x)|,
final stability residual, with bootstrap 95% CIs.

Protocol: betas {0.0,0.3,0.6,0.8}; N_SEEDS paired seeds; CMA-ES budget BUDGET;
DES horizon DES_BUDGET; drift_mag DRIFT. Same seeds reused across variants (paired).
Saves results/ablation.json.
"""
import os, json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import fluid as F
import des
import methods as M
import optimize_util as OU
import optimizers as OPT

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)

BETAS = [0.0, 0.30, 0.60, 0.80]
N_SEEDS = 40
BUDGET = 40          # CMA-ES evaluations per run
DES_BUDGET = 180     # DES horizon per candidate deployment
WARM = 25
WINDOW = 25
DRIFT = 0.08         # mild +8% exogenous linear load surge during the tuning window
GAMMA = 1.5          # stability-regularization weight (same scale as optimizers.py)
CONTROL_CFG = np.array([30.0, 2.5, 0.0, 120.0])  # stable reference replica (matches exp_settling)
VARIANTS = ["Single-Shot", "STS-Settle", "STS-Settle+Drift", "STS-Settle+Stability", "Full STS"]


X0 = F.round_cfg(F.X0)


def _ablation_eval(x, beta, variant, seed, diag):
    """Return fitness under a given ablation variant. diag collects deployment count,
    settling time, and evaluation error for reporting."""
    settle = variant != "Single-Shot"
    drift_corr = variant in ("STS-Settle+Drift", "Full STS")
    stability = variant in ("STS-Settle+Stability", "Full STS")

    if not settle:
        # Single-Shot (myopic): candidate scored on the FROZEN incumbent-induced
        # workload D(x0); the candidate is never deployed, so no settling and the
        # transient/performative bias is fully present. (paper single-shot semantics)
        c_score = F.myopic_cost(x, X0, beta)
        t_s = 0.0
    else:
        # Deploy candidate under exogenous drift; settle then score.
        d = des.deploy(x, beta, n_steps=DES_BUDGET, drift=DRIFT, seed=seed)
        diag["deploys"] += 1
        diag["steps"] += DES_BUDGET
        t_s = M.detect_settling(d["lam"], method="plateau")
        t_eff = min(t_s, DES_BUDGET - 1)
        post = d["cost"][t_eff:] if t_eff < DES_BUDGET - WARM else d["cost"][-WARM:]
        c_score = float(np.mean(post))
        if drift_corr:
            dc = des.deploy(CONTROL_CFG, beta, n_steps=DES_BUDGET, drift=DRIFT, seed=seed + 777)
            diag["deploys"] += 1
            diag["steps"] += DES_BUDGET
            # Temporal control differencing (paper Eq. drift): control's own late-minus-early
            # change measures the exogenous drift, config-agnostically.
            drift_bias = float(np.mean(dc["cost"][t_eff:]) - np.mean(dc["cost"][:WARM]))
            c_score -= drift_bias

    diag["t_s_sum"] += min(t_s, DES_BUDGET)
    diag["n_eval"] += 1

    # Evaluation-error diagnostic: how far the scored fitness is from the
    # drift-free realized performative cost of this candidate.
    true_c = F.true_cost(x, beta)
    diag["everr_sum"] += abs(c_score - true_c)

    fitness = c_score
    if stability:
        fitness += GAMMA * M.stability_residual(x, beta, n=24, seed=seed + 13)
    return fitness


def run_one(args):
    variant, beta, seed, c_opt = args
    rng_seed = seed
    diag = dict(deploys=0, steps=0, t_s_sum=0.0, n_eval=0, everr_sum=0.0)

    def evalf(u):
        x = OPT.to_cfg(u)
        return _ablation_eval(x, beta, variant, rng_seed, diag)

    cfg, _ = OPT.run_cma(evalf, budget=BUDGET, seed=seed, sigma=0.25)
    true_cost = float(F.true_cost(cfg, beta))
    regret = max(0.0, true_cost - c_opt)
    resid = float(M.stability_residual(cfg, beta, n=48, seed=seed + 5))
    mean_ts = diag["t_s_sum"] / max(1, diag["n_eval"])
    mean_everr = diag["everr_sum"] / max(1, diag["n_eval"])
    return (variant, beta, regret, mean_ts, diag["deploys"], mean_everr, resid)


def bootstrap_ci(arr, n_boot=2000, seed=0):
    arr = np.asarray(arr, float)
    rng = np.random.default_rng(seed)
    means = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    print(f"STS ablation: {len(VARIANTS)} variants x {len(BETAS)} betas x {N_SEEDS} seeds")
    # true optimum per beta
    opt_costs = {b: OU.find_optimum(b)[1] for b in BETAS}
    tasks = [(v, b, s, opt_costs[b]) for v in VARIANTS for b in BETAS for s in range(N_SEEDS)]
    print(f"Total ablation runs: {len(tasks)}")

    acc = {(v, b): {"regret": [], "ts": [], "deploys": [], "everr": [], "resid": []}
           for v in VARIANTS for b in BETAS}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for i, (v, b, reg, ts, dep, ee, rs) in enumerate(pool.map(run_one, tasks)):
            a = acc[(v, b)]
            a["regret"].append(reg); a["ts"].append(ts); a["deploys"].append(dep)
            a["everr"].append(ee); a["resid"].append(rs)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(tasks)} done", flush=True)

    rows = []
    for v in VARIANTS:
        for b in BETAS:
            a = acc[(v, b)]
            reg = np.array(a["regret"])
            lo, hi = bootstrap_ci(reg, seed=1)
            rows.append(dict(
                variant=v, beta=float(b),
                regret_mean=float(np.mean(reg)),
                regret_sem=float(np.std(reg) / np.sqrt(len(reg))),
                regret_ci_lo=lo, regret_ci_hi=hi,
                regret_median=float(np.median(reg)),
                settle_time_mean=float(np.mean(a["ts"])),
                deployments_mean=float(np.mean(a["deploys"])),
                eval_error_mean=float(np.mean(a["everr"])),
                eval_error_sem=float(np.std(a["everr"]) / np.sqrt(len(a["everr"]))),
                stability_residual_mean=float(np.mean(a["resid"])),
            ))
            print(f"  {v:22s} b={b:.2f}: regret={np.mean(reg):.4f} "
                  f"ts={np.mean(a['ts']):.0f} everr={np.mean(a['everr']):.3f} "
                  f"resid={np.mean(a['resid']):.3f}")

    out = dict(betas=BETAS, variants=VARIANTS, n_seeds=N_SEEDS, budget=BUDGET,
               des_budget=DES_BUDGET, drift=DRIFT, gamma=GAMMA, opt_costs={str(k): v for k, v in opt_costs.items()},
               rows=rows)
    with open(os.path.join(OUT_DIR, "ablation.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved ablation.json")


if __name__ == "__main__":
    main()
