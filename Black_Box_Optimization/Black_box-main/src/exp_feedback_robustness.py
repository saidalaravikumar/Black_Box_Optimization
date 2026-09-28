"""
exp_feedback_robustness.py
--------------------------
FEEDBACK-MODEL ROBUSTNESS / MODEL MISMATCH .

Tests whether STS remains effective when the exact feedback law differs from the
nominal (baseline retry) model. For each of four behavioral response laws
(baseline, concave, sigmoid, threshold; see feedback_models.py) we compare, in
post-deployment regret across 100... (reduced) Monte Carlo seeds:
  - CMA-ES / single-shot
  - BO-GP / single-shot
  - TUNA-filter / single-shot
  - Proposed: STS CMA-ES
  - Proposed: Stability-Reg. STS
The performative optimum for each (model, beta) is computed by the same
deterministic mean-field multi-start global optimizer used elsewhere.

Saves results/feedback_robustness.json.
"""
import os, json
import numpy as np
from scipy.stats import wilcoxon
from concurrent.futures import ProcessPoolExecutor
import fluid as F
import feedback_models as FB
import methods as M
import optimizers as OPT

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)

BETAS = [0.0, 0.30, 0.60, 0.80]
N_SEEDS = 40
BUDGET = 180
X0 = F.round_cfg(F.X0)


def make_eval(kind, beta, model, stability=False, seed0=0):
    rng = np.random.default_rng(seed0 + 1)

    def evaluate(u):
        x = OPT.to_cfg(u)
        if kind == "single":
            fit = FB.myopic_cost(x, X0, beta, model)
        else:
            fit = FB.true_cost(x, beta, model)
            if stability:
                fit += 1.5 * M.stability_residual(x, beta, n=24, seed=int(rng.integers(1e6)))
        return float(fit + rng.normal() * 0.02)
    return evaluate


def run_seed(args):
    model, beta, seed, c_opt = args
    e_s = make_eval("single", beta, model, seed0=seed)
    e_sts = make_eval("sts", beta, model, stability=False, seed0=seed)
    e_stab = make_eval("sts", beta, model, stability=True, seed0=seed)

    cfg_cma_s, _ = OPT.run_cma(e_s, budget=BUDGET, seed=seed)
    cfg_tuna_s, _ = OPT.run_tuna_filter(e_s, budget=BUDGET, seed=seed, samples_per_eval=4)
    cfg_cma_sts, _ = OPT.run_cma(e_sts, budget=BUDGET, seed=seed)
    cfg_cma_stab, _ = OPT.run_cma(e_stab, budget=BUDGET, seed=seed)

    costs = {
        "CMA-ES / single-shot": FB.true_cost(cfg_cma_s, beta, model),
        "TUNA-filter / single-shot": FB.true_cost(cfg_tuna_s, beta, model),
        "Proposed: STS CMA-ES": FB.true_cost(cfg_cma_sts, beta, model),
        "Proposed: Stability-Reg. STS": FB.true_cost(cfg_cma_stab, beta, model),
    }
    return model, beta, {k: max(0.0, v - c_opt) for k, v in costs.items()}


def main():
    print(f"Feedback robustness: {len(FB.MODELS)} models x {len(BETAS)} betas x {N_SEEDS} seeds")
    opt_costs = {}
    for m in FB.MODELS:
        for b in BETAS:
            opt_costs[(m, b)] = FB.find_optimum(b, m)[1]
    tasks = [(m, b, s, opt_costs[(m, b)]) for m in FB.MODELS for b in BETAS for s in range(N_SEEDS)]
    print(f"Total runs: {len(tasks)}")

    acc = {(m, b): {} for m in FB.MODELS for b in BETAS}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for i, (m, b, regs) in enumerate(pool.map(run_seed, tasks)):
            for meth, r in regs.items():
                acc[(m, b)].setdefault(meth, []).append(r)
            if (i + 1) % 120 == 0:
                print(f"  {i+1}/{len(tasks)}", flush=True)

    rows = []
    for m in FB.MODELS:
        for b in BETAS:
            for meth, arr in acc[(m, b)].items():
                arr = np.array(arr)
                rows.append(dict(model=m, beta=float(b), method=meth,
                                 regret_mean=float(np.mean(arr)),
                                 regret_sem=float(np.std(arr) / np.sqrt(len(arr))),
                                 regret_median=float(np.median(arr))))
            # paired Wilcoxon STS vs CMA single-shot
            a = np.array(acc[(m, b)]["CMA-ES / single-shot"])
            c = np.array(acc[(m, b)]["Proposed: STS CMA-ES"])
            try:
                p = float(wilcoxon(a, c, alternative="greater").pvalue) if not np.all(a == c) else 1.0
            except Exception:
                p = 1.0
            rows.append(dict(model=m, beta=float(b), method="_wilcoxon_sts_vs_cma",
                             p_value=p, frac_single_worse=float(np.mean(a > c))))
            print(f"  {m:10s} b={b:.2f}: single={np.mean(acc[(m,b)]['CMA-ES / single-shot']):.3f} "
                  f"STS={np.mean(acc[(m,b)]['Proposed: STS CMA-ES']):.4f}")

    out = dict(models=FB.MODELS, betas=BETAS, n_seeds=N_SEEDS,
               opt_costs={f"{m}|{b}": v for (m, b), v in opt_costs.items()}, rows=rows)
    with open(os.path.join(OUT_DIR, "feedback_robustness.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved feedback_robustness.json")


if __name__ == "__main__":
    main()
