"""
exp_scalability.py
------------------
CONFIGURATION-SPACE DIMENSIONALITY / SCALABILITY .

Evaluates how optimization behaves as the configuration space grows from the
original d=4 to d in {4,8,16,32}, using the scalable model (scalable.py): the
first four knobs drive the performative loop; the extra knobs are interpretable
static secondary parameters (mixed integer/continuous). Evaluation budget scales
with dimension (BUDGET_PER_DIM * d).

Compares, in post-deployment regret at beta in {0.6,0.8}:
  CMA-ES / single-shot, BO-GP / single-shot, TUNA-filter / single-shot,
  NSGA-II / single-shot, Proposed STS CMA-ES, Proposed Stability-Reg. STS.

Saves results/scalability.json.
"""
import os, json
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from concurrent.futures import ProcessPoolExecutor
import cma
import scalable as SC

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)

DIMS = [4, 8, 16, 32]
BETAS = [0.60, 0.80]
N_SEEDS = 10
BUDGET_PER_DIM = 20
GP_WINDOW = 70    # rolling window for BO-GP tractability at high d
BO_CAP = 100      # BO-GP evaluation cap (its myopic regret floor is budget-independent)


def cma_d(evalf, d, budget, seed, sigma=0.25):
    es = cma.CMAEvolutionStrategy(d * [0.5], sigma,
                                  {"bounds": [0, 1], "maxfevals": budget,
                                   "seed": seed + 1, "verbose": -9,
                                   "popsize": 4 + int(3 * np.log(d))})
    best_u, best_f, fe = None, np.inf, 0
    while not es.stop():
        C = es.ask(); V = [evalf(u) for u in C]; es.tell(C, V)
        for u, v in zip(C, V):
            fe += 1
            if v < best_f:
                best_f, best_u = v, np.array(u)
            if fe >= budget:
                break
        if fe >= budget:
            break
    return best_u


def bo_d(evalf, d, budget, seed, n_init=None):
    rng = np.random.default_rng(seed)
    budget = min(budget, BO_CAP)
    n_init = n_init or min(2 * d, 24)
    X = list(rng.random((n_init, d))); y = [evalf(u) for u in X]
    bi = int(np.argmin(y)); best_u, best_f = X[bi], y[bi]
    kernel = Matern(nu=2.5, length_scale=0.3) + WhiteKernel(noise_level=1e-3)
    gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, random_state=seed)
    for step in range(n_init, budget):
        Xa = np.array(X[-GP_WINDOW:]); ya = np.array(y[-GP_WINDOW:])
        gp.fit(Xa, ya)
        cand = rng.random((100, d))
        mu, sig = gp.predict(cand, return_std=True); sig = np.maximum(sig, 1e-6)
        imp = best_f - mu; z = imp / sig
        ei = imp * norm.cdf(z) + sig * norm.pdf(z)
        nu = cand[int(np.argmax(ei))]; nv = evalf(nu)
        X.append(nu); y.append(nv)
        if nv < best_f:
            best_f, best_u = nv, nu
    return best_u


def tuna_d(evalf, d, budget, seed, spe=4):
    eff = max(10, budget // spe)
    ef = lambda u: float(np.mean([evalf(u) for _ in range(spe)]))
    return cma_d(ef, d, eff, seed)


def nsga_d(evalf, d, budget, seed):
    rng = np.random.default_rng(seed)
    ps = 12; ng = max(1, budget // ps)
    pop = rng.random((ps, d)); fit = [evalf(u) for u in pop]
    bi = int(np.argmin(fit)); best_u, best_f = pop[bi], fit[bi]; fe = ps
    for _ in range(ng - 1):
        nxt = []
        for _ in range(ps):
            i1, i2 = rng.choice(ps, 2, replace=False)
            p1 = pop[i1] if fit[i1] < fit[i2] else pop[i2]
            i3, i4 = rng.choice(ps, 2, replace=False)
            p2 = pop[i3] if fit[i3] < fit[i4] else pop[i4]
            a = 0.3; g = (1 + 2 * a) * rng.random() - a
            ch = g * p1 + (1 - g) * p2
            if rng.random() < 0.3:
                ch += rng.normal(0, 0.1, size=d)
            nxt.append(np.clip(ch, 0, 1))
        pop = np.array(nxt); fit = []
        for u in pop:
            v = evalf(u); fit.append(v); fe += 1
            if v < best_f:
                best_f, best_u = v, np.array(u)
            if fe >= budget:
                break
        if fe >= budget:
            break
    return best_u


def run_cell(args):
    method, d, beta, seed, c_opt = args
    budget = BUDGET_PER_DIM * d
    rng = np.random.default_rng(seed + 1)
    single = method.endswith("single")
    x0 = SC.to_cfg(SC._norm(np.concatenate([[8.0, 0.25, 3.0, 200.0],
                    [SC.bounds(d)[4 + j].mean() for j in range(d - 4)]]), d), d)

    def evalf(u):
        x = SC.to_cfg(u, d)
        if single:
            v = SC.myopic_cost(x, x0, beta, d)
        else:
            v = SC.true_cost(x, beta, d)
            if method == "stab":
                v += 1.5 * SC.stability_residual(x, beta, d, n=24, seed=int(rng.integers(1e6)))
        return float(v + rng.normal() * 0.02)

    runner = {"cma-single": cma_d, "bo-single": bo_d, "tuna-single": tuna_d,
              "nsga-single": nsga_d, "sts": cma_d, "stab": cma_d}[method]
    u = runner(evalf, d, budget, seed)
    cfg = SC.to_cfg(u, d)
    regret = max(0.0, SC.true_cost(cfg, beta, d) - c_opt)
    return method, d, beta, regret


METHODS = ["cma-single", "bo-single", "tuna-single", "nsga-single", "sts", "stab"]


def main():
    print("Scalability across d in", DIMS)
    opt_costs = {(d, b): SC.optimum(b, d) for d in DIMS for b in BETAS}
    tasks = [(m, d, b, s, opt_costs[(d, b)]) for m in METHODS for d in DIMS
             for b in BETAS for s in range(N_SEEDS)]
    print(f"Total runs: {len(tasks)}")
    acc = {(m, d, b): [] for m in METHODS for d in DIMS for b in BETAS}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for i, (m, d, b, reg) in enumerate(pool.map(run_cell, tasks)):
            acc[(m, d, b)].append(reg)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(tasks)}", flush=True)
    rows = []
    for m in METHODS:
        for d in DIMS:
            for b in BETAS:
                a = np.array(acc[(m, d, b)])
                rows.append(dict(method=m, d=d, beta=float(b),
                                 regret_mean=float(np.mean(a)),
                                 regret_sem=float(np.std(a) / np.sqrt(len(a))),
                                 regret_median=float(np.median(a)),
                                 success_rate=float(np.mean(a < 0.05))))
            if m in ("cma-single", "sts", "stab"):
                print(f"  {m:12s} d={d} b=0.6: regret="
                      f"{np.mean(acc[(m,d,0.6)]):.4f}")
    out = dict(dims=DIMS, betas=BETAS, n_seeds=N_SEEDS, budget_per_dim=BUDGET_PER_DIM,
               methods=METHODS, opt_costs={f"{d}|{b}": v for (d, b), v in opt_costs.items()},
               rows=rows)
    with open(os.path.join(OUT_DIR, "scalability.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved scalability.json")


if __name__ == "__main__":
    main()
