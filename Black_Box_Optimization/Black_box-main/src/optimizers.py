"""
optimizers.py
-------------
Standard and state-of-the-art black-box optimizers coupled with two distinct
evaluation semantics:
  1. Single-Shot Semantics: Evaluates candidates on frozen incumbent trace (myopic).
  2. Settle-Then-Score (STS) Semantics: Deploys candidate, permits client population
     relaxation to steady-state equilibrium, and attributes background load drift.

Methods supported:
  - Random Search (Single-Shot / STS)
  - CMA-ES (Single-Shot / STS / Stability-Regularized)
  - BO-GP: Gaussian Process Bayesian Optimization with Expected Improvement (Guindani et al., IEEE TCC 2024)
  - TUNA-filter: Repeated multi-sample variance filtering on static trace (Freischuetz et al., ACM EuroSys 2025)
  - NSGA-II: Multi-objective & single-objective evolutionary search (Deb et al., TEVC)
"""

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
import cma
import fluid as F
import methods as M


def to_cfg(u):
    """Maps normalized vector u in [0, 1]^4 to bounded physical configuration."""
    u = np.clip(np.asarray(u, float), 0.0, 1.0)
    x = F.BOUNDS[:, 0] + u * (F.BOUNDS[:, 1] - F.BOUNDS[:, 0])
    return F.round_cfg(x)


def from_cfg(x):
    """Normalizes physical configuration to unit hypercube."""
    return (np.asarray(x, float) - F.BOUNDS[:, 0]) / (F.BOUNDS[:, 1] - F.BOUNDS[:, 0])


def make_eval_fluid(kind, beta, incumbent, stability=False, noise=0.02, seed0=0):
    """
    Constructs fitness evaluation callback for optimizers.
    Gaussian observation noise is added to reflect realistic telemetry uncertainty.
    """
    rng = np.random.default_rng(seed0 + 1)

    def evaluate(u):
        x = to_cfg(u)
        if kind == "single":
            fitness = F.myopic_cost(x, incumbent, beta)
        else:
            fitness = F.true_cost(x, beta)
            if stability:
                # Add stability residual regularization
                fitness += 1.5 * M.stability_residual(x, beta, n=24, seed=int(rng.integers(1e6)))
        return float(fitness + rng.normal() * noise)

    return evaluate


def run_cma(evalf, budget=180, seed=0, sigma=0.25, track_history=False, true_evalf=None):
    """Runs Covariance Matrix Adaptation Evolution Strategy (CMA-ES)."""
    es = cma.CMAEvolutionStrategy(
        4 * [0.5], sigma,
        {"bounds": [0, 1], "maxfevals": budget, "seed": seed + 1,
         "verbose": -9, "popsize": 8}
    )
    best_u, best_f = None, np.inf
    history = []
    fevals = 0

    while not es.stop():
        candidates = es.ask()
        values = [evalf(u) for u in candidates]
        es.tell(candidates, values)
        for u, v in zip(candidates, values):
            fevals += 1
            if v < best_f:
                best_f, best_u = v, np.array(u)
            if track_history:
                cur_metric = float(true_evalf(best_u)) if true_evalf is not None else float(best_f)
                history.append((fevals, cur_metric))
            if fevals >= budget:
                break
        if fevals >= budget:
            break

    if track_history:
        return to_cfg(best_u), float(best_f), history
    return to_cfg(best_u), float(best_f)


def run_random(evalf, budget=180, seed=0, track_history=False, true_evalf=None):
    """Uniform random sampling baseline across configuration bounds."""
    rng = np.random.default_rng(seed)
    best_u, best_f = None, np.inf
    history = []
    for i in range(1, budget + 1):
        u = rng.random(4)
        val = evalf(u)
        if val < best_f:
            best_f, best_u = val, u
        if track_history:
            cur_metric = float(true_evalf(best_u)) if true_evalf is not None else float(best_f)
            history.append((i, cur_metric))
    if track_history:
        return to_cfg(best_u), float(best_f), history
    return to_cfg(best_u), float(best_f)


def run_bo_gp(evalf, budget=180, seed=0, n_init=12, track_history=False, true_evalf=None):
    """
    Gaussian Process Bayesian Optimization with Expected Improvement (EI).
    Represents state-of-the-art cloud autotuners (e.g. Guindani et al., IEEE TCC 2024).
    """
    rng = np.random.default_rng(seed)
    # Initial quasi-random uniform sample
    X_obs = list(rng.random((n_init, 4)))
    y_obs = [evalf(u) for u in X_obs]

    best_idx = int(np.argmin(y_obs))
    best_f = y_obs[best_idx]
    best_u = X_obs[best_idx]
    history = []
    if track_history:
        for i in range(1, n_init + 1):
            cur_best_u = X_obs[int(np.argmin(y_obs[:i]))]
            cur_metric = float(true_evalf(cur_best_u)) if true_evalf is not None else float(min(y_obs[:i]))
            history.append((i, cur_metric))

    kernel = Matern(nu=2.5, length_scale=0.3) + WhiteKernel(noise_level=1e-3)
    gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, random_state=seed)

    for step in range(n_init + 1, budget + 1):
        X_arr = np.array(X_obs)
        y_arr = np.array(y_obs)
        gp.fit(X_arr, y_arr)

        # Candidate pool for Expected Improvement
        candidates = rng.random((100, 4))
        mu, sigma = gp.predict(candidates, return_std=True)
        sigma = np.maximum(sigma, 1e-6)

        # Expected Improvement for minimization: EI = (y_best - mu) * Phi(z) + sigma * phi(z)
        improvement = best_f - mu
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)

        next_u = candidates[np.argmax(ei)]
        next_val = evalf(next_u)

        X_obs.append(next_u)
        y_obs.append(next_val)

        if next_val < best_f:
            best_f = next_val
            best_u = next_u

        if track_history:
            cur_metric = float(true_evalf(best_u)) if true_evalf is not None else float(best_f)
            history.append((step, cur_metric))

    if track_history:
        return to_cfg(best_u), float(best_f), history
    return to_cfg(best_u), float(best_f)


def run_tuna_filter(evalf_raw, budget=180, seed=0, samples_per_eval=4, track_history=False, true_evalf=None):
    """
    TUNA-style repeated multi-sampling variance filtering (EuroSys 2025).
    Filters measurement noise by averaging repeated samples at each configuration,
    but evaluates under static trace-replay assumptions without settling.
    """
    effective_budget = max(10, budget // samples_per_eval)

    def eval_filtered(u):
        vals = [evalf_raw(u) for _ in range(samples_per_eval)]
        return float(np.mean(vals))

    res = run_cma(eval_filtered, budget=effective_budget, seed=seed, track_history=track_history, true_evalf=true_evalf)
    if track_history:
        cfg, best_f, hist = res
        scaled_hist = [(min(budget, step * samples_per_eval), val) for step, val in hist]
        return cfg, best_f, scaled_hist
    return res


def run_nsga(evalf, budget=180, seed=0, track_history=False):
    """
    Genetic Algorithm baseline (single-objective GA using tournament selection,
    blend crossover, and polynomial mutation).
    """
    rng = np.random.default_rng(seed)
    pop_size = 12
    n_gen = max(1, budget // pop_size)
    pop = rng.random((pop_size, 4))
    fitnesses = [evalf(u) for u in pop]

    best_idx = int(np.argmin(fitnesses))
    best_f = fitnesses[best_idx]
    best_u = pop[best_idx]
    history = []
    fevals = pop_size

    if track_history:
        for i in range(1, pop_size + 1):
            history.append((i, float(min(fitnesses[:i]))))

    for gen in range(n_gen - 1):
        next_pop = []
        for _ in range(pop_size):
            i1, i2 = rng.choice(pop_size, 2, replace=False)
            parent1 = pop[i1] if fitnesses[i1] < fitnesses[i2] else pop[i2]
            i3, i4 = rng.choice(pop_size, 2, replace=False)
            parent2 = pop[i3] if fitnesses[i3] < fitnesses[i4] else pop[i4]

            alpha = 0.3
            gamma = (1 + 2 * alpha) * rng.random() - alpha
            child = gamma * parent1 + (1 - gamma) * parent2

            if rng.random() < 0.3:
                child += rng.normal(0, 0.1, size=4)
            child = np.clip(child, 0.0, 1.0)
            next_pop.append(child)

        pop = np.array(next_pop)
        fitnesses = []
        for u in pop:
            val = evalf(u)
            fitnesses.append(val)
            fevals += 1
            if val < best_f:
                best_f, best_u = val, np.array(u)
            if track_history:
                history.append((fevals, float(best_f)))
            if fevals >= budget:
                break
        if fevals >= budget:
            break

    if track_history:
        return to_cfg(best_u), float(best_f), history
    return to_cfg(best_u), float(best_f)


def run_nsga_frontier(beta, budget=2500, seed=0):
    """
    Multi-objective evolutionary optimization (NSGA-II) mapping the
    Pareto frontier between performative operating cost and stability residual.
    """
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize as pymoo_min
    from pymoo.operators.sampling.rnd import FloatRandomSampling

    class PerfMOO(Problem):
        def __init__(self):
            super().__init__(n_var=4, n_obj=2, xl=0.0, xu=1.0)

        def _evaluate(self, U, out, *a, **k):
            X = np.array([to_cfg(u) for u in U])
            lam, _, _ = F.settle_vec(X, beta)
            cost = F.cost_vec(X, lam, beta)
            resids = np.array([M.stability_residual(x, beta, n=20, seed=seed) for x in X])
            out["F"] = np.column_stack([cost, resids])

    algo = NSGA2(pop_size=40, sampling=FloatRandomSampling())
    res = pymoo_min(PerfMOO(), algo, ("n_gen", max(1, budget // 40)),
                    seed=seed + 1, verbose=False)
    F_front = res.F
    X_front = np.array([to_cfg(u) for u in res.X])
    return X_front, F_front
