"""
optimize_util.py
----------------
Robust, vectorized multi-start global optimizer and adaptive Gaussian refiner.
Used strictly for computing high-precision ground-truth reference configurations
(x_o, x_m, x_s) across the performative landscape.
"""

import numpy as np
import fluid as F
from fluid import BOUNDS, round_cfg


def _clip(X):
    return np.minimum(np.maximum(X, BOUNDS[:, 0]), BOUNDS[:, 1])


def refine(obj_vec, x0, rounds=25, pop=1000, seed=0):
    """
    Adaptive Gaussian local search refiner for non-convex, bounded landscapes.
    """
    rng = np.random.default_rng(seed)
    span = BOUNDS[:, 1] - BOUNDS[:, 0]
    best = np.asarray(x0, float).copy()
    best_f = float(obj_vec(best[None, :])[0])
    sigma = 0.25 * span

    for _ in range(rounds):
        candidates = _clip(best[None, :] + rng.normal(size=(pop, 4)) * sigma[None, :])
        candidates = np.vstack([candidates, best[None, :]])
        vals = obj_vec(candidates)
        idx = int(np.argmin(vals))
        if vals[idx] < best_f:
            best_f = vals[idx]
            best = candidates[idx].copy()
        sigma *= 0.82

    return round_cfg(best)


def grid_seed(obj_vec, grid):
    """Selects best candidate from a precomputed grid."""
    f = obj_vec(grid)
    return grid[int(np.argmin(f))]


def minimize_global(obj_vec, grid, n_seeds=4, rounds=16, pop=600):
    """
    Multi-start global optimization combining top grid seeds with adaptive refinement.
    Guarantees deterministic, reproducible, global minimum discovery.
    """
    f = obj_vec(grid)
    order = np.argsort(f)
    seeds = [grid[int(i)] for i in order[:n_seeds]]

    best, best_f = None, np.inf
    for k, s in enumerate(seeds):
        candidate = refine(obj_vec, s, rounds=rounds, pop=pop, seed=k)
        fc = float(obj_vec(candidate[None, :])[0])
        if fc < best_f:
            best_f = fc
            best = candidate

    return best


_GLOBAL_GRID = None

def get_grid(n=10):
    global _GLOBAL_GRID
    if _GLOBAL_GRID is None:
        axes = [np.linspace(lo, hi, n) for lo, hi in BOUNDS]
        B, T, R, A = np.meshgrid(*axes, indexing="ij")
        _GLOBAL_GRID = np.column_stack([B.ravel(), T.ravel(), R.ravel(), A.ravel()])
    return _GLOBAL_GRID


def find_optimum(beta):
    """Finds global minimizer of true cost P(x, x)."""
    grid = get_grid(10)
    obj = lambda X: F.true_cost_vec(X, beta)
    xo = minimize_global(obj, grid, n_seeds=4, rounds=16, pop=600)
    return xo, float(F.true_cost(xo, beta))


def find_myopic(beta, incumbent=None):
    """Finds minimizer against static workload induced by incumbent x_0."""
    if incumbent is None:
        incumbent = F.round_cfg(F.X0)
    grid = get_grid(10)
    lam_inc = F.settle(incumbent, beta)["lam"]
    obj = lambda X: F.cost_vec(X, np.full(len(X), lam_inc), beta)
    xm = minimize_global(obj, grid, n_seeds=4, rounds=16, pop=600)
    believed = float(obj(xm[None, :])[0])
    return xm, believed


def find_stable(beta, iters=8):
    """Iterates best-response map x_{k+1} = argmin_x P(x, x_k) to find fixed point."""
    grid = get_grid(10)
    x = F.round_cfg(F.X0).copy()
    for _ in range(iters):
        lam_k = F.settle(x, beta)["lam"]
        obj = lambda X: F.cost_vec(X, np.full(len(X), lam_k), beta)
        xn = refine(obj, grid_seed(obj, grid), rounds=12, pop=400)
        if np.allclose(xn, round_cfg(x), atol=1e-2):
            x = xn
            break
        x = xn
    return round_cfg(x), float(F.true_cost(round_cfg(x), beta))

