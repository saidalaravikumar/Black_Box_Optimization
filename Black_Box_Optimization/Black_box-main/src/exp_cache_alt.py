"""
exp_cache_alt.py
----------------
ALTERNATIVE CACHE BROWSING-FEEDBACK FORMS .

The primary cache testbed (cache.py) uses a LINEAR browsing response:
    s* = S0 - beta_c * DELTA_S * (L_ref - L)/L_ref .
Here we add two alternative response laws and check whether the qualitative
performativity result (myopic tuner over-prefetches; STS throttles prefetch; a
performativity gap emerges) is robust to the functional form:

    saturating : s* = S0 - beta_c * DELTA_S * tanh(2*(L_ref - L)/L_ref)
    threshold  : s* = S0 - beta_c * DELTA_S * relu((L_ref - L)/L_ref - 0.15)/(1-0.15)

The existing linear mechanism remains the primary model. We recompute, per model
and per beta_c, the true optimal prefetch p*, the myopic prefetch p_m, and the
performativity gap.

Saves results/cache_alt.json.
"""
import os, json
import numpy as np
import cache as Ca

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = ["linear", "saturating", "threshold"]


def response(u, model):
    """u = (L_ref - L)/L_ref, the normalized speedup; returns the skew reduction factor."""
    if model == "linear":
        return u
    if model == "saturating":
        return np.tanh(2.0 * u)
    if model == "threshold":
        return np.maximum(0.0, u - 0.15) / (1.0 - 0.15)
    raise ValueError(model)


def settle_skew(y, beta_c, model, iters=250, tol=1e-10):
    y = Ca._clip(y)
    s = Ca.S0
    for _ in range(iters):
        h = Ca.hit_ratio(y, s)
        L = h * Ca.L_HIT + (1.0 - h) * Ca.L_MISS
        u = (Ca.L_REF - L) / Ca.L_REF
        s_new = float(np.clip(Ca.S0 - beta_c * Ca.DELTA_S * response(u, model), 0.4, 2.0))
        if abs(s_new - s) < tol:
            s = s_new; break
        s = 0.5 * s + 0.5 * s_new
    h = Ca.hit_ratio(y, s)
    L = h * Ca.L_HIT + (1.0 - h) * Ca.L_MISS
    return s, h, L


def cost_at(y, s, beta_c):
    C, p, ttl = Ca._clip(y)
    h = Ca.hit_ratio(y, s)
    L = h * Ca.L_HIT + (1.0 - h) * Ca.L_MISS
    bw = p * (1.0 - Ca.pref_conv(s))
    return float(Ca.W_MISS * (1.0 - h) + Ca.W_BW * Ca.BW_COST * bw + Ca.W_LATC * L)


def true_cost(y, beta_c, model):
    s, _, _ = settle_skew(y, beta_c, model)
    return cost_at(y, s, beta_c)


def make_grid(n=25):
    ax = [np.linspace(lo, hi, n) for lo, hi in Ca.BOUNDS]
    G = np.meshgrid(*ax, indexing="ij")
    return np.column_stack([g.ravel() for g in G])


def main():
    grid = make_grid(25)
    betas_c = np.round(np.linspace(0.0, 0.65, 14), 3)
    s_inc0 = None
    out_rows = []
    for model in MODELS:
        for bc in betas_c:
            bc = float(bc)
            # myopic incumbent-induced skew
            s_inc, _, _ = settle_skew(Ca.INCUMBENT, bc, model)
            # true optimum via grid
            tc = np.array([true_cost(y, bc, model) for y in grid])
            io = int(np.argmin(tc)); y_opt = grid[io]; c_opt = float(tc[io])
            # myopic optimum: minimize cost under frozen incumbent skew
            myo = np.array([cost_at(y, s_inc, bc) for y in grid])
            im = int(np.argmin(myo)); y_myo = grid[im]
            c_myo_real = true_cost(y_myo, bc, model)
            out_rows.append(dict(model=model, beta_c=bc, p_opt=float(y_opt[1]),
                                 p_myopic=float(y_myo[1]),
                                 perf_gap=float(c_myo_real - c_opt),
                                 skew_settled=float(settle_skew(y_opt, bc, model)[0])))
        r_hi = out_rows[-1]
        print(f"  {model:11s} beta_c=0.65: p_opt={r_hi['p_opt']:.2f} p_myo={r_hi['p_myopic']:.2f} gap={r_hi['perf_gap']:.3f}")

    out = dict(models=MODELS, betas_c=[float(b) for b in betas_c], rows=out_rows)
    with open(os.path.join(OUT_DIR, "cache_alt.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved cache_alt.json")


if __name__ == "__main__":
    main()
