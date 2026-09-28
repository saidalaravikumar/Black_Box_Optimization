"""
exp_thresholds.py
-----------------
THREE DISTINCT beta THRESHOLDS .

For the myopically-selected (aggressive) configuration x_m, computes and
distinguishes three conceptually different feedback thresholds:

  beta*(x)     Sufficient contraction bound: sup_lambda T_x'(lambda) = 1.
               A conservative, configuration-specific *sufficient* condition for a
               unique attracting workload fixed point (Theorem 1).
  beta_dagger  Saddle-node / fold: the multiplier AT the operating fixed point,
               T_x'(lambda*(beta)) = 1. This is the actual bifurcation where the
               stable low-latency branch loses tangential stability.
  beta_collapse Operational collapse of x_m: the beta at which the realized
               performative cost P(x_m,x_m) crosses an operational SLO ceiling.

In general beta*(x) <= beta_dagger <= beta_collapse: violating the contraction
BOUND does not itself prove a fold, and the fold need not coincide with the
operator-visible collapse of a particular configuration.

Saves results/thresholds.json.
"""
import os, json
import numpy as np
import fluid as F

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)

BETAS = np.linspace(0.0, 0.95, 96)


def T_map(lam, x, beta):
    q, _ = F.operating_point(lam, x)
    return F.LAMBDA0 * F.expected_attempts(q, x[2], beta)


def T_prime(lam, x, beta, h=1e-3):
    return (T_map(lam + h, x, beta) - T_map(lam - h, x, beta)) / (2 * h)


def sup_slope(x, beta, lam_hi_mult=2.0, n=400):
    """Sup of the map slope over the NOMINAL operating envelope [lam0, lam_hi_mult*lam0].
    (Restricting to the operating basin avoids pathological slope spikes deep in the
    already-collapsed region; it remains a sufficient contraction bound there.)"""
    lams = np.linspace(F.LAMBDA0, lam_hi_mult * F.LAMBDA0, n)
    return max(T_prime(l, x, beta) for l in lams)


def fixed_point(x, beta):
    return F.settle(x, beta)["lam"]


def cross_one(betas, vals):
    """First beta where vals crosses 1.0 (linear interp); None if never."""
    vals = np.asarray(vals)
    for i in range(1, len(vals)):
        if vals[i - 1] < 1.0 <= vals[i]:
            b0, b1 = betas[i - 1], betas[i]
            v0, v1 = vals[i - 1], vals[i]
            return float(b0 + (1.0 - v0) / (v1 - v0) * (b1 - b0))
    return None


def main():
    # Myopically-selected aggressive configuration (large batch, tight timeout,
    # retries enabled) -- the knife-edge config a trace-replay tuner favors.
    x_m = np.array([28.0, 0.35, 5.0, 300.0])
    print("x_m =", x_m)

    sup_L = []
    mult_fp = []       # multiplier at the operating fixed point
    cost = []
    lam_star = []
    for b in BETAS:
        sup_L.append(sup_slope(x_m, b))
        lstar = fixed_point(x_m, b)
        lam_star.append(lstar / F.LAMBDA0)
        mult_fp.append(T_prime(lstar, x_m, b))
        cost.append(F.true_cost(x_m, b))

    beta_star = cross_one(BETAS, sup_L)          # contraction bound
    beta_dagger = cross_one(BETAS, mult_fp)      # saddle-node fold
    # operational collapse: cost crosses an SLO ceiling (2x the low-feedback cost)
    ceiling = 2.0 * cost[0]
    beta_collapse = None
    for i in range(1, len(cost)):
        if cost[i - 1] < ceiling <= cost[i]:
            beta_collapse = float(BETAS[i - 1] + (ceiling - cost[i - 1]) /
                                  (cost[i] - cost[i - 1]) * (BETAS[i] - BETAS[i - 1]))
            break

    print(f"beta*(x_m)   (contraction bound) = {beta_star}")
    print(f"beta_dagger  (saddle-node fold)  = {beta_dagger}")
    print(f"beta_collapse(SLO ceiling {ceiling:.2f}) = {beta_collapse}")

    out = dict(
        x_m=[float(v) for v in x_m],
        betas=[float(b) for b in BETAS],
        sup_slope=[float(v) for v in sup_L],
        fixed_point_multiplier=[float(v) for v in mult_fp],
        lam_star_ratio=[float(v) for v in lam_star],
        true_cost=[float(v) for v in cost],
        slo_ceiling=float(ceiling),
        beta_star=beta_star,
        beta_dagger=beta_dagger,
        beta_collapse=beta_collapse,
    )
    with open(os.path.join(OUT_DIR, "thresholds.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved thresholds.json")


if __name__ == "__main__":
    main()
