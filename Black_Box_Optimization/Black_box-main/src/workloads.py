"""
workloads.py
------------
Exogenous base-arrival-rate generators for the serving DES .
All return a length-n series of the exogenous base rate lambda_0(t); the DES
adds Poisson-type demand noise and the endogenous performative amplification on
top. Used for workload-distribution robustness (beyond stationary Poisson).

  poisson  : stationary base rate (nominal).
  bursty   : ON/OFF telegraph process (alternating high/low regimes).
  diurnal  : slow sinusoidal time-of-day variation.
  heavytail: stationary mean with heavy-tailed (Pareto) multiplicative shocks.
"""
import numpy as np
import fluid as F


def make_series(kind, n, seed=0, base=None):
    base = F.LAMBDA0 if base is None else base
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    if kind == "poisson":
        return np.full(n, base)
    if kind == "bursty":
        # ON/OFF telegraph: regimes at 0.8x and 1.35x base, mean ~ base
        series = np.empty(n)
        state = 1.0
        p_switch = 0.03
        for i in range(n):
            if rng.random() < p_switch:
                state = 1.35 if state < 1.0 else 0.80
            series[i] = base * state
        return series
    if kind == "diurnal":
        period = max(60, n // 2)
        return base * (1.0 + 0.25 * np.sin(2 * np.pi * t / period))
    if kind == "heavytail":
        shocks = (rng.pareto(2.5, size=n) + 1.0)      # mean ~ 1.67
        shocks = shocks / shocks.mean()                # normalize to mean 1
        return base * (0.7 + 0.3 * np.clip(shocks, 0, 4))
    raise ValueError(kind)
