"""
cache.py
--------
Queueing-free performative testbed: Cache size and prefetch policy under
response-time-dependent browsing behavior.

Mechanical loop:
1. Faster responses (lower latency L) reduce user wait time, encouraging wider
   and more exploratory browsing (diluting request concentration -> lower Zipf skew s).
2. Lower skew s reduces cache hit ratio and severely degrades prefetch accuracy,
   since prefetch conversion requires concentrated, predictable request patterns.
3. Overly aggressive prefetching at high speed floods bandwidth with wasted traffic.

Configuration y = [C, p, ttl]:
    C:    Cache capacity as fraction of catalogue [0.02, 0.30]
    p:    Prefetch aggressiveness in [0.0, 1.0]
    ttl:  Admission patience / eviction delay in [0.20, 4.00]

beta_c in [0, 1] scales user behavioral feedback sensitivity.
"""

import numpy as np

# Physical constants of catalogue and latency model
N_ITEMS   = 1.0           # Normalized catalogue universe
S0        = 1.30          # Baseline Zipf popularity-skew exponent
L_HIT     = 0.5           # Normalized latency of a cache hit
L_MISS    = 6.0           # Normalized latency of a backend miss
BW_COST   = 3.0           # Cost weight penalty per unit wasted prefetch bandwidth
L_REF     = 2.6           # Reference latency around which browsing sensitivity pivots
DELTA_S   = 1.3           # Popularity skew sensitivity to response latency
PREF_EFF  = 0.75          # Maximum fraction of cache misses converted by prefetch

# Cost objective weights
W_MISS, W_BW, W_LATC = 2.0, 1.0, 0.25

# Configuration bounds: [C, p, ttl]
BOUNDS = np.array([
    [0.02, 0.30],    # C: cache size
    [0.00, 1.00],    # p: prefetch aggressiveness
    [0.20, 4.00]     # ttl: retention patience
])

# Canonical incumbent configuration: small cache, low prefetch (runs cold, high skew)
INCUMBENT = np.array([0.05, 0.10, 0.50])


def _clip(y):
    y = np.asarray(y, float)
    return np.minimum(np.maximum(y, BOUNDS[:, 0]), BOUNDS[:, 1])


def base_hit(C, ttl, s):
    """Che-type approximation of LRU cache hit ratio under Zipf skew s."""
    eff_C = C * (0.6 + 0.4 * np.tanh(ttl))
    return 1.0 - (1.0 - eff_C) ** (0.6 + s)


def pref_conv(s):
    """Prefetch conversion rate scales with request predictability (skew s)."""
    return PREF_EFF * float(np.clip(s / S0, 0.0, 1.0))


def hit_ratio(y, s):
    C, p, ttl = y
    h0 = base_hit(C, ttl, s)
    h = h0 + p * (1.0 - h0) * pref_conv(s)
    return float(min(h, 0.999))


def settle_cache(y, beta_c, iters=250, tol=1e-10):
    """
    Solves for the stationary fixed-point skew s* induced by policy y.
    s* = S0 - beta_c * DELTA_S * (L_REF - L(s*)) / L_REF.
    """
    y = _clip(y)
    s = S0
    for _ in range(iters):
        h = hit_ratio(y, s)
        L = h * L_HIT + (1.0 - h) * L_MISS
        s_new = S0 - beta_c * DELTA_S * (L_REF - L) / L_REF
        s_new = float(np.clip(s_new, 0.4, 2.0))
        if abs(s_new - s) < tol:
            s = s_new
            break
        s = 0.5 * s + 0.5 * s_new

    h = hit_ratio(y, s)
    L = h * L_HIT + (1.0 - h) * L_MISS
    return dict(s=s, hit=h, L=L)


def _cost_at_skew(y, s, beta_c):
    C, p, ttl = _clip(y)
    h = hit_ratio(y, s)
    L = h * L_HIT + (1.0 - h) * L_MISS
    bw = p * (1.0 - pref_conv(s))  # Bandwidth wasted on inaccurate prefetch
    return float(W_MISS * (1.0 - h) + W_BW * BW_COST * bw + W_LATC * L)


def true_cost(y, beta_c):
    """Long-run cost of running policy y under its own settled popularity skew."""
    d = settle_cache(y, beta_c)
    return _cost_at_skew(y, d["s"], beta_c)


def myopic_cost(y, y0, beta_c):
    """Cost of policy y scored against the frozen trace skew induced by y0."""
    d0 = settle_cache(y0, beta_c)
    return _cost_at_skew(y, d0["s"], beta_c)


def true_cost_vec(Y, beta_c, iters=150):
    """Vectorized calculation of performative cache cost for population search."""
    Y = np.atleast_2d(_clip(Y))
    C, p, ttl = Y[:, 0], Y[:, 1], Y[:, 2]
    s = np.full(len(Y), S0)

    for _ in range(iters):
        eff_C = C * (0.6 + 0.4 * np.tanh(ttl))
        h0 = 1.0 - (1.0 - eff_C) ** (0.6 + s)
        conv = PREF_EFF * np.clip(s / S0, 0.0, 1.0)
        h = np.minimum(h0 + p * (1.0 - h0) * conv, 0.999)
        L = h * L_HIT + (1.0 - h) * L_MISS
        s = np.clip(S0 - beta_c * DELTA_S * (L_REF - L) / L_REF, 0.4, 2.0)

    eff_C = C * (0.6 + 0.4 * np.tanh(ttl))
    h0 = 1.0 - (1.0 - eff_C) ** (0.6 + s)
    conv = PREF_EFF * np.clip(s / S0, 0.0, 1.0)
    h = np.minimum(h0 + p * (1.0 - h0) * conv, 0.999)
    L = h * L_HIT + (1.0 - h) * L_MISS
    bw = p * (1.0 - conv)
    return W_MISS * (1.0 - h) + W_BW * BW_COST * bw + W_LATC * L
