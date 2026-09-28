"""
fluid.py
--------
Deterministic mean-field (fluid) model of a request-serving loop with a
retrying / abandoning client population. This provides GROUND TRUTH for the
Performative System Configuration Optimization (PSCO) problem.

Given a configuration x = [b, T, r, adm], this module computes the unique
settled (fixed-point) workload distribution D(x) and its long-run cost P(x, x).

Configuration knobs:
    b    batch size            (capacity vs. batching-delay trade-off)
    T    client timeout (s)    (load-shedding vs. retry-amplification threshold)
    r    retry budget          (attempts allowed beyond initial request)
    adm  admission cap         (concurrency threshold before early shedding)

Coupled fixed-point mechanics:
1. Fast Physical Reneging: Requests not served within slack T - d_batch(b)
   renege before occupying server resources, loading the server at lambda*(1 - q).
2. Slow Behavioral Retry: Timed-out requests are re-issued with probability beta,
   amplifying arrival rate lambda = lambda_0 * E[attempts].
"""

import os
import numpy as np

# Physical server stack parameters (fixed system constants)
MU_MAX   = 210.0      # Asymptotic service rate (req/s) at infinite batch size
KAPPA    = 9.0        # Batch half-saturation constant
ALPHA_B  = 0.011      # Batch formation delay coefficient (s/req)
BACKOFF  = 0.20       # Client backoff duration added to timed-out retries (s)
K_TAIL   = 1.4        # Heavy-tail exponent for service sojourn distribution
C_TAIL   = 3.0        # Tail-quantile multiplier over mean sojourn
E0       = 1.0        # Normalized baseline energy per request
GAMMA_E  = 4.0        # Per-request fixed-overhead energy amortized by batching
P_TAIL   = 0.95       # Tail-latency quantile target
LAMBDA0  = float(os.environ.get("LAM0", 155.0))  # Exogenous base arrival rate (req/s)

# Multi-criteria cost weights: tail latency, energy per completed token, incompletion penalty
W_LAT, W_ENE, W_CMP = 1.0, 0.30, 2.0

# Normalization constants to balance objective scales
LAT_NORM, ENE_NORM = 0.9, 6.0

# Configuration space bounds: [b, T, r, adm]
BOUNDS = np.array([
    [1.0, 32.0],     # b: batch size
    [0.05, 4.0],     # T: timeout (s)
    [0.0, 4.0],      # r: retry budget
    [20.0, 400.0]    # adm: admission cap
])

# Canonical incumbent configuration: aggressive, tail-chasing policy
X0 = np.array([8.0, 0.25, 3.0, 200.0])
INCUMBENT = np.array([8.0, 0.25, 3.0, 200.0])


def mu_of_b(b):
    """Service capacity saturates with batch size according to Michaelis-Menten kinetics."""
    return MU_MAX * b / (b + KAPPA)


def batch_delay(b):
    """Linear batch-formation delay before execution."""
    return ALPHA_B * b


def _clip_cfg(x):
    """Project configuration into the feasible box domain."""
    x = np.asarray(x, dtype=float).copy()
    return np.minimum(np.maximum(x, BOUNDS[:, 0]), BOUNDS[:, 1])


def round_cfg(x):
    """Round discrete configuration coordinates (batch size, retry budget)."""
    x = _clip_cfg(x)
    x[0] = np.round(x[0])
    x[2] = np.round(x[2])
    return x


def operating_point(lam, x, iters=40, tol=1e-12):
    """
    Inner fixed point: Solves for timeout probability q and effective server drain
    under a given total arrival rate lam. Only surviving requests (1 - q) occupy
    the execution queue.
    """
    b, T, r, adm = x
    mu = mu_of_b(b)
    slack = T - batch_delay(b)
    if slack <= 0:
        return 1.0, max(mu, 1e-6)  # Immediate reneging if formation delay exceeds timeout

    q = 0.5
    for _ in range(iters):
        served_load = lam * (1.0 - q)
        drain = mu - served_load
        if drain <= 1e-6:
            q_new = 1.0
            drain = 1e-6
        else:
            q_new = 1.0 / (1.0 + (drain * slack) ** K_TAIL)
        
        # Admission cap soft penalty
        q_new = min(1.0, q_new + max(0.0, served_load / mu - adm / (adm + 25.0)) * 0.5)
        if abs(q_new - q) < tol:
            q = q_new
            break
        q = 0.5 * q + 0.5 * q_new

    served_load = lam * (1.0 - q)
    drain = max(mu - served_load, 1e-6)
    return float(q), float(drain)


def expected_attempts(q, r, beta):
    """
    Expected total attempts per exogenous request under geometric retry logic
    capped at r retries (r+1 total attempts) with retry probability beta.
    """
    bq = beta * q
    if bq >= 1.0 - 1e-12:
        return float(r + 1.0)
    return float((1.0 - bq ** (r + 1.0)) / (1.0 - bq))


def completion_prob(q, r, beta):
    """Probability that an exogenous request eventually completes successfully within r retries."""
    bq = beta * q
    if bq >= 1.0 - 1e-12:
        return float((1.0 - q) * (r + 1.0))
    return float((1.0 - q) * (1.0 - bq ** (r + 1.0)) / (1.0 - bq))


def settle(x, beta, lam0=LAMBDA0, iters=120, tol=1e-11):
    """
    Outer fixed point: Solves lambda* = lam0 * E[attempts](q(lambda*)).
    Returns the stationary induced workload state D(x).
    """
    x = _clip_cfg(x)
    b, T, r, adm = x
    lam = lam0
    for _ in range(iters):
        q, drain = operating_point(lam, x)
        lam_new = lam0 * expected_attempts(q, r, beta)
        lam_new = min(lam_new, 5.0 * lam0)  # Numerical safety boundary
        if abs(lam_new - lam) < tol:
            lam = lam_new
            break
        lam = 0.6 * lam + 0.4 * lam_new

    q, drain = operating_point(lam, x)
    attempts = expected_attempts(q, r, beta)
    completion = completion_prob(q, r, beta)
    retry_frac = 1.0 - lam0 / max(lam, 1e-9)

    return dict(
        lam=float(lam),
        q=float(q),
        drain=float(drain),
        attempts=float(attempts),
        completion=float(completion),
        retry_frac=float(retry_frac)
    )


def _cost_at_workload(x, lam, beta):
    """Computes operating cost of configuration x evaluated against arrival rate lam."""
    x = _clip_cfg(x)
    b, T, r, adm = x
    q, drain = operating_point(lam, x)
    tail_sojourn = batch_delay(b) + C_TAIL / max(drain, 0.25)
    c = completion_prob(q, r, beta)
    attempts = expected_attempts(q, r, beta)

    # End-to-end perceived tail latency including failed retry delay
    tail = min(tail_sojourn + (attempts - 1.0) * (T + BACKOFF), 8.0)

    # Energy amortized per completed request
    e_req = E0 * (1.0 + GAMMA_E / b)
    e_completed = min(e_req * attempts / max(c, 0.02), 24.0)

    cost = (W_LAT * tail / LAT_NORM
            + W_ENE * e_completed / ENE_NORM
            + W_CMP * (1.0 - c))

    return float(cost), dict(
        tail=tail,
        tail_sojourn=tail_sojourn,
        q=q,
        drain=drain,
        completion=c,
        attempts=attempts,
        energy=e_completed
    )


def true_cost(x, beta, lam0=LAMBDA0):
    """P(x, x): Stationary performative cost under self-induced workload D(x)."""
    d = settle(x, beta, lam0)
    return _cost_at_workload(x, d["lam"], beta)[0]


def true_cost_full(x, beta, lam0=LAMBDA0):
    """Returns P(x, x) along with full diagnostic state."""
    d = settle(x, beta, lam0)
    c, info = _cost_at_workload(x, d["lam"], beta)
    info["lam"] = d["lam"]
    return c, info


def myopic_cost(x, x0, beta, lam0=LAMBDA0):
    """P(x, x0): Cost of running candidate x evaluated on the incumbent's frozen workload D(x0)."""
    d0 = settle(x0, beta, lam0)
    return _cost_at_workload(x, d0["lam"], beta)[0]


# ---------------------------------------------------------------------------
# High-performance Vectorized Routines (Grid / Population Landscape Search)
# ---------------------------------------------------------------------------
def _op_vec(lam, X, iters=30):
    b, T, r, adm = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    mu = mu_of_b(b)
    slack = T - batch_delay(b)
    pos = slack > 0
    slack_s = np.where(pos, slack, 1.0)
    q = np.full(len(X), 0.5)

    for _ in range(iters):
        served = lam * (1.0 - q)
        drain = np.maximum(mu - served, 1e-6)
        q_new = 1.0 / (1.0 + (drain * slack_s) ** K_TAIL)
        floor = np.maximum(0.0, served / mu - adm / (adm + 25.0)) * 0.5
        q_new = np.minimum(1.0, q_new + floor)
        q_new = np.where(pos, q_new, 1.0)
        q = 0.5 * q + 0.5 * q_new

    served = lam * (1.0 - q)
    drain = np.maximum(mu - served, 1e-6)
    return q, drain


def _attempts_vec(q, r, beta):
    bq = np.clip(beta * q, 0.0, 1.0 - 1e-9)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return (1.0 - bq ** (r + 1.0)) / (1.0 - bq)


def _completion_vec(q, r, beta):
    bq = np.clip(beta * q, 0.0, 1.0 - 1e-9)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return (1.0 - q) * (1.0 - bq ** (r + 1.0)) / (1.0 - bq)


def settle_vec(X, beta, lam0=LAMBDA0, iters=80):
    X = np.atleast_2d(X).astype(float)
    r = X[:, 2]
    lam = np.full(len(X), float(lam0))
    for _ in range(iters):
        q, drain = _op_vec(lam, X)
        lam_new = np.minimum(lam0 * _attempts_vec(q, r, beta), 5.0 * lam0)
        lam = 0.6 * lam + 0.4 * lam_new
    q, drain = _op_vec(lam, X)
    return lam, q, drain


def cost_vec(X, lam, beta):
    X = np.atleast_2d(X).astype(float)
    b, T, r, adm = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    q, drain = _op_vec(lam, X)
    tail_sojourn = batch_delay(b) + C_TAIL / np.maximum(drain, 0.25)
    c = _completion_vec(q, r, beta)
    attempts = _attempts_vec(q, r, beta)
    tail = np.minimum(tail_sojourn + (attempts - 1.0) * (T + BACKOFF), 8.0)
    e_req = E0 * (1.0 + GAMMA_E / b)
    e_completed = np.minimum(e_req * attempts / np.maximum(c, 0.02), 24.0)
    return (W_LAT * tail / LAT_NORM
            + W_ENE * e_completed / ENE_NORM
            + W_CMP * (1.0 - c))


def true_cost_vec(X, beta, lam0=LAMBDA0):
    lam, q, drain = settle_vec(X, beta, lam0)
    return cost_vec(X, lam, beta)


def obj2_vec(X, lam, beta):
    X = np.atleast_2d(X).astype(float)
    b, T, r = X[:, 0], X[:, 1], X[:, 2]
    q, drain = _op_vec(lam, X)
    tail_sojourn = batch_delay(b) + C_TAIL / np.maximum(drain, 0.25)
    c = _completion_vec(q, r, beta)
    attempts = _attempts_vec(q, r, beta)
    tail = np.minimum(tail_sojourn + (attempts - 1.0) * (T + BACKOFF), 8.0)
    e_completed = np.minimum((1.0 + GAMMA_E / b) * attempts / np.maximum(c, 0.02), 24.0)
    return np.column_stack([tail / LAT_NORM, W_ENE * e_completed / ENE_NORM])


def obj2_settled(X, beta):
    lam, _, _ = settle_vec(X, beta)
    return obj2_vec(X, lam, beta)
