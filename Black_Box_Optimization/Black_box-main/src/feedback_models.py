"""
feedback_models.py
------------------
Alternative performative feedback laws for the serving testbed .

The nominal serving model couples workload to congestion through the per-attempt
retry propensity psi = beta * q inside the geometric attempt count
    A(q) = (1 - psi^(r+1)) / (1 - psi).
Here we keep the SAME queueing/cost machinery but replace the FUNCTIONAL FORM of
the behavioral response psi(q; beta), so that the arrival-rate feedback obeys a
different (but physically interpretable) law. This tests whether STS depends on
the exact feedback function used to build the nominal simulator.

Models (all reduce the coupling to a per-attempt reissue propensity psi in [0,1)):
  baseline  : psi = beta * q                          (linear in perceived congestion)
  concave   : psi = 0.707 * beta * sqrt(q)            (strong response even at low congestion)
  sigmoid   : psi = beta * sigma(6 (q - 0.5))         (smooth saturating response)
  threshold : psi = beta * 1.75 * max(0,(q-0.3)/0.7)  (weak below threshold, stronger above)

Calibration: all models satisfy psi(q=0.5) = 0.5 beta, matching the baseline's
feedback strength at mid-congestion, so beta is comparable across laws.
"""
import numpy as np
import fluid as F

MODELS = ["baseline", "concave", "sigmoid", "threshold"]


def psi_of_q(q, beta, model):
    q = np.clip(q, 0.0, 1.0)
    if model == "baseline":
        psi = beta * q
    elif model == "concave":
        psi = 0.7071 * beta * np.sqrt(q)
    elif model == "sigmoid":
        psi = beta / (1.0 + np.exp(-6.0 * (q - 0.5)))
    elif model == "threshold":
        psi = 1.75 * beta * np.maximum(0.0, (q - 0.3) / 0.7)
    else:
        raise ValueError(model)
    return np.clip(psi, 0.0, 1.0 - 1e-9)


def _attempts(psi, r):
    psi = np.clip(psi, 0.0, 1.0 - 1e-9)
    return (1.0 - psi ** (r + 1.0)) / (1.0 - psi)


def _completion(q, psi, r):
    psi = np.clip(psi, 0.0, 1.0 - 1e-9)
    return (1.0 - q) * (1.0 - psi ** (r + 1.0)) / (1.0 - psi)


def settle(x, beta, model, lam0=None, iters=160, tol=1e-11):
    """Outer fixed point lambda* = lam0 * A(q(lambda*)) under the chosen feedback law."""
    x = F._clip_cfg(x)
    lam0 = F.LAMBDA0 if lam0 is None else lam0
    r = x[2]
    lam = lam0
    for _ in range(iters):
        q, _ = F.operating_point(lam, x)
        psi = psi_of_q(q, beta, model)
        lam_new = min(lam0 * _attempts(psi, r), 5.0 * lam0)
        if abs(lam_new - lam) < tol:
            lam = lam_new
            break
        lam = 0.6 * lam + 0.4 * lam_new
    q, _ = F.operating_point(lam, x)
    return dict(lam=float(lam), q=float(q))


def _cost_at(x, lam, beta, model):
    x = F._clip_cfg(x)
    b, T, r, adm = x
    q, drain = F.operating_point(lam, x)
    psi = psi_of_q(q, beta, model)
    tail_sojourn = F.batch_delay(b) + F.C_TAIL / max(drain, 0.25)
    c = _completion(q, psi, r)
    attempts = _attempts(psi, r)
    tail = min(tail_sojourn + (attempts - 1.0) * (T + F.BACKOFF), 8.0)
    e_req = F.E0 * (1.0 + F.GAMMA_E / b)
    e_completed = min(e_req * attempts / max(c, 0.02), 24.0)
    return float(F.W_LAT * tail / F.LAT_NORM + F.W_ENE * e_completed / F.ENE_NORM + F.W_CMP * (1.0 - c))


def true_cost(x, beta, model):
    d = settle(x, beta, model)
    return _cost_at(x, d["lam"], beta, model)


def myopic_cost(x, x0, beta, model):
    d0 = settle(x0, beta, model)
    return _cost_at(x, d0["lam"], beta, model)


# ---- vectorized ground-truth optimum ----
def _op_vec(lam, X, iters=30):
    return F._op_vec(lam, X, iters)


def true_cost_vec(X, beta, model):
    X = np.atleast_2d(X).astype(float)
    r = X[:, 2]
    lam0 = F.LAMBDA0
    lam = np.full(len(X), float(lam0))
    for _ in range(120):
        q, drain = _op_vec(lam, X)
        psi = psi_of_q(q, beta, model)
        lam = 0.6 * lam + 0.4 * np.minimum(lam0 * _attempts(psi, r), 5.0 * lam0)
    b, T = X[:, 0], X[:, 1]
    q, drain = _op_vec(lam, X)
    psi = psi_of_q(q, beta, model)
    tail_sojourn = F.batch_delay(b) + F.C_TAIL / np.maximum(drain, 0.25)
    c = _completion(q, psi, r)
    attempts = _attempts(psi, r)
    tail = np.minimum(tail_sojourn + (attempts - 1.0) * (T + F.BACKOFF), 8.0)
    e_req = F.E0 * (1.0 + F.GAMMA_E / b)
    e_completed = np.minimum(e_req * attempts / np.maximum(c, 0.02), 24.0)
    return (F.W_LAT * tail / F.LAT_NORM + F.W_ENE * e_completed / F.ENE_NORM + F.W_CMP * (1.0 - c))


def find_optimum(beta, model, n=10):
    axes = [np.linspace(lo, hi, n) for lo, hi in F.BOUNDS]
    B, T, R, A = np.meshgrid(*axes, indexing="ij")
    grid = np.column_stack([B.ravel(), T.ravel(), R.ravel(), A.ravel()])
    # multi-start refine
    import optimize_util as OU
    obj = lambda X: true_cost_vec(X, beta, model)
    xo = OU.minimize_global(obj, grid, n_seeds=4, rounds=16, pop=600)
    return F.round_cfg(xo), float(true_cost(F.round_cfg(xo), beta, model))
