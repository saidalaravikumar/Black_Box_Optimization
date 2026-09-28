"""
scalable.py
-----------
Scalable d-dimensional serving configuration model .

The first four knobs are the ORIGINAL performative knobs x_core = [b, T, r, adm]
that drive the endogenous retry/queueing feedback loop (fluid.py). The remaining
d-4 knobs are interpretable *secondary* configuration parameters that add a
smooth, static (exogenous) cost component but do not alter the performative loop:

  worker threads, prefetch depth, connection-pool size, GC interval, log
  verbosity, compression level, ... (alternating integer / continuous knobs).

Each secondary knob z_j contributes a convex penalty minimized at a fixed
interior target t_j; the aggregate secondary weight is normalized so the total
secondary cost magnitude is comparable (~0.6) across dimensions d. Thus the
performative optimum is well defined for every d, and increasing d is a genuine
search-space scaling test while the performative structure is preserved.
"""
import numpy as np
import fluid as F

CORE_LABELS = ["batch b", "timeout T", "retries r", "admission adm"]
# Secondary interpretable knobs (label, is_integer, lo, hi)
SECONDARY = [
    ("worker threads", True, 1, 32),
    ("prefetch depth", True, 0, 16),
    ("connection pool", True, 8, 256),
    ("GC interval (s)", False, 0.5, 10.0),
    ("log verbosity", True, 0, 5),
    ("compression level", True, 0, 9),
    ("keepalive (s)", False, 1.0, 60.0),
    ("write buffer (MB)", False, 1.0, 64.0),
    ("read-ahead (KB)", True, 4, 512),
    ("shard count", True, 1, 64),
    ("flush threshold", False, 0.1, 0.9),
    ("index cache frac", False, 0.05, 0.5),
    ("thread affinity", True, 0, 3),
    ("numa nodes", True, 1, 4),
    ("io depth", True, 1, 64),
    ("spill ratio", False, 0.1, 0.95),
    ("bloom bits", True, 4, 20),
    ("compaction thr", True, 2, 12),
    ("prefetch window", True, 1, 32),
    ("socket buffer", True, 16, 512),
    ("retry jitter (ms)", False, 0.0, 100.0),
    ("cache lines", True, 8, 256),
    ("batch linger (ms)", False, 0.0, 50.0),
    ("max inflight", True, 1, 128),
    ("hedge delay (ms)", False, 0.0, 40.0),
    ("segment (MB)", False, 8.0, 256.0),
    ("ttl slack (s)", False, 0.1, 5.0),
    ("warmup reqs", True, 0, 500),
]


def bounds(d):
    assert 4 <= d <= 4 + len(SECONDARY)
    B = [list(F.BOUNDS[i]) for i in range(4)]
    for j in range(d - 4):
        _, _, lo, hi = SECONDARY[j]
        B.append([float(lo), float(hi)])
    return np.array(B)


def _targets(d):
    """Deterministic interior targets in normalized [0,1] for secondary knobs."""
    return np.array([0.35 + 0.4 * ((0.61803398 * (j + 1)) % 1.0) for j in range(d - 4)])


def to_cfg(u, d):
    """Normalized [0,1]^d -> physical config, rounding integer knobs."""
    u = np.clip(np.asarray(u, float), 0.0, 1.0)
    B = bounds(d)
    x = B[:, 0] + u * (B[:, 1] - B[:, 0])
    x[0] = round(x[0]); x[2] = round(x[2])   # core integers b, r
    for j in range(d - 4):
        if SECONDARY[j][1]:
            x[4 + j] = round(x[4 + j])
    return x


def _norm(x, d):
    B = bounds(d)
    return (np.asarray(x, float) - B[:, 0]) / (B[:, 1] - B[:, 0])


def _secondary_cost(u_sec, d):
    if d == 4:
        return 0.0
    t = _targets(d)
    w = 0.6 / (d - 4)
    return float(np.sum(w * (np.asarray(u_sec) - t) ** 2))


def true_cost(x, beta, d):
    x = np.asarray(x, float)
    core = F.true_cost(x[:4], beta)
    u = _norm(x, d)
    return core + _secondary_cost(u[4:], d)


def myopic_cost(x, x0, beta, d):
    x = np.asarray(x, float); x0 = np.asarray(x0, float)
    core = F.myopic_cost(x[:4], x0[:4], beta)
    u = _norm(x, d)
    return core + _secondary_cost(u[4:], d)


def optimum(beta, d):
    """Performative optimum: core optimum (via mean-field) + secondary targets."""
    import optimize_util as OU
    xo_core, c_core = OU.find_optimum(beta)
    # secondary at targets => zero secondary cost
    return c_core  # optimal total cost (secondary cost = 0 at targets)


def stability_residual(x, beta, d, radius=0.06, n=48, seed=0):
    """Fixed-point residual on the performative core knobs (the feedback lives there)."""
    import methods as M
    return M.stability_residual(np.asarray(x, float)[:4], beta, radius=radius, n=n, seed=seed)
