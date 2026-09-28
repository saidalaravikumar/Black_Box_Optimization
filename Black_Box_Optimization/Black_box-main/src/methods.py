"""
methods.py
----------
Core mechanisms transforming standard black-box optimizers into performativity-aware
and stability-preferring search frameworks:

1. Performative Screening Test (PST):
   Non-intrusive, paired-perturbation micro-trial executed prior to tuning budget
   commitment to measure empirical workload sensitivity kappa_p and establish
   statistical significance against exogenous drift noise.

2. Online Stationarity & Settling Detection:
   Dual-detector architecture combining a local linear regression slope/variation
   plateau test with a sequential two-sided CUSUM change detector on arrival rates.

3. Settle-Then-Score (STS) Evaluation:
   Deploys candidate, detects stationary convergence, differences out background
   exogenous drift using a parallel control replica, and applies a penalization
   to candidates that exhibit slow or incomplete settling (metastable risk).

4. Fixed-Point Stability Residual:
   Quantifies distance from the candidate to the neighborhood minimizer under its
   own induced workload, enabling bi-objective Pareto and stability-regularized tuning.
"""

import numpy as np
import fluid as F
import des


# --------------------------------------------------------------------------- #
# 1. Performative Screening Test (PST)                                        #
# --------------------------------------------------------------------------- #
def screening_test(x0, beta, k=10, delta=0.15, horizon=160, seed=0):
    """
    Executes k paired micro-perturbations around incumbent x0.
    Evaluates induced workload shift against natural exogenous variance
    measured on an unperturbed time-matched control replica.

    Returns:
        coefficient: normalized fractional workload sensitivity per unit perturbation
        induced_shift: raw mean shift in arrival rate
        exogenous_std: standard deviation of control replica arrival rate
        zscore: signal-to-noise ratio against exogenous background variation
    """
    rng = np.random.default_rng(seed)
    span = F.BOUNDS[:, 1] - F.BOUNDS[:, 0]
    signs = np.array([-1.0, 1.0])
    perturbs, loads, ctrl_loads = [], [], []

    for i in range(k):
        s = signs[i % 2]
        direction = rng.normal(size=4)
        direction /= (np.linalg.norm(direction) + 1e-9)
        xp = F._clip_cfg(x0 + s * delta * span * direction)

        # Deploy candidate perturbation
        d = des.deploy(xp, beta, n_steps=horizon, seed=int(rng.integers(1e6)))
        loads.append(d["lam"][-horizon // 3:].mean())

        # Time-matched control replica on incumbent
        dc = des.deploy(x0, beta, n_steps=horizon, seed=int(rng.integers(1e6)))
        ctrl_loads.append(dc["lam"][-horizon // 3:].mean())
        perturbs.append(s * delta)

    loads = np.array(loads)
    ctrl = np.array(ctrl_loads)
    base = ctrl.mean()
    ctrl_std = ctrl.std() + 1e-6

    induced = np.abs(loads - base).mean()
    coeff = induced / (abs(base) + 1e-9) / delta
    zscore = induced / ctrl_std

    return dict(
        coefficient=float(coeff),
        induced_shift=float(induced),
        exogenous_std=float(ctrl_std),
        zscore=float(zscore),
        loads=loads.tolist(),
        base=float(base)
    )


# --------------------------------------------------------------------------- #
# 2. Online Stationarity & Settling Detection                                 #
# --------------------------------------------------------------------------- #
def detect_settling(series, window=40, slope_tol=2e-3, var_tol=0.02, method="plateau"):
    """
    Returns time-step index at which the stream is declared stationary,
    or len(series) if settling is not achieved within the observation horizon.

    Methods:
        'plateau': Fits linear trend on sliding window. Declares settled when
                   normalized slope < slope_tol and coefficient of variation < var_tol.
        'cusum':   Two-sided cumulative sum change detector on increments. Declares
                   settled when no regime change occurs for `window` consecutive steps.
    """
    x = np.asarray(series, float)
    n = len(x)
    scale = np.median(np.abs(x)) + 1e-6

    if method == "plateau":
        for t in range(window, n):
            w = x[t - window:t]
            tt = np.arange(window)
            slope = np.polyfit(tt, w, 1)[0]
            cov = w.std() / (np.abs(w.mean()) + 1e-9)
            if (abs(slope) / scale) < slope_tol and cov < var_tol:
                return t
        return n
    else:  # cusum
        diffs = np.diff(x) / scale
        gp = gn = 0.0
        last_change = window
        thr = 0.02
        for t in range(1, n):
            e = diffs[t - 1]
            gp = max(0.0, gp + e - 0.001)
            gn = min(0.0, gn + e + 0.001)
            if gp > thr or gn < -thr:
                last_change = t
                gp = gn = 0.0
            if t - last_change > window:
                return t
        return n


# --------------------------------------------------------------------------- #
# 3. Settle-Then-Score (STS) Evaluation                                       #
# --------------------------------------------------------------------------- #
def settle_then_score(x, beta, lam_prev=None, budget=450, warm=25,
                      settle_penalty=0.02, drift=0.0, control=True,
                      control_cfg=None, seed=0):
    """
    Closed-loop evaluation: Deploys candidate x, detects stationary relaxation,
    isolates and removes exogenous drift using a parallel reference control,
    and returns penalized fitness.
    """
    d = des.deploy(x, beta, lam_prev=lam_prev, n_steps=budget, drift=drift, seed=seed)
    t_settle = detect_settling(d["lam"], method="plateau")
    censored = (t_settle >= budget)
    t_eff = min(t_settle, budget)

    post = d["cost"][t_eff:] if t_eff < budget - warm else d["cost"][-warm:]
    steady_cost = float(np.mean(post))

    # Control differencing for drift cancellation
    if control and drift != 0.0:
        ref = control_cfg if control_cfg is not None else np.array([30.0, 2.5, 0.0, 120.0])
        dc = des.deploy(ref, beta, n_steps=budget, drift=drift, seed=seed + 777)
        drift_bias = float(np.mean(dc["cost"][t_eff:]) - np.mean(dc["cost"][:warm]))
        steady_cost -= drift_bias

    # Penalty for slow convergence or censorship
    fitness = steady_cost + settle_penalty * (t_eff / budget) * (1.0 + float(censored))

    return dict(
        fitness=float(fitness),
        steady_cost=float(steady_cost),
        settle_time=int(t_eff),
        censored=bool(censored),
        lam_final=float(d["lam_final"])
    )


def single_shot_score(x, beta, lam_prev=None, warm=20, window=25, seed=0):
    """
    Conventional (myopic) evaluation: Evaluates candidate over an immediate short
    window before the system workload relaxes to the new induced state.
    """
    d = des.deploy(x, beta, lam_prev=lam_prev, n_steps=warm + window, seed=seed)
    return float(np.mean(d["cost"][warm:warm + window]))


# --------------------------------------------------------------------------- #
# 4. Fixed-Point Stability Residual                                           #
# --------------------------------------------------------------------------- #
def stability_residual(x, beta, radius=0.06, n=48, seed=0):
    """
    Estimates the fixed-point residual ||x - argmin_{x'} P(x', x)||_X.
    Scores a localized neighborhood of candidate x against the workload D(x)
    induced by x without redeployment.
    A zero residual signifies an exact performative fixed point (stable equilibrium).
    """
    x = F._clip_cfg(x)
    lam_x = F.settle(x, beta)["lam"]
    rng = np.random.default_rng(seed)
    span = F.BOUNDS[:, 1] - F.BOUNDS[:, 0]

    neighborhood = F._clip_cfg(x[None, :] + rng.normal(size=(n, 4)) * radius * span[None, :])
    candidates = np.vstack([x[None, :], neighborhood])

    costs = F.cost_vec(candidates, np.full(len(candidates), lam_x), beta)
    best_idx = int(np.argmin(costs))

    resid = np.linalg.norm((candidates[best_idx] - x) / span)
    return float(resid)
