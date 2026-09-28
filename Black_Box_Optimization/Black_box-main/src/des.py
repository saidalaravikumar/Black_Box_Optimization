"""
des.py
------
Stochastic closed-loop discrete-event simulator of the serving stack.
The deterministic mean-field limit of this process is fluid.py.

Simulates the finite relaxation dynamics toward the performative fixed point,
incorporating:
1. Dynamic settling inertia (kappa parameter setting client adjustment speed).
2. Stochastic demand / Poisson arrival noise.
3. Exogenous non-stationary load drift (linear or arbitrary series).

This enables rigorous validation of:
- Online stationarity and settling time detection.
- Dual-replica control differencing for drift cancellation.
- Censored evaluations when budget caps are reached during search.
"""

import numpy as np
import fluid as F


def induced_target(lam, x, beta, lam0):
    """Calculates instantaneous target arrival rate under current timeout probability."""
    q, _ = F.operating_point(lam, x)
    return lam0 * F.expected_attempts(q, beta=beta, r=x[2])


def deploy(x, beta, lam_prev=None, lam0_series=None, n_steps=500, kappa=0.06,
           noise=0.02, drift=0.0, seed=0, warm_from=None):
    """
    Simulates one closed-loop deployment of configuration x.

    Parameters:
        x: configuration array [b, T, r, adm]
        beta: performative feedback strength in [0, 1]
        lam_prev: initial arrival rate before deployment (default: lam0_series[0])
        lam0_series: time series of exogenous base arrival rate
        n_steps: duration of deployment in discrete time steps
        kappa: client population relaxation rate (inertia inverse)
        noise: relative demand noise standard deviation
        drift: linear drift factor if lam0_series is None
        seed: random seed for reproducibility

    Returns:
        Dictionary containing time-series arrays for arrival rate, timeout prob,
        and instantaneous measured cost.
    """
    x = F._clip_cfg(x)
    rng = np.random.default_rng(seed)

    if lam0_series is None:
        base = F.LAMBDA0
        lam0_series = base * (1.0 + drift * np.linspace(0.0, 1.0, n_steps))

    lam = float(lam_prev if lam_prev is not None else lam0_series[0])
    if warm_from is not None:
        lam = float(warm_from)

    lam_hist = np.empty(n_steps)
    q_hist = np.empty(n_steps)
    cost_hist = np.empty(n_steps)

    for t in range(n_steps):
        lam0 = lam0_series[t]
        target = min(induced_target(lam, x, beta, lam0), 5.0 * lam0)
        # Relaxation dynamics with square-root diffusion noise
        lam = lam + kappa * (target - lam) + rng.normal() * noise * np.sqrt(max(lam, 1.0))
        lam = max(lam, 1.0)

        q, _ = F.operating_point(lam, x)
        c, _ = F._cost_at_workload(x, lam, beta)

        lam_hist[t] = lam
        q_hist[t] = q
        cost_hist[t] = c + rng.normal() * 0.01  # Stochastic observation noise

    return dict(
        lam=lam_hist,
        q=q_hist,
        cost=cost_hist,
        lam0=lam0_series,
        lam_final=lam
    )


if __name__ == "__main__":
    # Quick self-test: DES steady-state mean matches fluid model
    test_cfg = np.array([32.0, 3.5, 0.0, 120.0])
    res = deploy(test_cfg, beta=0.7, n_steps=1500, seed=42)
    fl = F.settle(test_cfg, 0.7)
    print(f"DES steady lambda: {res['lam'][-300:].mean():.2f}, Fluid lambda: {fl['lam']:.2f}")
    print(f"DES steady cost: {res['cost'][-300:].mean():.3f}, Fluid cost: {F.true_cost(test_cfg, 0.7):.3f}")
