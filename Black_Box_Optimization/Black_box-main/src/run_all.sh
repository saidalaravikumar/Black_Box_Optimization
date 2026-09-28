#!/usr/bin/env bash
# Regenerate every result JSON in ../results (seed-controlled), then the matplotlib preview figures.
set -e
SD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PY:-python3}
cd "$SD"
echo "== Core benchmark =="
$PY exp_ground_truth.py          # reference optimum x_o, performativity gap, price of stability
$PY exp_optimizers.py            # primary benchmark, 5 single-shot tuners vs. 3 STS variants (Table II, Fig. 4)
$PY exp_convergence.py           # convergence traces
$PY exp_screening.py             # PST detection power / false-positive rate (Sec. V-D)
$PY exp_settling.py              # plateau/CUSUM settling time vs. ground truth, drift attribution
$PY exp_cache.py                 # Testbed B: queue-free cache/prefetch model
$PY exp_nsga.py                  # STS-NSGA-II cost-stability frontier
echo "== Robustness and diagnostics =="
$PY exp_thresholds.py            # beta*, beta-dagger, beta_col (Fig. 3)
$PY exp_des_validation.py        # independent discrete-event validation (Fig. 5)
$PY exp_ablation.py              # STS component ablation (Table III)
$PY exp_feedback_robustness.py   # four retry-response laws (Table IV)
$PY exp_repeated_eval.py         # fair deployment-time budget (Fig. 6)
$PY exp_sensitivity.py           # PST (k, delta) and stability weight gamma
$PY exp_scalability.py           # d in {4, 8, 16, 32}
$PY exp_seed_dist.py             # per-seed regret distribution (Fig. 7)
$PY exp_workload.py              # Poisson / bursty / diurnal / heavy-tailed demand
$PY exp_cache_alt.py             # alternative browsing-feedback forms
$PY exp_adaptive_workload.py     # adaptive stationarity-gated policy (Fig. 8)
echo "== Preview figures (matplotlib) =="
$PY plots.py
echo "Done. All simulation results saved in ../results/ and preview figures in ../figures/."
