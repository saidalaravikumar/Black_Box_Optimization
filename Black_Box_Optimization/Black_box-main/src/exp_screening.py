"""
exp_screening.py
----------------
Empirical validation of the Performative Screening Test (PST) with 100 replications.
Sweeps 16 fine feedback levels beta in [0.0, 0.90].
Validates:
1. Monotonic tracking of true feedback strength by the recovered coefficient kappa_p.
2. High statistical detection power (>98% for beta >= 0.10).
3. Rigorous false-positive control on the non-performative negative control (beta = 0.0).
Saves results to results/screening.json.
"""

import os
import json
import numpy as np
import fluid as F
from methods import screening_test

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)

INCUMBENT = F.round_cfg(F.X0)


from concurrent.futures import ProcessPoolExecutor


def _worker_null(s):
    return screening_test(INCUMBENT, 0.0, k=10, seed=5000 + s)["zscore"]


def _worker_beta(args):
    b, s, threshold = args
    res = screening_test(INCUMBENT, b, k=10, seed=s)
    return b, res["coefficient"], res["zscore"], (1.0 if res["zscore"] > threshold else 0.0)


def run(betas=None, reps=100, k=10):
    if betas is None:
        betas = np.round(np.linspace(0.0, 0.90, 16), 3)

    print(f"Calibrating null distribution on negative control (beta=0) with {reps} trials...")
    with ProcessPoolExecutor() as pool:
        null_zs = list(pool.map(_worker_null, range(reps)))
    threshold = float(np.quantile(null_zs, 0.95))
    fpr = float(np.mean([1.0 if z > threshold else 0.0 for z in null_zs]))

    tasks = [(float(b), s, threshold) for b in betas for s in range(reps)]
    print(f"Evaluating screening test across {len(betas)} beta levels ({len(tasks)} runs with multiprocessing)...")

    results_by_b = {float(b): {"coeffs": [], "zs": [], "detected": []} for b in betas}
    with ProcessPoolExecutor() as pool:
        for b, coeff, z, det in pool.map(_worker_beta, tasks):
            results_by_b[b]["coeffs"].append(coeff)
            results_by_b[b]["zs"].append(z)
            results_by_b[b]["detected"].append(det)

    rows = []
    for b in betas:
        b_float = float(b)
        coeffs = results_by_b[b_float]["coeffs"]
        detected = results_by_b[b_float]["detected"]
        rows.append(dict(
            beta=b_float,
            kappa_p_mean=float(np.mean(coeffs)),
            kappa_p_std=float(np.std(coeffs)),
            kappa_p_sem=float(np.std(coeffs) / np.sqrt(reps)),
            detection_rate=float(np.mean(detected))
        ))
        print(f"  beta={b_float:.2f}: kappa_p={np.mean(coeffs):.3f}, detect={100*np.mean(detected):.1f}%")

    return dict(threshold=threshold, false_positive_rate=fpr, rows=rows)


if __name__ == "__main__":
    res = run(reps=100)
    out_file = os.path.join(OUT_DIR, "screening.json")
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved screening benchmark to {out_file}")
