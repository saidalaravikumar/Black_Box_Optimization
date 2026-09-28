"""
exp_ground_truth.py
-------------------
Computes fine-grid ground truth metrics across feedback strength beta in [0.0, 0.95].
Evaluates:
  - Performative optimum x_o and P(x_o, x_o)
  - Myopic optimum x_m and P(x_m, x_m) vs believed P(x_m, x_0)
  - Performatively stable configuration x_s and P(x_s, x_s)
  - Performativity gap Gamma(beta)
  - Price of stability rho(beta)
  - Score error / optimism
Saves results to results/ground_truth.json.
"""

import os
import json
import numpy as np
import fluid as F
import optimize_util as OU

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)

betas = np.linspace(0.0, 0.95, 35).tolist()
results = []

print("Running ground-truth benchmark across 35 beta values...")
for b in betas:
    # 1. Performative optimum (minimizes true cost)
    x_opt, c_opt = OU.find_optimum(b)

    # 2. Incumbent configuration
    x_inc = F.round_cfg(F.X0)
    lam_inc = float(F.settle(x_inc, b)["lam"])

    # 3. Myopic optimum (minimizes cost on incumbent trace)
    x_myo, myo_believed = OU.find_myopic(b, x_inc)
    lam_myo = float(F.settle(x_myo, b)["lam"])
    myo_realised = F.true_cost(x_myo, b)

    # 4. Performatively stable configuration
    x_stab, c_stab = OU.find_stable(b)

    gap = float(myo_realised - c_opt)
    p_stab = float(c_stab - c_opt)
    score_err = float(myo_realised - myo_believed)

    lam_opt = float(F.settle(x_opt, b)["lam"])

    results.append({
        "beta": float(b),
        "x_opt": [float(v) for v in x_opt],
        "cost_opt": float(c_opt),
        "x_myopic": [float(v) for v in x_myo],
        "myopic_believed": float(myo_believed),
        "myopic_realised": float(myo_realised),
        "x_stable": [float(v) for v in x_stab],
        "cost_stable": float(c_stab),
        "performativity_gap": gap,
        "price_of_stability": p_stab,
        "score_error": score_err,
        "lam_incumbent": float(lam_inc),
        "lam_opt": float(lam_opt),
        "lam_myopic": float(lam_myo)
    })
    print(f"  beta={b:.3f} | Gap={gap:.4f} | PriceStab={p_stab:.4f} | ScoreErr={score_err:.4f}")

out_file = os.path.join(OUT_DIR, "ground_truth.json")
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved ground truth results to {out_file}")
