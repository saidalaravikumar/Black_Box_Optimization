"""
exp_seed_dist.py
----------------
SEED-LEVEL DISTRIBUTION ANALYSIS .

Post-processes the 100-seed optimizer benchmark (results/optimizers.json) at
beta = 0.60 and 0.80 to characterise the per-seed distribution of post-deployment
regret for CMA-ES/single-shot, BO-GP/single-shot, TUNA/single-shot, and the two
proposed STS methods. Reports median, IQR, bootstrap 95% CI on the median, and
the PAIRED fraction of seeds on which each STS method improves over each
single-shot baseline (same seed index). Also emits the raw per-seed regret
arrays for boxplot/violin rendering.

Saves results/seed_dist.json.
"""
import os, json
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
BETAS = [0.60, 0.80]
METHOD_MAP = {
    "CMA-ES / single-shot": "CMA-ES (Single-Shot)",
    "BO-GP / single-shot": "BO-GP (Single-Shot)",
    "TUNA-filter / single-shot": "TUNA-filter (Single-Shot)",
    "Proposed: CMA-ES / settle-then-score": "Proposed (STS CMA-ES)",
    "Proposed: CMA-ES / settle+stability": "Proposed (Stability-Reg. STS)",
}
STS_KEYS = ["Proposed: CMA-ES / settle-then-score", "Proposed: CMA-ES / settle+stability"]
SINGLE_KEYS = ["CMA-ES / single-shot", "BO-GP / single-shot", "TUNA-filter / single-shot"]


def bootstrap_median_ci(arr, n_boot=3000, seed=0):
    rng = np.random.default_rng(seed)
    meds = [np.median(rng.choice(arr, len(arr), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def main():
    with open(os.path.join(OUT_DIR, "optimizers.json")) as f:
        data = json.load(f)
    # index rows by (round(beta,2), method)
    by = {}
    for r in data["rows"]:
        by[(round(r["beta"], 2), r["method"])] = np.array(r["regrets"])

    out = {"betas": BETAS, "method_map": METHOD_MAP, "cells": []}
    for b in BETAS:
        # find nearest available beta key
        keyb = min({round(r["beta"], 2) for r in data["rows"]}, key=lambda x: abs(x - b))
        cell = {"beta": b, "beta_key": keyb, "dists": {}, "improvement": {}}
        for m in list(METHOD_MAP.keys()):
            arr = by[(keyb, m)]
            lo, hi = bootstrap_median_ci(arr, seed=1)
            cell["dists"][m] = dict(
                label=METHOD_MAP[m],
                regrets=[float(v) for v in arr],
                median=float(np.median(arr)),
                q1=float(np.percentile(arr, 25)),
                q3=float(np.percentile(arr, 75)),
                iqr=float(np.percentile(arr, 75) - np.percentile(arr, 25)),
                median_ci_lo=lo, median_ci_hi=hi,
                mean=float(np.mean(arr)),
            )
        # paired improvement fractions
        for sts in STS_KEYS:
            s_arr = by[(keyb, sts)]
            for base in SINGLE_KEYS:
                b_arr = by[(keyb, base)]
                n = min(len(s_arr), len(b_arr))
                frac = float(np.mean(b_arr[:n] > s_arr[:n]))
                cell["improvement"][f"{METHOD_MAP[sts]} vs {METHOD_MAP[base]}"] = frac
        out["cells"].append(cell)
        best_sts = "Proposed: CMA-ES / settle-then-score"
        print(f"beta={b} (key {keyb}): STS-CMA median={np.median(by[(keyb,best_sts)]):.4f}; "
              f"improve vs CMA-single = "
              f"{cell['improvement']['Proposed (STS CMA-ES) vs CMA-ES (Single-Shot)']*100:.0f}% of seeds")

    with open(os.path.join(OUT_DIR, "seed_dist.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved seed_dist.json")


if __name__ == "__main__":
    main()
