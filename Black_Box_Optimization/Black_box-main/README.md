# Black-Box Optimization Under Endogenous Workload Feedback

Code for the paper:  
**"Deployment-Aware Black-Box Optimization of Computing Systems Under Endogenous Workload Feedback"**  


---

## Overview

Automated system tuners typically assume incoming workloads are exogenous. However, in interactive and cloud computing systems, deployed configurations often reshape workloads through endogenous feedback loops (e.g., degraded latency causes timeouts, client retries, and traffic surges). A configuration that appears optimal on offline traces may trigger performance collapse when deployed live.

This repository provides the algorithms and simulation testbeds:
- **PST (Performative Screening Test)**: Paired incumbent micro-perturbations against a control replica to detect endogenous feedback loops.
- **STS (Settle-Then-Score)**: Candidate deployment with dual-replica drift correction, plateau detection, and CUSUM guard to evaluate configurations only after workload settling.
- **Adaptive Stationarity Gate**: Automatically routes evaluations to STS during steady regimes (arrival-rate CoV < 0.06) or to short-window drift-corrected scoring under transient drifts.

---

## Repository Structure

```text
Black_box/
├── src/
│   ├── fluid.py                 # Mean-field fluid queue serving model (Testbed A)
│   ├── des.py                   # Stochastic discrete-event simulator (DES)
│   ├── cache.py                 # Queue-free cache & prefetch model (Testbed B)
│   ├── scalable.py              # Scalable d-dimensional system model
│   ├── feedback_models.py       # Endogenous retry dynamics (baseline, concave, sigmoid, threshold)
│   ├── workloads.py             # Workload patterns (Poisson, bursty ON-OFF, diurnal, Pareto)
│   ├── methods.py               # Algorithms: PST, plateau/CUSUM settling detection, STS
│   ├── optimizers.py            # Black-box optimizers (CMA-ES, BO-GP, TUNA-filter, NSGA-II, Random)
│   ├── optimize_util.py         # Multi-start global optimizer for reference optimum x_o
│   ├── exp_*.py                 # Experiment drivers (one script per evaluation)
│   ├── plots.py                 # Figure generator (vector PDF and 600 DPI PNG)
│   └── run_all.sh               # Execution script to run all experiments
├── requirements.txt             # Python dependencies
└── README.md                    # Repository documentation and instructions
```

---

## How to Simulate

All experiments are seed-controlled and deterministic.

### 1. Run the Primary Benchmark
Compare single-shot tuners against STS-equipped optimizers across feedback intensities ($\beta \in [0.0, 0.90]$):
```bash
cd src
python3 exp_optimizers.py
```

### 2. Run Independent Discrete-Event (DES) Validation
Validate that mean-field optimization transfers without model exploitation to a stochastic DES environment:
```bash
cd src
python3 exp_des_validation.py
```

### 3. Run Specific Experiments
```bash
cd src
python3 exp_ablation.py             # STS component ablation (Table III)
python3 exp_feedback_robustness.py  # Robustness to 4 retry feedback laws (Table IV)
python3 exp_scalability.py          # Dimensional scaling: d in {4, 8, 16, 32}
python3 exp_cache.py                # Testbed B: caching and prefetching
python3 exp_adaptive_workload.py    # Non-stationary bursty/diurnal workload gating
```

### 4. Run the Full Suite
To execute the entire suite of benchmarks sequentially:
```bash
cd src
bash run_all.sh
```

### 5. Generate Figures
To render preview and publication figures:
```bash
cd src
python3 plots.py
```

---

## Paper Mapping

| Paper Item | Description | Script (`src/`) |
|:---|:---|:---|
| **Fig. 3** | Threshold dynamics ($\beta^*, \beta^\dagger, \beta_{\text{col}}$) | `exp_thresholds.py` |
| **Table II, Fig. 4** | Primary optimizer benchmark (Single-shot vs. STS) | `exp_optimizers.py` |
| **Convergence** | Optimizer convergence trajectories over evaluations | `exp_convergence.py` |
| **Fig. 5** | Independent discrete-event simulator validation | `exp_des_validation.py` |
| **Sec. V-D** | Performative Screening Test (PST) detection power & FPR | `exp_screening.py` |
| **Settling / Drift** | Plateau/CUSUM settling time vs. ground truth | `exp_settling.py` |
| **Table III** | STS component ablation (settling, drift, stability) | `exp_ablation.py` |
| **Table IV** | Robustness across 4 endogenous feedback laws | `exp_feedback_robustness.py` |
| **Fig. 6** | Equal deployment-time evaluation budget comparison | `exp_repeated_eval.py` |
| **Sensitivity** | Hyperparameter sensitivity ($k, \delta, \gamma$) | `exp_sensitivity.py` |
| **Scalability** | Dimensionality scaling up to $d=32$ | `exp_scalability.py` |
| **Fig. 7** | Per-seed regret distribution across 100 seeds | `exp_seed_dist.py` |
| **Testbed B** | Queue-free cache & prefetch model evaluations | `exp_cache.py`, `exp_cache_alt.py` |
| **Pareto Frontier** | Bi-objective STS-NSGA-II (cost vs. stability) | `exp_nsga.py` |
| **Fig. 8** | Non-stationary workloads & adaptive stationarity gate | `exp_adaptive_workload.py`, `exp_workload.py` |
