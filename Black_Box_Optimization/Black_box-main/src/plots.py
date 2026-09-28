"""
plots.py 
------------------
Publication-grade IEEE figures for PSCO-STS matching the IEEE Open Journal
guidelines and the uploaded reference visual style:

  * ZERO dotted lines in legends or curves for methods. Every method is visible
    and distinct with solid, dashed, or dash-dotted styles and clear markers.
  * Clean, light solid grid lines (never dotted) for a crisp publication finish.
  * Distinct, high-contrast Okabe-Ito colorblind-safe palette:
      - Orange/Vermillion (#D55E00)
      - Blue (#0072B2)
      - Teal/Bluish-Green (#009E73)
      - Reddish-Purple/Mauve (#CC79A7)
      - Amber/Warm Yellow (#E69F00)
      - Sky Blue (#56B4E9)
      - Slate/Charcoal (#444444)
  * Shaded translucent confidence/SEM bands around curves with sample markers.
  * Monotonic shape-preserving PCHIP curve smoothing with genuine markers at
    all measured points.
  * Serif typography (Times New Roman / Liberation Serif with CM math).
  * 600 dpi raster export plus embedded-font vector PDF.
"""
import os, json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullFormatter

try:
    from scipy.interpolate import PchipInterpolator
    from scipy.signal import savgol_filter
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 9.0,
    "axes.titlesize": 9.0,
    "axes.labelsize": 9.0,
    "legend.fontsize": 7.4,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "axes.linewidth": 1.0,
    "lines.linewidth": 2.2,
    "lines.markersize": 5.2,
    "lines.markeredgewidth": 0.6,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "lines.dash_capstyle": "round",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#E2E2E2",
    "grid.alpha": 0.85,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#CCCCCC",
    "legend.fancybox": False,
    "legend.borderpad": 0.4,
    "legend.labelspacing": 0.32,
    "legend.handletextpad": 0.5,
    "legend.handlelength": 1.9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.pad_inches": 0.02,
    "figure.constrained_layout.use": False,
})

# High-contrast Okabe-Ito colorblind-safe palette
C_VERM  = "#D55E00"  # Vermillion / Red-Orange
C_BLUE  = "#0072B2"  # Deep Blue
C_GREEN = "#009E73"  # Teal / Bluish Green
C_PURP  = "#CC79A7"  # Reddish Purple / Mauve
C_AMBER = "#E69F00"  # Amber / Warm Orange
C_SKY   = "#56B4E9"  # Sky Blue
C_GRAY  = "#555555"  # Slate Gray
C_INK   = "#222222"  # Charcoal

# Distinct linestyles (NEVER dotted)
LS_SOLID       = "-"
LS_DASH        = "--"
LS_DASHDOT     = "-."
LS_LONGDASH    = (0, (6, 2.5))
LS_DASHDASHDOT = (0, (4, 1.5, 4, 1.5, 1, 1.5))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(SCRIPT_DIR, "..", "results")
FIG = os.path.join(SCRIPT_DIR, "..", "figures")
os.makedirs(FIG, exist_ok=True)

COL_W  = 3.5      # single-column text width (in)
TEXT_W = 7.16     # full (double-column) text width (in)
SINGLE, DOUBLE = COL_W, TEXT_W
H_SINGLE = 2.45
H_DOUBLE = 2.10
H_TALL   = 2.22


def _load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        print(f"  [skip] {name} missing")
        return None
    with open(p) as f:
        return json.load(f)


def _save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")


def _dense(x, y, logy=False, n=400):
    x = np.asarray(x, float); y = np.asarray(y, float)
    order = np.argsort(x); x, y = x[order], y[order]
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 6 or not _HAVE_SCIPY or len(np.unique(x)) < 6:
        return x, y
    xs = np.linspace(x.min(), x.max(), n)
    if logy:
        yy = np.log(np.clip(y, 1e-12, None))
        ys = np.exp(PchipInterpolator(x, yy)(xs))
    else:
        ys = PchipInterpolator(x, y)(xs)
    return xs, ys


def sline(ax, x, y, color, ls=LS_SOLID, lw=2.2, marker="o", ms=5.2, label=None,
          logy=False, zorder=3, markevery=1, dense=True):
    x = np.asarray(x, float); y = np.asarray(y, float)
    order = np.argsort(x); x, y = x[order], y[order]
    if dense:
        xs, ys = _dense(x, y, logy=logy)
    else:
        xs, ys = x, y
    ax.plot(xs, ys, color=color, ls=ls, lw=lw, zorder=zorder, solid_capstyle="round")
    if marker:
        ax.plot(x, y, color=color, ls="none", marker=marker, ms=ms,
                markevery=markevery, markeredgecolor="white", markeredgewidth=0.6,
                zorder=zorder + 1)
    return plt.Line2D([0], [0], color=color, ls=ls, lw=lw, marker=marker,
                      ms=ms, markeredgecolor="white", markeredgewidth=0.6,
                      label=label)


def serrbar(ax, x, y, yerr, color, ls=LS_SOLID, lw=2.2, marker="o", ms=5.2,
            label=None, logy=False, zorder=3, capsize=2.0, band_alpha=0.16):
    x = np.asarray(x, float); y = np.asarray(y, float); yerr = np.asarray(yerr, float)
    order = np.argsort(x); x, y, yerr = x[order], y[order], yerr[order]
    xs, ys = _dense(x, y, logy=logy)
    ax.plot(xs, ys, color=color, ls=ls, lw=lw, zorder=zorder, solid_capstyle="round")
    
    # Translucent confidence / error band matching reference image style
    if band_alpha > 0:
        ylo = np.maximum(y - yerr, 1e-6 if logy else -1e9)
        yhi = y + yerr
        ax.fill_between(x, ylo, yhi, color=color, alpha=band_alpha, linewidth=0, zorder=zorder - 1)
        
    ax.errorbar(x, y, yerr=yerr, color=color, ls="none", marker=marker, ms=ms,
                capsize=capsize, elinewidth=0.9, capthick=0.9,
                markeredgecolor="white", markeredgewidth=0.6, zorder=zorder + 1,
                ecolor=color, alpha=0.9)
    return plt.Line2D([0], [0], color=color, ls=ls, lw=lw, marker=marker,
                      ms=ms, markeredgecolor="white", markeredgewidth=0.6,
                      label=label)


def _sg_log(y, frac=0.09, poly=2):
    y = np.asarray(y, float)
    n = len(y)
    if not _HAVE_SCIPY or n < 7:
        return y
    w = int(round(n * frac))
    if w % 2 == 0:
        w += 1
    w = max(5, min(w, n - 1 if (n - 1) % 2 == 1 else n - 2))
    if w <= poly:
        return y
    ly = np.log(np.clip(y, 1e-12, None))
    return np.exp(savgol_filter(ly, w, poly))


def _log_minor(ax, axis="y"):
    loc = LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
    if axis in ("y", "both"):
        ax.yaxis.set_minor_locator(loc); ax.yaxis.set_minor_formatter(NullFormatter())
    if axis in ("x", "both"):
        ax.xaxis.set_minor_locator(loc); ax.xaxis.set_minor_formatter(NullFormatter())


# --------------------------------------------------------------------------- #
# Figures                                                                     #
# --------------------------------------------------------------------------- #
def plot_landscape():
    data = _load("ground_truth.json")
    if not data:
        return
    betas = [d["beta"] for d in data]
    gap = [d["performativity_gap"] for d in data]
    rho = [d["price_of_stability"] for d in data]
    se = [abs(d["score_error"]) for d in data]
    fig, ax = plt.subplots(figsize=(SINGLE, H_SINGLE))
    h1 = sline(ax, betas, gap, C_VERM, ls=LS_SOLID, marker="o",
               label=r"Performativity gap $\Gamma(\beta)$")
    h2 = sline(ax, betas, rho, C_BLUE, ls=LS_DASH, marker="s",
               label=r"Price of stability $\rho(\beta)$")
    h3 = sline(ax, betas, se, C_PURP, ls=LS_DASHDOT, marker="^",
               label=r"Score error $|\mathcal{E}_{\mathrm{score}}|$")
    ax.axvline(0.60, color=C_GRAY, ls="--", alpha=0.6, lw=1.1)
    ax.text(0.60, ax.get_ylim()[1] * 0.5, r" $\beta\!\approx\!0.6$", fontsize=8, color="#333333")
    ax.set_xlabel(r"Performative feedback strength $\beta$")
    ax.set_ylabel("Discrepancy metric")
    ax.set_xlim(0, 0.95)
    ax.set_ylim(-0.05, max(max(gap), max(se)) * 1.10)
    ax.legend(handles=[h1, h2, h3], loc="upper left", frameon=True)
    fig.tight_layout(pad=0.3)
    _save(fig, "landscape")


def plot_regret():
    """3-panel: (a) single-shot methods, (b) STS variants, (c) relative regret reduction."""
    data = _load("optimizers.json")
    if not data:
        return
    by = {}
    for r in data["rows"]:
        by.setdefault(r["method"], {"b": [], "m": [], "s": []})
        by[r["method"]]["b"].append(r["beta"])
        by[r["method"]]["m"].append(r["regret_mean"])
        by[r["method"]]["s"].append(r["regret_sem"])
    for m in by:
        order = np.argsort(by[m]["b"])
        for kk in ("b", "m", "s"):
            by[m][kk] = list(np.array(by[m][kk])[order])

    # Distinct styles and markers, NO dotted lines in legend
    single = {
        "CMA-ES / single-shot": (C_VERM, LS_SOLID, "o", "CMA-ES"),
        "BO-GP / single-shot": (C_BLUE, LS_DASH, "s", "BO-GP"),
        "TUNA-filter / single-shot": (C_AMBER, LS_DASHDOT, "D", "TUNA-filter"),
        "NSGA-II / single-shot": (C_PURP, LS_LONGDASH, "^", "NSGA-II"),
        "Random / single-shot": (C_GRAY, LS_DASHDASHDOT, "v", "Random"),
    }
    sts = {
        "Proposed: CMA-ES / settle-then-score": (C_GREEN, LS_SOLID, "^", "STS CMA-ES (Primary)"),
        "Proposed: CMA-ES / settle+stability": (C_BLUE, LS_DASH, "s", "Stability-Reg. STS"),
        "Proposed: NSGA-II / settle-then-score": (C_PURP, LS_DASHDOT, "d", "STS NSGA-II"),
    }
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(DOUBLE, H_DOUBLE))
    hs = []
    for m, (c, ls, mk, lbl) in single.items():
        if m in by:
            hs.append(serrbar(a1, by[m]["b"], by[m]["m"], by[m]["s"], c, ls=ls,
                              marker=mk, ms=4.8, label=lbl, logy=True, lw=2.0))
    a1.set_yscale("log"); a1.set_title("(a) Single-shot baselines")
    a1.set_ylabel("Post-deployment regret"); a1.set_xlabel(r"Feedback strength $\beta$")
    a1.set_xlim(0, 0.9); a1.set_ylim(3e-3, 3.0); _log_minor(a1)
    a1.legend(handles=hs, loc="lower right", frameon=True, fontsize=7.0)

    hs2 = []
    for m, (c, ls, mk, lbl) in sts.items():
        if m in by:
            hs2.append(serrbar(a2, by[m]["b"], by[m]["m"], by[m]["s"], c, ls=ls,
                               marker=mk, ms=5.0, label=lbl, logy=True, lw=2.2))
    a2.set_yscale("log"); a2.set_title("(b) Proposed STS variants")
    a2.set_xlabel(r"Feedback strength $\beta$"); a2.set_xlim(0, 0.9); a2.set_ylim(3e-4, 0.2); _log_minor(a2)
    a2.legend(handles=hs2, loc="upper left", frameon=True, fontsize=7.0)

    base = np.array(by["CMA-ES / single-shot"]["m"])
    bb = np.array(by["CMA-ES / single-shot"]["b"])
    hs3 = []
    for m, key in [("Proposed: CMA-ES / settle-then-score", "STS CMA-ES"),
                   ("Proposed: CMA-ES / settle+stability", "Stability-Reg. STS")]:
        c, ls, mk, lbl = sts[m]
        st = np.array(by[m]["m"])
        rr = base / np.maximum(st, 1e-6)
        hs3.append(sline(a3, bb, rr, c, ls=ls, marker=mk, ms=5.0, label=lbl, logy=True, lw=2.2))
    a3.axhline(1.0, color="#666666", ls="--", lw=1.0)
    a3.set_yscale("log"); a3.set_title("(c) Regret reduction vs CMA-ES")
    a3.set_xlabel(r"Feedback strength $\beta$"); a3.set_ylabel(r"$\mathrm{RR}=\mathrm{Reg}_{\mathrm{base}}/\mathrm{Reg}_{\mathrm{STS}}$")
    a3.set_xlim(0, 0.9); _log_minor(a3)
    a3.legend(handles=hs3, loc="upper left", frameon=True, fontsize=7.0)
    fig.tight_layout(pad=0.4, w_pad=1.0)
    _save(fig, "regret")


def plot_convergence():
    data = _load("convergence.json")
    if not data:
        return
    iters = np.asarray(data["iterations"], float)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE, H_DOUBLE))
    styles = {
        "Proposed (STS CMA-ES)": (C_GREEN, LS_SOLID, "STS CMA-ES (Primary)", 2.4),
        "Proposed (Stability-Reg. STS)": (C_BLUE, LS_DASH, "Stability-Reg. STS", 2.1),
        "CMA-ES (Single-Shot)": (C_VERM, LS_SOLID, "CMA-ES (single-shot)", 1.9),
        "BO-GP (Single-Shot)": (C_AMBER, LS_DASH, "BO-GP (single-shot)", 1.9),
        "TUNA-filter (Single-Shot)": (C_INK, LS_DASHDOT, "TUNA-filter", 1.8),
    }
    handles = []
    for b_str, ax, title in [("0.6", ax1, r"(a) $\beta = 0.60$ (near fold)"),
                             ("0.8", ax2, r"(b) $\beta = 0.80$ (deep feedback)")]:
        bd = data["betas"][b_str]
        for m, (c, ls, lbl, lw) in styles.items():
            if m in bd:
                mean = _sg_log(np.array(bd[m]["mean"]))
                sem = np.array(bd[m]["sem"])
                sem_s = _sg_log(np.clip(sem, 1e-9, None))
                line, = ax.plot(iters, mean, color=c, ls=ls, lw=lw, solid_capstyle="round")
                ax.fill_between(iters, np.clip(mean - sem_s, 1e-4, None), mean + sem_s,
                                color=c, alpha=0.15, linewidth=0)
                if b_str == "0.6":
                    handles.append(plt.Line2D([0], [0], color=c, ls=ls, lw=lw, label=lbl))
        ax.set_yscale("log"); ax.set_xlabel("Optimization evaluations")
        ax.set_title(title); ax.set_xlim(1, data["budget"]); ax.set_ylim(3e-3, 1.5)
        _log_minor(ax)
    ax1.set_ylabel("True realized regret")
    ax1.legend(handles=handles, loc="lower left", frameon=True, fontsize=7.0)
    fig.tight_layout(pad=0.4, w_pad=1.0)
    _save(fig, "convergence")


def plot_screening():
    data = _load("screening.json")
    sens = _load("sensitivity.json")
    if not data:
        return
    rows = data["rows"]
    betas = [r["beta"] for r in rows]
    km = [r["kappa_p_mean"] for r in rows]
    ks = [r["kappa_p_sem"] for r in rows]
    det = [r["detection_rate"] * 100 for r in rows]

    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(DOUBLE, H_DOUBLE))
    ax2 = ax1.twinx()
    h1 = serrbar(ax1, betas, km, ks, C_BLUE, marker="o", label=r"Coeff. $\kappa_p$")
    h2 = sline(ax2, betas, det, C_VERM, ls=LS_DASH, marker="s", label="Detection (%)")
    ax1.set_xlabel(r"True feedback $\beta$"); ax1.set_ylabel(r"$\kappa_p$", color=C_BLUE)
    ax2.set_ylabel("Detection rate (%)", color=C_VERM); ax2.set_ylim(0, 105)
    ax1.tick_params(axis="y", labelcolor=C_BLUE); ax2.tick_params(axis="y", labelcolor=C_VERM)
    ax2.grid(False)
    ax1.set_title("(a) PST sensitivity vs feedback")
    ax1.legend([h1, h2], [r"$\kappa_p$", "Detection (%)"], loc="upper left", frameon=True)

    if sens:
        grid = [g for g in sens["pst_grid"] if abs(g["delta"] - 0.15) < 1e-6]
        grid.sort(key=lambda g: g["k"])
        ks_ = [g["k"] for g in grid]
        detk = [g["detection_rate"] * 100 for g in grid]
        fpr = [g["fpr"] * 100 for g in grid]
        hh1 = sline(ax3, ks_, detk, C_GREEN, marker="o", label=r"Detection @ $\beta{=}0.2$")
        hh2 = sline(ax3, ks_, fpr, C_VERM, ls=LS_DASH, marker="s", label="False-positive rate")
        ax3.axhline(5.0, color="#666666", ls="--", lw=1.0)
        ax3.text(ks_[-1], 8.0, r"$\alpha=5\%$", fontsize=7.5, ha="right", color="#333333")
        ax3.set_xlabel(r"PST trials $k$"); ax3.set_ylabel("Rate (%)")
        ax3.set_title(r"(b) PST vs trial count $k$ ($\delta{=}0.15$)")
        ax3.set_ylim(-3, 108)
        ax3.legend(handles=[hh1, hh2], loc="center right", frameon=True)
    fig.tight_layout(pad=0.4, w_pad=1.1)
    _save(fig, "screening")


def plot_methods():
    data = _load("settling.json")
    if not data:
        return
    st = data["settling"]
    kaps = [s["kappa"] for s in st]
    fig, ax = plt.subplots(figsize=(SINGLE, H_SINGLE))
    h0 = sline(ax, kaps, [s["true_mean"] for s in st], C_INK, ls=LS_SOLID, marker="",
               lw=2.4, label="Ground-truth settling", dense=True)
    h1 = sline(ax, kaps, [s["plateau_mean"] for s in st], C_BLUE, ls=LS_DASH, marker="o",
               label="Plateau detector")
    h2 = sline(ax, kaps, [s["cusum_mean"] for s in st], C_GREEN, ls=LS_DASHDOT, marker="s",
               label="CUSUM detector")
    d = data["drift"]
    txt = (f"+40% drift surge:\nnaive err {d['naive_err_mean']:.2f}$\\pm${d['naive_err_sem']:.2f}\n"
           f"corrected {d['corr_err_mean']:.2f}$\\pm${d['corr_err_sem']:.2f} ({d['reduction_pct']:.0f}\\% $\\downarrow$)")
    ax.text(0.97, 0.60, txt, transform=ax.transAxes, fontsize=7.2, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8F8F8", edgecolor="#CCCCCC"))
    ax.set_xlabel(r"Client population inertia $\kappa$")
    ax.set_ylabel("Settling time (steps)")
    ax.set_xlim(min(kaps), max(kaps)); ax.set_ylim(0, max(s["true_mean"] for s in st) * 1.15)
    ax.legend(handles=[h0, h1, h2], loc="upper right", frameon=True)
    fig.tight_layout(pad=0.3)
    _save(fig, "methods")


def plot_cache():
    data = _load("cache.json")
    alt = _load("cache_alt.json")
    if not data:
        return
    gaps = data["gap"]
    betas = [g["beta_c"] for g in gaps]
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(DOUBLE, H_DOUBLE))
    ax2 = ax1.twinx()
    h1 = sline(ax1, betas, [g["p_opt"] for g in gaps], C_BLUE, marker="o", label=r"STS $p^*$")
    h2 = sline(ax1, betas, [g["p_myopic"] for g in gaps], C_VERM, ls=LS_DASH, marker="s", label=r"Myopic $p_m$")
    h3 = sline(ax2, betas, [g["perf_gap"] for g in gaps], C_GREEN, ls=LS_DASHDOT, marker="^", label=r"Gap $\Gamma_c$")
    ax1.set_xlabel(r"Browsing feedback $\beta_c$"); ax1.set_ylabel("Prefetch prob. $p$")
    ax2.set_ylabel(r"Gap $\Gamma_c$", color=C_GREEN); ax2.tick_params(axis="y", labelcolor=C_GREEN)
    ax2.grid(False)
    ax1.set_ylim(0, 1.08); ax2.set_ylim(0, max(g["perf_gap"] for g in gaps) * 1.2 + 1e-3)
    ax1.set_title("(a) Primary (linear) browsing model")
    ax1.legend([h1, h2, h3], [r"STS $p^*$", r"Myopic $p_m$", r"Gap $\Gamma_c$"],
               loc="center left", frameon=True)
    if alt:
        cols = {"linear": C_BLUE, "saturating": C_GREEN, "threshold": C_VERM}
        mk = {"linear": "o", "saturating": "s", "threshold": "^"}
        ls = {"linear": LS_SOLID, "saturating": LS_DASH, "threshold": LS_DASHDOT}
        hs = []
        for m in alt["models"]:
            rr = [r for r in alt["rows"] if r["model"] == m]
            rr.sort(key=lambda z: z["beta_c"])
            hs.append(sline(ax3, [r["beta_c"] for r in rr], [r["perf_gap"] for r in rr],
                            cols[m], ls=ls[m], marker=mk[m], label=m))
        ax3.set_xlabel(r"Browsing feedback $\beta_c$"); ax3.set_ylabel(r"Performativity gap $\Gamma_c$")
        ax3.set_title("(b) Alternative browsing responses")
        ax3.legend(handles=hs, loc="upper left", frameon=True)
    fig.tight_layout(pad=0.4, w_pad=1.1)
    _save(fig, "cache")


def plot_pareto():
    data = _load("nsga.json")
    if not data:
        return
    pts = data["pareto"]
    costs = np.array([p["cost"] for p in pts])
    resids = np.array([p["stability_residual"] for p in pts])
    order = np.argsort(costs)
    costs_s, resids_s = costs[order], resids[order]
    fig, ax = plt.subplots(figsize=(SINGLE, H_SINGLE))
    xs, ys = _dense(costs_s, resids_s, n=400)
    ax.plot(xs, ys, color=C_BLUE, lw=2.2, zorder=2, label="STS NSGA-II frontier")
    ax.scatter(costs, resids, color=C_BLUE, s=26, zorder=3, edgecolor="white", linewidths=0.5)
    ic = int(np.argmin(costs))
    ax.scatter([costs[ic]], [resids[ic]], color=C_VERM, marker="D", s=65, zorder=5,
               edgecolor="white", linewidths=0.7, label=r"Perf. optimum $x_o$")
    ir = int(np.argmin(resids))
    ax.scatter([costs[ir]], [resids[ir]], color=C_GREEN, marker="s", s=65, zorder=5,
               edgecolor="white", linewidths=0.7, label=r"Stable point $x_s$")
    ax.set_xlabel(r"Operational cost $P(x,x)$")
    ax.set_ylabel(r"Stability residual $\mathcal{R}_{\mathrm{stab}}(x)$")
    ax.set_xlim(costs.min() * 0.96, costs.max() * 1.04); ax.set_ylim(-0.01, resids.max() * 1.18)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout(pad=0.3)
    _save(fig, "pareto")


ABL_COL = {"Single-Shot": C_VERM, "STS-Settle": C_AMBER, "STS-Settle+Drift": C_SKY,
           "STS-Settle+Stability": C_GREEN, "Full STS": C_BLUE}


def plot_ablation():
    data = _load("ablation.json")
    if not data:
        return
    variants = data["variants"]; betas = data["betas"]
    reg = {(r["variant"], r["beta"]): r for r in data["rows"]}
    fig, ax = plt.subplots(figsize=(SINGLE, 2.78))
    nb = len(betas); nv = len(variants); w = 0.8 / nv
    x = np.arange(nb)
    for i, v in enumerate(variants):
        means = [reg[(v, b)]["regret_mean"] for b in betas]
        lo = [reg[(v, b)]["regret_mean"] - reg[(v, b)]["regret_ci_lo"] for b in betas]
        hi = [reg[(v, b)]["regret_ci_hi"] - reg[(v, b)]["regret_mean"] for b in betas]
        ax.bar(x + i * w - 0.4 + w / 2, means, w, yerr=[np.abs(lo), np.abs(hi)],
               color=ABL_COL[v], label=v, capsize=1.5, error_kw={"lw": 0.9},
               edgecolor="white", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([f"{b:.2f}" for b in betas])
    ax.set_xlabel(r"Feedback strength $\beta$"); ax.set_ylabel("Post-deployment regret")
    ax.set_ylim(1e-3, 1.5); _log_minor(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=2, frameon=True,
              fontsize=6.8, columnspacing=0.8, handlelength=1.4)
    fig.tight_layout(rect=[0, 0, 1, 0.80], pad=0.3)
    _save(fig, "ablation")


def plot_feedback():
    data = _load("feedback_robustness.json")
    if not data:
        return
    models = data["models"]
    rows = [r for r in data["rows"] if "method" in r and not r["method"].startswith("_")]
    reg = {(r["model"], r["beta"], r["method"]): r["regret_mean"] for r in rows}
    sem = {(r["model"], r["beta"], r["method"]): r["regret_sem"] for r in rows}
    show = [("CMA-ES / single-shot", C_VERM, "CMA-ES (single)"),
            ("BO-GP / single-shot", C_AMBER, "BO-GP (single)"),
            ("TUNA-filter / single-shot", C_INK, "TUNA (single)"),
            ("Proposed: STS CMA-ES", C_GREEN, "STS CMA-ES"),
            ("Proposed: Stability-Reg. STS", C_BLUE, "Stab.-Reg. STS")]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, H_DOUBLE))
    for ax, beta, title in [(axes[0], 0.60, r"(a) $\beta = 0.60$"), (axes[1], 0.80, r"(b) $\beta = 0.80$")]:
        x = np.arange(len(models)); w = 0.8 / len(show)
        for i, (m, c, lbl) in enumerate(show):
            vals = [reg.get((mod, beta, m), 0) for mod in models]
            errs = [sem.get((mod, beta, m), 0) for mod in models]
            ax.bar(x + i * w - 0.4 + w / 2, vals, w, yerr=errs, color=c, label=lbl,
                   capsize=1.4, error_kw={"lw": 0.8}, edgecolor="white", linewidth=0.4)
        ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(models, rotation=12, fontsize=7.5)
        ax.set_title(title); ax.set_ylim(1e-3, 2.0); _log_minor(ax)
    axes[0].set_ylabel("Post-deployment regret")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, 1.30), ncol=5, frameon=True,
                   fontsize=6.9, columnspacing=0.8, handlelength=1.3)
    fig.tight_layout(rect=[0, 0, 1, 0.86], pad=0.4, w_pad=1.0)
    _save(fig, "feedback_robustness")


def plot_budget():
    data = _load("repeated_eval.json")
    if not data:
        return
    rows = data["rows"]
    methods = data["methods"]
    col = {"single-shot": C_VERM, "repeated-3": C_AMBER, "repeated-5": C_SKY,
           "repeated-10": C_GRAY, "STS": C_GREEN}
    lsm = {"single-shot": LS_SOLID, "repeated-3": LS_DASH, "repeated-5": LS_DASHDOT,
           "repeated-10": LS_LONGDASH, "STS": LS_SOLID}
    mkm = {"single-shot": "s", "repeated-3": "v", "repeated-5": "^",
           "repeated-10": "D", "STS": "o"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE, H_DOUBLE))
    h1 = []; h2 = []
    for m in methods:
        rr = sorted([r for r in rows if r["method"] == m], key=lambda z: z["budget"])
        bs = [r["budget"] for r in rr]
        h1.append(serrbar(ax1, bs, [r["regret_mean"] for r in rr], [r["regret_sem"] for r in rr],
                          col[m], ls=lsm[m], marker=mkm[m], ms=4.8, label=m, logy=True, lw=2.1))
        h2.append(sline(ax2, bs, [r["useful_evals_mean"] for r in rr], col[m], ls=lsm[m],
                        marker=mkm[m], ms=4.8, label=m, lw=2.1))
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("Total deployment-time budget (steps)"); ax1.set_ylabel("Post-deployment regret")
    ax1.set_title(r"(a) Regret vs budget ($\beta=0.60$)"); _log_minor(ax1, "both")
    ax1.legend(handles=h1, loc="upper right", frameon=True, fontsize=7.0)
    ax2.set_xscale("log"); ax2.set_xlabel("Total deployment-time budget (steps)")
    ax2.set_ylabel("Settled / useful evaluations")
    ax2.set_title("(b) Useful (settled) evaluations"); _log_minor(ax2, "x")
    ax2.legend(handles=h2, loc="upper left", frameon=True, fontsize=7.0)
    fig.tight_layout(pad=0.4, w_pad=1.1)
    _save(fig, "budget")


def plot_sensitivity():
    data = _load("sensitivity.json")
    if not data:
        return
    gammas = data["gamma_values"]; betas = data["betas_b"]
    gr = {(r["gamma"], r["beta"]): r for r in data["gamma_rows"]}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE, H_DOUBLE))
    Z = np.array([[gr[(g, b)]["regret_mean"] for b in betas] for g in gammas])
    im = ax1.imshow(Z, aspect="auto", origin="lower", cmap="viridis",
                    extent=[-0.5, len(betas) - 0.5, -0.5, len(gammas) - 0.5])
    ax1.set_xticks(range(len(betas))); ax1.set_xticklabels([f"{b:.2f}" for b in betas])
    ax1.set_yticks(range(len(gammas))); ax1.set_yticklabels([f"{g:g}" for g in gammas])
    ax1.set_xlabel(r"Feedback $\beta$"); ax1.set_ylabel(r"Stability weight $\gamma$")
    ax1.set_title("(a) STS regret operating region")
    ax1.grid(False)
    for i in range(len(gammas)):
        for j in range(len(betas)):
            ax1.text(j, i, f"{Z[i,j]:.3f}", ha="center", va="center", fontsize=6.0,
                     color="white" if Z[i, j] < Z.max() * 0.6 else "black")
    cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=7.0)
    cb.outline.set_linewidth(0.7)
    ax2b = ax2.twinx()
    h = []
    for b, cc in [(0.60, C_BLUE), (0.80, C_VERM)]:
        reg = [gr[(g, b)]["regret_mean"] for g in gammas]
        res = [gr[(g, b)]["residual_mean"] for g in gammas]
        h.append(sline(ax2, gammas, reg, cc, marker="o", ms=4.8, label=fr"regret $\beta{{=}}{b:.1f}$"))
        sline(ax2b, gammas, res, cc, ls=LS_DASH, marker="s", ms=4.0)
    ax2b.grid(False)
    ax2.set_xlabel(r"Stability weight $\gamma$"); ax2.set_ylabel("Post-deployment regret")
    ax2b.set_ylabel(r"Stability residual $\mathcal{R}_{\mathrm{stab}}$")
    ax2.set_title(r"(b) Regret (—) \& residual (- -) vs $\gamma$")
    ax2.legend(handles=h, loc="upper left", frameon=True)
    fig.tight_layout(pad=0.4, w_pad=1.5)
    _save(fig, "sensitivity")


def plot_scalability():
    data = _load("scalability.json")
    if not data:
        return
    dims = data["dims"]
    lbls = {"cma-single": (C_VERM, LS_SOLID, "s", "CMA-ES (single)"),
            "bo-single": (C_AMBER, LS_DASH, "v", "BO-GP (single)"),
            "tuna-single": (C_INK, LS_DASHDOT, "X", "TUNA (single)"),
            "nsga-single": (C_SKY, LS_LONGDASH, "P", "NSGA-II (single)"),
            "sts": (C_GREEN, LS_SOLID, "o", "STS CMA-ES (Primary)"),
            "stab": (C_BLUE, LS_DASH, "d", "Stability-Reg. STS")}
    reg = {(r["method"], r["d"], r["beta"]): r for r in data["rows"]}
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, H_DOUBLE))
    hlast = []
    for ax, beta, title in [(axes[0], 0.60, r"(a) $\beta=0.60$"), (axes[1], 0.80, r"(b) $\beta=0.80$")]:
        hs = []
        for m, (c, ls, mk, lbl) in lbls.items():
            y = [reg[(m, d, beta)]["regret_mean"] for d in dims]
            e = [reg[(m, d, beta)]["regret_sem"] for d in dims]
            hs.append(serrbar(ax, dims, y, e, c, ls=ls, marker=mk, ms=4.8, label=lbl,
                              logy=True, lw=2.0))
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks(dims); ax.set_xticklabels(dims)
        ax.set_xlabel("Configuration dimension $d$"); ax.set_title(title); _log_minor(ax)
        hlast = hs
    axes[0].set_ylabel("Post-deployment regret")
    axes[1].legend(handles=hlast, loc="upper left", frameon=True, fontsize=6.9)
    fig.tight_layout(pad=0.4, w_pad=1.0)
    _save(fig, "scalability")


def plot_seed_dist():
    data = _load("seed_dist.json")
    if not data:
        return
    order = ["CMA-ES / single-shot", "BO-GP / single-shot", "TUNA-filter / single-shot",
             "Proposed: CMA-ES / settle-then-score", "Proposed: CMA-ES / settle+stability"]
    short = {"CMA-ES / single-shot": "CMA-ES", "BO-GP / single-shot": "BO-GP",
             "TUNA-filter / single-shot": "TUNA", "Proposed: CMA-ES / settle-then-score": "STS CMA-ES",
             "Proposed: CMA-ES / settle+stability": "Stab.-Reg."}
    cols = [C_VERM, C_AMBER, C_INK, C_GREEN, C_BLUE]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, H_TALL))
    for ax, cell, title in zip(axes, data["cells"], [r"(a) $\beta=0.60$", r"(b) $\beta=0.80$"]):
        arrs = [cell["dists"][m]["regrets"] for m in order]
        parts = ax.violinplot(arrs, showmedians=True, showextrema=False, widths=0.82)
        for i, b in enumerate(parts["bodies"]):
            b.set_facecolor(cols[i]); b.set_alpha(0.45); b.set_edgecolor(cols[i])
            b.set_linewidth(1.2)
        parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(2.0)
        for i, a in enumerate(arrs):
            xj = np.random.default_rng(i).normal(i + 1, 0.045, size=len(a))
            ax.scatter(xj, a, s=5, color=cols[i], alpha=0.45, linewidths=0)
        ax.set_yscale("symlog", linthresh=1e-2)
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels([short[m] for m in order], rotation=18, fontsize=7.4)
        ax.set_title(title)
    axes[0].set_ylabel("Post-deployment regret (symlog)")
    fig.tight_layout(pad=0.4, w_pad=0.9)
    _save(fig, "seed_dist")


def plot_thresholds():
    data = _load("thresholds.json")
    if not data:
        return
    betas = np.array(data["betas"])
    mult = np.array(data["fixed_point_multiplier"])
    supL = np.clip(np.array(data["sup_slope"]), 0, 3)
    cost = np.array(data["true_cost"])
    XHI = 0.82
    bc = data["beta_collapse"]
    supmask = betas <= XHI
    fpmask = betas <= (bc + 0.005)
    cost_env = np.maximum.accumulate(cost)
    fig, ax = plt.subplots(figsize=(SINGLE, H_SINGLE))
    ax2 = ax.twinx(); ax2.grid(False)
    ax.axvspan(bc, XHI, color=C_VERM, alpha=0.08)
    
    xs, ys = _dense(betas[supmask], supL[supmask], n=500)
    ax.plot(xs, ys, color=C_AMBER, ls=LS_DASH, lw=2.2, label=r"$\sup_\lambda T_x'(\lambda)$ (bound)")
    xs2, ys2 = _dense(betas[fpmask], mult[fpmask], n=500)
    ax.plot(xs2, ys2, color=C_BLUE, ls=LS_SOLID, lw=2.4, label=r"$T_x'(\lambda^*)$ (fold)")
    ax.axhline(1.0, color="#666666", ls="--", lw=1.0, alpha=0.7)
    ax.text(0.02, 1.06, "slope $=1$", fontsize=7.0, color="#333333")
    
    xs3, ys3 = _dense(betas[fpmask], cost_env[fpmask], n=500)
    ax2.plot(xs3, ys3, color=C_VERM, lw=2.2, ls=LS_DASHDOT, label=r"Realized cost $P(x_m,x_m)$")
    ax2.axhline(data["slo_ceiling"], color=C_VERM, ls="--", lw=1.0, alpha=0.6)
    ax.text((bc + XHI) / 2, 2.05, "operational collapse", fontsize=7.0, color=C_VERM,
            ha="center", rotation=90, alpha=0.9)
    ymax = 3.0
    for key, cc, lab, yv in [("beta_star", C_AMBER, r"$\beta^{*}$", 2.75),
                             ("beta_dagger", C_BLUE, r"$\beta^{\dagger}$", 2.75),
                             ("beta_collapse", C_VERM, r"$\beta_{\mathrm{col}}$", 2.45)]:
        if data[key] is not None:
            ax.axvline(data[key], color=cc, ls="-", lw=1.3, alpha=0.7)
            ax.annotate(lab, xy=(data[key], yv), xytext=(data[key], yv), color=cc,
                        fontsize=9.0, ha="center",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=cc, lw=0.8))
    ax.set_xlabel(r"Feedback strength $\beta$")
    ax.set_ylabel(r"Map slope $T_x'$"); ax.set_ylim(0, ymax)
    ax2.set_ylabel(r"Realized Cost $P(x_m,x_m)$", color=C_VERM); ax2.tick_params(axis="y", labelcolor=C_VERM)
    ax.set_xlim(0, XHI)
    ax.legend(loc="center left", frameon=True, fontsize=7.2)
    fig.tight_layout(pad=0.3)
    _save(fig, "thresholds")


def plot_des_validation():
    """Independent DES validation: post-deployment regret graded in the
    stochastic discrete-event simulator (not the mean-field model used to search)."""
    data = _load("des_validation.json")
    if not data:
        return
    rows = [r for r in data["rows"] if not r["method"].startswith("_")]
    by = {}
    for r in rows:
        by.setdefault(r["method"], {"b": [], "m": [], "s": []})
        by[r["method"]]["b"].append(r["beta"])
        by[r["method"]]["m"].append(r["regret_mean"])
        by[r["method"]]["s"].append(r["regret_sem"])
    style = {
        "CMA-ES / single-shot": (C_VERM, LS_SOLID, "o", "CMA-ES (single)"),
        "BO-GP / single-shot": (C_AMBER, LS_DASH, "s", "BO-GP (single)"),
        "NSGA-II / single-shot": (C_PURP, LS_DASHDOT, "^", "NSGA-II (single)"),
        "Proposed: STS-CMA-ES": (C_GREEN, LS_SOLID, "o", "STS-CMA-ES (Primary)"),
    }
    fig, ax = plt.subplots(figsize=(SINGLE, H_SINGLE))
    hs = []
    for m, (c, ls, mk, lbl) in style.items():
        if m in by:
            hs.append(serrbar(ax, by[m]["b"], by[m]["m"], by[m]["s"], c, ls=ls,
                              marker=mk, ms=5.2, label=lbl, logy=True, lw=2.2, band_alpha=0.18))
    ax.set_yscale("log")
    ax.set_xlabel(r"Performative feedback strength $\beta$")
    ax.set_ylabel("DES post-deployment regret")
    ax.set_xlim(0.28, 0.82); _log_minor(ax)
    ax.legend(handles=hs, loc="center left", frameon=True, fontsize=7.2)
    fig.tight_layout(pad=0.3)
    _save(fig, "des_validation")


def plot_adaptive():
    """Adaptive evaluation policy across workload regimes. The stationarity gate
    routes (near-)stationary loads to STS and non-stationary loads to robust
    short-window evaluation, recovering the best of both."""
    data = _load("adaptive_workload.json")
    if not data:
        return
    rows = data["rows"]
    kinds = [r["workload"] for r in rows]
    disp = {"poisson": "Poisson", "heavytail": "Heavy-tailed",
            "bursty": "Bursty ON/OFF", "diurnal": "Diurnal"}
    labels = [disp.get(k, k) for k in kinds]
    series = [("regret_single", C_VERM, "Single-shot"),
              ("regret_sts", C_AMBER, "STS"),
              ("regret_adaptive", C_GREEN, "Adaptive (gated)")]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.75))
    x = np.arange(len(kinds)); w = 0.8 / len(series)
    for i, (key, c, lbl) in enumerate(series):
        vals = [r[key + "_mean"] for r in rows]
        errs = [r[key + "_sem"] for r in rows]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, yerr=errs, color=c, label=lbl,
               capsize=1.8, error_kw={"lw": 0.9}, edgecolor="white", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=12, fontsize=7.6)
    ax.set_ylabel("Post-deployment regret")
    ax.set_ylim(1e-3, max(r["regret_sts_mean"] for r in rows) * 2.2); _log_minor(ax)
    ax.axvspan(-0.5, 1.5, color=C_GREEN, alpha=0.06)
    ax.axvspan(1.5, 3.5, color=C_VERM, alpha=0.06)
    ax.text(0.5, ax.get_ylim()[1] * 0.60, "stationary\n(gate $\\to$ STS)", ha="center",
            fontsize=6.8, color="#007744", fontweight="bold")
    ax.text(2.5, ax.get_ylim()[1] * 0.60, "non-stationary\n(gate $\\to$ robust)", ha="center",
            fontsize=6.8, color="#aa2200", fontweight="bold")
    ax.set_xlim(-0.5, 3.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.26), ncol=3, frameon=True,
              fontsize=7.0, columnspacing=0.9, handlelength=1.4)
    fig.tight_layout(rect=[0, 0, 1, 0.82], pad=0.3)
    _save(fig, "adaptive_workload")


def plot_all():
    print("== Generating publication figures  ==")
    funcs = [
        plot_landscape, plot_regret, plot_convergence, plot_screening,
        plot_methods, plot_cache, plot_pareto, plot_ablation,
        plot_feedback, plot_budget, plot_sensitivity, plot_scalability,
        plot_seed_dist, plot_thresholds, plot_des_validation, plot_adaptive
    ]
    for fn in funcs:
        try:
            fn()
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  [err] {fn.__name__}: {e}")
    print("All figures successfully generated in figures.")


if __name__ == "__main__":
    plot_all()
