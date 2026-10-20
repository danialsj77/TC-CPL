"""Fig. 7 of the revision (cityNL overload assessment), from the submitted
per-seed evaluations (results/cityNL/eval_cache). Changes from the submitted
figure: violation frequencies are counted per seed-day (rule-based controllers
per day), and the error bars are 95% bootstrap intervals (seeds x days)
instead of +/- one standard deviation. Writes IEEE TTE/figs/safety_summary_rev.png.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(REV)
EC = os.path.join(REPO, "results", "cityNL", "eval_cache")
RNG = np.random.default_rng(11)
CTRL = [("CAFAP", "heuristic_cafap", "#999999", "x"), ("Round-Robin", "heuristic_roundrobin", "#E69F00", "v"),
        ("DDPG-PST", "ddpg_pst", "#56B4E9", "^"), ("PPO-PST", "ppo_pst", "#CC79A7", "P"),
        ("CPO", "cpo_pst", "#0072B2", "o"), ("Fixed-price\nTC-CPL", "fixedpenalty", "#D55E00", "d"),
        ("Self-tuning\nTC-CPL", "tccpl", "#009E73", "D")]
BANDS = [0.0, 0.5, 1, 2, 3, 5, 10, 15, 20]


def g(v):
    while isinstance(v, (list, tuple)):
        v = v[0]
    return float(v)


def load(key):
    seeds = [0] if key.startswith("heuristic") else [42, 43, 44, 45, 46]
    out = []
    for s in seeds:
        st = json.load(open(os.path.join(EC, f"{key}_s{s}_n100_o500.json")))["stats"]
        out.append(st)
    return out


def arr(runs, k):
    return np.array([[g(e[k]) for e in st] for st in runs])          # (S, D)


def key_of(b):
    return "total_transformer_overload" if b == 0 else f"overload_kWh_tol{b:g}"


def boot_ci(X):
    S, D = X.shape
    reps = []
    for _ in range(4000):
        s = RNG.integers(0, S, S)
        d = RNG.integers(0, D, D)
        reps.append(X[s][:, d].mean())
    return np.percentile(reps, [2.5, 97.5])


def main():
    plt.rcParams.update({"font.size": 8.5})
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14.0, 3.9), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    x = np.arange(len(CTRL))
    m10, lo10, hi10, p10, mi, loi, hii = [], [], [], [], [], [], []
    sweep = {}
    for lab, key, col, mk in CTRL:
        runs = load(key)
        X = arr(runs, key_of(10))
        E = arr(runs, "total_energy_charged") / 1000.0
        I = X / np.maximum(E, 1e-9)
        m10.append(X.mean()); l, h = boot_ci(X); lo10.append(l); hi10.append(h)
        p10.append((X > 1e-9).mean())
        mi.append(I.mean()); l, h = boot_ci(I); loi.append(l); hii.append(h)
        sweep[lab] = [arr(runs, key_of(b)).mean() for b in BANDS]
    cols = [c for _, _, c, _ in CTRL]
    labels = [l for l, _, _, _ in CTRL]

    def bars(ax, m, lo, hi, ylab, tag, fmt, probs=None):
        m, lo, hi = map(np.array, (m, lo, hi))
        ax.bar(x, m, yerr=[m - lo, hi - m], capsize=3, color=cols, alpha=0.85, edgecolor="black", lw=0.6)
        for xi, v in zip(x, m):
            ax.annotate(fmt.format(v), (xi, v), textcoords="offset points", xytext=(9, 3), ha="left", fontsize=7.5,
                        bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 1.22 * float(np.max(hi)))
        ax.grid(axis="y", alpha=0.3)
        ax.text(-0.10, 1.04, tag, transform=ax.transAxes, fontsize=10.5, fontweight="bold", va="bottom")
        if probs is not None:
            ax.set_ylim(0, 1.32 * float(np.max(hi)))
            for xi, p in zip(x, probs):
                ax.text(xi, 0.975, f"{100 * p:.1f}%", transform=ax.get_xaxis_transform(), ha="center",
                        va="top", fontsize=8, fontweight="bold", color="#B22222")
            ax.text(0.5, 1.02, "share of seed-days with overload beyond the band", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=8, color="#B22222")

    bars(a1, m10, lo10, hi10, "overload beyond 10-kW band (kWh/day)", "(a)", "{:.3f}", p10)
    bars(a2, mi, loi, hii, "overload beyond 10-kW band\n(kWh per MWh delivered)", "(b)", "{:.4f}")
    for lab, _, col, mk in CTRL:
        a3.plot(BANDS, sweep[lab], color=col, marker=mk, ms=4, lw=1.4, label=lab)
    a3.axvline(10, color="#B22222", ls=":", lw=1.0)
    a3.set_xlabel("band $b$ (kW above each transformer's limit)")
    a3.set_ylabel("mean overload beyond the band (kWh/day)")
    a3.set_ylim(bottom=0)
    a3.grid(alpha=0.3)
    a3.legend(fontsize=7, ncol=2, frameon=False, loc="upper right")
    a3.text(-0.10, 1.04, "(c)", transform=a3.transAxes, fontsize=10.5, fontweight="bold", va="bottom")
    fig.tight_layout()
    out = os.path.join(REPO, "IEEE TTE", "figs", "safety_summary_rev.png")
    fig.savefig(out, dpi=300)
    print("wrote", out)
    for lab, m, p in zip(labels, m10, p10):
        print(f"  {lab.replace(chr(10), ' '):20s} {m:.3f} kWh/day  p10 {100 * p:.1f}%")


if __name__ == "__main__":
    main()
