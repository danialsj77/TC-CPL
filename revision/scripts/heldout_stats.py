"""Paired statistics on the 100 held-out cityNL days (simulator seeds 5000-5099),
which no demonstration of the submitted runs used: TC-CPL minus the fixed-price
ablation, round-robin and CAFAP, with the estimator of analyze_submitted.paired
(per-day difference of seed means; two-level bootstrap over seeds and days).

    python scripts/heldout_stats.py        -> analysis/heldout_stats.json
"""
import os, json, glob
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EV = os.path.join(ROOT, "runs", "cityNL_eval")
B = 10_000
RNG = np.random.default_rng(0)
METRICS = {"etr": "energy_tracking_error", "sat": "average_user_satisfaction",
           "edel": "total_energy_charged", "ov": "total_transformer_overload", "b10": "ov_tol10"}
LABELS = {"tccpl": "TC-CPL", "fixedpenalty": "Fixed-price SAC", "RR": "Round-Robin", "CAFAP": "CAFAP"}


import re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import output_is_current
REPO = os.path.dirname(ROOT)


def load(label):
    files = sorted(f for f in glob.glob(os.path.join(EV, f"{label}_s*_n100_o5000.json"))
                   if output_is_current(f, REPO, "cityNL", label,
                                        int(re.search(r"_s(\d+)_n100", f).group(1))))
    days = [json.load(open(f))["days"] for f in files]
    return {m: np.array([[float(d[k]) for d in dd] for dd in days]) for m, k in METRICS.items()}, len(files)


def paired(xa, xb, binary=False):
    if binary:
        xa, xb = (xa > 1e-9).astype(float), (xb > 1e-9).astype(float)
    diff = xa.mean(0) - xb.mean(0)
    D = len(diff)
    idx = RNG.integers(0, D, size=(B, D))
    two = np.empty(B)
    for k in range(B):
        sa = RNG.integers(0, xa.shape[0], xa.shape[0])
        sb = RNG.integers(0, xb.shape[0], xb.shape[0])
        two[k] = xa[sa][:, idx[k]].mean() - xb[sb][:, idx[k]].mean()
    return dict(mean_a=float(xa.mean()), mean_b=float(xb.mean()), diff=float(diff.mean()),
                ci_seed_day=[float(np.percentile(two, 2.5)), float(np.percentile(two, 97.5))],
                p_seed_day=float(min(1.0, 2 * min((two <= 0).mean(), (two >= 0).mean()))))


if __name__ == "__main__":
    A, na = load("tccpl")
    out = {"days": "seeds 5000-5099", "tccpl_seeds": na, "summary": {}, "paired": {}}
    for lab in LABELS:
        X, n = load(lab)
        out["summary"][LABELS[lab]] = {"seeds": n, **{m: float(X[m].mean()) for m in METRICS},
                                       "p_ov": float((X["ov"] > 1e-9).mean()), "p_b10": float((X["b10"] > 1e-9).mean())}
        if lab != "tccpl":
            r = {m: paired(A[m], X[m]) for m in METRICS}
            r["p_b10"] = paired(A["b10"], X["b10"], binary=True)
            r["p_ov"] = paired(A["ov"], X["ov"], binary=True)
            out["paired"][LABELS[lab]] = r
    os.makedirs(os.path.join(ROOT, "analysis"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "analysis", "heldout_stats.json"), "w"), indent=1)
    for k, v in out["summary"].items():
        print(f"{k:16s} n={v['seeds']} " + " ".join(f"{m} {v[m]:.4g}" for m in ("etr", "sat", "edel", "ov", "b10", "p_ov", "p_b10")))
    for k, r in out["paired"].items():
        print(f"TC-CPL minus {k:16s} " + "  ".join(f"{m} {r[m]['diff']:+.4g} [{r[m]['ci_seed_day'][0]:+.4g},{r[m]['ci_seed_day'][1]:+.4g}]" for m in ("etr", "sat", "edel", "ov", "b10", "p_b10")))
