"""Paired statistics on PublicPST-S: TC-CPL (tau=0) minus each variant, and
the effect of the headroom projection (with minus without, same checkpoints).

Same estimator as analyze_submitted.paired: the seed-averaged per-day
difference with a paired bootstrap over days, and a two-level bootstrap that
resamples the seeds of each controller independently and the days jointly.
Rule-based controllers are deterministic (one "seed"). Per-seed differences
are given for learners trained with the same seeds.

    python scripts/stress_stats.py        -> analysis/stress_stats.json
"""
import os, json, glob
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs", "PublicPST_stress")
B = 10_000
RNG = np.random.default_rng(0)
METRICS = {"etr": "energy_tracking_error", "sat": "average_user_satisfaction",
           "edel": "total_energy_charged", "ov": "total_transformer_overload",
           "b10": "ov_tol10", "nameplate": "ov_nameplate"}


import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import stress_run_is_current


def load(var):
    runs = sorted(p for p in glob.glob(os.path.join(RUNS, f"{var}_s*", "eval.json"))
                  if stress_run_is_current(os.path.dirname(p)))
    if not runs:
        return None, []
    seeds = [int(p.split("_s")[-1].split("/")[0]) for p in runs]
    days = [json.load(open(p))["days"] for p in runs]
    n = min(len(d) for d in days)
    X = {m: np.array([[float(d[i][k]) for i in range(n)] for d in days]) for m, k in METRICS.items()}
    return X, seeds


def paired(xa, xb, binary=False):
    if binary:
        xa, xb = (xa > 1e-9).astype(float), (xb > 1e-9).astype(float)
    diff = xa.mean(0) - xb.mean(0)
    D = len(diff)
    idx = RNG.integers(0, D, size=(B, D))
    boot_day = diff[idx].mean(1)
    two = np.empty(B)
    for k in range(B):
        sa = RNG.integers(0, xa.shape[0], xa.shape[0])
        sb = RNG.integers(0, xb.shape[0], xb.shape[0])
        two[k] = xa[sa][:, idx[k]].mean() - xb[sb][:, idx[k]].mean()
    return dict(mean_a=float(xa.mean()), mean_b=float(xb.mean()), diff=float(diff.mean()),
                ci_day=[float(np.percentile(boot_day, 2.5)), float(np.percentile(boot_day, 97.5))],
                ci_seed_day=[float(np.percentile(two, 2.5)), float(np.percentile(two, 97.5))],
                p_seed_day=float(min(1.0, 2 * min((two <= 0).mean(), (two >= 0).mean()))))


if __name__ == "__main__":
    A, sa = load("tccpl")
    if A is None:
        raise SystemExit("no PublicPST-S run of the current tccpl variant yet (revision/stress_sweep.sh)")
    out = {"reference": "tccpl", "seeds": sa, "comparisons": {}}
    names = sorted({os.path.basename(os.path.dirname(p)).rsplit("_s", 1)[0]
                    for p in glob.glob(os.path.join(RUNS, "*", "eval.json"))
                    if stress_run_is_current(os.path.dirname(p))} - {"tccpl"})
    for var in names:
        Xb, sb = load(var)
        res = {"seeds": sb}
        for m in METRICS:
            res[m] = paired(A[m], Xb[m])
            common = [s for s in sa if s in sb]
            if len(sb) > 1 and common:
                res[m]["per_seed_diff"] = [float((A[m][sa.index(s)] - Xb[m][sb.index(s)]).mean())
                                           for s in common]
        res["p_ov"] = paired(A["ov"], Xb["ov"], binary=True)
        res["p_b10"] = paired(A["b10"], Xb["b10"], binary=True)
        out["comparisons"][var] = res
    # headroom projection: the same checkpoints and days without / with it
    out["shield"] = {}
    for p in sorted(glob.glob(os.path.join(RUNS, "*", "eval_shield.json"))):
        var = os.path.basename(os.path.dirname(p)).rsplit("_s", 1)[0]
        if var in out["shield"]:
            continue
        base = sorted(glob.glob(os.path.join(RUNS, f"{var}_s*", "eval.json")))
        shield = [b.replace("eval.json", "eval_shield.json") for b in base]
        if not all(os.path.exists(x) for x in shield):
            continue
        # make_env applies the projection only to residual policies, so an
        # eval_shield.json of a direct-action variant is an unshielded rerun
        if not json.load(open(shield[0]))["spec"].get("residual", True):
            continue
        def stack(files, key):
            days = [json.load(open(f))["days"] for f in files]
            n = min(len(d) for d in days)
            return np.array([[float(d[i][key]) for i in range(n)] for d in days])
        res = {"seeds": len(base)}
        for m, k in METRICS.items():
            xs, xb = stack(shield, k), stack(base, k)
            res[m] = paired(xs, xb)          # shield minus without
        res["max_excess_kW_shield"] = float(stack(shield, "max_excess_kW").max())
        res["p_ov_shield"] = float((stack(shield, "total_transformer_overload") > 1e-9).mean())
        out["shield"][var] = res
    path = os.path.join(ROOT, "analysis", "stress_stats.json")
    json.dump(out, open(path, "w"), indent=1)
    for var, res in out["comparisons"].items():
        line = [f"{var:16s} n={len(res['seeds'])}"]
        for m in ("etr", "sat", "edel", "ov", "p_ov"):
            r = res[m]
            line.append(f"{m} {r['diff']:+.3f} [{r['ci_seed_day'][0]:+.3f},{r['ci_seed_day'][1]:+.3f}]")
        print("  ".join(line))
    for var, res in out["shield"].items():
        print(f"shield {var:14s} n={res['seeds']} max_exc {res['max_excess_kW_shield']:.4f} kW  p_ov {res['p_ov_shield']:.3f}  " +
              "  ".join(f"{m} {res[m]['mean_b']:.3f}->{res[m]['mean_a']:.3f} ({res[m]['diff']:+.3f} [{res[m]['ci_seed_day'][0]:+.3f},{res[m]['ci_seed_day'][1]:+.3f}])" for m in ("ov", "sat", "edel", "etr")))
    print("wrote", path)
