"""Analyses that need NO new simulation: statistics on the submitted cityNL /
PublicPST evaluations (results/<scenario>/eval_cache, 5 seeds x 100 paired days)
and the dual-controller traces of all 5 seeds (logs/sweep/*.log).

    python scripts/analyze_submitted.py

Writes revision/analysis/submitted_stats.json and revision/analysis/*.png.
"""
import glob, json, os, re, sys
import numpy as np
from scipy import stats

REV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(REV)
OUT = os.path.join(REV, "analysis")
os.makedirs(OUT, exist_ok=True)
SEEDS = [42, 43, 44, 45, 46]
LEARNED = {"TC-CPL": "tccpl", "Fixed-price SAC": "fixedpenalty", "DDPG-PST": "ddpg_pst",
           "PPO-PST": "ppo_pst", "CPO": "cpo_pst"}
HEUR = {"Round-Robin": "heuristic_roundrobin_s0", "CAFAP": "heuristic_cafap_s0"}
METRICS = {"etr": "energy_tracking_error", "sqtr": "tracking_error",
           "sat": "average_user_satisfaction", "edel": "total_energy_charged",
           "ov": "total_transformer_overload", "b10": "overload_kWh_tol10",
           "b5": "overload_kWh_tol5", "b2": "overload_kWh_tol2", "b1": "overload_kWh_tol1",
           "rel1": "overload_kWh_rel1", "rel2": "overload_kWh_rel2",
           "rel3": "overload_kWh_rel3", "rel5": "overload_kWh_rel5",
           "nameplate": "overload_kWh_nameplate", "unmet": "unmet_demand_pct"}
RNG = np.random.default_rng(2026)
B = 10_000


def g(v):
    while isinstance(v, (list, tuple)):
        v = v[0]
    return float(v)


def load(scen, key):
    d = json.load(open(f"{REPO}/results/{scen}/eval_cache/{key}_n100_o500.json"))["stats"]
    return {m: np.array([g(e[k]) if k in e else np.nan for e in d]) for m, k in METRICS.items()}


def arrays(scen):
    """{label: {metric: (S, D) array}}; heuristics have S = 1."""
    A = {}
    for lab, key in LEARNED.items():
        per = [load(scen, f"{key}_s{s}") for s in SEEDS]
        A[lab] = {m: np.stack([p[m] for p in per]) for m in METRICS}
    for lab, key in HEUR.items():
        p = load(scen, key)
        A[lab] = {m: p[m][None, :] for m in METRICS}
    return A


def summary(A):
    out = {}
    for lab, M in A.items():
        r = {}
        for m, x in M.items():
            r[m] = float(np.nanmean(x))
            r[m + "_seed_means"] = [float(v) for v in np.nanmean(x, axis=1)]
        for m in ("ov", "b1", "b2", "b5", "b10", "rel1", "rel2", "rel3", "rel5", "nameplate"):
            v = M[m] > 1e-9
            r[f"p_{m}_pooled"] = float(v.mean())                     # over seed-days
            r[f"p_{m}_per_seed"] = [float(x) for x in v.mean(axis=1)]
            r[f"p_{m}_anyseed"] = float((np.nanmean(M[m], 0) > 1e-9).mean())  # as in submission
        out[lab] = r
    return out


def paired(A, a, b, m, binary=False):
    """TC-CPL (a) minus comparator (b) on metric m.
    day-level: seed-averaged per-day difference, paired bootstrap over days.
    two-level: resample seeds of each learner independently and days jointly."""
    xa, xb = A[a][m], A[b][m]
    if binary:
        xa, xb = (xa > 1e-9).astype(float), (xb > 1e-9).astype(float)
    da, db = np.nanmean(xa, 0), np.nanmean(xb, 0)
    diff = da - db
    D = len(diff)
    idx = RNG.integers(0, D, size=(B, D))
    boot_day = diff[idx].mean(1)
    Sa, Sb = xa.shape[0], xb.shape[0]
    two = np.empty(B)
    for k in range(B):
        sa = RNG.integers(0, Sa, Sa)
        sb = RNG.integers(0, Sb, Sb)
        dd = idx[k]
        two[k] = xa[sa][:, dd].mean() - xb[sb][:, dd].mean()
    res = dict(mean_a=float(np.nanmean(xa)), mean_b=float(np.nanmean(xb)),
               diff=float(diff.mean()),
               ci_day=[float(np.percentile(boot_day, 2.5)), float(np.percentile(boot_day, 97.5))],
               ci_seed_day=[float(np.percentile(two, 2.5)), float(np.percentile(two, 97.5))],
               p_two_sided_seed_day=float(2 * min((two <= 0).mean(), (two >= 0).mean())))
    nz = diff[np.abs(diff) > 1e-12]
    if len(nz) >= 5 and not binary:
        res["wilcoxon_p_day"] = float(stats.wilcoxon(nz).pvalue)
    # per-seed paired differences (learner vs learner only)
    if Sa == Sb == len(SEEDS):
        res["per_seed_diff"] = [float(np.nanmean(xa[s] - xb[s])) for s in range(Sa)]
    return res


# ── dual-controller traces from the training logs ───────────────────────────
LINE = re.compile(r"\[(M1|M1-frozen) @\s*(\d+)\] overload above tightened limit=([\d.]+) kWh "
                  r"\(true ([\d.na]+)\)\s+sat=([\d.]+) \| λ_ov=([\d.]+)\s+λ_sat=([\d.]+)")


def traces(scen, algo):
    T = {}
    for s in SEEDS:
        f = f"{REPO}/logs/sweep/{scen}_{algo}_s{s}.log"
        if not os.path.exists(f):
            continue
        rows = [LINE.search(l) for l in open(f, errors="ignore")]
        rows = [r for r in rows if r]
        if rows:
            T[s] = dict(step=[int(r.group(2)) for r in rows],
                        ov_tight=[float(r.group(3)) for r in rows],
                        ov_true=[float(r.group(4)) if r.group(4) != "nan" else np.nan for r in rows],
                        sat=[float(r.group(5)) for r in rows],
                        lam_ov=[float(r.group(6)) for r in rows],
                        lam_sat=[float(r.group(7)) for r in rows])
    return T


def trace_summary(T, cap=500.0):
    out = {}
    for s, t in T.items():
        lam = np.array(t["lam_ov"]); st = np.array(t["step"])
        n = len(lam); third = slice(2 * n // 3, n)
        at_cap = lam >= cap - 1e-6
        out[s] = dict(n_updates=n,
                      first_cap_step=int(st[at_cap.argmax()]) if at_cap.any() else None,
                      frac_updates_at_cap=float(at_cap.mean()),
                      lam_ov_final=float(lam[-1]), lam_sat_final=float(t["lam_sat"][-1]),
                      ov_tight_first=float(np.mean(t["ov_tight"][:3])),
                      ov_tight_last_third=float(np.mean(np.array(t["ov_tight"])[third])),
                      ov_true_last_third=float(np.nanmean(np.array(t["ov_true"])[third])),
                      sat_last_third=float(np.mean(np.array(t["sat"])[third])),
                      min_ov_tight=float(np.min(t["ov_tight"])),
                      any_zero_ov_tight_window=bool(np.any(np.array(t["ov_tight"]) <= 0)))
    return out


def main():
    res = {}
    for scen in ("cityNL", "PublicPST"):
        if not os.path.exists(f"{REPO}/results/{scen}/eval_cache"):
            res[scen] = {"traces": {a: trace_summary(traces(scen, a)) for a in ("tccpl", "fixedpenalty")},
                         "traces_raw": {a: traces(scen, a) for a in ("tccpl", "fixedpenalty")}}
            continue
        A = arrays(scen)
        res[scen] = {"summary": summary(A), "paired": {}}
        for comp in ("Fixed-price SAC", "DDPG-PST", "PPO-PST", "CPO", "Round-Robin", "CAFAP"):
            P = {}
            for m in ("etr", "sqtr", "sat", "edel", "ov", "b10", "b5", "rel2", "rel5", "unmet"):
                P[m] = paired(A, "TC-CPL", comp, m)
            for m in ("ov", "b10", "b5", "rel2", "rel5"):
                P["viol_" + m] = paired(A, "TC-CPL", comp, m, binary=True)
            res[scen]["paired"][comp] = P
        res[scen]["traces"] = {a: trace_summary(traces(scen, a))
                               for a in ("tccpl", "fixedpenalty")}
        res[scen]["traces_raw"] = {a: traces(scen, a) for a in ("tccpl", "fixedpenalty")}
    json.dump(res, open(os.path.join(OUT, "submitted_stats.json"), "w"), indent=1)
    plot_traces(res)
    print_report(res)


def plot_traces(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(12, 5.6), constrained_layout=True)
    for r, scen in enumerate(("cityNL", "PublicPST")):
        for algo, col, ls in (("tccpl", "#1f5fbf", "-"), ("fixedpenalty", "#d0711b", "--")):
            for s, t in res[scen]["traces_raw"][algo].items():
                x = np.array(t["step"]) / 1e6
                lab = ("TC-CPL" if algo == "tccpl" else "fixed-price SAC") if s == 42 else None
                ax[r, 0].plot(x, t["lam_ov"], color=col, ls=ls, lw=1, label=lab)
                ax[r, 1].plot(x, t["ov_tight"], color=col, ls=ls, lw=1, label=lab)
                ax[r, 2].plot(x, t["sat"], color=col, ls=ls, lw=1, label=lab)
        ax[r, 0].axhline(500, color="k", lw=0.8, ls=":")
        ax[r, 1].axhline(0, color="k", lw=0.8, ls=":")
        ax[r, 2].axhline(0.98, color="k", lw=0.8, ls=":")
        ax[r, 0].set_ylabel(f"{scen}\n" + r"$\lambda_{ov}$ (kWh-price)")
        ax[r, 1].set_ylabel("training overload above\n0.95 x limit (kWh/day)")
        ax[r, 2].set_ylabel("training satisfaction")
        for c in range(3):
            ax[r, c].set_xlabel("environment steps (millions)")
    ax[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(OUT, "dual_traces_5seeds.png"), dpi=200)


def print_report(res):
    for scen in ("cityNL", "PublicPST"):
        if "summary" not in res[scen]:
            print(f"\n===== {scen}: no per-seed eval cache; dual traces only =====")
            for a, T in res[scen]["traces"].items():
                for s, t in T.items():
                    print(f"  {a:12s} s{s}: n={t['n_updates']} lam_ov_end {t['lam_ov_final']:.1f} lam_sat_end {t['lam_sat_final']:.2f} ov_tight last-third {t['ov_tight_last_third']:.2f} sat last-third {t['sat_last_third']:.4f}")
            continue
        S = res[scen]["summary"]
        print(f"\n===== {scen}: means over 5 seeds x 100 days (heuristics: 1 run) =====")
        print(f"{'controller':16s} {'|etr|':>7s} {'sat':>6s} {'Edel':>7s} {'ov':>6s} {'b10':>6s} "
              f"{'P(ov)':>6s} {'P(b10)':>6s} {'P(b10)any':>9s} {'P(rel2)':>7s} {'P(rel5)':>7s} {'namepl':>6s}")
        for lab, r in S.items():
            print(f"{lab:16s} {r['etr']:7.0f} {r['sat']:6.4f} {r['edel']:7.0f} {r['ov']:6.3f} "
                  f"{r['b10']:6.3f} {r['p_ov_pooled']:6.3f} {r['p_b10_pooled']:6.3f} "
                  f"{r['p_b10_anyseed']:9.2f} {r['p_rel2_pooled']:7.3f} {r['p_rel5_pooled']:7.3f} "
                  f"{r['nameplate']:6.3f}")
        print("-- TC-CPL minus comparator: diff [95% CI day-bootstrap] [95% CI seed x day]")
        for comp, P in res[scen]["paired"].items():
            for m in ("etr", "sat", "edel", "ov", "b10", "viol_b10", "viol_ov"):
                p = P[m]
                print(f"  vs {comp:16s} {m:9s} {p['diff']:+9.4f}  [{p['ci_day'][0]:+.4f}, {p['ci_day'][1]:+.4f}]"
                      f"  [{p['ci_seed_day'][0]:+.4f}, {p['ci_seed_day'][1]:+.4f}]"
                      + (f"  per-seed {[round(v, 4) for v in p['per_seed_diff']]}" if 'per_seed_diff' in p else ""))
        print("-- dual traces")
        for a, T in res[scen]["traces"].items():
            for s, t in T.items():
                print(f"  {a:12s} s{s}: n={t['n_updates']} cap@{t['first_cap_step']} at-cap {t['frac_updates_at_cap']:.2f} "
                      f"lam_ov_end {t['lam_ov_final']:.1f} lam_sat_end {t['lam_sat_final']:.2f} "
                      f"ov_tight first {t['ov_tight_first']:.2f} last-third {t['ov_tight_last_third']:.2f} "
                      f"(min {t['min_ov_tight']:.2f}) true last-third {t['ov_true_last_third']:.2f} "
                      f"sat last-third {t['sat_last_third']:.4f}")


if __name__ == "__main__":
    main()
