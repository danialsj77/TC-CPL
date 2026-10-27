"""Collect every revision result that exists on disk into one JSON
(revision/analysis/results.json). Missing runs are reported, never invented.

    python scripts/collect_results.py
"""
import glob, json, os, re, sys
import numpy as np

REV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(REV)
OUT = os.path.join(REV, "analysis")
RUNS = os.path.join(REV, "runs")
RNG = np.random.default_rng(7)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import output_is_current, stress_run_is_current


def load_days(path):
    return json.load(open(path))["days"]


def day_metrics(days):
    a = lambda k: np.array([d[k] for d in days], float)
    sats = np.concatenate([np.array(d["sat_list"], float) for d in days]) if days else np.array([])
    return dict(
        n_days=len(days),
        etr=a("energy_tracking_error").mean(), sqtr=a("tracking_error").mean(),
        sat=a("average_user_satisfaction").mean(), edel=a("total_energy_charged").mean(),
        ov=a("total_transformer_overload").mean(), b10=a("ov_tol10").mean(), b5=a("ov_tol5").mean(),
        p_ov=(a("total_transformer_overload") > 1e-9).mean(), p_b10=(a("ov_tol10") > 1e-9).mean(),
        p_b5=(a("ov_tol5") > 1e-9).mean(), p_rel2=(a("ov_rel2") > 1e-9).mean(),
        p_rel5=(a("ov_rel5") > 1e-9).mean(), rel5=a("ov_rel5").mean(),
        nameplate=a("ov_nameplate").mean(),
        max_exc=a("max_excess_kW").max(), max_exc_mean=a("max_excess_kW").mean(),
        max_exc_pct=a("max_excess_pct_rating").max(),
        viol_steps=a("n_viol_steps").mean(), tr_viol=a("n_tr_violating").mean(),
        ov_grid=a("ov_gridside").mean(), losses=a("grid_losses_kWh").mean(),
        unmet_share=a("unmet_kWh").sum() / max(a("requested_kWh").sum(), 1e-9),
        unmet=a("unmet_kWh").mean(),
        ev_below_095=float((sats < 0.95).mean()) if sats.size else np.nan,
        ev_below_09=float((sats < 0.9).mean()) if sats.size else np.nan,
        ev_below_08=float((sats < 0.8).mean()) if sats.size else np.nan,
        ev_p5=float(np.percentile(sats, 5)) if sats.size else np.nan,
        n_dep=a("n_departed").mean(), infer_ms=np.nanmean(a("infer_ms_mean")))


def pool(paths):
    days = []
    for p in paths:
        days += load_days(p)
    return days


def per_tr_concentration(paths):
    P = None
    for p in paths:
        for d in load_days(p):
            v = np.array(d["per_tr_overload_kWh"])
            P = v if P is None else P + v
    if P is None or P.sum() <= 0:
        return {}
    s = np.sort(P)[::-1]
    return dict(n_tr_any=int((P > 0).sum()), n_tr=len(P), top3_share=float(s[:3].sum() / P.sum()))


def citynl():
    res = {"base": {}, "shield": {}, "heldout": {}, "perturb": {}, "missing": []}
    labels = {"TC-CPL": "tccpl", "Fixed-price SAC": "fixedpenalty", "DDPG-PST": "ddpg_pst",
              "PPO-PST": "ppo_pst", "CPO": "cpo_pst", "Round-Robin": "RR", "CAFAP": "CAFAP"}
    E = os.path.join(RUNS, "cityNL_eval")
    for lab, key in labels.items():
        seeds = (0,) if key in ("RR", "CAFAP") else (42, 43, 44, 45, 46)
        for kind, pat in (("base", "{k}_s{s}_n100.json"), ("shield", "{k}_s{s}_shield_n100.json"),
                          ("heldout", "{k}_s{s}_n100_o5000.json")):
            ps = [os.path.join(E, pat.format(k=key, s=s)) for s in seeds]
            # only outputs made from the checkpoints now in models/ (provenance.py)
            have = [p for p, s in zip(ps, seeds)
                    if os.path.exists(p) and output_is_current(p, REPO, "cityNL", key, s)]
            if len(have) < len(ps):
                res["missing"].append(f"{kind}:{key} {len(have)}/{len(ps)}")
            if have:
                m = day_metrics(pool(have))
                m["seeds"] = len(have)
                if kind == "base":
                    m.update(per_tr_concentration(have))
                res[kind][lab] = m
    # observation perturbations (3 seeds x 50 days) against the same 50 days unperturbed
    for f in sorted(glob.glob(os.path.join(E, "*_s4?_*_n50.json"))):
        m = re.match(r"(tccpl|fixedpenalty)_s(\d+)_(.+)_n50\.json", os.path.basename(f))
        if not m:
            continue
        key, seed, pert = m.group(1), int(m.group(2)), m.group(3)
        if not output_is_current(f, REPO, "cityNL", key, seed):
            res["missing"].append(f"perturb:{key} s{seed} {pert} (made from another checkpoint)")
            continue
        res["perturb"].setdefault(pert, {}).setdefault(key, []).append((seed, f))
    P = {}
    for pert, d in res["perturb"].items():
        P[pert] = {}
        for key, lst in d.items():
            seeds = sorted(s for s, _ in lst)
            pert_days = pool([f for _, f in sorted(lst)])
            base_days = []
            for s in seeds:
                b = os.path.join(E, f"{key}_s{s}_n100.json")
                if os.path.exists(b) and output_is_current(b, REPO, "cityNL", key, s):
                    base_days += load_days(b)[:50]
            P[pert][key] = dict(seeds=seeds, perturbed=day_metrics(pert_days),
                                nominal=day_metrics(base_days) if base_days else None)
    res["perturb"] = P
    return res


def publicpst(scen="PublicPST_S"):
    """The small scenario's main comparison (five learners x five seeds + heuristics)."""
    res, E = {}, os.path.join(RUNS, f"{scen}_eval")
    labels = {"TC-CPL": "tccpl", "Fixed-price SAC": "fixedpenalty", "DDPG-PST": "ddpg_pst",
              "PPO-PST": "ppo_pst", "CPO": "cpo_pst", "Round-Robin": "RR", "CAFAP": "CAFAP"}
    for lab, key in labels.items():
        seeds = (0,) if key in ("RR", "CAFAP") else (42, 43, 44, 45, 46)
        have = [os.path.join(E, f"{key}_s{s}_n100.json") for s in seeds
                if os.path.exists(os.path.join(E, f"{key}_s{s}_n100.json"))
                and output_is_current(os.path.join(E, f"{key}_s{s}_n100.json"), REPO, scen, key, s)]
        if have:
            m = day_metrics(pool(have))
            m["seeds"] = len(have)
            m["per_seed_etr"] = [day_metrics(load_days(p))["etr"] for p in have]
            m["per_seed_sat"] = [day_metrics(load_days(p))["sat"] for p in have]
            res[lab] = m
    return res


def offline():
    res = {}
    for scen in ("cityNL", "PublicPST", "PublicPST_stress"):
        R = []
        # offline_v2: rerun of 19 Oct 2026 after the desired-energy fix in offline_feasibility.py
        for f in glob.glob(os.path.join(RUNS, os.environ.get("OFFLINE_DIR", "offline_v2"), f"{scen}_*_*.json")):
            if re.fullmatch(rf"{scen}_\d+_\d+\.json", os.path.basename(f)) and \
                    not os.path.basename(f).endswith(("_0_1.json", "_0_2.json")):
                R += json.load(open(f))
        if not R:
            continue
        st = lambda k: [r[k].get("status") for r in R]
        P0, P1 = [r["P0"] for r in R], [r["P1"] for r in R]
        res[scen] = dict(
            n_days=len(R), n_vars_mean=float(np.mean([r["n_vars"] for r in R])),
            n_vars_max=int(np.max([r["n_vars"] for r in R])),
            sessions_mean=float(np.mean([r["n_sessions"] for r in R])),
            p0_optimal=st("P0").count("optimal"), p1_optimal=st("P1").count("optimal"),
            p2_095_optimal=st("P2_095").count("optimal"), p2_098_optimal=st("P2_098").count("optimal"),
            p0_sec_mean=float(np.mean([p["seconds"] for p in P0])),
            p0_sec_max=float(np.max([p["seconds"] for p in P0])),
            p0_iters_mean=float(np.mean([p["iters"] for p in P0])),
            p1_min=float(np.min([p["objective"] for p in P1])),
            p1_mean=float(np.mean([p["objective"] for p in P1])),
            p0_plan_sat_min=float(np.min([p["mean_sat"] for p in P0])),
            p2_098_rel_increase_max=float(max(
                (r["P2_098"]["objective"] - r["P0"]["objective"]) / max(r["P0"]["objective"], 1e-9)
                for r in R if r["P2_098"].get("objective") is not None)))
    return res


def stress():
    """PublicPST-S: every trained variant (mean over seeds, per-seed lists) + heuristics."""
    S = os.path.join(RUNS, "PublicPST_stress")
    res = {}
    for d in sorted(glob.glob(os.path.join(S, "*_s*"))):
        name = os.path.basename(d)
        m = re.match(r"(.+)_s(\d+)$", name)
        var, seed = m.group(1), int(m.group(2))
        ev = os.path.join(d, "eval.json")
        # a run counts only if it was produced with the variant as defined NOW
        # (folders of earlier protocols or retired variants are ignored)
        if not os.path.exists(ev) or not stress_run_is_current(d):
            continue
        r = res.setdefault(var, {"seeds": [], "per_seed": []})
        dm = day_metrics(load_days(ev))
        meta_p = os.path.join(d, "train_meta.json")
        if os.path.exists(meta_p):
            meta = json.load(open(meta_p))
            lam = np.array(meta["lambda_history"]) if meta["lambda_history"] else np.zeros((0, 2))
            vio = np.array(meta["violation_history"]) if meta["violation_history"] else np.zeros((0, 2))
            n = len(lam)
            spec = meta["spec"]
            dm.update(lam_ov_end=float(lam[-1, 0]) if n else np.nan,
                      lam_sat_end=float(lam[-1, 1]) if n else np.nan,
                      lam_ov_max=float(lam[:, 0].max()) if n else np.nan,
                      at_cap_frac=float((lam[:, 0] >= spec["lam_max"] - 1e-6).mean()) if n else np.nan,
                      lam_ov_last_third=float(lam[2 * n // 3:, 0].mean()) if n else np.nan,
                      train_ov_first_quarter=float(vio[: max(1, n // 4), 0].mean()) if n else np.nan,
                      train_ov_last_third=float(vio[2 * n // 3:, 0].mean()) if n else np.nan,
                      train_sat_last_third=float(vio[2 * n // 3:, 1].mean()) if n else np.nan,
                      true_ov_first_quarter=float(np.nanmean(meta["true_overload_history"][: max(1, n // 4)])) if n else np.nan,
                      minutes=meta["minutes"], n_updates=n,
                      lam_hist=lam.tolist(), vio_hist=vio.tolist())
        r["seeds"].append(seed)
        r["per_seed"].append(dm)
    for var, r in res.items():
        keys = [k for k, v in r["per_seed"][0].items() if isinstance(v, (int, float))]
        r["mean"] = {k: float(np.nanmean([p.get(k, np.nan) for p in r["per_seed"]])) for k in keys}
        r["min"] = {k: float(np.nanmin([p.get(k, np.nan) for p in r["per_seed"]])) for k in keys}
        r["max"] = {k: float(np.nanmax([p.get(k, np.nan) for p in r["per_seed"]])) for k in keys}
    return res


def paired_stress(res, a="tccpl", b="fixed_100_50", metric="etr"):
    """Seed-paired difference a - b on PublicPST-S (same seeds, same days)."""
    S = os.path.join(RUNS, "PublicPST_stress")
    seeds = sorted(set(res.get(a, {}).get("seeds", [])) & set(res.get(b, {}).get("seeds", [])))
    if not seeds:
        return None
    da = np.stack([[d[metric] for d in load_days(os.path.join(S, f"{a}_s{s}", "eval.json"))] for s in seeds])
    db = np.stack([[d[metric] for d in load_days(os.path.join(S, f"{b}_s{s}", "eval.json"))] for s in seeds])
    diff = (da - db).mean(0)
    boot = np.array([diff[RNG.integers(0, len(diff), len(diff))].mean() for _ in range(5000)])
    return dict(seeds=seeds, diff=float(diff.mean()),
                ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                per_seed=[float((da[i] - db[i]).mean()) for i in range(len(seeds))])


def main():
    out = {"cityNL": citynl(), "PublicPST_S": publicpst("PublicPST_S"), "offline": offline(), "stress": stress()}
    try:
        out["scaling"] = json.load(open(os.path.join(OUT, "scaling_bench.json")))
    except OSError:
        out["scaling"] = None
    pairs = {}
    for b in ("fixed_100_50", "fixed_500_5", "tau_2", "lmax_2000", "no_demos", "no_residual",
              "no_gridblock", "flat_encoder", "inc_cafap", "rho_0p25", "no_load_aware"):
        for mtr in ("etr", "average_user_satisfaction", "total_energy_charged",
                    "total_transformer_overload", "ov_rel5"):
            key = {"etr": "energy_tracking_error"}.get(mtr, mtr)
            p = paired_stress(out["stress"], "tccpl", b, key)
            if p:
                pairs.setdefault(b, {})[mtr] = p
    out["stress_paired_vs_tccpl"] = pairs
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=float)
    print("missing cityNL:", out["cityNL"]["missing"])
    print("stress variants done:", {k: v["seeds"] for k, v in out["stress"].items()})


if __name__ == "__main__":
    main()
