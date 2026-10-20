"""Calibrate the stressed single-transformer scenario: heuristic overload
statistics for candidate transformer ratings (Round-Robin and CAFAP)."""
import os, sys, json, re
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import exp_lib as X
INFL_MULT = float(os.environ.get("INFL_MULT", 1.0))
PV_MULT = float(os.environ.get("PV_MULT", 1.0))

def make_cfg(tag, kw, infl, pv, dr):
    src = open(os.path.join(X.REPO, X.CONFIGS["PublicPST"])).read()
    s = re.sub(r"(transformer:\s*\n\s*max_power:)\s*\d+", rf"\1 {kw}", src)
    s = re.sub(r"(inflexible_loads:[^\n]*\n\s*include:)\s*\w+", rf"\1 {infl}", s)
    s = re.sub(r"(solar_power:[^\n]*\n\s*include:)\s*\w+", rf"\1 {pv}", s)
    s = re.sub(r"(demand_response:[^\n]*\n\s*include:)\s*\w+", rf"\1 {dr}", s)
    s = s.replace("event_start_hour_mean: 12", "event_start_hour_mean: 18")
    # cityNL-like background conditions (cityNL.yaml)
    s = re.sub(r"inflexible_loads_capacity_multiplier_mean:\s*[\d.]+", f"inflexible_loads_capacity_multiplier_mean: {INFL_MULT}", s)
    s = re.sub(r"solar_power_capacity_multiplier_mean:\s*[\d.]+", f"solar_power_capacity_multiplier_mean: {PV_MULT}", s)
    s = re.sub(r"event_capacity_percentage_mean:\s*\d+", "event_capacity_percentage_mean: 30", s)
    s = re.sub(r"event_capacity_percentage_std:\s*\d+", "event_capacity_percentage_std: 8", s)
    s = re.sub(r"event_length_minutes_max:\s*\d+", "event_length_minutes_max: 90", s)
    path = os.path.join(X.REPO, "revision", "configs", f"calib_{tag}.yaml")
    open(path, "w").write(s)
    return os.path.relpath(path, X.REPO)

if __name__ == "__main__":
    kw, infl, pv, dr, n = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
    tag = f"{kw}_{infl[0]}{pv[0]}{dr[0]}_i{INFL_MULT}_p{PV_MULT}"
    X.CONFIGS[tag] = make_cfg(tag, kw, infl, pv, dr)
    spec = X.spec_of(scenario=tag)
    res = {}
    for h in ("RR", "CAFAP"):
        days = X.evaluate_heuristic(spec, h, n, seed_offset=500)
        a = lambda k: np.array([d[k] for d in days])
        res[h] = dict(raw=a("total_transformer_overload").mean(),
                      pviol=(a("total_transformer_overload") > 1e-9).mean(),
                      b10=a("ov_tol10").mean(), pviol10=(a("ov_tol10") > 1e-9).mean(),
                      b5=(a("ov_tol5") > 1e-9).mean(),
                      sat=a("average_user_satisfaction").mean(),
                      etr=a("energy_tracking_error").mean(),
                      edel=a("total_energy_charged").mean(),
                      maxexc=a("max_excess_kW").mean())
    print(json.dumps({"tag": tag, **{h: {k: round(float(v), 3) for k, v in r.items()} for h, r in res.items()}}))
