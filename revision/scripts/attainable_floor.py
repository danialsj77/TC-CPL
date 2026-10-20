"""Overload of fixed residual actions on the evaluation days.

delta = -1 on every port is the most conservative per-step action of the
anchored class a = clip(a0 + rho*delta, 0, 1): it gives an upper bound on the
lowest overload the class can attain (not the floor itself, since charging
earlier can change the state later). delta = 0 is the incumbent. Both are run
through the training interface, so the overload above the TIGHTENED limit
max{(1-m) P_tr, P_base} is exactly the quantity M1 prices. CAFAP is run
without the wrapper to record the peak transformer loading.

    python scripts/attainable_floor.py SCENARIO {minus1,zero,CAFAP} [--days N]

Writes revision/runs/floor/<SCENARIO>_<policy>_n<N>.json.
"""
import os, sys, json, time

threads = os.environ.get("EXP_THREADS", "1")
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, threads)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import exp_lib as X  # noqa: E402


def run_day(env, raw_of, act, seed):
    obs, _ = env.reset(seed=seed)
    raw = raw_of(env)
    e_ref = None
    ov_tight, peak, done, info = 0.0, 0.0, False, {}
    while not done:
        obs, r, term, trunc, info = env.step(act(env, raw))
        done = term or trunc
        comps = getattr(raw, "_reward_components", None)
        if comps is not None:
            if e_ref is None:
                e_ref = raw.reward_function._e_ref
            ov_tight += comps[1] * e_ref
        peak = max(peak, max(float(tr.current_power) for tr in raw.transformers))
    return {"ov_tight_kWh": ov_tight, "ov_raw_kWh": float(info["total_transformer_overload"]),
            "sat": float(info["average_user_satisfaction"]),
            "energy_kWh": float(info["total_energy_charged"]), "peak_tr_kW": peak}


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    scen, policy = args[0], args[1]
    n_days = int(sys.argv[sys.argv.index("--days") + 1]) if "--days" in sys.argv else 100
    if "--days" in sys.argv:
        args = [a for a in args if a != str(n_days)]
    os.chdir(X.REPO)
    spec = X.spec_of(scenario=scen)
    out_dir = os.path.join(X.REPO, "revision", "runs", "floor")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{scen}_{policy}_n{n_days}.json")
    t0, days = time.time(), []
    X.pin_first_day(500)
    for d in range(n_days):
        if policy == "CAFAP":
            from ev2gym.baselines.heuristics import ChargeAsFastAsPossible
            from ev2gym.rl_agent.reward import SquaredTrackingErrorReward
            env = X.TCCPLEnv(config_file=os.path.join(X.REPO, X.CONFIGS[scen]),
                             reward_function=SquaredTrackingErrorReward,
                             state_function=X.grid_aware_obs, save_replay=False,
                             save_plots=False, verbose=False)
            h = ChargeAsFastAsPossible()
            days.append(run_day(env, lambda e: e, lambda e, raw: h.get_action(raw), 500 + d))
        else:
            env = X.make_env(spec, X.make_costs(spec))
            val = -1.0 if policy == "minus1" else 0.0
            days.append(run_day(env, lambda e: e.unwrapped,
                                lambda e, raw: np.full(e.action_space.shape, val, np.float32),
                                500 + d))
        env.close()
    keys = days[0].keys()
    summary = {k: float(np.mean([r[k] for r in days])) for k in keys}
    summary["peak_tr_kW_max"] = float(max(r["peak_tr_kW"] for r in days))
    summary["ov_tight_days_positive"] = int(sum(r["ov_tight_kWh"] > 1e-9 for r in days))
    json.dump({"scenario": scen, "policy": policy, "n_days": n_days, "day0_pinned": True,
               "summary": summary, "days": days, "minutes": (time.time() - t0) / 60},
              open(out, "w"), indent=1)
    print(scen, policy, json.dumps({k: round(v, 3) for k, v in summary.items()}))
