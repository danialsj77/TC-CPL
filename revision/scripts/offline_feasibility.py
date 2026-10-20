"""Hindsight (perfect-foresight) analysis of every evaluation day.

For each day the full session list is recorded (CAFAP run with save_replay, as
the paper's offline benchmark does) and three convex programs are solved with
CVXPY/Clarabel under the HARD transformer limit (zero overload):

  P0  min sum_t (P_set - P_EV)^2                 = the paper's Optimal (Offline), Eq. (16)
  P1  max mean_k s_k,  s_k = min(1, E_dep,k / E_des,k)
                                                  -> best mean satisfaction any schedule can
                                                     reach with zero overload, even in hindsight
  P2  P0 + mean_k s_k >= 0.95  (and >= 0.98)     -> the offline benchmark WITH the paper's
                                                     satisfaction requirement; feasible iff P1 >= floor

    python scripts/offline_feasibility.py SCENARIO FIRST_DAY N_DAYS

Writes revision/runs/offline/<SCENARIO>_<first>_<n>.json.
"""
import os, sys, json, time, pickle
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
import scipy.sparse as sps
import cvxpy as cp
import exp_lib as X
from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.baselines.heuristics import ChargeAsFastAsPossible
from ev2gym.rl_agent.reward import SquaredTrackingErrorReward
from ev2gym.rl_agent.state import PublicPST

TMP = os.path.join(X.REPO, "revision", "replay_tmp", str(os.getpid()))


def record_day(cfg, seed):
    env = EV2Gym(config_file=cfg, reward_function=SquaredTrackingErrorReward,
                 state_function=PublicPST, save_replay=True, replay_save_path=TMP + "/",
                 save_plots=False, verbose=False)
    env.reset(seed=seed)
    a = ChargeAsFastAsPossible()
    done = False
    while not done:
        _, _, t, tr, _ = env.step(a.get_action(env))
        done = t or tr
    path = os.path.join(TMP, f"replay_{env.sim_name}.pkl")
    env.close()
    r = pickle.load(open(path, "rb"))
    os.remove(path)
    return r


def build(r):
    """Same model as the notebook's OptimalPSTOracle (delivered power d per
    occupied slot), plus the session structure needed for satisfaction."""
    T, C, P = r.sim_length, r.n_cs, int(r.max_n_ports)
    dt = r.timescale / 60.0
    u = r.u
    eta_all = getattr(r, "ev_ch_efficiency", None)
    if eta_all is None:
        eta_all = np.ones_like(u)
    cs_pmax = r.port_max_charge_current * r.voltages / 1000.0
    tr_pmax = np.array([r.tra_max_amps[k] * r.transformers[k].voltage / 1000.0
                        for k in range(r.n_transformers)])
    setp = np.asarray(r.power_setpoints, float).reshape(-1)[:T]
    slots = [(pt, i, t) for pt in range(P) for i in range(C) for t in range(T) if u[pt, i, t] == 1]
    idx = {s: k for k, s in enumerate(slots)}
    n = len(slots)
    eta = np.array([max(float(eta_all[pt, i, t]), 1e-3) for (pt, i, t) in slots])
    ub = np.array([max(min(r.ev_max_ch_power[pt, i, t], cs_pmax[i]), 0.0) for (pt, i, t) in slots]) * eta

    def group(keys_of, vals=None):
        keys = sorted(set(keys_of))
        row = {g: j for j, g in enumerate(keys)}
        data = np.ones(n) if vals is None else np.asarray(vals, float)
        return sps.csr_matrix((data, ([row[g] for g in keys_of], np.arange(n))), shape=(len(keys), n)), keys

    M = sps.csr_matrix((np.ones(n), ([t for (_, _, t) in slots], np.arange(n))), shape=(T, n))
    S, sk = group([(i, t) for (_, i, t) in slots], 1.0 / eta)
    tr_of = np.asarray(r.cs_transformer, int)
    G, gk = group([(int(tr_of[i]), t) for (_, i, t) in slots])
    sessions = []                                  # (slot indices, e0, E_cap, E_des)
    arr = np.asarray(r.ev_arrival)
    for pt in range(P):
        for i in range(C):
            t = 0
            while t < T:
                # a session starts where a port becomes occupied OR where a new
                # vehicle arrives at a port its predecessor has just left
                if u[pt, i, t] == 1 and (t == 0 or u[pt, i, t - 1] == 0 or arr[pt, i, t] == 1):
                    t0, ks = t, []
                    while t < T and u[pt, i, t] == 1 and (t == t0 or arr[pt, i, t] == 0):
                        ks.append(idx[(pt, i, t)])
                        t += 1
                    e0 = float(r.energy_at_arrival[pt, i, t0])
                    # the replay stores the desired energy at the DEPARTURE index
                    # (= t, the first step after the session), not at arrival
                    des = float(r.ev_des_energy[pt, i, t]) if t < T else 0.0
                    if des <= 0.0:
                        des = float(r.ev_max_energy[pt, i, t0])
                    sessions.append((ks, e0, float(r.ev_max_energy[pt, i, t0]), des))
                else:
                    t += 1
    rows = [j for j, s in enumerate(sessions) for _ in s[0]]
    cols = [k for s in sessions for k in s[0]]
    E = sps.csr_matrix((np.ones(len(cols)), (rows, cols)), shape=(len(sessions), n))
    e0 = np.array([s[1] for s in sessions])
    cap = np.array([max(s[2] - s[1], 0.0) for s in sessions])
    des = np.array([max(s[3], 1e-6) for s in sessions])
    return dict(n=n, T=T, dt=dt, setp=setp, M=M, ub=ub, S=S,
                s_rhs=np.array([cs_pmax[i] for (i, _) in sk]), G=G,
                g_rhs=np.maximum(np.array([tr_pmax[k, t] for (k, t) in gk]), 0.0),
                E=E, e0=e0, cap=cap, des=des)


def solve(m, mode, floor=None):
    p = cp.Variable(m["n"], nonneg=True)
    cons = [p <= m["ub"], m["S"] @ p <= m["s_rhs"], m["G"] @ p <= m["g_rhs"],
            m["E"] @ p * m["dt"] <= m["cap"] + 1e-9]
    z = cp.Variable(len(m["des"]))
    sat_cons = [z <= 1.0, z <= cp.multiply(1.0 / m["des"], m["e0"] + m["dt"] * (m["E"] @ p))]
    track = cp.sum_squares(m["setp"] - m["M"] @ p)
    if mode == "P0":
        prob = cp.Problem(cp.Minimize(track), cons)
    elif mode == "P1":
        prob = cp.Problem(cp.Maximize(cp.sum(z) / len(m["des"])), cons + sat_cons)
    else:
        prob = cp.Problem(cp.Minimize(track), cons + sat_cons + [cp.sum(z) / len(m["des"]) >= floor])
    t0 = time.time()
    try:
        prob.solve(solver=cp.CLARABEL)
    except Exception as e:                                       # noqa: BLE001
        return dict(status=f"error {type(e).__name__}", seconds=time.time() - t0)
    out = dict(status=prob.status, seconds=time.time() - t0,
               iters=getattr(prob.solver_stats, "num_iters", None))
    if prob.status in ("optimal", "optimal_inaccurate"):
        pv = np.maximum(np.asarray(p.value).reshape(-1), 0)
        e_dep = m["e0"] + m["dt"] * (m["E"] @ pv)
        s = np.minimum(e_dep / m["des"], 1.0)
        out.update(objective=float(prob.value), mean_sat=float(s.mean()),
                   frac_ev_below_095=float((s < 0.95).mean()),
                   delivered_kWh=float(pv.sum() * m["dt"]),
                   sq_tracking=float(np.sum((m["setp"] - m["M"] @ pv) ** 2)),
                   abs_tracking_kWh=float(np.sum(np.abs(m["setp"] - m["M"] @ pv)) * m["dt"]))
    return out


if __name__ == "__main__":
    scen, first, n_days = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    os.makedirs(TMP, exist_ok=True)
    os.chdir(X.REPO)
    cfg = X.CONFIGS[scen]
    out_dir = os.path.join(X.REPO, "revision", "runs", os.environ.get("OFFLINE_OUT", "offline"))
    os.makedirs(out_dir, exist_ok=True)
    res = []
    for d in range(first, first + n_days):
        seed = 500 + d
        t0 = time.time()
        r = record_day(cfg, seed)
        m = build(r)
        rec = dict(day=d, seed=seed, n_vars=m["n"], n_sessions=len(m["des"]),
                   P0=solve(m, "P0"), P1=solve(m, "P1"),
                   P2_095=solve(m, "P2", 0.95), P2_098=solve(m, "P2", 0.98))
        rec["seconds_total"] = time.time() - t0
        res.append(rec)
        print(d, rec["P0"].get("status"), round(rec["P0"].get("seconds", 0), 2), "maxsat",
              round(rec["P1"].get("objective", float("nan")), 4), "P2(.95)", rec["P2_095"].get("status"),
              flush=True)
    json.dump(res, open(os.path.join(out_dir, f"{scen}_{first}_{n_days}.json"), "w"), indent=1)
