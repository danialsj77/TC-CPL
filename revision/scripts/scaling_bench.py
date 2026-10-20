"""Computational footprint of the M3 networks versus the number of ports
Parameter counts (M3 vs a flat MLP), actor latency per
dispatch (batch 1) on CPU and MPS, SAC update time (B=256), replay memory, and
the end-to-end controller pipeline on the real PublicPST and cityNL simulators
with the trained 1M-step checkpoints (observation, incumbent, actor, projection).

Waits until the machine is quiet (1-min load average < MAX_LOAD) so timings are
not polluted by the training campaign; records the load it ran at.

    python scripts/scaling_bench.py      -> revision/analysis/scaling_bench.json
"""
import json, os, platform, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
MAX_LOAD = float(os.environ.get("BENCH_MAX_LOAD", 4.0))
MAX_WAIT_H = float(os.environ.get("BENCH_MAX_WAIT_H", 20))
t_wait = time.time()
while os.getloadavg()[0] > MAX_LOAD and time.time() - t_wait < MAX_WAIT_H * 3600:
    time.sleep(120)

import numpy as np
import torch as th
import exp_lib as X
from tccpl_learner import TCCPLActor, TCCPLCritic, TCCPLSAC, grid_aware_obs
from ev2gym.baselines.heuristics import RoundRobin

OUT = os.path.join(X.REPO, "revision", "analysis")
os.makedirs(OUT, exist_ok=True)
H = 128


def layout(N, W):
    rng = np.random.default_rng(0)
    tr = np.sort(np.concatenate([np.arange(W), rng.integers(0, W, N - W)]))
    return {"n_ports": N, "n_tr": W, "tr_of_port": tr.tolist(),
            "base_obs_dim": 4 + 4 * W + 3 * N, "obs_dim": 4 + 4 * W + 4 * N}


def fake_obs(L, B):
    N, W = L["n_ports"], L["n_tr"]
    o = th.rand(B, L["obs_dim"])
    oc = o[:, 4 + 4 * W:4 + 4 * W + 3 * N].view(B, N, 3)
    oc[..., 0] = th.where(th.rand(B, N) < 0.7, 0.5, 0.0)       # 70% occupied
    return o


def timeit(fn, reps, warm=10):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts) * 1e3
    return {"mean_ms": float(ts.mean()), "median_ms": float(np.median(ts)),
            "p99_ms": float(np.percentile(ts, 99)), "max_ms": float(ts.max())}


def bench_networks():
    rows = []
    for N in (20, 100, 600, 1200, 3000, 6000):
        W = max(1, round(N / 4.92)) if N > 20 else 1
        L = layout(N, W)
        actor = TCCPLActor(L, H, use_bf16=False).eval()
        critic = TCCPLCritic(L, H, use_bf16=False)
        flat = X.FlatActor(L, 256)
        r = {"N": N, "W": W, "obs_dim": L["obs_dim"],
             "actor_params": sum(p.numel() for p in actor.parameters()),
             "critic_params_each": sum(p.numel() for p in critic.q1.parameters()),
             "flat_actor_params": sum(p.numel() for p in flat.parameters()),
             "flat_critic_params_each": sum(p.numel() for p in X.FlatQ(L, 256).parameters()),
             "replay_GB_200k": 200_000 * (2 * L["obs_dim"] * 4 + N * 4 + 5 * 4 + 1 + 2 * 8) / 1e9}
        o1 = fake_obs(L, 1)
        for threads in (1, 4):
            th.set_num_threads(threads)
            with th.no_grad():
                r[f"actor_cpu{threads}"] = timeit(lambda: actor(o1, deterministic=True), 200)
        if th.backends.mps.is_available():
            a_m, o_m = actor.to("mps"), o1.to("mps")

            def f():
                with th.no_grad():
                    a_m(o_m, deterministic=True)
                th.mps.synchronize()
            r["actor_mps"] = timeit(f, 200)
            actor.to("cpu")
        # one SAC update (actor + twin critics), B = 256
        for dev in ["cpu"] + (["mps"] if th.backends.mps.is_available() else []):
            th.set_num_threads(4)
            a, c = TCCPLActor(L, H, False).to(dev), TCCPLCritic(L, H, False).to(dev)
            oa, oc_ = th.optim.Adam(a.parameters(), 3e-4), th.optim.Adam(c.parameters(), 3e-4)
            ob = fake_obs(L, 256).to(dev)

            def upd():
                d, lp, _ = a.sample(ob)
                q1, q2 = c(ob, d.detach())
                loss = ((q1 - 1) ** 2 + (q2 - 1) ** 2).mean()
                oc_.zero_grad(); loss.backward(); oc_.step()
                q1p, q2p = c(ob, a.sample(ob)[0])
                al = (0.1 * lp - th.min(q1p, q2p)).mean()
                oa.zero_grad(); al.backward(); oa.step()
                if dev == "mps":
                    th.mps.synchronize()
            reps = 20 if N <= 600 else 5
            try:
                r[f"update_B256_{dev}"] = timeit(upd, reps, warm=3)
            except RuntimeError as e:                       # out of memory at large N
                r[f"update_B256_{dev}"] = {"error": str(e)[:120]}
        rows.append(r)
        print(json.dumps({k: (v if not isinstance(v, dict) else round(v.get("mean_ms", -1), 3))
                          for k, v in r.items()}), flush=True)
    return rows


def bench_pipeline(scen, days=5):
    """Controller-side time per dispatch with the trained seed-42 checkpoint."""
    th.set_num_threads(1)
    os.chdir(X.REPO)
    spec = X.spec_of(scenario=scen)
    model = TCCPLSAC.load(os.path.join(X.REPO, "models", scen, "tccpl_sac.zip"), device="cpu")
    t_obs, t_inc, t_act, t_step = [], [], [], []
    for d in range(days):
        env = X.make_env(spec, X.SquaredTrackingErrorReward)
        obs, _ = env.reset(seed=500 + d)
        raw = env.unwrapped
        done = False
        while not done:
            t0 = time.perf_counter(); grid_aware_obs(raw); t_obs.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); env._incumbent.get_action(raw); t_inc.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); a = model.predict(obs, deterministic=True)[0]
            np.clip(env._anchor + 0.25 * a * env._ctrl_mask(), 0, 1); t_act.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); obs, _, te, tr, _ = env.step(a); t_step.append(time.perf_counter() - t0)
            done = te or tr
        env.close()
    f = lambda v: {"mean_ms": float(np.mean(v) * 1e3), "p99_ms": float(np.percentile(v, 99) * 1e3),
                   "max_ms": float(np.max(v) * 1e3), "n": len(v)}
    tot = np.array(t_obs) + np.array(t_inc) + np.array(t_act)
    return {"observation": f(t_obs), "incumbent": f(t_inc), "actor_and_projection": f(t_act),
            "controller_total": f(tot), "simulator_step_incl_powerflow": f(t_step)}


if __name__ == "__main__":
    res = {"platform": {"machine": platform.machine(), "processor": platform.processor(),
                        "python": platform.python_version(), "torch": th.__version__,
                        "cpu_count": os.cpu_count()},
           "load_at_start": os.getloadavg(), "waited_min": (time.time() - t_wait) / 60}
    try:
        import subprocess
        res["platform"]["chip"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                                 capture_output=True, text=True).stdout.strip()
    except Exception:                                                   # noqa: BLE001
        pass
    res["networks"] = bench_networks()
    res["pipeline"] = {s: bench_pipeline(s) for s in ("PublicPST", "cityNL")}
    res["load_at_end"] = os.getloadavg()
    json.dump(res, open(os.path.join(OUT, "scaling_bench.json"), "w"), indent=1)
    print(json.dumps(res["pipeline"], indent=1))
