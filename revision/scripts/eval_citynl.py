"""Re-evaluate the cityNL (or, with EXP_SCEN=PublicPST, the PublicPST) checkpoints in models/
on the same 100 paired days (env seeds 500..599) with extended per-day records,
optionally under an evaluation-time observation perturbation.

    python scripts/eval_citynl.py LABEL SEED [kind:p] [--days N]

LABEL: tccpl | fixedpenalty | ddpg_pst | ppo_pst | cpo_pst | RR | CAFAP
kind:p (TC-CPL / fixed-price only): status_flip:0.1, status_hide:0.3, status_delay:1,
nev_bias:0.2, rating_bias:-0.05, dr_notice:0 ...  (see exp_lib.ObsPerturbation)

Writes revision/runs/cityNL_eval/<LABEL>_s<SEED>[_<kind><p>].json; never
touches models/ or results/.
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
threads = os.environ.get("EXP_THREADS", "1")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = threads
import numpy as np
import torch
torch.set_num_threads(int(threads))
import gymnasium as gym
import exp_lib as X
sys.path.insert(0, HERE)
from provenance import checkpoint_stamp as _stamp
from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from ev2gym.rl_agent.reward import SquaredTrackingErrorReward
from ev2gym.rl_agent.state import PublicPST
from tccpl_learner import TCCPLEnv, TCCPLSAC, grid_aware_obs, BaselineFeatureExtractor, layout_from_env

SCEN = os.environ.get("EXP_SCEN", "cityNL")
OUT = os.path.join(X.REPO, "revision", "runs", f"{SCEN}_eval")


def paths(name, seed):
    suf = "" if seed == 42 else f"_s{seed}"
    return (os.path.join(X.REPO, "models", SCEN, f"{name}_sac{suf}.zip"),
            os.path.join(X.REPO, "models", SCEN, f"{name}{suf}", "vec_normalize.pkl"))


def checkpoint_stamp(label, seed):
    """Provenance of the checkpoint (see provenance.py); stored in every output."""
    return _stamp(X.REPO, SCEN, label, seed)


def plain_env(state_fn):
    return TCCPLEnv(config_file=os.path.join(X.REPO, X.CONFIGS[SCEN]),
                    reward_function=SquaredTrackingErrorReward, state_function=state_fn,
                    save_replay=False, save_plots=False, verbose=False)


def run(label, seed, perturb, n_days, offset=500):
    os.chdir(X.REPO)
    spec = X.spec_of(scenario=SCEN)
    days = []
    if label in ("RR", "CAFAP"):
        return X.evaluate_heuristic(spec, label, n_days, seed_offset=offset)
    if label in ("tccpl", "fixedpenalty"):
        zp, _ = paths(label, seed)
        # act through the residual radius the checkpoint was TRAINED with
        # (first-submission checkpoints carry no rho in their sidecar: 0.25)
        stamp = checkpoint_stamp(label, seed) or {}
        spec = X.spec_of(scenario=SCEN, rho=stamp.get("rho", 0.25),
                         tr_load_aware=stamp.get("tr_load_aware", False))
        print(f"  {label} s{seed}: residual radius {spec['rho']:g} from the checkpoint", flush=True)
        return X.evaluate_learned(spec, zp, n_days, seed_offset=offset, perturb=perturb)

    zp, vp = paths(label, seed)
    state_fn = PublicPST if label in ("ddpg_pst", "ppo_pst") else grid_aware_obs
    vn = VecNormalize.load(vp, DummyVecEnv([lambda: plain_env(state_fn)]))
    vn.training, vn.norm_reward = False, False
    if label == "ddpg_pst":
        model = DDPG.load(zp, device="cpu")
    elif label == "ppo_pst":
        model = PPO.load(zp, device="cpu")
    else:
        from cpo_pst import CPOPolicy
        probe = plain_env(grid_aware_obs)
        base = layout_from_env(probe)["base_obs_dim"]
        probe.close()
        sp = gym.spaces.Box(-np.inf, np.inf, (base,), np.float32)
        CPOPolicy.set_extractor_factory(lambda: (BaselineFeatureExtractor(sp, 256), 256))
        model = CPOPolicy.load(zp, device="cpu")

    def act(o, raw):
        return model.predict(vn.normalize_obs(o.reshape(1, -1))[0], deterministic=True)[0]

    X.pin_first_day(offset)
    for d in range(n_days):
        env = plain_env(state_fn)
        days.append(X.rollout_day(env, act, offset + d))
        env.close()
    return days


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n_days = 100
    if "--days" in sys.argv:
        n_days = int(sys.argv[sys.argv.index("--days") + 1])
        args = [a for a in args if a != str(n_days)]
    offset = 500
    if "--offset" in sys.argv:
        offset = int(sys.argv[sys.argv.index("--offset") + 1])
        args = [a for a in args if a != str(offset)]
    label, seed = args[0], int(args[1])
    perturb, tag = None, ""
    if len(args) > 2:
        kind, p = args[2].split(":")
        perturb = X.ObsPerturbation(kind, float(p), seed=seed)
        tag = f"_{kind}{p}"
    if os.environ.get("EXP_SHIELD"):
        tag += "_shield"
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"{label}_s{seed}{tag}_n{n_days}" + ("" if offset == 500 else f"_o{offset}") + ".json")
    stamp = checkpoint_stamp(label, seed)
    if os.path.exists(out) and not os.environ.get("EXP_FORCE"):
        try:
            with open(out) as fh:
                old_stamp = json.load(fh).get("checkpoint")
        except (OSError, ValueError):
            old_stamp = "unreadable"
        if old_stamp == stamp:
            print("exists", out); sys.exit(0)
        print(f"re-evaluating {out}: it was made from another checkpoint", flush=True)
    t0 = time.time()
    days = run(label, seed, perturb, n_days, offset)
    with open(out + ".tmp", "w") as f:
        json.dump({"label": label, "seed": seed, "perturb": tag, "n_days": n_days, "day0_pinned": True,
                   "checkpoint": stamp, "minutes": (time.time() - t0) / 60, "days": days}, f)
    os.replace(out + ".tmp", out)
    print(f"done {label} s{seed}{tag}: {(time.time() - t0) / 60:.1f} min")
