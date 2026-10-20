"""Experiment library for the TTE major revision (TTE-Reg-2026-09-3401).

Everything here REUSES the learner of the submitted paper (tccpl_learner.py)
unchanged and only adds switches around it, so every variant differs from
TC-CPL in exactly the component it ablates:

  adapt        M1 on/off (False = the fixed-price ablation, prices frozen)
  lam_ov0/lam_sat0, lam_max, kp, ki, kd, tau_ov, update_freq    M1 settings (kd > 0 = PID)
  incumbent    "RR" (Round-Robin, the paper) or "CAFAP"
  rho          residual radius of the anchored projection
  demos        M2 warm start (D zero-residual demonstrations) on/off
  residual     M2 residual anchoring on/off (off = absolute action, no anchor)
  grid_block   M3 transformer block o_w on/off (off = zeros in the observation)
  encoder      "m3" (shared per-port / per-transformer encoder) or "flat" (MLP)

Outputs never touch models/ or results/ of the paper: every run writes into
revision/runs/<tag>/.
"""
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

from ev2gym.rl_agent.reward import SquaredTrackingErrorReward
from ev2gym.baselines.heuristics import RoundRobin
from tccpl_learner import (TCCPLEnv, ResidualPSTWrapper, TCCPLCosts,
                           AdaptiveLagrangianCallback, prefill_buffer,
                           TCCPLReplayBuffer, TCCPLSAC, TCCPLPolicy,
                           layout_from_env, grid_aware_obs, resolve_device,
                           LOG_STD_MIN, LOG_STD_MAX, _LOG_2PI, E_SCALE, _p_ref)

CONFIGS = {
    "PublicPST": "ev2gym/example_config_files/PublicPST.yaml",
    "cityNL": "ev2gym/example_config_files/cityNL.yaml",
    "PublicPST_stress": "revision/configs/PublicPST_stress.yaml",
}

# The revised protocol (= the defaults of tccpl_runtime.CONFIG: residual radius 1,
# load-aware M3). A variant overrides keys; rho=0.25 / tr_load_aware=False = first submission.
DEFAULT_SPEC = dict(
    scenario="PublicPST_stress", seed=42, budget=300_000, device="cpu",
    n_envs=32, buffer_size=200_000, batch_size=256, lr=3e-4, tau=0.005,
    gamma=1.0, hidden=128, per_alpha=0.6, per_eps=1e-3, per_beta0=0.4,
    adapt=True, lam_ov0=10.0, lam_sat0=5.0, lam_max=500.0, kp=5.0, ki=2.0, kd=0.0,
    tau_ov=0.0, update_freq=10_000, ov_margin=0.05, sat_floor=0.98,
    aux_sat_coef=25.0, aux_anneal_frac=0.8,
    incumbent="RR", rho=1.0, demos=True, n_demos=100_000, residual=True,
    grid_block=True, encoder="m3", tr_load_aware=True,
)


# keys that do not change what a run computes (where it ran)
SPEC_IGNORE = ("device",)


def same_spec(a: Dict, b: Dict) -> bool:
    """True when two specs describe the same experiment (device ignored)."""
    keys = (set(a) | set(b)) - set(SPEC_IGNORE)
    return all(a.get(k) == b.get(k) for k in keys)


def spec_of(**over) -> Dict:
    s = dict(DEFAULT_SPEC)
    unknown = set(over) - set(s)
    assert not unknown, f"unknown spec keys {unknown}"
    s.update(over)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Incumbents
# ─────────────────────────────────────────────────────────────────────────────
class CAFAPIncumbent:
    """Charge-as-fast-as-possible, with the constructor signature the wrapper
    uses (EV2Gym's own class takes no env)."""

    def __init__(self, env, **kw):
        pass

    def get_action(self, env):
        return np.ones(env.number_of_ports)


INCUMBENTS = {"RR": RoundRobin, "CAFAP": CAFAPIncumbent}


# ─────────────────────────────────────────────────────────────────────────────
# Observation variants (M3 grid block off; evaluation-time perturbations)
# ─────────────────────────────────────────────────────────────────────────────
def grid_obs_no_w(env, *args) -> np.ndarray:
    """M3 ablation: the transformer block o_w is withheld (zeros)."""
    o = grid_aware_obs(env, *args)
    W = len(env.transformers)
    o[4:4 + 4 * W] = 0.0
    return o


class ObsPerturbation:
    """Evaluation-time corruption of what the POLICY observes; the plant, the
    costs and the reported metrics are untouched.

    kind:
      "status_flip"   occupied port's charging/full flag flipped with prob p
      "status_hide"   occupied port reported empty (all port features 0), prob p
      "status_delay"  port block one dispatch interval old
      "nev_bias"      observed non-EV load scaled by (1 + p)
      "nev_persist"   observed non-EV load is the previous interval's value
      "rating_bias"   observed transformer limits scaled by (1 + p)
      "dr_notice"     DR notification shortened to p steps (0 = no notice)
    """

    def __init__(self, kind: str, p: float, seed: int = 0):
        self.kind, self.p = kind, float(p)
        self.rng = np.random.default_rng(seed)
        self._prev_c = None

    def __call__(self, env, *args) -> np.ndarray:
        if self.kind == "dr_notice":
            return _grid_obs_horizon(env, int(self.p))
        o = grid_aware_obs(env, *args)
        W, N = len(env.transformers), int(env.number_of_ports)
        ow = o[4:4 + 4 * W].reshape(W, 4)
        oc = o[4 + 4 * W:4 + 4 * W + 3 * N].reshape(N, 3)
        if self.kind == "nev_persist":
            # previous interval's non-EV load instead of the realized value
            t = min(env.current_step, env.simulation_length - 1)
            tp = max(t - 1, 0)
            for w, tr in enumerate(env.transformers):
                scale = max(float(np.max(tr.max_power)), 1.0)
                nev = float(tr.inflexible_load[tp] + tr.solar_power[tp]) / scale
                ow[w, 2] = nev
                ow[w, 3] = ow[w, 0] - nev
        elif self.kind == "nev_bias":
            nev = ow[:, 2] * (1.0 + self.p)
            ow[:, 3] = ow[:, 0] - nev
            ow[:, 2] = nev
        elif self.kind == "rating_bias":
            ow[:, 0] *= (1.0 + self.p)
            ow[:, 1] *= (1.0 + self.p)
            ow[:, 3] = ow[:, 0] - ow[:, 2]
        elif self.kind == "status_flip":
            occ = oc[:, 0] > 0.25
            flip = occ & (self.rng.random(N) < self.p)
            oc[flip, 0] = np.where(oc[flip, 0] > 0.75, 0.5, 1.0)
        elif self.kind == "status_hide":
            occ = oc[:, 0] > 0.25
            hide = occ & (self.rng.random(N) < self.p)
            oc[hide, :] = 0.0
        elif self.kind == "status_delay":
            cur = oc.copy()
            if self._prev_c is not None and env.current_step > 0:
                oc[:, :] = self._prev_c
            self._prev_c = cur
        else:
            raise ValueError(self.kind)
        o[4:4 + 4 * W] = ow.reshape(-1)
        o[4 + 4 * W:4 + 4 * W + 3 * N] = oc.reshape(-1)
        return o


def _grid_obs_horizon(env, h_steps: int) -> np.ndarray:
    """grid_aware_obs with the DR look-ahead window shortened to h_steps
    (h_steps = 0: only the limit already in force is known)."""
    T = env.simulation_length
    t = min(env.current_step, T - 1)
    p_ref = _p_ref(env)
    setp = env.power_setpoints[env.current_step] if env.current_step < T else 0.0
    parts = [(env.current_step / T, float(setp) / p_ref,
              float(env.current_power_usage[env.current_step - 1]) / p_ref,
              float(env.charge_power_potential[t]) / p_ref)]
    for tr in env.transformers:
        scale = max(float(np.max(tr.max_power)), 1.0)
        if h_steps <= 0:
            lim = float(tr.max_power[t])
            lim_min = lim
        else:
            limits = tr.get_power_limits(step=t, horizon=h_steps)
            lim, lim_min = float(limits[0]), float(np.min(limits))
            # get_power_limits only reveals events announced <= h steps ahead;
            # the limit already in force is always known
            lim = min(lim, float(tr.max_power[t]))
            lim_min = min(lim_min, lim)
        non_ev = float(tr.inflexible_load[t] + tr.solar_power[t])
        parts.append((lim / scale, lim_min / scale, non_ev / scale, (lim - non_ev) / scale))
    for cs in env.charging_stations:
        for ev in cs.evs_connected:
            if ev is None:
                parts.append((0.0, 0.0, 0.0))
            else:
                parts.append((1.0 if ev.get_soc() >= 1 else 0.5,
                              float(ev.total_energy_exchanged) / E_SCALE,
                              (env.current_step - ev.time_of_arrival) / T))
    return np.hstack(parts).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation-time headroom shield (R4.2 / R4.5): a constraint-enforcement layer
# ─────────────────────────────────────────────────────────────────────────────
def shield_action(raw, a, margin: float = 0.0) -> np.ndarray:
    """Scale the commanded actions behind every transformer so that the largest
    EV power they can draw (station rating x sum of its commanded port fractions, normalized to <= 1)
    never exceeds that transformer's headroom (1 - margin) x limit - non-EV load
    in the interval being dispatched. EV acceptance and efficiency can only
    lower the realized power, so this removes overload whenever the non-EV load
    is known; it changes nothing on transformers with enough headroom."""
    T = raw.simulation_length
    t = min(raw.current_step, T - 1)
    a = np.asarray(a, dtype=float).reshape(-1).copy()
    trs = {tr.id: tr for tr in raw.transformers}
    upper = {w: 0.0 for w in trs}
    slices = []
    k = 0
    for cs in raw.charging_stations:
        sl = slice(k, k + cs.n_ports)
        k += cs.n_ports
        # a station shares one connection: fractions above 1 in total are
        # normalized (as the station itself does), so scaling is linear below
        tot = float(a[sl].sum())
        if tot > 1.0:
            a[sl] /= tot
        # every commanded port counts, occupied or not: a vehicle that arrives
        # within the interval draws the current already commanded for its port
        upper[cs.connected_transformer] += float(a[sl].sum()) * cs.get_max_power()
        slices.append((cs.connected_transformer, sl))
    scale = {}
    for w, tr in trs.items():
        head = (1.0 - margin) * float(tr.max_power[t]) - float(tr.inflexible_load[t] + tr.solar_power[t])
        scale[w] = 1.0 if upper[w] <= max(head, 0.0) + 1e-9 else max(head, 0.0) / upper[w]
    for w, sl in slices:
        a[sl] *= scale[w]
    return a.astype(np.float32)


class ShieldedResidualWrapper(ResidualPSTWrapper):
    """TC-CPL interface with the headroom shield applied to the executed action."""

    def step(self, delta):
        delta = np.asarray(delta, dtype=np.float32).reshape(-1)
        a = np.clip(self._anchor + self.rho * delta * self._ctrl_mask(), 0.0, 1.0)
        a = shield_action(self.env, a)
        obs, reward, terminated, truncated, info = self.env.step(a)
        if terminated or truncated:
            self._anchor = np.zeros_like(self._anchor)
        else:
            self._anchor = np.asarray(
                self._incumbent.get_action(self.env), dtype=np.float32).copy()
        return self._augment(obs), reward, terminated, truncated, info


# ─────────────────────────────────────────────────────────────────────────────
# M2 ablation: absolute actions, no incumbent in the loop
# ─────────────────────────────────────────────────────────────────────────────
class DirectPSTWrapper(ResidualPSTWrapper):
    """Same observation layout as the residual interface (so the networks are
    identical), but the anchor block is ZERO and the executed action is the
    absolute fraction a = (delta + 1)/2 on controllable ports. The incumbent is
    still evaluated (self.last_a0) only so demonstrations can be converted to
    absolute actions; the policy never sees it."""

    def _augment(self, obs) -> np.ndarray:
        n = self.env.number_of_ports
        return np.concatenate([np.asarray(obs, dtype=np.float32),
                               np.zeros(n, dtype=np.float32)]).astype(np.float32)

    @property
    def last_a0(self):
        return self._anchor

    def step(self, delta):
        delta = np.asarray(delta, dtype=np.float32).reshape(-1)
        a = np.clip(0.5 * (delta + 1.0), 0.0, 1.0) * self._ctrl_mask()
        obs, reward, terminated, truncated, info = self.env.step(a)
        if terminated or truncated:
            self._anchor = np.zeros_like(self._anchor)
        else:
            self._anchor = np.asarray(
                self._incumbent.get_action(self.env), dtype=np.float32).copy()
        return self._augment(obs), reward, terminated, truncated, info


# ─────────────────────────────────────────────────────────────────────────────
# M3 ablation: flat MLP actor / twin critics with the same interface
# ─────────────────────────────────────────────────────────────────────────────
def _flat_mlp(inp, hidden, out_gain=None):
    net = nn.Sequential(nn.Linear(inp, hidden), nn.ReLU(),
                        nn.Linear(hidden, hidden), nn.ReLU())
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
            nn.init.zeros_(m.bias)
    return net


class FlatActor(nn.Module):
    """One MLP over the whole observation (incl. anchor block) emitting a
    residual mean and log-std for every port; parameters grow with N."""

    def __init__(self, layout: Dict, hidden_dim: int = 256):
        super().__init__()
        self.N, self.W = int(layout["n_ports"]), int(layout["n_tr"])
        self.trunk = _flat_mlp(int(layout["obs_dim"]), hidden_dim)
        self.out = nn.Linear(hidden_dim, 2 * self.N)
        nn.init.orthogonal_(self.out.weight, gain=0.01)
        nn.init.zeros_(self.out.bias)

    def _ctrl(self, obs):
        W, N = self.W, self.N
        d = obs[:, 4 + 4 * W:4 + 4 * W + 3 * N].view(obs.shape[0], N, 3)[..., 0]
        return ((d > 0.25) & (d < 0.75)).to(obs.dtype)

    def _trunk(self, obs):
        out = self.out(self.trunk(obs.float()))
        mu, log_std = out[:, :self.N], out[:, self.N:]
        return mu, log_std.clamp(LOG_STD_MIN, LOG_STD_MAX), self._ctrl(obs)

    def forward(self, obs, deterministic: bool = False):
        mu, log_std, ctrl = self._trunk(obs)
        if deterministic:
            return th.tanh(mu) * ctrl
        return th.tanh(mu + log_std.exp() * th.randn_like(mu)) * ctrl

    def sample(self, obs):
        mu, log_std, ctrl = self._trunk(obs)
        std = log_std.exp()
        u = mu + std * th.randn_like(mu)
        delta = th.tanh(u)
        logp = (-0.5 * (((u - mu) / std) ** 2 + 2 * log_std + _LOG_2PI)
                - th.log(1.0 - delta ** 2 + 1e-6))
        return delta * ctrl, (logp * ctrl).sum(1, keepdim=True), ctrl.sum(1, keepdim=True)

    def action_log_prob(self, obs):
        d, j, _ = self.sample(obs)
        return d, j.squeeze(-1)


class FlatQ(nn.Module):
    def __init__(self, layout: Dict, hidden_dim: int = 256):
        super().__init__()
        n = int(layout["obs_dim"]) + int(layout["n_ports"])
        self.trunk = _flat_mlp(n, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, obs, delta):
        return self.out(self.trunk(th.cat([obs.float(), delta.float()], -1)))


class FlatCritic(nn.Module):
    def __init__(self, layout: Dict, hidden_dim: int = 256):
        super().__init__()
        self.q1, self.q2 = FlatQ(layout, hidden_dim), FlatQ(layout, hidden_dim)

    def forward(self, obs, actions):
        return self.q1(obs, actions), self.q2(obs, actions)


class FlatPolicy(BasePolicy):
    """TCCPLPolicy with the structured M3 networks replaced by flat MLPs."""

    def __init__(self, observation_space, action_space, lr_schedule,
                 layout: Optional[Dict] = None, hidden_dim: int = 256,
                 use_bf16: bool = False, use_sde: bool = False, **kwargs):
        super().__init__(observation_space, action_space, squash_output=True)
        self.layout, self.hidden_dim, self.use_bf16 = layout, hidden_dim, use_bf16
        self.actor = FlatActor(layout, hidden_dim)
        self.critic = FlatCritic(layout, hidden_dim)
        self.critic_target = FlatCritic(layout, hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False
        self.critic_target.train(False)
        self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=lr_schedule(1))
        self.critic.optimizer = self.optimizer_class(self.critic.parameters(), lr=lr_schedule(1))

    def _get_constructor_parameters(self) -> Dict:
        data = super()._get_constructor_parameters()
        data.update(layout=self.layout, hidden_dim=self.hidden_dim, use_bf16=self.use_bf16)
        return data

    def _predict(self, observation, deterministic: bool = False):
        return self.actor(observation, deterministic=deterministic)

    def forward(self, obs, deterministic: bool = False):
        return self._predict(obs, deterministic=deterministic)

    def set_training_mode(self, mode: bool) -> None:
        self.actor.train(mode)
        self.critic.train(mode)
        self.training = mode


# ─────────────────────────────────────────────────────────────────────────────
# Environment factory
# ─────────────────────────────────────────────────────────────────────────────
def state_fn_of(spec: Dict, perturb: Optional[ObsPerturbation] = None):
    if perturb is not None:
        return perturb
    return grid_aware_obs if spec["grid_block"] else grid_obs_no_w


def make_env(spec: Dict, reward_fn, perturb: Optional[ObsPerturbation] = None):
    env = TCCPLEnv(config_file=os.path.join(REPO, CONFIGS[spec["scenario"]]),
                   reward_function=reward_fn,
                   state_function=state_fn_of(spec, perturb),
                   save_replay=False, save_plots=False, verbose=False)
    inc = INCUMBENTS[spec["incumbent"]]
    if spec["residual"] and os.environ.get("EXP_SHIELD"):
        return ShieldedResidualWrapper(env, rho=spec["rho"], incumbent_cls=inc)
    if spec["residual"]:
        return ResidualPSTWrapper(env, rho=spec["rho"], incumbent_cls=inc)
    return DirectPSTWrapper(env, rho=spec["rho"], incumbent_cls=inc)


def make_costs(spec: Dict) -> TCCPLCosts:
    return TCCPLCosts(lambda_ov_kwh=spec["lam_ov0"], lambda_sat=spec["lam_sat0"],
                      sat_floor=spec["sat_floor"], ov_margin=spec["ov_margin"],
                      aux_ov_coef_kwh=0.0, aux_sat_coef=spec["aux_sat_coef"])


def collect_demos(spec: Dict, costs: TCCPLCosts, cache_dir: str) -> Dict[str, np.ndarray]:
    """M2 demonstrations: the incumbent through the SAME interface. Residual
    mode stores delta = 0; direct mode stores the incumbent's absolute action
    as delta = 2 a0 - 1 on controllable ports."""
    key = (f"{spec['scenario']}_{spec['incumbent']}_rho{spec['rho']}_res{int(spec['residual'])}"
           f"_grid{int(spec['grid_block'])}_m{spec['ov_margin']}_f{spec['sat_floor']}"
           f"_n{spec['n_demos']}_s{spec['seed']}_v2")
    path = os.path.join(cache_dir, f"demos_{key}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    env = make_env(spec, costs)
    n = env.action_space.shape[0]
    # Demonstration days come from a seed range DISJOINT from the evaluation
    # days (500..599 + n_days): 100000 + 10000 * (seed % 100) + episode.
    base = 100_000 + 10_000 * (spec["seed"] % 100)
    obs, _ = env.reset(seed=base)
    cols = {k: [] for k in ("observations", "actions", "rewards", "next_observations",
                            "dones", "r_base", "c_ov", "c_sat", "c_aux_ov", "c_aux_sat")}
    ep, got = 0, 0
    while got < spec["n_demos"]:
        if spec["residual"]:
            delta = np.zeros(n, dtype=np.float32)
        else:
            delta = ((2.0 * env.last_a0 - 1.0) * env._ctrl_mask()).astype(np.float32)
        nxt, r, term, trunc, info = env.step(delta)
        done = term or trunc
        cols["observations"].append(obs.copy()); cols["actions"].append(delta.copy())
        cols["rewards"].append(float(r)); cols["next_observations"].append(nxt.copy())
        cols["dones"].append(float(done))
        for k in ("r_base", "c_ov", "c_sat", "c_aux_ov", "c_aux_sat"):
            cols[k].append(float(info.get(k, 0.0)))
        obs = nxt
        got += 1
        if done and got < spec["n_demos"]:
            ep += 1
            obs, _ = env.reset(seed=base + ep)
    env.close()
    out = {k: np.asarray(v, dtype=np.float32) for k, v in cols.items()}
    os.makedirs(cache_dir, exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, path)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
class _Progress(BaseCallback):
    def __init__(self, path: str, every: int = 20_000):
        super().__init__(0)
        self.path, self.every, self._last, self.t0 = path, every, 0, time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last >= self.every:
            self._last = self.num_timesteps
            with open(self.path, "a") as f:
                f.write(f"{self.num_timesteps} {time.time() - self.t0:.0f}\n")
        return True


def train(spec: Dict, out_dir: str) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(REPO)
    set_random_seed(spec["seed"])
    th.manual_seed(spec["seed"])
    np.random.seed(spec["seed"])
    device = resolve_device(spec["device"])
    t0 = time.time()

    costs = make_costs(spec)
    venv = DummyVecEnv([(lambda: make_env(spec, costs)) for _ in range(spec["n_envs"])])
    venv.reset()
    probe = make_env(spec, costs)
    layout = layout_from_env(probe.unwrapped)
    probe.close()

    policy_cls = TCCPLPolicy if spec["encoder"] == "m3" else FlatPolicy
    hidden = spec["hidden"] if spec["encoder"] == "m3" else 256
    model = TCCPLSAC(
        policy=policy_cls, env=venv, learning_rate=spec["lr"],
        buffer_size=spec["buffer_size"],
        learning_starts=0 if spec["demos"] else 100 * spec["n_envs"],
        batch_size=spec["batch_size"], tau=spec["tau"], gamma=spec["gamma"],
        train_freq=(1, "step"), gradient_steps=-1, ent_coef="auto",
        seed=spec["seed"], verbose=0, device=device,
        policy_kwargs=dict(layout=layout, hidden_dim=hidden, use_bf16=False,
                           **({"tr_load_aware": spec["tr_load_aware"]} if spec["encoder"] == "m3" else {})),
        beta0=spec["per_beta0"],
        replay_buffer_class=TCCPLReplayBuffer,
        replay_buffer_kwargs=dict(alpha=spec["per_alpha"], eps=spec["per_eps"],
                                  beta0=spec["per_beta0"], lambda_ref=costs))
    if spec["demos"]:
        data = collect_demos(spec, costs, os.path.join(REPO, "revision", "cache"))
        prefill_buffer(model, data)

    dual = AdaptiveLagrangianCallback(
        costs, adapt=spec["adapt"], initial_lambda_ov=spec["lam_ov0"],
        initial_lambda_sat=spec["lam_sat0"], sat_floor=spec["sat_floor"],
        target_overload=spec["tau_ov"], ki=spec["ki"], kp=spec["kp"],
        kd=spec["kd"], lambda_max=spec["lam_max"], update_freq=spec["update_freq"],
        aux_anneal_frac=spec["aux_anneal_frac"], verbose=1)
    prog = _Progress(os.path.join(out_dir, "progress.txt"))
    model.learn(total_timesteps=spec["budget"], callback=[dual, prog], progress_bar=False)
    model.save(os.path.join(out_dir, "model.zip"))

    n_params = {
        "actor": int(sum(p.numel() for p in model.policy.actor.parameters())),
        "critic_twin": int(sum(p.numel() for p in model.policy.critic.parameters())),
    }
    meta = dict(spec=spec, layout={k: layout[k] for k in ("n_ports", "n_tr", "obs_dim")},
                minutes=(time.time() - t0) / 60, device=device, n_params=n_params,
                lambda_history=dual.lambda_history,
                violation_history=dual.violation_history,
                true_overload_history=dual.true_overload_history,
                update_freq=dual.update_freq, update_steps=list(dual.update_steps),
                finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(os.path.join(out_dir, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    venv.close()
    return meta


def load_model(spec: Dict, path: str):
    return TCCPLSAC.load(path, device="cpu",
                         custom_objects={"replay_buffer_class": TCCPLReplayBuffer})


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation with extended per-day records
# ─────────────────────────────────────────────────────────────────────────────
def _scalar(v):
    while isinstance(v, (tuple, list)):
        v = v[0]
    return float(np.asarray(v).reshape(-1)[0])


def day_record(raw, info: Dict, extra_grid: np.ndarray, p_tr: np.ndarray,
               t_infer: List[float]) -> Dict:
    """Per-day metrics: paper metrics, overload family, per-transformer and
    instantaneous violations, grid-side recount, per-EV service distribution."""
    T = int(raw.simulation_length)
    dt = raw.timescale / 60.0
    ov = np.asarray(raw.tr_overload, float)[:, :T]                    # (W, T) kW
    lim = np.array([np.asarray(tr.max_power, float)[:T] for tr in raw.transformers])
    nameplate = lim.max(axis=1, keepdims=True)
    rec = {k: _scalar(info[k]) for k in (
        "tracking_error", "energy_tracking_error", "power_tracker_violation",
        "average_user_satisfaction", "total_transformer_overload",
        "total_energy_charged", "energy_user_satisfaction") if k in info}
    for b in (0.5, 1, 2, 5, 10, 15, 20):
        rec[f"ov_tol{b:g}"] = float(np.maximum(ov - b, 0).sum() * dt)
    for p in (1, 2, 5, 10):
        rec[f"ov_rel{p:g}"] = float(np.maximum(ov - p / 100 * nameplate, 0).sum() * dt)
    rec["ov_nameplate"] = float(np.maximum(ov - (nameplate - lim), 0).sum() * dt)
    rec["max_excess_kW"] = float(ov.max()) if ov.size else 0.0
    rec["max_excess_pct_rating"] = float((ov / nameplate).max() * 100) if ov.size else 0.0
    rec["n_viol_steps"] = int((ov > 1e-9).sum())
    rec["n_viol_steps_10kW"] = int((ov > 10).sum())
    rec["n_tr_violating"] = int((ov.max(axis=1) > 1e-9).sum())
    rec["n_tr_violating_10kW"] = int((ov.max(axis=1) > 10).sum())
    rec["per_tr_overload_kWh"] = (ov.sum(axis=1) * dt).round(4).tolist()
    # grid-side recount: P_tr,grid = P_tr + sum_j P_batt,j (1/eta_j - 1)
    ov_grid = np.maximum(p_tr + extra_grid - lim, 0.0)
    rec["ov_gridside"] = float(ov_grid.sum() * dt)
    rec["ov_gridside_tol10"] = float(np.maximum(ov_grid - 10, 0).sum() * dt)
    rec["grid_losses_kWh"] = float(extra_grid.sum() * dt)
    # per-EV service of departed vehicles
    scores = [float(s) for cs in raw.charging_stations for s in cs.all_user_satisfaction]
    rec["sat_list"] = [round(s, 5) for s in scores]
    unmet, req = [], []
    for ev in raw.EVs:
        if ev.time_of_departure <= raw.current_step:
            want = float(ev.desired_capacity) - float(ev.battery_capacity_at_arrival)
            unmet.append(max(0.0, float(ev.desired_capacity) - float(ev.current_capacity)))
            req.append(max(want, 0.0))
    rec["n_departed"] = len(scores)
    rec["unmet_kWh"] = float(np.sum(unmet)) if unmet else 0.0
    rec["requested_kWh"] = float(np.sum(req)) if req else 0.0
    rec["infer_ms_mean"] = float(np.mean(t_infer) * 1e3) if t_infer else float("nan")
    rec["infer_ms_p99"] = float(np.percentile(t_infer, 99) * 1e3) if t_infer else float("nan")
    return rec


def rollout_day(env, act_fn, seed: int) -> Dict:
    """Run one evaluation day; act_fn(obs, raw_env) -> action for env.step."""
    obs, _ = env.reset(seed=seed)
    raw = env.unwrapped
    W, T = len(raw.transformers), int(raw.simulation_length)
    extra = np.zeros((W, T))
    p_tr = np.zeros((W, T))
    t_inf, done, info = [], False, {}
    while not done:
        t = raw.current_step
        t0 = time.perf_counter()
        a = act_fn(obs, raw)
        t_inf.append(time.perf_counter() - t0)
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        if t < T:
            for tr in raw.transformers:
                p_tr[tr.id, t] = tr.current_power
            for cs in raw.charging_stations:
                for ev in cs.evs_connected:
                    if ev is not None and ev.current_energy > 0:
                        eta = ev.charge_efficiency if not isinstance(ev.charge_efficiency, dict) else 1.0
                        p = ev.current_energy * 60.0 / raw.timescale
                        extra[cs.connected_transformer, t] += p * (1.0 / max(eta, 1e-3) - 1.0)
    return day_record(raw, dict(info), extra, p_tr, t_inf)


def pin_first_day(seed_offset: int) -> None:
    """Make the first evaluation day identical across controllers.

    With tr_seed -1 (PublicPST-S) the household load, PV and demand-response
    event of a day are drawn from the global numpy state when the env is built.
    From the second day on that state is fixed by the previous day's reset seed
    (the rollout itself draws nothing from it), so only the first day depends on
    whatever ran before; this pins it. No effect on cityNL (tr_seed 42)."""
    np.random.seed(seed_offset - 1)


def evaluate_learned(spec: Dict, model_path: str, n_days: int, seed_offset: int = 500,
                     perturb: Optional[ObsPerturbation] = None) -> List[Dict]:
    os.chdir(REPO)
    model = load_model(spec, model_path)
    out = []
    pin_first_day(seed_offset)
    for d in range(n_days):
        env = make_env(spec, SquaredTrackingErrorReward, perturb=perturb)
        if perturb is not None:
            perturb._prev_c = None
        out.append(rollout_day(env, lambda o, raw: model.predict(o, deterministic=True)[0],
                               seed_offset + d))
        env.close()
    return out


def evaluate_heuristic(spec: Dict, name: str, n_days: int, seed_offset: int = 500) -> List[Dict]:
    """Round-Robin / CAFAP on the same days (no residual wrapper)."""
    from ev2gym.baselines.heuristics import ChargeAsFastAsPossible
    os.chdir(REPO)
    out = []
    pin_first_day(seed_offset)
    for d in range(n_days):
        env = TCCPLEnv(config_file=os.path.join(REPO, CONFIGS[spec["scenario"]]),
                       reward_function=SquaredTrackingErrorReward,
                       state_function=grid_aware_obs, save_replay=False,
                       save_plots=False, verbose=False)
        holder = {}

        def act(o, raw):
            if "a" not in holder:
                holder["a"] = RoundRobin(raw) if name == "RR" else ChargeAsFastAsPossible()
            act_ = holder["a"].get_action(raw)
            return shield_action(raw, act_) if os.environ.get("EXP_SHIELD") else act_
        out.append(rollout_day(env, act, seed_offset + d))
        env.close()
    return out
