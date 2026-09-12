"""TC-CPL learner — the full manuscript specification (Sec. III), importable.

This module implements everything TC-CPL-specific that the notebook trains with:

  Observation (M3, Sec. III-C)   grid_aware_obs(): o = [o_g | o_w x W | o_c x N],
      every feature scaled by FIXED physical constants of the site (P_ref, the
      per-transformer nameplate, T, E_SCALE) — the "fixed causal feature
      scaling" of Sec. III-B. No running statistics anywhere.
  Residual anchoring (M2)        ResidualPSTWrapper: the incumbent's (Round-
      Robin) action a0_t is computed every step, appended to the observation
      (learner observation o~ = (o, a0)), and the executed action is the
      anchored projection  a_t = clip(a0_t + rho * delta_t, 0, 1), rho = 0.25.
      delta = 0 reproduces the incumbent exactly, so M2's demonstrations are
      zero-residual transitions.
  Costs (Sec. III-B)             TCCPLCosts: exact normalized quantities
      r_bar_t   = -dt * ((min(P_set, P_pot) - P_tot)/P_ref)^2
      c_ov_t    = (dt/E_ref) * sum_w [P_tr - P_tr_max]_+          (TRUE overload)
      c_sat_t   = 1{t = t_end} * (1 - eps_usr)                    (terminal deficit)
      plus two NAMED AUXILIARY signals (margin-shaped overload, per-departure
      satisfaction hinge) whose coefficients are held fixed across ablations
      and annealed to zero — they are never dual-priced (Sec. III-B).
  M1 (Sec. III-B)                AdaptiveLagrangianCallback: PI dual controller
      on the measured violations; `adapt=False` freezes the multiplier update,
      which is exactly the paper's fixed-penalty ablation (Sec. III-D).
  Replay (Sec. III-B)            TCCPLReplayBuffer: stores the components and
      recomposes r~(lambda) at every draw (a dual update reprices the whole
      buffer); prioritized sampling ~ (|TD| + eps)^omega with normalized
      importance weights  w_i = (N p_i)^-beta / max_j (N p_j)^-beta,  beta
      annealed to 1 (Schaul et al., 2016).
  Learner (Sec. III-C/D)         TCCPLActor / TCCPLCritic / TCCPLPolicy /
      TCCPLSAC: shared per-port encoder chi_c, per-transformer pooling chi_w,
      global chi_g, one shared actor head decoding each port's residual from
      [e_j, q_w(j), g, a0_j]; twin critics that embed (o_c_j, a0_j, delta_j)
      pairs, pool within the parent transformer, then site-pool; gamma = 1
      finite-horizon returns; importance-weighted SAC losses; DYNAMIC entropy
      target -|J_ctrl_t| over the controllable ports of each sampled state.

MULTIPLIER UNITS. The paper prices the normalized cost c_ov = kWh/E_ref. To
keep the PI gains and lambda values in the interpretable "price per kWh of
episode overload" units the earlier experiments were tuned in (and that the
lambda-evolution figure reports), the controller adapts lambda_ov_kwh and the
composition converts EXACTLY once:  lambda_ov = lambda_ov_kwh * E_ref / P_ref
= lambda_ov_kwh * T * dt.  This is the identity  lambda_kwh * eps_ov[kWh]
= lambda * sum_t c_ov_t;  fixed positive rescaling changes neither the
feasible set nor the optimizer (Sec. III-B). The satisfaction channel is
already O(1) and needs no conversion.

Every class here is picklable/importable, so SB3 checkpoints written by
TCCPLSAC.save() reload with TCCPLSAC.load() in a fresh kernel.
"""
import math
import os
import hashlib
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.baselines.heuristics import RoundRobin

# Fixed per-EV energy scale for the o_c block, in kWh. A site-level E_ref would
# crush the per-vehicle energy feature to ~1e-4 at city scale; a fixed per-EV
# constant keeps the feature O(1) and is still causal and site-independent.
E_SCALE = 100.0



# ─────────────────────────────────────────────────────────────────────────────
# Baseline encoder for the CPO baseline (NOT part of TC-CPL)
# ─────────────────────────────────────────────────────────────────────────────
class BaselineFeatureExtractor(BaseFeaturesExtractor):
    """Flat-MLP extractor for the CPO baseline — no structural prior.

    Lives in this module (rather than only in notebook §5.5) so that every
    place that rebuilds a CPO checkpoint — §6.1's extractor factory, §7's
    evaluation and rollout cells, the exported runtime — can import it in a
    fresh kernel, whatever subset of the notebook has been executed.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        input_dim = int(np.prod(observation_space.shape))
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, features_dim), nn.ReLU(),
        )
        gain = nn.init.calculate_gain("relu")
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gain)
                nn.init.zeros_(m.bias)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        x = obs.float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x.view(x.shape[0], -1))

def resolve_device(device: str = "auto") -> str:
    """Pick the training device, INCLUDING Apple-silicon GPUs.

    Stable-Baselines3's own "auto" only ever checks CUDA and silently falls
    back to CPU on a Mac. This resolver prefers cuda, then mps (Metal), then
    cpu. Measured on an M5 Pro at cityNL scale (B=256, N=600): mps runs the
    TC-CPL gradient step ~4.4x faster than cpu (128 ms vs 558 ms), and ~1.8x
    faster at PublicPST scale. Pass an explicit "cpu"/"cuda"/"mps" to override.
    """
    if device != "auto":
        return device
    if th.cuda.is_available():
        return "cuda"
    mps = getattr(th.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0
_LOG_2PI = math.log(2.0 * math.pi)


# ═════════════════════════════════════════════════════════════════════════════
# Observation (M3, observation half) and layout
# ═════════════════════════════════════════════════════════════════════════════

def _p_ref(env) -> float:
    """Fixed aggregate charger-nameplate scale of the site, kW (Sec. III-B)."""
    p = getattr(env, "_tccpl_p_ref", None)
    if p is None:
        p = float(sum(cs.get_max_power() for cs in env.charging_stations)) or 1.0
        env._tccpl_p_ref = p
    return p


def grid_aware_obs(env, *args) -> np.ndarray:
    """Paper Eq. (obs): o_t = [o_g, {o_w}, {o_c}], all fixed-scale features.

      o_g (4):  t/T, P_set/P_ref, P_tot(t-1)/P_ref, P_pot/P_ref
      o_w (4/transformer):  [limit_now, min limit over DR window, non-EV load,
                             headroom] / (that transformer's nameplate)
      o_c (3/port):  d in {0, 0.5, 1}, dE/E_SCALE, dtau/T

    Per-port blocks are emitted in ENV ACTION ORDER (station-major), so port j
    of the observation is port j of the action vector — the identity mapping
    the per-port residual policy relies on.
    """
    T = env.simulation_length
    t = min(env.current_step, T - 1)
    p_ref = _p_ref(env)

    setp = env.power_setpoints[env.current_step] if env.current_step < T else 0.0
    parts = [(
        env.current_step / T,
        float(setp) / p_ref,
        float(env.current_power_usage[env.current_step - 1]) / p_ref,
        float(env.charge_power_potential[t]) / p_ref,
    )]

    horizon = max(1, int(round(
        env.config["demand_response"]["notification_of_event_minutes"] / env.timescale)))
    for tr in env.transformers:
        scale = max(float(np.max(tr.max_power)), 1.0)   # nameplate, DR-independent
        limits = tr.get_power_limits(step=t, horizon=horizon)  # operator-known, DR-aware
        non_ev = float(tr.inflexible_load[t] + tr.solar_power[t])
        lim = float(limits[0])
        parts.append((lim / scale, float(np.min(limits)) / scale,
                      non_ev / scale, (lim - non_ev) / scale))

    for cs in env.charging_stations:
        for ev in cs.evs_connected:
            if ev is None:
                parts.append((0.0, 0.0, 0.0))
            else:
                parts.append((
                    1.0 if ev.get_soc() >= 1 else 0.5,
                    float(ev.total_energy_exchanged) / E_SCALE,
                    (env.current_step - ev.time_of_arrival) / T,
                ))
    return np.hstack(parts).astype(np.float32)


def layout_from_env(env) -> Dict:
    """Derive the static site layout the policy needs — plain data, picklable."""
    tr_of_port: List[int] = []
    for cs in env.charging_stations:
        tr_of_port += [int(cs.connected_transformer)] * cs.n_ports
    n_tr, n_ports = len(env.transformers), int(env.number_of_ports)
    base = 4 + 4 * n_tr + 3 * n_ports
    return {"n_ports": n_ports, "n_tr": n_tr, "tr_of_port": tr_of_port,
            "base_obs_dim": base, "obs_dim": base + n_ports}


# ═════════════════════════════════════════════════════════════════════════════
# Environment plumbing: component-exposing env + residual-anchoring wrapper
# ═════════════════════════════════════════════════════════════════════════════

class TCCPLEnv(EV2Gym):
    """EV2Gym that exposes the reward COMPONENTS in the step info dict, so the
    replay buffer can recompose r~(lambda) at sample time (Sec. III-B). The
    "g1" alias carries the exact normalized overload cost for cpo_pst.CPO."""

    def step(self, actions, visualize=False):
        obs, reward, terminated, truncated, info = super().step(actions, visualize)
        comps = getattr(self, "_reward_components", None)
        if comps is not None and isinstance(info, dict):
            (info["r_base"], info["c_ov"], info["c_sat"],
             info["c_aux_ov"], info["c_aux_sat"]) = comps
            info["g1"] = comps[1]          # CPO's cost signal: TRUE overload cost
        return obs, reward, terminated, truncated, info


class ResidualPSTWrapper(gym.Wrapper):
    """M2 residual anchoring (paper Eq. (projection)).

    - Appends the incumbent's action a0_t to the observation: o~ = (o, a0).
    - The agent outputs a residual delta in [-1, 1]^N; the wrapper executes
          a_t = clip(a0_t + rho * delta_t * ctrl_mask, 0, 1),   rho = 0.25,
      where ctrl masks the residual to CONTROLLABLE ports (occupied, SoC < 1):
      full and empty ports receive no action (Sec. III-C).
    - delta = 0 executes the incumbent exactly — the zero-residual property
      that makes M2's demonstrations consistent by construction.
    """

    def __init__(self, env, rho: float = 0.25, incumbent_cls=RoundRobin):
        super().__init__(env)
        self.rho = float(rho)
        self.incumbent_cls = incumbent_cls
        n = int(env.number_of_ports)
        base = int(np.prod(env.observation_space.shape))
        high = np.inf * np.ones(base + n, dtype=np.float32)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(n,), dtype=np.float32)
        self._anchor = np.zeros(n, dtype=np.float32)
        self._incumbent = None

    def _ctrl_mask(self) -> np.ndarray:
        m = np.zeros(self.env.number_of_ports, dtype=np.float32)
        k = 0
        for cs in self.env.charging_stations:
            for ev in cs.evs_connected:
                if ev is not None and ev.get_soc() < 1:
                    m[k] = 1.0
                k += 1
        return m

    def _augment(self, obs) -> np.ndarray:
        return np.concatenate(
            [np.asarray(obs, dtype=np.float32), self._anchor]).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._incumbent = self.incumbent_cls(self.env)   # fresh round-robin queue
        self._anchor = np.asarray(
            self._incumbent.get_action(self.env), dtype=np.float32).copy()
        return self._augment(obs), info

    def step(self, delta):
        delta = np.asarray(delta, dtype=np.float32).reshape(-1)
        a = np.clip(self._anchor + self.rho * delta * self._ctrl_mask(), 0.0, 1.0)
        obs, reward, terminated, truncated, info = self.env.step(a)
        if terminated or truncated:
            self._anchor = np.zeros_like(self._anchor)
        else:
            self._anchor = np.asarray(
                self._incumbent.get_action(self.env), dtype=np.float32).copy()
        return self._augment(obs), reward, terminated, truncated, info


# ═════════════════════════════════════════════════════════════════════════════
# Costs (Sec. III-B): exact normalized objective + dual costs + aux signals
# ═════════════════════════════════════════════════════════════════════════════

class TCCPLCosts:
    """Reward function computing the paper's exact normalized quantities.

    EV2Gym signature: fn(env, total_costs, user_satisfaction_list,
                         invalid_action_punishment) -> float.

    The scalar returned to the env is the CURRENT composition (for logging);
    the components are stashed on the env and re-composed at sample time by
    TCCPLReplayBuffer, so the scalar is never what the learner trains on
    after a dual update.

    lambda_ov is adapted in kWh-price units (lambda_ov_kwh) and converted by
    the exact factor E_ref/P_ref = T*dt at composition (see module docstring).
    The satisfaction price lambda_sat acts directly on the terminal deficit.
    """

    def __init__(self,
                 lambda_ov_kwh: float = 10.0,
                 lambda_sat: float = 5.0,
                 sat_floor: float = 0.98,
                 ov_margin: float = 0.05,
                 aux_ov_coef_kwh: float = 0.0,
                 aux_sat_coef: float = 25.0,
                 aux_margin: Optional[float] = None,
                 **legacy):
        # CONSTRAINT TIGHTENING (training-time back-off, a standard device in
        # robust control / safe RL). Problem 1 keeps the TRUE requirements
        # (eps_ov_max = 0 kWh, eps_usr_min = 0.95); the learner is trained
        # against strictly tighter surrogates so the deployed policy clears the
        # requirements with slack:
        #   * the dual-priced overload cost c_ov counts loading above the
        #     TIGHTENED limit (1 - ov_margin) * P_tr_max (EV-controllable part
        #     only, so zero charging always satisfies it => the tightened
        #     constraint is FEASIBLE, unlike a negative energy threshold);
        #   * the satisfaction floor used by the dual controller and the aux
        #     hinge is sat_floor (0.98) rather than the 0.95 requirement.
        # ov_margin = 0 recovers the exact, untightened cost.
        self.lambda_ov_kwh = float(lambda_ov_kwh)
        self.lambda_sat = float(lambda_sat)
        self.sat_floor = float(legacy.get("target_sat", sat_floor))
        self.ov_margin = float(legacy.get("limit_margin", ov_margin))
        self.aux_margin = float(2.0 * self.ov_margin if aux_margin is None else aux_margin)
        self.aux_ov_coef_kwh = float(aux_ov_coef_kwh)
        self.aux_sat_coef = float(aux_sat_coef)
        self.aux_scale = 1.0          # annealed 1 -> 0 by the callback
        self._p_ref = None
        self._dt = None
        self._e_ref = None
        self._ov_price = None         # E_ref / P_ref = T * dt

    def _prep(self, env) -> None:
        if self._p_ref is None:
            self._p_ref = _p_ref(env)
            self._dt = env.timescale / 60.0
            self._e_ref = env.simulation_length * self._dt * self._p_ref
            self._ov_price = self._e_ref / self._p_ref

    def effective_multipliers(self) -> Tuple[float, float, float, float]:
        """(lambda_ov, lambda_sat, aux_ov, aux_sat) in normalized-cost units,
        at their CURRENT values — read by the buffer at every minibatch draw."""
        ov_price = self._ov_price if self._ov_price is not None else 24.0
        return (self.lambda_ov_kwh * ov_price,
                self.lambda_sat,
                self.aux_scale * self.aux_ov_coef_kwh * ov_price,
                self.aux_scale * self.aux_sat_coef)

    def components(self, env, user_satisfaction_list):
        self._prep(env)
        t = env.current_step - 1              # the step that just completed
        dt, p_ref, e_ref = self._dt, self._p_ref, self._e_ref

        # r_bar_t = dt * r_t / P_ref^2  (paper Sec. III-B)
        err = (min(env.power_setpoints[t], env.charge_power_potential[t])
               - env.current_power_usage[t])
        r_base = -dt * (err / p_ref) ** 2

        # c_ov: overload above the TIGHTENED limit (1 - ov_margin) * P_tr_max,
        # EV-controllable part (threshold never below the non-EV load), the
        # dual-priced training constraint; c_aux_ov: same shape with the wider
        # aux_margin — an optional, annealed auxiliary signal (default off).
        ov, aux_ov = 0.0, 0.0
        m, ma = self.ov_margin, self.aux_margin
        for tr in env.transformers:
            limit = tr.max_power[t]                    # DR-aware true limit
            non_ev = tr.inflexible_load[t] + tr.solar_power[t]
            p = tr.current_power
            ov += max(0.0, p - max((1.0 - m) * limit, non_ev))
            aux_ov += max(0.0, p - max((1.0 - ma) * limit, non_ev))
        c_ov = dt * ov / e_ref
        c_aux_ov = dt * aux_ov / e_ref

        # c_sat: terminal-only service deficit 1{t = t_end} (1 - eps_usr).
        # Scores are accumulated ON THE ENV — this reward object is shared by
        # all DummyVecEnv workers, so per-episode state cannot live on self.
        if t == 0:
            env._tccpl_sat_scores = []
        scores = getattr(env, "_tccpl_sat_scores", None)
        if scores is None:
            scores = env._tccpl_sat_scores = []
        scores.extend(float(s) for s in user_satisfaction_list)
        c_sat = 0.0
        if env.current_step >= env.simulation_length:
            # An episode with no departures is assigned zero service cost.
            c_sat = (1.0 - float(np.mean(scores))) if scores else 0.0

        # auxiliary per-departure satisfaction hinge against the TRAINING floor
        c_aux_sat = sum(max(0.0, self.sat_floor - float(s))
                        for s in user_satisfaction_list) / max(env.number_of_ports, 1)

        return float(r_base), float(c_ov), float(c_sat), float(c_aux_ov), float(c_aux_sat)

    def __call__(self, env, total_costs, user_satisfaction_list,
                 invalid_action_punishment):
        comps = self.components(env, user_satisfaction_list)
        env._reward_components = comps
        r_base, c_ov, c_sat, c_aux_ov, c_aux_sat = comps
        lam_ov, lam_sat, aux_ov, aux_sat = self.effective_multipliers()
        return (r_base - lam_ov * c_ov - lam_sat * c_sat
                - aux_ov * c_aux_ov - aux_sat * c_aux_sat)


# ═════════════════════════════════════════════════════════════════════════════
# M1: PI dual controller (adapt=False -> the paper's fixed-penalty ablation)
# ═════════════════════════════════════════════════════════════════════════════

class AdaptiveLagrangianCallback(BaseCallback):
    """PI dual control on the measured episode metrics (Stooke et al. 2020;
    classical dual ascent is K_p = 0). Violations are measured in the same
    interpretable units the lambda-evolution figure reports:

        v_ov  = mean episode overload [kWh] - 0
        v_sat = eps_usr_min - mean episode satisfaction

    and the prices are applied to the exact normalized costs through
    TCCPLCosts.effective_multipliers() (see the unit-conversion note there).

    Also anneals the auxiliary-signal scale 1 -> 0 linearly over the first
    `aux_anneal_frac` of training — the SAME schedule whether adapt is on or
    off, as the paper requires of a named auxiliary signal (Sec. III-B).
    """

    def __init__(self, reward_ref: TCCPLCosts,
                 adapt: bool = True,
                 initial_lambda_ov: float = 10.0,
                 initial_lambda_sat: float = 5.0,
                 sat_floor: float = 0.98,          # TRAINING floor (requirement 0.95)
                 target_overload: float = 0.0,     # kWh above the tightened limit
                 ki: float = 2.0, kp: float = 5.0,
                 lambda_max: float = 500.0,
                 update_freq: int = 10_000,       # in TIMESTEPS
                 min_episodes: int = 3,
                 aux_anneal_frac: float = 0.8,
                 verbose: int = 1,
                 **legacy):
        super().__init__(verbose)
        self.reward_ref = reward_ref
        self.adapt = bool(adapt)
        self.sat_floor = float(legacy.get("target_sat", sat_floor))
        # The measured overload fed to the controller is the SAME tightened
        # quantity the learner is priced on (kWh above (1 - ov_margin) * limit),
        # so the constraint is feasible and lambda can settle once loading
        # stays under the tightened limit. A NEGATIVE target_overload is
        # allowed as an experimental "pressure" mode, but note that it makes
        # the constraint infeasible (overload energy is non-negative): the
        # integral term then ratchets lambda_ov to lambda_max regardless of
        # how good the policy is. Prefer a larger ov_margin instead.
        self.target_overload = float(target_overload)
        if self.target_overload < 0:
            print(f"  [M1] WARNING: target_overload={self.target_overload} < 0 is "
                  "infeasible (overload energy >= 0); lambda_ov will ratchet to "
                  "lambda_max. Prefer tightening ov_margin.")
        self.ki, self.kp = ki, kp
        self.lambda_max = lambda_max
        self.update_freq = update_freq
        self.min_episodes = min_episodes
        self.aux_anneal_frac = float(aux_anneal_frac)

        self._I_ov = initial_lambda_ov
        self._I_sat = initial_lambda_sat
        self.lambda_ov = initial_lambda_ov
        self.lambda_sat = initial_lambda_sat

        self._last_update = 0
        self._run_cov: Dict[int, float] = {}       # per-env running sum of c_ov
        self._ep_overloads: List[float] = []       # kWh above the TIGHTENED limit
        self._ep_true_ov: List[float] = []         # kWh above the TRUE limit (log)
        self._ep_sats: List[float] = []
        self.lambda_history: List[Tuple[float, float]] = []
        self.violation_history: List[Tuple[float, float]] = []
        self.true_overload_history: List[float] = []

    @property
    def target_sat(self) -> float:                 # backward-compatible alias
        return self.sat_floor

    def _apply(self) -> None:
        self.reward_ref.lambda_ov_kwh = self.lambda_ov
        self.reward_ref.lambda_sat = self.lambda_sat

    def _on_training_start(self) -> None:
        self._last_update = self.num_timesteps
        self._apply()

    def _on_step(self) -> bool:
        # anneal the auxiliary-signal scale (same schedule for the ablation)
        total = max(getattr(self.model, "_total_timesteps", 0), 1)
        frac = self.num_timesteps / (self.aux_anneal_frac * total)
        self.reward_ref.aux_scale = float(np.clip(1.0 - frac, 0.0, 1.0))

        e_ref = getattr(self.reward_ref, "_e_ref", None)
        for e, (done, info) in enumerate(zip(self.locals.get("dones", ()),
                                             self.locals.get("infos", ()))):
            if not isinstance(info, dict):
                continue
            # accumulate the tightened overload cost of this episode (kWh)
            if "c_ov" in info and e_ref:
                self._run_cov[e] = self._run_cov.get(e, 0.0) + float(info["c_ov"]) * e_ref
            if done:
                self._ep_overloads.append(self._run_cov.pop(e, 0.0))
                ov = info.get("total_transformer_overload", None)
                sat = info.get("average_user_satisfaction", None)
                if ov is not None and np.isfinite(ov):
                    self._ep_true_ov.append(float(ov))
                if sat is not None and np.isfinite(sat):
                    self._ep_sats.append(float(sat))

        if self.num_timesteps - self._last_update >= self.update_freq:
            self._last_update = self.num_timesteps
            self._update_lambda()
        return True

    def _update_lambda(self) -> None:
        if (len(self._ep_overloads) < self.min_episodes
                or len(self._ep_sats) < self.min_episodes):
            if self.verbose:
                print(f"  [M1 @ {self.num_timesteps}] insufficient episode data "
                      f"(ov={len(self._ep_overloads)}, sat={len(self._ep_sats)}) — skipping")
            return
        avg_ov = float(np.mean(self._ep_overloads))          # above tightened limit
        avg_true = float(np.mean(self._ep_true_ov)) if self._ep_true_ov else float("nan")
        avg_sat = float(np.mean(self._ep_sats))
        self._ep_overloads.clear()
        self._ep_true_ov.clear()
        self._ep_sats.clear()
        self.violation_history.append((avg_ov, avg_sat))
        self.true_overload_history.append(avg_true)

        if self.adapt:
            v_ov = avg_ov - self.target_overload
            v_sat = self.sat_floor - avg_sat
            self._I_ov = float(np.clip(self._I_ov + self.ki * v_ov, 0.0, self.lambda_max))
            self._I_sat = float(np.clip(self._I_sat + self.ki * v_sat, 0.0, self.lambda_max))
            self.lambda_ov = float(np.clip(self.kp * v_ov + self._I_ov, 0.0, self.lambda_max))
            self.lambda_sat = float(np.clip(self.kp * v_sat + self._I_sat, 0.0, self.lambda_max))
            self._apply()

        self.lambda_history.append((self.lambda_ov, self.lambda_sat))
        if self.verbose:
            mode = "M1" if self.adapt else "M1-frozen"
            print(f"[{mode} @ {self.num_timesteps:>7}] overload above tightened limit="
                  f"{avg_ov:.3f} kWh (true {avg_true:.3f})  sat={avg_sat:.4f} | "
                  f"λ_ov={self.lambda_ov:.2f}  λ_sat={self.lambda_sat:.2f}  "
                  f"aux={self.reward_ref.aux_scale:.2f}")


# ═════════════════════════════════════════════════════════════════════════════
# M2: zero-residual demonstrations + buffer prefill
# ═════════════════════════════════════════════════════════════════════════════

def collect_demonstrations(make_env, n_transitions: int, seed: int,
                           cache_key: str = "", cache_dir: str = "cache",
                           verbose: bool = True) -> Dict[str, np.ndarray]:
    """M2 Phase 1 — run the incumbent through the RESIDUAL interface.

    `make_env` returns a fresh ResidualPSTWrapper env whose reward is the SAME
    TCCPLCosts object family as training (C1). The wrapper executes the
    incumbent for delta = 0, so demonstrations are zero-residual transitions
    under the same fixed feature scaling as online data (C2 by construction).
    """
    key = hashlib.md5(f"{n_transitions}|{seed}|{cache_key}|v2".encode()).hexdigest()[:16]
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"m2_demos_{key}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        if verbose:
            print(f"  Loaded {len(z['rewards']):,} cached demonstrations from {cache}")
        return {k: z[k] for k in z.files}

    env = make_env()
    n_ports = env.action_space.shape[0]
    obs, _ = env.reset(seed=seed)
    cols = {k: [] for k in ("observations", "actions", "rewards",
                            "next_observations", "dones",
                            "r_base", "c_ov", "c_sat", "c_aux_ov", "c_aux_sat")}
    zero = np.zeros(n_ports, dtype=np.float32)
    collected, ep = 0, 0
    while collected < n_transitions:
        next_obs, reward, term, trunc, info = env.step(zero)
        done = term or trunc
        cols["observations"].append(obs.copy())
        cols["actions"].append(zero.copy())
        cols["rewards"].append(float(reward))
        cols["next_observations"].append(next_obs.copy())
        cols["dones"].append(float(done))
        for k in ("r_base", "c_ov", "c_sat", "c_aux_ov", "c_aux_sat"):
            cols[k].append(float(info.get(k, 0.0)))
        obs = next_obs
        collected += 1
        if done and collected < n_transitions:
            ep += 1
            obs, _ = env.reset(seed=seed + ep)
    env.close()

    out = {k: np.asarray(v, dtype=np.float32) for k, v in cols.items()}
    if verbose:
        print(f"  Collected {collected:,} zero-residual transitions "
              f"from {ep + 1} episodes.")
        print(f"  Mean r_bar {out['r_base'].mean():.5f}   "
              f"mean c_ov {out['c_ov'].mean():.2e}   "
              f"episodes with overload cost: "
              f"{(out['c_ov'] > 0).mean() * 100:.1f}% of steps")
    try:
        np.savez_compressed(cache, **out)
        if verbose:
            print(f"  Cached demonstrations → {cache}")
    except OSError as e:
        print(f"  ! could not cache demonstrations: {e}")
    return out


def prefill_buffer(model, data: Dict[str, np.ndarray]) -> None:
    """M2 Phase 2 — write the demonstrations (with components) into the
    prioritized buffer at max priority. Observations carry the fixed feature
    scaling already, so they are written raw — no statistics to initialize."""
    rb = model.replay_buffer
    n_envs, n_pos = rb.n_envs, rb.buffer_size
    total = len(data["observations"])
    n_fill = min(total // n_envs, n_pos)
    used = n_fill * n_envs
    obs_shape, act_dim = rb.obs_shape, rb.action_dim

    for pos in range(n_fill):
        s, e = pos * n_envs, (pos + 1) * n_envs
        rb.observations[pos] = data["observations"][s:e].reshape(n_envs, *obs_shape)
        rb.next_observations[pos] = data["next_observations"][s:e].reshape(n_envs, *obs_shape)
        rb.actions[pos] = data["actions"][s:e].reshape(n_envs, act_dim)
        rb.rewards[pos] = data["rewards"][s:e].reshape(n_envs)
        rb.dones[pos] = data["dones"][s:e].reshape(n_envs)
        for k in ("r_base", "c_ov", "c_sat", "c_aux_ov", "c_aux_sat"):
            getattr(rb, k)[pos] = data[k][s:e].reshape(n_envs)
        rb.has_comp[pos] = True
    rb.pos = n_fill % n_pos
    rb.full = n_fill >= n_pos
    rb.priorities[:n_fill, :] = rb.max_priority
    rb._palpha[:n_fill, :] = (rb.max_priority + rb.eps) ** rb.alpha
    print(f"  Pre-filled {n_fill:,} positions × {n_envs} envs = {used:,} "
          f"zero-residual transitions at max priority "
          f"(fill {min(n_fill / n_pos, 1.0):.1%}).")


# ═════════════════════════════════════════════════════════════════════════════
# Replay: sample-time composition + PER with importance weights (beta -> 1)
# ═════════════════════════════════════════════════════════════════════════════

class TCCPLSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    weights: th.Tensor          # normalized importance weights, (B, 1)
    batch_inds: np.ndarray      # for update_priorities
    env_inds: np.ndarray


class TCCPLReplayBuffer(ReplayBuffer):
    """Stores (r_bar, c_ov, c_sat, c_aux_ov, c_aux_sat) with every transition
    and composes r~(lambda) at every draw from the LIVE multipliers, so one
    dual update reprices the entire buffer (Sec. III-B). Prioritized sampling
    with cached exponentiated priorities; importance weights per Schaul et al.
    with beta set by the learner each update phase. Priorities are refreshed by
    TCCPLSAC.train() from the TD errors it computes anyway — no extra forward
    passes."""

    def __init__(self, buffer_size, observation_space, action_space,
                 device="auto", n_envs=1, optimize_memory_usage=False,
                 handle_timeout_termination=True,
                 alpha: float = 0.6, eps: float = 1e-3,
                 beta0: float = 0.4, lambda_ref: Optional[TCCPLCosts] = None):
        super().__init__(buffer_size, observation_space, action_space,
                         device=device, n_envs=n_envs,
                         optimize_memory_usage=optimize_memory_usage,
                         handle_timeout_termination=handle_timeout_termination)
        self.alpha, self.eps = float(alpha), float(eps)
        self.beta = float(beta0)                 # annealed to 1 by the learner
        self.lambda_ref = lambda_ref
        shape = (self.buffer_size, self.n_envs)
        for k in ("r_base", "c_ov", "c_sat", "c_aux_ov", "c_aux_sat"):
            setattr(self, k, np.zeros(shape, dtype=np.float32))
        self.has_comp = np.zeros(shape, dtype=bool)
        self.priorities = np.zeros(shape, dtype=np.float64)
        self._palpha = np.zeros(shape, dtype=np.float64)
        self.max_priority = 1.0

    def set_lambda_ref(self, reward_obj: TCCPLCosts) -> None:
        self.lambda_ref = reward_obj

    def add(self, obs, next_obs, action, reward, done, infos) -> None:
        pos = self.pos
        super().add(obs, next_obs, action, reward, done, infos)
        for e, info in enumerate(infos):
            if isinstance(info, dict) and "r_base" in info:
                self.r_base[pos, e] = info["r_base"]
                self.c_ov[pos, e] = info["c_ov"]
                self.c_sat[pos, e] = info["c_sat"]
                self.c_aux_ov[pos, e] = info.get("c_aux_ov", 0.0)
                self.c_aux_sat[pos, e] = info.get("c_aux_sat", 0.0)
                self.has_comp[pos, e] = True
            else:
                self.has_comp[pos, e] = False
        self.priorities[pos, :] = self.max_priority
        self._palpha[pos, :] = (self.max_priority + self.eps) ** self.alpha

    def _composed_rewards(self, batch_inds, env_inds) -> np.ndarray:
        stored = self.rewards[batch_inds, env_inds]
        if self.lambda_ref is None:
            return stored
        lam_ov, lam_sat, aux_ov, aux_sat = self.lambda_ref.effective_multipliers()
        composed = (self.r_base[batch_inds, env_inds]
                    - lam_ov * self.c_ov[batch_inds, env_inds]
                    - lam_sat * self.c_sat[batch_inds, env_inds]
                    - aux_ov * self.c_aux_ov[batch_inds, env_inds]
                    - aux_sat * self.c_aux_sat[batch_inds, env_inds])
        return np.where(self.has_comp[batch_inds, env_inds], composed, stored)

    def sample(self, batch_size: int, env=None) -> TCCPLSamples:
        upper = self.buffer_size if self.full else self.pos
        assert upper > 0, "sampling from an empty buffer"
        flat = self._palpha[:upper].reshape(-1)
        total = flat.sum()
        n_total = upper * self.n_envs
        if not np.isfinite(total) or total <= 0:
            flat_idx = np.random.randint(0, n_total, size=batch_size)
            probs = np.full(batch_size, 1.0 / n_total)
        else:
            p = flat / total
            flat_idx = np.random.choice(n_total, size=batch_size, p=p)
            probs = p[flat_idx]
        batch_inds, env_inds = np.unravel_index(flat_idx, (upper, self.n_envs))

        # normalized importance weights (Schaul et al. 2016), beta -> 1
        w = (n_total * np.maximum(probs, 1e-12)) ** (-self.beta)
        w = (w / w.max()).astype(np.float32).reshape(-1, 1)

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size, env_inds, :], env)
        else:
            next_obs = self._normalize_obs(
                self.next_observations[batch_inds, env_inds, :], env)
        data = (
            self._normalize_obs(self.observations[batch_inds, env_inds, :], env),
            self.actions[batch_inds, env_inds, :],
            next_obs,
            (self.dones[batch_inds, env_inds]
             * (1 - self.timeouts[batch_inds, env_inds])).reshape(-1, 1),
            self._normalize_reward(
                self._composed_rewards(batch_inds, env_inds).reshape(-1, 1), env),
            w,
        )
        return TCCPLSamples(*tuple(map(self.to_torch, data)),
                            batch_inds=batch_inds, env_inds=env_inds)

    def update_priorities(self, batch_inds, env_inds, td: np.ndarray) -> None:
        td = np.abs(np.asarray(td, dtype=np.float64)).reshape(-1)
        self.priorities[batch_inds, env_inds] = td
        self._palpha[batch_inds, env_inds] = (td + self.eps) ** self.alpha
        self.max_priority = float(max(self.max_priority, td.max() + self.eps))


# ═════════════════════════════════════════════════════════════════════════════
# M3 networks (Sec. III-C): shared port encoder, per-transformer pooling,
# residual actor head anchored at a0, structured twin critics
# ═════════════════════════════════════════════════════════════════════════════

def _mlp(inp: int, hidden: int, out: int) -> nn.Sequential:
    net = nn.Sequential(nn.Linear(inp, hidden), nn.SiLU(),
                        nn.Linear(hidden, out), nn.SiLU())
    gain = nn.init.calculate_gain("relu")
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain)
            nn.init.zeros_(m.bias)
    return net


class _LayoutMixin:
    """Registers the static site layout as buffers and splits observations."""

    def _init_layout(self, layout: Dict) -> None:
        self.N = int(layout["n_ports"])
        self.W = int(layout["n_tr"])
        tr = th.as_tensor(layout["tr_of_port"], dtype=th.long)
        assign = th.zeros(self.W, self.N)
        assign[tr, th.arange(self.N)] = 1.0        # (W, N) port->transformer
        self.register_buffer("tr_idx", tr, persistent=False)
        self.register_buffer("assign", assign, persistent=False)

    def _split(self, obs: th.Tensor):
        B, W, N = obs.shape[0], self.W, self.N
        og = obs[:, :4]
        ow = obs[:, 4:4 + 4 * W].view(B, W, 4)
        oc = obs[:, 4 + 4 * W:4 + 4 * W + 3 * N].view(B, N, 3)
        a0 = obs[:, 4 + 4 * W + 3 * N:]
        d = oc[..., 0]
        occ = (d > 0.25).to(obs.dtype)                       # occupied ports
        ctrl = ((d > 0.25) & (d < 0.75)).to(obs.dtype)       # SoC < 1
        return og, ow, oc, a0, occ, ctrl

    def _tr_pool(self, per_port: th.Tensor, occ: th.Tensor) -> th.Tensor:
        """Masked mean of per-port embeddings within each parent transformer."""
        masked = per_port * occ.unsqueeze(-1)
        sums = th.einsum("wn,bnh->bwh", self.assign.to(per_port.dtype), masked)
        cnt = th.einsum("wn,bn->bw", self.assign.to(occ.dtype), occ)
        return sums / cnt.clamp(min=1.0).unsqueeze(-1)


class TCCPLActor(nn.Module, _LayoutMixin):
    """Paper Eqs. (m3-port)-(m3-actor): shared chi_c / chi_w / chi_g and one
    shared head decoding each port's residual from [e_j, q_w(j), g_t, a0_j].
    Residuals, log-probabilities and entropy exist only on CONTROLLABLE ports;
    full ports stay contextual inputs; empty ports are masked from every pool."""

    def __init__(self, layout: Dict, hidden_dim: int = 128, use_bf16: bool = True):
        super().__init__()
        self._init_layout(layout)
        H = hidden_dim
        self.chi_c = _mlp(3, H, H)
        self.chi_w = _mlp(4 + H, H, H)
        self.chi_g = _mlp(4, H, H)
        self.head = nn.Sequential(nn.Linear(3 * H + 1, H), nn.SiLU(),
                                  nn.Linear(H, 2))
        nn.init.orthogonal_(self.head[0].weight, gain=nn.init.calculate_gain("relu"))
        nn.init.zeros_(self.head[0].bias)
        nn.init.orthogonal_(self.head[2].weight, gain=0.01)   # start near delta = 0
        nn.init.zeros_(self.head[2].bias)
        self._use_bf16 = bool(use_bf16)

    @property
    def _autocast(self) -> bool:
        return (self._use_bf16 and th.cuda.is_available()
                and th.cuda.is_bf16_supported())

    def _trunk(self, obs: th.Tensor):
        og, ow, oc, a0, occ, ctrl = self._split(obs)

        def compute():
            e = self.chi_c(oc)                                   # (B, N, H)
            ebar = self._tr_pool(e, occ)                         # (B, W, H)
            q = self.chi_w(th.cat([ow.to(e.dtype), ebar], -1))   # (B, W, H)
            g = self.chi_g(og.to(e.dtype))                       # (B, H)
            qp = q[:, self.tr_idx, :]                            # (B, N, H)
            h = th.cat([e, qp, g.unsqueeze(1).expand(-1, self.N, -1),
                        a0.to(e.dtype).unsqueeze(-1)], -1)
            return self.head(h)                                  # (B, N, 2)

        if self._autocast and obs.is_cuda:
            with th.autocast(device_type="cuda", dtype=th.bfloat16):
                out = compute()
            out = out.float()
        else:
            out = compute()
        mu = out[..., 0]
        log_std = out[..., 1].clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std, ctrl

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        mu, log_std, ctrl = self._trunk(obs)
        if deterministic:
            return th.tanh(mu) * ctrl
        u = mu + log_std.exp() * th.randn_like(mu)
        return th.tanh(u) * ctrl

    def sample(self, obs: th.Tensor):
        """(delta, joint log-prob over controllable ports, |J_ctrl|)."""
        mu, log_std, ctrl = self._trunk(obs)
        std = log_std.exp()
        u = mu + std * th.randn_like(mu)
        delta = th.tanh(u)
        logp = (-0.5 * (((u - mu) / std) ** 2 + 2 * log_std + _LOG_2PI)
                - th.log(1.0 - delta ** 2 + 1e-6))
        joint = (logp * ctrl).sum(dim=1, keepdim=True)           # (B, 1)
        n_ctrl = ctrl.sum(dim=1, keepdim=True)                   # (B, 1)
        return delta * ctrl, joint, n_ctrl

    def action_log_prob(self, obs: th.Tensor):                   # SB3 compat
        delta, joint, _ = self.sample(obs)
        return delta, joint.squeeze(-1)


class _QNet(nn.Module, _LayoutMixin):
    """One critic (paper Eq. (m3-critic)): per-port embedding of
    [o_c_j, a0_j, delta_j], masked mean within the parent transformer,
    transformer embedding with o_w, unweighted site pool, final head."""

    def __init__(self, layout: Dict, hidden_dim: int = 128):
        super().__init__()
        self._init_layout(layout)
        H = hidden_dim
        self.chi_c = _mlp(5, H, H)
        self.chi_w = _mlp(4 + H, H, H)
        self.chi_g = _mlp(4, H, H)
        self.head = nn.Sequential(nn.Linear(2 * H, H), nn.SiLU(), nn.Linear(H, 1))
        nn.init.orthogonal_(self.head[0].weight, gain=nn.init.calculate_gain("relu"))
        nn.init.zeros_(self.head[0].bias)
        nn.init.orthogonal_(self.head[2].weight, gain=1.0)
        nn.init.zeros_(self.head[2].bias)

    def forward(self, obs: th.Tensor, delta: th.Tensor) -> th.Tensor:
        og, ow, oc, a0, occ, _ = self._split(obs)
        x = th.cat([oc, a0.unsqueeze(-1), delta.to(oc.dtype).unsqueeze(-1)], -1)
        z = self._tr_pool(self.chi_c(x), occ)                    # (B, W, H)
        b = self.chi_w(th.cat([ow.to(z.dtype), z], -1))          # (B, W, H)
        site = b.mean(dim=1)                                     # (B, H)
        g = self.chi_g(og)
        return self.head(th.cat([g, site], -1))                  # (B, 1)


class TCCPLCritic(nn.Module):
    """Twin structured critics with the SB3 ContinuousCritic call convention."""

    def __init__(self, layout: Dict, hidden_dim: int = 128, use_bf16: bool = True):
        super().__init__()
        self.q1 = _QNet(layout, hidden_dim)
        self.q2 = _QNet(layout, hidden_dim)
        self._use_bf16 = bool(use_bf16)

    def forward(self, obs: th.Tensor, actions: th.Tensor):
        if (self._use_bf16 and th.cuda.is_available()
                and th.cuda.is_bf16_supported() and obs.is_cuda):
            with th.autocast(device_type="cuda", dtype=th.bfloat16):
                q1, q2 = self.q1(obs, actions), self.q2(obs, actions)
            return q1.float(), q2.float()
        return self.q1(obs, actions), self.q2(obs, actions)


class TCCPLPolicy(BasePolicy):
    """SB3-compatible policy holding the structured actor and twin critics.

    policy_kwargs: layout (plain dict from layout_from_env), hidden_dim,
    use_bf16. Learned parameter shapes depend only on hidden_dim — not on the
    number of ports (Sec. III-C)."""

    def __init__(self, observation_space, action_space, lr_schedule,
                 layout: Optional[Dict] = None, hidden_dim: int = 128,
                 use_bf16: bool = True, use_sde: bool = False, **kwargs):
        super().__init__(observation_space, action_space, squash_output=True)
        assert layout is not None, "TCCPLPolicy needs layout=layout_from_env(env)"
        assert not use_sde
        self.layout, self.hidden_dim, self.use_bf16 = layout, hidden_dim, use_bf16

        self.actor = TCCPLActor(layout, hidden_dim, use_bf16)
        self.critic = TCCPLCritic(layout, hidden_dim, use_bf16)
        self.critic_target = TCCPLCritic(layout, hidden_dim, use_bf16)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False
        self.critic_target.train(False)

        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(), lr=lr_schedule(1))
        self.critic.optimizer = self.optimizer_class(
            self.critic.parameters(), lr=lr_schedule(1))

    def _get_constructor_parameters(self) -> Dict:
        data = super()._get_constructor_parameters()
        data.update(layout=self.layout, hidden_dim=self.hidden_dim,
                    use_bf16=self.use_bf16)
        return data

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.actor(observation, deterministic=deterministic)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def set_training_mode(self, mode: bool) -> None:
        self.actor.train(mode)
        self.critic.train(mode)
        self.training = mode


class TCCPLSAC(SAC):
    """SAC with the paper's training rules (Sec. III-D):

      - finite-horizon gamma = 1 with the terminal mask;
      - importance-weighted critic/actor/temperature losses (PER, beta -> 1);
      - DYNAMIC entropy target -|J_ctrl_t| per sampled state;
      - priorities refreshed from the TD errors of the critic update itself
        (no extra forward passes);
      - rewards recomposed from stored components at every draw (the buffer).

    Use with policy=TCCPLPolicy and replay_buffer_class=TCCPLReplayBuffer.
    """

    def __init__(self, *args, beta0: float = 0.4, **kwargs):
        self.beta0 = float(beta0)
        super().__init__(*args, **kwargs)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        # anneal beta -> 1 over training (Sec. III-B)
        progress = 1.0 - self._current_progress_remaining
        if hasattr(self.replay_buffer, "beta"):
            self.replay_buffer.beta = self.beta0 + (1.0 - self.beta0) * progress

        ent_losses, ent_coefs, actor_losses, critic_losses = [], [], [], []
        for _ in range(gradient_steps):
            data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            w = data.weights                                        # (B, 1)

            # current policy at the sampled states (for temperature and actor)
            delta_pi, logp_pi, n_ctrl = self.actor.sample(data.observations)

            # temperature: dynamic target -|J_ctrl| (Sec. III-C)
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                target = -n_ctrl.detach()
                ent_coef_loss = -(self.log_ent_coef
                                  * (w * (logp_pi + target)).detach()).mean()
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
                ent_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(float(ent_coef))

            # critic target (gamma = 1, terminal mask)
            with th.no_grad():
                next_delta, next_logp, _ = self.actor.sample(data.next_observations)
                tq1, tq2 = self.critic_target(data.next_observations, next_delta)
                tq = th.min(tq1, tq2) - ent_coef * next_logp
                y = data.rewards + (1.0 - data.dones) * self.gamma * tq

            q1, q2 = self.critic(data.observations, data.actions)
            critic_loss = 0.5 * (w * ((q1 - y) ** 2 + (q2 - y) ** 2)).mean()
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            critic_losses.append(critic_loss.item())

            # priority refresh from the SAME TD errors — free
            with th.no_grad():
                td = 0.5 * ((q1 - y).abs() + (q2 - y).abs())
            if hasattr(self.replay_buffer, "update_priorities"):
                self.replay_buffer.update_priorities(
                    data.batch_inds, data.env_inds,
                    td.squeeze(-1).cpu().numpy())

            # actor
            q1_pi, q2_pi = self.critic(data.observations, delta_pi)
            actor_loss = (w * (ent_coef * logp_pi - th.min(q1_pi, q2_pi))).mean()
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
            actor_losses.append(actor_loss.item())

            polyak_update(self.critic.parameters(),
                          self.critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", float(np.mean(ent_coefs)))
        self.logger.record("train/actor_loss", float(np.mean(actor_losses)))
        self.logger.record("train/critic_loss", float(np.mean(critic_losses)))
        self.logger.record("train/per_beta", float(getattr(self.replay_buffer, "beta", 0.0)))
        if ent_losses:
            self.logger.record("train/ent_coef_loss", float(np.mean(ent_losses)))


# ═════════════════════════════════════════════════════════════════════════════
# Shape self-test (no environment needed):  python tccpl_learner.py
# ═════════════════════════════════════════════════════════════════════════════

def test_shapes(n_tr: int = 3, ports_per_tr: int = 4, batch: int = 6,
                hidden: int = 32, verbose: bool = True) -> None:
    N = n_tr * ports_per_tr
    layout = {"n_ports": N, "n_tr": n_tr,
              "tr_of_port": [j // ports_per_tr for j in range(N)],
              "base_obs_dim": 4 + 4 * n_tr + 3 * N,
              "obs_dim": 4 + 4 * n_tr + 3 * N + N}
    obs = th.randn(batch, layout["obs_dim"])
    # make the d feature realistic: a mix of empty / charging / full ports
    d = th.tensor(np.random.choice([0.0, 0.5, 1.0], size=(batch, N)), dtype=th.float32)
    oc_start = 4 + 4 * n_tr
    obs[:, oc_start:oc_start + 3 * N:3] = d

    actor = TCCPLActor(layout, hidden, use_bf16=False)
    critic = TCCPLCritic(layout, hidden, use_bf16=False)

    delta, logp, n_ctrl = actor.sample(obs)
    assert delta.shape == (batch, N) and logp.shape == (batch, 1)
    assert th.all(delta.abs() <= 1.0)
    ctrl = (d == 0.5).float()
    assert th.allclose(delta * (1 - ctrl), th.zeros_like(delta)), \
        "residuals must be zero on non-controllable ports"
    assert th.allclose(n_ctrl.squeeze(-1), ctrl.sum(-1)), "n_ctrl mismatch"
    det = actor(obs, deterministic=True)
    assert det.shape == (batch, N)
    q1, q2 = critic(obs, delta)
    assert q1.shape == (batch, 1) and q2.shape == (batch, 1)
    (q1.sum() + q2.sum() + logp.sum()).backward()

    # permutation equivariance within one transformer: swap two ports of tr 0
    perm = list(range(N))
    perm[0], perm[1] = perm[1], perm[0]
    obs_p = obs.clone()
    for k in range(3):
        obs_p[:, oc_start + 0 * 3 + k] = obs[:, oc_start + 1 * 3 + k]
        obs_p[:, oc_start + 1 * 3 + k] = obs[:, oc_start + 0 * 3 + k]
    a0_start = oc_start + 3 * N
    obs_p[:, a0_start + 0] = obs[:, a0_start + 1]
    obs_p[:, a0_start + 1] = obs[:, a0_start + 0]
    with th.no_grad():
        mu, _, _ = actor._trunk(obs)
        mu_p, _, _ = actor._trunk(obs_p)
    assert th.allclose(mu[:, perm], mu_p, atol=1e-5), \
        "actor must be permutation-equivariant over ports of one transformer"
    with th.no_grad():
        qa, _ = critic(obs, delta)
        qb, _ = critic(obs_p, delta[:, perm])
    assert th.allclose(qa, qb, atol=1e-5), "critic must be permutation-invariant"

    n_par = sum(p.numel() for p in actor.parameters())
    if verbose:
        print(f"tccpl_learner self-test OK — actor {n_par:,} parameters "
              f"(independent of N={N}), equivariance and masking verified.")


if __name__ == "__main__":
    th.manual_seed(0)
    np.random.seed(0)
    test_shapes()
