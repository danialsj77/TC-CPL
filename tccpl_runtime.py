"""AUTO-GENERATED from TC-CPL.ipynb -- DO NOT EDIT BY HAND.

Regenerate with:  python export_training_module.py

Definition cells 4-32 are reproduced verbatim at module level; each training
cell becomes a `train_<name>()` function. Importing this module reads the seed
from the TCCPL_SEED environment variable (see the CONFIG cell), so set it
BEFORE importing.
"""
import matplotlib
matplotlib.use("Agg")   # training runs are headless


# ====================================================================
# notebook cell 4
# ====================================================================
import os
import json
import time
import copy
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed

from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.rl_agent.reward import SquaredTrackingErrorReward
from ev2gym.rl_agent.state import PublicPST
from ev2gym.baselines.heuristics import RoundRobin

# ── The TC-CPL learner stack (manuscript Sec. III) — see tccpl_learner.py ────
from tccpl_learner import (
    TCCPLEnv, ResidualPSTWrapper, TCCPLCosts, AdaptiveLagrangianCallback,
    collect_demonstrations, prefill_buffer, TCCPLReplayBuffer,
    TCCPLPolicy, TCCPLSAC, grid_aware_obs, layout_from_env, E_SCALE,
    resolve_device,
)

# ── Global seed ──────────────────────────────────────────────────────────────
SEED = 42
set_random_seed(SEED)
th.manual_seed(SEED)
np.random.seed(SEED)

# ── TF32 on Ampere+ GPUs: much faster fp32 matmuls at RL-safe precision ──────
# Implementation detail only (like the bf16 encoder autocast): no architecture,
# loss or hyperparameter changes. No effect on CPU or pre-Ampere GPUs.
th.backends.cuda.matmul.allow_tf32 = True
th.backends.cudnn.allow_tf32 = True

print("All libraries loaded. Seed:", SEED)


# ====================================================================
# notebook cell 6
# ====================================================================
import matplotlib as mpl

# ── Publication style: serif + STIX math (IEEE-like), quiet axes ─────────────
mpl.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset":  "stix",
    "font.size":         10,
    "axes.titlesize":    10,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   8.5,
    "axes.linewidth":    0.8,
    "grid.linewidth":    0.5,
    "grid.alpha":        0.3,
    "lines.linewidth":   1.6,
    "figure.dpi":        110,
    "savefig.dpi":       600,
    "savefig.bbox":      "tight",
    "legend.frameon":    False,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── One identity per algorithm, used by EVERY figure (Okabe–Ito palette) ─────
#    key substrings are matched against the labels used in this notebook, in
#    order, so put the more specific key first if two could ever collide.
#    "label" is for figures (mathtext); "plain" is the same name without
#    mathtext, for printed tables and CSV headers.
ALGO_STYLE = [
    # (key,           figure label,            plain label,           colour,    mk,  ls)
    ("CAFAP",         "CAFAP",                 "CAFAP",               "#999999", "x", ":"),
    ("Round-Robin",   "Round-Robin",           "Round-Robin",         "#E69F00", "v", ":"),
    ("DDPG",          "DDPG-PST",              "DDPG-PST",            "#56B4E9", "^", "-"),
    ("PPO",           "PPO-PST",               "PPO-PST",             "#CC79A7", "P", "-"),
    ("CPO",           "CPO",                   "CPO",                 "#0072B2", "o", "--"),
    ("Fixed",         "SAC (fixed $\\lambda$)",  "SAC (fixed lambda)",  "#D55E00", "d", "--"),
    ("TC-CPL",       "TC-CPL (M1+M2+M3)",    "TC-CPL (M1+M2+M3)",  "#009E73", "D", "-"),
    ("Optimal",       "Optimal (offline)",     "Optimal (offline)",   "#000000", "*", "-."),
]

def get_style(label: str) -> dict:
    """Canonical colour/marker/linestyle for an algorithm label.

    "label" carries mathtext and is for figures; "plain" is the ASCII form and
    is what the printed tables and CSV headers use.
    """
    for key, disp, plain, color, marker, ls in ALGO_STYLE:
        if key in label:
            return {"label": disp, "plain": plain, "color": color,
                    "marker": marker, "ls": ls}
    return {"label": label, "plain": label, "color": "#777777",
            "marker": ".", "ls": "-"}

# ── Metric symbols in mathematical notation (paper Table I definitions) ──────
METRIC_LABELS = {
    "tracking_error":             r"$\epsilon_{tr}$  (kW$^2$)",
    "energy_tracking_error":      r"$|\epsilon_{tr}|$  (kWh)",
    "power_tracker_violation":    r"$\epsilon_{sur}$  (kW)",
    "average_user_satisfaction":  r"$\epsilon_{usr}$",
    "total_transformer_overload": r"Transformer overload  (kWh)",
    "overload_intensity":         r"Overload intensity  (kWh / MWh delivered)",
    "unmet_demand_pct":           r"Unmet demand  (%)",
    "total_reward":               r"Episode return",
    "voltage_violation":          r"Voltage violation  (p.u.)",
    "total_energy_charged":       r"Energy charged  (kWh)",
    "total_profits":              r"Profit  (€)",
    "battery_degradation":        r"Battery degradation",
}

RESULTS_DIR = "results"   # re-pointed to results/<scenario> by the CONFIG cell


def save_fig(fig, name: str) -> None:
    """Save a figure as a 600-dpi PNG — the only render the manuscript uses."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig.savefig(f"{RESULTS_DIR}/{name}.png")
    print(f"Saved → {RESULTS_DIR}/{name}.png")

print("Publication figure style active "
      "(serif + STIX math, Okabe–Ito colours, PNG output).")


# ====================================================================
# notebook cell 7
# ====================================================================
# ── TC-CPL environment plumbing (defined in tccpl_learner.py) ────────────────
# TCCPLEnv           EV2Gym exposing the reward COMPONENTS (r̄, c_ov, c_sat,
#                    c_aux_ov, c_aux_sat) in every step info dict, so the
#                    replay buffer can recompose r̃(λ) at sample time
#                    (paper Sec. III-B). info["g1"] aliases c_ov for CPO.
# ResidualPSTWrapper M2 residual anchoring — computes the incumbent's action
#                    a⁰ₜ every step, appends it to the observation
#                    (õ = (o, a⁰)), and executes the anchored projection
#                        aₜ = clip(a⁰ₜ + ρ·δₜ·ctrl, 0, 1),   ρ = 0.25,
#                    where ctrl masks residuals to CONTROLLABLE ports
#                    (occupied, SoC < 1). δ = 0 reproduces the incumbent.
print("Environment plumbing ready: TCCPLEnv (components in info) + "
      "ResidualPSTWrapper (anchored projection).")


# ====================================================================
# notebook cell 9
# ====================================================================
# The paper observation (M3, observation half) and the static site layout are
# implemented in tccpl_learner.py:
#   grid_aware_obs(env)   o = [o_g | o_w × W | o_c × N] with FIXED physical
#                         scaling — P_ref for powers, each transformer's own
#                         nameplate for its block, E_SCALE per EV, T for time.
#                         Per-port blocks are in ENV ACTION ORDER, so
#                         observation port j IS action port j (the identity
#                         the per-port residual policy relies on).
#   layout_from_env(env)  {n_ports, n_tr, tr_of_port, base_obs_dim, obs_dim}
# The live-scenario probe runs in §6.1, after CONFIG selects the scenario.
print(f"grid_aware_obs ready (fixed per-EV energy scale E_SCALE = {E_SCALE:.0f} kWh).")


# ====================================================================
# notebook cell 11
# ====================================================================
# ── Scenario: BOTH validation cases of the manuscript ────────────────────────
# Run interactively (Jupyter / VS Code), this cell ASKS which scenario to use:
# it lists the two validation configurations, prints the main characteristics
# of the one you pick (read from its YAML and from what is already on disk)
# and waits for a "yes"; any other answer offers the choice again. Headless
# runs — train_one.py, runall.sh, run_sweep.py, papermill — set TCCPL_SCENARIO
# and are never prompted. Every artefact path (models/, logs/, results/) is
# namespaced by scenario, so the two studies never collide.
#   PublicPST : 20 stations × 1 port, one 100-kW transformer, no grid sim
#   cityNL    : 300 stations × 2 ports, 122 substations, full AC power flow
import glob
import sys
import yaml

SCENARIO_CATALOGUE = {   # name → (what it is, what it costs)
    "PublicPST": ("single-transformer public facility of the PST paper [8]",
                  "fast — minutes per learner at smoke budgets; the small-scale study"),
    "cityNL":    ("city-scale Dutch MV feeder: 122 substations, full AC power flow",
                  "slow — about 28 h per seed for all five learners at 300k steps on an "
                  "M5 Pro, ~3x at the 1M-step protocol; the headline study"),
}
SCENARIO_DEFAULT = "cityNL"


def scenario_facts(name: str) -> dict:
    """Main characteristics of one scenario, read from its YAML and from the
    artefacts already on disk — nothing here is hard-coded."""
    path = f"ev2gym/example_config_files/{name}.yaml"
    with open(path, encoding="utf-8") as fh:
        c = yaml.safe_load(fh)
    n_cs, ppc = int(c["number_of_charging_stations"]), int(c["number_of_ports_per_cs"])
    steps, dt = int(c["simulation_length"]), int(c["timescale"])
    tr = c.get("transformer") or {}
    on = lambda sec: "on" if (c.get(sec) or {}).get("include", False) else "off"
    if c.get("simulate_grid"):
        grid = f"full AC power flow ({os.path.basename(c['network_info']['bus_info_file'])})"
    else:
        grid = "off — aggregate transformer model"
    if tr.get("size_from_feeder"):
        trafo = (f"{c['number_of_transformers']} secondary substations, each sized from its "
                 f"feeder node (target loading {tr.get('target_peak_loading')}, IEC 60076 ladder)")
    else:
        trafo = f"{c['number_of_transformers']} × {tr.get('max_power')} kW"
    trained = sorted(os.path.basename(p).split("_sac")[0]
                     for p in glob.glob(f"models/{name}/*_sac*.runmeta.json"))
    evaluated = os.path.exists(f"results/{name}/per_episode_metrics.json")
    return {
        "config file":          path,
        "charge points":        f"{n_cs} stations × {ppc} port(s) = {n_cs * ppc} charge points",
        "transformers":         trafo,
        "grid simulation":      grid,
        "episode":              (f"{steps} steps × {dt} min = {steps * dt / 60:.0f} h from "
                                 f"{int(c['hour']):02d}:{int(c['minute']):02d}, {c['simulation_days']}"),
        "EV arrivals":          (f"ElaadNL '{c['scenario']}' distributions, spawn multiplier "
                                 f"{c['spawn_multiplier']}, V2G {'on' if c.get('v2g_enabled') else 'off'}"),
        "setpoint flexibility": f"{c['power_setpoint_flexiblity']} %",
        "loads / PV / DR":      (f"inflexible loads {on('inflexible_loads')}, solar {on('solar_power')}, "
                                 f"demand response {on('demand_response')}"),
        "trained on disk":      ", ".join(trained) if trained else "nothing yet",
        "evaluated (§7)":       f"yes — results/{name}/" if evaluated else "not yet",
    }


def select_scenario(default: str = SCENARIO_DEFAULT) -> str:
    """Which scenario this kernel works on. Precedence: TCCPL_SCENARIO (set by
    every headless launcher — never asks) → interactive prompt (Jupyter) → default."""
    forced = os.environ.get("TCCPL_SCENARIO")
    if forced:
        assert forced in SCENARIO_CATALOGUE, f"unknown TCCPL_SCENARIO={forced!r}"
        return forced
    if "ipykernel" not in sys.modules:          # plain python: nobody to ask
        return default
    names = list(SCENARIO_CATALOGUE)
    while True:
        print("Which validation scenario should this notebook run?  "
              "(config files: ev2gym/example_config_files/)")
        for i, n in enumerate(names, 1):
            what, cost = SCENARIO_CATALOGUE[n]
            print(f"  [{i}] {n:<10} {what}\n      {'':<10} {cost}")
        try:
            ans = input(f"Scenario — number or name [Enter = {default}]: ").strip()
        except Exception as exc:                # no stdin (nbconvert / papermill)
            print(f"  no interactive input available ({type(exc).__name__}) → {default}")
            return default
        if ans == "":
            choice = default
        elif ans.isdigit() and 1 <= int(ans) <= len(names):
            choice = names[int(ans) - 1]
        else:
            match = [n for n in names if n.lower() == ans.lower()]
            if not match:
                print(f"  '{ans}' is not an option — choose again.\n")
                continue
            choice = match[0]
        print(f"\n{choice} — main characteristics")
        print("-" * 78)
        for k, v in scenario_facts(choice).items():
            print(f"  {k:<21} {v}")
        print("-" * 78)
        try:
            ok = input(f"Use {choice}? [yes / no]: ").strip().lower()
        except Exception:
            ok = "yes"
        if ok in ("y", "yes"):
            print(f"→ {choice} selected.\n")
            return choice
        print("→ choosing again.\n")


SCENARIO_NAME = select_scenario()
assert SCENARIO_NAME in ("cityNL", "PublicPST"), SCENARIO_NAME
os.environ["TCCPL_SCENARIO"] = SCENARIO_NAME    # shell cells / subprocesses agree
RESULTS_DIR = f"results/{SCENARIO_NAME}"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Training budget: ONE number for every learner (protocol parity). The
#    reported protocol is 1,000,000 environment steps × 5 seeds, run on a
#    cluster (run_sweep.py / slurm_sweep.sh). TCCPL_TOTAL_TIMESTEPS overrides
#    it for smaller local runs; run_sweep.py treats a model trained at any
#    other budget as NOT finished, so old 300k checkpoints are retrained.
_BUDGET = int(os.environ.get("TCCPL_TOTAL_TIMESTEPS", 1_000_000))
_NUM_ENVS = 32
_SEEDS = [int(s) for s in os.environ.get("TCCPL_SEEDS", "42,43,44,45,46").split(",")]
assert _SEEDS[0] == 42, ("42 must stay the FIRST seed: its artefacts carry no "
                          "suffix (models/<scenario>/<name>_sac.zip), every other "
                          "seed is suffixed _s<seed>")

CONFIG = {
    "scenario":            SCENARIO_NAME,
    "config_file":         f"ev2gym/example_config_files/{SCENARIO_NAME}.yaml",
    "reward_function":     SquaredTrackingErrorReward,  # DDPG/PPO + all eval envs
    "state_function":      PublicPST,                   # paper-faithful state (DDPG/PPO)
    "state_function_safe": grid_aware_obs,              # M3 observation (constrained models)

    # ── Reproducibility protocol ─────────────────────────────────────────────
    # Five training seeds. Train once per seed (TCCPL_SEED=42…46, or
    # run_sweep.py / slurm_sweep.sh); §7 aggregates over every seed it finds
    # on disk (§7.3 learning curves and §7.8 λ trajectories show the mean and
    # the min–max band across seeds; §7.4 averages the per-episode metrics).
    "seeds":            _SEEDS,
    "run_seed":         int(os.environ.get("TCCPL_SEED", 42)),

    # ── Training device. SB3's "auto" NEVER selects Apple-silicon GPUs, so
    #    resolve_device prefers cuda → mps (Metal) → cpu. Measured on an
    #    M5 Pro: mps is ~4.4x faster than cpu per gradient step at cityNL
    #    scale, ~1.8x at PublicPST scale. Override with TCCPL_DEVICE=cpu/mps.
    "device":            resolve_device(os.environ.get("TCCPL_DEVICE", "auto")),

    # ── Vectorized envs — in-process, so all workers share ONE TCCPLCosts
    #    object and M1's dual update reaches every env by mutating it ─────────
    "num_envs":          _NUM_ENVS,

    # ── Training budgets: _BUDGET (1,000,000 by default) for every learner ──
    "total_timesteps_fixed": _BUDGET,   # SAC (fixed λ) — frozen-multiplier ablation
    "total_timesteps_tccpl": _BUDGET,   # TC-CPL (M1+M2+M3)
    "total_timesteps_ddpg":  _BUDGET,   # DDPG-PST literature baseline
    "total_timesteps_ppo":   _BUDGET,   # PPO-PST on-policy baseline
    "total_timesteps_cpo":   _BUDGET,   # CPO (Achiam 2017) constrained baseline
    # Training-time EvalCallback episodes (the protocol value is 5). A smoke
    # budget evaluates at almost every vector step, so it uses one episode.
    "eval_episodes_train":   5 if _BUDGET >= 10_000 else 1,

    # ── TC-CPL learner (paper Sec. III-D) ────────────────────────────────────
    "gamma":          1.0,     # FINITE-HORIZON return with the terminal mask
    "learning_rate":  3e-4,
    "buffer_size":    200_000,
    "batch_size":     256,
    "tau":            0.005,   # Polyak target-update rate
    "hidden_dim":     128,     # width of every χ network (M3)
    "use_bf16":       True,    # bf16 autocast inside the M3 encoders — CUDA
    #                           Ampere+ only; mps and cpu run fp32
    "rho":            0.25,    # residual radius, paper Eq. (projection)

    # ── Problem 1 REQUIREMENTS (what is evaluated and reported) ─────────────
    "target_sat":         0.95,   # ε_usr_min — the service requirement
    "target_overload":    0.0,    # ε_ov_max  — overload must vanish

    # ── CONSTRAINT TIGHTENING (what the learner is trained against) ──────────
    #    A standard back-off: solve a strictly tighter surrogate so the deployed
    #    policy clears the requirements with slack.
    #    * ov_margin: the dual-priced overload counts loading above
    #      (1 − ov_margin) × the DR-aware transformer limit (EV-controllable
    #      part). Feasible by construction — unlike a NEGATIVE energy threshold,
    #      which no policy can satisfy (λ_ov would ratchet to λ_max forever).
    #    * train_sat_floor: satisfaction floor used by the dual controller.
    #    Override for sweeps with TCCPL_OV_MARGIN / TCCPL_TRAIN_SAT_FLOOR.
    "ov_margin":          float(os.environ.get("TCCPL_OV_MARGIN", 0.05)),
    "train_sat_floor":    float(os.environ.get("TCCPL_TRAIN_SAT_FLOOR", 0.98)),
    "train_target_overload": 0.0, # kWh above the TIGHTENED limit (<0 = infeasible
    #                               "pressure" mode; prefer a larger ov_margin)

    # ── M1 PI dual controller — gains and λ in kWh-PRICE units. The exact
    #    conversion to the paper's normalized-cost units (× E_ref/P_ref = T·Δt)
    #    happens inside TCCPLCosts; see the tccpl_learner.py module docstring.
    "lambda_init_ov":     10.0,
    "lambda_init_sat":     5.0,
    "lambda_ki":           2.0,   # integral gain K_i (classic dual-ascent step)
    "lambda_kp":           5.0,   # proportional gain K_p (Stooke et al. 2020)
    "lambda_max":        500.0,
    "lambda_update_freq": 10_000, # in TIMESTEPS

    # ── Fixed-penalty ablation: Algorithm 1 with line 15 FROZEN ─────────────
    "lambda_ov_fixed":    100.0,
    "lambda_sat_fixed":    50.0,

    # ── Optional auxiliary signals (annealed to zero, never dual-priced).
    #    The overload margin is now dual-priced (ov_margin above), so the
    #    auxiliary overload channel is OFF by default; the per-departure
    #    satisfaction hinge (against train_sat_floor) stays on. ────────────────
    "aux_ov_coef":        0.0,    # kWh-price units, converted like λ_ov
    "aux_sat_coef":       25.0,
    "aux_anneal_frac":    0.8,    # 1 → 0 over the first 80% of the budget

    # ── PER (Sec. III-B): p ∝ (|TD|+ε)^ω; weights ϖ=(Np)^{−β}/max, β → 1 ────
    "per_alpha":          0.6,    # ω
    "per_eps":            1e-3,   # ε_PER
    "per_beta0":          0.4,    # β annealed from here to 1 over training

    # ── M2: incumbent warm start (zero-residual demonstrations) ──────────────
    "offline_transitions": int(os.environ.get("TCCPL_OFFLINE_TRANSITIONS", 100_000)),
    #                      (env override exists for smoke tests only)

    # ── DDPG-PST literature baseline (Yılmaz, Orfanoudakis & Vergara, 2024) ──
    "ddpg_gamma":         0.99,
    "ddpg_tau":           5e-4,
    "ddpg_lr":            1e-3,
    "ddpg_batch_size":    64,
    "ddpg_buffer_size":   int(os.environ.get("TCCPL_DDPG_BUFFER",
                                             min(1_000_000, _BUDGET))),
    #                     [8] uses 1e6. A buffer at least as large as the budget
    #                     never discards a transition, so this is behaviour-
    #                     identical to [8] for any budget ≤ 1e6 — at a RAM cost:
    #                     obs + next_obs at cityNL's obs_dim ≈ 2291 take ~18 GB
    #                     float32 for 1e6 transitions (fine on a cluster node).
    #                     On a laptop set TCCPL_DDPG_BUFFER=300000 (a sliding
    #                     window of the last 300k transitions, ~5.5 GB).
    "ddpg_ou_sigma":      0.2,
    "ddpg_actor_arch":    [128, 128],   # paper: two FC layers of 128
    "ddpg_critic_arch":   [64, 64],     # paper: critic width 64

    # ── PPO-PST on-policy baseline (standard SB3 PPO on the paper's MDP) ─────
    "ppo_gamma":          0.99,
    "ppo_lr":             3e-4,
    "ppo_n_steps":        min(1024, max(8, _BUDGET // _NUM_ENVS)),  # 1024 at the protocol;
    #                     capped so a smoke budget is not one full 32k-step rollout
    "ppo_batch_size":     256,
    "ppo_n_epochs":       10,
    "ppo_gae_lambda":     0.95,
    "ppo_clip_range":     0.2,
    "ppo_arch":           [128, 128],

    # ── CPO rollout per env (256 at the protocol; capped like ppo_n_steps) ──
    "cpo_n_steps":        min(256, max(8, _BUDGET // _NUM_ENVS)),
}

# ── Seed of THIS run, and the per-seed artefact naming ───────────────────────
SEED = CONFIG["run_seed"]
set_random_seed(SEED); th.manual_seed(SEED); np.random.seed(SEED)
SUF = "" if SEED == CONFIG["seeds"][0] else f"_s{SEED}"


def run_paths(name: str) -> dict:
    """All artefact paths of one algorithm for the current scenario + seed."""
    d = f"./models/{SCENARIO_NAME}/{name}{SUF}"
    return {
        "dir":     d + "/",
        "zip":     f"./models/{SCENARIO_NAME}/{name}_sac{SUF}",
        "best":    f"{d}/best_model",
        "vecnorm": f"{d}/vec_normalize.pkl",
        "log":     f"./logs/{SCENARIO_NAME}/{name}{SUF}/",
    }


def lambda_history_path(tag: str) -> str:
    return f"{RESULTS_DIR}/lambda_history_{tag}{SUF}.json"


def eval_every(total_timesteps: int, n_points: int = 20) -> int:
    """eval_freq for EvalCallback, in VEC steps (SB3 multiplies by num_envs)."""
    return max(total_timesteps // (n_points * max(CONFIG["num_envs"], 1)), 1)


print(f"Scenario: {SCENARIO_NAME}   config: {CONFIG['config_file']}")
print(f"Device:   {CONFIG['device']}")
print(f"Requirements: overload = 0 kWh, ε_usr ≥ {CONFIG['target_sat']}   |   "
      f"training thresholds: limit × {1 - CONFIG['ov_margin']:.2f}, "
      f"ε_usr ≥ {CONFIG['train_sat_floor']}")
print(f"Run seed: {SEED}  (protocol seeds: {CONFIG['seeds']})   "
      f"artefact suffix: '{SUF or '(none)'}'   results → {RESULTS_DIR}/")
print(f"Budget: {_BUDGET:,} environment steps per learner   "
      f"(dual update every {CONFIG['lambda_update_freq']:,} steps → "
      f"{_BUDGET // CONFIG['lambda_update_freq']} multiplier updates)")


# ====================================================================
# notebook cell 14
# ====================================================================
def build_env(seed: Optional[int] = None, state_fn=None, reward_fn=None,
              wrap_residual: bool = False):
    """One environment of the active scenario.

    wrap_residual=True gives the TC-CPL interface: grid-aware observation with
    the incumbent anchor appended, residual actions through the anchored
    projection. There is NO VecNormalize on this path — the paper's fixed
    causal feature scaling lives inside the observation itself.
    """
    env = TCCPLEnv(
        config_file     = CONFIG["config_file"],
        reward_function = reward_fn or CONFIG["reward_function"],
        state_function  = state_fn or CONFIG["state_function"],
        save_replay     = False,
        save_plots      = False,
        verbose         = False,
    )
    if wrap_residual:
        env = ResidualPSTWrapper(env, rho=CONFIG["rho"])
    if seed is not None:
        env.reset(seed=seed)
    return env


def build_tccpl_vec_env(reward_fn, n_envs: int = None) -> DummyVecEnv:
    """Training envs for TC-CPL and its frozen-λ ablation: in-process workers
    SHARING one TCCPLCosts object — mutating it is how M1's dual update
    reaches every worker instantly."""
    n = n_envs or CONFIG["num_envs"]
    venv = DummyVecEnv([
        (lambda: build_env(state_fn=CONFIG["state_function_safe"],
                           reward_fn=reward_fn, wrap_residual=True))
        for _ in range(n)])
    venv.reset()
    print(f"  TC-CPL vec env: {n} residual-wrapped workers  "
          f"obs {venv.observation_space.shape}  act {venv.action_space.shape}")
    return venv


def build_baseline_vec_env(reward_fn=None, state_fn=None,
                           n_envs: int = None) -> VecNormalize:
    """Training envs for DDPG-PST / PPO-PST / CPO — VecNormalize-wrapped, as
    those baselines are conventionally run."""
    n  = n_envs or CONFIG["num_envs"]
    rf = reward_fn or CONFIG["reward_function"]
    sf = state_fn or CONFIG["state_function"]
    venv = DummyVecEnv([(lambda: build_env(state_fn=sf, reward_fn=rf))
                        for _ in range(n)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, gamma=0.99)
    venv.reset()
    print(f"  Baseline vec env: {n} workers  state {sf.__name__}  "
          f"obs {venv.observation_space.shape}")
    return venv


def build_eval_env(state_fn=None, wrap_residual: bool = False,
                   vecnormalize: bool = False):
    """Single evaluation env. The reward is ALWAYS the tracking-only objective,
    so every learner's evaluation curve measures the same quantity."""
    def _init():
        return build_env(state_fn=state_fn,
                         reward_fn=SquaredTrackingErrorReward,
                         wrap_residual=wrap_residual)
    env = DummyVecEnv([_init])
    if vecnormalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=False, training=False)
    return env


# ====================================================================
# notebook cell 16
# ====================================================================
import gc
import psutil


def free_mem(*names) -> None:
    """Close and drop training objects by GLOBAL name, then report RAM.
    Shared by every post-training cleanup cell in §6."""
    for name in names:
        obj = globals().pop(name, None)
        if obj is not None and hasattr(obj, "close"):
            try:
                obj.close()
            except Exception:
                pass
    gc.collect()
    if th.cuda.is_available():
        th.cuda.empty_cache()
    print(f"RAM now: {psutil.Process(os.getpid()).memory_info().rss / 1024**3:.2f} GB")


# ====================================================================
# notebook cell 19
# ====================================================================
def make_costs(lambda_ov_kwh: float, lambda_sat: float) -> TCCPLCosts:
    """The paper's exact normalized objective and costs (Sec. III-B):

        r̄ₜ    = −Δt · ((min(P_set, P_pot) − P_tot)/P_ref)²
        c_ovₜ  = (Δt/E_ref) · Σ_w [P_tr − (1−m)·P̄_tr]₊  ← TIGHTENED overload, dual-priced
        c_satₜ = 1{t = t_end} · (1 − ε_usr)             ← terminal deficit, dual-priced

    Constraint tightening: the learner is trained against the limit backed off
    by m = CONFIG["ov_margin"] and the satisfaction floor train_sat_floor
    (0.98), while Problem 1's requirements (0 kWh, 0.95) are what §7 reports.
    The optional auxiliary signals (annealed, never dual-priced) stay available.

    λ_ov is adapted in kWh-price units and converted internally by the exact
    factor E_ref/P_ref = T·Δt (see tccpl_learner.py). One factory serves
    TC-CPL (λ adapted), the frozen-λ ablation, and CPO (λ = 0: pure r̄ with
    c_ov exposed as info["g1"], its cost signal).
    """
    return TCCPLCosts(
        lambda_ov_kwh   = lambda_ov_kwh,
        lambda_sat      = lambda_sat,
        sat_floor       = CONFIG["train_sat_floor"],
        ov_margin       = CONFIG["ov_margin"],
        aux_ov_coef_kwh = CONFIG["aux_ov_coef"],
        aux_sat_coef    = CONFIG["aux_sat_coef"],
    )


print(f"TCCPLCosts factory ready: r̄ + dual costs (overload above "
      f"{1 - CONFIG['ov_margin']:.0%} of the limit, terminal deficit) "
      f"+ optional annealed auxiliary signals.")


# ====================================================================
# notebook cell 20
# ====================================================================
def make_dual_callback(reward_ref: TCCPLCosts,
                       adapt: bool = True) -> AdaptiveLagrangianCallback:
    """M1 when adapt=True; the paper's fixed-penalty ablation when
    adapt=False — Algorithm 1 with the multiplier update of line 15 frozen,
    while measurement, aux-signal annealing and everything else stay
    identical (Sec. III-D)."""
    return AdaptiveLagrangianCallback(
        reward_ref         = reward_ref,
        adapt              = adapt,
        initial_lambda_ov  = reward_ref.lambda_ov_kwh,
        initial_lambda_sat = reward_ref.lambda_sat,
        sat_floor          = CONFIG["train_sat_floor"],        # training floor
        target_overload    = CONFIG["train_target_overload"],  # above tightened limit
        ki                 = CONFIG["lambda_ki"],
        kp                 = CONFIG["lambda_kp"],
        lambda_max         = CONFIG["lambda_max"],
        update_freq        = CONFIG["lambda_update_freq"],
        aux_anneal_frac    = CONFIG["aux_anneal_frac"],
        verbose            = 1,
    )


# ====================================================================
# notebook cell 22
# ====================================================================
def collect_m2_demos(reward_fn: TCCPLCosts, verbose: bool = True):
    """M2 Phase 1 — zero-residual demonstrations from the incumbent, collected
    through the SAME residual interface and fixed feature scaling as training:
    C1 (reward consistency) and C2 (scaling consistency) hold by construction.

    The cache key deliberately EXCLUDES λ: the stored components are
    λ-independent and repriced at sample time, so TC-CPL and the frozen-λ
    ablation share one demonstration set (≈17 min saved per run)."""
    return collect_demonstrations(
        make_env=lambda: build_env(state_fn=CONFIG["state_function_safe"],
                                   reward_fn=reward_fn, wrap_residual=True),
        n_transitions=CONFIG["offline_transitions"],
        seed=SEED,
        cache_key=(f"{CONFIG['scenario']}|rho{CONFIG['rho']}|"
                   f"sat{CONFIG['train_sat_floor']}|m{CONFIG['ov_margin']}"),
        verbose=verbose,
    )


# ====================================================================
# notebook cell 24
# ====================================================================
def buffer_kwargs(lambda_ref: TCCPLCosts) -> dict:
    """TCCPLReplayBuffer (tccpl_learner.py): stores the reward components and
    recomposes r̃(λ) at every draw from the LIVE multipliers; prioritized
    sampling ∝ (|TD|+ε)^ω with normalized importance weights, β annealed to 1.
    Priorities are refreshed inside TCCPLSAC.train() from the TD errors the
    critic update computes anyway — no extra forward passes."""
    return dict(
        replay_buffer_class  = TCCPLReplayBuffer,
        replay_buffer_kwargs = dict(alpha=CONFIG["per_alpha"],
                                    eps=CONFIG["per_eps"],
                                    beta0=CONFIG["per_beta0"],
                                    lambda_ref=lambda_ref),
    )


print(f"Replay: PER ω={CONFIG['per_alpha']}, β {CONFIG['per_beta0']}→1, "
      "sample-time λ-composition, TD-refresh inside train().")


# ====================================================================
# notebook cell 26
# ====================================================================
# ── M3 learner sanity check (no environment needed) ──────────────────────────
# Verifies output shapes, the controllable-port masking, permutation
# equivariance of the actor / invariance of the critics, and gradient flow.
from tccpl_learner import test_shapes, TCCPLActor as _A

test_shapes()

_toy_s = {"n_ports": 8, "n_tr": 2, "tr_of_port": [0] * 4 + [1] * 4,
          "base_obs_dim": 4 + 8 + 24, "obs_dim": 4 + 8 + 24 + 8}
_toy_l = {"n_ports": 600, "n_tr": 122, "tr_of_port": [0] * 600,
          "base_obs_dim": 4 + 488 + 1800, "obs_dim": 4 + 488 + 1800 + 600}
_n_s = sum(p.numel() for p in _A(_toy_s, CONFIG["hidden_dim"]).parameters())
_n_l = sum(p.numel() for p in _A(_toy_l, CONFIG["hidden_dim"]).parameters())
assert _n_s == _n_l
print(f"Actor parameters at N=8 and N=600: {_n_s:,} == {_n_l:,} "
      "(size-independent, Sec. III-C)")
del _toy_s, _toy_l, _n_s, _n_l


# ====================================================================
# notebook cell 28
# ====================================================================
# ── Baseline encoder for CPO: a flat MLP with no structural prior ────────────
# The class is defined in tccpl_learner.py so that §6.1 and §7 can rebuild a
# CPO checkpoint in ANY kernel, even one in which this section was skipped.
from tccpl_learner import BaselineFeatureExtractor

_bfe = BaselineFeatureExtractor(gym.spaces.Box(-np.inf, np.inf, (12,), np.float32), 256)
print(f"BaselineFeatureExtractor ready: {sum(p.numel() for p in _bfe.parameters()):,} "
      f"parameters for a 12-dim probe input (flat MLP 256-128-256, orthogonal init)")
del _bfe


# ====================================================================
# notebook cell 32
# ====================================================================
# ── Site layout — DERIVED from a live env, never hard-coded ──────────────────
_probe_env = build_env(state_fn=CONFIG["state_function_safe"], seed=0)
LAYOUT = layout_from_env(_probe_env)
assert int(np.prod(_probe_env.observation_space.shape)) == LAYOUT["base_obs_dim"]
P_REF = float(sum(cs.get_max_power() for cs in _probe_env.charging_stations))
E_REF = _probe_env.simulation_length * _probe_env.timescale / 60.0 * P_REF
print(f"{CONFIG['scenario']}: ports={LAYOUT['n_ports']}  "
      f"transformers={LAYOUT['n_tr']}  base obs dim={LAYOUT['base_obs_dim']}  "
      f"learner obs dim={LAYOUT['obs_dim']} (+a⁰ anchor block)")
print(f"Fixed scales: P_ref={P_REF:,.0f} kW  E_ref={E_REF:,.0f} kWh  "
      f"λ-unit conversion E_ref/P_ref = {E_REF / P_REF:.1f}")
_probe_env.close()

tccpl_policy_kwargs = dict(layout=LAYOUT, hidden_dim=CONFIG["hidden_dim"],
                           use_bf16=CONFIG["use_bf16"])

# ── PublicPST-state geometry (DDPG-PST / PPO-PST) ────────────────────────────
_probe_paper = build_env(state_fn=CONFIG["state_function"], seed=0)
print(f"PublicPST state (DDPG/PPO): obs dim = "
      f"{int(np.prod(_probe_paper.observation_space.shape))}")
_probe_paper.close()
del _probe_env, _probe_paper

# ── CPO extractor: flat MLP on the grid-aware observation (no anchor) ────────
# Registered here so §7 can rebuild the network when loading a CPO checkpoint
# in a fresh kernel. NOT the M3 encoder — that is a TC-CPL contribution and is
# withheld from every baseline.
from cpo_pst import CPOPolicy as _CPOPolicy
from tccpl_learner import BaselineFeatureExtractor as _BaselineFeatureExtractor


def cpo_make_extractor():
    _space = gym.spaces.Box(low=-np.inf, high=np.inf,
                            shape=(LAYOUT["base_obs_dim"],), dtype=np.float32)
    return _BaselineFeatureExtractor(_space, features_dim=256), 256


_CPOPolicy.set_extractor_factory(cpo_make_extractor)
print("CPO extractor factory registered (flat MLP on the grid-aware observation).")


# ====================================================================
# training job: fixedpenalty  (notebook cells 34)
# ====================================================================
def train_fixedpenalty():
    # --- notebook cell 34 ---
    P = run_paths("fixedpenalty")
    os.makedirs(P["dir"], exist_ok=True)
    print("=" * 60)
    print(f"TRAINING: SAC (fixed λ) — Algorithm 1 with line 15 FROZEN at "
          f"λ_ov={CONFIG['lambda_ov_fixed']}, λ_sat={CONFIG['lambda_sat_fixed']} "
          f"— seed {SEED}")
    print("=" * 60)

    fixed_reward = make_costs(CONFIG["lambda_ov_fixed"], CONFIG["lambda_sat_fixed"])

    # M2 warm start — IDENTICAL to §6.3 (the demonstrations are λ-independent)
    offline_data = collect_m2_demos(fixed_reward)

    train_env_fixed = build_tccpl_vec_env(reward_fn=fixed_reward)
    fixed_model = TCCPLSAC(
        policy          = TCCPLPolicy,
        env             = train_env_fixed,
        learning_rate   = CONFIG["learning_rate"],
        buffer_size     = CONFIG["buffer_size"],
        learning_starts = 0,
        batch_size      = CONFIG["batch_size"],
        tau             = CONFIG["tau"],
        gamma           = CONFIG["gamma"],
        train_freq      = (1, "step"),
        gradient_steps  = -1,          # one update per environment transition
        ent_coef        = "auto",
        seed            = SEED,
        verbose         = 1,
        device          = CONFIG["device"],
        policy_kwargs   = tccpl_policy_kwargs,
        beta0           = CONFIG["per_beta0"],
        tensorboard_log = "./logs/tensorboard/",
        **buffer_kwargs(lambda_ref=fixed_reward),
    )
    prefill_buffer(fixed_model, offline_data)

    frozen_cb = make_dual_callback(fixed_reward, adapt=False)   # ← THE ablation
    eval_cb_fixed = EvalCallback(
        build_eval_env(state_fn=CONFIG["state_function_safe"], wrap_residual=True),
        best_model_save_path = P["dir"],
        log_path             = P["log"],
        eval_freq            = eval_every(CONFIG["total_timesteps_fixed"]),
        n_eval_episodes      = CONFIG["eval_episodes_train"],
        deterministic        = True,
        verbose              = 0,
    )
    fixed_model.learn(total_timesteps=CONFIG["total_timesteps_fixed"],
                      callback=[frozen_cb, eval_cb_fixed], progress_bar=False)
    fixed_model.save(P["zip"])
    with open(lambda_history_path("fixedpenalty"), "w") as f:
        json.dump({"lambda_history": frozen_cb.lambda_history,
                   "violation_history": frozen_cb.violation_history,
                   "update_freq": frozen_cb.update_freq, "seed": SEED}, f)
    print("SAC (fixed λ) training complete.")


# ====================================================================
# training job: tccpl  (notebook cells 37, 38)
# ====================================================================
def train_tccpl():
    # --- notebook cell 37 ---
    P = run_paths("tccpl")
    os.makedirs(P["dir"], exist_ok=True)
    print("=" * 60)
    print(f"TRAINING: TC-CPL (M1+M2+M3) — seed {SEED}")
    print("=" * 60)

    tccpl_reward = make_costs(CONFIG["lambda_init_ov"], CONFIG["lambda_init_sat"])

    # ── Phase 1 (M2): zero-residual demonstrations from the incumbent ────────────
    offline_data = collect_m2_demos(tccpl_reward)

    # ── Phase 2: model + prioritized prefill ─────────────────────────────────────
    train_env_tccpl = build_tccpl_vec_env(reward_fn=tccpl_reward)
    tccpl_model = TCCPLSAC(
        policy          = TCCPLPolicy,
        env             = train_env_tccpl,
        learning_rate   = CONFIG["learning_rate"],
        buffer_size     = CONFIG["buffer_size"],
        learning_starts = 0,           # learning starts from the demonstrations
        batch_size      = CONFIG["batch_size"],
        tau             = CONFIG["tau"],
        gamma           = CONFIG["gamma"],
        train_freq      = (1, "step"),
        gradient_steps  = -1,          # one update per environment transition
        ent_coef        = "auto",
        seed            = SEED,
        verbose         = 1,
        device          = CONFIG["device"],
        policy_kwargs   = tccpl_policy_kwargs,
        beta0           = CONFIG["per_beta0"],
        tensorboard_log = "./logs/tensorboard/",
        **buffer_kwargs(lambda_ref=tccpl_reward),
    )
    prefill_buffer(tccpl_model, offline_data)

    # ── C1/C2 verification ───────────────────────────────────────────────────────
    rb = tccpl_model.replay_buffer
    _n = rb.buffer_size if rb.full else max(rb.pos, 1)
    assert rb.has_comp[:_n].any(), "demonstrations lost their reward components"
    _anchor = rb.observations[0, 0, -LAYOUT["n_ports"]:]
    print(f"C1 OK — components stored, repriced by the live λ at every draw.")
    print(f"C2 OK — fixed feature scaling by construction; demo anchor block "
          f"mean |a⁰| = {np.abs(_anchor).mean():.3f}")

    # ── Phase 3: online learning with the PI dual controller (M1) ────────────────
    lagrangian_cb = make_dual_callback(tccpl_reward, adapt=True)
    eval_cb_tccpl = EvalCallback(
        build_eval_env(state_fn=CONFIG["state_function_safe"], wrap_residual=True),
        best_model_save_path = P["dir"],
        log_path             = P["log"],
        eval_freq            = eval_every(CONFIG["total_timesteps_tccpl"]),
        n_eval_episodes      = CONFIG["eval_episodes_train"],
        deterministic        = True,
        verbose              = 0,
    )
    tccpl_model.learn(total_timesteps=CONFIG["total_timesteps_tccpl"],
                      callback=[lagrangian_cb, eval_cb_tccpl], progress_bar=False)
    tccpl_model.save(P["zip"])
    print(f"TC-CPL training complete: λ updated {len(lagrangian_cb.lambda_history)} "
          f"times → λ_ov={lagrangian_cb.lambda_ov:.1f}, "
          f"λ_sat={lagrangian_cb.lambda_sat:.1f} (kWh-price units)")

    # --- notebook cell 38 ---
    with open(lambda_history_path("tccpl"), "w") as f:
        json.dump({
            "lambda_history":    lagrangian_cb.lambda_history,
            "violation_history": lagrangian_cb.violation_history,
            "update_freq":       lagrangian_cb.update_freq,
            "seed":              SEED,
        }, f)
    print(f"Lambda history saved → {lambda_history_path('tccpl')}")


# ====================================================================
# training job: ddpg  (notebook cells 41)
# ====================================================================
# imports hoisted out of the job body (see hoist_imports)
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
def train_ddpg():
    # --- notebook cell 41 ---
    # [hoisted to module scope] from stable_baselines3 import DDPG
    # [hoisted to module scope] from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

    P = run_paths("ddpg_pst")
    os.makedirs(P["dir"], exist_ok=True)
    print(f"Building DDPG-PST training environment (paper-faithful PublicPST state, seed {SEED})...")
    train_env_ddpg = build_baseline_vec_env(reward_fn=CONFIG["reward_function"],
                                            state_fn=CONFIG["state_function"])
    eval_env_ddpg  = build_eval_env(state_fn=CONFIG["state_function"], vecnormalize=True)

    n_actions = train_env_ddpg.action_space.shape[-1]
    ou_noise  = OrnsteinUhlenbeckActionNoise(
        mean  = np.zeros(n_actions),
        sigma = CONFIG["ddpg_ou_sigma"] * np.ones(n_actions),
    )

    eval_cb_ddpg = EvalCallback(
        eval_env_ddpg,
        best_model_save_path = P["dir"],
        log_path             = P["log"],
        eval_freq            = eval_every(CONFIG["total_timesteps_ddpg"]),   # vec-steps, not timesteps
        n_eval_episodes      = CONFIG["eval_episodes_train"],
        deterministic        = True,
        verbose              = 0,
    )

    # Plain flat-MLP policy and UNIFORM replay — exactly as published.
    ddpg_model = DDPG(
        policy          = "MlpPolicy",
        env             = train_env_ddpg,
        learning_rate   = CONFIG["ddpg_lr"],
        buffer_size     = CONFIG["ddpg_buffer_size"],
        learning_starts = 5_000,
        batch_size      = CONFIG["ddpg_batch_size"],
        tau             = CONFIG["ddpg_tau"],
        gamma           = CONFIG["ddpg_gamma"],
        train_freq      = (1, "step"),
        gradient_steps  = -1,   # Algorithm 1: one update per environment transition
        action_noise    = ou_noise,
        seed            = SEED,
        verbose         = 1,
        device          = CONFIG["device"],
        policy_kwargs   = dict(net_arch=dict(pi=CONFIG["ddpg_actor_arch"],
                                             qf=CONFIG["ddpg_critic_arch"])),
        tensorboard_log = "./logs/tensorboard/",
    )

    print("\n" + "="*60)
    print(f"TRAINING: DDPG-PST (literature baseline, Yilmaz et al. 2024) — seed {SEED}")
    print("="*60)
    ddpg_model.learn(
        total_timesteps = CONFIG["total_timesteps_ddpg"],
        callback        = eval_cb_ddpg,
        progress_bar    = False,
    )
    ddpg_model.save(P["zip"])
    train_env_ddpg.save(P["vecnorm"])
    print("DDPG-PST training complete.")


# ====================================================================
# training job: ppo  (notebook cells 44)
# ====================================================================
# imports hoisted out of the job body (see hoist_imports)
from stable_baselines3 import PPO
def train_ppo():
    # --- notebook cell 44 ---
    # [hoisted to module scope] from stable_baselines3 import PPO

    P = run_paths("ppo_pst")
    os.makedirs(P["dir"], exist_ok=True)
    print(f"Building PPO-PST training environment (PublicPST state, seed {SEED})...")
    train_env_ppo = build_baseline_vec_env(reward_fn=CONFIG["reward_function"],
                                           state_fn=CONFIG["state_function"])
    eval_env_ppo  = build_eval_env(state_fn=CONFIG["state_function"], vecnormalize=True)

    eval_cb_ppo = EvalCallback(
        eval_env_ppo,
        best_model_save_path = P["dir"],
        log_path             = P["log"],
        eval_freq            = eval_every(CONFIG["total_timesteps_ppo"]),   # vec-steps, not timesteps
        n_eval_episodes      = CONFIG["eval_episodes_train"],
        deterministic        = True,
        verbose              = 0,
    )

    ppo_model = PPO(
        policy          = "MlpPolicy",
        env             = train_env_ppo,
        learning_rate   = CONFIG["ppo_lr"],
        n_steps         = CONFIG["ppo_n_steps"],
        batch_size      = CONFIG["ppo_batch_size"],
        n_epochs        = CONFIG["ppo_n_epochs"],
        gamma           = CONFIG["ppo_gamma"],
        gae_lambda      = CONFIG["ppo_gae_lambda"],
        clip_range      = CONFIG["ppo_clip_range"],
        ent_coef        = 0.0,
        seed            = SEED,
        verbose         = 1,
        device          = CONFIG["device"],
        policy_kwargs   = dict(net_arch=dict(pi=CONFIG["ppo_arch"], vf=CONFIG["ppo_arch"])),
        tensorboard_log = "./logs/tensorboard/",
    )

    print("\n" + "="*60)
    print(f"TRAINING: PPO-PST (on-policy baseline) — seed {SEED}")
    print("="*60)
    ppo_model.learn(
        total_timesteps = CONFIG["total_timesteps_ppo"],
        callback        = eval_cb_ppo,
        progress_bar    = False,
    )
    ppo_model.save(P["zip"])
    train_env_ppo.save(P["vecnorm"])
    print("PPO-PST training complete.")


# ====================================================================
# training job: cpo  (notebook cells 47)
# ====================================================================
# imports hoisted out of the job body (see hoist_imports)
from cpo_pst import CPO
def train_cpo():
    # --- notebook cell 47 ---
    # [hoisted to module scope] from cpo_pst import CPO

    P = run_paths("cpo_pst")
    os.makedirs(P["dir"], exist_ok=True)

    # λ = 0 and no auxiliary signals: CPO's reward is the pure normalized tracking
    # objective r̄. TCCPLEnv exposes the TRUE overload cost as info["g1"]; it is
    # rescaled to per-step kWh (× E_ref) so CPO's dual arithmetic works on O(1)
    # numbers, with cost_limit = ε_ov_max = 0 unchanged by the rescaling.
    cpo_reward = make_costs(0.0, 0.0)
    cpo_reward.ov_margin = 0.0          # CPO constrains the TRUE overload, as published
    cpo_reward.aux_ov_coef_kwh = 0.0
    cpo_reward.aux_sat_coef = 0.0

    train_env_cpo = build_baseline_vec_env(reward_fn=cpo_reward,
                                           state_fn=CONFIG["state_function_safe"])

    cpo_agent = CPO(
        train_env_cpo,
        cpo_make_extractor,
        cost_limit = CONFIG["target_overload"],       # ε_ov_max = 0
        cost_fn    = lambda i: float(i.get("g1", 0.0)) * E_REF,   # per-step kWh
        n_steps    = CONFIG["cpo_n_steps"],           # per env (256 at the protocol)
        gamma      = 0.99,
        delta      = 0.01,                            # KL trust region
        device     = CONFIG["device"],
        verbose    = 1,
    )

    print("\n" + "=" * 60)
    print(f"TRAINING: CPO (trust-region CMDP baseline) — seed {SEED}")
    print("=" * 60)
    cpo_agent.learn(total_timesteps=CONFIG["total_timesteps_cpo"])
    cpo_agent.save(P["zip"])
    train_env_cpo.save(P["vecnorm"])

    with open(f"{RESULTS_DIR}/cpo_history{SUF}.json", "w") as f:
        json.dump({"history": cpo_agent.history, "seed": SEED}, f, indent=2)
    print(f"CPO training complete → {P['zip']}.zip")
    free_mem("train_env_cpo", "cpo_agent", "cpo_reward")



TRAIN_JOBS = {
    "fixedpenalty": train_fixedpenalty,
    "tccpl": train_tccpl,
    "ddpg": train_ddpg,
    "ppo": train_ppo,
    "cpo": train_cpo,
}
