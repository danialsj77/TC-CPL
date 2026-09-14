# TC-CPL — Transformer-Constrained Charging Policy Learning

Code and experiments for the manuscript **"Power Setpoint Tracking for
Large-Scale EV Charging: Transformer-Constrained Charging Policy Learning with
Self-Tuning Constraint Prices"**, submitted to IEEE Transactions on
Transportation Electrification, built on the
[EV2Gym](https://github.com/StavrosOrf/EV2Gym) simulator.

**Authors:** Daniyal Sajedi-Hosseini and Xun Shen (Tokyo University of
Agriculture and Technology), Xianbang Chen (Cornell University), Shubham Singh,
Lei Zhou and Katsuki Fujisawa (Institute of Science Tokyo), Sebastien Gros
(Norwegian University of Science and Technology), and Zhengmao Li (Aalto
University).

The code implements the manuscript's Sec. III **exactly**: the grid-aware
observation with fixed physical feature scaling, residual anchoring at the
incumbent's action (ρ = 0.25), the exact normalized costs with the terminal
service deficit, the PI dual controller (M1), prioritized replay with
importance weights, per-transformer pooled actor and twin critics with a
shared per-port residual head (M3), the dynamic entropy target and
finite-horizon γ = 1 returns.

---

## How to run the study

The study is **5 algorithms × 5 seeds × 2 scenarios = 50 independent training
jobs** of 1,000,000 environment steps each, followed by one evaluation per
scenario. Nothing needs to be edited inside the code: every knob is a command
line flag or an environment variable. Python **3.10 or 3.11** is required
(the pinned numpy/pandas have no wheels for 3.12+).

### 1. Install and rehearse (about 15 minutes)

```bash
git clone https://github.com/danialsj77/TC-CPL.git && cd TC-CPL
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tccpl_learner.py                # analytic self-test, prints "self-test OK"
SMOKE=1 ./runall.sh PublicPST          # all five learners at 300 steps, a few minutes
SMOKE=1 ./runall.sh cityNL             # the same on the city scenario, ~10 minutes
rm -rf models logs results             # discard the rehearsal artefacts
```

A smoke run writes real artefacts but marks them as smoke runs, so the
completion checks below never mistake them for finished models. Torch uses a
CUDA GPU automatically when one is visible; `TCCPL_DEVICE=cpu` forces CPU
(works, only slower).

### 2. Time one job before submitting fifty

One job is one process:

```bash
python train_one.py --algo tccpl --seed 42 --scenario cityNL --threads 8
```

Its log prints a progress table with `total_timesteps` and `fps`; the wall
time of one job is `1e6 / fps`. **There is no mid-run checkpoint**: a job
killed by a scheduler time limit restarts from zero, so the per-job time limit
must exceed that estimate. cityNL is the slow scenario (AC power flow every
step); PublicPST is roughly ten times faster. If the cluster's longest
allowed limit is shorter than one cityNL job, stop and ask before running
anything: the budget can be lowered for *all* jobs with
`TCCPL_TOTAL_TIMESTEPS=<steps>`, but that changes the reported protocol.

Memory per job is about 8 GB, except DDPG on cityNL, whose 1e6-transition
replay buffer needs about 18 GB (request 32 GB for that job, or for all).

### 3. Run the 50 jobs

Job number *i* (0–49) means: scenario `i // 25` (0 = cityNL, 1 = PublicPST),
seed `42 + (i % 25) // 5`, algorithm `i % 5` in the order fixedpenalty, tccpl,
ddpg, ppo, cpo. Pick whichever launcher fits the machine:

| Cluster | Command |
|---|---|
| **SLURM** | edit the `#SBATCH` header of `slurm_sweep.sh` (partition, GPU line, time limit) and its `source .venv/bin/activate` line, then `sbatch slurm_sweep.sh` (all 50) or `sbatch --array=0-24 slurm_sweep.sh` (cityNL only) |
| **Grid Engine / `qsub`** (e.g. TSUBAME) | edit the `#$` header of `tsubame_sweep.sh` (resource type, `h_rt`) and its activation line, then `qsub -g <group> tsubame_sweep.sh` (tasks 1–50) or `qsub -g <group> -t 1-25 tsubame_sweep.sh` (cityNL only) |
| **One big machine, no scheduler** | `nohup python run_sweep.py --jobs 4 > sweep.log 2>&1 &` (four jobs at a time, RAM permitting) |

All three launchers skip a job whose artefacts already exist at the full
budget, so after a failure or a time-out **resubmit the same command** and only
the missing jobs run. A job is finished when
`models/<scenario>/<name>_sac[_s<seed>].runmeta.json` records
`"timesteps": 1000000` and `"smoke_test": false`; `python run_sweep.py --dry-run`
lists what is done and what is missing without starting anything.

### 4. Send the artefacts back

```bash
tar czf tccpl_artefacts.tgz models logs results      # about 0.5–1 GB
```

That archive holds the checkpoints with their run metadata, the periodic
evaluation logs (learning curves) and the multiplier histories.

### 5. Evaluate

Unpack the archive into the repository, open `TC-CPL.ipynb` in Jupyter, answer
`cityNL` (or `PublicPST`) at the prompt of §2, and run **§7** (skip §6, which
would retrain). §7 finds every seed on disk automatically, evaluates each
controller on the same 100 paired days, averages the per-episode metrics over
seeds, draws the learning curves and multiplier trajectories with an
across-seed band, prints the per-seed final multipliers with a convergence
trend, and writes every figure and table of the manuscript to
`results/<scenario>/`. Do not run the whole notebook headlessly for
evaluation: that executes §6 and retrains everything.

Environment variables understood by every entry point:

| Variable | Default | Meaning |
|---|---|---|
| `TCCPL_SCENARIO` | prompt in Jupyter, `cityNL` headless | scenario |
| `TCCPL_SEED` | `42` | seed of this run (42 is unsuffixed: `tccpl_sac.zip`; others `tccpl_sac_s43.zip`, …) |
| `TCCPL_SEEDS` | `42,43,44,45,46` | the protocol's seed list (42 must stay first) |
| `TCCPL_TOTAL_TIMESTEPS` | `1000000` | training budget of every learner; also scales rollout and evaluation sizes for smoke runs |
| `TCCPL_DEVICE` | `auto` (cuda → mps → cpu) | training device |
| `TCCPL_DDPG_BUFFER` | `min(1e6, budget)` | DDPG replay size (set `300000` on a laptop) |
| `TCCPL_OV_MARGIN`, `TCCPL_TRAIN_SAT_FLOOR` | `0.05`, `0.98` | constraint-tightening knobs (see below) |
| `TCCPL_OFFLINE_TRANSITIONS`, `TCCPL_EVAL_EPISODES`, `TCCPL_N_OPT` | `100000`, `100`, `100` | smoke-test knobs only |

---

## Layout

| Path | Role |
|---|---|
| `TC-CPL.ipynb` | The end-to-end pipeline: mechanisms (§5), training of all learners (§6), evaluation, statistics and every figure/table of the manuscript (§7–8). |
| `tccpl_learner.py` | **The TC-CPL learner stack** — observation, residual wrapper, costs, PI dual callback, demonstrations, prioritized buffer, structured actor/critics, `TCCPLSAC`. Run `python tccpl_learner.py` for a self-test (shapes, masking, permutation equivariance, gradients). |
| `cpo_pst.py` | Constrained Policy Optimization baseline (Achiam et al., 2017). |
| `export_training_module.py` | Regenerates `tccpl_runtime.py` from the notebook. **Re-run after editing notebook cells 4–47.** |
| `tccpl_runtime.py` | AUTO-GENERATED — notebook definitions + one `train_<algo>()` per job, importable outside Jupyter. Do not edit by hand. |
| `train_one.py` | One controller × one seed × one scenario as a standalone process. |
| `runall.sh` | All five algorithms of one scenario at the same time (one process each): `./runall.sh cityNL 42`; `SMOKE=1` rehearses everything; finished jobs are skipped on re-run. |
| `run_sweep.py` | The 50-job protocol with bounded concurrency on one machine. |
| `slurm_sweep.sh`, `tsubame_sweep.sh` | The 50-job protocol as a SLURM array / a Grid-Engine (`qsub`) array, one `train_one.py` per task. |
| `ev2gym/` | The EV2Gym simulator package (environment, Dutch field data, heuristics, plotting). Kept intact. |
| `models/<scenario>/`, `logs/<scenario>/`, `results/<scenario>/`, `cache/` | Created by runs — everything is namespaced by scenario so the two studies never collide. |

Regenerable and therefore not in git: `models/` (checkpoints), `cache/` (M2
demonstration buffers, rebuilt on the first run), `results/**/*.pkl`, `*.npz`,
`*.mp4` (day captures and the rollout video) and `logs/`.

## The two validation scenarios

When the notebook starts in Jupyter, its §2 cell asks which scenario to run;
headless launchers select it with `TCCPL_SCENARIO` and are never prompted.

- **`PublicPST`** — the single-transformer public facility of the PST paper:
  20 charge points behind one 100-kW transformer, 112 × 15-min steps, no grid
  simulation. Fast; the small-scale study.
- **`cityNL`** — the city-scale scenario: 122 MV/LV substations on a real
  feeder model, 300 dual-socket stations (600 ports), full AC power flow,
  96 × 15-min steps. The headline study.

Controllers: TC-CPL (M1+M2+M3) · SAC (fixed λ) — Algorithm 1 with the
multiplier update frozen, everything else identical, so the pair isolates M1 ·
DDPG-PST and PPO-PST on the published `PublicPST` state · CPO on the grid-aware
observation with a flat MLP · CAFAP/ALAP/Round-Robin heuristics · the offline
convex-QP perfect-foresight benchmark.

## What §7 writes

Every table and figure goes to `results/<scenario>/` as PNG + CSV/JSON. §7.4
also records the overload beyond dead bands of 0.5–20 kW, 1–5 % of nameplate
and the nameplate rating per day; both §7.9 figures report the 10-kW band. The
design-region figure (§7.9) is one row of four panels per scenario —
(a) $|\epsilon_{tr}|$ vs. overload, (b) vs. user satisfaction, (c) vs.
delivered energy, (d) a compact legend — with every controller drawn and a
bold red frame marking the magnified view of the desired region.

## Constraint tightening (requirements vs. training thresholds)

Problem 1's **requirements** — the quantities §7 evaluates and the paper reports —
are `target_overload = 0` kWh and `target_sat = 0.95`. The learner is trained
against strictly **tighter surrogates** (a standard back-off): the dual-priced
overload counts loading above `(1 − ov_margin) ×` the DR-aware transformer limit
(EV-controllable part; `ov_margin = 0.05`), and the PI dual controller targets
`train_sat_floor = 0.98`. Both are overridable per run with `TCCPL_OV_MARGIN`
and `TCCPL_TRAIN_SAT_FLOOR`.

Why tighten the *limit* and not the *threshold*: a negative energy threshold
(e.g. "overload ≤ −3 kWh") is infeasible because overload energy is
non-negative, so the integral term would ratchet λ_ov to λ_max regardless of
the policy and the Lagrangian analysis would no longer hold. Tightening the
limit gives the same pressure toward zero while keeping the constraint
satisfiable, so λ can settle once loading stays under the tightened limit — at
which point true overload is exactly zero.

## Multiplier units (read before re-tuning)

The paper prices the normalized cost c_ov = kWh/E_ref. The PI controller adapts
λ_ov in interpretable **kWh-price units** (`lambda_init_ov`, `lambda_ki`, … in
CONFIG), and `TCCPLCosts` converts once by the exact factor
E_ref/P_ref = T·Δt (24 on cityNL, 28 on PublicPST). Fixed positive rescaling
changes neither the feasible set nor the optimizer (paper Sec. III-B). The two
auxiliary-signal coefficients (`aux_ov_coef`, `aux_sat_coef`) use the same
convention and are annealed to zero over the first 80 % of training —
identically for TC-CPL and its ablation.

## Device and speed notes

- Stable-Baselines3's `device="auto"` only ever checks CUDA and silently falls
  back to CPU on a Mac. `tccpl_learner.resolve_device()` prefers
  **cuda → mps (Apple GPU) → cpu**; override with `TCCPL_DEVICE`. Measured on a
  MacBook Pro M5 Pro, one TC-CPL gradient step at cityNL scale takes 558 ms on
  CPU and 128 ms on mps. The bf16 encoder autocast is CUDA-only.
- TF32 matmuls + bf16 autocast inside the M3 encoders (Ampere+ GPUs); PER
  priorities are cached in exponentiated form and refreshed from the TD errors
  the critic update computes anyway.
- The remaining wall-clock cost is structural: one gradient step per
  environment transition (protocol parity) and serial `DummyVecEnv` stepping.
  `SubprocVecEnv` is not used because M1's dual update reaches every worker by
  mutating one shared reward object, which requires in-process envs;
  parallelism across *runs* comes from the launchers instead.

## Verified end to end (2026-09-14)

In a fresh Python 3.11 environment built only from `requirements.txt`
(torch 2.14, Stable-Baselines3 2.9, numpy 1.24, cvxpy 1.7): the learner
self-test; `SMOKE=1 ./runall.sh` on both scenarios (all five learners through
`train_one.py`, run metadata written, finished jobs skipped on re-run);
`slurm_sweep.sh` and `tsubame_sweep.sh` executed locally as single array tasks;
and the **whole notebook** executed headlessly on `PublicPST` with a smoke
budget and two seeds present — all 46 code cells, every figure, table, test
and the rollout video, with §7 aggregating over the seeds it finds. The
`cityNL` results in the manuscript were produced by the same notebook.

## License and attribution

This work is released under the MIT License (see `LICENSE`). The `ev2gym/`
package is a fork of [EV2Gym](https://github.com/StavrosOrf/EV2Gym) by Stavros
Orfanoudakis et al., used and redistributed under the same license; the
`cityNL` scenario, the grid-aware observation, the TC-CPL learner and every
file listed in the layout table above are additions of this work. If you use
this code, please cite both the EV2Gym paper and the TC-CPL manuscript.
