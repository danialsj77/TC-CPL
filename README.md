# TC-CPL — Transformer-Constrained Charging Policy Learning

<p align="center">
  <img src="figs/policy_simulation_preview.gif" width="100%"
       alt="Nine charging controllers following the same power setpoint over one simulated day on the cityNL scenario">
  <br>
  <em>One simulated day on the cityNL scenario (122 substations, 300 stations,
  600 charge points): nine charging controllers, TC-CPL among them, follow the
  same power setpoint side by side.</em>
  <br>
  <a href="https://github.com/danialsj77/TC-CPL/raw/main/figs/policy_simulation.mp4">Download the full-quality video (MP4, 9 MB)</a>
</p>

Code for the manuscript **"Power Setpoint Tracking for Large-Scale EV
Charging: Transformer-Constrained Charging Policy Learning with Self-Tuning
Constraint Prices"**, submitted to *IEEE Transactions on Transportation
Electrification*. Built on the [EV2Gym](https://github.com/StavrosOrf/EV2Gym)
simulator.

**Authors:** Daniyal Sajedi-Hosseini, Yoshihiko Fujisawa, Shubham Singh, Lei
Zhou, Katsuki Fujisawa, Sebastien Gros, Mazaher Karimi, and Xun Shen.

**Affiliations:** Daniyal Sajedi-Hosseini and Xun Shen, Tokyo University of
Agriculture and Technology; Yoshihiko Fujisawa, Shubham Singh, Lei Zhou and
Katsuki Fujisawa, Institute of Science Tokyo; Sebastien Gros, Norwegian
University of Science and Technology; Mazaher Karimi, University of Vaasa.

---

## How to run it

You need **Python 3.10 or 3.11** (3.12 and newer do not work with the pinned
libraries), about 10 GB of disk, and optionally a GPU. Every command below is
run from the repository folder.

### Step 1 — Install (5 minutes)

Linux / Mac:

```bash
git clone https://github.com/danialsj77/TC-CPL.git
cd TC-CPL
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell or Command Prompt):

```bat
git clone https://github.com/danialsj77/TC-CPL.git
cd TC-CPL
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If `python3.11` / `py -3.11` is not found, install Python 3.11 from
python.org (Mac: `brew install python@3.11`). To check that everything is in
place, run `python tccpl_learner.py`; it must print `self-test OK`.

### Step 2 — Run all 50 jobs

The study is 50 jobs: one job trains one algorithm, with one seed, on one
scenario, for 1,000,000 steps. Pick one line, depending on the machine:

```bash
sbatch slurm_sweep.sh                   # SLURM cluster
qsub -g <your-group> tsubame_sweep.sh   # Grid Engine cluster (TSUBAME)
python run_sweep.py --jobs 4            # one machine (Linux, Mac or Windows), 4 jobs at a time
```

What the cluster lines mean:

- `sbatch` (SLURM) and `qsub` (Grid Engine) hand the script to the cluster's
  scheduler, which runs it on compute nodes when resources are free, instead
  of on the login machine you are typing on.
- `-g <your-group>` is required on TSUBAME: it names the group (lab or project
  account, e.g. `tga-xxxx`) whose computing budget pays for the jobs. Replace
  the placeholder with your own group name; `t4-user-info group list` shows
  it. Without `-g` the job runs in trial mode, which is limited to 10 minutes.
- The script itself is an *array job*: it asks the scheduler for 50 tasks and
  each task runs one `train_one.py` job.

Before submitting, open the script once and check the lines at the top:
the resource type (`gpu_1` = one GPU; a CPU-only type works, just slower),
the run-time limit per job (`h_rt` on TSUBAME, `--time` on SLURM) and the
line that activates `.venv`.

If some jobs fail or run out of time, **run the same line again**: finished
jobs are skipped and only the missing ones run. To see what is finished:

```bash
python run_sweep.py --dry-run
```

Memory: about 8 GB per job, except DDPG on cityNL, which needs about 18 GB.
A job cannot be paused and resumed, so give each job a long enough time limit.

### Step 2b — The revision run (October 2026): 20 jobs

For the revision only the two TC-CPL learners are trained again (the DDPG, PPO and
CPO checkpoints do not change): **Self-tuning TC-CPL** (`tccpl`) and
**Fixed-price TC-CPL** (`fixedpenalty`), on both scenarios, five seeds each, so
2 x 5 x 2 = 20 jobs. They are independent, so **submit all 20 at once**. The defaults
of this version are the settings of the study (residual radius 1, load-aware
networks, demonstration days disjoint from the evaluation days, overload tolerance 0):

```bash
TCCPL_ALGOS=fixedpenalty:tccpl sbatch --array=0-19 slurm_sweep.sh                       # SLURM
qsub -g <your-group> -t 1-20 -v TCCPL_ALGOS=fixedpenalty:tccpl tsubame_sweep.sh         # Grid Engine
python run_sweep.py --algos fixedpenalty tccpl --jobs 4                                 # one machine
```

A cityNL job took 21-30 hours on one GPU in the first study (PublicPST 8-21 hours), and a
job cannot be paused, so `tsubame_sweep.sh` requests 48 hours; lower it only if the site caps
it and a job fits. To see which task runs what, without running anything:
`TCCPL_DRY=1 SGE_TASK_ID=7 bash tsubame_sweep.sh` (SLURM: `TCCPL_DRY=1 SLURM_ARRAY_TASK_ID=6 bash slurm_sweep.sh`).

When the jobs are done, check them before sending anything:

```bash
python check_cluster_results.py        # prints ok / FAIL per run; exit code 0 = all 20 fine
```

### Step 3 — Send the results back

```bash
tar czf tccpl_results.tgz models logs results
```

The archive is about 0.5-1 GB (about 0.3 GB after the revision run, which trains only 20 models).

### Step 4 — Evaluate (Daniyal's side)

Unpack the archive into the repository, open `TC-CPL.ipynb` in Jupyter, type
`cityNL` or `PublicPST` when asked, and run **Section 7** only. It evaluates
every seed it finds on the same 100 days and writes every figure and table of
the paper to `results/<scenario>/`. Do not run Section 6: it trains again.

---

## Settings

All settings are environment variables; the defaults are the paper's protocol.

| Variable | Default | What it does |
|---|---|---|
| `TCCPL_SCENARIO` | asks in Jupyter; `cityNL` in scripts | which scenario |
| `TCCPL_SEED` | `42` | seed of this run |
| `TCCPL_SEEDS` | `42,43,44,45,46` | the five seeds of the study (42 must stay first) |
| `TCCPL_TOTAL_TIMESTEPS` | `1000000` | training steps per job |
| `TCCPL_DEVICE` | `auto` | `cuda`, `mps` or `cpu` |
| `TCCPL_DDPG_BUFFER` | `min(1e6, budget)` | DDPG replay size; use `300000` on a laptop |
| `TCCPL_OV_MARGIN`, `TCCPL_TRAIN_SAT_FLOOR` | `0.05`, `0.98` | training-time constraint tightening |
| `TCCPL_TR_LOAD_AWARE` | `1` | transformer embeddings see how many of their ports are occupied, and port embeddings the incumbent's command (`0` = networks of the first submission) |
| `TCCPL_TRAIN_TARGET_OVERLOAD` | `0` | overload tolerance of the dual controller, kWh per episode above the tightened limit; runs with different values write to the same model files, so use one copy of the repository per value |
| `TCCPL_DEMO_SEED_OFFSET` | `100000` | demonstration days start at simulator seed `run seed + offset`, disjoint from the evaluation (500-599) and held-out (5000-5099) days (`0` = first submission) |
| `TCCPL_RHO` | `1` | residual radius of the anchored projection; `1` lets every port cover its full range, `0.25` = first submission |
| `TCCPL_EVAL_SEED_OFFSET` | `500` | first evaluation day; `5000` = the 100 held-out days (demonstrations use seeds from 100000) |
| `TCCPL_EVAL_SHIELD` | `0` | `1` evaluates with the headroom projection at dispatch, no retraining |
| `TCCPL_KD` | `0` | derivative gain of the dual controller (`0` = the PI law of the paper) |
| `TCCPL_ALGOS` | all five | cluster arrays only: e.g. `fixedpenalty:tccpl` runs just the two TC-CPL learners (20 tasks) |

Smoke-test only: `TCCPL_OFFLINE_TRANSITIONS`, `TCCPL_EVAL_EPISODES`, `TCCPL_N_OPT`.

## Files

| File | What it is |
|---|---|
| `TC-CPL.ipynb` | The whole study: method (§5), training (§6), evaluation and every figure/table (§7–8). |
| `tccpl_learner.py` | The TC-CPL learner (observation, costs, dual controller, replay, networks). `python tccpl_learner.py` runs its self-test. |
| `cpo_pst.py` | The CPO baseline. |
| `train_one.py` | Trains one algorithm × one seed × one scenario. |
| `runall.sh` | Trains all five algorithms of one scenario at once. `SMOKE=1` for a quick test. |
| `run_sweep.py`, `slurm_sweep.sh`, `tsubame_sweep.sh` | The 50-job study on one machine, on SLURM, or on Grid Engine. |
| `export_training_module.py` | Rebuilds `tccpl_runtime.py` (auto-generated, do not edit) from the notebook. Run it after editing notebook cells 4–47. |
| `ev2gym/` | The simulator and its data. |
| `models/`, `logs/`, `results/`, `cache/` | Created by runs; not in git. |

## The two scenarios

- **PublicPST** — 20 charge points behind one 100-kW transformer, no grid
  simulation. Fast.
- **cityNL** — a Dutch city feeder: 122 substations, 300 stations (600 ports),
  full AC power flow. About ten times slower. The main result of the paper.

Controllers compared: TC-CPL, SAC with fixed multipliers (the ablation),
DDPG-PST, PPO-PST, CPO, the CAFAP and Round-Robin heuristics, and the offline
optimum with perfect foresight.

<details>
<summary><b>Technical notes</b> (click to open)</summary>

### Constraint tightening

The paper's requirements are overload = 0 kWh and satisfaction ≥ 0.95, and
those are what §7 reports. Training uses tighter targets: overload is priced
above `(1 − ov_margin)` × the transformer limit (`ov_margin = 0.05`) and the
dual controller aims at satisfaction ≥ 0.98. The limit is tightened rather
than the threshold because a negative overload threshold can never be met, so
the multiplier would only grow; a tightened limit can be met, so the
multiplier can settle.

### Multiplier units

The dual controller works in kWh-price units (`lambda_init_ov`, `lambda_ki`,
… in the notebook's CONFIG) and converts once to the paper's normalized cost
by the factor `E_ref / P_ref = T·Δt`. The two auxiliary shaping signals are
annealed to zero over the first 80 % of training, identically for TC-CPL and
its ablation.

### Device and speed

Torch uses `cuda` when available, then `mps` (Apple GPU), then `cpu`. One
gradient step per environment step and serial environment stepping are part
of the protocol, so the wall time is dominated by the simulator; parallelism
comes from running jobs side by side.

### Verified (2026-09-14)

In a fresh Python 3.11 environment from `requirements.txt`: the self-test;
`SMOKE=1 ./runall.sh` on both scenarios; `slurm_sweep.sh` and
`tsubame_sweep.sh` as single tasks; and the whole notebook, headless, on
PublicPST with two seeds present (all cells, every figure, table, test and
the video). The cityNL results in the manuscript were produced by the same
notebook.

</details>

## License

MIT (see `LICENSE`). `ev2gym/` is a fork of
[EV2Gym](https://github.com/StavrosOrf/EV2Gym) by Stavros Orfanoudakis et al.,
under the same license. If you use this code, please cite both the EV2Gym
paper and the TC-CPL manuscript.
