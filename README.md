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

### Step 2 — Run the whole campaign (all jobs at once)

The study is two array jobs, submitted together, from the repository folder:

| Array | Jobs | What one job does | Resources | Time per job |
|---|---|---|---|---|
| `tsubame_sweep.sh` / `slurm_sweep.sh` | 50 = 5 algorithms × 5 seeds × 2 scenarios | trains one controller for 1,000,000 steps | 1 GPU, 8 cores, 32 GB | up to 30 h (cityNL TC-CPL), most far less |
| `revision/stress_sweep.sh` | 66 PublicPST-S experiments | trains one variant for 200,000 steps and evaluates it | 8 cores, no GPU, 16 GB | 1–2 h |

The jobs are independent, so **submit both arrays at the same time**. Use a fresh clone in a
new folder for this campaign. The scripts would retrain an old folder anyway (only runs of
this code version count as finished), but a fresh folder keeps the archive clean.

```bash
mkdir -p logs/qsub                                     # Grid Engine (TSUBAME): the log folder must exist first
qsub -g <your-group> tsubame_sweep.sh                  # main study, 50 tasks (GPU)
qsub -g <your-group> revision/stress_sweep.sh          # PublicPST-S, 66 tasks (CPU)
```

```bash
mkdir -p logs/slurm                                    # SLURM
sbatch slurm_sweep.sh
sbatch revision/stress_sweep.sh
```

```bash
python run_sweep.py --jobs 4                           # ONE machine instead of the cluster: all 50 jobs, 4 running at a time
```

What the cluster lines mean:

- `sbatch` (SLURM) and `qsub` (Grid Engine) hand the script to the cluster's
  scheduler, which runs it on compute nodes when resources are free, instead
  of on the login machine you are typing on.
- `-g <your-group>` is required on TSUBAME: it names the group (lab or project
  account, e.g. `tga-xxxx`) whose computing budget pays for the jobs. Replace
  the placeholder with your own group name; `t4-user-info group list` shows
  it. Without `-g` the job runs in trial mode, which is limited to 10 minutes.
- Each script is an *array job*: it asks the scheduler for many tasks and each
  task runs one job.

Before submitting, open both scripts once and check the lines at the top: the
resource type (`gpu_1` = one GPU for the main study, `cpu_8` = 8 cores for
PublicPST-S; use the names of your site), the run-time limit (`h_rt` on TSUBAME,
`--time` on SLURM; 48 h and 8 h are requested) and the line that activates `.venv`.
A job cannot be paused and resumed, so do not lower the limits below the times above.
To see what a task runs, without running anything:

```bash
TCCPL_DRY=1 SGE_TASK_ID=7 bash tsubame_sweep.sh             # or SLURM_ARRAY_TASK_ID=6 bash slurm_sweep.sh
TCCPL_DRY=1 SGE_TASK_ID=3 bash revision/stress_sweep.sh
```

If some jobs fail or run out of time, **submit the same array again**: finished
jobs exit at once and only the missing ones run (`python run_sweep.py --dry-run`
lists the state of the main study).

Memory: about 8 GB per main job, except DDPG on cityNL, which needs about 18 GB.

### Step 3 — Check and send the results back

```bash
python check_cluster_results.py        # ok / FAIL for each of the 50 + 66 jobs; exit code 0 = complete
tar czf tccpl_results.tgz models logs results revision/runs/PublicPST_stress
```

Send the archive (about 1 GB) only when the check passes; if it lists failures,
submit the arrays again first.

### Step 4 — Evaluate and build the figures and tables (Daniyal's side)

1. Move the previous results aside, then unpack: `tar xzf tccpl_results.tgz`, and
   run `python check_cluster_results.py` once more.
2. Open `TC-CPL.ipynb`, run it for `cityNL` and then for `PublicPST`, **Section 7
   only** (Section 6 trains again). It evaluates every seed on the same 100 days and
   writes the figures and per-day results to `results/<scenario>/`.
3. The re-evaluations with per-interval records, the headroom projection, the
   observation errors and the held-out days (150 jobs, a few hours):
   `cd revision && python scripts/campaign.py jobs.json 4`
4. Statistics, tables and Fig. 6, from `revision/`:

```bash
python scripts/analyze_submitted.py && python scripts/collect_results.py && python scripts/stress_stats.py
python scripts/heldout_stats.py && python scripts/make_tables.py && python scripts/fig_safety.py
```

Every step uses only results made from the checkpoints now in `models/`; outputs of an
earlier model are re-made, not reused.

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
| `TCCPL_PROTOCOL` | `2026-10` | campaign stamp written into every run's sidecar; only runs carrying it count as finished |

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
| `revision/stress_sweep.sh` | The 66 PublicPST-S experiments as one array (Grid Engine or SLURM); tasks in `revision/stress_tasks.txt`. |
| `check_cluster_results.py` | Checks that every job of the campaign finished with the current settings. |
| `export_training_module.py` | Rebuilds `tccpl_runtime.py` (auto-generated, do not edit) from the notebook. Run it after editing notebook cells 4–47. |
| `ev2gym/` | The simulator and its data. |
| `models/`, `logs/`, `results/`, `cache/` | Created by runs; not in git. |

## The two scenarios

- **PublicPST** — 20 charge points behind one 100-kW transformer, no grid
  simulation. Fast.
- **cityNL** — a Dutch city feeder: 122 substations, 300 stations (600 ports),
  full AC power flow. About ten times slower. The main result of the paper.

Controllers compared: Self-tuning TC-CPL, Fixed-price TC-CPL (the same learner
with frozen multipliers, the ablation), DDPG-PST, PPO-PST, CPO, the CAFAP and
Round-Robin heuristics, and the offline optimum with perfect foresight.

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

### Verified (2026-10-20, campaign version)

On a copy of the repository as a fresh clone has it (including the earlier
PublicPST-S records in git), with small budgets: `slurm_sweep.sh` for all five
algorithms on PublicPST and `tsubame_sweep.sh` for both TC-CPL learners on cityNL
(sidecars carry the campaign stamp, radius 1 and load-aware networks; a resubmitted
finished task is skipped, a first-submission model is retrained);
`revision/stress_sweep.sh` for seven tasks covering every code path (the
self-tuning learner, no anchoring, no grid block, flat encoder, radius 0.25, no
load-aware pooling, a heuristic, the headroom projection); `eval_citynl.py` on
the new checkpoints of all five learners (radius read from the checkpoint, outputs
re-made when the checkpoint changes); `check_cluster_results.py`; the actor and
critic under bf16 autocast; and Section 7 of the notebook on new checkpoints. The
first protocol (2026-09-14) was verified the same way and produced the
first-submission results.

</details>

## License

MIT (see `LICENSE`). `ev2gym/` is a fork of
[EV2Gym](https://github.com/StavrosOrf/EV2Gym) by Stavros Orfanoudakis et al.,
under the same license. If you use this code, please cite both the EV2Gym
paper and the TC-CPL manuscript.
