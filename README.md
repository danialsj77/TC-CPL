# TC-CPL — Transformer-Constrained Charging Policy Learning

Code for the manuscript **"Power Setpoint Tracking for Large-Scale EV
Charging: Transformer-Constrained Charging Policy Learning with Self-Tuning
Constraint Prices"**, submitted to *IEEE Transactions on Transportation
Electrification*. Built on the [EV2Gym](https://github.com/StavrosOrf/EV2Gym)
simulator.

**Authors:** Daniyal Sajedi-Hosseini and Xun Shen (Tokyo University of
Agriculture and Technology), Xianbang Chen (Cornell University), Shubham Singh,
Lei Zhou and Katsuki Fujisawa (Institute of Science Tokyo), Sebastien Gros
(Norwegian University of Science and Technology), and Zhengmao Li (Aalto
University).

---

## How to run it

You need **Python 3.10 or 3.11**, about 10 GB of disk, and optionally a GPU.
Every command below is run from the repository folder.

### Step 1 — Install (5 minutes)

```bash
git clone https://github.com/danialsj77/TC-CPL.git
cd TC-CPL
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2 — Test that it works (5 minutes)

```bash
python tccpl_learner.py          # prints "self-test OK"
SMOKE=1 ./runall.sh PublicPST    # trains all five learners for 300 steps, ~3 min
rm -rf models logs results       # delete the test output
```

### Step 3 — Measure one job (optional, but do it on a cluster)

The study is 50 jobs. One job trains one algorithm, with one seed, on one
scenario, for 1,000,000 steps. Run one job by hand to see how long it takes:

```bash
python train_one.py --algo tccpl --seed 42 --scenario cityNL
```

Its log prints `fps`. Hours per job = 1,000,000 / fps / 3600. A job cannot be
paused and resumed, so the cluster's time limit per job must be longer than
that. If it is not, stop here and tell Daniyal.

### Step 4 — Run all 50 jobs

Pick one line, depending on the machine:

```bash
sbatch slurm_sweep.sh                   # SLURM cluster
qsub -g <your-group> tsubame_sweep.sh   # Grid Engine cluster (TSUBAME)
python run_sweep.py --jobs 4            # one big machine, 4 jobs at a time
```

On a cluster, first open the script and edit the lines at the top (queue or
resource type, GPU, time limit, and the line that activates `.venv`).

If some jobs fail or run out of time, **run the same line again**: finished
jobs are skipped and only the missing ones run. To see what is finished:

```bash
python run_sweep.py --dry-run
```

Memory: about 8 GB per job, except DDPG on cityNL, which needs about 18 GB.

### Step 5 — Send the results back

```bash
tar czf tccpl_results.tgz models logs results
```

The archive is about 0.5–1 GB.

### Step 6 — Evaluate (Daniyal's side)

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
