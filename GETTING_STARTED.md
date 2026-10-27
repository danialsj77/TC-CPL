# Getting started with TC-CPL on one computer

This page is for anyone who wants to run the code on an ordinary computer: a laptop, a
workstation or a machine with one GPU, under Windows, macOS or Linux. No cluster is needed.
Reproducing every number of the paper (116 training jobs) is a separate procedure, described
in [RUN_ON_SUPERCOMPUTER.md](RUN_ON_SUPERCOMPUTER.md).

1. [What the code does](#1-what-the-code-does)
2. [What you need](#2-what-you-need)
3. [Install](#3-install-10-minutes)
4. [First run: a smoke test](#4-first-run-a-smoke-test-2-minutes)
5. [Train a controller](#5-train-a-controller)
6. [Evaluate and draw the tables and figures](#6-evaluate-and-draw-the-tables-and-figures)
7. [What the result files mean](#7-what-the-result-files-mean)
8. [The variant experiments](#8-the-variant-experiments-optional)
9. [Changing settings](#9-changing-settings)
10. [Problems and fixes](#10-problems-and-fixes)
11. [Citing](#11-citing)

## 1. What the code does

A charging site has many charge points behind one or more transformers. A controller decides,
every 15 minutes, how much power each charge point delivers. It must follow a power setpoint
from the grid operator, give the drivers the energy they asked for, and never overload a
transformer.

The repository contains:

- **The simulator**: a fork of [EV2Gym](https://github.com/StavrosOrf/EV2Gym) (folder
  `ev2gym/`) with EV arrivals drawn from Dutch charging data, household load, PV,
  demand-response events and, for the city scenario, a full AC power flow.
- **TC-CPL**, the controller of the paper (`tccpl_learner.py`): a reinforcement-learning
  policy whose constraint prices tune themselves during training, so nothing has to be
  hand-tuned per site. Its three mechanisms are M1 (self-tuning constraint prices),
  M2 (warm start from an incumbent controller and residual anchoring) and M3 (a structured
  encoder that pools charge points per transformer).
- **The baselines** it is compared with: Fixed-price TC-CPL (the same learner with frozen
  prices), DDPG-PST, PPO-PST, CPO, the CAFAP and Round-Robin heuristics, and an offline
  optimum with perfect foresight.
- **Two scenarios**: PublicPST-S (20 charge points behind one 42-kW transformer: small and
  fast) and cityNL (a Dutch city feeder, 122 substations, 600 charge points: the main result,
  about ten times slower).
- **The evaluation outputs of the paper**, in `results/` and `revision/runs/`, so you can
  look at the numbers without training anything (Section 7).

Three kinds of runs are possible: a smoke test (seconds), training one controller (hours) and
evaluating trained controllers (minutes to hours). Each is one command or one notebook section.

## 2. What you need

| | |
|---|---|
| Python | **3.10 or 3.11**. 3.12 and newer do not work: the pinned numpy and pandas have no wheels for them. |
| Operating system | Linux, macOS or Windows. The `.sh` scripts need bash (Linux, macOS, or WSL / Git Bash on Windows); every step below also has a plain `python` command that works everywhere. |
| Disk | about 10 GB: packages, the demonstration cache, checkpoints and logs. |
| Memory | 8 GB free for one training job (DDPG on cityNL: 18 GB); 3 GB for a variant run. |
| GPU | optional. CUDA, Apple Silicon (MPS) and plain CPU all work and are picked automatically. |
| Gurobi | not needed. The `gurobipy` package installs without a licence; the offline optimum uses the free CLARABEL or OSQP solvers unless a licensed Gurobi is found. |
| Jupyter | only for evaluation (Section 6): `pip install notebook`. |

## 3. Install (10 minutes)

Linux / macOS:

```bash
git clone https://github.com/danialsj77/TC-CPL.git
cd TC-CPL
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tccpl_learner.py
```

Windows (PowerShell or Command Prompt):

```bat
git clone https://github.com/danialsj77/TC-CPL.git
cd TC-CPL
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python tccpl_learner.py
```

The last command must print `self-test OK` twice; it takes a few seconds. If `python3.11` or
`py -3.11` is not found, install Python 3.11 from python.org (macOS: `brew install
python@3.11`) or use 3.10 in the same way.

Windows notes: if PowerShell answers `running scripts is disabled on this system` at the
`activate` line, run once `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`,
or use Command Prompt instead, where the same line runs `activate.bat`. Environment variables
are set with `set NAME=value` in Command Prompt and `$env:NAME="value"` in PowerShell; the
`NAME=value python …` form used below is for Linux and macOS.

Every command on this page is run from the repository folder with the environment activated.
In every new terminal, activate it again: `source .venv/bin/activate` (Linux / macOS) or
`.venv\Scripts\activate` (Windows).

## 4. First run: a smoke test (2 minutes)

This trains Self-tuning TC-CPL for 300 steps on PublicPST-S. It exercises the whole chain
(simulator, demonstrations, learner, checkpoint) and takes 10 to 20 seconds once the imports
are done:

```bash
python train_one.py --algo tccpl --seed 1 --scenario PublicPST_S --timesteps 300 --offline-transitions 100
```

It ends with `=== done: tccpl seed 1 in 0.2 min (300 timesteps) ===` and writes

- `models/PublicPST_S/tccpl_sac_s1.zip`, the checkpoint, with a sidecar
  `tccpl_sac_s1.runmeta.json` that records `"smoke_test": true`, so the campaign scripts never
  count this tiny model as a finished run;
- `models/PublicPST_S/tccpl_s1/` and `logs/PublicPST_S/tccpl_s1/` (training records),
  `results/PublicPST_S/lambda_history_tccpl_s1.json` (its constraint prices) and `cache/` (the
  demonstrations it collected).

Seed 1 is used so that nothing of the paper is touched: seed 42 would overwrite the paper's
own price history `results/PublicPST_S/lambda_history_tccpl.json` (if that happens, restore it
with `git checkout results/PublicPST_S`). Delete the `_s1` files before you evaluate in
Section 6, because the notebook evaluates every checkpoint it finds, this one included.

All five algorithms at once, about two minutes (Linux / macOS):

```bash
SMOKE=1 ./runall.sh PublicPST_S 1
```

On Windows, run the `python` command above once per algorithm, with `--algo fixedpenalty`,
`ddpg`, `ppo` and `cpo` in turn.

## 5. Train a controller

One algorithm, one seed, one scenario, at the paper's budget of 1,000,000 steps:

```bash
python train_one.py --algo tccpl --seed 42 --scenario PublicPST_S
```

- `--algo`: `fixedpenalty` (Fixed-price TC-CPL), `tccpl` (Self-tuning TC-CPL), `ddpg`, `ppo` or `cpo`.
- `--scenario`: `PublicPST_S` or `cityNL` (`PublicPST`, the 100-kW scenario of the first submission, still works).
- `--seed`: the paper uses 42, 43, 44, 45 and 46.
- `--threads N`: CPU threads of this process (default 4). Keep the total over parallel runs at or below the physical cores.

**How long it takes.** Times of the paper's campaign, one job per GPU with two CPU threads in
most cases, several jobs sharing each server:

| Scenario | Self-tuning / Fixed-price TC-CPL | DDPG-PST | PPO-PST | CPO |
|---|---|---|---|---|
| PublicPST-S | 10–21 h | 1.5–3.5 h | 20 min | 20 min |
| cityNL | 12–30 h | 7–9 h | 4.5–6 h | 5–7 h |

The time goes into the simulator, not into the neural network, so a faster GPU helps little,
and a CPU-only machine is slower by a factor of a few. For a first real run on a laptop, train
for fewer steps:

```bash
TCCPL_TOTAL_TIMESTEPS=200000 python train_one.py --algo tccpl --seed 42 --scenario PublicPST_S
```

(Windows: `set TCCPL_TOTAL_TIMESTEPS=200000` in Command Prompt or
`$env:TCCPL_TOTAL_TIMESTEPS="200000"` in PowerShell on its own line, then the `python` command.)
200,000 steps is the budget of the paper's variant experiments; a TC-CPL run of that size took
1.5 to 12 hours on CPU on the campaign's busy servers.

All five algorithms of one scenario in parallel (Linux / macOS; about 8 GB of RAM per job,
`THREADS=2` to give each process fewer cores):

```bash
./runall.sh PublicPST_S 42
```

Finished runs are skipped when the command is repeated, so an interrupted batch continues where
it stopped (`FORCE=1` retrains). On Windows, open five terminals, activate the environment in
each, and run `python train_one.py --algo <A> --seed 42 --scenario PublicPST_S --threads 2` with
`<A>` = `fixedpenalty`, `tccpl`, `ddpg`, `ppo`, `cpo`.

**What a run writes:**

| Path | What |
|---|---|
| `models/<scenario>/<name>_sac.zip` (`_s43` … for the other seeds) | the checkpoint; `<name>` is `tccpl`, `fixedpenalty`, `ddpg_pst`, `ppo_pst` or `cpo_pst` |
| `models/<scenario>/<name>_sac.runmeta.json` | how it was trained: steps, seed, settings, minutes, whether it was a smoke test |
| `models/<scenario>/<name>/vec_normalize.pkl` | observation statistics (DDPG, PPO, CPO) |
| `logs/<scenario>/<name>/evaluations.npz` | the periodic evaluation returns during training, from which the learning curves are drawn |
| `logs/tensorboard/` | TensorBoard event files of every run (`tensorboard --logdir logs/tensorboard`) |
| `results/<scenario>/lambda_history_<algo>.json` | the constraint prices during training (the two TC-CPL learners) |
| `results/<scenario>/cpo_history.json` | CPO's constraint value during training |
| `cache/` | the demonstration buffer of this scenario and seed (about 90 MB), reused by the other TC-CPL learner and by repeated runs of the same seed |

Files of the other seeds carry the suffix `_s43`, `_s44`, … (seed 42, the first seed of the
protocol, has none). Training overwrites the checkpoint and the price history of the same
algorithm, seed and scenario, so a seed-42 run replaces the paper's `lambda_history_*.json`
in `results/<scenario>/`; `git checkout results/<scenario>` brings the shipped files back. The
other evaluation outputs of the paper in `results/` stay until you run Section 6.

The trained checkpoints of the paper (50 models, 440 MB) are not in the repository; the
numbers they produced are (Section 7).

## 6. Evaluate and draw the tables and figures

Evaluation is Section 7 of the notebook `TC-CPL.ipynb`. It evaluates every checkpoint found
in `models/<scenario>/` on the same 100 simulated days, together with the two heuristics and
the offline optimum, and writes the tables and figures to `results/<scenario>/`.

```bash
pip install notebook
jupyter notebook TC-CPL.ipynb
```

1. Remove smoke-test models first (the `_s1` files of Section 4): the notebook evaluates every
   checkpoint in `models/<scenario>/` by file name, whatever its budget.
2. The configuration cell (Section 2) asks which scenario to use; type its number or name and
   confirm. To skip the question, set `TCCPL_SCENARIO=PublicPST_S` (or `cityNL`) in the
   terminal before starting Jupyter.
3. Run the cells from the top through Section 6.1 (Policy Configuration). Section 0 lists any
   package the kernel is missing and the `%pip` command that installs it.
4. Skip Sections 6.2 to 6.6: they train, which takes hours.
5. Run Section 7 cell by cell, or select its first cell and use "Run all below". Section 8
   then writes `evaluation_results.json`.

How long: PublicPST-S well under an hour; cityNL several hours (an AC power flow for 122
substations at every step). Each finished (model, seed) pair is cached in
`results/<scenario>/eval_cache/`, so an interrupted run resumes where it stopped; delete that
folder to evaluate the controllers from scratch (and `results/<scenario>/optimal_stats.json` to
solve the offline optimum again).

Train Self-tuning TC-CPL (`--algo tccpl`) first: Sections 7.12 and 7.13 compare every other
controller with it and stop if it is missing. With that single checkpoint, the tables contain
it, CAFAP, Round-Robin and the optimum; every further checkpoint is added. The paper's tables
were made from five seeds of each learner; `headline_results.json` records which seeds went
into the shipped numbers.

## 7. What the result files mean

`results/cityNL/` and `results/PublicPST_S/` hold the paper's evaluation: five seeds of every
learner, final checkpoints, the same 100 days for every controller.

| File | What |
|---|---|
| `table_II_comparison.csv` / `.json` | mean and standard deviation over the 100 days of the eight metrics below (`<column>_mean`, `<column>_std`), one row per controller |
| `table_III_metric_comparison.csv` / `.json` | twelve metrics × every controller, with display names: the eight below plus voltage violation, profit, battery degradation and episode return (means in the CSV, mean and standard deviation in the JSON) |
| `headline_results.json` | the key numbers (unmet demand, delivered energy, overload, clean-day share, satisfaction, significance) and the seeds that were evaluated |
| `per_episode_metrics.json` | every metric of every controller on every day (averaged over seeds) |
| `significance_tests.json` | paired tests of every controller against Self-tuning TC-CPL over the 100 days: McNemar on overload occurrence, Wilcoxon on overload, overload intensity and unmet demand, Holm-adjusted |
| `evaluation_results.json` | mean and standard deviation of every metric for every controller under machine-readable names (`tc_cpl_m1_m2_m3`, `ddpg_pst_paper`, …), written by Section 8 of the notebook |
| `optimal_stats.json` | the offline optimum, day by day (also its cache) |
| `lambda_history_*.json` | the constraint prices during training, one file per TC-CPL learner and seed |
| `scenario_identity.json` | the network every controller ran on (a check that the comparison is paired) |
| `*.png` | figures. `results/cityNL/` holds the full set: learning curves, metric boxplot, setpoint tracking, safety summary, the design panels of both scenarios, price trajectories, policy snapshot. `results/PublicPST_S/` holds its learning curves and metric boxplot; its design panel is `results/cityNL/design_panels_PublicPST_S.png` |

Metric columns:

| Column | Meaning |
|---|---|
| `eps_tr_kW2` | squared setpoint-tracking error over the day (kW²) |
| `abs_eps_tr_kWh` | energy tracking error over the day (kWh) |
| `eps_sur_kW` | power drawn above the setpoint (kW) |
| `eps_usr` | user satisfaction at departure, 1 = every EV received all the energy it asked for; requirement ≥ 0.95 |
| `overload_kWh` | transformer energy above its limit over the day (kWh); requirement 0 |
| `overload_per_MWh` | the same per MWh delivered, so controllers that deliver different amounts can be compared |
| `delivered_kWh` | energy delivered over the day (kWh) |
| `unmet_demand_pct` | share of the achievable demand that was not delivered (%) |

`revision/runs/PublicPST_stress/<variant>_s<seed>/` holds the variant experiments of Section 8:
`train_meta.json` (settings, prices during training, minutes), `eval.json` (the 100 days) and,
where present, `eval_shield.json` (the same days with the headroom projection at dispatch).

## 8. The variant experiments (optional)

`revision/` holds the ablations and sensitivity runs on PublicPST-S: 25 variants of TC-CPL
(frozen prices at several values, no demonstrations, no residual anchoring, no transformer
block, a flat encoder, other gains, caps and tolerances, …), each trained for 200,000 steps on
CPU and evaluated on the same 100 days, plus the reference run and the two heuristics: 66 runs
in all. `revision/README.md` describes the scripts.

One variant, from the `revision/` folder:

```bash
cd revision
python scripts/train_variant.py no_demos 45 budget=200000 update_freq=3072 device=cpu
```

Results go to `revision/runs/PublicPST_stress/<variant>_s<seed>/`. A folder whose records
already match the variant's definition and holds its model is skipped without a message (seed
45 of `no_demos` is not in the paper, so it runs); `EXP_FORCE=1` repeats a run.
`revision/stress_tasks.txt` lists the 66 runs of the paper, and
`python -c "import variants; print(sorted(variants.VARIANTS))"` (in `revision/`) prints every
variant the code knows: 31, including the reference, its ablation and four `pid_*` settings
that were explored but are not part of the paper.

## 9. Changing settings

Every setting is an environment variable with the paper's protocol as default; the full list
is the [Settings table of the README](README.md#settings). The ones most people need:

| Variable | Example | Effect |
|---|---|---|
| `TCCPL_TOTAL_TIMESTEPS` | `200000` | training steps per run (paper: 1,000,000) |
| `TCCPL_DEVICE` | `cpu`, `cuda`, `mps` | force a device instead of the automatic choice |
| `TCCPL_SCENARIO` | `PublicPST_S` | the scenario of a notebook kernel (scripts take `--scenario`) |
| `TCCPL_DDPG_BUFFER` | `300000` | smaller DDPG replay buffer on machines with little RAM |

Set them in the terminal before the command: `export NAME=value` on Linux / macOS (or
`NAME=value python …` on one line, as in Section 5), `set NAME=value` in Command Prompt,
`$env:NAME="value"` in PowerShell.

## 10. Problems and fixes

| Symptom | Cause and fix |
|---|---|
| `pip` cannot find `numpy==1.24` or `pandas==2.0` | Python 3.12 or newer. Create the environment with 3.10 or 3.11. |
| `ModuleNotFoundError: No module named 'pkg_resources'` | setuptools 80 removed it: `pip install "setuptools<80"` (already pinned in `requirements.txt`). |
| `No module named torch` (or any package) when running a script | the environment is not activated in this terminal. |
| `./runall.sh: Permission denied` | `chmod +x runall.sh`, or run `bash runall.sh …`. |
| PowerShell: `running scripts is disabled on this system` at `activate` | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once, or use Command Prompt. |
| Windows and the `.sh` scripts | run the `python train_one.py …` commands yourself, one per algorithm (Section 5). In Git Bash the scripts work with the venv's interpreter named explicitly: `PYTHON=.venv/Scripts/python bash runall.sh PublicPST_S 42`. WSL is a Linux system of its own: repeat Section 3 (Linux commands) inside it. |
| Windows: `UnicodeEncodeError: 'charmap' codec can't encode character` | output redirected to a file uses the old code page; set `PYTHONUTF8=1` (`set PYTHONUTF8=1`, `$env:PYTHONUTF8="1"` or `export PYTHONUTF8=1`) and run again. |
| `OpenBLAS: Memory allocation still failed`, or the machine thrashes | too many threads for parallel runs: `--threads 2` for `train_one.py`, `THREADS=2 ./runall.sh …`, or fewer jobs at once. |
| Out of memory with DDPG on cityNL | `TCCPL_DDPG_BUFFER=300000`. |
| The notebook kernel cannot import a package that pip installed | the kernel is another Python; run the `%pip install …` line that Section 0 prints, inside the notebook. |
| The notebook runs the wrong scenario, or you do not want the question | set `TCCPL_SCENARIO` before starting Jupyter. |
| Evaluation on cityNL is slow | that is the power flow; it caches every finished model in `results/cityNL/eval_cache/` and resumes after an interruption. |
| Gurobi licence warnings | harmless; the optimum uses CLARABEL or OSQP instead. |
| A run was interrupted | repeat the same command: `train_one.py` retrains that job; `runall.sh` and `run_sweep.py` skip finished jobs and run only the missing ones. |
| Apple Silicon: the GPU (MPS) misbehaves | `TCCPL_DEVICE=cpu`. |

## 11. Citing

If you use this code, please cite both the TC-CPL manuscript and the EV2Gym paper, whose
simulator this repository extends:

- D. Sajedi-Hosseini, Y. Fujisawa, S. Singh, L. Zhou, K. Fujisawa, S. Gros, M. Karimi and
  X. Shen, "Power Setpoint Tracking for Large-Scale EV Charging: Transformer-Constrained
  Charging Policy Learning with Self-Tuning Constraint Prices," submitted to *IEEE Transactions
  on Transportation Electrification*, 2026.
- S. Orfanoudakis et al., EV2Gym, https://github.com/StavrosOrf/EV2Gym.

Questions and problems: open an issue at https://github.com/danialsj77/TC-CPL/issues.
