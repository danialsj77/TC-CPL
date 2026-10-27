# Revision experiments

Everything added for the major revision lives here. Nothing in `models/`, `results/` or `logs/` of the paper is written by these scripts; every run writes into `revision/runs/`. Run the scripts from this folder with the repository's environment activated (Python 3.10 or 3.11, `pip install -r ../requirements.txt`).

| Path | What |
|---|---|
| `exp_lib.py` | Switches around the unchanged learner (`../tccpl_learner.py`): fixed vs adaptive prices, cap, gains, tolerance τ, margin, incumbent, ρ, demonstrations, residual, transformer block, flat encoder; extended evaluation records; observation perturbations; headroom shield |
| `variants.py` | Named variants (each overrides only the listed keys of the protocol); 31 defined, 25 of them in the paper (`stress_tasks.txt`), the `pid_*` ones exploratory |
| `../ev2gym/example_config_files/PublicPST_S.yaml` | PublicPST-S: PublicPST with a 42-kW transformer and cityNL's household load, PV and demand response |
| `stress_sweep.sh`, `stress_tasks.txt` | The 66 variant runs of the paper as one Grid Engine / SLURM array (`VARIANT SEED [shield]` per line, written by `scripts/make_jobs.py`) |
| `scripts/train_variant.py` | Train one variant/seed on PublicPST-S, then evaluate it on the 100 paired days (`EXP_SHIELD=1` evaluates with the headroom projection; `EXP_FORCE=1` repeats a finished run) |
| `scripts/eval_citynl.py` | Re-evaluate the main checkpoints with per-interval records (`EXP_SCEN=PublicPST_S` for PublicPST-S, default cityNL; `EXP_SHIELD=1` for the shield; `kind:p` for a perturbation; `--offset 5000` for held-out days) |
| `scripts/offline_feasibility.py` | Hindsight QPs per evaluation day (tracking optimum, best satisfaction under zero overload, optimum with a satisfaction floor) |
| `scripts/attainable_floor.py` | Overload above the tightened limit of the incumbent (`zero`) and of the most conservative residual (`minus1`) on the evaluation days; CAFAP peak loading (`runs/floor/`) |
| `scripts/scaling_bench.py` | Parameter counts, latency, update time, memory vs number of ports (`analysis/scaling_bench.json`) |
| `scripts/pin_day0.py` | Re-evaluates the first PublicPST-S day of evaluations made before `exp_lib.pin_first_day` (days 2-100 already coincide across controllers) |
| `scripts/campaign.py`, `scripts/make_jobs.py`, `jobs.json` | Unattended job queue for the 150 re-evaluations (restartable; finished jobs are skipped); `make_jobs.py` also writes `stress_tasks.txt` |
| `scripts/analyze_submitted.py` | Paired statistics and 5-seed dual traces from `../results/<scenario>/eval_cache` (`analysis/submitted_stats.json`, `analysis/*.png`) |
| `scripts/collect_results.py` | Gathers every run into `analysis/results.json`; missing runs are reported, never invented |
| `scripts/stress_stats.py` | PublicPST-S paired statistics: TC-CPL minus each variant, and with minus without the headroom projection (`analysis/stress_stats.json`) |
| `scripts/heldout_stats.py` | cityNL comparison on the 100 held-out days, seeds 5000-5099 (`analysis/heldout_stats.json`) |
| `scripts/make_tables.py`, `scripts/fig_safety.py` | LaTeX tables and the safety figure of the manuscript, written into a local `../IEEE TTE/` folder (not part of the repository) |
| `scripts/provenance.py` | Decides whether a record was made from the checkpoint and variant definition now on disk; used by every consumer above |
| `scripts/calibrate_stress.py`, `scripts/collect_demos.py`, `scripts/pilot_status.py` | Helpers: transformer-rating calibration of PublicPST-S, demonstration collection, status of a pilot run |
| `runs/` | In git: `PublicPST_stress/<variant>_s<seed>/` (the 66 variant runs: `train_meta.json`, `eval.json`, `eval_shield.json` where listed; the `model.zip` files are not in git), `floor/` (attainable-overload floors), `offline_v2/` (hindsight study). Written by `jobs.json`, not in git: `cityNL_eval/`, `PublicPST_S_eval/` |
| `analysis/` | Created by the statistics scripts above |
| `cache/`, `logs/` | Demonstrations and job logs of the runs; not in git |

Queue status while `campaign.py` runs: `jobs.json.status.json`.
