# Revision experiments

Everything added for the major revision lives here. Nothing in `models/`, `results/` or `logs/` of the paper is written by these scripts.

| Path | What |
|---|---|
| `exp_lib.py` | Switches around the unchanged learner (`../tccpl_learner.py`): fixed vs adaptive prices, cap, gains, tolerance τ, margin, incumbent, ρ, demonstrations, residual, transformer block, flat encoder; extended evaluation records; observation perturbations; headroom shield |
| `variants.py` | Named variants (each overrides only the listed keys of the submitted protocol) |
| `configs/PublicPST_stress.yaml` | PublicPST-S: PublicPST with a 50-kW transformer and cityNL's household load, PV and demand response |
| `scripts/train_variant.py` | Train one variant/seed on PublicPST-S, then evaluate it on the 100 paired days |
| `scripts/eval_citynl.py` | Re-evaluate the submitted checkpoints (`EXP_SCEN=PublicPST` for PublicPST; `EXP_SHIELD=1` for the shield; `kind:p` for a perturbation; `--offset 5000` for held-out days) |
| `scripts/offline_feasibility.py` | Hindsight QPs per evaluation day (tracking optimum, best satisfaction under zero overload, optimum with a satisfaction floor) |
| `scripts/analyze_submitted.py` | Paired statistics and 5-seed dual traces from the submitted results |
| `scripts/scaling_bench.py` | Parameter counts, latency, update time, memory vs number of ports |
| `scripts/attainable_floor.py` | Overload above the tightened limit of the incumbent (`zero`) and of the most conservative residual (`minus1`) on the evaluation days; CAFAP peak loading (`runs/floor/`) |
| `scripts/pin_day0.py` | Re-evaluates the first PublicPST-S day of evaluations made before `exp_lib.pin_first_day` (days 2-100 already coincide across controllers) |
| `scripts/stress_stats.py` | PublicPST-S paired statistics: TC-CPL minus each variant, and with minus without the headroom projection (`analysis/stress_stats.json`) |
| `scripts/heldout_stats.py` | cityNL comparison on the 100 held-out days, seeds 5000-5099 (`analysis/heldout_stats.json`) |
| `scripts/campaign.py`, `scripts/make_jobs.py`, `jobs.json` | Unattended job queue (restartable; finished jobs are skipped) |
| `runs/` | Outputs: `PublicPST_stress/<variant>_s<seed>/`, `cityNL_eval/`, `PublicPST_eval/`, `offline/` |
| `analysis/` | Tables and figures computed from the runs |
| `cache/` | M2 demonstrations (`v1_overlapping_days/` = first attempt, superseded) |
| `runs_smoke/` | Smoke tests and benchmarks, not results |

Environment: `~/.venvs/tccpl311` (Python 3.11, `pip install -r ../requirements.txt`). Queue status: `jobs.json.status.json`.
