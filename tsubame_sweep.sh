#!/bin/sh
# ─────────────────────────────────────────────────────────────────────────────
# tsubame_sweep.sh — the full TC-CPL protocol as ONE Grid-Engine array job
# (Altair/Sun Grid Engine `qsub`, e.g. TSUBAME): 50 tasks = 2 scenarios ×
# 5 seeds × 5 algorithms, each `train_one.py` at 1,000,000 steps.
#
#   qsub -g <group> tsubame_sweep.sh            # all 50 tasks
#   qsub -g <group> -t 1-25 tsubame_sweep.sh    # cityNL only
#   qsub -g <group> -t 26-50 tsubame_sweep.sh   # PublicPST only
#   qsub -g <group> -t 1-20 -v TCCPL_ALGOS=fixedpenalty:tccpl tsubame_sweep.sh
#                                               # only the two TC-CPL learners
#   TCCPL_DRY=1 SGE_TASK_ID=7 bash tsubame_sweep.sh   # print what task 7 runs
#
# Task t (1-based) → i = t-1 → scenario i/25, seed 42 + (i%25)/5, algorithm i%5
# (fixedpenalty, tccpl, ddpg, ppo, cpo). Finished tasks exit at once, so a
# partly failed array is simply resubmitted. There is NO mid-run checkpoint:
# h_rt must exceed one full job: cityNL took 21-30 h per job on one GPU in the first
# protocol, so 48 h is requested (lower it only if the site caps it and a job fits).
# Edit the resource type, h_rt and the environment lines for the site.
# ─────────────────────────────────────────────────────────────────────────────
#$ -cwd
#$ -N tccpl
#$ -l gpu_1=1
#$ -l h_rt=48:00:00
#$ -t 1-50
#$ -j y
#$ -o logs/qsub/
set -eu
mkdir -p logs/qsub

# ── environment ─────────────────────────────────────────────────────────────
# module load cuda                    # if the site needs it
# . .venv/bin/activate                # python3.11 -m venv .venv && pip install -r requirements.txt
THREADS="${THREADS:-8}"               # CPU cores of the requested resource type

export TCCPL_TOTAL_TIMESTEPS="${TCCPL_TOTAL_TIMESTEPS:-1000000}"
export TCCPL_SEEDS="42,43,44,45,46"

SCENARIOS=(cityNL PublicPST)
SEEDS=(42 43 44 45 46)
# TCCPL_ALGOS restricts the array, e.g. TCCPL_ALGOS=fixedpenalty:tccpl -> 20 tasks
# (2 scenarios x 5 seeds x 2 algorithms); ':' ',' or ' ' separate the names.
_algos="${TCCPL_ALGOS:-fixedpenalty tccpl ddpg ppo cpo}"
read -r -a ALGOS <<< "${_algos//[,:]/ }"
n_per_scen=$(( ${#SEEDS[@]} * ${#ALGOS[@]} ))
i=$(( ${SGE_TASK_ID:-1} - 1 ))
if [ "$i" -ge $(( ${#SCENARIOS[@]} * n_per_scen )) ]; then
  echo "task $((i+1)) is beyond the $(( ${#SCENARIOS[@]} * n_per_scen ))-task grid; nothing to do"; exit 0
fi
scen="${SCENARIOS[$(( i / n_per_scen ))]}"
seed="${SEEDS[$(( (i % n_per_scen) / ${#ALGOS[@]} ))]}"
algo="${ALGOS[$(( i % ${#ALGOS[@]} ))]}"
if [ -n "${TCCPL_DRY:-}" ]; then echo "task $((i+1)): $algo $scen seed $seed"; exit 0; fi

# ── skip a job already finished at THIS budget (same rule as run_sweep.py) ──
if python - "$algo" "$seed" "$scen" <<'PYEOF'
import json, os, sys
algo, seed, scen = sys.argv[1:4]
name = {"ddpg": "ddpg_pst", "ppo": "ppo_pst", "cpo": "cpo_pst"}.get(algo, algo)
suf = "" if seed == "42" else f"_s{seed}"
budget = int(os.environ.get("TCCPL_TOTAL_TIMESTEPS", 1_000_000))
try:
    info = json.load(open(os.path.join("models", scen, f"{name}_sac{suf}.runmeta.json")))
    done = (not info.get("smoke_test")) and int(info.get("timesteps", -1)) == budget
    if algo in ("tccpl", "fixedpenalty"):
        done = done and bool(info.get("tr_load_aware", False)) == bool(int(os.environ.get("TCCPL_TR_LOAD_AWARE", "1"))) \
            and int(info.get("demo_seed_offset", 0)) == int(os.environ.get("TCCPL_DEMO_SEED_OFFSET", 100000)) \
            and abs(float(info.get("rho", 0.25)) - float(os.environ.get("TCCPL_RHO", 1.0))) < 1e-9
    if algo == "tccpl":
        done = done and abs(float(info.get("train_target_overload", 0.0)) - float(os.environ.get("TCCPL_TRAIN_TARGET_OVERLOAD", 0.0))) < 1e-9
    sys.exit(0 if done else 1)
except Exception:
    sys.exit(1)
PYEOF
then
  echo "task $((i+1)): $algo $scen seed $seed already trained at $TCCPL_TOTAL_TIMESTEPS steps — skipping"
  exit 0
fi

echo "task $((i+1)): $algo | $scen | seed $seed | $THREADS threads | budget $TCCPL_TOTAL_TIMESTEPS"
python train_one.py --algo "$algo" --seed "$seed" --scenario "$scen" --threads "$THREADS"
