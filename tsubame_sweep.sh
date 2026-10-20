#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# tsubame_sweep.sh — the full TC-CPL protocol as ONE Grid-Engine array job
# (Altair/Sun Grid Engine `qsub`, e.g. TSUBAME): 50 tasks = 2 scenarios (cityNL,
# PublicPST_S) × 5 seeds × 5 algorithms, each `train_one.py` at 1,000,000 steps.
#
#   qsub -g <group> tsubame_sweep.sh            # all 50 tasks
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
#$ -S /bin/bash
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

# cityNL and PublicPST_S (which replaces the first submission's PublicPST)
read -r -a SCENARIOS <<< "${TCCPL_SCENARIOS:-cityNL PublicPST_S}"
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

# ── skip a job already finished under THIS protocol (run_sweep.is_complete) ──
# Only a sidecar written by this code version at the full budget counts, so a
# folder that still holds models of the first submission is retrained, not reused.
if python run_sweep.py --check "$algo" "$seed" "$scen"; then
  echo "task $((i+1)): $algo $scen seed $seed already finished in this campaign — skipping"
  exit 0
fi

echo "task $((i+1)): $algo | $scen | seed $seed | $THREADS threads | budget $TCCPL_TOTAL_TIMESTEPS"
python train_one.py --algo "$algo" --seed "$seed" --scenario "$scen" --threads "$THREADS"
