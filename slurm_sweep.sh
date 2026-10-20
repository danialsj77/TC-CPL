#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# slurm_sweep.sh — the full TC-CPL protocol as ONE SLURM job array:
#   2 scenarios (cityNL, PublicPST_S) × 5 seeds × 5 algorithms = 50 tasks, each `train_one.py` at
#   1,000,000 environment steps (TCCPL_TOTAL_TIMESTEPS overrides).
#
#   sbatch slurm_sweep.sh                 # everything
#   TCCPL_ALGOS=fixedpenalty:tccpl sbatch --array=0-19 slurm_sweep.sh   # only the two TC-CPL learners
#   TCCPL_DRY=1 SLURM_ARRAY_TASK_ID=7 bash slurm_sweep.sh             # print what task 7 runs
#
# Task index → (scenario, seed, algorithm): index = scen·25 + seed_i·5 + algo_i.
# Finished jobs (runmeta sidecar at the current budget) exit at once, so a
# partly failed array is simply resubmitted. Edit the #SBATCH header and the
# environment activation for the target system before submitting.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=tccpl
#SBATCH --array=0-49
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
# (no mid-run checkpoint: a job killed by the time limit restarts from zero)
#SBATCH --time=96:00:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
set -euo pipefail
# SLURM runs a spooled COPY of this script, so $0 is not in the repository:
# start from the submission folder (submit from the repository root)
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f train_one.py ] || { echo "submit from the repository root (no train_one.py in $PWD)"; exit 1; }
mkdir -p logs/slurm

# ── environment: activate the venv/conda env that has the training stack ────
# source .venv/bin/activate     # python3.11 -m venv .venv && pip install -r requirements.txt
# module load cuda               # if the site needs it

# cityNL and PublicPST_S (which replaces the first submission's PublicPST)
read -r -a SCENARIOS <<< "${TCCPL_SCENARIOS:-cityNL PublicPST_S}"
SEEDS=(${TCCPL_SEEDS_LIST:-42 43 44 45 46})
# TCCPL_ALGOS restricts the array, e.g. TCCPL_ALGOS=fixedpenalty:tccpl -> --array=0-19
_algos="${TCCPL_ALGOS:-fixedpenalty tccpl ddpg ppo cpo}"
read -r -a ALGOS <<< "${_algos//[,:]/ }"
export TCCPL_TOTAL_TIMESTEPS="${TCCPL_TOTAL_TIMESTEPS:-1000000}"
export TCCPL_SEEDS="$(IFS=,; echo "${SEEDS[*]}")"

i="${SLURM_ARRAY_TASK_ID:-0}"
n_per_scen=$(( ${#SEEDS[@]} * ${#ALGOS[@]} ))
if [ "$i" -ge $(( ${#SCENARIOS[@]} * n_per_scen )) ]; then
  echo "task $i is beyond the $(( ${#SCENARIOS[@]} * n_per_scen ))-task grid; nothing to do"; exit 0
fi
scen="${SCENARIOS[$(( i / n_per_scen ))]}"
seed="${SEEDS[$(( (i % n_per_scen) / ${#ALGOS[@]} ))]}"
algo="${ALGOS[$(( i % ${#ALGOS[@]} ))]}"
if [ -n "${TCCPL_DRY:-}" ]; then echo "task $i: $algo $scen seed $seed"; exit 0; fi

# ── skip a job already finished under THIS protocol (run_sweep.is_complete) ──
# Only a sidecar written by this code version at the full budget counts, so a
# folder that still holds models of the first submission is retrained, not reused.
if python run_sweep.py --check "$algo" "$seed" "$scen"; then
  echo "task $i: $algo $scen seed $seed already finished in this campaign — skipping"
  exit 0
fi

echo "task $i: $algo | $scen | seed $seed | ${SLURM_CPUS_PER_TASK:-?} cpus | budget $TCCPL_TOTAL_TIMESTEPS"
python train_one.py --algo "$algo" --seed "$seed" --scenario "$scen" \
    --threads "${SLURM_CPUS_PER_TASK:-4}"
