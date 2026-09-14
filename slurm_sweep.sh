#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# slurm_sweep.sh — the full TC-CPL protocol as ONE SLURM job array:
#   2 scenarios × 5 seeds × 5 algorithms = 50 tasks, each `train_one.py` at
#   1,000,000 environment steps (TCCPL_TOTAL_TIMESTEPS overrides).
#
#   sbatch slurm_sweep.sh                 # everything
#   sbatch --array=0-24 slurm_sweep.sh    # cityNL only
#   sbatch --array=25-49 slurm_sweep.sh   # PublicPST only
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
cd "$(dirname "$(readlink -f "$0")")"
mkdir -p logs/slurm

# ── environment: activate the venv/conda env that has the training stack ────
# source .venv/bin/activate     # python3.11 -m venv .venv && pip install -r requirements.txt
# module load cuda               # if the site needs it

SCENARIOS=(cityNL PublicPST)
SEEDS=(${TCCPL_SEEDS_LIST:-42 43 44 45 46})
ALGOS=(fixedpenalty tccpl ddpg ppo cpo)
export TCCPL_TOTAL_TIMESTEPS="${TCCPL_TOTAL_TIMESTEPS:-1000000}"
export TCCPL_SEEDS="$(IFS=,; echo "${SEEDS[*]}")"

i="${SLURM_ARRAY_TASK_ID:-0}"
n_per_scen=$(( ${#SEEDS[@]} * ${#ALGOS[@]} ))
scen="${SCENARIOS[$(( i / n_per_scen ))]}"
seed="${SEEDS[$(( (i % n_per_scen) / ${#ALGOS[@]} ))]}"
algo="${ALGOS[$(( i % ${#ALGOS[@]} ))]}"

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
    sys.exit(0 if done else 1)
except Exception:
    sys.exit(1)
PYEOF
then
  echo "task $i: $algo $scen seed $seed already trained at $TCCPL_TOTAL_TIMESTEPS steps — skipping"
  exit 0
fi

echo "task $i: $algo | $scen | seed $seed | ${SLURM_CPUS_PER_TASK:-?} cpus | budget $TCCPL_TOTAL_TIMESTEPS"
python train_one.py --algo "$algo" --seed "$seed" --scenario "$scen" \
    --threads "${SLURM_CPUS_PER_TASK:-4}"
