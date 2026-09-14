#!/bin/sh
# ─────────────────────────────────────────────────────────────────────────────
# tsubame_sweep.sh — the full TC-CPL protocol as ONE Grid-Engine array job
# (Altair/Sun Grid Engine `qsub`, e.g. TSUBAME): 50 tasks = 2 scenarios ×
# 5 seeds × 5 algorithms, each `train_one.py` at 1,000,000 steps.
#
#   qsub -g <group> tsubame_sweep.sh            # all 50 tasks
#   qsub -g <group> -t 1-25 tsubame_sweep.sh    # cityNL only
#   qsub -g <group> -t 26-50 tsubame_sweep.sh   # PublicPST only
#
# Task t (1-based) → i = t-1 → scenario i/25, seed 42 + (i%25)/5, algorithm i%5
# (fixedpenalty, tccpl, ddpg, ppo, cpo). Finished tasks exit at once, so a
# partly failed array is simply resubmitted. There is NO mid-run checkpoint:
# h_rt must exceed one full job (measure one with train_one.py first).
# Edit the resource type, h_rt and the environment lines for the site.
# ─────────────────────────────────────────────────────────────────────────────
#$ -cwd
#$ -N tccpl
#$ -l gpu_1=1
#$ -l h_rt=24:00:00
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

i=$(( ${SGE_TASK_ID:-1} - 1 ))
case $(( i / 25 )) in 0) scen=cityNL ;; *) scen=PublicPST ;; esac
seed=$(( 42 + (i % 25) / 5 ))
case $(( i % 5 )) in
  0) algo=fixedpenalty ;; 1) algo=tccpl ;; 2) algo=ddpg ;; 3) algo=ppo ;; *) algo=cpo ;;
esac

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
  echo "task $((i+1)): $algo $scen seed $seed already trained at $TCCPL_TOTAL_TIMESTEPS steps — skipping"
  exit 0
fi

echo "task $((i+1)): $algo | $scen | seed $seed | $THREADS threads | budget $TCCPL_TOTAL_TIMESTEPS"
python train_one.py --algo "$algo" --seed "$seed" --scenario "$scen" --threads "$THREADS"
