#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# revision/stress_sweep.sh — the PublicPST-S experiments (paper table "stress") as ONE
# array job, for Grid Engine (qsub, e.g. TSUBAME) or SLURM (sbatch). CPU only.
# Submit FROM THE REPOSITORY ROOT, at the same time as the main 50-task array (tsubame_sweep.sh / slurm_sweep.sh):
#
#   mkdir -p logs/qsub  && qsub -g <group> revision/stress_sweep.sh     # Grid Engine: 66 tasks
#   mkdir -p logs/slurm && sbatch revision/stress_sweep.sh              # SLURM
#   TCCPL_DRY=1 SGE_TASK_ID=3 bash revision/stress_sweep.sh             # print what task 3 runs
#
# Task t (Grid Engine 1-based, SLURM 0-based) runs line t of revision/stress_tasks.txt
# (VARIANT SEED [shield], written by revision/scripts/make_jobs.py): it trains the
# variant for 200k steps, evaluates it on the 100 shared days and, if marked, again
# with the headroom projection. Results: revision/runs/PublicPST_stress/<VARIANT>_s<SEED>/.
# A task whose results already match its current definition exits at once, so a
# partly failed array is simply resubmitted. One task takes about 1-2 h on 8 cores.
# ─────────────────────────────────────────────────────────────────────────────
#$ -cwd
#$ -S /bin/bash
#$ -N tccpl_stress
#$ -l cpu_8=1
#$ -l h_rt=8:00:00
#$ -t 1-66
#$ -j y
#$ -o logs/qsub/
#SBATCH --job-name=tccpl_stress
#SBATCH --array=0-65
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
set -eu
# Grid Engine starts in the submission folder (-cwd); SLURM gives it in SLURM_SUBMIT_DIR
# (both run a spooled COPY of this script, so $0 does not point into the repository)
cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f train_one.py ] || [ ! -f revision/stress_tasks.txt ]; then
  echo "run this from the repository root (found no train_one.py / revision/stress_tasks.txt in $PWD)"; exit 1
fi

# ── environment ─────────────────────────────────────────────────────────────
# . .venv/bin/activate                # python3.11 -m venv .venv && pip install -r requirements.txt
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-8}}"     # CPU cores of the requested resource type

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then t=$(( SLURM_ARRAY_TASK_ID + 1 )); else t="${SGE_TASK_ID:-1}"; fi
line="$(grep -v '^#' revision/stress_tasks.txt | sed -n "${t}p")"
if [ -z "$line" ]; then echo "task $t: beyond the task list; nothing to do"; exit 0; fi
read -r variant seed shield <<< "$line"
if [ -n "${TCCPL_DRY:-}" ]; then echo "task $t: $variant seed $seed${shield:+ (+ headroom projection)}"; exit 0; fi

SETTINGS="${STRESS_SETTINGS:-budget=200000 update_freq=3072 device=cpu}"   # = CAMPAIGN in revision/variants.py
export EXP_THREADS="$THREADS"
echo "task $t: $variant | seed $seed | $THREADS threads | $SETTINGS${shield:+ | + headroom projection}"
cd revision
python scripts/train_variant.py "$variant" "$seed" $SETTINGS
if [ -n "${shield:-}" ]; then
  EXP_SHIELD=1 python scripts/train_variant.py "$variant" "$seed" $SETTINGS
fi
echo "task $t: done"
