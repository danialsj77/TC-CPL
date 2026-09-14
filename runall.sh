#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# runall.sh — train ALL FIVE algorithms of one scenario AT THE SAME TIME.
#
#   ./runall.sh                     # cityNL,    seed 42  (the defaults)
#   ./runall.sh PublicPST           # PublicPST, seed 42
#   ./runall.sh cityNL 43           # cityNL,    seed 43
#
#   SMOKE=1  ./runall.sh PublicPST  # ~2-min tiny-budget rehearsal of the
#                                   # whole thing (300 steps per algorithm)
#   FORCE=1  ./runall.sh            # retrain even if a finished run exists
#   THREADS=4 PYTHON=.venv/bin/python ./runall.sh    # knobs
#
#   Constraint-tightening knobs (read by the notebook's CONFIG cell; the
#   requirements stay 0 kWh / 0.95, these set what the learner TRAINS against):
#   TCCPL_OV_MARGIN=0.05         # dual-priced overload above (1 - m) x limit
#   TCCPL_TRAIN_SAT_FLOOR=0.98   # satisfaction floor used by the dual controller
#   e.g.  TCCPL_OV_MARGIN=0.10 ./runall.sh cityNL 42
#
# One process per algorithm (fixedpenalty, tccpl, ddpg, ppo, cpo), each writing
# to logs/runall/<scenario>_<algo>_s<seed>.log. Jobs already completed at the
# FULL budget are skipped, so an interrupted batch resumes where it stopped.
# Ctrl-C stops every job. caffeinate keeps the Mac awake while training runs.
#
# Full protocol = 5 seeds x 1,000,000 steps per scenario (run_sweep.py or
# slurm_sweep.sh on a cluster; here one seed at a time):
#   for s in 42 43 44 45 46; do ./runall.sh cityNL $s; done
# then the same for PublicPST, then run notebook §7 once per scenario.
# TCCPL_TOTAL_TIMESTEPS=300000 ./runall.sh ...   # smaller local budget
# ─────────────────────────────────────────────────────────────────────────────
set -u
cd "$(dirname "$0")"

SCENARIO="${1:-cityNL}"
SEED="${2:-42}"
ALGOS=(fixedpenalty tccpl ddpg ppo cpo)
PYTHON="${PYTHON:-python3}"
THREADS="${THREADS:-3}"          # per process: 5 jobs x 3 threads = 15 cores

case "$SCENARIO" in
  cityNL|PublicPST) ;;
  *) echo "usage: ./runall.sh [cityNL|PublicPST] [seed]"; exit 2 ;;
esac

# ── Fail fast if this interpreter lacks the training stack ───────────────────
if ! "$PYTHON" -c "import torch, stable_baselines3, ev2gym, tccpl_learner" 2>/dev/null; then
  echo "!! '$PYTHON' cannot import the training stack."
  echo "   Activate your venv first, or point PYTHON at it:"
  echo "     PYTHON=/path/to/venv/bin/python ./runall.sh $SCENARIO $SEED"
  exit 2
fi

EXTRA=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  EXTRA=(--timesteps 300 --offline-transitions 100)
  echo ">> SMOKE run: 300 timesteps per algorithm (never mistaken for a real"
  echo ">> run — the completion check ignores smoke artefacts)"
fi

# ── Skip finished full-budget runs (mirrors run_sweep.is_complete): the
#    sidecar must record the CURRENT budget, not just "not a smoke run" ──────
is_done() {
  "$PYTHON" - "$1" "$SEED" "$SCENARIO" <<'PYEOF'
import json, os, sys
algo, seed, scen = sys.argv[1:4]
name = {"ddpg": "ddpg_pst", "ppo": "ppo_pst", "cpo": "cpo_pst"}.get(algo, algo)
suf = "" if seed == "42" else f"_s{seed}"
budget = int(os.environ.get("TCCPL_TOTAL_TIMESTEPS", 1_000_000))
try:
    info = json.load(open(os.path.join("models", scen, f"{name}_sac{suf}.runmeta.json")))
    ok = (not info.get("smoke_test")) and int(info.get("timesteps", -1)) == budget
    sys.exit(0 if ok else 1)
except Exception:
    sys.exit(1)
PYEOF
}

mkdir -p logs/runall
command -v caffeinate >/dev/null 2>&1 && caffeinate -ims -w $$ &

PIDS=()
NAMES=()
trap 'echo; echo "interrupted — stopping all jobs";
      [ ${#PIDS[@]} -gt 0 ] && kill ${PIDS[@]+"${PIDS[@]}"} 2>/dev/null;
      exit 130' INT TERM

echo "── $SCENARIO · seed $SEED · $THREADS threads/process · python: $PYTHON"
T0=$(date +%s)
for algo in "${ALGOS[@]}"; do
  if [[ "${FORCE:-0}" != "1" ]] && is_done "$algo"; then
    echo "skip  $algo — already trained at the full budget (FORCE=1 to redo)"
    continue
  fi
  LOG="logs/runall/${SCENARIO}_${algo}_s${SEED}.log"
  "$PYTHON" train_one.py --algo "$algo" --seed "$SEED" --scenario "$SCENARIO" \
      --threads "$THREADS" ${EXTRA[@]+"${EXTRA[@]}"} >"$LOG" 2>&1 &
  PIDS+=($!)
  NAMES+=("$algo")
  echo "start $algo  (pid $!)  →  $LOG"
done

if [ ${#PIDS[@]} -eq 0 ]; then
  echo "nothing to do — every job is already trained."
  exit 0
fi

echo
echo "watch progress in another terminal with, e.g.:"
echo "  tail -f logs/runall/${SCENARIO}_tccpl_s${SEED}.log"
echo

FAIL=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "DONE  ${NAMES[$i]}  ($(( ($(date +%s) - T0) / 60 )) min elapsed)"
  else
    echo "FAIL  ${NAMES[$i]} — see logs/runall/${SCENARIO}_${NAMES[$i]}_s${SEED}.log"
    FAIL=1
  fi
done

echo
if [ "$FAIL" -eq 0 ]; then
  echo "All jobs finished. Models → models/${SCENARIO}/   Next:"
  echo "  - remaining seeds:  for s in 43 44 45 46; do ./runall.sh $SCENARIO \$s; done"
  echo "  - then evaluate:    open TC-CPL.ipynb with TCCPL_SCENARIO=$SCENARIO and run §7"
else
  echo "Some jobs FAILED — check the logs above, then re-run ./runall.sh $SCENARIO $SEED"
  echo "(finished jobs are skipped automatically)."
fi
exit "$FAIL"
