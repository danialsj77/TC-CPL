"""Run the training sweep with a bounded number of concurrent processes.

    python run_sweep.py                      # all algos x all seeds, 2 at a time
    python run_sweep.py --jobs 3
    python run_sweep.py --algos tccpl fixedpenalty --seeds 42
    python run_sweep.py --dry-run

Each job is a separate `train_one.py` process. Finished jobs are skipped, so an
interrupted sweep resumes where it stopped (use --force to retrain regardless).
Per-job output goes to logs/sweep/<algo>_s<seed>.log.

Concurrency is bounded by RAM, not by cores: each run holds its own replay buffer
(~3.6 GB at 200k transitions x 2291 dims in float32 -- it was 7.3 GB before the
observation dtype was fixed) plus its envs and its own CUDA context.
"""
import argparse
import json
import os
import subprocess
import sys
import time

ALGOS = ["fixedpenalty", "tccpl", "ddpg", "ppo", "cpo"]
SEEDS = [42, 43, 44]
SCENARIOS = ["cityNL", "PublicPST"]
FIRST_SEED = SEEDS[0]


def artefact(algo, seed, scenario):
    """The .zip train_one.py will write, mirroring the notebook's run_paths()."""
    suffix = "" if seed == FIRST_SEED else f"_s{seed}"
    name = {"ddpg": "ddpg_pst", "ppo": "ppo_pst", "cpo": "cpo_pst"}.get(algo, algo)
    return os.path.join("models", scenario, f"{name}_sac{suffix}.zip")


def is_complete(algo, seed, scenario):
    """A job counts as done only if its sidecar says it ran the FULL budget.

    A bare .zip is not enough: a smoke run (--timesteps) writes to the same path,
    and skipping on file existence alone would silently leave a 6000-step model
    in the results table.
    """
    zip_path = artefact(algo, seed, scenario)
    if not os.path.exists(zip_path):
        return False, "no model"
    meta = zip_path[:-4] + ".runmeta.json"
    if not os.path.exists(meta):
        return False, "no runmeta sidecar (pre-sidecar or interrupted run)"
    try:
        with open(meta, encoding="utf-8") as f:
            info = json.load(f)
    except (OSError, ValueError) as e:
        return False, f"unreadable sidecar ({e})"
    if info.get("smoke_test"):
        return False, f"smoke run ({info.get('timesteps')} steps)"
    return True, f"{info.get('timesteps')} steps, {info.get('minutes')} min"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=2,
                    help="concurrent training processes (default 2)")
    ap.add_argument("--algos", nargs="+", default=ALGOS, choices=ALGOS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--scenarios", nargs="+", default=SCENARIOS, choices=SCENARIOS)
    ap.add_argument("--threads", type=int, default=None,
                    help="threads per process (default: 16 physical cores // jobs)")
    ap.add_argument("--force", action="store_true",
                    help="retrain even if the artefact already exists")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    os.makedirs(os.path.join("logs", "sweep"), exist_ok=True)

    threads = args.threads or max(1, 16 // max(args.jobs, 1))

    queue = []
    skipped = []
    for scenario in args.scenarios:
        for seed in args.seeds:
            for algo in args.algos:
                done_already, why = is_complete(algo, seed, scenario)
                if done_already and not args.force:
                    skipped.append((algo, seed, scenario, why))
                else:
                    queue.append((algo, seed, scenario))

    print(f"{len(queue)} job(s) to run, {len(skipped)} already done, "
          f"{args.jobs} at a time, {threads} threads each")
    for algo, seed, scenario, why in skipped:
        print(f"  skip  {algo:<13} {scenario:<10} seed {seed}  ({why})")
    for algo, seed, scenario in queue:
        _, why = is_complete(algo, seed, scenario)
        print(f"  queue {algo:<13} {scenario:<10} seed {seed}  ({why})")
    if args.dry_run or not queue:
        return

    running = []          # (proc, algo, seed, log_handle, start_time)
    done, failed = [], []
    t_start = time.perf_counter()

    def launch(algo, seed, scenario):
        log_path = os.path.join("logs", "sweep", f"{scenario}_{algo}_s{seed}.log")
        fh = open(log_path, "w", encoding="utf-8", buffering=1)
        proc = subprocess.Popen(
            [sys.executable, "train_one.py", "--algo", algo,
             "--seed", str(seed), "--scenario", scenario,
             "--threads", str(threads)],
            stdout=fh, stderr=subprocess.STDOUT, cwd=here,
        )
        print(f"[{time.strftime('%H:%M:%S')}] start {algo} {scenario} seed {seed} "
              f"(pid {proc.pid}) -> {log_path}", flush=True)
        return (proc, algo, seed, scenario, fh, time.perf_counter())

    while queue or running:
        while queue and len(running) < args.jobs:
            running.append(launch(*queue.pop(0)))

        time.sleep(5)

        for entry in list(running):
            proc, algo, seed, scenario, fh, t0 = entry
            if proc.poll() is None:
                continue
            running.remove(entry)
            fh.close()
            mins = (time.perf_counter() - t0) / 60
            if proc.returncode == 0:
                done.append((algo, seed, scenario))
                print(f"[{time.strftime('%H:%M:%S')}] DONE  {algo} {scenario} seed {seed} "
                      f"in {mins:.1f} min ({len(done)}/{len(done)+len(failed)+len(queue)+len(running)} finished)",
                      flush=True)
            else:
                failed.append((algo, seed, scenario))
                print(f"[{time.strftime('%H:%M:%S')}] FAIL  {algo} {scenario} seed {seed} "
                      f"(exit {proc.returncode}) after {mins:.1f} min -- see "
                      f"logs/sweep/{scenario}_{algo}_s{seed}.log", flush=True)

    total = (time.perf_counter() - t_start) / 60
    print(f"\nsweep finished in {total:.1f} min: {len(done)} ok, {len(failed)} failed")
    for algo, seed, scenario in failed:
        print(f"  FAILED {algo} {scenario} seed {seed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
