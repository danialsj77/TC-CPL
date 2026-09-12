"""Train ONE controller at ONE seed on ONE scenario, as a standalone process.

    python train_one.py --algo tccpl --seed 42 --scenario cityNL

Exists so several runs can proceed concurrently: a Jupyter kernel is a single
process, and SubprocVecEnv cannot be used here (each worker re-imports torch and
overruns the Windows paging file). Separate top-level processes have neither
problem, and measurements show the training loop is not GPU-compute-bound, so
concurrent runs have real headroom.

Artefacts land in exactly the paths the notebook uses (models/<name>_sac<suffix>
.zip, models/<name><suffix>/, logs/<name><suffix>/), so section 7 of the notebook
picks them up with no changes.
"""
import argparse
import json
import os
import sys
import time

# algo -> (run_paths name, CONFIG key holding its training budget)
JOB_INFO = {
    "fixedpenalty": ("fixedpenalty", "total_timesteps_fixed"),
    "tccpl":        ("tccpl",        "total_timesteps_tccpl"),
    "ddpg":         ("ddpg_pst",     "total_timesteps_ddpg"),
    "ppo":          ("ppo_pst",      "total_timesteps_ppo"),
    "cpo":          ("cpo_pst",      "total_timesteps_cpo"),
}


def meta_path(zip_path):
    """Sidecar recording how a saved model was produced."""
    return zip_path[:-4] + ".runmeta.json" if zip_path.endswith(".zip") \
        else zip_path + ".runmeta.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", required=True,
                    choices=["fixedpenalty", "tccpl", "ddpg", "ppo", "cpo"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--scenario", default="cityNL",
                    choices=["cityNL", "PublicPST"])
    ap.add_argument("--threads", type=int, default=4,
                    help="BLAS/torch threads for THIS process. Keep the product "
                         "over concurrent runs at or below the 16 physical cores.")
    ap.add_argument("--timesteps", type=int, default=None,
                    help="override the training budget (all total_timesteps_* "
                         "keys). For smoke tests; omit for the real run.")
    ap.add_argument("--offline-transitions", type=int, default=None,
                    help="override CONFIG['offline_transitions'] (M2 demos).")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    sys.path.insert(0, here)

    # Must precede every numpy/torch import. Without a cap each process sizes its
    # BLAS pool to all 32 logical cores, and concurrent runs thrash or abort with
    # "OpenBLAS: Memory allocation still failed after 10 retries".
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(args.threads)

    # Read by the notebook's CONFIG cell, so both must be set before the import.
    os.environ["TCCPL_SEED"] = str(args.seed)
    os.environ["TCCPL_SCENARIO"] = args.scenario

    t0 = time.perf_counter()
    print(f"=== {args.algo} | {args.scenario} | seed {args.seed} | "
          f"{args.threads} threads ===", flush=True)

    import torch as th
    th.set_num_threads(args.threads)

    import tccpl_runtime as rt            # importing runs the definition cells

    if rt.SEED != args.seed:
        raise SystemExit(f"seed mismatch: runtime came up at {rt.SEED}, "
                         f"expected {args.seed}")

    # The training cells read CONFIG at call time, so overriding it here works.
    if args.timesteps is not None:
        for key in [k for k in rt.CONFIG if k.startswith("total_timesteps_")]:
            rt.CONFIG[key] = args.timesteps
        print(f"!! SMOKE TEST: training budget overridden to "
              f"{args.timesteps} timesteps", flush=True)
    if args.offline_transitions is not None:
        rt.CONFIG["offline_transitions"] = args.offline_transitions
        print(f"!! offline_transitions overridden to "
              f"{args.offline_transitions}", flush=True)

    name, budget_key = JOB_INFO[args.algo]
    budget = rt.CONFIG[budget_key]

    rt.TRAIN_JOBS[args.algo]()

    dt = time.perf_counter() - t0

    # Sidecar so a short smoke run can never be mistaken for a finished run:
    # run_sweep.py skips a job only when this records the full budget.
    zip_path = rt.run_paths(name)["zip"] + ".zip"
    try:
        with open(meta_path(zip_path), "w", encoding="utf-8") as f:
            json.dump({
                "algo": args.algo,
                "scenario": args.scenario,
                "seed": args.seed,
                "timesteps": budget,
                "smoke_test": args.timesteps is not None,
                "offline_transitions": rt.CONFIG.get("offline_transitions"),
                "config_file": rt.CONFIG["config_file"],
                "target_sat": rt.CONFIG["target_sat"],
                "train_sat_floor": rt.CONFIG.get("train_sat_floor"),
                "ov_margin": rt.CONFIG.get("ov_margin"),
                "minutes": round(dt / 60, 2),
                "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=2)
    except OSError as e:
        print(f"  ! could not write run metadata: {e}", flush=True)

    print(f"=== done: {args.algo} seed {args.seed} in {dt/60:.1f} min "
          f"({budget} timesteps) ===", flush=True)


if __name__ == "__main__":
    main()
