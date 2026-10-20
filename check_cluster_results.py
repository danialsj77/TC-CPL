"""Check the TC-CPL runs that come back from the cluster before they are used.

    python check_cluster_results.py                      # both scenarios, five seeds
    python check_cluster_results.py --scenarios cityNL --seeds 42 43

For every Self-tuning (tccpl) and Fixed-price (fixedpenalty) TC-CPL run it verifies the
checkpoint, the run-metadata sidecar (full budget, no smoke test, load-aware networks,
residual radius, demonstration days disjoint from the evaluation days) and the saved
multiplier history (one entry per dual update, recorded update steps). Exit code 0 = all fine.
"""
import argparse, json, os, sys

ALGOS = {"tccpl": "Self-tuning TC-CPL", "fixedpenalty": "Fixed-price TC-CPL"}
ap = argparse.ArgumentParser()
ap.add_argument("--scenarios", nargs="+", default=["cityNL", "PublicPST"])
ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
ap.add_argument("--budget", type=int, default=1_000_000)
ap.add_argument("--rho", type=float, default=1.0)
ap.add_argument("--tau", type=float, default=0.0, help="overload tolerance expected of the self-tuning runs")
ap.add_argument("--updates", type=int, default=99, help="dual updates expected (budget / 10,000 - 1)")
args = ap.parse_args()

bad = 0
for scen in args.scenarios:
    for algo, label in ALGOS.items():
        for seed in args.seeds:
            suf = "" if seed == 42 else f"_s{seed}"
            tag = f"{scen:9s} {label:19s} seed {seed}"
            problems = []
            zip_p = f"models/{scen}/{algo}_sac{suf}.zip"
            meta_p = f"models/{scen}/{algo}_sac{suf}.runmeta.json"
            hist_p = f"results/{scen}/lambda_history_{algo}{suf}.json"
            if not os.path.exists(zip_p):
                problems.append("no checkpoint")
            try:
                m = json.load(open(meta_p))
                if m.get("smoke_test"): problems.append("smoke run")
                if int(m.get("timesteps", -1)) != args.budget: problems.append(f"budget {m.get('timesteps')}")
                if not m.get("tr_load_aware"): problems.append("networks of the first submission")
                if abs(float(m.get("rho", 0.25)) - args.rho) > 1e-9: problems.append(f"rho {m.get('rho', 0.25)}")
                if int(m.get("demo_seed_offset", 0)) < 5100: problems.append("demonstration days overlap the evaluation days")
                if algo == "tccpl" and abs(float(m.get("train_target_overload", 0.0)) - args.tau) > 1e-9:
                    problems.append(f"tolerance {m.get('train_target_overload')}")
            except (OSError, ValueError):
                problems.append("no runmeta")
            try:
                h = json.load(open(hist_p))
                n = len(h.get("lambda_history", []))
                if n != args.updates: problems.append(f"{n} dual updates (expected {args.updates})")
                if len(h.get("update_steps", [])) != n: problems.append("update steps not recorded")
                if algo == "fixedpenalty" and any(abs(x[0] - 100.0) > 1e-6 for x in h.get("lambda_history", [])):
                    problems.append("fixed-price multiplier moved")
            except (OSError, ValueError):
                problems.append("no multiplier history")
            if problems:
                bad += 1
            print(f"{'FAIL' if problems else 'ok  '} {tag}" + (f": {'; '.join(problems)}" if problems else ""))
print(f"\n{bad} run(s) with problems")
sys.exit(1 if bad else 0)
