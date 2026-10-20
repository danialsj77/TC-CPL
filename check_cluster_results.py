"""Check everything the cluster campaign must return before it is packed and sent.

    python check_cluster_results.py                 # the 25 cityNL runs + the 70 PublicPST-S tasks
    python check_cluster_results.py --main-only     # only the 25 cityNL runs of the five algorithms
    python check_cluster_results.py --scenarios cityNL --seeds 42 43

Main runs (train_one.py; five algorithms x five seeds on cityNL): checkpoint, a
run-metadata sidecar of THIS campaign (protocol, full budget, no smoke test), for the two
TC-CPL learners the revised settings (residual radius 1, load-aware networks, demonstration
days disjoint from the evaluation days, tolerance 0) and the saved multiplier history; the
observation statistics DDPG/PPO/CPO need at evaluation; the learning-curve file.
PublicPST-S tasks (revision/stress_sweep.sh): trained model, training record and evaluation
of the variant as defined now, plus the headroom-projection evaluation where listed.
Prints ok / FAIL per item; exit code 0 = everything present and consistent.
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROTOCOL = os.environ.get("TCCPL_PROTOCOL", "2026-10")      # = train_one.PROTOCOL
ALGOS = {"tccpl": ("tccpl", "Self-tuning TC-CPL"), "fixedpenalty": ("fixedpenalty", "Fixed-price TC-CPL"),
         "ddpg": ("ddpg_pst", "DDPG-PST"), "ppo": ("ppo_pst", "PPO-PST"), "cpo": ("cpo_pst", "CPO")}

ap = argparse.ArgumentParser()
ap.add_argument("--scenarios", nargs="+", default=["cityNL"])
ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
ap.add_argument("--algos", nargs="+", default=list(ALGOS), choices=list(ALGOS))
ap.add_argument("--budget", type=int, default=1_000_000)
ap.add_argument("--rho", type=float, default=1.0)
ap.add_argument("--tau", type=float, default=0.0, help="overload tolerance expected of the self-tuning runs")
ap.add_argument("--updates", type=int, default=99, help="dual updates expected (budget / 10,000 - 1)")
ap.add_argument("--main-only", action="store_true", help="skip the PublicPST-S tasks")
args = ap.parse_args()
os.chdir(HERE)

bad = 0


def report(tag, problems):
    global bad
    bad += bool(problems)
    print(f"{'FAIL' if problems else 'ok  '} {tag}" + (f": {'; '.join(problems)}" if problems else ""))


print("== main runs (train_one.py) ==")
for scen in args.scenarios:
    for algo in args.algos:
        name, label = ALGOS[algo]
        for seed in args.seeds:
            suf = "" if seed == 42 else f"_s{seed}"
            problems = []
            if not os.path.exists(f"models/{scen}/{name}_sac{suf}.zip"):
                problems.append("no checkpoint")
            try:
                m = json.load(open(f"models/{scen}/{name}_sac{suf}.runmeta.json"))
                if m.get("protocol") != PROTOCOL: problems.append(f"not from this campaign (protocol {m.get('protocol')})")
                if m.get("smoke_test"): problems.append("smoke run")
                if int(m.get("timesteps", -1)) != args.budget: problems.append(f"budget {m.get('timesteps')}")
                if algo in ("tccpl", "fixedpenalty"):
                    if not m.get("tr_load_aware"): problems.append("networks of the first submission")
                    if abs(float(m.get("rho", 0.25)) - args.rho) > 1e-9: problems.append(f"rho {m.get('rho', 0.25)}")
                    if int(m.get("demo_seed_offset", 0)) < 5100: problems.append("demonstration days overlap the evaluation days")
                if algo == "tccpl" and abs(float(m.get("train_target_overload", 0.0)) - args.tau) > 1e-9:
                    problems.append(f"tolerance {m.get('train_target_overload')}")
            except (OSError, ValueError):
                problems.append("no runmeta")
            if algo in ("ddpg", "ppo", "cpo") and not os.path.exists(f"models/{scen}/{name}{suf}/vec_normalize.pkl"):
                problems.append("no vec_normalize.pkl (needed to evaluate)")
            if algo != "cpo" and not os.path.exists(f"logs/{scen}/{name}{suf}/evaluations.npz"):
                problems.append("no learning-curve file logs/.../evaluations.npz")
            if algo in ("tccpl", "fixedpenalty"):
                try:
                    h = json.load(open(f"results/{scen}/lambda_history_{algo}{suf}.json"))
                    if h.get("tr_load_aware") is not True or "kp" not in h:
                        problems.append("multiplier history from an earlier run (git copy)")
                    n = len(h.get("lambda_history", []))
                    if n != args.updates: problems.append(f"{n} dual updates (expected {args.updates})")
                    if len(h.get("update_steps", [])) != n: problems.append("update steps not recorded")
                    if algo == "fixedpenalty" and any(abs(x[0] - 100.0) > 1e-6 for x in h.get("lambda_history", [])):
                        problems.append("fixed-price multiplier moved")
                    if algo == "tccpl" and n and max(x[0] for x in h["lambda_history"]) >= h.get("lambda_max", 500.0) - 1e-6:
                        print(f"     note: {scen} {label} seed {seed} reached the cap at some update")
                except (OSError, ValueError):
                    problems.append("no multiplier history")
            report(f"{scen:9s} {label:19s} seed {seed}", problems)

if not args.main_only:
    print("\n== PublicPST-S tasks (revision/stress_sweep.sh) ==")
    sys.path.insert(0, os.path.join(HERE, "revision", "scripts"))
    from provenance import stress_run_is_current, stress_shield_is_current
    lines = [l.split() for l in open("revision/stress_tasks.txt") if l.strip() and not l.startswith("#")]
    for parts in lines:
        var, seed, shield = parts[0], int(parts[1]), len(parts) > 2
        d = f"revision/runs/PublicPST_stress/{var}_s{seed}"
        problems = []
        need = ["eval.json"] + ([] if var.startswith("heur_") else ["model.zip", "train_meta.json"])
        need += ["eval_shield.json"] if shield else []
        problems += [f"no {f}" for f in need if not os.path.exists(os.path.join(d, f))]
        if not problems and not stress_run_is_current(d):
            problems.append("results do not match the current definition of the variant")
        elif not problems and shield and not stress_shield_is_current(d):
            problems.append("headroom-projection evaluation is from an earlier run")
        report(f"{var:18s} seed {seed}", problems)

print(f"\n{bad} item(s) with problems")
sys.exit(1 if bad else 0)
