"""Write revision/jobs.json, the ordered job list of the revision campaign.

Order = priority: demonstrations -> core stress-test training interleaved with
the cityNL re-evaluation of the submitted checkpoints -> sensitivity training
interleaved with the cityNL observation-perturbation tests.
"""
import json, os, itertools

REV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.expanduser("~/.venvs/tccpl311/bin/python")
STRESS = "budget=200000 update_freq=3584"          # 200k steps, one dual update per 32 episodes

CORE = ["tccpl", "fixed_100_50", "fixed_500_5", "tau_2", "lmax_2000",
        "no_demos", "no_residual", "no_gridblock", "flat_encoder", "inc_cafap"]
SENS = ["fixed_10_5", "fixed_10_50", "fixed_100_5", "fixed_500_50", "fixed_2000_50",
        "lmax_100", "tau_0p5", "pi_kp0", "pi_ki0p5", "pi_ki8", "no_m2",
        "inc_cafap_rho1", "inc_rr_rho1", "margin0", "pi_al", "tau_2_lmax2000"]
CORE_SEEDS, SENS_SEEDS = [42, 43, 44], [42, 43]
DEMO_KEYS = {"tccpl": CORE_SEEDS, "no_residual": CORE_SEEDS, "no_gridblock": CORE_SEEDS,
             "inc_cafap": CORE_SEEDS, "inc_cafap_rho1": SENS_SEEDS, "inc_rr_rho1": SENS_SEEDS}
PERTURB = ["status_flip:0.1", "status_flip:0.3", "status_hide:0.1", "status_hide:0.3",
           "status_delay:1", "nev_bias:0.2", "nev_bias:-0.2", "rating_bias:0.05",
           "rating_bias:-0.05", "dr_notice:0", "dr_notice:2", "nev_persist:1"]

_dev = itertools.cycle(["mps", "cpu"])


def train_job(v, s):
    dev = next(_dev)
    return {"name": f"train_{v}_s{s}",
            "cmd": f"{PY} scripts/train_variant.py {v} {s} {STRESS} device={dev}",
            "done_if": f"runs/PublicPST_stress/{v}_s{s}/eval.json",
            "env": {"EXP_THREADS": 1 if dev == "mps" else 2}}


def citynl_job(label, s, pert=None, days=100):
    tag = "" if pert is None else "_" + pert.replace(":", "")
    arg = "" if pert is None else f" {pert}"
    return {"name": f"citynl_{label}_s{s}{tag}_n{days}",
            "cmd": f"{PY} scripts/eval_citynl.py {label} {s}{arg} --days {days}",
            "done_if": f"runs/cityNL_eval/{label}_s{s}{tag}_n{days}.json",
            "env": {"EXP_THREADS": 1}}


def interleave(a, b):
    out = []
    for x, y in itertools.zip_longest(a, b):
        out += [z for z in (x, y) if z is not None]
    return out


jobs = []
for v, seeds in DEMO_KEYS.items():
    for s in seeds:
        jobs.append({"name": f"demos_{v}_s{s}", "cmd": f"{PY} scripts/collect_demos.py {v} {s}",
                     "done_if": f"logs/jobs/demos_{v}_s{s}.done", "env": {"EXP_THREADS": 1}})
        jobs[-1]["cmd"] += f" && touch logs/jobs/demos_{v}_s{s}.done"
for h in ("RR", "CAFAP"):
    jobs.append({"name": f"heur_{h}", "cmd": f"{PY} scripts/train_variant.py heur_{h} 0",
                 "done_if": f"runs/PublicPST_stress/heur_{h}_s0/eval.json", "env": {"EXP_THREADS": 1}})

core_train = [train_job(v, s) for s in CORE_SEEDS for v in CORE]
city_base = [citynl_job(l, s) for l in ("tccpl", "fixedpenalty", "ddpg_pst", "ppo_pst", "cpo_pst")
             for s in (42, 43, 44, 45, 46)] + [citynl_job(h, 0) for h in ("RR", "CAFAP")]
jobs += interleave(core_train, city_base)

sens_train = [train_job(v, s) for s in SENS_SEEDS for v in SENS]
city_pert = [citynl_job(l, s, p, 50) for p in PERTURB for s in (42, 43, 44)
             for l in ("tccpl", "fixedpenalty")]
jobs += interleave(sens_train, city_pert)

# held-out days disjoint from every demonstration seed of the submitted runs
# (those used simulator seeds 42..~1100, which include 500..599)
for l, seeds in (("tccpl", (42, 43, 44, 45, 46)), ("fixedpenalty", (42, 43, 44, 45, 46)),
                 ("RR", (0,)), ("CAFAP", (0,))):
    for s in seeds:
        jobs.append({"name": f"citynl_{l}_s{s}_heldout", "env": {"EXP_THREADS": 1},
                     "cmd": f"{PY} scripts/eval_citynl.py {l} {s} --days 100 --offset 5000",
                     "done_if": f"runs/cityNL_eval/{l}_s{s}_n100_o5000.json"})

# headroom shield on PublicPST-S (evaluation only, after training)
for v, seeds in (("heur_RR", (0,)), ("heur_CAFAP", (0,)), ("tccpl", CORE_SEEDS),
                 ("fixed_100_50", CORE_SEEDS), ("no_residual", CORE_SEEDS), ("tau_2", CORE_SEEDS)):
    for s in seeds:
        jobs.append({"name": f"shield_{v}_s{s}", "env": {"EXP_SHIELD": 1, "EXP_THREADS": 1},
                     "cmd": f"{PY} scripts/train_variant.py {v} {s} {STRESS}",
                     "done_if": f"runs/PublicPST_stress/{v}_s{s}/eval_shield.json"})

json.dump(jobs, open(os.path.join(REV, "jobs.json"), "w"), indent=1)
print(len(jobs), "jobs:", sum(j["name"].startswith("train_") for j in jobs), "training,",
      sum(j["name"].startswith("citynl_") for j in jobs), "cityNL evaluations")
