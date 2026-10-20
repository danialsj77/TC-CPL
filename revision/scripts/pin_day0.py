"""Re-evaluate the first day of PublicPST-S evaluations made before the first
day was pinned (exp_lib.pin_first_day) and replace it in place.

With tr_seed -1 the household load, PV and demand-response event of a day are
drawn from the global numpy state when the env is built. Days 2..N already
coincide across controllers (that state is fixed by the previous day's reset
seed); the first day depended on the process history. This rewrites days[0]
of every runs/PublicPST_stress/*/eval*.json without "day0_pinned" and keeps the
old record under "day0_unpinned".

    python scripts/pin_day0.py [--dry]
"""
import os, sys, json, glob

threads = os.environ.get("EXP_THREADS", "1")
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, threads)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import exp_lib as X  # noqa: E402

ROOT = os.path.join(X.REPO, "revision", "runs", "PublicPST_stress")


def first_day(path, rec):
    run = os.path.dirname(path)
    name = os.path.basename(run).rsplit("_s", 1)[0]
    spec = rec["spec"]
    shield = os.path.basename(path) == "eval_shield.json"
    if shield:
        os.environ["EXP_SHIELD"] = "1"
    else:
        os.environ.pop("EXP_SHIELD", None)
    if name.startswith("heur_"):
        return X.evaluate_heuristic(spec, name[5:], 1)[0]
    return X.evaluate_learned(spec, os.path.join(run, "model.zip"), 1)[0]


if __name__ == "__main__":
    dry = "--dry" in sys.argv
    todo = []
    for p in sorted(glob.glob(os.path.join(ROOT, "*", "eval*.json"))):
        if p.endswith(".tmp"):
            continue
        rec = json.load(open(p))
        if not rec.get("day0_pinned"):
            todo.append((p, rec))
    print(len(todo), "evaluations to fix")
    for p, rec in todo:
        if dry:
            print("  ", os.path.relpath(p, ROOT))
            continue
        new = first_day(p, rec)
        old = rec["days"][0]
        rec["day0_unpinned"] = old
        rec["days"][0] = new
        rec["day0_pinned"] = True
        json.dump(rec, open(p + ".tmp", "w"))
        os.replace(p + ".tmp", p)
        print(f"  {os.path.relpath(p, ROOT)}: overload {old['total_transformer_overload']:.2f}"
              f" -> {new['total_transformer_overload']:.2f} kWh", flush=True)
