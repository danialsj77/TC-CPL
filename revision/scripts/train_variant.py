"""Train one named variant at one seed, then evaluate it on the paired days.

    python scripts/train_variant.py VARIANT SEED [key=value ...]
    python scripts/train_variant.py heur_RR 0        # Round-Robin / CAFAP on the same days

Writes revision/runs/<scenario>/<VARIANT>_s<SEED>/{model.zip, train_meta.json, eval.json}.
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
threads = os.environ.get("EXP_THREADS", "2")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = threads
import torch
torch.set_num_threads(int(threads))
torch.set_num_interop_threads(1)
import exp_lib as X
from variants import VARIANTS

N_EVAL_DAYS = int(os.environ.get("EXP_EVAL_DAYS", 100))


def parse(v):
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    return {"True": True, "False": False}.get(v, v)


if __name__ == "__main__":
    name, seed = sys.argv[1], int(sys.argv[2])
    over = {} if name.startswith("heur_") else dict(VARIANTS[name])
    over["seed"] = seed
    for kv in sys.argv[3:]:
        k, v = kv.split("=", 1)
        over[k] = parse(v)
    spec = X.spec_of(**over)
    out = os.path.join(X.REPO, "revision", "runs", spec["scenario"], f"{name}_s{seed}")
    os.makedirs(out, exist_ok=True)
    ev_path = os.path.join(out, "eval_shield.json" if os.environ.get("EXP_SHIELD") else "eval.json")
    if name.startswith("heur_"):
        try:
            if X.same_spec(json.load(open(ev_path)).get("spec", {}), spec) and not os.environ.get("EXP_FORCE"):
                print("exists", ev_path); sys.exit(0)
        except (OSError, ValueError):
            pass
        t0 = time.time()
        days = X.evaluate_heuristic(spec, name[5:], N_EVAL_DAYS)
        json.dump({"spec": spec, "days": days, "day0_pinned": True, "minutes": (time.time() - t0) / 60},
                  open(ev_path, "w"))
        print("done", name)
        sys.exit(0)

    def _same(path):
        """An existing result counts only if it came from THIS spec (a folder can
        hold results of an earlier protocol, e.g. checked into git)."""
        try:
            with open(path) as fh:
                return X.same_spec(json.load(fh).get("spec", {}), spec)
        except (OSError, ValueError):
            return False

    meta_p = os.path.join(out, "train_meta.json")
    # the shield is an evaluation-time projection: never train with it on, so a
    # shield call needs a finished model of THIS spec
    if os.environ.get("EXP_SHIELD") and not (_same(meta_p) and os.path.exists(os.path.join(out, "model.zip"))):
        print("no trained model of this spec yet for the shield evaluation:", out)
        sys.exit(3)
    retrained = False
    if not _same(meta_p) or os.environ.get("EXP_FORCE"):
        for stale in ("eval.json", "eval_shield.json"):          # results of the old model
            if os.path.exists(os.path.join(out, stale)):
                os.remove(os.path.join(out, stale))
        retrained = True
        print("training", name, seed, json.dumps(over), flush=True)
        meta = X.train(spec, out)
        lam = meta["lambda_history"][-1] if meta["lambda_history"] else None
        print(f"trained {name} s{seed}: {meta['minutes']:.1f} min  lambda_end={lam}", flush=True)
    if retrained or not _same(ev_path) or os.environ.get("EXP_FORCE"):
        t0 = time.time()
        days = X.evaluate_learned(spec, os.path.join(out, "model.zip"), N_EVAL_DAYS)
        json.dump({"spec": spec, "days": days, "day0_pinned": True, "minutes": (time.time() - t0) / 60},
                  open(ev_path + ".tmp", "w"))
        os.replace(ev_path + ".tmp", ev_path)
        print(f"evaluated {name} s{seed} on {N_EVAL_DAYS} days in {(time.time() - t0) / 60:.1f} min")
