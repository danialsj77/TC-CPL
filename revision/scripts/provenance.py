"""Which checkpoint an evaluation output was made from (no heavy imports).

Shared by eval_citynl.py, which writes the stamp into every output, and
collect_results.py, which uses an output only if its stamp matches the checkpoint
now in models/. Outputs of an earlier model are therefore never mixed into the tables.
"""
import json, os

NAMES = {"ddpg": "ddpg_pst", "ppo": "ppo_pst", "cpo": "cpo_pst"}


def checkpoint_stamp(repo, scen, label, seed):
    """None for the rule-based controllers, else the identity of the checkpoint."""
    if label in ("RR", "CAFAP"):
        return None
    name = NAMES.get(label, label)
    suf = "" if int(seed) == 42 else f"_s{seed}"
    zp = os.path.join(repo, "models", scen, f"{name}_sac{suf}.zip")
    try:
        with open(zp[:-4] + ".runmeta.json") as fh:
            m = json.load(fh)
    except (OSError, ValueError):
        return {"runmeta": None, "zip_mtime": os.path.getmtime(zp) if os.path.exists(zp) else None}
    return {"protocol": m.get("protocol"), "finished": m.get("finished"),
            "rho": float(m.get("rho", 0.25)), "tr_load_aware": bool(m.get("tr_load_aware", False))}


def output_is_current(path, repo, scen, label, seed):
    """True if the evaluation output at `path` was made from the checkpoint now in models/.
    Outputs written before stamps existed count as current only while the checkpoint is
    still one of the first submission (its sidecar has no protocol)."""
    try:
        with open(path) as fh:
            rec = json.load(fh).get("checkpoint", "missing")
    except (OSError, ValueError):
        return False
    now = checkpoint_stamp(repo, scen, label, seed)
    if rec == "missing":
        return now is None or now.get("protocol") is None
    return rec == now


def stress_run_is_current(run_dir):
    """True if a PublicPST-S run folder <variant>_s<seed> holds results of the variant
    as defined NOW in variants.py with the campaign settings (heuristics always count).
    Folders of earlier protocols or retired variants are ignored by every consumer."""
    import re, sys
    name = os.path.basename(os.path.normpath(run_dir))
    m = re.match(r"(.+)_s(\d+)$", name)
    if not m:
        return False
    var, seed = m.group(1), int(m.group(2))
    if var.startswith("heur_"):
        return os.path.exists(os.path.join(run_dir, "eval.json"))
    rev = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if rev not in sys.path:
        sys.path.insert(0, rev)
    import exp_lib as X                      # heavy (torch, simulator): imported only here
    from variants import VARIANTS, CAMPAIGN
    if var not in VARIANTS:
        return False
    want = X.spec_of(**VARIANTS[var], seed=seed, **CAMPAIGN)
    try:
        with open(os.path.join(run_dir, "train_meta.json")) as fh:
            got_train = json.load(fh)["spec"]
        with open(os.path.join(run_dir, "eval.json")) as fh:
            got_eval = json.load(fh).get("spec", {})
    except (OSError, ValueError, KeyError):
        return False
    return X.same_spec(got_train, want) and X.same_spec(got_eval, want)
