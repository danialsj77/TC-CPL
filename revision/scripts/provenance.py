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


def _expected_stress_spec(run_dir):
    """(spec the run must have been made with, is_heuristic) or (None, None) for a folder that
    is not a run of the current campaign. Imports the learner stack (torch) only here."""
    import re, sys
    name = os.path.basename(os.path.normpath(run_dir))
    m = re.match(r"(.+)_s(\d+)$", name)
    if not m:
        return None, None
    var, seed = m.group(1), int(m.group(2))
    rev = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if rev not in sys.path:
        sys.path.insert(0, rev)
    import exp_lib as X
    from variants import VARIANTS, CAMPAIGN
    if var.startswith("heur_"):
        return X.spec_of(seed=seed, **CAMPAIGN), True
    if var not in VARIANTS:
        return None, None
    return X.spec_of(**VARIANTS[var], seed=seed, **CAMPAIGN), False


def _spec_in(path):
    try:
        with open(path) as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return None
    return d.get("spec")


def stress_run_is_current(run_dir):
    """True if a PublicPST-S run folder <variant>_s<seed> holds results of the variant as
    defined NOW in variants.py with the campaign settings: training record and evaluation
    for a learned variant, evaluation for a heuristic. Folders of earlier protocols or of
    retired variants are ignored by every consumer."""
    want, heur = _expected_stress_spec(run_dir)
    if want is None:
        return False
    from exp_lib import same_spec
    files = ["eval.json"] if heur else ["train_meta.json", "eval.json"]
    for f in files:
        got = _spec_in(os.path.join(run_dir, f))
        if got is None or not same_spec(got, want):
            return False
    return True


def stress_shield_is_current(run_dir):
    """True if the headroom-projection evaluation of the folder belongs to the current run."""
    want, _ = _expected_stress_spec(run_dir)
    if want is None:
        return False
    from exp_lib import same_spec
    got = _spec_in(os.path.join(run_dir, "eval_shield.json"))
    return got is not None and same_spec(got, want) and stress_run_is_current(run_dir)
