"""Pre-collect the M2 demonstrations of a variant/seed into revision/cache/ (shared by
every variant whose demonstration key matches), so concurrent training jobs do not
collect the same data twice."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("OMP_NUM_THREADS", "1")
import exp_lib as X
from variants import VARIANTS

if __name__ == "__main__":
    name, seed = sys.argv[1], int(sys.argv[2])
    spec = X.spec_of(**dict(VARIANTS[name], seed=seed))
    os.chdir(X.REPO)
    d = X.collect_demos(spec, X.make_costs(spec), os.path.join(X.REPO, "revision", "cache"))
    print("demos", name, seed, len(d["rewards"]), "mean c_ov", float(d["c_ov"].mean()))
