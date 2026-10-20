"""Run the TC-CPL campaign with a bounded number of concurrent processes, on one
machine or split over several servers that do not share a disk.

    python run_sweep.py --dry-run                     # state of all 95 jobs
    python run_sweep.py --jobs 4                      # everything on ONE machine, 4 at a time
    python run_sweep.py --plan 7 --jobs 2             # how 7 servers would share the work
    python run_sweep.py --part 3/7 --jobs 2           # on server 3 of 7: run its share
    python run_sweep.py --part 3/7 --pack part3.tgz   # on server 3: pack ONLY what it produced
    python run_sweep.py --algos tccpl fixedpenalty --seeds 42 --what main

The campaign: 25 main jobs on cityNL (5 algorithms x 5 seeds, train_one.py, 1,000,000 steps,
GPU) and 70 PublicPST-S jobs (revision/stress_tasks.txt via revision/stress_sweep.sh, 200,000
steps, CPU). PublicPST-S replaces the unstressed PublicPST of the first submission, whose
transformer never binds; `--scenarios cityNL PublicPST` still runs PublicPST as well. --part K/N gives server K the same balanced share on every server, as
long as all servers use the same options apart from K (shares are balanced by the job times
of the first campaign; the longest jobs start first). A job is "finished" only when its
results come from THIS code version at the full budget (see is_complete), so a folder with
models of an earlier campaign is retrained, never reused; --force retrains regardless.

Merging the servers: on each server `--part K/N --pack partK.tgz` packs exactly the files of
its own jobs (a plain tar of whole folders would also carry the earlier copies of other
servers' results that come with a git clone, and unpacking those last would overwrite new
results). Unpack every partK.tgz into one fresh clone, then run check_cluster_results.py.

Per-job output goes to logs/sweep/<scenario>_<algo>_s<seed>.log and
logs/sweep/stress_<variant>_s<seed>.log. Concurrency is bounded by RAM, not by cores:
about 8 GB per main job (DDPG on cityNL about 18 GB), about 3 GB per PublicPST-S job.
On a SLURM / Grid Engine cluster with a shared disk the array scripts slurm_sweep.sh /
tsubame_sweep.sh and revision/stress_sweep.sh do the same with one task per job.
"""
import argparse
import json
import os
import subprocess
import sys
import tarfile
import time

ALGOS = ["fixedpenalty", "tccpl", "ddpg", "ppo", "cpo"]
SEEDS = [int(s) for s in os.environ.get("TCCPL_SEEDS", "42,43,44,45,46").split(",")]
SCENARIOS = ["cityNL"]              # the campaign (PublicPST-S replaces PublicPST, see revision/)
ALL_SCENARIOS = ["cityNL", "PublicPST"]
FIRST_SEED = 42                     # its artefacts carry no _s<seed> suffix
BUDGET = int(os.environ.get("TCCPL_TOTAL_TIMESTEPS", 1_000_000))
PROTOCOL = os.environ.get("TCCPL_PROTOCOL", "2026-10")   # = train_one.PROTOCOL
HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {"ddpg": "ddpg_pst", "ppo": "ppo_pst", "cpo": "cpo_pst"}

# Hours of one job on one GPU in the first campaign (runmeta minutes); used only to
# balance --part and to order the queue, never to decide anything else.
EST_HOURS = {("cityNL", "fixedpenalty"): 29.5, ("cityNL", "tccpl"): 25.0, ("cityNL", "ddpg"): 8.4,
             ("cityNL", "ppo"): 6.0, ("cityNL", "cpo"): 5.7,
             ("PublicPST", "fixedpenalty"): 8.6, ("PublicPST", "tccpl"): 9.0, ("PublicPST", "ddpg"): 1.5,
             ("PublicPST", "ppo"): 0.3, ("PublicPST", "cpo"): 0.3}
STRESS_HOURS, HEUR_HOURS = 1.5, 0.1


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
    if info.get("protocol") != PROTOCOL:
        return False, f"earlier campaign (protocol {info.get('protocol')!r}, current {PROTOCOL!r})"
    if int(info.get("timesteps", -1)) != BUDGET:
        return False, (f"budget mismatch ({info.get('timesteps')} steps on disk, "
                       f"protocol is {BUDGET})")
    if algo in ("tccpl", "fixedpenalty"):
        want_la = bool(int(os.environ.get("TCCPL_TR_LOAD_AWARE", "1")))
        want_tau = float(os.environ.get("TCCPL_TRAIN_TARGET_OVERLOAD", 0.0))
        if bool(info.get("tr_load_aware", False)) != want_la:
            return False, f"architecture mismatch (tr_load_aware={info.get('tr_load_aware', False)} on disk)"
        want_off = int(os.environ.get("TCCPL_DEMO_SEED_OFFSET", 100_000))
        want_rho = float(os.environ.get("TCCPL_RHO", 1.0))
        if abs(float(info.get("rho", 0.25)) - want_rho) > 1e-9:
            return False, f"residual radius mismatch (rho={info.get('rho', 0.25)} on disk)"
        if int(info.get("demo_seed_offset", 0)) != want_off:
            return False, f"demonstration days mismatch (offset {info.get('demo_seed_offset', 0)} on disk)"
        # the tolerance acts only through the multiplier update, so it matters
        # for the self-tuning learner, not for the frozen-price ablation
        if algo == "tccpl" and abs(float(info.get("train_target_overload", 0.0)) - want_tau) > 1e-9:
            return False, f"tolerance mismatch ({info.get('train_target_overload', 0.0)} on disk)"
    return True, f"{info.get('timesteps')} steps, {info.get('minutes')} min"




# ── the job list ─────────────────────────────────────────────────────────────
class Job:
    """One main job (train_one.py) or one PublicPST-S task (revision/stress_sweep.sh)."""

    def __init__(self, kind, **kw):
        self.kind = kind
        self.__dict__.update(kw)
        if kind == "main":
            self.key = f"{self.scenario}_{self.algo}_s{self.seed}"
            self.label = f"{self.algo:<13} {self.scenario:<10} seed {self.seed}"
            self.hours = EST_HOURS.get((self.scenario, self.algo), 5.0)
            self.log = os.path.join("logs", "sweep", f"{self.key}.log")
        else:
            self.key = f"stress_{self.variant}_s{self.seed}"
            self.label = f"PublicPST-S   {self.variant:<18} seed {self.seed}" + (" +projection" if self.shield else "")
            self.hours = HEUR_HOURS if self.variant.startswith("heur_") else STRESS_HOURS
            self.log = os.path.join("logs", "sweep", f"{self.key}.log")

    def status(self):
        if self.kind == "main":
            return is_complete(self.algo, self.seed, self.scenario)
        d = os.path.join("revision", "runs", "PublicPST_stress", f"{self.variant}_s{self.seed}")
        if not os.path.isdir(d):
            return False, "no run"
        sys.path.insert(0, os.path.join(HERE, "revision", "scripts"))
        from provenance import stress_run_is_current, stress_shield_is_current   # imports the learner stack once
        if not stress_run_is_current(d):
            return False, "no result of the current definition"
        if self.shield and not stress_shield_is_current(d):
            return False, "headroom-projection evaluation missing"
        return True, "current"

    def artefacts(self):
        """The files this job produces: exactly what --pack puts into a server's archive."""
        if self.kind == "stress":
            return [os.path.join("revision", "runs", "PublicPST_stress", f"{self.variant}_s{self.seed}"), self.log]
        name = NAMES.get(self.algo, self.algo)
        suf = "" if self.seed == FIRST_SEED else f"_s{self.seed}"
        out = [f"models/{self.scenario}/{name}_sac{suf}.zip", f"models/{self.scenario}/{name}_sac{suf}.runmeta.json",
               f"models/{self.scenario}/{name}{suf}", f"logs/{self.scenario}/{name}{suf}", self.log]
        if self.algo in ("tccpl", "fixedpenalty"):
            out.append(f"results/{self.scenario}/lambda_history_{self.algo}{suf}.json")
        return out

    def command(self, threads, force):
        env = dict(os.environ)
        if self.kind == "main":
            return [sys.executable, "train_one.py", "--algo", self.algo, "--seed", str(self.seed),
                    "--scenario", self.scenario, "--threads", str(threads)], env
        # stress_sweep.sh picks its line from SGE_TASK_ID; a SLURM task id inherited from
        # an enclosing allocation would take precedence, so drop it
        for v in ("SLURM_ARRAY_TASK_ID", "SLURM_SUBMIT_DIR"):
            env.pop(v, None)
        env.update(SGE_TASK_ID=str(self.task), THREADS=str(threads), CUDA_VISIBLE_DEVICES="")
        if force:
            env["EXP_FORCE"] = "1"
        return ["bash", os.path.join("revision", "stress_sweep.sh")], env


def all_jobs(args):
    jobs = []
    if args.what in ("all", "main"):
        jobs += [Job("main", algo=a, seed=s, scenario=sc)
                 for sc in args.scenarios for s in args.seeds for a in args.algos]
    if args.what in ("all", "stress"):
        lines = [l.split() for l in open(os.path.join(HERE, "revision", "stress_tasks.txt"))
                 if not l.startswith("#")]            # the same lines stress_sweep.sh numbers
        jobs += [Job("stress", variant=p[0], seed=int(p[1]), shield=len(p) > 2, task=t)
                 for t, p in enumerate(lines, start=1) if p]
    return jobs


def split(jobs, n):
    """Balanced, deterministic shares: longest job first to the least-loaded server.
    Every server computes the same split from the same options."""
    parts, load = [[] for _ in range(n)], [0.0] * n
    for j in sorted(jobs, key=lambda j: (-j.hours, j.key)):
        k = min(range(n), key=lambda i: (load[i], i))
        parts[k].append(j)
        load[k] += j.hours
    return parts


def makespan(jobs, slots):
    """Estimated wall time of a share run longest-first with `slots` jobs at a time."""
    free = [0.0] * max(slots, 1)
    for j in sorted(jobs, key=lambda j: -j.hours):
        i = free.index(min(free))
        free[i] += j.hours
    return max(free) if jobs else 0.0


def pack(jobs, path):
    have, missing = [], []
    for j in jobs:
        done, why = j.status()
        (have if done else missing).append(j)
    with tarfile.open(path, "w:gz") as tar:
        for j in have:
            for p in j.artefacts():
                if os.path.exists(p):
                    tar.add(p)
    print(f"packed {len(have)} finished job(s) into {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    for j in missing:
        print(f"  NOT packed (not finished): {j.label}  ({j.status()[1]})")
    return 0 if not missing else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=2,
                    help="jobs running at the same time on THIS machine (default 2)")
    ap.add_argument("--what", choices=["all", "main", "stress"], default="all",
                    help="the main cityNL jobs, the PublicPST-S jobs, or both (default)")
    ap.add_argument("--algos", nargs="+", default=ALGOS, choices=ALGOS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--scenarios", nargs="+", default=SCENARIOS, choices=ALL_SCENARIOS,
                    help="main-job scenarios (default: cityNL; PublicPST is optional)")
    ap.add_argument("--part", metavar="K/N", default=None,
                    help="run only share K of N (servers without a shared disk; same options on every server)")
    ap.add_argument("--plan", type=int, metavar="N", default=None,
                    help="print how N servers would share the jobs, and the estimated hours at --jobs")
    ap.add_argument("--pack", metavar="FILE.tgz", default=None,
                    help="pack the files of the finished jobs of this share (or of all jobs) and exit")
    ap.add_argument("--gpus", default=None,
                    help="comma-separated GPU ids of this machine, e.g. 0,1,2,3: main jobs are spread "
                         "over them (default: every job uses the first visible GPU)")
    ap.add_argument("--threads", type=int, default=None,
                    help="threads per process (default: available cores // jobs)")
    ap.add_argument("--force", action="store_true",
                    help="retrain even if the artefact already exists")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", nargs=3, metavar=("ALGO", "SEED", "SCENARIO"),
                    help="exit 0 if this one job is finished under the current protocol "
                         "(used by slurm_sweep.sh / tsubame_sweep.sh), else 1")
    args = ap.parse_args()
    if args.check:
        algo, seed, scen = args.check
        done, why = is_complete(algo, int(seed), scen)
        print(f"{algo} {scen} seed {seed}: {'finished' if done else 'to run'} ({why})")
        sys.exit(0 if done else 1)

    os.chdir(HERE)
    jobs = all_jobs(args)

    if args.plan:
        parts = split(jobs, args.plan)
        tot = sum(j.hours for j in jobs)
        print(f"{len(jobs)} jobs, about {tot:.0f} job-hours, over {args.plan} servers "
              f"running {args.jobs} job(s) at a time each:")
        for k, p in enumerate(parts, start=1):
            print(f"  --part {k}/{args.plan}: {len(p):3d} jobs, {sum(j.hours for j in p):6.1f} job-hours, "
                  f"about {makespan(p, args.jobs):5.1f} h of wall time")
        return

    if args.part:
        k, n = (int(x) for x in args.part.split("/"))
        if not 1 <= k <= n:
            raise SystemExit(f"--part {args.part}: K must be between 1 and N")
        jobs = split(jobs, n)[k - 1]
        print(f"share {k}/{n}: {len(jobs)} jobs, {sum(j.hours for j in jobs):.1f} job-hours "
              f"(same options on every server, only K differs)")

    if args.pack:
        sys.exit(pack(jobs, args.pack))

    os.makedirs(os.path.join("logs", "sweep"), exist_ok=True)
    try:                                   # cgroup/SLURM-aware where available
        n_cores = len(os.sched_getaffinity(0))
    except AttributeError:
        n_cores = os.cpu_count() or 16
    threads = args.threads or max(1, n_cores // max(args.jobs, 1))

    queue, skipped = [], []
    for j in sorted(jobs, key=lambda j: (-j.hours, j.key)):      # longest first
        done_already, why = j.status()
        (skipped if done_already and not args.force else queue).append((j, why))

    print(f"{len(queue)} job(s) to run, {len(skipped)} already done, "
          f"{args.jobs} at a time, {threads} threads each, "
          f"budget {BUDGET:,} steps, seeds {args.seeds}; "
          f"about {makespan([j for j, _ in queue], args.jobs):.0f} h if run here")
    for j, why in skipped:
        print(f"  skip  {j.label}  ({why})")
    for j, why in queue:
        print(f"  queue {j.label}  ({why})")
    if args.dry_run or not queue:
        return

    gpus = [g.strip() for g in args.gpus.split(",")] if args.gpus else None
    running = []          # (proc, job, log handle, start time, gpu)
    done, failed = [], []
    t_start = time.perf_counter()

    def launch(j):
        cmd, env = j.command(threads, args.force)
        gpu = None
        if gpus and j.kind == "main":
            busy = [r[4] for r in running]
            gpu = min(gpus, key=lambda g: (busy.count(g), gpus.index(g)))
            env["CUDA_VISIBLE_DEVICES"] = gpu
        fh = open(j.log, "w", encoding="utf-8", buffering=1)
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=HERE, env=env)
        print(f"[{time.strftime('%H:%M:%S')}] start {j.label} (pid {proc.pid}"
              f"{', GPU ' + gpu if gpu is not None else ''}) -> {j.log}", flush=True)
        return (proc, j, fh, time.perf_counter(), gpu)

    queue = [j for j, _ in queue]
    while queue or running:
        while queue and len(running) < args.jobs:
            running.append(launch(queue.pop(0)))

        time.sleep(5)

        for entry in list(running):
            proc, j, fh, t0, _ = entry
            if proc.poll() is None:
                continue
            running.remove(entry)
            fh.close()
            mins = (time.perf_counter() - t0) / 60
            if proc.returncode == 0:
                done.append(j)
                print(f"[{time.strftime('%H:%M:%S')}] DONE  {j.label} in {mins:.1f} min "
                      f"({len(done)}/{len(done) + len(failed) + len(queue) + len(running)} finished)", flush=True)
            else:
                failed.append(j)
                print(f"[{time.strftime('%H:%M:%S')}] FAIL  {j.label} (exit {proc.returncode}) "
                      f"after {mins:.1f} min -- see {j.log}", flush=True)

    total = (time.perf_counter() - t_start) / 60
    print(f"\nsweep finished in {total:.1f} min: {len(done)} ok, {len(failed)} failed")
    for j in failed:
        print(f"  FAILED {j.label}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
