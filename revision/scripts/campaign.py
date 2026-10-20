"""Unattended job queue for the revision experiments.

    nohup python scripts/campaign.py JOBFILE MAX_PARALLEL > logs/campaign.log 2>&1 &

JOBFILE is a JSON list of {"name", "cmd", "done_if", "env"(optional), "stamp"(optional)};
a job whose done_if file exists (and, with "stamp", was made from the current checkpoint)
is skipped, so the queue can be restarted at any time.
Jobs run in listed order with at most MAX_PARALLEL at once; each job's output
goes to logs/jobs/<name>.log. A status file logs/campaign_status.json is
rewritten after every change.
"""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)


class _External:
    """Stand-in for a Popen of a process started by an earlier campaign."""

    def __init__(self, pid, done_file):
        self.pid, self.done_file, self.returncode = pid, done_file, None

    def poll(self):
        try:
            os.kill(self.pid, 0)
            return None
        except OSError:
            self.returncode = 0 if os.path.exists(self.done_file) else 1
            return self.returncode


def is_done(j):
    """done_if exists -- and, for a job with "stamp": [scenario, label, seed], the
    output was made from the checkpoint now in models/ (provenance.py), so the
    outputs of an earlier model are re-made instead of skipped."""
    path = os.path.join(REV, j["done_if"])
    if not os.path.exists(path):
        return False
    if "stamp" in j:
        sys.path.insert(0, HERE)
        from provenance import output_is_current
        return output_is_current(path, os.path.dirname(REV), *j["stamp"])
    return True


def main(jobfile, max_par):
    os.makedirs(os.path.join(REV, "logs", "jobs"), exist_ok=True)
    status_path = jobfile + ".status.json"
    seen, pending = set(), []
    running, done, failed = {}, [], []

    def refresh():
        """(Re)read the job file; jobs appended later are queued too."""
        try:
            jobs = json.load(open(jobfile))
        except (OSError, ValueError):
            return
        for j in jobs:
            if j["name"] in seen:
                continue
            seen.add(j["name"])
            if is_done(j):
                done.append(j["name"] + " (skipped: done)")
            else:
                pending.append(j)
    refresh()
    t0 = time.time()

    def dump():
        json.dump({"elapsed_min": round((time.time() - t0) / 60, 1),
                   "pending": [j["name"] for j in pending],
                   "running": {n: round((time.time() - s) / 60, 1) for n, (_, s) in running.items()},
                   "done": done, "failed": failed}, open(status_path, "w"), indent=1)

    while pending or running or os.path.exists(jobfile + ".keepalive"):
        while pending and len(running) < max_par:
            j = pending.pop(0)
            # a job left running by an earlier (stopped) campaign: adopt it
            probe = subprocess.run(["pgrep", "-f", j["cmd"].split(" ", 1)[1]],
                                   capture_output=True, text=True).stdout.split()
            if probe:
                running[j["name"]] = (_External(int(probe[0]), os.path.join(REV, j["done_if"])), time.time())
                print(time.strftime("%H:%M:%S"), "adopt", j["name"], flush=True)
                continue
            env = dict(os.environ, **{k: str(v) for k, v in j.get("env", {}).items()})
            log = open(os.path.join(REV, "logs", "jobs", j["name"] + ".log"), "w")
            p = subprocess.Popen(j["cmd"], shell=True, cwd=REV, env=env,
                                 stdout=log, stderr=subprocess.STDOUT)
            running[j["name"]] = (p, time.time())
            print(time.strftime("%H:%M:%S"), "start", j["name"], flush=True)
            dump()
        time.sleep(5)
        refresh()
        for n, (p, s) in list(running.items()):
            if p.poll() is not None:
                del running[n]
                (done if p.returncode == 0 else failed).append(n)
                print(time.strftime("%H:%M:%S"), "end", n, p.returncode,
                      f"{(time.time() - s) / 60:.1f} min", flush=True)
                dump()
    dump()
    print("campaign finished", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
