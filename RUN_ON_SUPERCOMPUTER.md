# Running the revision campaign on several servers

This page is the complete procedure for the October 2026 campaign. There are two ways to run
it; **use only one of them**:

- **Several separate servers, each with its own disk** (how the first campaign was run):
  follow Steps 1–6 below.
- **TSUBAME batch jobs** (`qsub`, all nodes see the same folder): follow only the last
  section, *Alternative: TSUBAME batch jobs*.

Doing both would train every job twice. In Steps 1–6 every command is copied as it is; only
the server number `K` changes.

**What runs.** 116 independent jobs, all from scratch (nothing of the first campaign is reused):

| Jobs | What one job does | Needs |
|---|---|---|
| 50 main jobs: 5 algorithms × 5 seeds on cityNL and on PublicPST-S | trains one controller for 1,000,000 steps | GPU, about 8 GB RAM (DDPG: 18 GB) |
| 66 PublicPST-S variant jobs | trains one TC-CPL variant for 200,000 steps and evaluates it | CPU only, about 3 GB RAM |

**How long.** About 30 hours with 7 servers running 5 jobs at a time each (the longest single
job takes about 30 hours, so nothing finishes sooner; even 5 servers would finish in about 30 hours).
This needs, on every server, about **64 GB of RAM and 20 or more CPU cores** for the 5 jobs; on a
smaller server use `--jobs 2` (about 41 hours with 7 servers).

---

## Step 1 — Install, on every server (10 minutes)

Use a **new folder**. Do not reuse the folder of the first campaign.

```bash
git clone https://github.com/danialsj77/TC-CPL.git TC-CPL-rev
cd TC-CPL-rev
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tccpl_learner.py
```

The last command must print `self-test OK` twice. Python must be 3.10 or 3.11.

## Step 2 — Look at the split (on any one server, optional)

```bash
python run_sweep.py --plan 7 --jobs 5
```

It prints which share each of the 7 servers gets and the estimated hours. If you use a
different number of servers, replace `7` by that number **everywhere below**.

## Step 3 — Start the jobs, on every server

On server 1:

```bash
cd TC-CPL-rev && source .venv/bin/activate
nohup python run_sweep.py --part 1/7 --jobs 5 > sweep_part1.log 2>&1 &
```

On server 2 the same with `--part 2/7` and `sweep_part2.log`, and so on up to `--part 7/7`.

- Every server must use **the same options**; only the number before `/7` changes. Each
  server then runs a different share and together they run all 116 jobs exactly once.
- `--jobs 5` = five jobs at the same time on that server (not the total number of jobs). It needs
  about 64 GB of RAM and 20 or more cores; each job gets an equal share of the cores. On a smaller
  server use `--jobs 2`: the share is the same, it just takes longer.
- A server with several GPUs: add `--gpus 0,1,2,3` (its GPU numbers) so the jobs are spread
  over them; without it all jobs use the first GPU.
- `nohup ... &` keeps the jobs running after you log out.

## Step 4 — Follow the progress

```bash
tail -n 20 sweep_part1.log                            # START / DONE / FAIL lines
python run_sweep.py --part 1/7 --dry-run              # what is finished and what is left
```

If a job fails, or a server restarts: **run the Step 3 command again**. Finished jobs are
skipped and only the missing ones run. The log of each job is in `logs/sweep/`.

## Step 5 — Pack the results, on every server, when its share is finished

```bash
python run_sweep.py --part 1/7 --pack part1.tgz       # server 2: --part 2/7 --pack part2.tgz ...
```

It prints `packed N finished job(s)`. If it also lists `NOT packed` lines, those jobs are not
finished: go back to Step 3 on that server.

Please use this command and **not** `tar` of the whole folder: the folder also contains old
copies of some results that come with `git clone`, and unpacking those last would overwrite
the new results of another server.

## Step 6 — Merge and check, on one server

```bash
git clone https://github.com/danialsj77/TC-CPL.git TC-CPL-merge
cd TC-CPL-merge
python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
tar xzf /path/to/part1.tgz            # repeat for part2.tgz ... part7.tgz (any order)
python check_cluster_results.py
```

The check must end with `0 item(s) with problems`. Then pack everything into one file:

```bash
python run_sweep.py --pack tccpl_results.tgz
```

and send `tccpl_results.tgz` (about 1 GB) to Daniyal. Sending the seven `partK.tgz` files
instead is also fine.

## If the check reports a problem

| Message | Meaning | What to do |
|---|---|---|
| `no checkpoint`, `no run` | the job did not run or did not finish | run Step 3 again on the server of that share, then Step 5 and Step 6 |
| `not from this campaign` | a model of the first campaign | the folder was reused: use a fresh clone (Step 1) |
| `multiplier history from an earlier run` | the part archive of that job is missing | unpack all `partK.tgz` files (Step 6) |
| `budget 12000`, `smoke run` | a short test run | run the job again with Step 3 |

---

### Alternative: TSUBAME batch jobs (shared disk)

Use this **instead of** Steps 2–5 if all jobs should run as TSUBAME batch jobs. After Step 1
(one fresh clone on the shared disk), from the repository folder (edit the `.venv` line in
both scripts first):

```bash
mkdir -p logs/qsub
qsub -g <your-group> tsubame_sweep.sh               # the 50 main jobs (GPU, 48 h limit each)
qsub -g <your-group> revision/stress_sweep.sh       # the 66 PublicPST-S variant jobs (CPU, 8 h limit each)
```

Then Step 6 from `python check_cluster_results.py` on, in that same folder.
