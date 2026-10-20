"""Status of the load-aware pilots against the pre-registered criteria
(revision/pilot_loadaware/PASS_CRITERIA.md).

    ~/.venvs/tccpl311/bin/python revision/scripts/pilot_status.py

Reads the dual updates printed in pilot_loadaware/tau*/train.log and the old
seed-43 history, and prints progress, an ETA, and C1-C3 once update 30 is in."""
import json, os, re, time
import numpy as np

REV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIL = os.path.join(REV, "pilot_loadaware")
OLD = os.path.join(os.path.dirname(REV), "results", "cityNL", "lambda_history_tccpl_s43.json")
PAT = re.compile(r"\[M1 @\s*(\d+)\] overload above tightened limit=([\d.]+) kWh \(true ([\d.na]+)\)\s+"
                 r"sat=([\d.]+) \| λ_ov=([\d.]+)\s+λ_sat=([\d.]+)")

old = json.load(open(OLD))
old_lam = np.array(old["lambda_history"]); old_vio = np.array(old["violation_history"])
W = slice(20, 30)
print(f"old seed 43 (first-submission networks), updates 21-30: overload {old_vio[W, 0].mean():.2f} kWh/day, "
      f"λ_ov {old_lam[20, 0]:.0f}→{old_lam[29, 0]:.0f} ({(old_lam[29, 0] - old_lam[20, 0]) / 9:+.1f}/update)\n")
# run: (tolerance, C1/R1 overload threshold, label of the lambda criterion)
RUNS = {"tau0": (0.0, 2.0, "C2"), "tau1": (1.0, 2.0, "C3"), "rho1": (0.0, 1.0, "R2")}
for run, (tau, c1_max, crit) in RUNS.items():
    log = os.path.join(PIL, run, "train.log")
    if not os.path.exists(log):
        print(f"{run}: no log"); continue
    txt = open(log, errors="ignore").read()
    rows = [(int(a), float(b), float(d), float(e), float(f)) for a, b, c, d, e, f in PAT.findall(txt)]
    alive = time.time() - os.path.getmtime(log) < 1800        # log written in the last 30 min
    err = "Traceback" in txt
    if not alive and "[M1 @" not in txt:
        print(f"{run}: not running\n"); continue
    st = os.stat(log)
    age_h = (time.time() - getattr(st, "st_birthtime", st.st_ctime)) / 3600   # creation time on macOS
    print(f"{run} (tolerance {tau:g}): {len(rows)} dual updates, log age {age_h:.1f} h"
          + ("  ERROR in log" if err else "") + ("" if alive else "  (no training process found)"))
    if not rows:
        print("   still collecting demonstrations / before the first update\n"); continue
    steps = np.array([r[0] for r in rows]); ov = np.array([r[1] for r in rows]); lam = np.array([r[3] for r in rows])
    rate = steps[-1] / max(age_h, 1e-6)                      # steps per hour (incl. setup)
    print(f"   latest: step {steps[-1]:,}  overload {ov[-1]:.2f}  λ_ov {lam[-1]:.1f}  λ_def {rows[-1][4]:.2f}  "
          f"| ~{rate:,.0f} steps/h → update 30 in ~{max(0, (300_000 - steps[-1]) / rate):.0f} h, 10^6 in ~{(1e6 - steps[-1]) / rate:.0f} h")
    k = min(len(rows), len(old_lam))
    print("   update:      " + " ".join(f"{i:>6d}" for i in range(1, k + 1)[-8:]))
    print("   λ_ov new:    " + " ".join(f"{x:6.0f}" for x in lam[:k][-8:]))
    print("   λ_ov old:    " + " ".join(f"{x:6.0f}" for x in old_lam[:k, 0][-8:]))
    print("   overload new:" + " ".join(f"{x:6.2f}" for x in ov[:k][-8:]))
    print("   overload old:" + " ".join(f"{x:6.2f}" for x in old_vio[:k, 0][-8:]))
    if len(rows) >= 30:
        c1 = ov[W].mean() <= c1_max
        rise = (lam[29] - lam[20]) / 9
        line = f"   {'R1' if crit == 'R2' else 'C1'} overload 21-30 = {ov[W].mean():.2f} (≤ {c1_max:g}): {'PASS' if c1 else 'fail'}"
        if crit == "R2":
            c = rise <= 2.6 and lam[29] < 500
            sat = np.array([r[2] for r in rows])[W].mean()
            line += (f" | R2 λ rise {rise:+.1f}/update (≤ +2.6), λ_ov {lam[29]:.0f} (< 500): {'PASS' if c else 'fail'}"
                     f" | training satisfaction {sat:.4f}")
        elif tau == 0:
            c = rise <= 2.6; line += f" | C2 λ rise {rise:+.1f}/update (≤ +2.6): {'PASS' if c else 'fail'}"
        else:
            viol = ov[W].mean() - tau
            c = rise <= 1.0 and abs(viol) <= 0.5
            line += f" | C3 λ rise {rise:+.1f}/update (≤ +1), violation {viol:+.2f} (±0.5): {'PASS' if c else 'fail'}"
        print(line)
    else:
        print(f"   criteria are judged at update 30 ({30 - len(rows)} to go)")
    print()
