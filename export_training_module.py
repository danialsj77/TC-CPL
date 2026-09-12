"""Generate `tccpl_runtime.py` from TC-CPL.ipynb.

The notebook stays the single source of truth. This script lifts its definition
cells (imports, CONFIG, env builders, M1/M2/M3, encoders, policy geometry) and
each of its training cells into an importable module, so a training run can be
launched as a plain process instead of only from inside Jupyter. That is what
makes it possible to run several runs concurrently -- Jupyter is one process, and
SubprocVecEnv is not usable on this machine (each worker re-imports torch and
exhausts the Windows paging file).

Re-run this after editing any of the notebook cells listed below:

    python export_training_module.py
"""
import io
import json
import os
import re

NOTEBOOK = "TC-CPL.ipynb"
OUTPUT = "tccpl_runtime.py"

# Every code cell in this INCLUSIVE index range is a definition cell.
DEFINITION_RANGE = (4, 32)

# One entry per training job: the cells that make up the job, in order.
# 38 is not cleanup -- it persists the lambda history of the TC-CPL run.
JOBS = {
    "fixedpenalty": [34],
    "tccpl":        [37, 38],
    "ddpg":         [41],
    "ppo":          [44],
    "cpo":          [47],
}

HEADER = '''"""AUTO-GENERATED from {nb} -- DO NOT EDIT BY HAND.

Regenerate with:  python export_training_module.py

Definition cells {lo}-{hi} are reproduced verbatim at module level; each training
cell becomes a `train_<name>()` function. Importing this module reads the seed
from the TCCPL_SEED environment variable (see the CONFIG cell), so set it
BEFORE importing.
"""
import matplotlib
matplotlib.use("Agg")   # training runs are headless
'''


def cell_source(cell):
    """Cell source with IPython magics/shell escapes dropped and the Jupyter
    progress bar disabled (headless runs must not require tqdm/rich)."""
    lines = []
    for line in "".join(cell["source"]).splitlines():
        stripped = line.lstrip()
        if stripped.startswith("!") or stripped.startswith("%"):
            lines.append("# [notebook magic removed] " + line)
        else:
            lines.append(line.replace("progress_bar=True", "progress_bar=False")
                             .replace("progress_bar    = True", "progress_bar    = False"))
    return "\n".join(lines)


def indent(text, prefix="    "):
    return "\n".join(prefix + l if l.strip() else l for l in text.splitlines())


def hoist_imports(src):
    """Lift column-0 imports out of a cell body, returning (imports, new_body).

    A notebook cell is module scope, so `import os` half way down still works for
    code above it. Wrapped in a function it does NOT: the import makes the name
    local to the WHOLE function, so any earlier use raises

        UnboundLocalError: cannot access local variable 'os'

    which is exactly what cell 41's `import json, os` did to train_safe(), whose
    body starts with os.makedirs(). Hoisting to module scope restores notebook
    semantics. Indented imports (inside try/if) are left alone -- those are
    deliberate and already scoped.
    """
    imports, body = [], []
    for line in src.splitlines():
        if re.match(r"^(?:import|from)\s+\S", line):
            imports.append(line)
            body.append(f"# [hoisted to module scope] {line}")
        else:
            body.append(line)
    return imports, "\n".join(body)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    with io.open(NOTEBOOK, encoding="utf-8") as f:
        nb = json.load(f)
    cells = nb["cells"]

    lo, hi = DEFINITION_RANGE
    out = [HEADER.format(nb=NOTEBOOK, lo=lo, hi=hi)]

    n_def = 0
    for i in range(lo, hi + 1):
        if i >= len(cells) or cells[i]["cell_type"] != "code":
            continue
        src = cell_source(cells[i])
        if not src.strip():
            continue
        out.append(f"\n# {'=' * 68}\n# notebook cell {i}\n# {'=' * 68}\n{src}\n")
        n_def += 1

    for name, idxs in JOBS.items():
        body, hoisted = [], []
        for i in idxs:
            if cells[i]["cell_type"] != "code":
                raise SystemExit(f"cell {i} for job {name!r} is not a code cell")
            imports, cleaned = hoist_imports(cell_source(cells[i]))
            hoisted.extend(imports)
            body.append(f"# --- notebook cell {i} ---\n{cleaned}")
        joined = "\n\n".join(body)
        preamble = ""
        if hoisted:
            uniq = list(dict.fromkeys(hoisted))
            preamble = ("# imports hoisted out of the job body (see hoist_imports)\n"
                        + "\n".join(uniq) + "\n")
        out.append(
            f"\n# {'=' * 68}\n# training job: {name}  (notebook cells "
            f"{', '.join(map(str, idxs))})\n# {'=' * 68}\n"
            f"{preamble}def train_{name}():\n{indent(joined)}\n"
        )

    out.append(
        "\n\nTRAIN_JOBS = {\n"
        + "".join(f'    "{n}": train_{n},\n' for n in JOBS)
        + "}\n"
    )

    text = "\n".join(out)
    compile(text, OUTPUT, "exec")          # fail loudly rather than write garbage

    with io.open(OUTPUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)

    print(f"wrote {OUTPUT}: {n_def} definition cells, {len(JOBS)} jobs, "
          f"{len(text.splitlines())} lines")
    print(f"jobs: {', '.join(JOBS)}")


if __name__ == "__main__":
    main()
