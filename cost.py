"""
Spend ledger. Modal has no billing CLI, so we account for it ourselves from
container-seconds and the published per-second rates.

    python3 cost.py add prep-probe --cpu 2 --mem 8 --secs 210 --n 48
    python3 cost.py add pretrain --gpu H100 --secs 18000
    python3 cost.py                       # show the ledger

Rates verified against https://modal.com/pricing on 2026-08-30.
"""

import json
import os
import sys
import time

GPU = {  # $/sec
    "B300": 0.001972, "B200": 0.001736, "H200": 0.001261, "H100": 0.001097,
    "RTX6000": 0.000842, "A100-80": 0.000694, "A100-40": 0.000583,
    "L40S": 0.000542, "A10": 0.000306, "L4": 0.000222, "T4": 0.000164,
}
CPU_CORE_S = 0.0000131
MEM_GIB_S = 0.00000222
BUDGET = 30.00
# Modal bills the whole container lifetime -- image pull, startup, volume commit --
# and re-bills a preempted container that restarts. Estimating from the loop's own
# wall clock ran 30% low against the dashboard on the CPU fan-outs, so quote
# forecasts with this on top and trust the dashboard for actuals.
OVERHEAD = 1.30
LEDGER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spend.json")


def cost(gpu=None, cpu=0.125, mem=0.25, secs=0.0, n=1, gpus=1, overhead=OVERHEAD):
    """n = number of identical containers; gpus = GPUs inside each container."""
    c = (GPU[gpu] * gpus if gpu else 0.0) + max(cpu, 0.125) * CPU_CORE_S + mem * MEM_GIB_S
    return c * secs * n * overhead


def load():
    if not os.path.exists(LEDGER):
        return []
    with open(LEDGER) as fh:
        return json.load(fh)


def add(label, **kw):
    rows = load()
    kw["cost"] = round(cost(**kw), 4)
    kw["label"] = label
    kw["at"] = time.strftime("%H:%M:%S")
    rows.append(kw)
    with open(LEDGER, "w") as fh:
        json.dump(rows, fh, indent=1)
    return kw


def show():
    rows = load()
    tot = sum(r["cost"] for r in rows)
    print(f"{'when':<10}{'item':<26}{'gpu':<10}{'secs':>8}{'n':>4}{'$':>9}")
    for r in rows:
        g = str(r.get("gpu") or "-")
        if r.get("gpus", 1) > 1:
            g += f"x{r['gpus']}"
        print(f"{r['at']:<10}{r['label']:<26}{g:<10}"
              f"{r.get('secs', 0):>8.0f}{r.get('n', 1):>4}{r['cost']:>9.3f}")
    print(f"{'':<49}{'spent':>8}{tot:>9.3f}")
    print(f"{'':<49}{'left':>8}{BUDGET - tot:>9.3f}")
    if tot > BUDGET * 0.85:
        print("\n!! over 85% of budget")
    return tot


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] == "show":
        show()
    elif a[0] == "add":
        kw = {}
        i = 2
        while i < len(a):
            k = a[i].lstrip("-")
            v = a[i + 1]
            kw[k] = v if k == "gpu" else float(v)
            i += 2
        for k in ("n", "gpus"):
            if k in kw:
                kw[k] = int(kw[k])
        print(add(a[1], **kw))
        show()
