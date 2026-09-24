#!/usr/bin/env python3
"""Fold the run_all.sh logs into Tables 4 and 5 of the paper.

The benches print one scene per run; the paper's tables combine eight.
This is the step in between. It reproduces the published figures of
Table 4 (the restart detector) and Table 5 exactly, which is what shows
the aggregation was the one described.

Table 4: 8 sequences x 3 fields of view x 57 edges = 1368.
Table 5: 8 sequences, every station = 526 edges.

Log files keep the tags the figure pipeline also reads:
tIV_* is Table 4, tV_* is Table 5, vb_* is Section 5.2.
"""
import re, sys, os

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
SEQS = ["apartment", "hauptgebaude", "plain", "stairs",
        "gazebo_summer", "gazebo_winter", "wood_summer", "wood_autumn"]

def table4():
    print("Table 4 - the restart detector (nine pushes)\n")
    print(f"{'sequence':<18}{'edges':>7}{'wrong basin':>13}{'caught':>9}{'FA':>5}")
    tot = [0, 0, 0, 0]
    for s in SEQS:
        E = W = C = F = 0
        for fov in (360, 90, 40):
            t = open(f"{RES}/tIV_{s}_{fov}.log").read()
            m = re.search(r"(\d+) edges, (\d+) in the wrong basin", t)
            E += int(m.group(1)); W += int(m.group(2))
            r = re.search(r"^\s+restart spread\s+([\d.]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$", t, re.M)
            C += int(r.group(2)); F += int(r.group(4))
        print(f"{s:<18}{E:>7}{W:>13}{C:>9}{F:>5}")
        for i, v in enumerate((E, W, C, F)): tot[i] += v
    print(f"{'total':<18}{tot[0]:>7}{tot[1]:>13}{tot[2]:>9}{tot[3]:>5}")

def table5():
    print("\n\nTable 5 - walking the survey against searching from nothing\n")
    print(f"{'sequence':<18}{'edges':>7}{'walk':>6}{'search':>7}{'silent':>7}{'FA':>5}")
    tot = [0] * 5; unres = 0
    for s in SEQS:
        t = open(f"{RES}/tV_{s}.log").read()
        E = int(re.search(r"(\d+) edges over", t).group(1))
        walk = int(re.search(r"walking the survey\s+(\d+)", t).group(1))
        srch = int(re.search(r"searching, no guess\s+(\d+)", t).group(1))
        silent = int(re.search(r"of the \d+ the search lost, \d+ said so [^)]*\) and (\d+) did not", t).group(1))
        fa = int(re.search(r"false alarms: (\d+) of", t).group(1))
        u = re.search(r"of the \d+ silent, (\d+) fit at least as well", t)
        unres += int(u.group(1)) if u else 0
        print(f"{s:<18}{E:>7}{walk:>6}{srch:>7}{silent:>7}{fa:>5}")
        for i, v in enumerate((E, walk, srch, silent, fa)): tot[i] += v
    print(f"{'total':<18}{tot[0]:>7}{tot[1]:>6}{tot[2]:>7}{tot[3]:>7}{tot[4]:>5}")
    print(f"\nunresolvable (the wrong place fits at least as well): {unres}")

def section_5_2():
    print("\n\nSection 5.2 - how far the prediction understates the error\n")
    print(f"{'sequence':<18}{'directions':>13}{'median':>9}{'rank corr.':>12}")
    for s in sorted(SEQS):
        t = open(f"{RES}/vb_{s}.log").read()
        m = re.search(r"converged only: directions (\d+), median (\d+)×", t)
        r = re.search(r"rank correlation.*?mean ([+-][\d.]+)", t, re.S)
        d, med = (m.group(1), m.group(2) + "×") if m else ("0", "—")
        print(f"{s:<18}{d:>13}{med:>9}{r.group(1) if r else '—':>12}")

if __name__ == "__main__":
    table4(); table5(); section_5_2()
