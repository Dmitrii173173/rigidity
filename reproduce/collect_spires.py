#!/usr/bin/env python3
"""Fold the second-sensor runs into the shape of Tables 4 and 5.

The Hesai QT64 sequences of the Oxford Spires Dataset, laid out by
datasets/spires/prepare_spires.py and run as t4_<sequence>_<fov>.txt (basin,
first 30 stations, 360 / 90 / 40 degrees, the Hokuyo thresholds unchanged) and
t5_<sequence>_30.txt (global, first 30 stations, search net +-2 m at 0.5 m).

Table 4: 10 sequences x 3 fields of view x 57 edges = 1710.
Table 5: 10 sequences x 57 edges = 570.

Intervals resample sequences, 20,000 times, as the paper does for the Hokuyo
tables: the edges of one sequence are not independent of each other.

    python3 collect_spires.py <directory with the logs> [t4 | t4n0.025]

The second argument picks the Table 4 logs: t4 (the Hokuyo thresholds) or
t4n0.025 (the median-residual threshold at the Hesai's own noise).
"""
import glob, os, re, sys
import numpy as np

RES = sys.argv[1] if len(sys.argv) > 1 else "."
T4 = sys.argv[2] if len(sys.argv) > 2 else "t4"
DRAWS = 20000


def sequences():
    names = sorted({re.sub(rf"^{re.escape(T4)}_(.*)_(360|90|40)\.txt$", r"\1", os.path.basename(p))
                    for p in glob.glob(os.path.join(RES, f"{T4}_*_*.txt"))})
    return [n for n in names if all(os.path.exists(os.path.join(RES, f"{T4}_{n}_{f}.txt")) for f in (360, 90, 40))]


def detector(text, name):
    m = re.search(rf"^\s+{name}\s+([\d.]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$", text, re.M)
    return [int(m.group(i)) for i in (2, 3, 4, 5)]  # caught, missed, false alarms, quiet


def interval(per_sequence, numerator, denominator, rng):
    a = np.array([p[numerator] for p in per_sequence], float)
    b = np.array([p[denominator] for p in per_sequence], float)
    idx = rng.integers(0, len(a), size=(DRAWS, len(a)))
    num, den = a[idx].sum(1), b[idx].sum(1)
    rates = num[den > 0] / den[den > 0]
    if rates.size == 0:
        return float("nan"), float("nan")
    return 100 * np.percentile(rates, 2.5), 100 * np.percentile(rates, 97.5)


def table4(seqs, rng):
    print(f"Table 4 (second sensor, {T4} logs) - restart spread and median residual\n")
    print(f"{'sequence':<34}{'edges':>6}{'wrong':>7}{'restart caught':>16}{'FA':>5}{'median caught':>15}{'FA':>5}")
    rows = []
    for s in seqs:
        r = dict(E=0, W=0, rc=0, rf=0, mc=0, mf=0)
        for fov in (360, 90, 40):
            t = open(os.path.join(RES, f"{T4}_{s}_{fov}.txt"), encoding="utf-8", errors="replace").read()
            m = re.search(r"(\d+) edges, (\d+) in the wrong basin", t)
            r["E"] += int(m.group(1)); r["W"] += int(m.group(2))
            c, _, f, _ = detector(t, "restart spread"); r["rc"] += c; r["rf"] += f
            c, _, f, _ = detector(t, "median residual"); r["mc"] += c; r["mf"] += f
        r["S"] = r["E"] - r["W"]
        rows.append(r)
        print(f"{s:<34}{r['E']:>6}{r['W']:>7}{r['rc']:>16}{r['rf']:>5}{r['mc']:>15}{r['mf']:>5}")
    T = {k: sum(r[k] for r in rows) for k in rows[0]}
    print(f"{'total':<34}{T['E']:>6}{T['W']:>7}{T['rc']:>16}{T['rf']:>5}{T['mc']:>15}{T['mf']:>5}")
    print(f"\n  restart: catches {T['rc']} of {T['W']} ({100*T['rc']/T['W']:.1f} %, "
          f"{interval(rows,'rc','W',rng)[0]:.1f}-{interval(rows,'rc','W',rng)[1]:.1f}), "
          f"{T['rf']} false alarms of {T['S']} sound ({100*T['rf']/T['S']:.1f} %, "
          f"{interval(rows,'rf','S',rng)[0]:.1f}-{interval(rows,'rf','S',rng)[1]:.1f})")
    print(f"  median residual: catches {T['mc']} of {T['W']} ({100*T['mc']/T['W']:.1f} %, "
          f"{interval(rows,'mc','W',rng)[0]:.1f}-{interval(rows,'mc','W',rng)[1]:.1f}), "
          f"{T['mf']} false alarms of {T['S']} sound ({100*T['mf']/T['S']:.1f} %, "
          f"{interval(rows,'mf','S',rng)[0]:.1f}-{interval(rows,'mf','S',rng)[1]:.1f})")
    print("  (per cent, 95 % interval resampling sequences)")


def table5(seqs, rng):
    print("\n\nTable 5 (second sensor) - walking the survey against searching from nothing\n")
    print(f"{'sequence':<34}{'edges':>6}{'walk':>6}{'search':>7}{'silent':>7}{'FA':>5}{'unres.':>7}")
    rows = []
    for s in seqs:
        p = os.path.join(RES, f"t5_{s}_30.txt")
        if not os.path.exists(p):
            continue
        t = open(p, encoding="utf-8", errors="replace").read()
        if "edges over" not in t:
            print(f"{s:<34}  not finished")
            continue
        r = dict(E=int(re.search(r"(\d+) edges over", t).group(1)),
                 walk=int(re.search(r"walking the survey\s+(\d+)", t).group(1)),
                 search=int(re.search(r"searching, no guess\s+(\d+)", t).group(1)),
                 silent=int(re.search(r"of the \d+ the search lost, \d+ said so [^)]*\) and (\d+) did not", t).group(1)),
                 fa=int(re.search(r"false alarms: (\d+) of", t).group(1)))
        u = re.search(r"of the \d+ silent, (\d+) fit at least as well", t)
        r["unres"] = int(u.group(1)) if u else 0
        r["sound"] = r["E"] - r["search"]
        rows.append(r)
        print(f"{s:<34}{r['E']:>6}{r['walk']:>6}{r['search']:>7}{r['silent']:>7}{r['fa']:>5}{r['unres']:>7}")
    if not rows:
        return
    T = {k: sum(r[k] for r in rows) for k in rows[0]}
    print(f"{'total':<34}{T['E']:>6}{T['walk']:>6}{T['search']:>7}{T['silent']:>7}{T['fa']:>5}{T['unres']:>7}")
    for name, num, den in (("walk loses", "walk", "E"), ("search loses", "search", "E"),
                           ("silent among the search's failures", "silent", "search"),
                           ("false alarms among sound edges", "fa", "sound"),
                           ("unresolvable", "unres", "E")):
        lo, hi = interval(rows, num, den, rng)
        print(f"  {name}: {T[num]} of {T[den]} ({100*T[num]/max(T[den],1):.1f} %, {lo:.1f}-{hi:.1f})")


if __name__ == "__main__":
    rng = np.random.default_rng(20261007)
    seqs = sequences()
    table4(seqs, rng)
    if T4 == "t4":
        table5(seqs, rng)
