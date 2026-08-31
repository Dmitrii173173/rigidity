#!/usr/bin/env python3
"""Свёртка логов run_all.sh в Table IV и Table V статьи.

В репозитории такого скрипта нет: стенды печатают по одной сцене, а
таблицы агрегируют восемь. Эта свёртка воспроизводит опубликованные
числа Table IV (рестарт) и Table V точно, что и подтверждает, что
агрегация делалась именно так.

Table IV: 8 последовательностей × 3 поля зрения × 57 рёбер = 1368.
Table V:  8 последовательностей, все станции = 526 рёбер.
"""
import re, sys, os

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
SEQS = ["apartment", "hauptgebaude", "plain", "stairs",
        "gazebo_summer", "gazebo_winter", "wood_summer", "wood_autumn"]

def table_iv():
    print("Table IV — рестарт-детектор (девять толчков)\n")
    print(f"{'последовательность':<18}{'рёбер':>7}{'чужой басс.':>13}{'поймано':>9}{'ЛТ':>5}")
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
    print(f"{'ИТОГО':<18}{tot[0]:>7}{tot[1]:>13}{tot[2]:>9}{tot[3]:>5}")

def table_v():
    print("\n\nTable V — ход съёмки против поиска\n")
    print(f"{'последовательность':<18}{'рёбер':>7}{'ход':>6}{'поиск':>7}{'молч.':>7}{'ЛТ':>5}")
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
    print(f"{'ИТОГО':<18}{tot[0]:>7}{tot[1]:>6}{tot[2]:>7}{tot[3]:>7}{tot[4]:>5}")
    print(f"\nнеразрешимых (подходят не хуже истины): {unres}")

def v_b():
    print("\n\nV-B — во сколько раз предсказание занижает ошибку\n")
    print(f"{'последовательность':<18}{'направлений':>13}{'медиана':>9}{'ранг.корр.':>12}")
    for s in sorted(SEQS):
        t = open(f"{RES}/vb_{s}.log").read()
        m = re.search(r"converged only: directions (\d+), median (\d+)×", t)
        r = re.search(r"rank correlation.*?mean ([+-][\d.]+)", t, re.S)
        d, med = (m.group(1), m.group(2) + "×") if m else ("0", "—")
        print(f"{s:<18}{d:>13}{med:>9}{r.group(1) if r else '—':>12}")

if __name__ == "__main__":
    table_iv(); table_v(); v_b()
