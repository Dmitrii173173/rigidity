#!/usr/bin/env python3
"""Lay the Oxford Spires sequences out the way the benches read a survey.

The Oxford Spires Dataset (Tao et al., 2025; fetch_spires.sh downloads it) is a
handheld Hesai QT64 walked around Oxford sites that a Leica RTC360 surveyed. Per
sequence it ships the SLAM keyframes as motion-undistorted clouds in the sensor's
base frame, processed/vilens-slam/undist-clouds/cloud_<sec>_<nsec>.pcd, and
processed/trajectory/gt-tum.txt: the base frame's pose in the TLS map, from
registering every undistorted cloud to that map, stated to 1-2 cm. The reference
carries a pose at every keyframe's own timestamp, so nothing is interpolated.

The benches read Hokuyo_<i>.<ext> and one pose per station in
pose_scanner_leica.csv, so this writes, per sequence, <sequence>_asl/ holding hard
links Hokuyo_0.pcd ... to the keyframes in time order and the poses. Hard links
need no privilege on Windows and cost no space; the extracted clouds must sit on
the same volume.

Two things make a keyframe list into a survey. The SLAM keeps keyframes 0.35 m
apart in two Keble sequences and about 1.07 m apart everywhere else, so stations
are thinned to at least MIN_STEP apart and every sequence is walked at one pace.
And the reference covers only what the TLS map covers: where it is missing, the
next station with a pose can be tens of metres on, which a survey that starts each
leg from the one before cannot cross. The sequence is cut at every such jump and
the longest unbroken stretch is kept. The three Blenheim Palace sequences have no
stretch longer than 29 stations and are not used.

    python3 prepare_spires.py                  # every downloaded sequence
    python3 prepare_spires.py <sequence> ...
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HEADER = "poseId, timestamp, " + ", ".join(f"T{r}{c}" for r in range(4) for c in range(4))
MATCH = 1e-3     # seconds: a keyframe without a reference pose this close is dropped
MIN_STEP = 0.9   # metres between kept stations
JUMP = 2.5       # metres: a step longer than this breaks the walk


def matrix(tx, ty, tz, qx, qy, qz, qw):
    w, x, y, z = np.array([qw, qx, qy, qz]) / np.linalg.norm([qw, qx, qy, qz])
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = [tx, ty, tz]
    return T


def stamp(name):
    _, sec, nsec = name[:-4].split("_")
    return int(sec) + int(nsec) * 1e-9


def prepare(sequence):
    root = os.path.join(HERE, sequence)
    clouds = os.path.join(root, "vilens-slam", "undist-clouds")
    truth = os.path.join(root, "trajectory", "gt-tum.txt")
    if not (os.path.isdir(clouds) and os.path.exists(truth)):
        return None
    reference = np.loadtxt(truth)
    names = sorted((n for n in os.listdir(clouds) if n.endswith(".pcd")), key=stamp)

    posed, dropped = [], 0
    for name in names:
        t = stamp(name)
        k = int(np.argmin(np.abs(reference[:, 0] - t)))
        if abs(reference[k, 0] - t) > MATCH:
            dropped += 1
            continue
        posed.append((name, t, matrix(*reference[k, 1:8])))

    # one pace, then cut at the jumps and keep the longest stretch
    kept = []
    for station in posed:
        if not kept or np.linalg.norm(station[2][:3, 3] - kept[-1][2][:3, 3]) >= MIN_STEP:
            kept.append(station)
    runs, run = [], []
    for station in kept:
        if run and np.linalg.norm(station[2][:3, 3] - run[-1][2][:3, 3]) > JUMP:
            runs.append(run)
            run = []
        run.append(station)
    runs.append(run)
    walk = max(runs, key=len)

    out = os.path.join(HERE, f"{sequence}_asl")
    os.makedirs(out, exist_ok=True)
    for old in os.listdir(out):
        os.remove(os.path.join(out, old))
    rows = [HEADER]
    for i, (name, t, pose) in enumerate(walk):
        os.link(os.path.join(clouds, name), os.path.join(out, f"Hokuyo_{i}.pcd"))
        rows.append(f"{i}, {t:.9f}, " + ", ".join(f"{v:.9f}" for v in pose.reshape(-1)))
    with open(os.path.join(out, "pose_scanner_leica.csv"), "w") as f:
        f.write("\n".join(rows) + "\n")
    return out, len(walk), dropped, len(kept), len(runs)


if __name__ == "__main__":
    wanted = sys.argv[1:] or sorted(d for d in os.listdir(HERE)
                                    if os.path.isdir(os.path.join(HERE, d)) and not d.endswith("_asl"))
    for sequence in wanted:
        done = prepare(sequence)
        if done is None:
            print(f"{sequence}: no clouds or no reference")
        else:
            out, n, dropped, kept, runs = done
            print(f"{sequence}: {n} stations -> {out} (longest of {runs} stretches of {kept} "
                  f"stations after thinning; {dropped} keyframes without a pose)")
