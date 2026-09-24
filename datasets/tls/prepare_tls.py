#!/usr/bin/env python3
"""Lay the ETH TLS registration benchmark out the way the benches read a survey.

The benchmark (Theiler et al., 2015; fetch_tls.sh downloads it) ships each scene
as s1.ply ... sN.ply and a pairwise reference, groundtruth/sa-sb.tfm for every
ordered pair. The benches read Hokuyo_<i>.<ext> and one pose per station in
pose_scanner_leica.csv, so this writes, for each scene, a directory <scene>_asl/
holding symbolic links Hokuyo_0.ply ... to the scans and the poses.

Which way a .tfm points was settled by the loops, not assumed: sa-sb maps the
coordinates of scan a into the frame of scan b, because with that reading every
triangle of the facade closes to 0.0 mm, of the office to 17.6 mm and of the arch
to 9.1 mm, while the other reading leaves metres. So with s1 as the world, the
pose of station j is sj-s1.tfm, and s1 is the identity.

    python3 prepare_tls.py            # office, arch, facade, whichever are extracted
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HEADER = "poseId, timestamp, " + ", ".join(f"T{r}{c}" for r in range(4) for c in range(4))

def stations(scene):
    n = 0
    while os.path.exists(os.path.join(HERE, scene, f"s{n + 1}.ply")):
        n += 1
    return n

def prepare(scene):
    n = stations(scene)
    if n == 0:
        return None
    out = os.path.join(HERE, f"{scene}_asl")
    os.makedirs(out, exist_ok=True)
    rows = [HEADER]
    for j in range(1, n + 1):
        pose = np.eye(4) if j == 1 else np.loadtxt(os.path.join(HERE, scene, "groundtruth", f"s{j}-s1.tfm"))
        link = os.path.join(out, f"Hokuyo_{j - 1}.ply")
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(os.path.join("..", scene, f"s{j}.ply"), link)
        rows.append(f"{j - 1}, 0.0, " + ", ".join(f"{v:.9f}" for v in pose.reshape(-1)))
    with open(os.path.join(out, "pose_scanner_leica.csv"), "w") as f:
        f.write("\n".join(rows) + "\n")
    return out, n

if __name__ == "__main__":
    for scene in sys.argv[1:] or ["office", "arch", "facade"]:
        done = prepare(scene)
        print(f"{scene}: " + (f"{done[1]} stations -> {done[0]}" if done else "not extracted"))
