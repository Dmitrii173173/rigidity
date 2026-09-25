# Recomputing the tables

The benches print one scene per run; the tables of the paper combine eight.
This directory is the step in between, which Section 4.3 of the paper points at.

```sh
cargo build --release -p rigidity-cli --example real_data \
            -p rigidity-graph --example basin --example global --example eth_survey
./reproduce/run_all.sh          # 80 runs, about 40 minutes on an M5
python3 reproduce/collect.py results
python3 reproduce/table3.py    results
```

All eight sequences are needed: `datasets/fetch.sh`, and for the long ones
`datasets/fetch_chunked.sh` — the ETH server drops long connections, and
`curl -C -` restarts from the beginning when it does.

The benches look for the pose file beside the scans: if `pose_scanner_leica.csv`
sits only in `csv_global`, copy it into `csv_local`.

## What produces what

| Table in the paper | Bench | Conditions | Aggregation |
|---|---|---|---|
| Table 3 | `eth_survey` | 30 stations, five fields of view | `table3.py` |
| Table 4 | `basin` | 30 stations, 360°/±90°/±40° | `collect.py` |
| Table 5 | `global` | every station, 360° | `collect.py` |
| Section 5.2 | `real_data` | 360°, 30 pairs | `collect.py` |

Log file names keep the tags the figure pipeline under `paper-isprs-jprs/figures`
also reads, so they still carry the Roman numbering of an earlier draft:
`tIII_` is Table 3, `tIV_` is Table 4, `tV_` is Table 5 and `vb_` is Section 5.2.

Table 2 has nothing to aggregate — `bias` prints its whole row:

```sh
RIGIDITY_INIT_SIGMA=0.10 cargo run --release -p rigidity-graph --example bias -- <directory> 12
```

Section 5.1 comes from `monte_carlo`, which needs no dataset, and Fig. 5 from
`residuals`.

## The second instrument

The terrestrial-scanner rows of Table 2 and the arch of Section 5.5 come from the
ETH TLS registration benchmark (Theiler et al., 2015). It is public and ships each
scene as `s1.ply` … with a pairwise reference `groundtruth/sa-sb.tfm`, which maps
the coordinates of scan a into the frame of scan b — the direction the loops of
triangles settle, not an assumption; `prepare_tls.py` says how.

```sh
datasets/tls/fetch_tls.sh                     # office 308 MB, arch 915 MB, facade 1.5 GB
(cd datasets/tls && unzip office.zip && unzip arch.zip && unzip facade.zip)
python3 datasets/tls/prepare_tls.py           # <scene>_asl/: Hokuyo_<i>.ply links and the poses
for s in 0.02 0.10 0.30; do
  RIGIDITY_SCAN_EXT=ply RIGIDITY_ALL_PAIRS=1 RIGIDITY_INIT_SIGMA=$s \
    ./target/release/examples/bias datasets/tls/office_asl 4
done
RIGIDITY_SCAN_EXT=ply RIGIDITY_ALL_PAIRS=1 RIGIDITY_INIT_SIGMA=0.10 \
  ./target/release/examples/bias datasets/tls/arch_asl 4
```

`repeat` probes a scene for a wrong place that fits as well as the right one and
reads the restart on it; it is run on the corridor, where the answer is known,
and on the facade:

```sh
RIGIDITY_PROBE_EDGES=11-13,12-14,16-18,17-19,21-23,0-1,11-12,5-7 \
  ./target/release/examples/repeat <hauptgebaude csv_local> 36
RIGIDITY_SCAN_EXT=ply RIGIDITY_PROBE_RANGE=8 \
  ./target/release/examples/repeat datasets/tls/facade_asl 7
```

## The literature search

`literature/` holds the protocol written before the search, the harvest script,
every record and every screening decision behind the negative claim of Section 2.
Its README gives the counts.

## The margin in Table 3

`table3.py` calls an improvement of the worst station by more than a stated
margin a win. The margin is the author's choice and the caption of Table 3 says
which one: fifteen per cent. The bench itself calls a variant better on any
improvement whatever, and at a floor of zero the weighting is JᵀWJ itself, so by
that rule it declares itself better; that is arithmetic noise, and by it the
count of "wins" is an order larger than the real one. Both counts are printed,
`wins` at the margin and `any` at the bench's own rule. Pass a different margin
as the second argument to recompute all five rows:

```sh
python3 reproduce/table3.py results 0.05
```
