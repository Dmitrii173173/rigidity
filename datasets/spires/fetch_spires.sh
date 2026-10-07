#!/bin/sh
# Oxford Spires Dataset (Tao et al., 2025), University of Oxford, CC BY-NC-SA 4.0.
# Only what the benches read: per sequence the reference trajectory and the SLAM keyframes as
# motion-undistorted clouds, about 0.25-0.65 GB each; the thirteen sequences that ship a reference
# trajectory, some 5 GB in all. The raw recordings (1.1 TB) are not needed.
#
#   sh fetch_spires.sh            # all thirteen
#   sh fetch_spires.sh <sequence> # one
#
# Then python3 prepare_spires.py lays them out as <sequence>_asl/.
cd "$(dirname "$0")"
BASE=https://huggingface.co/datasets/ori-drs/oxford_spires_dataset/resolve/main/sequences
SEQUENCES="${*:-2024-03-12-keble-college-02 2024-03-12-keble-college-03 2024-03-12-keble-college-04
  2024-03-12-keble-college-05 2024-03-13-observatory-quarter-01 2024-03-13-observatory-quarter-02
  2024-03-14-blenheim-palace-01 2024-03-14-blenheim-palace-02 2024-03-14-blenheim-palace-05
  2024-03-18-christ-church-02 2024-03-18-christ-church-03 2024-03-20-christ-church-05
  2024-05-20-bodleian-library-02}"
for s in $SEQUENCES; do
  for f in trajectory/gt-tum.txt vilens-slam/undist-clouds.zip; do
    mkdir -p "$s/$(dirname $f)"
    n=0
    until curl -s -f -L -C - --retry 5 --retry-delay 5 -o "$s/$f" "$BASE/$s/processed/$f"; do
      n=$((n+1)); [ $n -ge 20 ] && { echo "$s $f FAILED" >> progress.txt; continue 2; }
      sleep 5
    done
  done
  (cd "$s/vilens-slam" && unzip -q -o undist-clouds.zip) && echo "$s ready" >> progress.txt
done
echo DONE >> progress.txt
