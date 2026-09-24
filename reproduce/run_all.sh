#!/bin/sh
# Re-run the four benches over the eight ETH ASL sequences. The conditions are
# the ones the table captions of the paper state:
#   Table 3        eth_survey  30 stations, five fields of view
#   Table 4        basin       30 stations, three fields of view (360, ±90, ±40)
#   Table 5        global      every station, 360°
#   Section 5.2    real_data   360°, the field of view recorded
# Log names keep the tags the figure pipeline also reads: tIII_ is Table 3,
# tIV_ is Table 4, tV_ is Table 5, vb_ is Section 5.2.
set -u
cd "$(dirname "$0")"
R="$(pwd)/results"; mkdir -p "$R"
BIN="$(cd "$(dirname "$0")/.." && pwd)/target/release/examples"
DS="$(cd "$(dirname "$0")/.." && pwd)/datasets"

seq_dir() {
  case "$1" in
    plain)         d="$DS/plain_extract" ;;
    hauptgebaude)  d="$DS/haupt_extract" ;;
    apartment)     d="$DS/apartment_extract" ;;
    stairs)        d="$DS/stairs_extract" ;;
    gazebo_summer) d="$DS/gazebo_summer_extract" ;;
    gazebo_winter) d="$DS/gazebo_winter_extract" ;;
    wood_summer)   d="$DS/wood_summer_extract" ;;
    wood_autumn)   d="$DS/wood_autumn_extract" ;;
  esac
  find "$d" -maxdepth 2 -type d -name csv_local 2>/dev/null | head -1
}
stations() { ls "$1"/Hokuyo_*.csv 2>/dev/null | wc -l | tr -d ' '; }

run() { # log_name command...
  log="$R/$1.log"; shift
  start=$(date +%s)
  "$@" > "$log" 2>&1
  code=$?
  printf "  %-34s %4s s  exit=%s\n" "$(basename "$log")" "$(( $(date +%s) - start ))" "$code"
}

SEQS="plain hauptgebaude apartment stairs gazebo_summer gazebo_winter wood_summer wood_autumn"

echo "══ Section 5.2 (real_data), 360°, 30 pairs ══"
for s in $SEQS; do d=$(seq_dir "$s"); [ -n "$d" ] || { echo "  $s: NO DATA"; continue; }
  RIGIDITY_SECTOR= run "vb_$s" "$BIN/real_data" "$d" 30; done

echo "══ Table 4 (basin), 30 stations ══"
for s in $SEQS; do d=$(seq_dir "$s"); [ -n "$d" ] || { echo "  $s: NO DATA"; continue; }
  for fov in 360 90 40; do
    if [ "$fov" = 360 ]; then unset RIGIDITY_SECTOR; else RIGIDITY_SECTOR=$fov; export RIGIDITY_SECTOR; fi
    run "tIV_${s}_${fov}" "$BIN/basin" "$d" 30
  done; unset RIGIDITY_SECTOR; done

echo "══ Table 3 (eth_survey), 30 stations ══"
for s in $SEQS; do d=$(seq_dir "$s"); [ -n "$d" ] || { echo "  $s: NO DATA"; continue; }
  for fov in 360 90 60 40 30; do
    if [ "$fov" = 360 ]; then unset RIGIDITY_SECTOR; else RIGIDITY_SECTOR=$fov; export RIGIDITY_SECTOR; fi
    run "tIII_${s}_${fov}" "$BIN/eth_survey" "$d" 30
  done; unset RIGIDITY_SECTOR; done

echo "══ Table 5 (global), every station, 360° ══"
for s in $SEQS; do d=$(seq_dir "$s"); [ -n "$d" ] || { echo "  $s: NO DATA"; continue; }
  run "tV_$s" "$BIN/global" "$d" "$(stations "$d")"; done

echo "══ done ══"
