#!/bin/sh
# Полный пересчёт четырёх стендов на восьми последовательностях ETH ASL.
# Условия взяты из подписей к таблицам статьи:
#   Table III  eth_survey  30 станций, пять полей зрения
#   Table IV   basin       30 станций, три поля зрения (360, ±90, ±40)
#   Table V    global      все станции, 360°
#   V-B        real_data   360°, поле зрения записано
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

run() { # имя_лога команда...
  log="$R/$1.log"; shift
  start=$(date +%s)
  "$@" > "$log" 2>&1
  code=$?
  printf "  %-34s %4s s  exit=%s\n" "$(basename "$log")" "$(( $(date +%s) - start ))" "$code"
}

SEQS="plain hauptgebaude apartment stairs gazebo_summer gazebo_winter wood_summer wood_autumn"

echo "══ V-B (real_data), 360°, 30 пар ══"
for s in $SEQS; do d=$(seq_dir "$s"); [ -n "$d" ] || { echo "  $s: НЕТ ДАННЫХ"; continue; }
  RIGIDITY_SECTOR= run "vb_$s" "$BIN/real_data" "$d" 30; done

echo "══ Table IV (basin), 30 станций ══"
for s in $SEQS; do d=$(seq_dir "$s"); [ -n "$d" ] || { echo "  $s: НЕТ ДАННЫХ"; continue; }
  for fov in 360 90 40; do
    if [ "$fov" = 360 ]; then unset RIGIDITY_SECTOR; else RIGIDITY_SECTOR=$fov; export RIGIDITY_SECTOR; fi
    run "tIV_${s}_${fov}" "$BIN/basin" "$d" 30
  done; unset RIGIDITY_SECTOR; done

echo "══ Table III (eth_survey), 30 станций ══"
for s in $SEQS; do d=$(seq_dir "$s"); [ -n "$d" ] || { echo "  $s: НЕТ ДАННЫХ"; continue; }
  for fov in 360 90 60 40 30; do
    if [ "$fov" = 360 ]; then unset RIGIDITY_SECTOR; else RIGIDITY_SECTOR=$fov; export RIGIDITY_SECTOR; fi
    run "tIII_${s}_${fov}" "$BIN/eth_survey" "$d" 30
  done; unset RIGIDITY_SECTOR; done

echo "══ Table V (global), все станции, 360° ══"
for s in $SEQS; do d=$(seq_dir "$s"); [ -n "$d" ] || { echo "  $s: НЕТ ДАННЫХ"; continue; }
  run "tV_$s" "$BIN/global" "$d" "$(stations "$d")"; done

echo "══ готово ══"
