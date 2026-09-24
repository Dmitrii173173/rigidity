#!/bin/sh
# Download in chunks, checking the length of each.
#
# The ETH server drops long connections, so one curl cannot take the file:
# `-C -` falls back to the beginning when it breaks. Here each chunk goes to a
# file of its own, its length is checked against the one expected, and only
# then is it appended. curl's exit code cannot be trusted: it can be zero on an
# incomplete transfer.
set -u
cd "$(dirname "$0")/.."
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36"

# Name, UUID and exact length. The length is required: without it a broken
# transfer cannot be told from a finished one, which is exactly what happens
# here. The values come from the collection's API; see fetch.sh.
if [ "$#" -ne 3 ]; then
  echo "usage: fetch_chunked.sh <name> <uuid> <bytes>"
  echo "  apartment     fb20daee-c4da-4521-9502-34bb924654c7  3146589806"
  echo "  stairs        745b3beb-5294-4914-b7b6-b7a749d12634  2919579746"
  echo "  gazebo_summer 20e4fcd9-42d2-470e-8f90-598690fe65e2  1332460435"
  echo "  hauptgebaude  3e5895e0-c557-4d8e-b023-d7e9d37dca34  3706859919"
  exit 2
fi
NAME="$1"
URL="https://www.research-collection.ethz.ch/server/api/core/bitstreams/$2/content"
OUT="datasets/$NAME.zip"
PART="datasets/.chunk-$NAME"
TOTAL="$3"
CHUNK=33554432

rm -f "$OUT" "$PART"
offset=0
reported=-1
while [ "$offset" -lt "$TOTAL" ]; do
  end=$((offset + CHUNK - 1))
  [ "$end" -ge "$TOTAL" ] && end=$((TOTAL - 1))
  want=$((end - offset + 1))

  tries=0
  while [ "$tries" -lt 10 ]; do
    rm -f "$PART"
    curl -sS --max-time 240 --connect-timeout 30 -A "$UA" -r "$offset-$end" -o "$PART" "$URL" >/dev/null 2>&1
    got=$(stat -f%z "$PART" 2>/dev/null || echo 0)
    [ "$got" -eq "$want" ] && break
    tries=$((tries + 1))
  done
  if [ "$tries" -ge 10 ]; then
    echo "could not fetch the chunk at offset $offset"
    exit 1
  fi

  cat "$PART" >> "$OUT"
  offset=$((end + 1))
  pct=$((offset * 100 / TOTAL))
  if [ "$pct" -ge $((reported + 20)) ]; then
    echo "$NAME: ${pct}%"
    reported=$pct
  fi
done
rm -f "$PART"
echo "downloaded $(stat -f%z "$OUT") bytes of $TOTAL"
unzip -l "$OUT" >/dev/null 2>&1 && echo "archive intact" || echo "archive damaged"
