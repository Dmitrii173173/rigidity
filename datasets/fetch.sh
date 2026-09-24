#!/bin/sh
# Download the ETH ASL Challenging Datasets, resuming.
# The server drops the connection on large files, so one curl is not enough;
# for the ones that still fail, datasets/fetch_chunked.sh.
set -u
cd "$(dirname "$0")/.."
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36"
API="https://www.research-collection.ethz.ch/server/api/core/bitstreams"

fetch() {
  name="$1"; uuid="$2"
  attempt=1
  while [ "$attempt" -le 12 ]; do
    curl -sSL -C - -A "$UA" --retry 5 --retry-all-errors --retry-delay 3 \
         --connect-timeout 30 -o "datasets/$name.zip" "$API/$uuid/content" || true
    if unzip -l "datasets/$name.zip" >/dev/null 2>&1; then
      echo "$name: done, $(stat -f%z "datasets/$name.zip") bytes"
      return 0
    fi
    echo "$name: attempt $attempt, $(stat -f%z "datasets/$name.zip" 2>/dev/null || echo 0) bytes so far"
    attempt=$((attempt + 1))
  done
  echo "$name: FAILED"
  return 1
}

# All eight sequences of the one set: one sensor (a Hokuyo UTM-30LX), one
# reference (a Leica TS15 total station), one format. Sizes are given because
# eighteen gigabytes do not download in passing.
fetch plain          f493ec1b-b55c-43b7-be34-d230133fdd43   # 0.9 GB, open terrain, weak vertical constraint
fetch hauptgebaude   3e5895e0-c557-4d8e-b023-d7e9d37dca34   # 3.7 GB, corridor with repeating structure
fetch apartment      fb20daee-c4da-4521-9502-34bb924654c7   # 3.2 GB, apartment: the only sequence with genuine revisits
fetch stairs         745b3beb-5294-4914-b7b6-b7a749d12634   # 2.9 GB, stairs: strong changes in visible volume
fetch gazebo_summer  20e4fcd9-42d2-470e-8f90-598690fe65e2   # 1.3 GB, gazebo, sparse vegetation
fetch gazebo_winter  873a7c94-a0cf-4628-a0a1-cb6e2a6c0c8b   # 2.9 GB, the same gazebo in winter
fetch wood_summer    929362d6-65c7-4884-950f-c4089e748b7a   # 4.7 GB, wood: the least structured scene
fetch wood_autumn    3cc8fe4d-276b-4d19-8727-aec057592c95   # 3.1 GB, the same wood in autumn
echo "all downloads finished"
