#!/bin/sh
# ETH TLS registration benchmark (Theiler et al., 2015), ETH Zurich PRS group.
# office and arch are the scenes of Table 2 and Section 5.5; facade and courtyard are the two scenes the
# probe of Section 5.6 finds no copy in.
cd "$(dirname "$0")"
while read scene url; do
  f=$(basename "$url")
  n=0
  until curl -s -f -L -C - --retry 5 --retry-delay 5 -o "$f" "$url"; do
    n=$((n+1)); [ $n -ge 5 ] && { echo "$scene FAILED" >> progress.txt; continue 2; }
    sleep 5
  done
  echo "$scene downloaded $(stat -f %z "$f") bytes" >> progress.txt
done < urls.txt
echo DONE >> progress.txt
