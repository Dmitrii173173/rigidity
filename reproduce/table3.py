#!/usr/bin/env python3
"""Fold the eth_survey logs into Table 3 of the paper.

A win is an improvement of the worst station by more than MARGIN. The paper
states 15 per cent and the caption of Table 3 says so; the bench itself counts
any improvement at all, which counts arithmetic noise, and that count is kept
here as a second column so the difference stays visible. Pass another margin as
the second argument to recompute all five rows.
"""
import re, os, sys
RES = sys.argv[1]
MARGIN = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15   # Table 3 of the paper
SEQS=["apartment","hauptgebaude","plain","stairs","gazebo_summer","gazebo_winter","wood_summer","wood_autumn"]
FOVS=[360,90,60,40,30]
WORST=re.compile(r"^\s*([^:]{1,24}):\s+worst\s+([\d.]+) m", re.M)
ASK=re.compile(r"^\s+asking\s+(\S+\s*mm),\s+(\d+) of (\d+) directions dropped\s*$", re.M)
HDR=re.compile(r"^  (?:the same edges weighted[^\n]*|the same again[^\n]*|the crate's own[^\n]*|Hatleskog[^\n]*|Λ = Σ[^\n]*|the nulls:|weighted by conditioning[^\n]*)$", re.M)

def sections(t):
    hs=[(m.start(), m.group(0).strip()) for m in HDR.finditer(t)]
    return [(n, t[p:(hs[i+1][0] if i+1<len(hs) else len(t))]) for i,(p,n) in enumerate(hs)]

rows={k:[] for k in ["threshold","floor","probabilistic","s11","no_spectrum"]}
cells=notcomp=0
for s in SEQS:
    for f in FOVS:
        p=f"{RES}/tIII_{s}_{f}.log"
        if not os.path.exists(p): continue
        t=open(p).read(); secs=sections(t); base=None
        for n,b in secs:
            if n.startswith("the same edges weighted"):
                m=WORST.search(b); base=float(m.group(2)) if m else None
        if not base: continue
        for n,b in secs:
            if n.startswith("weighted by conditioning"):
                for (lab,lost,tot),(_,w) in zip(ASK.findall(b), WORST.findall(b)):
                    cells+=1
                    if int(lost)*2 < int(tot): rows["threshold"].append(float(w)/base)
                    else: notcomp+=1
            elif n.startswith("Λ = Σ vvᵀ/(spread² + floor²)"):
                rows["floor"] += [float(w)/base for _,w in WORST.findall(b)]
            elif n.startswith("Hatleskog"):
                rows["probabilistic"] += [float(w)/base for lab,w in WORST.findall(b) if lab.strip()=="probabilistic"]
            elif "the bias S11 measured" in n:
                rows["s11"] += [float(w)/base for _,w in WORST.findall(b)]
            elif n.startswith("the nulls"):
                rows["no_spectrum"] += [float(w)/base for lab,w in WORST.findall(b) if lab.strip()=="no spectrum"]

names={"threshold":"Threshold on predicted spread","floor":"Additive floor",
       "probabilistic":"Probabilistic attenuation","s11":"Floor from the measured bias",
       "no_spectrum":"Discard the spectrum"}
# comparisons and wins at MARGIN=0.15, as Table 3 prints them
paper={"threshold":(98,3),"floor":(360,0),"probabilistic":(40,5),"s11":(40,0),"no_spectrum":(40,0)}
print(f"a win is an improvement of the worst station by more than {MARGIN:.0%}\n")
print(f"{'':<32}{'compar.':>8}{'wins':>6}{'any':>5}{'best':>7}{'worst':>8}   {'paper':>9}")
TC=TB=0
for k in ["threshold","floor","probabilistic","s11","no_spectrum"]:
    v=rows[k]; n=len(v)
    b=sum(1 for r in v if r < 1.0 - MARGIN)          # wins at the stated margin
    any_=sum(1 for r in v if r < 0.9995)             # what the bench itself would call better
    pn,pb=paper[k]
    mk="✓" if (n,b)==(pn,pb) else ("~" if n==pn else "≠")
    print(f"{names[k]:<32}{n:>8}{b:>6}{any_:>5}{min(v):>7.2f}{max(v):>8.1f}   {pn:>4}/{pb}  {mk}")
    TC+=n; TB+=b
print(f"{'total':<32}{TC:>8}{TB:>6}   {'578/8':>21}")
print(f"threshold cells {cells}, not comparisons {notcomp}")
