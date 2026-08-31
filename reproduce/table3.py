#!/usr/bin/env python3
"""Свёртка логов eth_survey в Table III статьи."""
import re, os, sys
RES = sys.argv[1]
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
paper={"threshold":(97,16),"floor":(360,52),"probabilistic":(40,14),"s11":(40,18),"no_spectrum":(40,10)}
print(f"{'':<32}{'сравн.':>7}{'лучше':>7}{'best':>7}{'worst':>8}   {'в статье':>12}")
TC=TB=0
for k in ["threshold","floor","probabilistic","s11","no_spectrum"]:
    v=rows[k]; n=len(v); b=sum(1 for r in v if r<0.9995); pn,pb=paper[k]
    mk="✓" if (n,b)==(pn,pb) else ("~" if n==pn else "≠")
    print(f"{names[k]:<32}{n:>7}{b:>7}{min(v):>7.2f}{max(v):>8.1f}   {pn:>5}/{pb}  {mk}")
    TC+=n; TB+=b
print(f"{'ИТОГО':<32}{TC:>7}{TB:>7}   {'577/110':>12}")
print(f"ячеек порога {cells}, не сравнения {notcomp}")
