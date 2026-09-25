#!/usr/bin/env python3
"""Harvest step of the search protocol (notes/search-protocol.md), run 2026-09-23.

Deviations from the protocol, recorded here and in the paper:
- Scopus and Web of Science: no subscription available to the author.
- IEEE Xplore and ScienceDirect refuse automated access (HTTP 418 / CPE00001);
  their records are searched through OpenAlex, which carries abstracts for
  96-99 % of IEEE RA-L papers (checked 2026-09-23).
- OpenAlex has no wildcards; each `word*` of the protocol is replaced by the
  stemmed forms listed below (OpenAlex stems search terms itself).
- Google Scholar is not used for citation tracking; forward citations come
  from OpenAlex `cites:` and from OpenCitations (COCI), union of the two.
"""
import json, time, urllib.request, urllib.parse, re, sys, os
from xml.etree import ElementTree as ET

OUT = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "literature-search/1.0 (systematic search for a paper)"}

def get(url, tries=6):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            return urllib.request.urlopen(req, timeout=60).read()
        except Exception as e:
            time.sleep(2 + 3 * k)
    raise RuntimeError("failed: " + url)

def oa(params):
    return json.loads(get("https://api.openalex.org/works?" + urllib.parse.urlencode(params)))

FIELDS = "id,doi,title,publication_year,type,abstract_inverted_index,primary_location,referenced_works"

def oa_all(filt, label, records):
    cursor, n = "*", 0
    while cursor:
        d = oa({"filter": filt, "per-page": 200, "cursor": cursor, "select": FIELDS})
        for w in d["results"]:
            rec = records.setdefault(w["id"], {"w": w, "found_by": []})
            if label not in rec["found_by"]:
                rec["found_by"].append(label)
            n += 1
        cursor = d["meta"].get("next_cursor")
        if not d["results"]:
            break
        time.sleep(0.15)
    return n

# ---- the protocol's strings, translated for OpenAlex (no wildcards) ----
A = '("point cloud registration" OR "scan matching" OR "point set registration" OR "point cloud alignment" OR "iterative closest point")'
B = ('(perturbation OR perturbed OR perturb OR restart OR restarts OR restarted OR "multi-start" OR multistart '
     'OR "multiple initializations" OR "multiple initialisations" OR "Monte Carlo" OR "basin of convergence" OR "convergence basin")')
C = ('(verification OR verify OR verifying OR validation OR validate OR validating OR detection OR detect OR detecting '
     'OR diagnosis OR diagnose OR diagnostic OR introspection OR introspective OR "failure detection" OR "self-assessment" '
     'OR reliability OR reliable OR confidence)')
SEPARATE = {
    "S1 loop closure verification": '"loop closure verification" AND (lidar OR "point cloud")',
    "S2 registration failure":      '"registration failure" AND (detection OR detect OR detecting OR prediction OR predict OR predicting)',
    "S3 alignment verification":    '"alignment verification" AND (3D OR "point cloud")',
    "S4 degeneracy detection":      '"degeneracy detection" AND registration',
    "S5 introspective":             '(introspective OR introspection) AND (registration OR "scan matching")',
}
YEARS = "publication_year:1992-2026"

ANCHORS = {  # the protocol's nine, section 7 step 4
    "Censi 2007": "10.1109/ROBOT.2007.363961", "Zhang 2016": "10.1109/ICRA.2016.7487211",
    "CELLO-3D 2019": "10.1109/ICRA.2019.8793516", "Brossard 2020": "10.1109/LRA.2020.2965391",
    "X-ICP 2024": "10.1109/TRO.2023.3335691", "OverlapNet 2020": "10.15607/RSS.2020.XVI.009",
    "TBV 2023": "10.1109/LRA.2023.3268040", "CorAl 2022": "10.1016/j.robot.2022.104136",
    "Iversen 2017": "10.1109/IROS.2017.8206335",
}

def main():
    records, runs = {}, []
    stamp = time.strftime("%Y-%m-%d")
    n = oa_all("title_and_abstract.search:" + A + " AND " + B + " AND " + C + "," + YEARS, "M main A AND B AND C", records)
    runs.append(("OpenAlex", "M main A AND B AND C", stamp, n))
    for label, s in SEPARATE.items():
        n = oa_all("title_and_abstract.search:" + s + "," + YEARS, label, records)
        runs.append(("OpenAlex", label, stamp, n))

    # citation tracking, one level each way
    anchor_ids = {}
    for name, doi in ANCHORS.items():
        w = oa({"filter": "doi:" + doi, "select": FIELDS})["results"]
        if not w:
            runs.append(("OpenAlex", f"anchor {name}", stamp, "not found")); continue
        w = w[0]; wid = w["id"].split("/")[-1]; anchor_ids[name] = wid
        rec = records.setdefault(w["id"], {"w": w, "found_by": []}); rec["found_by"].append(f"anchor {name}")
        nf = oa_all("cites:" + wid, f"F cites {name}", records)
        refs = [r.split("/")[-1] for r in (w.get("referenced_works") or [])]
        nb = 0
        for i in range(0, len(refs), 50):
            nb += oa_all("openalex:" + "|".join(refs[i:i+50]), f"B cited by {name}", records)
        # OpenCitations COCI, forward
        oc = json.loads(get("https://api.opencitations.net/index/v2/citations/doi:" + doi))
        dois = sorted({m.group(1).lower() for c in oc for m in [re.search(r"doi:(\S+)", c.get("citing", ""))] if m})
        known = {(r["w"].get("doi") or "").lower().replace("https://doi.org/", "") for r in records.values()}
        missing = [d for d in dois if d not in known]
        no = 0
        for i in range(0, len(missing), 40):
            no += oa_all("doi:" + "|".join(missing[i:i+40]), f"F cites {name} (OpenCitations)", records)
        runs.append(("OpenAlex+OpenCitations", f"anchor {name}", stamp, f"forward {nf} (+{no} via OpenCitations of {len(dois)}), backward {nb}"))
        time.sleep(0.3)

    # arXiv, the protocol's string
    q = ('(abs:"point cloud registration" OR abs:"scan matching") AND '
         '(abs:perturb OR abs:perturbed OR abs:perturbation OR abs:restart OR abs:"multi-start" OR abs:"Monte Carlo") AND '
         '(abs:verify OR abs:verification OR abs:validate OR abs:validation OR abs:detect OR abs:detection OR abs:introspective OR abs:failure)')
    x = get("https://export.arxiv.org/api/query?" + urllib.parse.urlencode({"search_query": q, "max_results": 500}))
    ns = {"a": "http://www.w3.org/2005/Atom"}
    arx = []
    for e in ET.fromstring(x).findall("a:entry", ns):
        arx.append({"id": e.find("a:id", ns).text, "title": " ".join(e.find("a:title", ns).text.split()),
                    "abstract": " ".join(e.find("a:summary", ns).text.split()),
                    "year": e.find("a:published", ns).text[:4]})
    runs.append(("arXiv", "protocol string", stamp, len(arx)))

    with open(os.path.join(OUT, "records.jsonl"), "w") as f:
        for r in records.values():
            f.write(json.dumps(r) + "\n")
    json.dump(arx, open(os.path.join(OUT, "arxiv.json"), "w"), indent=1)
    json.dump(runs, open(os.path.join(OUT, "runs.json"), "w"), indent=1)
    for r in runs: print(r)
    print("unique OpenAlex records:", len(records), "| arXiv:", len(arx))

if __name__ == "__main__":
    main()
