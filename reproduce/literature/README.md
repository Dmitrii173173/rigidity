# The literature search behind Section 2

Section 2 of the paper claims that no published work reads the spread of re-registrations from
perturbed starts as a verdict on whether a registration is in the right basin, at a stated
threshold, with error rates measured out of sample. A claim that something has not been done
cannot be checked by a reader unless the search is, so the search was written down before it was
run and every count is here.

| File | What it is |
|---|---|
| `protocol-2026-09-01.ru.md` | The protocol as written on 1 September 2026, before the search, in Russian and unedited. Section numbers in it refer to an earlier draft (its "Section V-E" is Section 5.5). |
| `harvest.py` | The harvest, rerunnable: the protocol's strings against OpenAlex and arXiv, and the citation tracking. It writes `records.jsonl`, abstracts included, next to itself. |
| `records.csv` | The 1401 records returned on 23 September 2026, without abstracts, and which query found each. |
| `arxiv.json` | The eight arXiv records; all eight are works already in OpenAlex. |
| `runs.json` | The count returned by every query and every anchor. |
| `titles.tsv` | All 1409 titles, as screened. |
| `screening.csv` | The decision on each of the 168 records that reached their abstracts, with the reason. |

## The claim and the criterion

The criterion was fixed in advance, in section 8 of the protocol. The claim is withdrawn, not
reworded, if a work is found that

- thresholds a quantity computed from several registrations of one pair from different
  initialisations, **and**
- reports at least one error rate — caught or false alarms — against an independent check.

A Monte Carlo covariance does not meet it: it gives a quantity and no verdict. A work that gives a
verdict without rates weakens the claim to a measurement but does not remove it, and is recorded
separately.

## Where the run departed from the protocol

| Protocol | What was done | Why |
|---|---|---|
| Scopus, Web of Science | not run | no subscription available to the author |
| IEEE Xplore, ScienceDirect | reached through OpenAlex | both refuse automated queries; OpenAlex carries abstracts for 96–99 % of IEEE Robotics and Automation Letters (checked for 2019, 2022 and 2025) |
| wildcards (`perturb*`) | the stemmed forms listed | OpenAlex has no wildcards and stems its terms |
| Google Scholar for citations | OpenAlex `cites:` together with OpenCitations (COCI) | Scholar refuses automated access; OpenCitations adds 30 citing works OpenAlex misses, 20 of them for CELLO-3D alone |

## The funnel

All queries on 23 September 2026, titles and abstracts, 1992 to 2026.

| Stage | Records |
|---|---|
| Returned, after merging (OpenAlex identifiers; the 8 arXiv records all matched) | 1401 |
| Titles read | 1409 |
| Past the titles: 70 selected by hand and all 98 records of the main string | 168 |
| Past the abstracts: included | 20 |
| — verified only through the map or trajectory (exclusion 3, counted separately) | 7 |
| — not assessable: no abstract and no accessible text | 2 |
| Read in full | 2 |
| Meeting the criterion | **0** |

Queries: the main string (blocks A AND B AND C) returned 107; the separate strings 3, 56, 23, 10
and 120; arXiv 8; one level of citations forward from the nine anchors 1046 and back 286.

## The nearest works

- **Aoki et al. (2023)**, J. Robotics and Mechatronics 35(2), 435–444. NDT is started from several
  initial poses and the covariance of the poses it converges to is read as a sign of localisation
  risk. The threshold they state only decides whether to add starts; using the covariance to switch
  localisation off is left to future work; no rate is reported. Nearest to the device. Read in
  full.
- **Steiner et al. (2021)**, IEEE RA-L 6(4), 8710–8717. A certainty computed over several candidate
  registrations is thresholded to end a global localisation, and success rates are reported. The
  candidates are different places in the map, not restarts of one pair: the question answered is
  which place, not whether the answer returns. Nearest to the use. The full text could not be
  obtained; judged on the abstract.
- **Liao et al. (2021)**, IEEE TPAMI 43(9), 3229–3246. A ratio of the cross-cloud to the intrinsic
  fuzzy-centre distance, at most one for an aligned pair, computed from one registration and used
  as a stop criterion. A measure of fit, like CorAl. Read in full.

The remaining seventeen included records judge a pair from one registration — CorAl, Almqvist et
al. (2018), Akai et al. (2022), Nobili et al. (2018), TEASER's certificate and others — and fail
the criterion's first clause. `screening.csv` gives each.

## Outcome

No work meets the criterion; the claim stands, and the nearest two are named in Section 2.
