# Assignment coverage

Source: CSCI461 Assignment 1, Fall 2026, all three pages.

| Requirement | Implementation / evidence |
|---|---|
| Team of 3–4; each understands everything | `TEAM_VERIFICATION.md`, `DISCUSSION_GUIDE.md`; roster pending actual names |
| One approved dataset | Option 3, YouTube Spam Collection; `data/README.md`, `SOURCE.json` |
| Required project layout | Root Dockerfile, requirements, README; data, notebook, output, screenshots |
| Host-prepared empty notebook and data before first build | Initial Git commit and `csci461-a1:scaffold`; Docker evidence reports zero initial cells |
| Required Jupyter base; additional packages; folder and WORKDIR | `Dockerfile`, pinned `requirements.txt` |
| Notebook runs entirely in container | Notebook environment assertion; offline fresh-container evidence |
| Restart and Run All | 31 ordered cells; all 15 code cells execute; Jupyter screenshot |
| Image/container and port explanation | README Docker concepts; discussion guide |
| Persistent output bind mount | README run command; reproduction checks hashes after container removal |
| images, ps, ps -a, logs, inspect | `output/evidence/docker_all.log`; screenshots |
| Docker Hub tag and push | `lbebol/csci461-assignment-1:fall2026-v1`; publication record and push log under `output/evidence/` |
| Different teammate pulls and runs | Explicitly pending actual teammate evidence in `TEAM_VERIFICATION.md` |
| head, shape, columns, dtypes, info, describe; row/column meanings | Notebook §2 |
| Missing values, duplicates, justified treatment, before/after shape | §3 |
| loc, iloc, np.r_ | §3.1 |
| At least three transformations | §3.1: conversion, encoding, normalization, features, binning |
| Random / stratified / cluster sampling and distribution comparison | §4; equal-size samples and video-based clusters |
| Four numerical text features; interpreted heatmap; causation caveat | §5 |
| Preserve original text; cleaned_text; punctuation/case/URLs/numbers | §3, original-text preservation assertion |
| Tokenization; total/unique/mean tokens | §6 |
| Top 20 table and chart; stop-word comparison | §6 |
| Two groups: mean lengths/counts and top 10 words | §6.1 |
| Five evidence-backed written insights | Six numbered insights in §6.2 and `output/text_insights.md` |
| Labeled train/test split; TF-IDF; classifier | §7, grouped stratification and training-only TF-IDF |
| Accuracy, precision, recall, F1, confusion matrix, interpretation/limitation | §7.2 and metrics output |
| K-Means with justified k; sizes; silhouette; top terms; interpretations | §8; k sweep, SVD, semantic interpretation |
| Compare clusters to known labels without equating IDs | §8.1, crosstab and adjusted Rand index |
| Required cleaned CSV and two metrics text files | `output/cleaned_data.csv`, `model_metrics.txt`, `cluster_summary.txt` |
| At least four required figures | Eight PNG figures; all four specified categories included |
| Build, images, running-container, notebook screenshots | `screenshots/` |
| Complete project ZIP, including redistributable dataset | `python3 scripts/package_submission.py` |
| GitHub bonus | Public `lBebol/csci461-assignment-1` repository |
| Model-extension bonus | Two classifiers using the same split; unigram/bigram features |

## Ambiguities in the handout

- It says six approved options but lists eight. Options 7–8 are clinical numerical
  datasets that do not naturally satisfy the text requirements; approved option 3
  satisfies all requested text tasks.
- The cover advertises +2 bonus marks, while the detailed bonus section lists two
  +0.5 activities, totaling +1. Both explicitly listed activities are included.
- The empty-notebook build instruction is respected by the initial scaffold build;
  the final distributed image is rebuilt with the completed notebook.
