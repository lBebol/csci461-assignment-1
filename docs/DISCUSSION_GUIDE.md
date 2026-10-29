# Team discussion guide

Use the actual notebook and outputs while rehearsing these questions.

## Docker

- **Dockerfile → image → container:** the Dockerfile is the build recipe; an image
  is a reusable filesystem/configuration template; a container is a running or
  stopped instance with its own writable layer and process environment.
- **Layers:** `COPY` and `RUN` produce cached filesystem layers. Copying pinned
  requirements before the changing notebook allows dependency installation to be
  reused. Changing a file invalidates its layer and subsequent dependent layers.
- **Registry / repository / tag:** Docker Hub is a registry. A repository is
  `username/csci461-assignment-1`; a tag such as `fall2026-v1` names an image version.
  Tags can move; a digest identifies exact image content.
- **`-p 8888:8888`:** left is the host port, right is the container port. The browser
  connects to the host port, which Docker forwards to Jupyter. Our command uses
  `127.0.0.1:8888:8888`, adding a host-interface binding. Another host port, e.g.
  `8889:8888`, works without changing the internal Jupyter port.
- **Bind mount versus volume:** a bind mount maps a chosen host path into the
  container. A named volume is storage managed by Docker. This assignment uses a
  bind mount so exported results are directly visible in the project directory.
- **Lifecycle:** `images` lists images; `ps` lists running containers; `ps -a`
  includes stopped containers; `logs` shows process logs; `inspect` shows metadata,
  state, port mappings, and mounts. `stop` stops the process; `rm` removes the
  container. Neither deletes bind-mounted host output.
- **Why rebuild?** A notebook copied into an image is a snapshot. Host edits appear
  immediately only when bind-mounted. Rebuild the final image to distribute edits
  to teammates who pull without mounting your source tree.

## Data and text

- **Missing dates:** 245 raw dates are blank. They remain unknown (`NaT`); imputing
  timestamps would invent information. A `has_date` flag records availability.
  Cleaning leaves 243 missing dates because some removed rows had missing dates.
- **Duplicates:** 3 duplicate records are removed. Repeated text from different
  comment IDs is retained because promotional repetition is meaningful. All
  identical cleaned text stays in one train/test partition.
- **Eight empty texts after cleaning:** some comments contain only symbols or
  other content removed by this text representation. They cannot support word
  features and are explicitly excluded, leaving 1,945 usable rows.
- **`loc` versus `iloc`:** `loc` uses labels or boolean conditions; `iloc` uses
  zero-based positions. `np.r_[0:3, n-3:n]` combines the first/last three positions.
- **Transformations:** timestamp conversion, target/category encoding, URL/date
  indicators, text normalization, word-length features, and length bins.
- **Sampling:** random selects records; stratified selects within labels to preserve
  proportions; cluster selects whole videos. Here the three sample sizes match.
  Selecting whole videos can shift label balance because source videos differ.
- **Correlation:** Pearson correlation measures linear association. Character and
  word counts share mathematical/source-text structure. Association is not causal
  evidence, and low Pearson correlation does not rule out nonlinear relationships.
- **Text cleaning:** preserve `CONTENT`; lowercase and normalize into `cleaned_text`,
  decode HTML, replace URLs/numbers, remove punctuation/extra spaces. These choices
  trade simplicity for loss of some potentially useful signals.
- **Stop words:** removing common function words improves topic charts. Classifiers
  retain them to preserve potentially useful unigram/bigram signals.

## Classification

- **TF-IDF:** document term frequency multiplied by inverse document frequency;
  common-in-one-document but rare-in-the-corpus terms receive more weight. The
  vocabulary and IDF must be learned only from training text.
- **Data leakage:** fitting TF-IDF on all rows or letting duplicated templates
  cross the holdout boundary can make evaluation too optimistic. Pipelines and a
  stratified grouped split address these two issues. Near-duplicates may remain.
- **NB versus logistic regression:** NB combines class/word probabilities with a
  conditional-independence approximation. Logistic regression learns linear
  feature weights for class probabilities. We compare both on identical test rows.
- **Confusion matrix:** rows are actual labels, columns predictions. Logistic
  Regression produces 180 true ham, 8 false spam alarms, 11 missed spam, and
  190 correctly detected spam on 389 test comments.
- **Precision / recall:** precision is `TP/(TP+FP)`; recall is `TP/(TP+FN)`. Raising
  a detection threshold often trades recall for precision. F1 combines both;
  macro F1 treats each class equally. Accuracy alone can hide imbalance.
- **Interpret the result:** Logistic Regression reaches 95.12% accuracy versus
  51.67% for the majority baseline, with macro F1 0.9511. This is one fixed split
  from five older music videos, not a guaranteed future-platform accuracy.

## K-Means

- **Why unsupervised?** Training uses TF-IDF/SVD vectors, not spam labels. Labels are
  inspected afterwards to compare cluster composition, never to select k.
- **SVD and normalization:** SVD projects sparse vocabulary to 50 dimensions;
  L2 normalization focuses comparisons on direction. Retained variance is only
  about 38.58%, so information loss is a real limitation.
- **Choosing k:** test 2–6; choose highest silhouette. k=6 wins within this range,
  but its boundary position means we have not established a global optimum.
- **Silhouette:** roughly `(other-cluster distance − own-cluster distance) /
  max(the two distances)`, averaged over records. Here 0.128 indicates weak
  separation. It is computed on a reproducible sample of 1,500 eligible rows.
- **Why 105 records have cluster -1:** stop-word removal and minimum-document
  frequency leave them with zero retained TF-IDF terms. They are excluded from
  clustering, not silently counted as a substantive cluster.
- **Interpretation:** view counts; subscription requests; video/playlist promotion;
  a large mixed music/promotion group; song opinions; and fan affection. The mixed
  cluster has 1,063 of 1,840 clustered records. ARI ≈ 0.051 shows weak overall
  agreement with the labels. Cluster IDs are arbitrary and are not predictions.

## Practice modifications

1. Change the sampling seed and explain the new label proportions.
2. Change a `loc` threshold or build a different `np.r_` positional selection.
3. Explain how preserving URLs as markers differs from deleting them completely.
4. Compare NB's higher spam recall against logistic regression's fewer false alarms.
5. Expand the exploratory k range, rerun, and revise the cluster interpretation.
6. Use host port 8889 while keeping the container port 8888.
7. Stop/remove a test container and show that its mounted output still exists.

If changing the notebook through the UI, save it before rebuilding the image.
If instead changing `scripts/build_notebook.py`, regenerate and rerun the notebook;
the generator replaces its previous contents and outputs.
