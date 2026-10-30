# Team and independent reproduction

The assignment requires **3–4 students**, and every member must be able to explain
all components. The GitHub maintainer is [@lBebol](https://github.com/lBebol).
Student names/IDs have not been supplied; add the actual roster before submission.

## Required independent teammate check — PENDING

The published image was successfully pulled and reproduced offline on the author's
machine. Its exact digest and public-access check are recorded in
[`dockerhub_publication.json`](../output/evidence/dockerhub_publication.json),
and the full execution/persistence check is recorded in
[`output/evidence/reproduction.json`](../output/evidence/reproduction.json).
It does **not** satisfy the requirement that another teammate pull the image.

Another teammate must run the following on their own machine. The image reference
is `lbebol/csci461-assignment-1:fall2026-v1`; its exact immutable digest is recorded
in [`dockerhub_publication.json`](../output/evidence/dockerhub_publication.json).

```bash
docker pull lbebol/csci461-assignment-1:fall2026-v1
mkdir -p output
docker run --rm --network none --no-healthcheck \
  --mount type=bind,src="$(pwd)/output",dst=/home/jovyan/bd-a1/output \
  lbebol/csci461-assignment-1:fall2026-v1 \
  python scripts/run_analysis.py
docker image inspect lbebol/csci461-assignment-1:fall2026-v1 \
  --format '{{json .RepoDigests}}'
```

Record real evidence here:

| Item | Verification record |
|---|---|
| Teammate name / student ID | Pending |
| Date and machine / OS | Pending |
| Pulled image tag and digest | Image reference is above; teammate's pull confirmation is pending |
| Notebook completed without errors | Not yet independently verified |
| Output files persisted on the host | Not yet independently verified |
| Screenshots / terminal log | Add the teammate's actual evidence |

The final completion claim must wait until the teammate has performed this check.

## Discussion preparation

Use [DISCUSSION_GUIDE.md](DISCUSSION_GUIDE.md), rerun the notebook, and practice
explaining every section. Dividing implementation work does not divide the scope
of questions each member may be asked.
