# Team and independent reproduction

The assignment requires **3–4 students**, and every member must be able to explain
all components. The GitHub maintainer is [@lBebol](https://github.com/lBebol).
Student names/IDs have not been supplied; add the actual roster before submission.

## Required independent teammate check — PENDING

The author-machine offline reproduction is recorded in
[`output/evidence/reproduction.json`](../output/evidence/reproduction.json).
It does **not** satisfy the requirement that another teammate pull the image.

After Docker Hub publication, another teammate must run the following on their
own machine, using the actual image reference recorded in the README:

```bash
docker pull DOCKERHUB_USERNAME/csci461-assignment-1:fall2026-v1
mkdir -p output
docker run --rm --network none --no-healthcheck \
  --mount type=bind,src="$(pwd)/output",dst=/home/jovyan/bd-a1/output \
  DOCKERHUB_USERNAME/csci461-assignment-1:fall2026-v1 \
  python scripts/run_analysis.py
docker image inspect DOCKERHUB_USERNAME/csci461-assignment-1:fall2026-v1 \
  --format '{{json .RepoDigests}}'
```

Record real evidence here:

| Item | Verification record |
|---|---|
| Teammate name / student ID | Pending |
| Date and machine / OS | Pending |
| Pulled image tag and digest | Pending publication and pull |
| Notebook completed without errors | Not yet independently verified |
| Output files persisted on the host | Not yet independently verified |
| Screenshots / terminal log | Add the teammate's actual evidence |

The final completion claim must wait until the teammate has performed this check.

## Discussion preparation

Use [DISCUSSION_GUIDE.md](DISCUSSION_GUIDE.md), rerun the notebook, and practice
explaining every section. Dividing implementation work does not divide the scope
of questions each member may be asked.
