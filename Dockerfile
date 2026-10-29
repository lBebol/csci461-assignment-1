FROM quay.io/jupyter/scipy-notebook:latest

LABEL org.opencontainers.image.title="CSCI461 Assignment 1: YouTube Spam Analysis" \
      org.opencontainers.image.source="https://github.com/lBebol/csci461-assignment-1"

USER ${NB_UID}
RUN mkdir -p /home/jovyan/bd-a1
WORKDIR /home/jovyan/bd-a1

COPY --chown=${NB_UID}:${NB_GID} requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

# These files are prepared on the HOST before docker build.
COPY --chown=${NB_UID}:${NB_GID} data/ ./data/
COPY --chown=${NB_UID}:${NB_GID} notebook/ ./notebook/
COPY --chown=${NB_UID}:${NB_GID} scripts/ ./scripts/
COPY --chown=${NB_UID}:${NB_GID} docs/ ./docs/
COPY --chown=${NB_UID}:${NB_GID} README.md ./README.md
RUN mkdir -p output/figures screenshots

ENV PYTHONHASHSEED=42 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MPLCONFIGDIR=/tmp/matplotlib

EXPOSE 8888
CMD ["start-notebook.py", "--ServerApp.root_dir=/home/jovyan/bd-a1"]
