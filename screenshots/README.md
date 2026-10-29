# Screenshot evidence

Screenshots capture actual Docker commands and the actual container-served
JupyterLab notebook. They are not terminal mockups or rendered substitutes.

Expected evidence:

1. `01-docker-build.png` — successful final custom-image build.
2. `02-docker-images.png` — custom image and image metadata.
3. `03-running-container.png` — running container, published port, and bind mounts.
4. `04-jupyter-notebook.png` — notebook open in JupyterLab with executed output.
5. `05-jupyter-results.png` — executed analysis results in JupyterLab.
6. `06-jupyter-completion.png` — final success message and exported artifacts.

The notebook was executed through JupyterLab's **Restart Kernel and Run All Cells**
menu, then saved. Code was temporarily collapsed through the View menu for readable
screenshots and expanded again afterward. Docker screenshots were taken from a
real Kitty terminal with Grim; JupyterLab screenshots were taken from Chromium.

The corresponding actual CLI output is retained under `output/evidence/`.
Ephemeral Jupyter authentication tokens are redacted from saved terminal logs.
Independent teammate pull/run screenshots must be added by that teammate after
Docker Hub publication; the local screenshots do not substitute for their check.
