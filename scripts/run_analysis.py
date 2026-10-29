#!/usr/bin/env python3
"""Execute the embedded notebook and verify exports inside a fresh container."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
output = ROOT / 'output'
output.mkdir(parents=True, exist_ok=True)
subprocess.run([
    'jupyter', 'nbconvert', '--to', 'notebook', '--execute',
    '--ExecutePreprocessor.timeout=600', '--output', 'analysis.executed.ipynb',
    '--output-dir', str(output), str(ROOT / 'notebook/analysis.ipynb'),
], cwd=ROOT, check=True)
subprocess.run([
    sys.executable, str(ROOT / 'scripts/verify_outputs.py'),
    '--notebook', str(output / 'analysis.executed.ipynb'),
], cwd=ROOT, check=True)
