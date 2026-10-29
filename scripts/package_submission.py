#!/usr/bin/env python3
"""Create the required ZIP with one bd-a1/ root, including data and evidence."""

import hashlib
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
destination = ROOT.parent / 'csci461-assignment-1.zip'
root_files = ['Dockerfile', 'requirements.txt', 'README.md', '.dockerignore', '.gitignore', '.gitattributes']
directories = ['data', 'notebook', 'output', 'screenshots', 'docs', 'scripts']
files = [ROOT / name for name in root_files]
for directory in directories:
    files.extend(path for path in (ROOT / directory).rglob('*')
                 if path.is_file() and not {'.ipynb_checkpoints', '__pycache__'}.intersection(path.parts))
with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for path in sorted(files):
        archive.write(path, Path('bd-a1') / path.relative_to(ROOT))
with zipfile.ZipFile(destination) as archive:
    assert archive.testzip() is None
    assert 'bd-a1/notebook/analysis.ipynb' in archive.namelist()
digest = hashlib.sha256(destination.read_bytes()).hexdigest()
destination.with_suffix('.zip.sha256').write_text(f'{digest}  {destination.name}\n')
print(f'Created {destination} ({destination.stat().st_size:,} bytes, {len(files)} files)')
print(f'SHA-256: {digest}')
