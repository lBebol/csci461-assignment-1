#!/usr/bin/env python3
"""Host-side fresh-image execution, offline reproduction, and persistence check."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import uuid

from verify_outputs import compare, verify

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', default='csci461-a1:latest')
    parser.add_argument('--temp-dir', type=Path)
    args = parser.parse_args()
    name = f'csci461-reproduce-{uuid.uuid4().hex[:8]}'
    evidence = ROOT / 'output/evidence'
    evidence.mkdir(parents=True, exist_ok=True)
    transcript = []

    def run(command):
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        transcript.append('$ ' + ' '.join(command) + '\n' + result.stdout)
        print(result.stdout, end='')
        result.check_returncode()
        return result.stdout

    with tempfile.TemporaryDirectory(prefix='csci461-reproduction-', dir=args.temp_dir) as temporary:
        output = Path(temporary) / 'output'
        output.mkdir()
        created = False
        try:
            run(['docker', 'create', '--name', name, '--network', 'none', '--no-healthcheck', '--mount',
                 f'type=bind,src={output},dst=/home/jovyan/bd-a1/output',
                 args.image, 'python', 'scripts/run_analysis.py'])
            created = True
            run(['docker', 'start', '-a', name])
            # docker start -a alone is not relied upon to propagate the exit status.
            state = json.loads(run(['docker', 'inspect', name, '--format', '{{json .State}}']))
            assert state['ExitCode'] == 0 and state['Status'] == 'exited', state
            run(['docker', 'ps', '-a', '--filter', f'name=^/{name}$'])
            verify(output, output / 'analysis.executed.ipynb')
            compare(output, ROOT / 'output')
            artifact_hashes = {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sorted(output.rglob('*')) if path.is_file()}
            image_id = run(['docker', 'image', 'inspect', args.image, '--format', '{{.Id}}']).strip()
            run(['docker', 'rm', name])
            created = False
            assert all((output / path).is_file() and hashlib.sha256((output / path).read_bytes()).hexdigest() == digest
                       for path, digest in artifact_hashes.items())
            print('PASS: bind-mounted outputs are unchanged after removing the container.')
            report = {
                'verified_at_utc': datetime.now(timezone.utc).isoformat(),
                'image': args.image, 'image_id': image_id, 'network': 'none',
                'only_host_mount': 'fresh empty output directory',
                'container_exit_code': state['ExitCode'],
                'artifact_validation': 'passed', 'reference_comparison': '9 files byte-identical',
                'container_removed': True, 'output_persistence': 'passed',
                'independent_teammate_verification': False,
                'artifact_sha256': artifact_hashes,
            }
            (evidence / 'reproduction.json').write_text(json.dumps(report, indent=2) + '\n')
            transcript.append('PASS: exports validated; 9 artifacts byte-identical; outputs persist after container removal.\n')
        finally:
            if created:
                subprocess.run(['docker', 'rm', '-f', name], check=False)
            (evidence / 'reproduction.log').write_text('\n'.join(transcript))


if __name__ == '__main__':
    main()
