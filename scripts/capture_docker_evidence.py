#!/usr/bin/env python3
"""Capture real host Docker commands, redacting ephemeral Jupyter login tokens."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def redact(text):
    return re.sub(r'(?i)(token=)[^\s&]+', r'\1[REDACTED]', text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container', default='csci461-a1')
    parser.add_argument('--image', default='csci461-a1:latest')
    parser.add_argument('--stage', choices=['all', 'build', 'images', 'containers'], default='all')
    parser.add_argument('--include-scaffold', action='store_true', help='Include author-machine initial empty-image evidence.')
    args = parser.parse_args()
    evidence = ROOT / 'output/evidence'
    evidence.mkdir(parents=True, exist_ok=True)
    stages = {
        'build': [['docker', 'build', '--progress=plain', '-t', args.image, '.']],
        'images': [
            ['docker', 'images', 'csci461-a1', '--no-trunc'],
            ['docker', 'image', 'inspect', args.image, '--format',
             'Image={{.Id}}\nCreated={{.Created}}\nWorkingDir={{.Config.WorkingDir}}\nUser={{.Config.User}}'],
        ],
        'containers': [
            ['docker', 'ps', '--filter', f'name=^/{args.container}$'],
            ['docker', 'ps', '-a', '--filter', 'name=csci461'],
            ['docker', 'inspect', args.container, '--format',
             'Image={{.Config.Image}}\nStatus={{.State.Status}}\nPorts={{json .NetworkSettings.Ports}}\nMounts={{json .Mounts}}'],
        ],
    }
    commands = stages.get(args.stage)
    if args.stage == 'all':
        commands = stages['images'] + stages['containers'] + [
            ['docker', 'logs', '--tail', '35', args.container],
        ]
    if args.include_scaffold:
        commands += [
            ['docker', 'run', '--rm', 'csci461-a1:scaffold', 'python', '-c',
             'import csv,json; print("Initial scaffold notebook cells:",len(json.load(open("notebook/analysis.ipynb"))["cells"])); print("Included dataset rows:",sum(1 for _ in csv.DictReader(open("data/dataset.csv"))))'],
        ]
    transcript = [f'Actual Docker evidence captured {datetime.now(timezone.utc).isoformat()}\n']
    for command in commands:
        print('$ ' + shlex.join(command), flush=True)
        result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        text = redact(result.stdout)
        print(text, flush=True)
        transcript.append('$ ' + shlex.join(command) + '\n' + text)
        result.check_returncode()
    (evidence / f'docker_{args.stage}.log').write_text('\n'.join(transcript))
    print('SUCCESS: all displayed Docker commands completed.', flush=True)


if __name__ == '__main__':
    main()
