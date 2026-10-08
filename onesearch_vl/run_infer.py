#!/usr/bin/env python3
"""Unified modality router; underlying inference implementations are preserved."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--modality', required=True, choices=['single_image', 'multi_image', 'video'])
    parser.add_argument('--dry-run', action='store_true', help='Print the command without importing model backends.')
    args, remainder = parser.parse_known_args()
    entry = {'single_image': 'run_infer_single.py', 'multi_image': 'run_infer_mi.py', 'video': 'run_infer_video.py'}[args.modality]
    command = [sys.executable, str(Path(__file__).resolve().parent / entry), *remainder]
    if args.dry_run:
        print(shlex.join(command))
        return 0
    return subprocess.call(command)


if __name__ == '__main__':
    raise SystemExit(main())
