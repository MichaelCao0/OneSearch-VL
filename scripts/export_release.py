#!/usr/bin/env python3
"""Export current source files under Git's publish rules without Git history."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DIRECTORIES = {'.git', '__pycache__', '.pytest_cache', 'tests', 'outputs',
                     'cache', 'saves', 'checkpoints', 'wandb', 'build', 'dist'}


def is_local_artifact(path):
    """Also reject private artifacts force-added to Git despite ignore rules."""
    return (
        bool(set(path.parts) & LOCAL_DIRECTORIES)
        or path.name == '.env'
        or (path.name.startswith('.env.') and path.name != '.env.example')
        or path.suffix in {'.parquet', '.arrow', '.safetensors', '.pt', '.mp4', '.pyc'}
        or '.hope' in path.suffixes
        or path.as_posix().startswith('RL/rllm/rllm/data/datasets/')
        or (path.parent.as_posix() == 'RL/rllm/rllm/registry' and path.suffix == '.json')
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path,
                        help='New directory outside this repository; never overwritten.')
    args = parser.parse_args(argv)
    destination = args.output.expanduser().resolve()
    if destination == ROOT or ROOT in destination.parents:
        parser.error('--output must be outside the source repository')
    if destination.exists():
        parser.error('--output already exists; choose a new directory')
    top = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=ROOT, text=True).strip()
    if Path(top).resolve() != ROOT:
        parser.error('Run the exporter from a OneSearch-VL Git checkout')
    listed = subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard', '--deduplicate'], cwd=ROOT,
    ).decode().split('\0')
    files = []
    for name in sorted(set(listed) - {''}):
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            parser.error('Git returned an unsafe path')
        source = ROOT / relative
        if any(parent.is_symlink() for parent in [source, *source.parents] if parent != ROOT):
            parser.error(f'Symlink is not publishable: {name}')
        if source.is_file():
            if is_local_artifact(relative):
                parser.error(f'Local artifact is tracked or not ignored: {name}')
            files.append((source, relative))
    destination.mkdir(parents=True)
    for source, relative in files:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    print(f'Exported {len(files)} source files. No Git history was copied.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
