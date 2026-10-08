"""OneSearch-VL entry point for registering the prepared RL splits."""
import argparse
from pathlib import Path
import subprocess
import sys

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[4]
    if not (root / 'RL/register_dataset.py').is_file():
        root = root.parent / 'OneSearch-VL'
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--code-root', type=Path, default=root)
    args, remaining = parser.parse_known_args()
    script = args.code_root / 'RL/register_dataset.py'
    if not script.is_file():
        raise FileNotFoundError('Provide --code-root pointing to the OneSearch-VL code repository')
    subprocess.run([sys.executable, str(script), *remaining], check=True)
