#!/usr/bin/env python3
"""Resolve published image/video paths for LLaMA-Factory.

Follows the relative-path relocation workflow of OpenSearch-VL:
https://huggingface.co/datasets/OpenSearch-VL/Search-VL-SFT-36K/blob/main/relocate_images.py
This implementation supports three modalities, streams large JSON arrays,
and writes new training files without overwriting the downloaded release.
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path

MODALITIES = ('image', 'multi_image', 'video')


def json_rows(path):
    decoder = json.JSONDecoder()
    with path.open(encoding='utf-8') as f:
        buf = ''; pos = 0; eof = False; opened = False
        while True:
            while pos < len(buf) and buf[pos] in ' \r\n\t,':
                pos += 1
            if not opened and pos < len(buf):
                if buf[pos] != '[':
                    raise ValueError('Expected a JSON array')
                opened = True; pos += 1; continue
            if opened and pos < len(buf) and buf[pos] == ']':
                if buf[pos + 1:].strip() or f.read().strip():
                    raise ValueError('Unexpected trailing content')
                return
            try:
                row, end = decoder.raw_decode(buf, pos)
            except json.JSONDecodeError:
                if eof:
                    raise ValueError('Truncated JSON array')
                chunk = f.read(4 * 1024 * 1024)
                buf = buf[pos:] + chunk; pos = 0; eof = not chunk
                continue
            yield row; pos = end


@lru_cache(maxsize=None)
def resolve(base, modality, field, value):
    relative = Path(value)
    if relative.is_absolute() or '..' in relative.parts or relative.parts[:1] != (field,):
        raise ValueError(f'Invalid {field} path: {value}')
    path = base / modality / relative
    if not path.is_file():
        raise FileNotFoundError(f'Missing media (extract all ZIP shards first): {path}')
    return str(path.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output-dir', type=Path, help='Usually the code repository SFT/data directory')
    parser.add_argument('--modalities', nargs='+', choices=MODALITIES, default=list(MODALITIES))
    parser.add_argument('--dry-run', action='store_true', help='Validate all references without writing JSON')
    args = parser.parse_args()
    if not args.dry_run and args.output_dir is None:
        parser.error('--output-dir is required unless --dry-run is set')
    base = args.base_dir.resolve()
    planned = []
    for modality in args.modalities:
        rel = Path(modality) / f'{modality}_llama_factory.json'
        src = base / rel
        if not src.is_file():
            raise FileNotFoundError(src)
        dst = args.output_dir.resolve() / rel if args.output_dir else None
        if not args.dry_run and (dst.exists() or dst.with_suffix('.json.partial').exists()):
            raise FileExistsError(f'Refusing to overwrite: {dst}')
        planned.append((modality, src, dst))
    for modality, src, dst in planned:
        n = 0; f = None
        partial = dst.with_suffix('.json.partial') if dst else None
        try:
            if not args.dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                f = partial.open('x', encoding='utf-8'); f.write('[\n')
            for row in json_rows(src):
                for field in ('images', 'videos'):
                    if row.get(field) is not None:
                        row[field] = [resolve(base, modality, field, v) for v in row[field]]
                if f:
                    if n: f.write(',\n')
                    f.write(json.dumps(row, ensure_ascii=False))
                n += 1
            if f:
                f.write('\n]\n'); f.close(); f = None
                partial.rename(dst)
        except BaseException:
            if f: f.close()
            if not args.dry_run and partial: partial.unlink(missing_ok=True)
            raise
        print(f'{modality}: {n:,} records validated' + ('' if args.dry_run else f' -> {dst}'), flush=True)


if __name__ == '__main__':
    main()
